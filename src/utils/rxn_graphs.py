from pathlib import Path
from functools import lru_cache
from typing import List, Tuple
from dataclasses import dataclass
from rdkit import Chem
from utils.mol_features import get_atom_features, get_bond_features
import torch
import torch.nn.functional as F
import math
from kgcl.resources.functional_groups import get_functional_group_resource_config, resolve_functional_group_resources

@dataclass(frozen=True)
class FunctionalGroupResources:
    smarts: tuple
    smart2name: dict
    embeddings: dict


def load_embedding_mapping(path):
    import pickle
    try:
        with path.open("rb") as fh:
            data = pickle.load(fh)
    except Exception as exc:
        raise ValueError(f"Could not load KGCL functional-group embedding mapping from {path}: {exc}") from exc
    if not hasattr(data, "keys"):
        raise ValueError(f"KGCL functional-group embedding mapping at {path} must be a mapping")
    return data

@lru_cache(maxsize=8)
def _load_cached(use_rxn_class: bool, resource_root_key: str | None, root_dir_key: str | None) -> FunctionalGroupResources:
    paths = resolve_functional_group_resources(
        use_rxn_class=use_rxn_class,
        resource_root=Path(resource_root_key) if resource_root_key else None,
        root_dir=Path(root_dir_key) if root_dir_key else None,
    )
    entries = []
    for line_no, raw in enumerate(paths.definitions.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) < 2:
            raise ValueError(f"Malformed functional-group definition {paths.definitions}:{line_no}: {raw}")
        mol = Chem.MolFromSmarts(parts[1])
        if mol is None:
            raise ValueError(f"Could not parse SMARTS in {paths.definitions}:{line_no}: {parts[1]}")
        entries.append((parts[0], mol))
    embeddings = load_embedding_mapping(paths.embeddings)
    missing = [name for name, _ in entries if name not in embeddings]
    if missing:
        raise ValueError(f"Functional-group definitions in {paths.definitions} are missing embedding keys in {paths.embeddings}: {missing[:5]}")
    names = tuple(name for name, _ in entries)
    smarts = tuple(mol for _, mol in entries)
    return FunctionalGroupResources(smarts=smarts, smart2name=dict(zip(smarts, names)), embeddings=embeddings)

def load_functional_group_resources(use_rxn_class: bool) -> FunctionalGroupResources:
    cfg = get_functional_group_resource_config()
    return _load_cached(bool(use_rxn_class), str(cfg.resource_root) if cfg.resource_root else None, str(cfg.root_dir) if cfg.root_dir else None)


def get_functional_group_resources(use_rxn_class: bool) -> FunctionalGroupResources:
    return load_functional_group_resources(use_rxn_class)


def match_fg(mol, use_rxn_class):
    fg_names = []
    fg_emb = []
    resource = load_functional_group_resources(bool(use_rxn_class))
    for sm in resource.smarts:
        if mol.HasSubstructMatch(sm):
            name = resource.smart2name[sm]
            fg_emb.append(resource.embeddings[name].tolist())
            fg_names.append(name)
    return fg_emb, fg_names

def attention(query, key, mask=None, dropout=None):

    pad_rows = query.size(0) - key.size(0)
    if pad_rows > 0:
        zero_padding = torch.zeros(pad_rows, key.size(1))
        key_pad = key.clone()
        key = torch.cat((key_pad, zero_padding), dim=0)

    value = key.clone()

    d_k = key.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    # Res coonect
    a = torch.matmul(p_attn, value)
    out = query + torch.matmul(p_attn, value)
    return out, p_attn


class MolGraph:
    """
    'MolGraph' represents the graph structure and featurization of a single molecule.
    """

    def __init__(self, mol: Chem.Mol, rxn_class: int = None, use_rxn_class: bool = False) -> None:
        """
        Parameters
        ----------
        mol: Chem.Mol,
            Molecule
        rxn_class: int, default None,
            Reaction class for this reaction.
        use_rxn_class: bool, default False,
            Whether to use reaction class as additional input
        """
        self.mol = mol
        self.rxn_class = rxn_class
        self.use_rxn_class = use_rxn_class
        self._build_mol()
        self._build_graph()

    def _build_mol(self) -> None:
        """Builds the molecule attributes."""
        self.num_atoms = self.mol.GetNumAtoms()
        self.num_bonds = self.mol.GetNumBonds()
        self.amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx()
                            for atom in self.mol.GetAtoms()}
        self.idx_to_amap = {value: key for key,
                                           value in self.amap_to_idx.items()}

    def _build_graph(self):
        """Builds the graph attributes."""
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        # mapping from bond index to concat(in_atom, bond) features
        self.f_bonds = []
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond

        # functional group embedding
        self.f_fgs, self.fg_names = match_fg(self.mol, self.use_rxn_class)
        self.atoms = []

        # Get atom features
        self.f_atoms = [get_atom_features(
            atom, rxn_class=self.rxn_class, use_rxn_class=self.use_rxn_class) for atom in self.mol.GetAtoms()]
        self.n_atoms = len(self.f_atoms)
        for atom in self.mol.GetAtoms():
            self.atoms.append(atom.GetSymbol())
        # Initialize atom to bond mapping for each atom
        for _ in range(self.n_atoms):
            self.a2b.append([])

        # add group knowledge
        if self.f_fgs:
            temp_tensor = torch.tensor(self.f_atoms)
            f_fgs_tensor = torch.tensor(self.f_fgs)
            fuse_f_atoms, self.attn_score = attention(temp_tensor, f_fgs_tensor)
            self.f_atoms = fuse_f_atoms.tolist()

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = self.mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = get_bond_features(bond)

                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2

class RxnGraph:
    """
    RxnGraph contains the information of a reaction, like reactants, products. The edits associated with the reaction are also captured in edit labels.
    """

    def __init__(self, prod_mol: Chem.Mol, edit_to_apply: Tuple, edit_atom: List = [], reac_mol: Chem.Mol = None,
                 rxn_class: int = None, use_rxn_class: bool = False) -> None:
        """
        Parameters
        ----------
        prod_mol: Chem.Mol,
            Product molecule
        reac_mol: Chem.Mol, default None
            Reactant molecule(s)
        edit_to_apply: Tuple,
            Edits to apply to the product molecule
        edit_atom: List,
            Edit atom of product molecule
        rxn_class: int, default None,
            Reaction class for this reaction.
        use_rxn_class: bool, default False,
            Whether to use reaction class as additional input
        """
        self.prod_graph = MolGraph(
            mol=prod_mol, rxn_class=rxn_class, use_rxn_class=use_rxn_class)
        if reac_mol is not None:
            self.reac_mol = reac_mol
        self.edit_to_apply = edit_to_apply
        self.edit_atom = edit_atom
        self.rxn_class = rxn_class

    def get_components(self, attrs: List = ['prod_graph', 'edit_to_apply', 'edit_atom']) -> Tuple:
        """ 
        Returns the components associated with the reaction graph. 
        """
        attr_tuple = ()
        for attr in attrs:
            if hasattr(self, attr):
                attr_tuple += (getattr(self, attr),)
            else:
                print(f"Does not have attr {attr}")

        return attr_tuple


class Vocab:
    """
    Vocab class to deal with vocabularies and other attributes.
    """

    def __init__(self, elem_list: List) -> None:
        """
        Parameters
        ----------
        elem_list: List, default ATOM_LIST
            Element list used for setting up the vocab
        """
        self.elem_list = elem_list
        if isinstance(elem_list, dict):
            self.elem_list = list(elem_list.keys())
        self.elem_to_idx = {a: idx for idx, a in enumerate(self.elem_list)}
        self.idx_to_elem = {idx: a for idx, a in enumerate(self.elem_list)}

    def __getitem__(self, a_type: Tuple) -> int:
        return self.elem_to_idx[a_type]

    def get(self, elem: Tuple, idx: int = None) -> int:
        """Returns the index of the element, else a None for missing element.

        Parameters
        ----------
        elem: str,
            Element to query
        idx: int, default None
            Index to return if element not in vocab
        """
        return self.elem_to_idx.get(elem, idx)

    def get_elem(self, idx: int) -> Tuple:
        """Returns the element at given index.

        Parameters
        ----------
        idx: int,
            Index to return if element not in vocab
        """
        return self.idx_to_elem[idx]

    def __len__(self) -> int:
        return len(self.elem_list)

    def get_index(self, elem: Tuple) -> int:
        """Returns the index of the element.

        Parameters
        ----------
        elem: str,
            Element to query
        """
        return self.elem_to_idx[elem]

    def size(self) -> int:
        """Returns length of Vocab."""
        return len(self.elem_list)
