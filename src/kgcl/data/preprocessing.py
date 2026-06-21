import sys
from collections import Counter
from collections.abc import Sequence
from typing import List

import joblib
import pandas as pd
from rdkit import Chem

from kgcl.config import KGCLConfig
from kgcl.config.paths import ProjectPaths
from utils.generate_edits import generate_reaction_edits


def check_edits(edits: List):
    for edit in edits:
        if edit[0] == 'Add Bond':
            return False

    return True


def is_valid_reaction_side(mol: Chem.Mol | None) -> bool:
    return mol is not None and mol.GetNumAtoms() > 1 and mol.GetNumBonds() > 1


def preprocessing(rxns: Sequence[str], args: KGCLConfig, rxn_classes: Sequence[int] | None = None, rxns_id: Sequence[str] | None = None) -> None:
    """Preprocess reactions data to get edits."""
    rxn_classes = [] if rxn_classes is None else list(rxn_classes)
    rxns_id = [] if rxns_id is None else list(rxns_id)
    if args.dataset == 'uspto_50k' and (len(rxn_classes) != len(rxns) or len(rxns_id) != len(rxns)):
        raise ValueError('USPTO-50K preprocessing requires rxn_classes and rxn_ids for every reaction.')
    rxns_data = []
    counter = []
    all_edits = {}

    paths = ProjectPaths(args.root_dir)
    savedir = paths.split_dir(args.dataset, args.mode)
    savedir.mkdir(parents=True, exist_ok=True)

    for idx, rxn_smi in enumerate(rxns):
        r, p = rxn_smi.split('>>')
        prod_mol = Chem.MolFromSmiles(p)

        if not is_valid_reaction_side(prod_mol):
            print(
                f'Product has 0 or 1 atom or 1 bond, Skipping reaction {idx}')
            print()
            sys.stdout.flush()
            continue

        react_mol = Chem.MolFromSmiles(r)

        if not is_valid_reaction_side(react_mol):
            print(
                f'Reactant has 0 or 1 atom or 1 bond, Skipping reaction {idx}')
            print()
            sys.stdout.flush()
            continue

        try:
            if args.dataset == 'uspto_50k':
                rxn_data = generate_reaction_edits(rxn_smi, kekulize=args.kekulize, rxn_class=int(
                    rxn_classes[idx]) - 1, rxn_id=rxns_id[idx])
            else:
                rxn_data = generate_reaction_edits(
                    rxn_smi, kekulize=args.kekulize)
        except Exception as exc:
            print(f'Failed to extract reaction data for reaction {idx}: {exc}')
            print()
            sys.stdout.flush()
            continue

        edits_accepted = check_edits(rxn_data.edits)
        if not edits_accepted:
            print(f'Edit: Add new bond. Skipping reaction {idx}')
            print()
            sys.stdout.flush()
            continue

        if args.dataset == 'uspto_full':
            if len(rxn_data.edits) > args.max_steps or len(rxn_data.edits) == 1:
                print(f'Edits step exceed max_steps or edit step is 1. Skipping reaction {idx}')
                print()
                sys.stdout.flush()
                continue

        if args.dataset == 'uspto_mit':
            if len(rxn_data.edits) > args.max_steps or len(rxn_data.edits) == 1:
                print(f'Edits step exceed max_steps or edit step is 1. Skipping reaction {idx}')
                print()
                sys.stdout.flush()
                continue

        rxns_data.append(rxn_data)

        if (idx % args.print_every == 0) and idx:
            print(f'{idx}/{len(rxns)} {args.mode} reactions processed.')
            sys.stdout.flush()

    print(f'All {args.mode} reactions complete.')
    sys.stdout.flush()

    save_file = paths.serialized_reactions_file(args.dataset, args.mode, args.kekulize)

    if args.mode == 'train':
        for idx, rxn_data in enumerate(rxns_data):
            for edit in rxn_data.edits:
                if edit not in all_edits:
                    all_edits[edit] = 1
                else:
                    all_edits[edit] += 1

        atom_edits = []
        bond_edits = []
        lg_edits = []
        atom_lg_edits = []

        if args.dataset == 'uspto_50k':
            for edit, num in all_edits.items():
                if edit[0] == 'Change Atom':
                    atom_edits.append(edit)
                    atom_lg_edits.append(edit)
                elif edit[0] == 'Delete Bond' or edit[0] == 'Change Bond' or edit[0] == 'Add Bond':
                    bond_edits.append(edit)
                elif edit[0] == 'Attaching LG':
                    lg_edits.append(edit)
            atom_lg_edits.extend(lg_edits)

        elif args.dataset == 'uspto_full':
            for edit, num in all_edits.items():
                if edit[0] == 'Change Atom':
                    atom_edits.append(edit)
                    atom_lg_edits.append(edit)
                elif edit[0] == 'Delete Bond' or edit[0] == 'Change Bond' or edit[0] == 'Add Bond':
                    bond_edits.append(edit)
                elif edit[0] == 'Attaching LG' and num >= 50:
                    lg_edits.append(edit)
            atom_lg_edits.extend(lg_edits)

        # elif args.dataset == 'uspto_mit':
        #     for edit, num in all_edits.items():
        #         if edit[0] == 'Change Atom':
        #             atom_edits.append(edit)
        #             atom_lg_edits.append(edit)
        #         elif edit[0] == 'Delete Bond' or edit[0] == 'Change Bond' or edit[0] == 'Add Bond':
        #             bond_edits.append(edit)
        #         elif edit[0] == 'Attaching LG' and num >= 20:
        #             lg_edits.append(edit)
        #     atom_lg_edits.extend(lg_edits)

        print(atom_edits)
        print(bond_edits)
        print(lg_edits)

        filter_rxns_data = []
        for idx, rxn_data in enumerate(rxns_data):
            for edit in rxn_data.edits:
                if edit[0] == 'Attaching LG' and edit not in lg_edits:
                    print(
                        f'The number of {edit} in training set is very small, skipping reaction')
                    rxn_data = None
            if rxn_data is not None:
                counter.append(len(rxn_data.edits))
                filter_rxns_data.append(rxn_data)

        print(Counter(counter))

        joblib.dump(filter_rxns_data, save_file, compress=3)
        joblib.dump(atom_edits, savedir / 'atom_vocab.txt')
        joblib.dump(bond_edits, savedir / 'bond_vocab.txt')
        joblib.dump(lg_edits, savedir / 'lg_vocab.txt')
        joblib.dump(atom_lg_edits, savedir / 'atom_lg_vocab.txt')
    else:
        bond_vocab_file = paths.vocab_file(args.dataset, 'bond_vocab.txt')
        atom_vocab_file = paths.vocab_file(args.dataset, 'atom_lg_vocab.txt')
        bond_vocab = joblib.load(bond_vocab_file)
        atom_vocab = joblib.load(atom_vocab_file)
        bond_vocab.extend(atom_vocab)
        all_edits = bond_vocab

        cover_num = 0
        for idx, rxn_data in enumerate(rxns_data):
            cover = True
            for edit in rxn_data.edits:
                if edit != 'Terminate' and edit not in all_edits:
                    print(f'{edit} in {args.mode} is not in train set')
                    cover = False
            if cover:
                cover_num += 1

            counter.append(len(rxn_data.edits))

        print(Counter(counter))
        print(f'The cover rate is {cover_num}/{len(rxns_data)}')
        joblib.dump(rxns_data, save_file, compress=3)


def run(args: KGCLConfig):
    args.print_every = args.preprocess_print_every
    paths = ProjectPaths(args.root_dir)
    rxn_key = 'reactants>reagents>production'
    input_csv = paths.canonicalized_csv(args.dataset, args.mode)
    if not input_csv.exists():
        raise FileNotFoundError(f'Canonicalized input CSV not found: {input_csv}')
    df = pd.read_csv(input_csv)
    if args.dataset == 'uspto_50k':
        preprocessing(rxns=df[rxn_key], args=args, rxn_classes=df['class'], rxns_id=df['id'])
    else:
        preprocessing(rxns=df[rxn_key], args=args)
