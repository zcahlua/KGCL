"""Shared molecule edit application logic used by data prep, models, and beam search."""
from __future__ import annotations

from typing import Any, Tuple

from rdkit import Chem

from utils.reaction_actions import AddGroupAction, AtomEditAction, BondEditAction


def apply_edit_to_mol(mol: Chem.Mol, edit: Tuple, edit_atom: Any) -> Chem.Mol:
    """Apply a KGCL edit tuple to an RDKit molecule without changing legacy behavior."""
    if edit[0] == 'Change Atom':
        edit_exe = AtomEditAction(edit_atom, *edit[1], action_vocab='Change Atom')
        new_mol = edit_exe.apply(mol)

    if edit[0] == 'Delete Bond':
        edit_exe = BondEditAction(*edit_atom, *edit[1], action_vocab='Delete Bond')
        new_mol = edit_exe.apply(mol)

    if edit[0] == 'Change Bond':
        edit_exe = BondEditAction(*edit_atom, *edit[1], action_vocab='Change Bond')
        new_mol = edit_exe.apply(mol)

    if edit[0] == 'Add Bond':
        edit_exe = BondEditAction(*edit_atom, *edit[1], action_vocab='Add Bond')
        new_mol = edit_exe.apply(mol)

    if edit[0] == 'Attaching LG':
        edit_exe = AddGroupAction(edit_atom, edit[1], action_vocab='Attaching LG')
        new_mol = edit_exe.apply(mol)

    return new_mol
