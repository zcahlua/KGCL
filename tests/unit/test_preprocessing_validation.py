from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("joblib")
pytest.importorskip("pandas")
pytest.importorskip("rdkit")

from kgcl.data import preprocessing as pp


class FakeMol:
    def __init__(self, atoms, bonds):
        self._atoms = atoms
        self._bonds = bonds
    def GetNumAtoms(self):
        return self._atoms
    def GetNumBonds(self):
        return self._bonds


def test_is_valid_reaction_side():
    assert pp.is_valid_reaction_side(FakeMol(2, 2))
    assert not pp.is_valid_reaction_side(None)
    assert not pp.is_valid_reaction_side(FakeMol(1, 2))
    assert not pp.is_valid_reaction_side(FakeMol(2, 1))


def test_reactant_bond_count_is_validated(monkeypatch, tmp_path, capsys):
    args = SimpleNamespace(dataset="uspto_full", mode="train", root_dir=str(tmp_path), kekulize=True, max_steps=9, print_every=100)
    def mol_from_smiles(smi):
        return FakeMol(2, 1) if smi == "R" else FakeMol(2, 2)
    monkeypatch.setattr(pp.Chem, "MolFromSmiles", mol_from_smiles)
    gen = Mock()
    monkeypatch.setattr(pp, "generate_reaction_edits", gen)
    pp.preprocessing(["R>>P"], args)
    assert not gen.called
    assert "Reactant" in capsys.readouterr().out


def test_exception_diagnostics_include_index_and_message(monkeypatch, tmp_path, capsys):
    args = SimpleNamespace(dataset="uspto_full", mode="train", root_dir=str(tmp_path), kekulize=True, max_steps=9, print_every=100)
    monkeypatch.setattr(pp.Chem, "MolFromSmiles", lambda _s: FakeMol(2, 2))
    monkeypatch.setattr(pp, "generate_reaction_edits", Mock(side_effect=RuntimeError("boom")))
    pp.preprocessing(["R>>P"], args)
    assert "Failed to extract reaction data for reaction 0: boom" in capsys.readouterr().out
