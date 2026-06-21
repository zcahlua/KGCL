import pytest

pytest.importorskip("rdkit")
pytest.importorskip("torch")
pytestmark = pytest.mark.runtime


def test_historical_imports():
    from models.KGCL import KGCL
    from models.beam_search import BeamSearch
    from utils.generate_edits import ReactionData
    from utils.rxn_graphs import MolGraph, RxnGraph, Vocab
    from utils.reaction_actions import BondEditAction

    assert KGCL and BeamSearch and MolGraph and RxnGraph and Vocab and ReactionData and BondEditAction
