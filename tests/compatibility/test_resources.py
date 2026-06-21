import pytest

rdkit = pytest.importorskip("rdkit")

def test_functional_group_resources_are_lazy_and_cwd_independent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from utils.rxn_graphs import load_functional_group_resources
    without = load_functional_group_resources(False)
    with_class = load_functional_group_resources(True)
    assert without.smarts
    assert with_class.smarts
    assert without is not with_class
    assert without.smart2name is not with_class.smart2name
