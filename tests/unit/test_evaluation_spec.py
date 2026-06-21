from kgcl.config import KGCLConfig
from kgcl.evaluation.full import run as full_run
from kgcl.evaluation.standard import run as standard_run
import kgcl.evaluation.full as full
import kgcl.evaluation.standard as standard


def test_standard_spec(monkeypatch):
    seen = {}
    monkeypatch.setattr(standard, 'run_exact_match_evaluation', lambda cfg, spec: seen.setdefault('spec', spec) or 0)
    standard_run(KGCLConfig())
    assert seen['spec'].top_k_values == (1, 3, 5, 10, 50)
    assert seen['spec'].include_max_frag is True
    assert seen['spec'].output_kind == 'standard'


def test_full_spec(monkeypatch):
    seen = {}
    monkeypatch.setattr(full, 'run_exact_match_evaluation', lambda cfg, spec: seen.setdefault('spec', spec) or 0)
    full_run(KGCLConfig())
    assert seen['spec'].top_k_values == (1, 3, 5, 10)
    assert seen['spec'].include_max_frag is False
    assert seen['spec'].output_kind == 'full'
