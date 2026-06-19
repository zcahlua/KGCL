import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def test_config_public_import():
    from kgcl.config import KGCLConfig, load_config
    assert KGCLConfig().dataset == load_config().dataset
