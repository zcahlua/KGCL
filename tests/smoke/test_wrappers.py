import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_root_help_wrappers_are_lightweight():
    for script in ["train.py", "preprocess.py", "prepare_data.py", "eval.py", "eval-full.py", "eval-rtacc.py", "canonicalize_prod.py"]:
        result = subprocess.run([sys.executable, str(ROOT / script), "--help"], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
