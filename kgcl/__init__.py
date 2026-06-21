"""Source-checkout shim for the ``kgcl`` src-layout package."""
from pathlib import Path

_src_package = Path(__file__).resolve().parents[1] / "src" / "kgcl"
if str(_src_package) not in __path__:
    __path__.append(str(_src_package))
