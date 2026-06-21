"""Source-checkout shim for historical ``models`` imports."""
from pathlib import Path

_src_package = Path(__file__).resolve().parents[1] / "src" / "models"
if str(_src_package) not in __path__:
    __path__.append(str(_src_package))

__all__ = ["KGCL", "BeamSearch"]


def __getattr__(name):
    if name == "KGCL":
        from models.KGCL import KGCL
        return KGCL
    if name == "BeamSearch":
        from models.beam_search import BeamSearch
        return BeamSearch
    raise AttributeError(name)
