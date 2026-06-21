__all__ = ["KGCL", "BeamSearch"]


def __getattr__(name):
    if name == "KGCL":
        from models.KGCL import KGCL
        return KGCL
    if name == "BeamSearch":
        from models.beam_search import BeamSearch
        return BeamSearch
    raise AttributeError(name)
