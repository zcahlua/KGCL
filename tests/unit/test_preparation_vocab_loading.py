from kgcl.data.preparation import process_batch
import pytest


def test_process_batch_rejects_empty_batch_before_vocab_use():
    with pytest.raises(ValueError, match='at least one'):
        process_batch([], config=object(), bond_vocab=object(), atom_vocab=object())
