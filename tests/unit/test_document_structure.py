from pathlib import Path


def test_docs_have_no_case_insensitive_duplicate_names():
    names = [path.name.casefold() for path in Path("docs").iterdir() if path.is_file()]
    assert len(names) == len(set(names))
