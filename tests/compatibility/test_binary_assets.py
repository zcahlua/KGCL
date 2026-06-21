from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests" / "fixtures" / "binary-assets.sha256"


def _manifest_entries():
    entries = []
    for line in MANIFEST.read_text().splitlines():
        digest, path = line.split("  ", 1)
        entries.append((digest, ROOT / path))
    return entries


def test_immutable_binary_asset_hashes_match_manifest():
    for expected, path in _manifest_entries():
        assert path.is_file(), path
        assert sha256(path.read_bytes()).hexdigest() == expected


def test_read_only_checkpoint_discovery_manifest_entries_exist():
    checkpoints = [path for _, path in _manifest_entries() if path.suffix == ".pt"]
    assert checkpoints
    assert all(path.parts[-4] == "experiments" or "experiments" in path.parts for path in checkpoints)


def test_checkpoint_metadata_read_only_when_torch_available():
    pytest = __import__("pytest")
    torch = pytest.importorskip("torch")
    for _, path in _manifest_entries():
        if path.suffix != ".pt":
            continue
        checkpoint = torch.load(path, map_location="cpu")
        assert isinstance(checkpoint, dict)
        assert "state" in checkpoint
