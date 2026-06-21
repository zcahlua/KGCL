from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from kgcl.cli.help import maybe_print_help
maybe_print_help(Path(__file__).name, sys.argv)
from kgcl.cli.evaluate import main
if __name__ == "__main__":
    raise SystemExit(main())
