import importlib
import sys
import pytest

COMMANDS = [
    'kgcl.cli.train','kgcl.cli.preprocess','kgcl.cli.prepare_data','kgcl.cli.canonicalize',
    'kgcl.cli.evaluate','kgcl.cli.evaluate_full','kgcl.cli.evaluate_round_trip',
]
HEAVY = {'torch','rdkit','numpy','pandas','joblib','tqdm','models','utils'}

@pytest.mark.parametrize('module_name', COMMANDS)
def test_cli_imports_are_lightweight(module_name):
    for name in list(HEAVY):
        sys.modules.pop(name, None)
    module = importlib.import_module(module_name)
    module.build_parser()
    assert not (HEAVY & set(sys.modules)), HEAVY & set(sys.modules)

@pytest.mark.parametrize('module_name', COMMANDS)
def test_cli_help_uses_command_parser(module_name):
    module = importlib.import_module(module_name)
    with pytest.raises(SystemExit) as exc:
        module.main(['--help'])
    assert exc.value.code == 0
