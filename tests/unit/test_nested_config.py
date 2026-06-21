import json

import pytest

from kgcl.config import load_config_file


def test_flat_yaml_and_json_work(tmp_path):
    y = tmp_path / 'c.yaml'
    y.write_text('device: cuda:0\ndataset: uspto_50k\n')
    j = tmp_path / 'c.json'
    j.write_text(json.dumps({'device': 'cuda:1'}))
    assert load_config_file(y)['device'] == 'cuda:0'
    assert load_config_file(j)['device'] == 'cuda:1'


def test_nested_mapping_rejected(tmp_path):
    p = tmp_path / 'c.yaml'
    p.write_text('training:\n  device: cuda:0\nevaluation:\n  device: cuda:1\n')
    with pytest.raises(ValueError, match='Nested configuration sections'):
        load_config_file(p)


def test_empty_yaml_works(tmp_path):
    p = tmp_path / 'empty.yaml'
    p.write_text('')
    assert load_config_file(p) == {}


def test_unknown_flat_key_fails(tmp_path):
    p = tmp_path / 'bad.yaml'
    p.write_text('unknown: 1\n')
    with pytest.raises(ValueError, match='Unknown KGCL configuration key'):
        load_config_file(p)
