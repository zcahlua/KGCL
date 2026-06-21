from pathlib import Path
from kgcl.config.paths import ProjectPaths


def test_project_paths_cover_workflow_locations(tmp_path):
    p = ProjectPaths(tmp_path)
    assert p.raw_csv('USPTO_50k','test') == tmp_path/'data'/'uspto_50k'/'raw_test.csv'
    assert p.canonicalized_csv('uspto_50k','test').name == 'canonicalized_test.csv'
    assert p.serialized_reactions_file('uspto_50k','valid', True).name == 'valid.file.kekulized'
    assert p.serialized_reactions_file('uspto_50k','valid', False).name == 'valid.file'
    assert p.vocab_file('uspto_50k','bond_vocab.txt').name == 'bond_vocab.txt'
    assert p.prepared_shard_dir('uspto_50k','train', False).name == 'without_rxn_class'
    assert p.experiment_dir('uspto_50k', True, 'BEST').parts[-2:] == ('with_rxn_class','BEST')
    assert p.round_trip_prediction_dir('uspto_50k', False, 'BEST').name == 'pred_text1'
    assert p.forward_prediction_input('uspto_50k', False, 'BEST').name == 'forward_predictions_50k_top50.txt'
