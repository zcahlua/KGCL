from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationSpec:
    beam_size: int
    step_beam_size: int
    top_k_values: tuple[int, ...]
    include_max_frag: bool
    track_stereo: bool
    output_kind: str


def canonicalize_smiles_clear_map(smiles, return_max_frag=True):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            return ('', '') if return_max_frag else ''
        if return_max_frag:
            sub_smi = smi.split('.')
            sub_mol = [Chem.MolFromSmiles(smiles) for smiles in sub_smi]
            sub_mol_size = [(sub_smi[i], len(m.GetAtoms())) for i, m in enumerate(sub_mol) if m is not None]
            if len(sub_mol_size) > 0:
                return smi, canonicalize_smiles_clear_map(sorted(sub_mol_size, key=lambda x: x[1], reverse=True)[0][0], return_max_frag=False)
            return smi, ''
        return smi
    return ('', '') if return_max_frag else ''


def _assert_model_cuda(model) -> None:
    assert all(parameter.device.type == 'cuda' for parameter in model.parameters())
    assert all(buffer.device.type == 'cuda' for buffer in model.buffers())


def run_exact_match_evaluation(config, spec: EvaluationSpec) -> int:
    import joblib
    import numpy as np
    import torch
    from rdkit import Chem, RDLogger
    from tqdm import tqdm

    RDLogger.logger().setLevel(4)
    from kgcl.evaluation.common import evaluation_setup
    context = evaluation_setup(config, output_kind=spec.output_kind)
    from models import BeamSearch, KGCL

    device = context.cuda.torch_device
    test_data = joblib.load(context.test_data_path)
    checkpoint = torch.load(context.checkpoint_path, map_location=device)
    model_config = checkpoint['saveables']
    model = KGCL(**model_config, device=str(device))
    model.load_state_dict(checkpoint['state'])
    model.to(device)
    _assert_model_cuda(model)
    model.eval()

    top_k = np.zeros(spec.beam_size)
    maxfrag_top_k = np.zeros(spec.beam_size)
    edit_steps_cor = []
    counter = []
    stereo_rxn = []
    stereo_rxn_cor = []
    beam_model = BeamSearch(model=model, step_beam_size=spec.step_beam_size, beam_size=spec.beam_size, use_rxn_class=config.use_rxn_class)
    p_bar = tqdm(list(range(len(test_data))))

    with open(str(context.output_path), 'a') as fp:
        for idx in p_bar:
            rxn_data = test_data[idx]
            rxn_smi = rxn_data.rxn_smi
            rxn_class = rxn_data.rxn_class
            edit_steps = len(rxn_data.edits)
            counter.append(edit_steps)
            r, p = rxn_smi.split('>>')
            r_mol = Chem.MolFromSmiles(r)
            [a.ClearProp('molAtomMapNumber') for a in r_mol.GetAtoms()]
            r_mol = Chem.MolFromSmiles(Chem.MolToSmiles(r_mol))
            r_smi = Chem.MolToSmiles(r_mol, isomericSmiles=True)
            r_set = set(r_smi.split('.'))
            r_maxfrag = None
            if spec.include_max_frag:
                _, r_maxfrag = canonicalize_smiles_clear_map(r)
            with torch.no_grad():
                top_k_results = beam_model.run_search(prod_smi=p, max_steps=config.max_steps, rxn_class=rxn_class)
            fp.write(f'({idx}) {rxn_smi}\n')
            beam_matched = False
            maxfrag_beam_matched = False
            for beam_idx, path in enumerate(top_k_results):
                pred_smi = path['final_smi']
                prob = path['prob']
                pred_set = set(pred_smi.split('.'))
                correct = pred_set == r_set
                str_edits = '|'.join(f'({str(edit)};{p})' for edit, p in zip(path['rxn_actions'], path['edits_prob']))
                fp.write(f'{beam_idx} prediction_is_correct:{correct} probability:{prob} {pred_smi} {str_edits}\n')
                if correct and not beam_matched:
                    top_k[beam_idx] += 1
                    beam_matched = True
                if spec.include_max_frag:
                    _, pred_maxfrag = canonicalize_smiles_clear_map(pred_smi)
                    if pred_maxfrag == r_maxfrag and not maxfrag_beam_matched:
                        maxfrag_top_k[beam_idx] += 1
                        maxfrag_beam_matched = True
            fp.write('\n')
            if beam_matched:
                edit_steps_cor.append(edit_steps)
            if spec.track_stereo:
                for edit in rxn_data.edits:
                    if edit[1] in {(1, 1), (1, 2), (0, 1), (0, 2), (2, 2), (2, 3)}:
                        stereo_rxn.append(idx)
                        if beam_matched:
                            stereo_rxn_cor.append(idx)
            msg = 'average score'
            for beam_idx in spec.top_k_values:
                match_acc = np.sum(top_k[:beam_idx]) / (idx + 1)
                msg += ', t%d: %.3f' % (beam_idx, match_acc)
                if spec.include_max_frag:
                    maxfrag_match_acc = np.sum(maxfrag_top_k[:beam_idx]) / (idx + 1)
                    msg += ', MaxFrag:' + 't%d: %.3f' % (beam_idx, maxfrag_match_acc)
            p_bar.set_description(msg)
        fp.write(f'edit_steps_reaction_number:{Counter(counter)}\n')
        fp.write(f'edit_steps_reaction_prediction_correct:{Counter(edit_steps_cor)}\n')
        fp.write(f'stereo_reaction_idx:{stereo_rxn}\n')
        fp.write(f'stereo_reaction_prediction_correct:{stereo_rxn_cor}\n')
    return 0
