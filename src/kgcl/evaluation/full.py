import numpy as np
import joblib
from tqdm import tqdm
from collections import Counter
import torch
from rdkit import Chem, RDLogger

lg = RDLogger.logger()
lg.setLevel(4)


def run(args):
    from kgcl.evaluation.common import evaluation_setup
    context = evaluation_setup(args, output_kind='full')

    # Import only after functional-group resources are configured.
    from models import KGCL, BeamSearch
    device = context.device
    test_data = joblib.load(context.test_data_path)
    checkpoint_path = context.checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['saveables']

    model = KGCL(**config, device=device)
    model.load_state_dict(checkpoint['state'])
    model.to(device)
    model.eval()

    top_k = np.zeros(args.full_beam_size)
    edit_steps_cor = []
    counter = []
    stereo_rxn = []
    stereo_rxn_cor = []
    beam_model = BeamSearch(model=model, step_beam_size=args.step_beam_size,
                            beam_size=args.full_beam_size, use_rxn_class=args.use_rxn_class)
    p_bar = tqdm(list(range(len(test_data))))
    pred_file = str(context.output_path)

    with open(pred_file, 'a') as fp:
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

            with torch.no_grad():
                top_k_results = beam_model.run_search(
                    prod_smi=p, max_steps=args.max_steps, rxn_class=rxn_class)

            fp.write(f'({idx}) {rxn_smi}\n')

            beam_matched = False
            for beam_idx, path in enumerate(top_k_results):
                pred_smi = path['final_smi']
                prob = path['prob']
                pred_set = set(pred_smi.split('.'))
                correct = pred_set == r_set
                str_edits = '|'.join(f'({str(edit)};{p})'for edit, p in zip(
                    path['rxn_actions'], path['edits_prob']))
                fp.write(
                    f'{beam_idx} prediction_is_correct:{correct} probability:{prob} {pred_smi} {str_edits}\n')
                if correct and not beam_matched:
                    top_k[beam_idx] += 1
                    beam_matched = True

            fp.write('\n')
            if beam_matched:
                edit_steps_cor.append(edit_steps)

            for edit in rxn_data.edits:
                if edit[1] == (1, 1) or edit[1] == (1, 2) or edit[1] == (0, 1) or edit[1] == (0, 2) or edit[1] == (2, 2) or edit[1] == (2, 3):
                    stereo_rxn.append(idx)
                    if beam_matched:
                        stereo_rxn_cor.append(idx)

            msg = 'average score'
            for beam_idx in [1, 3, 5, 10]:
                match_acc = np.sum(top_k[:beam_idx]) / (idx + 1)
                msg += ', t%d: %.3f' % (beam_idx, match_acc)
            p_bar.set_description(msg)

        edit_steps = Counter(counter)
        edit_steps_correct = Counter(edit_steps_cor)
        fp.write(f'edit_steps_reaction_number:{edit_steps}\n')
        fp.write(
            f'edit_steps_reaction_prediction_correct:{edit_steps_correct}\n')
        fp.write(f'stereo_reaction_idx:{stereo_rxn}\n')
        fp.write((f'stereo_reaction_prediction_correct:{stereo_rxn_cor}\n'))


