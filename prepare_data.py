import sys as _kgcl_sys
from pathlib import Path as _kgcl_Path
_kgcl_src = _kgcl_Path(__file__).resolve().parent / "src"
if _kgcl_src.exists() and str(_kgcl_src) not in _kgcl_sys.path:
    _kgcl_sys.path.insert(0, str(_kgcl_src))
from kgcl.cli.help import maybe_print_help as _kgcl_maybe_print_help
_kgcl_maybe_print_help(__file__.rsplit('/', 1)[-1], _kgcl_sys.argv)

import argparse
import sys
import copy
import os
from kgcl.config import add_arguments, add_config_argument, load_config
from typing import Any

import joblib
import torch
from rdkit import Chem

from utils.collate_fn import get_batch_graphs, prepare_edit_labels
from utils.reaction_actions import Termination
from utils.rxn_graphs import MolGraph, RxnGraph, Vocab


from kgcl.chemistry.edit_application import apply_edit_to_mol

def process_batch(batch_graphs, args):
    lengths = torch.tensor([len(graph_seq)
                           for graph_seq in batch_graphs], dtype=torch.long)
    max_length = max([len(graph_seq) for graph_seq in batch_graphs])

    bond_vocab_file = os.path.join(args.root_dir, 'data', args.dataset, args.mode, 'bond_vocab.txt')
    atom_vocab_file = os.path.join(args.root_dir, 'data', args.dataset, args.mode, 'atom_lg_vocab.txt')
    bond_vocab = Vocab(joblib.load(bond_vocab_file))
    atom_vocab = Vocab(joblib.load(atom_vocab_file))

    graph_seq_tensors = []
    edit_seq_labels = []
    seq_mask = []

    for idx in range(max_length):
        graphs_idx = [copy.deepcopy(batch_graphs[i][min(idx, length-1)]).get_components(attrs=['prod_graph', 'edit_to_apply', 'edit_atom'])
                      for i, length in enumerate(lengths)]
        mask = (idx < lengths).long()
        prod_graphs, edits, edit_atoms = list(zip(*graphs_idx))
        assert all([isinstance(graph, MolGraph) for graph in prod_graphs])

        edit_labels = prepare_edit_labels(
            prod_graphs, edits, edit_atoms, bond_vocab, atom_vocab)
        current_graph_tensors = get_batch_graphs(
            prod_graphs, use_rxn_class=args.use_rxn_class)

        graph_seq_tensors.append(current_graph_tensors)
        edit_seq_labels.append(edit_labels)
        seq_mask.append(mask)

    seq_mask = torch.stack(seq_mask).long()
    assert seq_mask.shape[0] == max_length
    assert seq_mask.shape[1] == len(batch_graphs)

    return graph_seq_tensors, edit_seq_labels, seq_mask


def prepare_data(args: Any) -> None:
    """ 
    prepare data batches for edits prediction
    """
    datafile = os.path.join(args.root_dir, 'data', args.dataset, args.mode, f'{args.mode}.file')
    if args.kekulize:
        datafile += '.kekulized'
    rxns_data = joblib.load(datafile)

    batch_graphs = []
    batch_num = 0

    if args.use_rxn_class:
        savedir = os.path.join(args.root_dir, 'data', args.dataset, args.mode, 'with_rxn_class')
    else:
        savedir = os.path.join(args.root_dir, 'data', args.dataset, args.mode, 'without_rxn_class')
    os.makedirs(savedir, exist_ok=True)

    for idx, rxn_data in enumerate(rxns_data):
        graph_seq = []
        rxn_smi = rxn_data.rxn_smi
        r, p = rxn_smi.split('>>')
        r_mol = Chem.MolFromSmiles(r)
        p_mol = Chem.MolFromSmiles(p)
        if args.kekulize:
            Chem.Kekulize(p_mol)

        if len(rxn_data.edits) > args.max_steps:
            print(f'Edits step exceed max_steps. Skipping reaction {idx}')
            print()
            sys.stdout.flush()
            continue

        int_mol = p_mol
        for i, edit in enumerate(rxn_data.edits):
            if int_mol is None:
                print("Interim mol is None")
                break
            if edit == 'Terminate':
                graph = RxnGraph(prod_mol=Chem.Mol(
                    int_mol), edit_to_apply=edit, reac_mol=Chem.Mol(r_mol), rxn_class=rxn_data.rxn_class, use_rxn_class=args.use_rxn_class)
                graph_seq.append(graph)
                edit_exe = Termination(action_vocab='Terminate')
                try:
                    pred_mol = edit_exe.apply(Chem.Mol(int_mol))
                    final_smi = Chem.MolToSmiles(pred_mol)
                except Exception as e:
                    final_smi = None
            else:
                graph = RxnGraph(prod_mol=Chem.Mol(int_mol), edit_to_apply=edit,
                                 edit_atom=rxn_data.edits_atom[i], reac_mol=Chem.Mol(r_mol), rxn_class=rxn_data.rxn_class, use_rxn_class=args.use_rxn_class)
                graph_seq.append(graph)
                int_mol = apply_edit_to_mol(
                    Chem.Mol(int_mol), edit, rxn_data.edits_atom[i])

        if len(graph_seq) == 0 or final_smi is None:
            print(f"No valid states found. Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        batch_graphs.append(graph_seq)
        if (idx % args.print_every == 0) and idx:
            print(f"{idx}/{len(rxns_data)} {args.mode} reactions processed.")
            sys.stdout.flush()

        if (len(batch_graphs) % args.batch_size == 0) and len(batch_graphs):
            batch_tensors = process_batch(batch_graphs, args)
            torch.save(batch_tensors, os.path.join(
                savedir, f'batch-{batch_num}.pt'))

            batch_num += 1
            batch_graphs = []

    print(f"All {args.mode} reactions complete.")
    sys.stdout.flush()

    batch_tensors = process_batch(batch_graphs, args)
    print("Saving..")
    torch.save(batch_tensors, os.path.join(savedir, f'batch-{batch_num}.pt'))


def main():
    parser = argparse.ArgumentParser(description='Prepare KGCL tensors from preprocessed reactions')
    add_config_argument(parser)
    add_arguments(parser, ['dataset', 'root_dir', 'mode', 'use_rxn_class', 'preprocess_batch_size', 'max_steps', 'preprocess_print_every', 'kekulize'])
    parsed = parser.parse_args()
    overrides = vars(parsed).copy()
    config_file = overrides.pop('config_file')
    args = load_config(config_file=config_file, cli_overrides=overrides)
    # Backward-compatible attribute names used by the original implementation.
    args.batch_size = args.preprocess_batch_size
    args.print_every = args.preprocess_print_every
    prepare_data(args=args)


if __name__ == "__main__":
    main()
