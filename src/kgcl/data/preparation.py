from kgcl.config.paths import ProjectPaths
import sys
import copy
from typing import Any


def process_batch(batch_graphs, *, config, bond_vocab, atom_vocab):
    if not batch_graphs:
        raise ValueError("process_batch requires at least one reaction graph sequence")
    import torch
    from utils.rxn_graphs import MolGraph
    lengths = torch.tensor([len(graph_seq)
                           for graph_seq in batch_graphs], dtype=torch.long)
    max_length = max([len(graph_seq) for graph_seq in batch_graphs])

    from utils.collate_fn import get_batch_graphs, prepare_edit_labels

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
            prod_graphs, use_rxn_class=config.use_rxn_class)

        graph_seq_tensors.append(current_graph_tensors)
        edit_seq_labels.append(edit_labels)
        seq_mask.append(mask)

    seq_mask = torch.stack(seq_mask).long()
    assert seq_mask.shape[0] == max_length
    assert seq_mask.shape[1] == len(batch_graphs)

    return graph_seq_tensors, edit_seq_labels, seq_mask


def prepare_data(config: Any) -> None:
    """ 
    prepare data batches for edits prediction
    """
    import joblib
    import torch
    from rdkit import Chem
    from utils.reaction_actions import Termination
    from utils.rxn_graphs import RxnGraph
    from kgcl.chemistry.edit_application import apply_edit_to_mol
    paths = ProjectPaths(config.root_dir)
    datafile = paths.serialized_reactions_file(config.dataset, config.mode, config.kekulize)
    if not datafile.exists():
        raise FileNotFoundError(f'Missing serialized reaction file: {datafile}')
    rxns_data = joblib.load(datafile)

    from utils.rxn_graphs import Vocab
    bond_vocab = Vocab(joblib.load(paths.vocab_file(config.dataset, 'bond_vocab.txt')))
    atom_vocab = Vocab(joblib.load(paths.vocab_file(config.dataset, 'atom_lg_vocab.txt')))

    batch_graphs = []
    batch_num = 0

    savedir = paths.prepared_shard_dir(config.dataset, config.mode, config.use_rxn_class)
    savedir.mkdir(parents=True, exist_ok=True)

    for idx, rxn_data in enumerate(rxns_data):
        graph_seq = []
        final_smi = None
        rxn_smi = rxn_data.rxn_smi
        r, p = rxn_smi.split('>>')
        r_mol = Chem.MolFromSmiles(r)
        p_mol = Chem.MolFromSmiles(p)
        if config.kekulize:
            Chem.Kekulize(p_mol)

        if len(rxn_data.edits) > config.max_steps:
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
                    int_mol), edit_to_apply=edit, reac_mol=Chem.Mol(r_mol), rxn_class=rxn_data.rxn_class, use_rxn_class=config.use_rxn_class)
                graph_seq.append(graph)
                edit_exe = Termination(action_vocab='Terminate')
                try:
                    pred_mol = edit_exe.apply(Chem.Mol(int_mol))
                    final_smi = Chem.MolToSmiles(pred_mol)
                except Exception:
                    final_smi = None
            else:
                graph = RxnGraph(prod_mol=Chem.Mol(int_mol), edit_to_apply=edit,
                                 edit_atom=rxn_data.edits_atom[i], reac_mol=Chem.Mol(r_mol), rxn_class=rxn_data.rxn_class, use_rxn_class=config.use_rxn_class)
                graph_seq.append(graph)
                int_mol = apply_edit_to_mol(
                    Chem.Mol(int_mol), edit, rxn_data.edits_atom[i])

        if len(graph_seq) == 0 or final_smi is None:
            print(f"No valid states found. Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        batch_graphs.append(graph_seq)
        if (idx % config.preprocess_print_every == 0) and idx:
            print(f"{idx}/{len(rxns_data)} {config.mode} reactions processed.")
            sys.stdout.flush()

        if (len(batch_graphs) % config.preprocess_batch_size == 0) and len(batch_graphs):
            batch_tensors = process_batch(batch_graphs, config=config, bond_vocab=bond_vocab, atom_vocab=atom_vocab)
            torch.save(batch_tensors, savedir / f'batch-{batch_num}.pt')

            batch_num += 1
            batch_graphs = []

    print(f"All {config.mode} reactions complete.")
    sys.stdout.flush()

    if batch_graphs:
        batch_tensors = process_batch(batch_graphs, config=config, bond_vocab=bond_vocab, atom_vocab=atom_vocab)
        print("Saving..")
        torch.save(batch_tensors, savedir / f'batch-{batch_num}.pt')
    else:
        print("No remaining reactions to save.")


def run(args):
    from kgcl.resources.functional_groups import functional_group_resource_context
    with functional_group_resource_context(resource_root=getattr(args, 'resource_root', None), root_dir=getattr(args, 'root_dir', None)):
        prepare_data(config=args)
