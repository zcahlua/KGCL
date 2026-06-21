from __future__ import annotations
from datetime import datetime

from kgcl.config.device import resolve_device
from kgcl.config.paths import ProjectPaths
from kgcl.resources.functional_groups import configure_functional_group_resources
from kgcl.training.context import TrainingContext


def build_model_config(args):
    from utils.mol_features import ATOM_FDIM, BOND_FDIM
    atom_fdim = ATOM_FDIM + 10 if args.get('use_rxn_class', False) else ATOM_FDIM
    return {
        'n_atom_feat': atom_fdim,
        'n_bond_feat': BOND_FDIM if args.get('atom_message', False) else atom_fdim + BOND_FDIM,
        'mpn_size': args['mpn_size'], 'mlp_size': args['mlp_size'], 'depth': args['depth'],
        'dropout_mlp': args['dropout_mlp'], 'dropout_mpn': args['dropout_mpn'],
        'atom_message': args['atom_message'], 'use_attn': args['use_attn'], 'n_heads': args['n_heads'],
    }


def run_training(config):
    if hasattr(config, 'to_dict'):
        args = config.to_dict()
    else:
        args = dict(config)
    if args.get('train_batch_size', 1) != 1:
        raise ValueError('train_batch_size greater than 1 is not supported; serialized shard DataLoader batch size is fixed at 1. Use preprocess_batch_size to control reaction shard size.')
    paths = ProjectPaths(args.get('root_dir', './'))
    configure_functional_group_resources(resource_root=args.get('resource_root'), root_dir=paths.root)

    import sys
    import joblib
    import torch
    from rdkit import RDLogger
    from torch.optim import Adam, lr_scheduler
    from models import KGCL
    from models.model_utils import CSVLogger
    from utils.datasets import RetroEditDataset, RetroEvalDataset
    from utils.rxn_graphs import Vocab
    from kgcl.training.checkpointing import save_checkpoint
    from kgcl.training.engine import train_epoch, validate
    from kgcl.training.objectives import build_objectives

    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    device = resolve_device(args.get('device', 'auto'))
    if args.get('lr') is None and args['dataset'] == 'uspto_50k': args['lr'] = 0.001
    elif args.get('lr') is None and args['dataset'] == 'uspto_full': args['lr'] = 0.0001

    run_started_at = datetime.now()
    out_dir = paths.experiment_dir(args['dataset'], args.get('use_rxn_class', False), run_started_at.strftime('%d-%m-%Y--%H-%M-%S'))
    out_dir.mkdir(parents=True, exist_ok=True)
    context = TrainingContext(device=device, paths=paths, run_started_at=run_started_at, output_dir=out_dir)

    csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'valid_acc', 'valid_first_step_acc', 'train_loss'], filename=str(context.output_dir / 'logs.csv'))
    bond_vocab = Vocab(joblib.load(paths.vocab_file(args['dataset'], 'bond_vocab.txt')))
    atom_vocab = Vocab(joblib.load(paths.vocab_file(args['dataset'], 'atom_lg_vocab.txt')))
    train_dir = paths.prepared_shard_dir(args['dataset'], 'train', args.get('use_rxn_class', False))
    eval_dir = paths.split_dir(args['dataset'], 'valid')
    train_data = RetroEditDataset(data_dir=str(train_dir)).loader(batch_size=1, num_workers=args['num_workers'], shuffle=True)
    valid_file = paths.serialized_reactions_file(args['dataset'], 'valid', args.get('kekulize', True)).name
    valid_data = RetroEvalDataset(data_dir=str(eval_dir), data_file=valid_file, use_rxn_class=args['use_rxn_class']).loader(batch_size=1, num_workers=args['num_workers'])
    if len(train_data) == 0:
        raise ValueError(f'Training dataset is empty: {train_dir}')
    if len(valid_data) == 0:
        raise ValueError(f'Validation dataset is empty: {eval_dir / valid_file}')

    model = KGCL(config=build_model_config(args), atom_vocab=atom_vocab, bond_vocab=bond_vocab, device=device)
    print(f'Converting model to device: {device}'); sys.stdout.flush(); model.to(device)
    print('Param Count: ', sum([x.nelement() for x in model.parameters()]) / 10 ** 6, 'M'); print()
    objectives = build_objectives()
    optimizer = Adam(model.parameters(), lr=args['lr'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=args['patience'], factor=args['factor'], threshold=args['thresh'], threshold_mode='abs')
    best_acc = 0
    for epoch in range(args['epochs']):
        train_loss, train_acc = train_epoch(args, epoch, model, train_data, objectives, optimizer, device=device)
        valid_acc, valid_first_step_acc = validate(model, valid_data)
        scheduler.step(valid_acc)
        print('epoch %d/%d, validation accuracy: %.4f, validation_first_acc: %.4f' % (epoch + 1, args['epochs'], valid_acc, valid_first_step_acc))
        print('---------------------------------------------------------'); print()
        csv_logger.writerow({'epoch': str(epoch + 1), 'train_acc': str(train_acc), 'valid_acc': str(valid_acc), 'valid_first_step_acc': str(valid_first_step_acc), 'train_loss': str(train_loss)})
        if valid_acc >= best_acc:
            print(f'Best eval accuracy so far. Saving best model from epoch {epoch + 1} (acc={valid_acc})')
            print('---------------------------------------------------------'); print(); save_checkpoint(model, context.output_dir, epoch); best_acc = valid_acc
    csv_logger.close(); print('Experiment finished!')
