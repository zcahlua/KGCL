from __future__ import annotations
import random

def train_epoch(args, epoch, model, train_data, objectives, optimizer, *, device: str):
    import torch
    import torch.nn as nn
    from models.model_utils import get_seq_edit_accuracy
    if len(train_data) == 0:
        raise ValueError('Training dataset is empty; run preprocessing/prepare-data and check data paths.')
    torch.cuda.empty_cache()
    model.train(); train_loss = 0; train_acc = 0
    seq_loss_fn = objectives['sequence']
    loss_adnce = objectives['adnce_with_rxn_class'] if args.get('use_rxn_class', False) else objectives['adnce_without_rxn_class']
    adnce_weight = 0.4 if args.get('use_rxn_class', False) else 0.3
    for batch_id, batch_data in enumerate(train_data):
        graph_seq_tensors, seq_labels, seq_mask = batch_data
        seq_mask = seq_mask.to(device)
        seq_edit_scores, batch_graph_outs = model(graph_seq_tensors)
        max_seq_len, batch_size = seq_mask.size(); seq_losses = []
        for idx in range(max_seq_len):
            edit_labels_idx = model.to_device(seq_labels[idx])
            loss_batch = [seq_mask[idx][i] * seq_loss_fn(seq_edit_scores[idx][i].unsqueeze(0), torch.argmax(edit_labels_idx[i]).unsqueeze(0).long()).sum() for i in range(batch_size)]
            seq_losses.append(torch.stack(loss_batch, dim=0).mean())
        p_features = torch.stack([element for element in batch_graph_outs[0]])
        r_feature_list = []
        seq_mask_list = seq_mask.tolist()
        for col_idx in range(batch_size):
            first_element = batch_graph_outs[0][col_idx]
            non_zero_elements = [row[col_idx] for row_idx, row in enumerate(batch_graph_outs) if not torch.equal(row[col_idx], first_element) and seq_mask_list[row_idx][col_idx]]
            r_feature_list.append(random.choice(non_zero_elements) if non_zero_elements else first_element)
        r_features = torch.stack(r_feature_list)
        total_loss = torch.stack(seq_losses).mean() + adnce_weight * loss_adnce(p_features, r_features)
        accuracy = get_seq_edit_accuracy(seq_edit_scores, seq_labels, seq_mask)
        train_loss += total_loss.item(); train_acc += accuracy
        optimizer.zero_grad(); total_loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), args['max_clip']); optimizer.step()
        if (batch_id + 1) % args['print_every'] == 0:
            print('\repoch %d/%d, batch %d/%d, loss: %.4f, accuracy: %.4f' % (epoch + 1, args['epochs'], batch_id + 1, len(train_data), train_loss / (batch_id + 1), train_acc / (batch_id + 1)), end='', flush=True)
    train_loss = float('%.4f' % (train_loss / len(train_data))); train_acc = float('%.4f' % (train_acc / len(train_data)))
    print('\nepoch %d/%d, train loss: %.4f, train accuracy: %.4f' % (epoch + 1, args['epochs'], train_loss, train_acc))
    return train_loss, train_acc

def validate(model, valid_data):
    import torch
    if len(valid_data) == 0:
        raise ValueError('Validation dataset is empty; run preprocessing/prepare-data and check validation paths.')
    model.eval(); total_accuracy = 0.0; first_step_accuracy = 0.0
    with torch.no_grad():
        for batch_data in valid_data:
            prod_smi_batch, edits_batch, edits_atom_batch, rxn_classes = batch_data
            for idx, prod_smi in enumerate(prod_smi_batch):
                if rxn_classes is None: edits, edits_atom = model.predict(prod_smi)
                else: edits, edits_atom = model.predict(prod_smi, rxn_class=rxn_classes[idx])
                if edits == edits_batch[idx] and edits_atom == edits_atom_batch[idx]: total_accuracy += 1.0
                if edits[0] == edits_batch[idx][0] and edits_atom[0] == edits_atom_batch[idx][0]: first_step_accuracy += 1.0
    return float('%.4f' % (total_accuracy / len(valid_data))), float('%.4f' % (first_step_accuracy / len(valid_data)))
