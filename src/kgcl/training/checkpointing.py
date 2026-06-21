from __future__ import annotations

def save_checkpoint(model, path, epoch):
    import torch
    save_dict = {'state': model.state_dict()}
    if hasattr(model, 'get_saveables'):
        save_dict['saveables'] = model.get_saveables()
    torch.save(save_dict, path / f'epoch_{epoch + 1}.pt')
