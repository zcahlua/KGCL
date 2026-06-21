from __future__ import annotations

def build_objectives():
    import torch.nn as nn
    from utils.ADNCE import ADNCE
    return {
        'sequence': nn.CrossEntropyLoss(reduction='none'),
        'adnce_without_rxn_class': ADNCE(temperature=0.3, reduction='mean', mu=0.5, sigma=1.0),
        'adnce_with_rxn_class': ADNCE(temperature=0.3, reduction='mean', mu=0.3, sigma=1.0),
    }
