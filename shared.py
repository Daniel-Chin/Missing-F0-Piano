from functools import lru_cache

import torch

GUI_PITCHES = [40, 48, 60]

@lru_cache(1)
def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# @lru_cache(1)
# def device():
#     return torch.device('cpu')
