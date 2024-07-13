import typing as tp
from functools import lru_cache
from itertools import count

import torch
from torch import Tensor
from matplotlib import pyplot as plt
from tqdm import tqdm

from music import *
from human_freq_response import logFreqResponseDb, HUMAN_RANGE
from measure_player_piano import MIN_POWER

DEFAULT_PERCEPTION_TOLERANCE = 0.1
DEFAULT_FORGIVE_STRANGERS = 1.0
DEFAULT_LR = 2e-2

@lru_cache(1)
def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def getOvertones(pitch: int):
    f0 = pitch2freq(pitch)
    return freq2pitch_batch(torch.arange(
        f0, HUMAN_RANGE[1], f0, 
    ))

def solve(
    target_pitch: int, 
    perception_tolerance: float = DEFAULT_PERCEPTION_TOLERANCE, 
    forgive_strangers: float = DEFAULT_FORGIVE_STRANGERS, 
    lr: float = DEFAULT_LR,
):
    '''
    `perception_tolerance`: in semitones.  
    `forgive_strangers`: how much to forgive the strangers, i.e., produced undesired frequencies.  
    '''
    assert target_pitch in PIANO_RANGE
    target_overtones = getOvertones(target_pitch)[1:]
    pitches_above = range(target_pitch + 1, PIANO_RANGE.stop)
    contributions = torch.zeros((
        len(pitches_above), 
        len(target_overtones) + 1, # last one for strangers
    ))
    for i_above, pitch in enumerate(pitches_above):
        for overtone in getOvertones(pitch):
            deviate = (target_overtones - overtone).abs()
            i: int = deviate.argmin().item()   # type: ignore
            if deviate[i] < perception_tolerance:
                contributions[i_above, i] += 1.0
            else:
                contributions[i_above, -1] += logFreqResponseDb(
                    pitch2freq_batch(overtone).log().unsqueeze(0), 
                ).exp()[0]
    
    # GD
    contributions = contributions.to(device())
    def init(lr_: float):
        activations = torch.ones(
            (len(pitches_above), ), requires_grad=True, 
            device=device(), 
        )
        optim = torch.optim.Adam([activations], lr=lr_)
        return activations, optim
    response_envelope = logFreqResponseDb(pitch2freq_batch(
        target_overtones, 
    ).log()).exp().to(device())
    def getLoss(activations: Tensor):
        powers = activations.square()
        produced = contributions.T @ powers
        loss_needed = (
            (produced[:-1] + 1e-4).log().square() * response_envelope
        ).sum()
        loss_stranger = (produced[-1] + forgive_strangers).log()
        return loss_needed + loss_stranger
    def oneEpoch(activations: Tensor, optim: torch.optim.Optimizer):
        loss = getLoss(activations)
        optim.zero_grad()
        loss.backward()
        grad = activations.grad
        optim.step()
        return loss, grad

    # tune lr
    # for try_lr, c in zip(
    #     torch.linspace(-4, -1, 6).exp(), 
    #     'rygcbm', 
    # ):
    #     label = f'lr={try_lr:.2e}'
    #     activations, optim = init(try_lr.item())
    #     losses = []
    #     for _ in tqdm([*range(1000)], desc=label):
    #         loss, grad = oneEpoch(activations, optim)
    #         losses.append(loss.item())
    #     plt.plot(losses, label=label, c=c)
    # plt.legend()
    # plt.show()
    # return

    activations, optim = init(lr)
    for epoch in count():
        loss, grad = oneEpoch(activations, optim)
        if grad is not None and grad.abs().max() < 0.005:
            print('converged at epoch', epoch)
            break
    assert (activations.square() < 1.0).all()

if __name__ == '__main__':
    solve(72)
    solve(60)
    solve(48)
