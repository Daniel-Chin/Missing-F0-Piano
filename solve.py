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

TUNE_LR = False

DEFAULT_PERCEPTION_TOLERANCE = 0.1
DEFAULT_FORGIVE_STRANGERS = 1.0
DEFAULT_LR = 2e-2

# @lru_cache(1)
# def device():
#     if torch.cuda.is_available():
#         return torch.device('cuda')
#     return torch.device('cpu')

@lru_cache(1)
def device():
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
    tune_lr: bool = TUNE_LR, 
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
    response_envelope = logFreqResponseDb(pitch2freq_batch(
        target_overtones, 
    ).log()).exp().to(device())

    class Trainee:
        def __init__(
            self, lr_: float, 
            activations: Tensor | None = None,
            lock_mask: Tensor | None = None,
            lock_value: Tensor | None = None,
        ):
            self.lr = lr_
            if activations is None:
                self.activations = torch.ones(
                    (len(pitches_above), ), requires_grad=True, 
                    device=device(), 
                )
            else:
                self.activations = activations
            self.optim = torch.optim.Adam([self.activations], lr=lr_)
            if lock_mask is None:
                self.lock_mask = torch.ones_like(self.activations)
            else:
                self.lock_mask = lock_mask
            if lock_value is None:
                self.lock_value = torch.zeros_like(self.activations)
            else:
                self.lock_value = lock_value
        
        def clone(self):
            activations = self.activations.detach().clone()
            activations.requires_grad = True
            return Trainee(
                self.lr, 
                activations, 
                self.lock_mask .clone(), 
                self.lock_value.clone(), 
            )
        
        def applyLock(self):
            with torch.no_grad():
                self.activations.mul_(
                    self.lock_mask, 
                ).add_(self.lock_value)

        def forward(self):
            powers = self.activations.square()
            produced = contributions.T @ powers
            loss_needed = (
                (produced[:-1] + 1e-4).log().square() * response_envelope
            ).sum()
            loss_stranger = (produced[-1] + forgive_strangers).log()
            return loss_needed + loss_stranger, produced.detach()
    
        def oneEpoch(self):
            loss, produced = self.forward()
            self.optim.zero_grad()
            loss.backward()
            assert self.activations.grad is not None
            self.activations.grad.mul_(self.lock_mask)
            self.optim.step()
            self.applyLock()
            return loss.detach(), self.activations.grad, produced

        def train(self):
            flat_combo = 0
            for epoch in tqdm(count()):
                loss, grad, produced = self.oneEpoch()
                if grad is not None and grad.abs().max() < 0.005:
                    flat_combo += 1
                else:
                    flat_combo = 0
                if flat_combo > 10:
                    break
            print('converged at epoch', epoch)
            powers = self.activations.square()
            assert (powers < 1.0).all()
            return loss, produced
        
    if tune_lr:
        for try_lr, c in zip(
            torch.linspace(-4, -1, 6).exp(), 
            'rygcbm', 
        ):
            label = f'lr={try_lr:.2e}'
            trainee = Trainee(try_lr.item())
            losses = []
            for _ in tqdm([*range(1000)], desc=label):
                loss, _, _ = trainee.oneEpoch()
                losses.append(loss.item())
            plt.plot(losses, label=label, c=c)
        plt.legend()
        plt.show()
        return None, None

    candidates = [Trainee(lr)]
    while True:
        losses = torch.tensor([x.train()[0] for x in candidates])
        winner_i = losses.argmin()
        winner: Trainee = candidates[winner_i]
        print('locked', 'down' if winner_i == 0 else 'up')
        with torch.no_grad():
            powers = winner.activations.square()
            unlocked = powers + (1.0 - winner.lock_mask) * 69.0
            legals = (unlocked >= MIN_POWER).float()
            if legals.all():
                break
            illegals = unlocked + legals * 69.0
            near_0 = illegals.argmin()
            near_min = (illegals - MIN_POWER).square().argmin()
            lock_0   = winner.clone()
            lock_min = winner.clone()
            lock_0  .lock_mask [near_0  ] = 0.0
            lock_min.lock_mask [near_min] = 0.0
            lock_0  .lock_value[near_0  ] = 0.0
            lock_min.lock_value[near_min] = MIN_POWER
            lock_0  .applyLock()
            lock_min.applyLock()
            candidates = [lock_0, lock_min]
    
    return powers, winner

def inspect(pitch: int):
    powers, trainee = solve(pitch)
    assert powers  is not None
    assert trainee is not None
    powers = powers.cpu()

    print(powers)

    plt.bar(*zip(*enumerate(powers)))
    plt.show()

    _, produced = trainee.forward()
    plt.plot(produced[:-1].cpu(), 'o')
    plt.axhline(1.0, c='r')
    plt.show()

if __name__ == '__main__':
    inspect(72)
    inspect(60)
    inspect(48)
