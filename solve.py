from functools import lru_cache
from itertools import count
import typing as tp

import torch
from torch import Tensor
from matplotlib import pyplot as plt
from tqdm import tqdm

from shared import *
from music import *
from human_freq_response import logFreqResponseDb, HUMAN_RANGE
from measure_player_piano import MIN_POWER

DEFAULT_PERCEPTION_TOLERANCE = 0.1
DEFAULT_PENALIZE_STRANGERS = 1.0
DEFAULT_LR = 4e-2

FREE_BUT_SMALL = 0.05

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

@lru_cache(len(GUI_PITCHES))    # don't mutate out tensor
def discretize(
    target_pitch: int, perception_tolerance: float, 
):
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
                ).exp().squeeze(0)
    response_envelope = logFreqResponseDb(pitch2freq_batch(
        target_overtones, 
    ).log()).exp()
    return contributions, response_envelope, pitches_above

class Trainee:
    def __init__(
        self, 
        target_pitch: int, 
        perception_tolerance: float, 
        penalize_strangers: float, 
        lr: float, 
        activations: Tensor | None = None,
        lock_mask  : Tensor | None = None,
        lock_value : Tensor | None = None,
    ):
        self.target_pitch = target_pitch
        self.perception_tolerance = perception_tolerance
        self.penalize_strangers = penalize_strangers
        self.lr = lr

        contributions, response_envelope, pitches_above = discretize(
            target_pitch, perception_tolerance, 
        )
        
        self.contributions = contributions.to(device())
        self.response_envelope = response_envelope.to(device())

        if activations is None:
            self.activations = torch.ones(
                (len(pitches_above), ), device=device(), 
            ) * FREE_BUT_SMALL
        else:
            self.activations = activations
        self.activations.requires_grad = True
        self.optim = torch.optim.Adam([self.activations], lr=lr)

        if lock_mask is None:
            self.lock_mask = torch.ones_like(self.activations)
        else:   # 0: locked, 1: unlocked
            self.lock_mask = lock_mask
        if lock_value is None:
            self.lock_value = torch.zeros_like(self.activations)
        else:
            self.lock_value = lock_value
    
    def clone(self):
        activations = self.activations.detach().clone()
        activations.requires_grad = True
        return Trainee(
            self.target_pitch,
            self.perception_tolerance, 
            self.penalize_strangers,
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
    
    def breakFree(self):
        THRESHOLD = FREE_BUT_SMALL
        with torch.no_grad():
            self.activations[torch.logical_and(
                self.activations < THRESHOLD, 
                self.lock_mask == 1.0, 
            )] = THRESHOLD

    def forward(self):
        powers = self.activations.square()
        produced = self.contributions.T @ powers
        loss_needed = (
            (produced[:-1] + 1e-4).log().square() * self.response_envelope
        ).sum()
        loss_stranger_cool = (produced[-1] + 1e-6).log()
        loss_stranger_adhoc = produced[-1] * self.penalize_strangers
        return (
            loss_needed + 
            loss_stranger_cool + 
            loss_stranger_adhoc
        ), produced.detach()

    def oneEpochGivenForward(
        self, loss: Tensor, 
    ):
        self.optim.zero_grad()
        loss.backward()
        assert self.activations.grad is not None
        self.activations.grad.mul_(self.lock_mask)
        self.optim.step()
        self.applyLock()
        return self.activations.grad

    def train(self, verbose: bool = False):
        losses = []
        minimum = (self.activations, torch.inf)
        boredom = 0
        counter = count()
        if verbose:
            counter = tqdm(counter, desc='epoch')
        for epoch in counter:
            loss, produced = self.forward()
            loss_item = loss.detach().item()
            losses.append(loss_item)
            new_low = loss_item < minimum[1]
            if new_low:
                minimum = (self.activations.detach().clone(), loss_item)
            grad = self.oneEpochGivenForward(loss)
            max_grad = grad.abs().max().item()
            if new_low and max_grad > 0.005:
                boredom = 0
            else:
                boredom += 1
                if boredom > 20:
                    break
            if epoch > 100000:
                plt.plot(losses)
                plt.title('loss')
                plt.show()
                print(f'{max_grad = }')
                import pdb; pdb.set_trace()
        self.activations = minimum[0]
        powers = self.activations.square()
        assert (powers < 1.0).all(), powers
        return minimum[1], produced

def tuneLR(target_pitch: int, try_lrs: Tensor, colors: tp.Iterable[str]):
    for try_lr, c in zip(try_lrs, colors):
        label = f'lr={try_lr:.2e}'
        trainee = Trainee(
            target_pitch, 
            DEFAULT_PERCEPTION_TOLERANCE, 
            DEFAULT_PENALIZE_STRANGERS, 
            try_lr.item(), 
        )
        losses = []
        for _ in tqdm([*range(1000)], desc=label):
            loss, _ = trainee.forward()
            trainee.oneEpochGivenForward(loss)
            losses.append(loss.item())
        plt.plot(losses, label=label, c=c)
    plt.legend()
    plt.title('Loss')
    plt.show()

def solve(
    target_pitch: int, 
    perception_tolerance: float = DEFAULT_PERCEPTION_TOLERANCE, 
    penalize_strangers: float = DEFAULT_PENALIZE_STRANGERS, 
    lr: float = DEFAULT_LR, 
    verbose: bool = False,
):
    '''
    `perception_tolerance`: in semitones.  
    `penalize_strangers`: how much to penalize the strangers, i.e., produced undesired frequencies.  
    '''
    candidates = [Trainee(
        target_pitch, perception_tolerance,
        penalize_strangers, lr, 
    )]
    while True:
        losses = torch.tensor([x.train()[0] for x in candidates])
        winner_i: int = losses.argmin().item()  # type: ignore
        winner = candidates[winner_i]
        assert isinstance(winner, Trainee)  # dumb type checker doesn't take the LFS...
        if verbose:
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
            lock_0  .breakFree()
            lock_min.breakFree()
            candidates = [lock_0, lock_min]
    
    return powers, winner

def inspect(pitch: int):
    powers, trainee = solve(pitch, verbose=True)
    powers = powers.cpu()

    print(powers)

    plt.bar(*zip(*enumerate(powers)))
    plt.show()

    _, produced = trainee.forward()
    plt.plot(produced[:-1].cpu(), 'o')
    plt.axhline(1.0, c='r')
    plt.show()

if __name__ == '__main__':
    # try_lrs = torch.linspace(-3, -2, 6).exp()
    # colors = 'rygcbm'
    # tuneLR(72, try_lrs, colors)
    # tuneLR(60, try_lrs, colors)
    # tuneLR(48, try_lrs, colors)

    inspect(72)
    inspect(60)
    inspect(48)
