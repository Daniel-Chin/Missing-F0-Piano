import math
from functools import lru_cache

from torch import Tensor

from shared import *

PIANO_RANGE = range(21, 109)

def __p2t(pitch):
    return (pitch + 36.37631656229591) * 0.0577622650466621

@lru_cache()
def pitch2freq(pitch: float | int):
    return math.exp(__p2t(pitch))

@lru_cache(len(GUI_PITCHES))    # don't mutate out tensor
def pitch2freq_batch(pitch: Tensor):
    return __p2t(pitch).exp()

def __t2p(t):
    return t * 17.312340490667562 - 36.37631656229591

@lru_cache()
def freq2pitch(f: float):
    return __t2p(math.log(f))

def freq2pitch_batch(f: Tensor):
    return __t2p(f.log())
