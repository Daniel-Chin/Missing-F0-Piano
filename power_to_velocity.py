import torch
from torch import Tensor

from piecewise_linear import PiecewiseLinear

from measure_player_piano import MIN_POWER

# results from ./measure_player_piano.py
MILESTONES = torch.tensor([
    (  1, MIN_POWER), 
    ( 20, 0.0678832 / 2.9), 
    ( 30, 0.1436872 / 2.9), 
    ( 40, 0.2313259 / 2.9), 
    ( 50, 0.3359315 / 2.9), 
    ( 60, 0.5382606 / 2.9), 
    ( 70, 0.8646125 / 2.9), 
    ( 80, 1.3882559 / 2.9), 
    ( 90, 1.8521190 / 2.9), 
    (100, 2.5902359 / 2.9), 
    (110, 2.8861124 / 2.9), 
    (127, 1.0), 
])

piecewiseLinear = PiecewiseLinear(
    MILESTONES[:, 1], MILESTONES[:, 0], 
)

def power2velocity(power: Tensor):
    x = piecewiseLinear.forward(power)
    x = x.round().clamp(1, 127).to(torch.int)
    x[power < MIN_POWER] = 0
    return x
