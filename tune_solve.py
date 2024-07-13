from time import sleep

import mido
from torch import Tensor

from music import PIANO_RANGE
from disklavier import Disklavier
from solve import solve
from power_to_velocity import power2velocity

PITCH = 40

def play(port, pitch: int, velocities: Tensor):
    for pitch, velocity_tensor in zip(
        range(PITCH + 1, PIANO_RANGE.stop), velocities, 
    ):
        velocity: int = velocity_tensor.item()  # type: ignore
        if velocity > 0:
            port.send(mido.Message(
                'note_on', note=pitch, velocity=velocity, 
            ))
    sleep(3)
    port.panic()

def main():
    with Disklavier() as port:
        for perception_tolerance in (0.1, ):
            print(f'{perception_tolerance = }')
            for forgive_strangers in (1.0, ):
                print(f'{forgive_strangers = }')
                powers, _ = solve(PITCH, perception_tolerance, forgive_strangers)
                velocities = power2velocity(powers)
                play(port, PITCH, velocities)
