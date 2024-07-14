import typing as tp

from torch import Tensor
import pretty_midi

from music import *
from power_to_velocity import power2velocity

class MissingF0MidiWriter:
    def __init__(self):
        self.midi = pretty_midi.PrettyMIDI()
        self.piano = pretty_midi.Instrument(0, name='Piano')
        self.midi.instruments.append(self.piano)
        self.solved: tp.Dict[int, Tensor] = {}
    
    def registerSolution(self, pitch: int, powers: Tensor):
        self.solved[pitch] = powers
    
    def addRaw(
        self, pitch: int, start: float, end: float, 
        velocity: int = 127, 
    ):
        note = pretty_midi.Note(
            velocity=velocity, pitch=pitch, 
            start=start, end=end, 
        )
        self.piano.notes.append(note)
    
    def add(
        self, pitch: int, 
        start: float, end: float, 
        scale_power: float = 1.0, 
    ):
        if scale_power != 1.0:
            print('Warning: scale_power != 1.0 is ill-solved.')
        solution = self.solved[pitch]
        velocities = power2velocity(solution * scale_power)
        for using_pitch, velocity_tensor in zip(
            range(pitch + 1, PIANO_RANGE.stop), velocities, 
        ):
            velocity: int = velocity_tensor.item()  # type: ignore
            if velocity == 0:
                continue
            self.addRaw(using_pitch, start, end, velocity)
