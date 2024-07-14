from os import path
from subprocess import Popen

import pretty_midi
import torchaudio
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch

from music import PIANO_RANGE

NOTE_TIME = 0.5

TEMP_MID = './temp/ePiano.mid'
TEMP_WAV = './temp/ePiano.wav'

def compose():
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(0, name='Piano')
    midi.instruments.append(piano)
    cursor = 0.0
    for pitch in PIANO_RANGE:
        for velocity in reversed(range(128)):
            note = pretty_midi.Note(
                velocity=velocity, pitch=pitch, 
                start=cursor, end=cursor + NOTE_TIME, 
            )
            cursor += NOTE_TIME
            piano.notes.append(note)
    midi.write(TEMP_MID)

def synth():
    with Popen([
        'fluidsynth', 
        '-ni', 
        path.expanduser('~/roaming_linux_daniel/soundfonts/Yamaha-Grand-Lite-v2.0.sf2'), 
        TEMP_MID, 
        '-F', TEMP_WAV, 
        '-r', '44100', 
    ]) as p:
        p.wait()

def analyze():
    print('load audio...')
    stereo, sr = torchaudio.load(TEMP_WAV)
    mono = stereo.mean(dim=0)
    cursor = 0.0
    results = []
    for pitch in tqdm(PIANO_RANGE):
        velocities = torch.zeros(128)
        powers = torch.zeros(128)
        for i, velocity in enumerate(reversed(range(128))):
            start = round(sr * (cursor + 0.1))
            stop  = round(sr * (cursor + NOTE_TIME - 0.01))
            power = mono[start:stop].square().mean()
            velocities[i] = velocity
            powers[i] = power
            cursor += NOTE_TIME
        results.append((pitch, velocities, powers / powers.mean()))
    
    for pitch, velocities, power_ratios in results:
        k = (pitch - PIANO_RANGE.start) / len(PIANO_RANGE)
        c = (k, 1.0 - k, 0.0)
        plt.plot(velocities, power_ratios, c=c, label=pitch)
    X = torch.arange(128, dtype=torch.float)
    Y = X.pow(4)
    # Y = (X / 128).exp()
    Y = Y / Y.mean()
    plt.plot(X, Y, 'o', c='k', label='fit')
    plt.xlabel('Velocity')
    plt.ylabel('Power')
    # plt.legend()
    plt.show()

if __name__ == '__main__':
    compose()
    synth()
    analyze()
