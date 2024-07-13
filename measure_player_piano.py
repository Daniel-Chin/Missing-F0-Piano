from subprocess import Popen, PIPE, DEVNULL
from time import sleep
import typing as tp
import random
import json

import torch
import torchaudio
import mido
from tqdm import tqdm
from matplotlib import pyplot as plt

from abs_sleep import AbsSleep

from music import *
from disklavier import DISKLAVIER, Disklavier

TIME_SUSTAIN = 0.7
TIME_REST = 0.3
TIME_MUTE = 7.0 # wait for the high register (without damper felt) to decay. Needed when the next event is quieter than the last.  

VELOCITIES = [
    1, 2, 3, 5, 10, 
    *range(20, 121, 10), 
    127, 
]

MAX_CHORD_SIZE = 6
COLORS = 'rygcbm'
assert len(COLORS) >= MAX_CHORD_SIZE

LOG_FILE = './log.json'
RECORD_FILE = './piano_measure.wav'

def playPiano(absSleep: AbsSleep):
    with Disklavier() as port:
        with open(LOG_FILE, 'w') as f:
            while True:
                try:
                    pitches: tp.Set[int] = set((torch.randn((
                        random.randint(1, MAX_CHORD_SIZE), 
                    )) * 18 + 60).clamp(
                        PIANO_RANGE.start + 2, 
                        PIANO_RANGE.stop  - 2, 
                    ).round().to(torch.int).tolist())
                    print(pitches)
                    for velocity in VELOCITIES:
                        absSleep.sleep(TIME_REST)
                        for pitch in pitches:
                            msg = mido.Message(
                                'note_on', note=pitch, velocity=velocity, 
                            )
                            port.send(msg)
                        absSleep.sleep(TIME_SUSTAIN)
                        port.panic()
                    absSleep.sleep(TIME_MUTE)
                    json.dump(list(pitches), f)
                    f.write('\n')
                except KeyboardInterrupt:
                    print('bye')
                    port.panic()
                    break

def confirm(msg: str):
    print()
    assert input(msg + ' y/n >').lower() == 'y'

def takeData():
    confirm('Make sure the piano volume is at 2.5/5.0   LED: OO*__')
    confirm('Mute the laptop.')
    confirm("The laptop should stay awake.")
    confirm('Get ready to retreat to a non-recorded zone.')
    for _ in tqdm(range(100), desc='迅速撤离'):
        sleep(0.03)
    with Popen([
        '/usr/bin/ffmpeg', 
        '-f', 'alsa', 
        '-i', 'default', 
        '-y', 
        RECORD_FILE, 
    ], stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL) as p:
        absSleep = AbsSleep()
        try:
            playPiano(absSleep)
        finally:
            if p.stdin is not None:
                p.stdin.write(b'q')
                p.stdin.flush()
                p.stdin.close()
            p.wait()
    print('ok')

def analyze():
    START = 1.1
    TAKE = 0.4

    with open(LOG_FILE) as f:
        pitches_tape = [json.loads(line) for line in f]
    print('loading audio...')
    stereo, sr = torchaudio.load(RECORD_FILE)
    print(f'{sr = }')
    mono = stereo.mean(dim=0)
    n_samples, = mono.shape
    print('audio time (sec):', n_samples / sr)
    def trim(start: float):
        return mono[
            round(sr * start) : 
            round(sr * (start + TAKE))
        ]

    results: tp.List[tp.Tuple[
        tp.List[int], Tensor, 
    ]] = []
    cursor = START
    for pitches in tqdm(pitches_tape):
        pitches: tp.List[int]
        powers = torch.zeros((len(VELOCITIES), ))
        for i in range(len(VELOCITIES)):
            clip = trim(cursor)
            powers[i] = clip.square().mean()
            cursor += TIME_REST + TIME_SUSTAIN
        ratios = powers / powers.mean()
        results.append((sorted(pitches), ratios))
        cursor += TIME_MUTE
    
    for pitches, ratios in sorted(results, key=lambda x: len(x[0])):
        color = COLORS[len(pitches) - 1]
        plt.plot(VELOCITIES, ratios, label=str(pitches), c=color)
    mean = torch.stack([ratios for _, ratios in results], dim=1).mean(dim=1)
    plt.plot(VELOCITIES, mean, label='mean', c='k', linestyle='--', linewidth=3)
    plt.legend()
    plt.xlabel('Velocity')
    plt.ylabel('Power Ratio')
    plt.title(DISKLAVIER)
    for v, m in zip(VELOCITIES, mean):
        print(v, ', ', m.item(), sep='')
    plt.show()

# eyeball result
MIN_POWER = 0.019

if __name__ == '__main__':
    print(f'{VELOCITIES = }')
    print('unit duration (sec):', (
        TIME_SUSTAIN + TIME_REST
    ) * len(VELOCITIES) + TIME_MUTE)

    # takeData()
    analyze()
