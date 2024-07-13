from subprocess import Popen, PIPE, DEVNULL
import signal
import typing as tp
import random
import json

import torch
import mido

from abs_sleep import AbsSleep

from music import *

TIME_SUSTAIN = 0.7
TIME_REST = 0.3
TIME_MUTE = 8.0 # wait for the high register (without damper felt) to decay. Needed when the next event is quieter than the last.  

VELOCITIES = [*range(0, 128, 8), 127]

INPUT = ''
KEYWORD = 'Disklavier'

LOG_FILE = './log.json'
RECORD_FILE = './piano_measure.wav'

def findDevice(devices: tp.List[str]):
    matched = [x for x in devices if KEYWORD.lower() in x.lower()]
    assert len(matched) == 1, devices
    return matched[0]

def playPiano(absSleep: AbsSleep):
    outs = mido.get_output_names()  # type: ignore
    device = findDevice(outs)
    with mido.open_output(device) as port:  # type: ignore
        with open(LOG_FILE, 'w') as f:
            while True:
                try:
                    pitches: tp.List[int] = (torch.randn((
                        random.randint(1, 6), 
                    )) * 18 + 60).clamp(
                        PIANO_RANGE.start + 2, 
                        PIANO_RANGE.stop  - 2, 
                    ).round().to(torch.int).tolist()
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
                    json.dump(pitches, f, indent=2)
                    f.write('\n')
                except KeyboardInterrupt:
                    print('bye')
                    port.panic()
                    break

def main():
    print('Make sure the piano volume is at 4/5: OOOO_')
    input('Press Enter to confirm >')
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

if __name__ == '__main__':
    main()
