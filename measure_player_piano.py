from subprocess import Popen
import signal
import typing as tp
from time import sleep, perf_counter
import random

import torch
import mido

TIME_SUSTAIN = 0.7
TIME_REST = 0.3
TIME_MUTE = 2.0 # wait for the high register (without damper felt) to decay. Needed when the next event is quieter than the last.  

VELOCITIES = [*range(0, 128, 8), 127]

INPUT = ''
KEYWORD = 'Disklavier'

RECORD_FILE = './piano_measure.wav'

def findDevice(devices: tp.List[str]):
    matched = [x for x in devices if KEYWORD.lower() in x.lower()]
    assert len(matched) == 1, devices
    return matched[0]

def playPiano():
    outs = mido.get_output_names()  # type: ignore
    device = findDevice(outs)
    with mido.open_output(device) as port:  # type: ignore
        while True:
            try:
                pitches = (torch.randn((random.randint(2, 6), )) * 18 + 60).round()
                print(pitches)
                start_time = perf_counter()
                for velocity in VELOCITIES:
                    for pitch_tensor in pitches:
                        pitch: int = pitch_tensor.item()    # type: ignore
                        msg = mido.Message(
                            'note_on', note=pitch, velocity=velocity, 
                        )
                        port.send(msg)
                    sleep(TIME_SUSTAIN)
                    port.panic()
                    sleep(TIME_REST)
                sleep(TIME_MUTE)
                dt = perf_counter() - start_time
                print('elapsed sec:', dt)
            except KeyboardInterrupt:
                print('bye')
                break

def main():
    with Popen([
        '/usr/bin/ffmpeg', '-f', 'alsa', '-i', 'default', 
        RECORD_FILE, 
    ]) as p:
        playPiano()
        p.send_signal(signal.SIGINT)
        p.wait()
    print('ok')

if __name__ == '__main__':
    main()
