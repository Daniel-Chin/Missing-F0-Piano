from subprocess import Popen, DEVNULL
from itertools import product

from disklavier import playMidiOnDisklavier
from solve import solve
from missing_f0_midi_writer import MissingF0MidiWriter

PITCH = 40
PERCEPTION_TOLERANCES = (0.25, )
PENALIZE_STRANGERSES = (0.0, )

USE_SYNTH_NOT_DISKLAVIER = True

TEMP_MIDI = 'temp/%d.mid'

def main():
    def experiments():
        return enumerate(product(
            PERCEPTION_TOLERANCES, PENALIZE_STRANGERSES, 
        ))
    for i, (perception_tolerance, penalize_strangers) in experiments():
        writer = MissingF0MidiWriter()
        print('solving... ', end='', flush=True)
        powers, _ = solve(
            PITCH, perception_tolerance, penalize_strangers, 
            # verbose=True,
        )
        print('ok')
        writer.registerSolution(PITCH, powers)
        writer.addRaw(PITCH + 12, 0.0, 0.5)
        writer.add(PITCH, 1.0, 2.0)
        writer.midi.write(TEMP_MIDI % i)

    input('Press Enter to play...')

    for i, (perception_tolerance, penalize_strangers) in experiments():
        print(f'{perception_tolerance = }')
        print(f'{penalize_strangers = }')
        if USE_SYNTH_NOT_DISKLAVIER:
            with Popen([
                'synth-midi', TEMP_MIDI % i, 
            ], stdout=DEVNULL, stderr=DEVNULL) as p:
                p.wait()
        else:
            playMidiOnDisklavier(TEMP_MIDI % i, verbose=False)

if __name__ == '__main__':
    main()
