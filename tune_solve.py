from subprocess import Popen, DEVNULL
from itertools import product

from disklavier import playMidiOnDisklavier
from solve import solve
from missing_f0_midi_writer import MissingF0MidiWriter

PITCH = 48
PERCEPTION_TOLERANCES = (6e-2, )
PENALIZE_STRANGERSES = (7.4, )

USE_SYNTH_NOT_DISKLAVIER = True

TEMP_MIDI = 'temp/tune_solve.mid'

def main():
    def experiments():
        return enumerate(product(
            PERCEPTION_TOLERANCES, PENALIZE_STRANGERSES, 
        ))
    writer = MissingF0MidiWriter()
    cursor = 0.0
    for i, (perception_tolerance, penalize_strangers) in experiments():
        print('solving... ', end='', flush=True)
        powers, _ = solve(
            PITCH, perception_tolerance, penalize_strangers, 
            # verbose=True,
        )
        print('ok')
        writer.registerSolution(PITCH, powers)
        writer.addRaw(PITCH + 12, cursor, cursor + 0.5)
        cursor += 1.0
        writer.add(PITCH, cursor, cursor + 1.0)
        cursor += 2.0
    writer.midi.write(TEMP_MIDI)

    input('Press Enter to play...')

    for i, (perception_tolerance, penalize_strangers) in experiments():
        print()
        print(f'{perception_tolerance = }')
        print(f'{penalize_strangers = }')
    if USE_SYNTH_NOT_DISKLAVIER:
        with Popen([
            'synth-midi', TEMP_MIDI, 
        ], stdout=DEVNULL, stderr=DEVNULL) as p:
            p.wait()
    else:
        playMidiOnDisklavier(TEMP_MIDI, verbose=False)

if __name__ == '__main__':
    main()
