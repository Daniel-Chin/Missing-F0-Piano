from subprocess import Popen, DEVNULL
from itertools import product

from disklavier import playMidiOnDisklavier
from solve import solve
from missing_f0_midi_writer import MissingF0MidiWriter

PITCH = 50
PERCEPTION_TOLERANCES = (0.2, )
FORGIVE_STRANGERSES = (1.0, 2.0, 4.0)
USE_SYNTH_NOT_DISKLAVIER = True

TEMP_MIDI = 'temp/%d.mid'

def main():
    iter_ = enumerate(product(
        PERCEPTION_TOLERANCES, FORGIVE_STRANGERSES, 
    ))
    for i, (perception_tolerance, forgive_strangers) in iter_:
        writer = MissingF0MidiWriter()
        print('solving... ', end='', flush=True)
        powers, _ = solve(
            PITCH, perception_tolerance, forgive_strangers, 
            verbose=True,
        )
        print('ok')
        writer.registerSolution(PITCH, powers)
        writer.addRaw(PITCH + 12, 0.0, 0.5)
        writer.add(PITCH, 1.0, 3.0)
        writer.midi.write(TEMP_MIDI % i)

    for i, (perception_tolerance, forgive_strangers) in iter_:
        print(f'{perception_tolerance = }')
        print(f'{forgive_strangers = }')
        if USE_SYNTH_NOT_DISKLAVIER:
            with Popen([
                'synth-midi', TEMP_MIDI, 
            ], stdout=DEVNULL, stderr=DEVNULL) as p:
                p.wait()
        else:
            playMidiOnDisklavier(TEMP_MIDI, verbose=False)

if __name__ == '__main__':
    main()
