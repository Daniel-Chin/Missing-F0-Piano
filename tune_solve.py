from time import sleep

from disklavier import playMidiOnDisklavier
from solve import solve
from missing_f0_midi_writer import MissingF0MidiWriter

PITCH = 40

TEMP_MIDI = 'temp.mid'

def main():
    for perception_tolerance in (0.1, ):
        print(f'{perception_tolerance = }')
        for forgive_strangers in (1.0, ):
            print(f'{forgive_strangers = }')
            writer = MissingF0MidiWriter()
            powers, _ = solve(PITCH, perception_tolerance, forgive_strangers)
            writer.registerSolution(PITCH, powers)
            writer.add(PITCH, 0.0, 3.0)
            writer.midi.write(TEMP_MIDI)
            playMidiOnDisklavier(TEMP_MIDI, verbose=False)
            sleep(1)
