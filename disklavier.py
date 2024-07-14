import typing as tp

import mido

from playMidi import main as playMidi

DISKLAVIER = 'Disklavier'

def findDevice(devices: tp.List[str]):
    matched = [x for x in devices if DISKLAVIER.lower() in x.lower()]
    assert len(matched) == 1, devices
    return matched[0]

def findDeviceOut():
    outs = mido.get_output_names()  # type: ignore
    return findDevice(outs)

def Disklavier():
    return mido.open_output(findDeviceOut()) # type: ignore

def playMidiOnDisklavier(
    filename: str | None = None, verbose: bool = True, 
):
    return playMidi(
        filename, findDeviceOut(), verbose=verbose, 
    )
