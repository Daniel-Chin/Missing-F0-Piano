import typing as tp

import mido

DISKLAVIER = 'Disklavier'

def findDevice(devices: tp.List[str]):
    matched = [x for x in devices if DISKLAVIER.lower() in x.lower()]
    assert len(matched) == 1, devices
    return matched[0]

def Disklavier():
    outs = mido.get_output_names()  # type: ignore
    device = findDevice(outs)
    return mido.open_output(device) # type: ignore
