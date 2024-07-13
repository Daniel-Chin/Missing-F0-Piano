import typing as tp

DISKLAVIER = 'Disklavier'

def findDevice(devices: tp.List[str]):
    matched = [x for x in devices if DISKLAVIER.lower() in x.lower()]
    assert len(matched) == 1, devices
    return matched[0]
