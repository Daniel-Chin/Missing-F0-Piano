from enum import Enum

class PedalEvent(Enum): 
    DOWN = 'DOWN'
    UP = 'UP'

BREATHE = [PedalEvent.UP, PedalEvent.DOWN]

RAINBOW = [
    PedalEvent.DOWN, [[0+1j, 12], [[11, [7, 9]], [11, 12]]], 
    [BREATHE, [0+1j, 9], 7], 
    [BREATHE, [-3+1j, 5], [[4, [0, 2]], [4, 5]]], 
    [BREATHE, [[2, [-1, 0]], [2, 4]], 0], 
]
