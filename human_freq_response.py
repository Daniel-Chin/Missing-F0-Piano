'''
Reference:
https://www.compadre.org/nexusph/course/Frequency_response_of_the_human_ear
blog by Bill Dorland.
'''

from math import log

from matplotlib import pyplot as plt
import torch

from piecewise_linear import PiecewiseLinear

MILESTONES = torch.tensor([
    (log(   19), -40), 
    (log(   20), -10), 
    (log(  100), 0), 
    (log( 1000), 0), 
    (log( 3000), 7.4), 
    (log( 4000), 2.5), 
    (log( 7400), 4), 
    (log( 9000), 2), 
    (log(10000), -8), 
    (log(14000), -8), 
    (log(20000), -20), 
    (log(20001), -40), 
])

freqResponseDb = PiecewiseLinear(
    MILESTONES[:, 0], MILESTONES[:, 1], 
)

def visualize():
    x = torch.linspace(log(10), log(21000), 1000).exp()
    y = freqResponseDb.forward(x.log())
    plt.plot(x, y)
    plt.xscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Response (dB)')
    plt.show()

if __name__ == '__main__':
    visualize()
