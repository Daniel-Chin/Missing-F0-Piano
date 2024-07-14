import typing as tp

import numpy as np
import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from music import *

PITCHES = [40, 48, 60]

WINDOW_WIDTH = 1000
SR = 22050
NYQUIST = SR / 2
PAGE_LEN = 1024
N_FREQ_BINS = PAGE_LEN // 2 + 1
FREQ_BINS = np.linspace(0, NYQUIST, N_FREQ_BINS)

t_GetValue = tp.Callable[[], float]

def initFig():
    fig, axes = plt.subplots(nrows = len(PITCHES), sharex=True)
    lines: tp.List[Line2D] = []
    for ax, pitch in zip(axes, PITCHES):
        ax: Axes
        line, = ax.plot(FREQ_BINS, np.zeros(N_FREQ_BINS))
        lines.append(line)
        f0 = pitch2freq(pitch)
        partials = np.arange(f0, NYQUIST, f0)
        ax.plot(partials, np.ones_like(partials), 'o', c='k', markersize=1)
        ax.set_ylim(bottom=0.0)
        ax.set_xlim(left=0.0, right=NYQUIST)
        ax.set_ylabel('Amplitude')
        ax.set_title(str(pitch))
    ax.set_xlabel('Frequency (Hz)')
    fig.tight_layout(rect=(-.04, 0, 1.02, 1))
    return fig, lines

def Root():
    root = tk.Tk()
    root.title('Solver Tuner')
    fig, lines = initFig()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(
        fill=tk.BOTH, 
        # expand=True, 
    )
    getPercTole = HParamControl(root, 'Perception Tolerance', -5, np.log(0.5))
    getPenaStra = HParamControl(root, 'Penalize Stranger', -3, 2)
    def animate_(_):
        animate(
            getPercTole, 
            getPenaStra, 
            lines, 
        )
        return lines
    anim = FuncAnimation(
        fig, animate_, 
        interval=33, blit=False, cache_frame_data=False, 
    )
    return root, anim

def HParamControl(
    parent: tk.Tk, text: str, 
    log_from: float, log_to: float, 
):
    default = (log_from + log_to) / 2
    var = tk.DoubleVar(value=default)
    def getValue():
        return np.exp(var.get())
    
    subroot = tk.Frame(parent)
    subroot.pack()
    upper = tk.Frame(subroot)
    upper.pack()
    tk.Label(upper, text=text + ': ').pack(side='left')
    display = tk.Label(upper)
    display.pack(side='right')
    def updateDisplay(_ = None):
        display.config(text=f'{getValue():.2e}')
    updateDisplay()
    
    scale = tk.Scale(
        subroot, variable=var, 
        from_=log_from, to=log_to, 
        resolution = (log_to - log_from) / 100, 
        showvalue=False, orient='horizontal', 
        length=WINDOW_WIDTH, 
        command=updateDisplay, 
    )
    scale.pack()
    return getValue

def animate(
    getPercTole: t_GetValue, 
    getPenaStra: t_GetValue, 
    lines: tp.List[Line2D], 
):
    ...

def main():
    root, anim = Root()
    root.mainloop()
    assert anim is not None # keep alive

if __name__ == '__main__':
    main()
