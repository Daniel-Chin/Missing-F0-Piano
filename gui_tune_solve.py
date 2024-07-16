'''
Does not consider MIN_POWER.  
'''

from __future__ import annotations

import typing as tp
from os import path
from contextlib import contextmanager
from time import time

import numpy as np
import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
from torch import Tensor
import fluidsynth

from my import profileFrequency

from shared import *
from music import *
from solve import Trainee
from power_to_velocity import power2velocity

N_EPOCHS_PER_FRAME = 16

WINDOW_WIDTH = 1000
SR = 22050
NYQUIST = SR / 2
PAGE_LEN = 2048
N_FREQ_BINS = PAGE_LEN // 2 + 1
FREQ_BINS = np.linspace(0, NYQUIST, N_FREQ_BINS)

TWO_PI_OVER_SR = 2 * torch.pi / SR
TIME_LADDER = TWO_PI_OVER_SR * torch.arange(
    PAGE_LEN, 
).unsqueeze(0).unsqueeze(1).contiguous().to(device())

def initFig():
    fig, axes = plt.subplots(nrows = len(GUI_PITCHES), sharex=True)
    lines: tp.List[Line2D] = []
    for ax, pitch in zip(axes, GUI_PITCHES):
        ax: Axes
        line, = ax.plot(
            FREQ_BINS, np.zeros(N_FREQ_BINS), 
            'o', markersize=0.5, 
        )
        lines.append(line)
        f0 = pitch2freq(pitch)
        partials = np.arange(f0, NYQUIST, f0)
        ax.plot(partials, np.ones_like(partials), 'o', c='k', markersize=1)
        ax.set_ylim(bottom=0.0, top=1.2)
        ax.set_xlim(left=0.0, right=NYQUIST)
        ax.set_ylabel('Amplitude')
        ax.set_title(str(pitch))
    ax.set_xlabel('Frequency (Hz)')
    fig.tight_layout(rect=(-.04, 0, 1.02, 1))
    return fig, lines

def Root(audioStateMachine: AudioStateMachine):
    root = tk.Tk()
    root.title('Solver Tuner')
    fig, lines = initFig()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(
        fill=tk.BOTH, 
        # expand=True, 
    )
    trainees_persistent = [
        Trainee(pitch, 0.1, 0.0, 0.0)
        for pitch in GUI_PITCHES
    ]
    def onSlide(_):
        for trainee in trainees_persistent:
            trainee.breakFree()
    getPercTole = HParamControl(
        root, onSlide, 'Perception Tolerance', -5, np.log(0.5), 
        log_default=-2.8,
    )
    getPenaStra = HParamControl(
        root, onSlide, 'Penalize Stranger', -1, 5, 
        log_default=2.0,
    )
    getLearRate = HParamControl(
        root, onSlide, 'Learning Rate', -9, -3, 
        log_default=-3.6,
    )
    getRespResp = HParamControl(
        root, onSlide, 'Respect Response', -5, 0, 
        log_default=0.0,
    )
    def animate_(_):
        animate(
            getPercTole(), 
            getPenaStra(), 
            getLearRate(), 
            getRespResp(), 
            lines, 
            trainees_persistent, 
        )
        audioStateMachine.loop([
            x.activations for x in trainees_persistent
        ])
        return lines
    anim = FuncAnimation(
        fig, animate_, 
        interval=33, blit=False, cache_frame_data=False, 
    )
    return root, anim

def HParamControl(
    parent: tk.Tk, slideCallback: tp.Callable[[float], None], 
    text: str, 
    log_from: float, log_to: float, 
    allow_zero: bool = True, 
    log_default: float | None = None, 
):
    if log_default is None:
        log_default = (log_from + log_to) / 2
    var = tk.DoubleVar(value=log_default)
    def getValue() -> float:
        ex = var.get()
        if allow_zero and abs(ex - log_from) < 1e-6:
            return 0.0
        return np.exp(ex)
    
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

    def onSlide(x: str, /):
        updateDisplay()
        slideCallback(float(x))
    
    scale = tk.Scale(
        subroot, variable=var, 
        from_=log_from, to=log_to, 
        resolution = (log_to - log_from) / 100, 
        showvalue=False, orient='horizontal', 
        length=WINDOW_WIDTH, 
        command=onSlide, 
    )
    scale.pack()
    return getValue

@profileFrequency(format_str='.0f')
def animate(
    perc_tole: float, 
    pena_stra: float, 
    lr: float,
    respect_response: float, 
    lines: tp.List[Line2D], 
    trainees_persistent: tp.List[Trainee], 
):
    for line, trainee_persistent, pitch in zip(
        lines, trainees_persistent, GUI_PITCHES, 
    ):
        trainee = Trainee(
            pitch, perc_tole, pena_stra, lr, 
            respect_response, 
            trainee_persistent.activations.detach().clone(), 
        )
        for epoch in range(N_EPOCHS_PER_FRAME):
            loss, _ = trainee.forward()
            trainee.oneEpochGivenForward(loss)
        activations = trainee.activations.detach()
        trainee_persistent.activations = activations
        f0s = pitch2freq_batch(torch.arange(
            pitch + 1, PIANO_RANGE.stop, 
            device=device(), 
        ))
        max_n_partials = int(NYQUIST / f0s[0].item())
        fns = f0s.unsqueeze(1) * (torch.arange(
            1, max_n_partials + 1, 
            device=device(), 
        ).unsqueeze(0))
        ans = activations.unsqueeze(1).repeat((
            1, max_n_partials, 
        ))
        ans[fns > NYQUIST] = 0.0
        waves = ((
            fns.unsqueeze(2) * TIME_LADDER
        ).sin() * ans.unsqueeze(2))
        mixdown = waves.sum(dim=1).sum(dim=0)
        spectrum: Tensor = torch.fft.rfft(
            mixdown, norm='forward', 
        )
        data = spectrum.abs().cpu().numpy()
        line.set_ydata(data)

def main():
    with fluidsynthContext() as fs:
        audioStateMachine = AudioStateMachine(fs)
        root, anim = Root(audioStateMachine)
        root.mainloop()
    assert anim is not None # keep alive

@contextmanager
def fluidsynthContext():
    fs = fluidsynth.Synth()
    fs.start()
    sfid = fs.sfload(path.expanduser(
        '~/roaming_linux_daniel/soundfonts/Yamaha-Grand-Lite-v2.0.sf2',
    ))
    fs.program_select(chan=0, sfid=sfid, bank=0, preset=0)
    try:
        yield fs
    finally:
        fs.delete()

class AudioStateMachine:
    def __init__(self, fs: fluidsynth.Synth):
        self.fs = fs

        self.last_play_time = 0
        self.down_keys: tp.List[int] = []
    
    def play(self, pitch: int, velocity: int):
        self.down_keys.append(pitch)
        self.fs.noteon(0, pitch, velocity)
    
    def panic(self):
        for pitch in self.down_keys:
            self.fs.noteoff(0, pitch)
        self.down_keys.clear()

    def loop(self, activationses: tp.List[Tensor]):
        now = int(time())
        if now != self.last_play_time:
            self.last_play_time = now
            self.panic()
            phase = (now % (len(GUI_PITCHES) + 1))
            try:
                pitch = GUI_PITCHES[phase]
            except IndexError:
                self.play(max(GUI_PITCHES) + 12, 127)
                return
            velocities = power2velocity(
                activationses[phase].square(), 
            )
            for p, v in zip(range(
                pitch + 1, PIANO_RANGE.stop,
            ), velocities, 
            ):
                v_: int = v.item()  # type: ignore
                self.play(p, v_)

if __name__ == '__main__':
    main()
