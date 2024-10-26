# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:14:27 2024

@author: Carlo
"""
# %% imports function definitions

import plotly.express as px
import numpy as np
import scipy
from scipy import fft

import plotly.io as pio
pio.renderers.default = 'svg'
# pio.renderers.default='browser'


def plot(*args, **kwargs):
    plot_title = kwargs.pop("plot_title", "")
    if len(args) == 1:
        y = args[0]
        x = range(len(y))
    else:
        x = args[0]
        y = args[1]
    fig = px.line(x=x, y=y, markers=True, **kwargs)
    fig.update_layout(
        template='plotly_white',
        title=plot_title, title_x=0.5
    )
    fig.show()

# return the amplitude of the interferogram for a single Lorentzian peak of the given shift and width


def signal_single_peak(pos_mm, shift_GHz, width_GHz):
    # time delay between the two arms of the Michelson (in ns)
    tao = 2e6*pos_mm/scipy.constants.c
    return np.exp(-np.pi*np.abs(tao)*width_GHz)*np.cos(2*np.pi*shift_GHz*tao)


def spectrum_fft(large_step_um, A):
    """
    Calculate the spctrum from the interferogram A sampled with 2*large_step_um steps
    """
    N = 1024
    # spectral axis in GHz
    freq = 1e-3*scipy.constants.c*fft.fftfreq(N, 2*large_step_um)[0:N//2]
    spectrum = fft.hfft(A-np.mean(A), n=N)[0:N//2]

    return freq, spectrum


def flip_sign(A, i):
    """
    flips the sign of th i-th element in the array A
    """
    Af = np.copy(A)
    Af[i] *= -1
    return Af


# %% simulation
if __name__ == "__main__":
    # number of large steps
    N = 20
    # large step
    S = 5e3  # um

    pos_mm = S*1e-3*np.arange(N)
    A = signal_single_peak(pos_mm, 3.49, 1.6)

    freq, spectrum = spectrum_fft(S, A)

    i = 5
    Ai = flip_sign(A, i)
    freq, spectrum_i = spectrum_fft(S, Ai)

    difference = spectrum_i-spectrum
    plot(difference,
         plot_title=f"Difference between original spectrum and spectrum with sign flip at {i}")
