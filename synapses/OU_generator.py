# -*- coding: utf-8 -*-
"""
Adds Ornstein-Uhlenbeck (OU) noise to current/voltage traces.
For BBP references see Barros-Zulaica et al. 2019 and Ecker et al. 2020
author: András Ecker, last update: 03.2021
"""

import numpy as np
from numba import jit

TAU = 28.2  # 28.2+/-3.5 ms from Barros-Zulaica et al. 2019 (mean of 33 in vitro L5_TTPC voltage traces)
SIGMA = 0.22  # 0.22+/-0.1 ms from Barros-Zulaica et al. 2019 (mean of 33 in vitro L5_TTPC voltage traces)


@jit(nopython=True)
def ou_generator(t, tau, sigma, rngs, y0):
    """
    Generates OU noise using forward Euler
    :param t: numpy array - representing time
    :param tau: float - tau parameter of OU noise (extracted from in vitro traces)
    :param sigma: float - sigma parameter of OU noise (extracted from in vitro traces)
    :param rngs: numpy array (same size as `t`) - with random draws from N(0, 1)
                 has to be a parameter of the function since numba doesn't support `np.random.normal()`...
    :param y0: float - mean/initial value of the noise
    :return: y: numpy array (same size as `t` and `rngs`) - generated (OU) noise
    """

    dt = t[1] - t[0]
    tmp = sigma * np.sqrt(2*dt/tau)
    y = np.zeros_like(t, dtype=np.float32)
    y[0] = y0
    for i in range(1, int(t.shape[0])):
        y[i] = y[i-1] + (dt/tau)*(y0-y[i-1]) + tmp*rngs[i-1]
    return y


def add_ou_noise(t, traces, tau=TAU, sigma=SIGMA):
    """Adds noise to current/voltage traces (see also `ou_generator()`)"""
    for i in range(traces.shape[0]):
        np.random.seed(12345+i)
        rngs = np.random.normal(size=t.shape[0])
        noise = ou_generator(t-np.min(t), tau, sigma, rngs, 0.)
        traces[i, :] += noise
    return traces
