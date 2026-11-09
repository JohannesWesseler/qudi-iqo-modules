# -*- coding: utf-8 -*-
"""
Least-squares fit of a sine of known frequency, and harmonic distortion.

    fit_sine(x, frequency, t=...)            x(t) = A sin(2 pi f t + phi) + c (+ s (t - mean t))
    fit_sine(x, frequency, sample_rate=...)  the same for uniformly sampled x, t = t0 + n / fs
    harmonic_distortion(x, frequency, sample_rate)

The fit is linear in a sin(wt) + b cos(wt) + c, so it has no start values and no iterations.
The phasor of the result is a + i b = A exp(i phi). For a sine whose frequency is known exactly
(a generator output, a response to it) this is the optimal estimator of amplitude and phase in
white noise; it is what a lock-in amplifier approximates with a low-pass filter.

The time origin sets the phase: fit two signals on the same time axis to compare their phases.

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ['SineFit', 'fit_sine', 'harmonic_distortion']


@dataclass(frozen=True)
class SineFit:
    amplitude: float      # A (peak)
    phase: float          # rad, x = A sin(w t + phase) + ...
    offset: float         # c
    slope: float          # s (per unit of t), 0 without trend
    resid_rms: float      # rms of x minus the fit

    @property
    def phasor(self) -> complex:
        """A exp(i phase)."""
        return complex(self.amplitude * np.exp(1j * self.phase))

    @property
    def rel_resid(self) -> float:
        """resid_rms / rms of the sine (~ THD + N for a clean signal)."""
        return self.resid_rms / (self.amplitude / np.sqrt(2)) if self.amplitude > 0 else np.inf


def fit_sine(x, frequency: float, t=None, sample_rate: Optional[float] = None, t0: float = 0.0,
             trend: bool = False) -> SineFit:
    """Fit x = A sin(2 pi f t + phase) + c, optionally + s (t - mean t).

    Give either the sample times `t` (any spacing) or `sample_rate` (then t = t0 + n / fs).
    trend: also fit a linear drift. Without it a drift s biases the sine by about s / (pi f)
    when the record holds only a few periods."""
    x = np.asarray(x, float)
    if t is None:
        if sample_rate is None:
            raise ValueError('give t or sample_rate')
        t = t0 + np.arange(x.size) / float(sample_rate)
    else:
        t = np.asarray(t, float)
    w = 2 * np.pi * float(frequency)
    cols = [np.sin(w * t), np.cos(w * t), np.ones_like(t)]
    if trend:
        cols.append(t - t.mean())
    m = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(m, x, rcond=None)
    r = x - m @ coef
    return SineFit(amplitude=float(np.hypot(coef[0], coef[1])),
                   phase=float(np.arctan2(coef[1], coef[0])), offset=float(coef[2]),
                   slope=float(coef[3]) if trend else 0.0,
                   resid_rms=float(np.sqrt(np.mean(r ** 2))))


def harmonic_distortion(x, frequency: float, sample_rate: float, n_harmonics: int = 5) -> float:
    """THD as an amplitude ratio: rms sum of harmonics 2..n_harmonics over the fundamental,
    all fitted at their known frequencies."""
    x = np.asarray(x, float)
    t = np.arange(x.size) / float(sample_rate)
    cols = [np.ones_like(t)]
    for k in range(1, n_harmonics + 1):
        cols += [np.sin(2 * np.pi * k * frequency * t), np.cos(2 * np.pi * k * frequency * t)]
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), x, rcond=None)
    amps = [np.hypot(coef[2 * k - 1], coef[2 * k]) for k in range(1, n_harmonics + 1)]
    return float(np.sqrt(np.sum(np.square(amps[1:]))) / amps[0]) if amps[0] > 0 else np.inf
