# -*- coding: utf-8 -*-
"""
Simulated current source (CurrentSourceInterface): a voltage amplifier (gain 16) into an R-L
load (default 3.43 Ohm, 20.5 uH, the Helmholtz load of the real source), current and output
voltage measured with noise. Same limits as the real source (current limit, OUT1 voltage limit,
burst/offset rule), no supply. For developing logic, GUIs and scripts without hardware.

Example config:

    current_source_dummy:
        module.Class: 'dummy.current_source_dummy.CurrentSourceDummy'
        options:
            noise: 2e-4          # A rms of the current measurement

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

import copy
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

from qudi.core.configoption import ConfigOption
from qudi.util.mutex import RecursiveMutex
from qudi.util.sine_fit import fit_sine
from qudi.interface.current_source_interface import (CurrentSourceInterface, CurrentSourceConstraints,
                                                     CommandRefused, WAVEFORMS)

FS = 125e6
N = 16384
ARB = 16384


class CurrentSourceDummy(CurrentSourceInterface):

    _noise = ConfigOption('noise', default=2e-4, missing='nothing')
    _r = ConfigOption('resistance', default=3.43, missing='nothing')
    _l = ConfigOption('inductance', default=20.5e-6, missing='nothing')
    _gain = ConfigOption('gain', default=16.0, missing='nothing')
    _limit0 = ConfigOption('current_limit', default=1.66, missing='nothing')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = RecursiveMutex()
        self._limit = 1.66
        self._powered = False
        self._out = dict(on=False, waveform='sine', frequency=1000.0, amplitude=0.0, offset=0.0,
                         v_amplitude=0.0, v_offset=0.0, predicted_peak=0.0)
        self._burst = None
        self._sync = False
        self._arb = np.sin(2 * np.pi * np.arange(ARB) / ARB)
        self._t_start = time.time()
        self._armed = None

    def on_activate(self):
        self._limit = float(self._limit0)

    def on_deactivate(self):
        self._powered = False

    # --- description ----------------------------------------------------------------------
    @property
    def constraints(self):
        return CurrentSourceConstraints(WAVEFORMS, (0.0, 1e6), self._limit, 2.0, 0.8, 60.0, 60.0,
                                    1.5, 600.0, 300.0, N, FS, ARB, 50000, 50000)

    def _gm(self, f):
        z = self._r + 2j * np.pi * f * self._l
        return self._gain / z

    def expected_current(self, frequency, amplitude):
        gm = self._gm(frequency)
        return float(amplitude) * gm / abs(gm)

    def max_current(self, waveform, frequency):
        gm = abs(self._gm(frequency)) if waveform == 'sine' else self._gain / self._r
        return float(min(self._limit * 0.98, 0.76 * gm))

    @property
    def current_limit(self):
        return self._limit

    @current_limit.setter
    def current_limit(self, value):
        if not 0 < value <= 2.0:
            raise CommandRefused('current limit must be in (0, 2] A')
        self._limit = float(value)

    # --- power ----------------------------------------------------------------------------
    def power_on(self):
        self._powered = True
        self.sigStateChanged.emit(self.output_state)
        return self.read_supply()

    def power_off(self):
        self.output_off()
        self._powered = False
        self.sigStateChanged.emit(self.output_state)

    @property
    def powered(self):
        return self._powered

    # --- output ---------------------------------------------------------------------------
    def set_output(self, waveform, frequency=0.0, amplitude=0.0, offset=0.0, ramp=True):
        waveform = waveform.lower()
        if waveform not in WAVEFORMS:
            raise CommandRefused(f'unknown waveform {waveform}')
        if not self._powered:
            raise CommandRefused('the amplifier supply is off: power_on() first')
        if waveform == 'dc':
            amplitude, frequency = 0.0, self._out['frequency']
        if self._burst is not None and offset:
            raise CommandRefused('bursts only with offset 0')
        gm = abs(self._gm(frequency)) if waveform == 'sine' else self._gain / self._r
        gm_dc = self._gain / self._r
        v_amp, v_off = abs(amplitude) / gm, offset / gm_dc
        peak = (abs(amplitude) + abs(offset)) / 0.98
        if abs(v_amp) + abs(v_off) > 0.76 + 1e-9:
            raise CommandRefused(f'OUT1 {abs(v_amp) + abs(v_off):.3f} V > 0.76 V clean limit')
        if peak > self._limit + 1e-9:
            raise CommandRefused(f'predicted peak {peak:.3f} A > current limit {self._limit} A')
        with self._lock:
            if waveform != self._out['waveform'] or frequency != self._out['frequency']:
                self._t_start = time.time()
            self._out.update(on=True, waveform=waveform, frequency=float(frequency),
                             amplitude=abs(amplitude), offset=offset, v_amplitude=v_amp,
                             v_offset=v_off, predicted_peak=peak)
        self.sigStateChanged.emit(self.output_state)
        return self.output_state

    def output_off(self):
        with self._lock:
            self._out.update(on=False, amplitude=0.0, offset=0.0, v_amplitude=0.0, v_offset=0.0,
                             predicted_peak=0.0)
        self.sigStateChanged.emit(self.output_state)

    @property
    def output_state(self):
        st = copy.deepcopy(self._out)
        st.update(powered=self._powered, burst=self._burst, sync_output=self._sync,
                  current_limit=self._limit, high_time={'dc': 0.0, 'ac': 0.0})
        return st

    def set_arbitrary_waveform(self, samples: Sequence[float]):
        x = np.asarray(samples, float)
        t = np.interp(np.arange(ARB) * len(x) / ARB, np.arange(len(x) + 1), np.append(x, x[0]))
        self._arb = t / np.abs(t).max()

    def configure_burst(self, n_cycles, n_repetitions=1, period=None):
        self._burst = None if n_cycles is None else (int(n_cycles), int(n_repetitions), period)
        self.sigStateChanged.emit(self.output_state)

    def set_sync_output(self, enabled):
        self._sync = bool(enabled)

    def trigger(self):
        self._t_start = time.time()
        if self._armed is not None:
            self._armed['triggered'] = True

    # --- simulation -----------------------------------------------------------------------
    def _shape(self, phase):
        w = self._out['waveform']
        if w == 'sine':
            return np.sin(2 * np.pi * phase)
        if w == 'square':
            return np.where((phase % 1) < 0.5, 1.0, -1.0)
        if w == 'triangle':
            p = phase % 1
            return np.where(p < 0.25, 4 * p, np.where(p < 0.75, 2 - 4 * p, 4 * p - 4))
        if w == 'arbitrary':
            return self._arb[((phase % 1) * ARB).astype(int) % ARB]
        return np.zeros_like(phase)

    def _simulate(self, dec, t0, gate=None):
        fs = FS / dec
        t = t0 + np.arange(N) / fs
        o = self._out
        if not o['on'] or not self._powered:
            i = np.zeros(N)
        else:
            f = o['frequency']
            # generator voltage -> amplifier (gain) -> R-L load: first-order current response
            u = o['v_amplitude'] * self._shape(f * t) if o['waveform'] != 'dc' else 0 * t
            if gate is not None:
                u = u * gate(t)
            from scipy.signal import lfilter
            a = np.exp(-1 / (fs * self._l / self._r))
            i = lfilter([1 - a], [1, -a], self._gain * u / self._r) + o['offset']
        i = i + np.random.normal(0, self._noise, N)
        di = np.gradient(i, 1 / fs)
        v = self._r * i + self._l * di
        return t - t0, i, v, fs

    def measure(self, n_average=1):
        o = self._out
        dec = 64 if o['waveform'] != 'sine' else max(1, int(np.ceil(20 * FS / (N * max(o['frequency'], 1)))))
        t, i, v, fs = self._simulate(dec, time.time() - self._t_start)
        res = dict(t=time.time(), waveform=o['waveform'], frequency=o['frequency'],
                   amplitude_set=o['amplitude'], offset_set=o['offset'], on=o['on'],
                   burst=self._burst is not None, decimation=dec, sample_rate=fs,
                   i_mean=float(i.mean()), i_rms=float(np.sqrt(np.mean(i ** 2))),
                   i_peak=float(np.abs(i).max()), v_mean=float(v.mean()),
                   v_rms=float(np.sqrt(np.mean(v ** 2))), v_peak=float(np.abs(v).max()),
                   i_mean_output=float(v.mean()) / self._r,
                   i_mean_command=float(o['offset']) if o['on'] else 0.0,
                   supply=self.read_supply(), supply_power=0.0)
        if o['waveform'] == 'sine' and o['on']:
            fi = fit_sine(i, o['frequency'], sample_rate=fs)
            fv = fit_sine(v, o['frequency'], sample_rate=fs)
            z_load = abs(self._r + 2j * np.pi * o['frequency'] * self._l)
            z = fv.phasor / fi.phasor if fi.amplitude > 0 else np.nan
            res.update(i_amp=fi.amplitude, v_amp=fv.amplitude, i_amp_std=0.0,
                       i_amp_output=fv.amplitude / z_load, i_amp_command=float(o['amplitude']),
                       phase_i_minus_v=float(np.angle(fi.phasor / fv.phasor)) if fv.amplitude > 0 else 0.0,
                       i_offset=fi.offset, z_re=float(np.real(z)), z_im=float(np.imag(z)),
                       i_thd=0.0, v_thd=0.0, i_rel_resid=fi.rel_resid)
        return res

    def capture(self, decimation=64):
        t, i, v, fs = self._simulate(int(decimation), time.time() - self._t_start)
        return dict(t=t, current=i, output=v, sample_rate=fs, decimation=int(decimation),
                    trigger_index=0, time=time.time())

    def arm_capture(self, decimation=64, trigger='generator', pre_trigger=0):
        self._armed = dict(dec=int(decimation), pre=int(pre_trigger), triggered=False)

    def read_capture(self, timeout=5.0):
        if self._armed is None:
            raise RuntimeError('no capture armed')
        a, self._armed = self._armed, None
        if not a['triggered']:
            time.sleep(min(timeout, 0.05))
            raise TimeoutError('dummy: no trigger')
        fs = FS / a['dec']
        gate = None
        if self._burst is not None:
            n_cyc, n_rep, per = self._burst
            dur = n_cyc / max(self._out['frequency'], 1e-9)
            per = per or dur
            nrep = n_rep if n_rep > 0 else 10 ** 9
            gate = lambda t: ((t >= 0) & ((t % per) < dur) & (t < nrep * per)).astype(float)
        t0 = -a['pre'] / fs
        t, i, v, fs = self._simulate(a['dec'], t0, gate)
        return dict(t=t + t0, current=i, output=v, sample_rate=fs, decimation=a['dec'],
                    trigger_index=a['pre'], time=time.time())

    def read_supply(self):
        on = int(self._powered)
        i = 0.03 + 0.5 * abs(self._out['amplitude']) * self._out['on']
        return dict(v1=16.0 * on, i1=i * on, trip1=0, on1=on, v4=16.0 * on, i4=i * on, trip4=0, on4=on)
