# -*- coding: utf-8 -*-
"""
Waveform current source built from a Red Pitaya (stock OS, SCPI) and an LM1875 power amplifier,
for loads of a few ohms such as test-field coils: DC to ~80 kHz, up to 2 A peak, continuous or
in triggered bursts with a sync marker, with live measurement of current and output voltage.

The amplifier is a voltage amplifier (gain 16). This module sets a current by dividing it by
the transconductance I/OUT1(f) measured for the connected load, so the current is only as good
as that measurement: every load needs its own load file (calibration/load_*.json). The current
is read back on a shunt and the output voltage on a divider (MON). At small currents (mA) these
readings are less accurate than the setting; README.md, "Accuracy".

Everything that protects the hardware is enforced in this module, below any logic, GUI or
script: output-voltage and current limits, ramping, heating limits of the load, plausibility
checks of every acquisition, supply checks, and the power on/off order (README.md, "Limits").

Threading: all access to the Red Pitaya runs under one lock, so logic threads and the monitor
timer (main thread) can call the module at the same time. Long operations (a capture waiting
for its trigger, the first trigger of a burst setting) hold the lock; the monitor skips its
tick meanwhile.

Full documentation (circuit, calibration, accuracy, known problems): README.md in this folder.

Example config:

    current_source:
        module.Class: 'current_source.redpitaya_lm1875_source.RedPitayaLM1875CurrentSource'
        connect:
            power_supply: ngp824
        options:
            host: '10.203.129.30'
            load: 'load_helmholtz_r17_5mm.json'   # in calibration/, or an absolute path
            # optional:
            # amplifier_calibration: 'amplifier.json'
            # current_limit: 1.66         # A peak; default: rated_current of the load
            # shunt_input: 1              # physical Red Pitaya input of the shunt
            # monitor_input: 2            # and of MON
            # rail_channels: [1, 4]       # supply channels for +16 V and -16 V
            # monitor_interval: 1.0       # s, 0 = off
            # stop_server_on_deactivate: True

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
import json
import os
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np
from PySide2 import QtCore

from qudi.core.configoption import ConfigOption
from qudi.core.connector import Connector
from qudi.util.mutex import RecursiveMutex
from qudi.util.sine_fit import fit_sine, harmonic_distortion
from qudi.interface.current_source_interface import (CurrentSourceInterface, CurrentSourceConstraints,
                                                     CurrentSourceAbort, CommandRefused, WAVEFORMS)
from .redpitaya_scpi import RedPitayaScpi, choose_decimation, FS_ADC, N_BUF, ARB_LENGTH
from .load_model import LoadModel

CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration')


def calibration_path(name):
    """A file name in calibration/ next to this module, or an absolute path."""
    return name if os.path.isabs(name) else os.path.join(CALIBRATION_DIR, name)


def load_amplifier_calibration(path):
    """Read amplifier.json: offset_null (V), adc_offsets [IN1, IN2] (V), out1_clean,
    out1_max (V)."""
    with open(path) as fh:
        d = json.load(fh)
    missing = [k for k in ('offset_null', 'adc_offsets', 'out1_clean', 'out1_max') if k not in d]
    if missing or len(d['adc_offsets']) != 2:
        raise ValueError(f'{path}: missing or malformed {missing or "adc_offsets"}')
    if not 0 < d['out1_clean'] <= d['out1_max'] < 1.0:
        raise ValueError(f'{path}: need 0 < out1_clean <= out1_max < 1 V')
    return d


class RedPitayaLM1875CurrentSource(CurrentSourceInterface):
    """Red Pitaya + LM1875 current source with the safety limits found in its characterization."""

    # --- the circuit as built (README.md, "Circuit"); a change here is a change of the board --
    SHUNT_OHM = 0.5              # R11, Kelvin-connected
    MON_RATIO = 16.0             # output / MON = (R12 + R13) / R13 = 16 k / 1 k
    CURRENT_LIMIT_ABSOLUTE = 2.0  # A: the shunt reading reaches the +-1 V input range at 2 A
    FREQUENCY_MAX = 1e6          # Hz, generator setting accepted
    # supply: two channels in series for +-16 V
    RAIL_VOLTAGE = 16.0
    RAIL_CURRENT_LIMIT = 2.1     # A; the NGP800 fuse trips at the current limit
    FUSE_DELAY = 0.05            # s
    FUSE_DELAY_INIT = 0.1        # s after switch-on (470 uF per rail charge)
    RAIL_MIN_VOLTAGE = 15.5
    # protection thresholds
    SHUNT_ABORT_V = 0.9          # shunt input (1.8 A)
    MON_ABORT_V = 0.85           # MON input (13.6 V output)
    CLIP_V = 0.98                # Red Pitaya input near full scale
    RAIL_CURRENT_ABORT = 2.0     # A per rail
    IDLE_CURRENT_ABORT = 0.150   # A per rail at zero output (idle 41/38 mA; more = oscillation)
    POWER_CHECK_TIME = 3.0       # s of idle check after switch-on
    GAIN_MARGIN = 0.98           # predicted current = requested / margin (large-signal gain +0.5 %)
    RAMP_STEP = 1.25             # max. ratio between ramp steps

    _supply = Connector(name='power_supply', interface='RohdeSchwarzNGP800')

    # installation
    _host = ConfigOption('host', missing='error')
    _port = ConfigOption('port', default=5000, missing='nothing')
    _shunt_input = ConfigOption('shunt_input', default=1, checker=lambda x: x in (1, 2))
    _monitor_input = ConfigOption('monitor_input', default=2, checker=lambda x: x in (1, 2))
    _rail_channels = ConfigOption('rail_channels', default=[1, 4], missing='nothing',
                                  checker=lambda x: len(x) == 2)
    _stop_server = ConfigOption('stop_server_on_deactivate', default=True, missing='nothing')
    # calibration of this unit and the connected load (files, see calibration_path)
    _amp_file = ConfigOption('amplifier_calibration', default='amplifier.json', missing='nothing')
    _load_file = ConfigOption('load', missing='error')
    # operation
    _current_limit_option = ConfigOption('current_limit', default=None, missing='nothing')
    _monitor_interval = ConfigOption('monitor_interval', default=1.0, missing='nothing')

    _sigStartMonitor = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = RecursiveMutex()
        self._rp = None
        self._load = None
        self._amp = {}
        self._heat = {}
        self._current_limit = 1.0
        self.allow_near_rail = False
        self._out = self._zero_state()
        self._burst = None                     # (n_cycles, n_repetitions, period) or None
        self._sync_out = False
        self._arb_table = np.zeros(ARB_LENGTH)
        self._arb_loaded = False
        self._gen_dirty = False          # settings written that apply only at the next trigger
        self._powered = False
        self._capture_armed = False
        self._armed_since = 0.0
        self._capture_pre = 0
        self._high = {'dc': [None, -1e9], 'ac': [None, -1e9]}   # [since, last end]
        self._last_measure = {}
        self._last_checked = None              # raw traces of the last check (abort diagnosis)
        self._timer = None

    # =========================================================================================
    # qudi
    # =========================================================================================
    def on_activate(self):
        if self._shunt_input == self._monitor_input:
            raise ValueError('shunt_input and monitor_input must be different Red Pitaya inputs')
        self._amp = load_amplifier_calibration(calibration_path(self._amp_file))
        self._offset_null = float(self._amp['offset_null'])
        self._adc_offset_shunt = float(self._amp['adc_offsets'][self._shunt_input - 1])
        self._adc_offset_mon = float(self._amp['adc_offsets'][self._monitor_input - 1])
        self._load = LoadModel.from_file(calibration_path(self._load_file))
        info = self._load.info
        self._heat = {k: float(v) for k, v in info['heating'].items() if k != 'note'}
        limit = self._current_limit_option
        limit = float(info['rated_current'] if limit is None else limit)
        if not 0 < limit <= self.CURRENT_LIMIT_ABSOLUTE:
            raise ValueError(f'current_limit {limit} A outside (0, {self.CURRENT_LIMIT_ABSOLUTE}]')
        self._current_limit = limit
        self.log.info(f'load: {info.get("description", "")} ({os.path.basename(info["file"])}, '
                      f'measured {info.get("measured", "?")}); current limit {limit} A')
        self._rp = RedPitayaScpi(self._host, self._port, logger=self.log)
        try:
            board = self._rp.board_info()
            self.log.info(f'current source Red Pitaya: {self._rp.idn}, '
                          f'{board.get("description", "")} {board.get("version", "")}')
        except Exception as e:
            raise IOError(f'{self._host} does not answer /get_info like the stock OS ({e}); '
                          f'is pyrpl running on it?')
        with self._lock:
            self._rp.gen_off()
            self._rp.set_burst(None)
            self._rp.set_trigger_source('internal')
            self._rp.set_trigger_output(False)
            self._rp.set_func('sine')
            self._rp.set_freq(1000.0)
            self._out = self._zero_state()
            states = self._supply().output_states(self._rails)
            self._powered = all(states.values())
            if any(states.values()) and not self._powered:
                self.log.warning(f'supply channels partly on ({states}): switching them off')
                self._supply().switch_off(self._rails)
        self._timer = QtCore.QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._monitor_tick)
        self._sigStartMonitor.connect(self._start_timer, QtCore.Qt.QueuedConnection)
        self._sigStartMonitor.emit()

    def on_deactivate(self):
        if self._timer is not None:
            self._timer.stop()
            self._timer.timeout.disconnect()
            self._timer = None
        self._sigStartMonitor.disconnect()
        with self._lock:
            try:
                self.power_off()
            except Exception:
                self.log.exception('power off on deactivation failed:')
            finally:
                if self._stop_server:
                    self._rp.stop_server()
                else:
                    self._rp.close()
                self._rp = None

    @property
    def _rails(self):
        return [int(c) for c in self._rail_channels]

    @staticmethod
    def _zero_state():
        return dict(on=False, waveform='sine', frequency=1000.0, amplitude=0.0, offset=0.0,
                    v_amplitude=0.0, v_offset=0.0, predicted_peak=0.0)

    # =========================================================================================
    # description
    # =========================================================================================
    @property
    def constraints(self) -> CurrentSourceConstraints:
        h = self._heat
        return CurrentSourceConstraints(
            waveforms=WAVEFORMS, frequency_range=(0.0, self.FREQUENCY_MAX),
            current_limit=self._current_limit, current_limit_absolute=self.CURRENT_LIMIT_ABSOLUTE,
            dc_continuous_current=h['dc_continuous_current'], dc_high_max_time=h['dc_high_max_time'],
            dc_high_rest_time=h['dc_high_rest_time'],
            ac_continuous_current=h['ac_continuous_current'], ac_high_max_time=h['ac_high_max_time'],
            ac_high_rest_time=h['ac_high_rest_time'],
            capture_length=N_BUF, capture_sample_rate=FS_ADC, arbitrary_length=ARB_LENGTH,
            burst_max_cycles=50000, burst_max_repetitions=50000)

    @property
    def current_limit(self) -> float:
        return self._current_limit

    @current_limit.setter
    def current_limit(self, value: float) -> None:
        value = float(value)
        if not 0 < value <= self.CURRENT_LIMIT_ABSOLUTE:
            raise CommandRefused(f'current limit must be in (0, {self.CURRENT_LIMIT_ABSOLUTE}] A')
        if value < self._out['predicted_peak'] - 1e-9:
            raise CommandRefused('lower the output before lowering the limit')
        self._current_limit = value
        self.log.info(f'current limit {value} A')

    @property
    def load_model(self) -> LoadModel:
        return self._load

    @property
    def amplifier_calibration(self) -> Dict[str, Any]:
        return copy.deepcopy(self._amp)

    def transconductance(self, frequency: float) -> complex:
        """Measured I/OUT1 (A/V, complex) of the load at this frequency (0: DC)."""
        return complex(self._load.transconductance(frequency)[0])

    def expected_current(self, frequency: float, amplitude: float) -> complex:
        """amplitude at the phase of the measured I/OUT1(f): set_output divides the amplitude
        by |I/OUT1|, so the current follows the command with that phase."""
        gm = self.transconductance(frequency)
        return float(amplitude) * (gm / abs(gm) if abs(gm) > 0 else 1.0)

    def _gm_for(self, waveform, f):
        """|I/OUT1| used to convert a requested amplitude: sine at f, all other shapes their
        low-frequency peak (DC transconductance)."""
        return abs(self.transconductance(f)) if waveform == 'sine' else self._load.gm_dc

    def _v_limit(self):
        return float(self._amp['out1_max'] if self.allow_near_rail else self._amp['out1_clean'])

    def max_current(self, waveform: str, frequency: float) -> float:
        """Largest requested peak current (offset 0): the current limit (times GAIN_MARGIN, so
        the predicted current stays within it) or the OUT1 voltage limit."""
        v_avail = self._v_limit() - abs(self._offset_null)
        gm = self._gm_for(waveform, frequency)
        return float(min(self._current_limit * self.GAIN_MARGIN, v_avail * gm))

    def volts_for(self, waveform, frequency, amplitude, offset):
        """OUT1 amplitude and offset (V) for a requested current, and the predicted peak (A)."""
        gm = self._gm_for(waveform, frequency)
        gm_dc = self._load.gm_dc
        v_amp = 0.0 if waveform == 'dc' else float(amplitude) / gm
        v_offs = float(offset) / gm_dc + self._offset_null
        peak = (v_amp * gm + abs(v_offs - self._offset_null) * gm_dc) / self.GAIN_MARGIN
        return v_amp, v_offs, peak

    def _check_setting(self, waveform, frequency, v_amp, v_offs, peak):
        if waveform not in WAVEFORMS:
            raise CommandRefused(f'unknown waveform {waveform!r}; one of {WAVEFORMS}')
        if waveform != 'dc' and not 0 < frequency <= self.FREQUENCY_MAX:
            raise CommandRefused(f'frequency {frequency} Hz outside (0, {self.FREQUENCY_MAX}]')
        v = abs(v_amp) + abs(v_offs)
        if v > self._amp['out1_max'] + 1e-9:
            raise CommandRefused(f'OUT1 {v:.3f} V > {self._amp["out1_max"]} V cap')
        if v > self._v_limit() + 1e-9:
            raise CommandRefused(f'OUT1 {v:.3f} V > {self._amp["out1_clean"]} V clean limit '
                                 '(allow_near_rail = True to go up to the cap)')
        if peak > self._current_limit + 1e-9:
            raise CommandRefused(f'predicted peak {peak:.3f} A > current limit '
                                 f'{self._current_limit} A')

    # =========================================================================================
    # power
    # =========================================================================================
    @property
    def powered(self) -> bool:
        return self._powered

    def power_on(self) -> Dict[str, Any]:
        with self._lock:
            sup = self._supply()
            if self._powered:
                r = sup.measure(self._rails)
                self._check_supply(r)
                return r
            self._zero(off=True)
            rep = sup.configure_channels(self._rails, self.RAIL_VOLTAGE, self.RAIL_CURRENT_LIMIT,
                                         fuse=True, fuse_delay=self.FUSE_DELAY,
                                         fuse_delay_init=self.FUSE_DELAY_INIT, link_fuses=True)
            if not rep['ok']:
                raise RuntimeError(f'supply configuration read-back mismatch: {rep}')
            sup.switch_on(self._rails)
            # the first ~0.3 s can show the 470 uF rail capacitors charging
            time.sleep(0.5)
            t0 = time.time()
            r = None
            while time.time() - t0 < self.POWER_CHECK_TIME:
                r = sup.measure(self._rails)
                bad = [c for c in self._rails if r[f'i{c}'] > self.IDLE_CURRENT_ABORT or r[f'trip{c}']]
                if bad:
                    sup.switch_off(self._rails)
                    self._powered = False
                    self._emit_abort(f'idle check failed on Ch{bad}: {r}', True)
                    raise CurrentSourceAbort(f'idle check failed: {r}', supply_off=True)
            if min(r[f'v{c}'] for c in self._rails) < self.RAIL_MIN_VOLTAGE:
                sup.switch_off(self._rails)
                self._powered = False
                raise CurrentSourceAbort(f'supply did not come up: {r}', supply_off=True)
            self._powered = True
            self.log.info(f'current source powered: {r}')
        self._emit_state()
        return r

    def power_off(self) -> None:
        err = None
        with self._lock:
            try:
                self._zero(off=True)
            except Exception as e:
                err = e
            finally:
                try:
                    self._supply().switch_off(self._rails)
                finally:
                    self._powered = False
        self._emit_state()
        if err is not None:
            raise err

    # =========================================================================================
    # output
    # =========================================================================================
    @property
    def output_state(self) -> Dict[str, Any]:
        st = copy.deepcopy(self._out)
        st.update(powered=self._powered, burst=self._burst, sync_output=self._sync_out,
                  current_limit=self._current_limit, high_time=self._high_times())
        return st

    def _send(self, waveform, f, v_amp, v_offs):
        """Write the generator. Waveform or frequency changes take effect only after a trigger
        (OS 3.00): the caller restarts the generator."""
        rp = self._rp
        changed = False
        if waveform != self._out['waveform']:
            rp.set_func(waveform)
            changed = True
            if waveform == 'arbitrary':
                rp.set_arbitrary(self._arb_table)
                self._arb_loaded = True
        elif waveform == 'arbitrary' and not self._arb_loaded:
            rp.set_arbitrary(self._arb_table)
            self._arb_loaded = True
            changed = True
        if waveform != 'dc' and f != self._out['frequency']:
            rp.set_freq(f)
            changed = True
        rp.set_amp_offs(v_amp, v_offs)
        self._out.update(waveform=waveform, v_amplitude=v_amp, v_offset=v_offs)
        if waveform != 'dc':
            self._out['frequency'] = f
        return changed

    def _magnitude(self):
        return abs(self._out['v_amplitude']) + abs(self._out['v_offset'] - self._offset_null)

    def set_output(self, waveform: str, frequency: float = 0.0, amplitude: float = 0.0,
                   offset: float = 0.0, ramp: bool = True) -> Dict[str, Any]:
        waveform = str(waveform).lower()
        frequency = float(frequency) if waveform != 'dc' else self._out['frequency']
        amplitude, offset = abs(float(amplitude)), float(offset)
        if waveform == 'dc':
            amplitude = 0.0
        v_amp, v_offs, peak = self.volts_for(waveform, frequency, amplitude, offset)
        self._check_setting(waveform, frequency, v_amp, v_offs, peak)
        if self._burst is not None and abs(offset) > 0:
            raise CommandRefused('bursts only with offset 0 (OUT1 is 0 V between bursts)')
        with self._lock:
            if not self._powered:
                raise CommandRefused('the amplifier supply is off: power_on() first')
            self._check_heating(waveform, amplitude, offset)
            rp = self._rp
            if self._burst is not None:
                # burst: the generator waits for trigger(); no ramp (nothing plays until then).
                # OS 3.00 applies waveform, frequency and burst settings only at a trigger and
                # until then keeps playing the old (possibly continuous) waveform at the new
                # amplitude: apply them with one trigger at amplitude 0 and let that empty
                # burst finish before the amplitude is set (measured 2026-11-06).
                if (not self._out['on'] or self._gen_dirty or waveform != self._out['waveform']
                        or frequency != self._out['frequency']):
                    rp.set_amp_offs(0.0, 0.0)
                    self._send(waveform, frequency, 0.0, 0.0)
                    if not self._out['on']:
                        rp.output(True)
                        self._out['on'] = True
                    rp.trigger_now()
                    time.sleep(self._burst[0] / frequency + 0.02)
                    self._gen_dirty = False
                rp.set_amp_offs(v_amp, v_offs)
                self._out.update(v_amplitude=v_amp, v_offset=v_offs)
            else:
                same_shape = (self._out['on'] and waveform == self._out['waveform']
                              and (waveform == 'dc' or frequency == self._out['frequency']))
                new_mag = abs(v_amp) + abs(v_offs - self._offset_null)
                old_mag = self._magnitude() if self._out['on'] else 0.0
                small_step = new_mag <= max(old_mag * self.RAMP_STEP, 1e-4)
                # same shape, scaled up (amplitude and offset by the same factor): ramp from
                # the present level instead of through zero
                k0 = old_mag / new_mag if new_mag > 0 else 0.0
                proportional = (same_shape and old_mag > 1e-9 and k0 < 1
                                and abs(k0 * v_amp - self._out['v_amplitude']) < 1e-6
                                and abs(k0 * (v_offs - self._offset_null)
                                        - (self._out['v_offset'] - self._offset_null)) < 1e-6)
                if ramp and proportional and not small_step:
                    for k in self._ramp_factors(new_mag):
                        if k > k0:
                            self._send(waveform, frequency, k * v_amp,
                                       self._offset_null + k * (v_offs - self._offset_null))
                            time.sleep(0.01)
                            self._quick_check()
                elif ramp and not (same_shape and small_step):
                    if old_mag > 1e-9:
                        self._ramp_down()
                    if not self._out['on']:
                        self._send(waveform, frequency, 0.0, self._offset_null)
                        rp.output(True)
                        rp.trigger_now()
                        self._out['on'] = True
                    elif self._send(waveform, frequency, 0.0, self._offset_null) or self._gen_dirty:
                        rp.trigger_now()
                    self._gen_dirty = False
                    for k in self._ramp_factors(new_mag):
                        self._send(waveform, frequency, k * v_amp,
                                   self._offset_null + k * (v_offs - self._offset_null))
                        time.sleep(0.01)
                        self._quick_check()
                else:
                    # waveform/frequency changes apply only at a trigger: trigger at zero
                    # amplitude first, then set the amplitude
                    if not self._out['on']:
                        self._send(waveform, frequency, 0.0, self._offset_null)
                        rp.output(True)
                        rp.trigger_now()
                        self._out['on'] = True
                    elif (self._gen_dirty or waveform != self._out['waveform']
                          or (waveform != 'dc' and frequency != self._out['frequency'])):
                        self._send(waveform, frequency, 0.0, self._offset_null)
                        rp.trigger_now()
                    self._gen_dirty = False
                    self._send(waveform, frequency, v_amp, v_offs)
            self._out.update(amplitude=amplitude, offset=offset, predicted_peak=peak)
            self._track_heating()
            if self._burst is None:
                self._quick_check()
        self._emit_state()
        return self.output_state

    def _ramp_factors(self, target_mag):
        """Scale factors up to 1 with <= RAMP_STEP per step, from ~1 % (>= 4 steps)."""
        if target_mag <= 0:
            return [1.0]
        ks = [1.0]
        while ks[-1] / self.RAMP_STEP >= 0.01 and (len(ks) < 4 or ks[-1] > 0.25):
            ks.append(ks[-1] / self.RAMP_STEP)
        return ks[::-1]

    def _ramp_down(self, steps=5):
        a, o = self._out['v_amplitude'], self._out['v_offset'] - self._offset_null
        for k in range(steps - 1, -1, -1):
            self._rp.set_amp_offs(a * k / steps, self._offset_null + o * k / steps)
            time.sleep(0.005)
        self._out.update(v_amplitude=0.0, v_offset=self._offset_null, amplitude=0.0, offset=0.0,
                         predicted_peak=0.0)

    def output_off(self) -> None:
        with self._lock:
            if self._out['on'] and self._magnitude() > 1e-9:
                self._ramp_down()
            self._zero(off=True)
        self._emit_state()

    def _zero(self, off=True):
        try:
            if off:
                self._rp.gen_off()
            else:
                self._rp.set_amp_offs(0.0, 0.0)
        finally:
            self._out.update(amplitude=0.0, offset=0.0, v_amplitude=0.0, v_offset=0.0,
                             predicted_peak=0.0)
            if off:
                self._out['on'] = False
            self._track_heating()

    def set_arbitrary_waveform(self, samples: Sequence[float]) -> None:
        x = np.asarray(samples, float).ravel()
        if len(x) < 2 or not np.all(np.isfinite(x)) or np.abs(x).max() == 0:
            raise ValueError('arbitrary waveform needs >= 2 finite samples, not all zero')
        n = len(x)
        table = np.interp(np.arange(ARB_LENGTH) * n / ARB_LENGTH, np.arange(n + 1),
                          np.append(x, x[0]))
        table /= np.abs(table).max()
        with self._lock:
            if self._out['on'] and self._out['waveform'] == 'arbitrary' and self._magnitude() > 0:
                raise CommandRefused('change the arbitrary table only at zero output')
            self._arb_table = table
            self._arb_loaded = False
            if self._out['waveform'] == 'arbitrary':
                self._rp.set_arbitrary(table)
                self._arb_loaded = True
                self._gen_dirty = True

    def configure_burst(self, n_cycles: Optional[int], n_repetitions: int = 1,
                        period: Optional[float] = None) -> None:
        with self._lock:
            if self._out['on'] and self._magnitude() > 1e-9:
                raise CommandRefused('configure bursts only at zero output')
            if n_cycles is None:
                self._rp.set_burst(None)
                self._burst = None
            else:
                n_cycles, n_repetitions = int(n_cycles), int(n_repetitions)
                if not 1 <= n_cycles <= 50000 or not 0 <= n_repetitions <= 50000:
                    raise CommandRefused('burst: 1..50000 cycles, 0..50000 repetitions')
                if period is not None and period <= 0:
                    raise CommandRefused('burst period must be > 0')
                self._rp.set_burst(n_cycles, n_repetitions, period)
                self._burst = (n_cycles, n_repetitions, period)
            self._gen_dirty = True
        self._emit_state()

    def set_sync_output(self, enabled: bool) -> None:
        with self._lock:
            self._rp.set_trigger_output(bool(enabled), 'DAC')
            self._sync_out = bool(enabled)
        self._emit_state()

    def trigger(self) -> None:
        with self._lock:
            if not self._out['on']:
                raise CommandRefused('output is off')
            self._rp.trigger_now()

    # =========================================================================================
    # heating rules (limits from the load file)
    # =========================================================================================
    def _classes(self, waveform, amplitude, offset):
        out = set()
        if abs(offset) > self._heat['dc_continuous_current'] + 1e-9:
            out.add('dc')
        if waveform != 'dc' and amplitude + abs(offset) > self._heat['ac_continuous_current'] + 1e-9:
            out.add('ac')
        return out

    def _limits(self, cls):
        return self._heat[f'{cls}_high_max_time'], self._heat[f'{cls}_high_rest_time']

    def _check_heating(self, waveform, amplitude, offset):
        now = time.time()
        for cls in self._classes(waveform, amplitude, offset):
            since, end = self._high[cls]
            t_max, rest = self._limits(cls)
            if since is None and now - end < rest:
                raise CommandRefused(f'{cls.upper()} above the continuous limit needs {rest:.0f} s '
                                     f'rest ({now - end:.0f} s so far)')
            if since is not None and now - since > t_max:
                raise CommandRefused(f'{cls.upper()} above the continuous limit for '
                                     f'{now - since:.0f} s already (max {t_max:.0f} s)')

    def _track_heating(self):
        now = time.time()
        on = self._out['on']
        active = self._classes(self._out['waveform'], self._out['amplitude'],
                               self._out['offset']) if on else set()
        for cls in ('dc', 'ac'):
            if cls in active and self._high[cls][0] is None:
                self._high[cls][0] = now
            elif cls not in active and self._high[cls][0] is not None:
                self._high[cls] = [None, now]

    def _high_times(self):
        now = time.time()
        return {cls: (now - since if since is not None else 0.0) for cls, (since, _) in self._high.items()}

    # =========================================================================================
    # measurement
    # =========================================================================================
    def _acquire(self, dec):
        self._rp.acq_setup(dec)
        raw = self._rp.acquire_now()
        return raw[self._shunt_input - 1], raw[self._monitor_input - 1]

    def _to_physical(self, shunt, mon):
        return ((np.asarray(shunt) - self._adc_offset_shunt) / self.SHUNT_OHM,
                (np.asarray(mon) - self._adc_offset_mon) * self.MON_RATIO)

    def _decimation(self):
        f = self._out['frequency'] if self._out['waveform'] != 'dc' else 0.0
        if self._burst is not None:
            return 64
        return choose_decimation(f)

    def _quick_check(self):
        if self._capture_armed:
            return
        dec = self._decimation()
        shunt, mon = self._acquire(dec)
        self._check_traces(shunt, mon, FS_ADC / dec)
        self._check_supply(self.read_supply())

    def _guard_armed(self):
        """An immediate acquisition (ACQ:RST) would discard an armed triggered capture."""
        if self._capture_armed:
            if time.time() - self._armed_since < 10.0:
                raise CommandRefused('a triggered capture is armed: read_capture() first')
            self.log.warning('armed capture was never read (> 10 s): released')
            self._capture_armed = False

    def measure(self, n_average: int = 1) -> Dict[str, Any]:
        """Readings (keys as in the interface docstring):
        i_* shunt current; v_* output voltage (MON x 16); i_*_output = output voltage / Z(f) of
        the load file; i_*_command = the setting. Sine (continuous): *_amp, phase_i_minus_v,
        z_re/z_im (output / current), i_thd, v_thd; otherwise only mean/rms/peak."""
        with self._lock:
            self._guard_armed()
            out = self._out
            f = out['frequency'] if (out['waveform'] == 'sine' and out['on'] and self._burst is None) else 0.0
            dec = self._decimation()
            self._rp.acq_setup(dec)
            fs = self._rp.fs
            cur, vout, fits = [], [], []
            for _ in range(max(1, int(n_average))):
                raw = self._rp.acquire_now()
                shunt, mon = raw[self._shunt_input - 1], raw[self._monitor_input - 1]
                self._check_traces(shunt, mon, fs)
                i, v = self._to_physical(shunt, mon)
                cur.append(i)
                vout.append(v)
                if f > 0:
                    fits.append((fit_sine(i, f, sample_rate=fs), fit_sine(v, f, sample_rate=fs)))
            sup = self.read_supply()
            self._check_supply(sup)
            ci, cv = np.array(cur), np.array(vout)
            on = out['on']
            res = dict(t=time.time(), waveform=out['waveform'], frequency=out['frequency'],
                       amplitude_set=out['amplitude'], offset_set=out['offset'], on=on,
                       burst=self._burst is not None, decimation=dec, sample_rate=fs,
                       i_mean=float(ci.mean()), i_rms=float(np.sqrt(np.mean(ci ** 2))),
                       i_peak=float(np.abs(ci).max()), v_mean=float(cv.mean()),
                       v_rms=float(np.sqrt(np.mean(cv ** 2))), v_peak=float(np.abs(cv).max()),
                       i_mean_output=float(cv.mean()) / self._load.r_dc,
                       i_mean_command=float(out['offset']) if on else 0.0,
                       supply=sup, supply_power=self._supply_power(sup))
            if f > 0:
                pi = np.array([a.phasor for a, b in fits])
                pv = np.array([b.phasor for a, b in fits])
                ref = pv / np.abs(pv)
                z = np.mean(pv / pi)
                z_load = complex(self._load.impedance(f)[0])
                res.update(i_amp=float(np.mean(np.abs(pi))), v_amp=float(np.mean(np.abs(pv))),
                           i_amp_std=float(np.std(np.abs(pi))),
                           i_amp_output=float(np.mean(np.abs(pv))) / abs(z_load),
                           i_amp_command=float(out['amplitude']),
                           phase_i_minus_v=float(np.angle(np.mean(pi / ref))),
                           i_offset=float(np.mean([a.offset for a, b in fits])),
                           z_re=float(z.real), z_im=float(z.imag),
                           i_thd=harmonic_distortion(ci[0], f, fs),
                           v_thd=harmonic_distortion(cv[0], f, fs),
                           i_rel_resid=float(np.mean([a.rel_resid for a, b in fits])))
            self._last_measure = res
            return res

    def capture(self, decimation: int = 64) -> Dict[str, Any]:
        with self._lock:
            self._guard_armed()
            shunt, mon = self._acquire(int(decimation))
            fs = self._rp.fs
            self._check_traces(shunt, mon, fs)
            i, v = self._to_physical(shunt, mon)
            return dict(t=np.arange(len(i)) / fs, current=i, output=v, sample_rate=fs,
                        decimation=int(decimation), trigger_index=0, time=time.time())

    def arm_capture(self, decimation: int = 64, trigger: str = 'generator',
                    pre_trigger: int = 0) -> None:
        if trigger not in ('generator', 'external'):
            raise ValueError("trigger must be 'generator' or 'external'")
        with self._lock:
            self._rp.acq_setup(int(decimation), int(pre_trigger))
            self._rp.arm(trigger)
            self._capture_armed = True
            self._armed_since = time.time()
            self._capture_pre = int(pre_trigger)

    def read_capture(self, timeout: float = 5.0) -> Dict[str, Any]:
        with self._lock:
            if not self._capture_armed:
                raise RuntimeError('no capture armed')
            try:
                raw = self._rp.wait_and_read(timeout)
            finally:
                self._capture_armed = False
            fs = self._rp.fs
            shunt, mon = raw[self._shunt_input - 1], raw[self._monitor_input - 1]
            k = self._capture_pre
            self._check_traces(shunt, mon, fs, k)
            i, v = self._to_physical(shunt, mon)
            return dict(t=(np.arange(len(i)) - k) / fs, current=i, output=v, sample_rate=fs,
                        decimation=self._rp.decimation, trigger_index=k, time=time.time())

    def read_supply(self) -> Dict[str, Any]:
        try:
            return self._supply().measure(self._rails)
        except Exception as e:     # a read-back failure is not itself a reason to abort
            self.log.warning(f'supply read-back failed: {e}')
            return {}

    def _supply_power(self, sup):
        try:
            return float(sum(sup[f'v{c}'] * sup[f'i{c}'] for c in self._rails))
        except KeyError:
            return float('nan')

    # =========================================================================================
    # checks and abort
    # =========================================================================================
    def _emit_abort(self, reason, supply_off):
        self.sigSafetyAbort.emit(reason, supply_off)

    def _abort(self, reason, supply=False):
        self.log.error(f'CURRENT SOURCE ABORT: {reason}')
        try:
            self._zero(off=True)
        finally:
            if supply:
                try:
                    self._supply().switch_off(self._rails)
                finally:
                    self._powered = False
            self._emit_abort(reason, supply)
            self._emit_state()
        raise CurrentSourceAbort(reason, supply_off=supply)

    def _commanded_zero(self):
        return (not self._out['on']) or self._magnitude() < 1e-6

    @property
    def last_checked_traces(self) -> Optional[Dict[str, Any]]:
        """Raw input traces (V) of the last safety check, kept so that an abort can be
        examined: dict(shunt, mon, sample_rate, pre_trigger, time, output_state)."""
        return copy.deepcopy(self._last_checked)

    def _check_traces(self, shunt, mon, fs, pre_trigger=0):
        shunt, mon = np.asarray(shunt), np.asarray(mon)
        self._last_checked = dict(shunt=shunt.copy(), mon=mon.copy(), sample_rate=float(fs),
                                  pre_trigger=int(pre_trigger),
                                  time=time.time(), output_state=copy.deepcopy(self._out))
        lim = min(self.SHUNT_ABORT_V, self.SHUNT_OHM * self._current_limit * 1.08)
        if np.abs(shunt).max() > lim:
            self._abort(f'shunt peak {np.abs(shunt).max():.3f} V > {lim:.3f} V')
        if np.abs(mon).max() > self.MON_ABORT_V:
            self._abort(f'MON peak {np.abs(mon).max():.3f} V > {self.MON_ABORT_V} V')
        if np.abs(shunt).max() > self.CLIP_V or np.abs(mon).max() > self.CLIP_V:
            self._abort('Red Pitaya input near full scale (clipping)')
        if self._commanded_zero() and self._burst is None:
            s1, s2 = shunt.std(), mon.std()
            if s1 > 0.005 or s2 > 0.005:
                self._abort(f'signal with OUT1 = 0: shunt {s1 * 1e3:.1f} mV rms, MON {s2 * 1e3:.1f} '
                            f'mV rms (oscillation?)', supply=True)
            return
        i, v = self._to_physical(shunt, mon)
        ratio, v_expected = self._load.output_ratio(i, v, fs)
        if v_expected > 0.1:            # ~ 30 mA: below, the ratio is dominated by noise
            if not 0.33 < ratio < 3.0:
                self._abort(f'measured output / output expected from the current = {ratio:.2f} '
                            '(open load, short, or wrong load file?)')
        elif np.sqrt(np.mean((v * np.hanning(len(v))) ** 2)) > 1.0:
            self._abort(f'amplifier output {np.abs(v).max():.2f} V without current (open load?)')

    def _check_supply(self, sup):
        if not sup:
            return
        idle = self._commanded_zero()
        for ch in self._rails:
            i = sup.get(f'i{ch}')
            if sup.get(f'trip{ch}'):
                self._abort(f'supply Ch{ch} fuse tripped', supply=True)
            if i is not None and i > self.RAIL_CURRENT_ABORT:
                self._abort(f'supply Ch{ch} current {i:.3f} A > {self.RAIL_CURRENT_ABORT} A',
                            supply=True)
            if idle and self._burst is None and i is not None and i > self.IDLE_CURRENT_ABORT:
                self._abort(f'supply Ch{ch} idle current {i * 1e3:.0f} mA > '
                            f'{self.IDLE_CURRENT_ABORT * 1e3:.0f} mA', supply=True)
        if self._powered and not all(sup.get(f'on{c}', 1) for c in self._rails):
            self.log.error('supply channels went off')
            self._powered = False
            self._zero(off=True)
            self._emit_state()

    # =========================================================================================
    # monitor (main thread timer)
    # =========================================================================================
    @QtCore.Slot()
    def _start_timer(self):
        if self._timer is not None and self._monitor_interval > 0:
            self._timer.start(int(1000 * self._monitor_interval))

    @QtCore.Slot()
    def _monitor_tick(self):
        try:
            if self._powered and not self._capture_armed and self._lock.tryLock():
                try:
                    now = time.time()
                    for cls, (since, _) in self._high.items():
                        t_max, _rest = self._limits(cls)
                        if since is not None and now - since > t_max:
                            self.log.error(f'{cls.upper()} above the continuous current for '
                                           f'{t_max:.0f} s: output ramped down')
                            self.output_off()
                            self._emit_abort(f'{cls.upper()} time limit reached', False)
                    if self._out['on']:
                        self.measure(1)
                    else:
                        self._check_supply(self.read_supply())
                finally:
                    self._lock.unlock()
        except CurrentSourceAbort:
            pass                        # logged and signalled by _abort
        except Exception as e:
            self.log.warning(f'current source monitor: {e}')
        finally:
            self._start_timer()

    def _emit_state(self):
        self.sigStateChanged.emit(self.output_state)
