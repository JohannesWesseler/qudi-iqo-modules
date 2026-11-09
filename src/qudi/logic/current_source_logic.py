# -*- coding: utf-8 -*-
"""
Control of a waveform current source (CurrentSourceInterface, e.g. the Red Pitaya + LM1875
source in hardware/current_source): output in A, live monitor of current, output voltage and
supply, scope traces, automatic switch-off when idle, and saving.

The hardware module enforces every limit; this logic adds convenience (trimming the current to
a measured target, the live view) and the idle rule. It knows nothing about the load: what the
current does there (a field, heat) belongs to the experiment's logic.

Script use (blocking calls, run in the caller's thread; exceptions propagate):

    src = current_source_logic
    src.power_on()
    src.set_output('sine', 10e3, amplitude=0.5)               # 0.5 A peak, 10 kHz
    src.set_output('sine', 10e3, amplitude=0.5, trim=True)    # trimmed to the measured current
    m = src.measure(n_average=5)        # dict: i_amp, i_amp_output, i_amp_command, supply, ...
    src.output_off(); src.power_off()

GUI use goes through request(...) (queued to the logic thread; errors -> sigError).

Example config:

    current_source_logic:
        module.Class: 'current_source_logic.CurrentSourceLogic'
        connect:
            source: current_source
        options:
            idle_off_minutes: 30      # supply off after this long without a command (0 = never)
            monitor_interval: 0.5     # s between live readings (0 = no live monitor)
            scope_every: 2            # a scope trace every n-th live reading
            history_length: 3600

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
from collections import deque

import numpy as np
from PySide2 import QtCore
from matplotlib.figure import Figure

from qudi.core.module import LogicBase
from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.util.mutex import RecursiveMutex
from qudi.util.datastorage import TextDataStorage
from qudi.util.sine_fit import fit_sine

CAPTURE_LENGTH = 16384          # samples per capture (Red Pitaya); used only to choose decimations
CAPTURE_RATE = 125e6

_RESPONSE_KEYS = ('f', 'amplitude', 'cycles', 'repeats', 'v_out1', 'gm_abs', 'gm_ph_deg',
                  'gm_abs_std', 'gm_ph_std_deg', 'z_re', 'z_im', 'gain_abs', 'gain_ph_deg',
                  'i_amp', 'v_amp', 'i_resid', 'v_resid', 'i_pre_mean', 'v_pre_mean',
                  'decimation', 'samples')


def burst_response(capture, frequency, v_out1, t_end, skip=50e-6):
    """One captured sine burst (time 0 = generator start, OUT1 = v_out1 sin(2 pi f t)):
    phasors of the current and the output voltage fitted over [skip, t_end], and
    gm = I / OUT1 (A/V), Z = V / I (Ohm), gain = V / OUT1. Also the means before the start
    (pre-trigger samples, output at zero)."""
    t = np.asarray(capture['t'], float)
    i = np.asarray(capture['current'], float)
    v = np.asarray(capture['output'], float)
    sel = (t >= skip) & (t <= t_end)
    if np.count_nonzero(sel) < 16:
        raise ValueError(f'only {np.count_nonzero(sel)} samples in the fit window')
    fi = fit_sine(i[sel], frequency, t=t[sel])
    fv = fit_sine(v[sel], frequency, t=t[sel])
    pre = t < 0
    return dict(i=fi.phasor, v=fv.phasor, gm=fi.phasor / v_out1, z=fv.phasor / fi.phasor,
                gain=fv.phasor / v_out1, i_resid=fi.resid_rms, v_resid=fv.resid_rms,
                i_pre_mean=float(i[pre].mean()) if pre.any() else np.nan,
                v_pre_mean=float(v[pre].mean()) if pre.any() else np.nan,
                samples=int(np.count_nonzero(sel)))


_HISTORY_KEYS = ('t', 'i_amp', 'i_amp_output', 'i_mean', 'i_mean_output', 'i_rms', 'v_amp', 'v_rms',
                 'phase_i_minus_v',
                 'supply_power', 'i_supply_pos', 'i_supply_neg')


class CurrentSourceLogic(LogicBase):
    """Current source control with live monitor."""

    _source = Connector(name='source', interface='CurrentSourceInterface')

    _idle_off_minutes = ConfigOption('idle_off_minutes', default=30.0, missing='nothing')
    _monitor_interval = ConfigOption('monitor_interval', default=0.5, missing='nothing')
    _scope_every = ConfigOption('scope_every', default=2, missing='nothing')
    _history_length = ConfigOption('history_length', default=3600, missing='nothing')

    _live = StatusVar('live_monitor', default=True)
    _scope_decimation = StatusVar('scope_decimation', default=0)     # 0: automatic

    sigMeasurement = QtCore.Signal(dict)          # every live reading
    sigHistory = QtCore.Signal(dict)              # arrays of the history (t relative, ...)
    sigScope = QtCore.Signal(dict)                # t, current, output
    sigStateChanged = QtCore.Signal(dict)         # output state of the hardware
    sigError = QtCore.Signal(str)
    sigAbort = QtCore.Signal(str, bool)

    _sigRequest = QtCore.Signal(str, dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = RecursiveMutex()
        self._timer = None
        self._history = deque()
        self._t0 = time.time()
        self._last_command = time.time()
        self._tick = 0
        self._last_measurement = {}
        self._last_scope = None
        self._stop = False
        self._last_response = None

    # --- qudi --------------------------------------------------------------------------------
    def on_activate(self):
        self._history = deque(maxlen=int(self._history_length))
        self._last_command = time.time()
        src = self._source()
        src.sigStateChanged.connect(self.sigStateChanged, QtCore.Qt.QueuedConnection)
        src.sigSafetyAbort.connect(self._on_abort, QtCore.Qt.QueuedConnection)
        self._sigRequest.connect(self._handle_request, QtCore.Qt.QueuedConnection)
        self._timer = QtCore.QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._tick_live)
        self._timer.start(int(1000 * max(self._monitor_interval, 0.5)))

    def on_deactivate(self):
        if self._timer is not None:
            self._timer.stop()
            self._timer.timeout.disconnect()
            self._timer = None
        src = self._source()
        src.sigStateChanged.disconnect(self.sigStateChanged)
        src.sigSafetyAbort.disconnect(self._on_abort)
        self._sigRequest.disconnect()

    # --- state -------------------------------------------------------------------------------
    @property
    def output_state(self):
        return self._source().output_state

    @property
    def constraints(self):
        return self._source().constraints

    def max_current(self, waveform, frequency):
        return self._source().max_current(waveform, frequency)

    @property
    def live_monitor(self):
        return bool(self._live)

    @live_monitor.setter
    def live_monitor(self, on):
        self._live = bool(on)

    @property
    def scope_decimation(self):
        return int(self._scope_decimation)

    @scope_decimation.setter
    def scope_decimation(self, dec):
        self._scope_decimation = int(dec)

    @property
    def last_measurement(self):
        return copy.deepcopy(self._last_measurement)

    @property
    def last_scope(self):
        return copy.deepcopy(self._last_scope)

    def history(self):
        with self._lock:
            rows = list(self._history)
        out = {k: np.array([r.get(k, np.nan) for r in rows], float) for k in _HISTORY_KEYS}
        return out

    # --- commands (blocking, caller's thread) ---------------------------------------------------
    def _touch(self):
        self._last_command = time.time()

    def power_on(self):
        self._touch()
        return self._source().power_on()

    def power_off(self):
        self._touch()
        self._source().power_off()

    def set_output(self, waveform, frequency=0.0, amplitude=0.0, offset=0.0, ramp=True, trim=False,
                   tolerance=0.01, max_iterations=4):
        """amplitude, offset in A. trim (sine only): adjust the amplitude until the measured
        current amplitude is within `tolerance` of the target (the conversion uses the measured
        small-signal transconductance; the large-signal gain is ~0.5 % higher)."""
        self._touch()
        src = self._source()
        st = src.set_output(waveform, frequency, amplitude, offset, ramp)
        if trim and waveform == 'sine' and amplitude > 0:
            target = float(amplitude)
            cmd = target
            for _ in range(int(max_iterations)):
                m = src.measure(2)
                err = target / m['i_amp']
                if abs(err - 1) < tolerance:
                    break
                cmd = min(cmd * err, src.max_current('sine', frequency))
                st = src.set_output(waveform, frequency, cmd, offset, ramp)
            st['trimmed_amplitude_command'] = cmd
        return st

    def output_off(self):
        self._touch()
        self._source().output_off()

    def measure(self, n_average=1):
        self._touch()
        m = self._source().measure(n_average)
        self._record(m)
        return m

    def capture(self, decimation=None):
        self._touch()
        dec = int(decimation or self._auto_decimation())
        c = self._source().capture(dec)
        self._last_scope = c
        self.sigScope.emit(c)
        return c

    def arm_capture(self, decimation=64, trigger='generator', pre_trigger=0):
        """Capture on the next generator start (see the hardware's arm_capture)."""
        self._touch()
        self._source().arm_capture(int(decimation), trigger, int(pre_trigger))

    def read_capture(self, timeout=5.0):
        self._touch()
        return self._source().read_capture(timeout)

    def expected_current(self, frequency, amplitude):
        """Current phasor the source expects for a sine of this amplitude (see the interface)."""
        return complex(self._source().expected_current(float(frequency), float(amplitude)))

    def configure_burst(self, n_cycles, n_repetitions=1, period=None):
        self._touch()
        self._source().configure_burst(n_cycles, n_repetitions, period)

    def set_sync_output(self, enabled):
        self._touch()
        self._source().set_sync_output(enabled)

    def set_arbitrary_waveform(self, samples):
        self._touch()
        self._source().set_arbitrary_waveform(samples)

    def trigger(self):
        self._touch()
        self._source().trigger()

    def set_current_limit(self, value):
        self._touch()
        self._source().current_limit = float(value)

    # --- characterization ------------------------------------------------------------------------
    def stop_measurement(self):
        """Ask a running frequency_response to stop after the current burst."""
        self._stop = True

    def frequency_response(self, frequencies, amplitudes, periods=5, repeats=3, skip=50e-6,
                           pre_trigger=200, save=True, nametag='frequency_response',
                           progress=None):
        """Response of the source with its load to sine bursts, for every frequency (Hz) and
        amplitude (A peak). Each burst plays `periods` periods after `skip` (s, the start
        transient); a capture triggered on the generator start records current and output, so
        all phases are relative to OUT1 = v_out1 sin(2 pi f t). Per point (mean of `repeats`
        bursts): gm = I/OUT1 (A/V), Z = V/I (Ohm), gain = V/OUT1, the fit residuals, and the
        current and output before the start (pre_trigger samples at zero output).

        The amplitude is converted to OUT1 with the present load file; points above
        max_current are skipped. Needs the supply on. Returns dict(rows=[...], meta). The rows
        give the columns of a load-file table (z_re, z_im, gm_abs, gm_ph_deg)."""
        src = self._source()
        if not src.powered:
            raise RuntimeError('power_on() first')
        self._stop = False
        live, self._live = self._live, False          # no live acquisitions in between
        rows, t0 = [], time.time()
        freqs = [float(f) for f in np.atleast_1d(frequencies)]
        amps = [float(a) for a in np.atleast_1d(amplitudes)]
        error = None
        try:
            src.output_off()
            for f in freqs:
                n_cyc = int(np.ceil(periods + skip * f))
                t_end = n_cyc / f
                dec = int(np.ceil(t_end * 1.05 * CAPTURE_RATE / (CAPTURE_LENGTH - pre_trigger)))
                if dec > 65536:
                    raise ValueError(f'{f} Hz: {n_cyc} periods do not fit in one capture')
                src.output_off()
                src.configure_burst(n_cyc, 1, None)
                for a in amps:
                    if self._stop:
                        raise RuntimeError('stopped')
                    if a > src.max_current('sine', f):
                        self.log.info(f'{f:g} Hz, {a:g} A: above max_current, skipped')
                        continue
                    self._touch()
                    st = src.set_output('sine', f, a)
                    v_out1 = float(st['v_amplitude'])
                    res = []
                    for _ in range(int(repeats)):
                        src.arm_capture(dec, 'generator', pre_trigger)
                        src.trigger()
                        cap = src.read_capture(t_end * 1.2 + 5.0)
                        res.append(burst_response(cap, f, v_out1, t_end, skip))
                        time.sleep(0.02)
                    gm = np.array([r['gm'] for r in res])
                    gm_m = complex(np.mean(gm))
                    z = complex(np.mean([r['z'] for r in res]))
                    gain = complex(np.mean([r['gain'] for r in res]))
                    row = dict(f=f, amplitude=a, cycles=n_cyc, repeats=len(res), v_out1=v_out1,
                               gm_abs=abs(gm_m), gm_ph_deg=float(np.degrees(np.angle(gm_m))),
                               gm_abs_std=float(np.std(np.abs(gm))),
                               gm_ph_std_deg=float(np.degrees(np.std(np.angle(gm / gm_m)))),
                               z_re=z.real, z_im=z.imag, gain_abs=abs(gain),
                               gain_ph_deg=float(np.degrees(np.angle(gain))),
                               i_amp=float(np.mean([abs(r['i']) for r in res])),
                               v_amp=float(np.mean([abs(r['v']) for r in res])),
                               i_resid=float(np.mean([r['i_resid'] for r in res])),
                               v_resid=float(np.mean([r['v_resid'] for r in res])),
                               i_pre_mean=float(np.mean([r['i_pre_mean'] for r in res])),
                               v_pre_mean=float(np.mean([r['v_pre_mean'] for r in res])),
                               decimation=cap['decimation'], samples=res[0]['samples'])
                    rows.append(row)
                    msg = (f"{f:9.2f} Hz {a * 1e3:7.1f} mA: gm {row['gm_abs']:.4f} A/V "
                           f"{row['gm_ph_deg']:+7.2f} deg (+-{row['gm_ph_std_deg']:.2f}), "
                           f"Z {z.real:.3f}{z.imag:+.3f}j, gain {row['gain_abs']:.3f}")
                    self.log.info(msg)
                    if progress is not None:
                        progress(msg)
                src.output_off()
        except Exception as e:          # keep and save what was measured, then re-raise
            error = e
        finally:
            try:
                src.output_off()
                src.configure_burst(None)
            finally:
                self._live = live
        meta = dict(kind='frequency_response', start=time.strftime('%Y-%m-%dT%H:%M:%S',
                                                                    time.localtime(t0)),
                    duration_s=time.time() - t0, periods=periods, repeats=repeats, skip=skip,
                    pre_trigger=pre_trigger, frequencies=freqs, amplitudes=amps,
                    error=None if error is None else f'{type(error).__name__}: {error}')
        out = dict(rows=rows, meta=meta)
        self._last_response = out
        if save and rows:
            storage = TextDataStorage(root_dir=self.module_default_data_dir)
            data = np.array([[r[k] for k in _RESPONSE_KEYS] for r in rows], float)
            path, _, _ = storage.save_data(data, metadata=meta, nametag=nametag,
                                           column_headers=list(_RESPONSE_KEYS))
            storage.save_thumbnail(self.plot_frequency_response(rows), path.rsplit('.', 1)[0])
            out['file'] = path
        if error is not None:
            raise error
        return out

    @property
    def last_frequency_response(self):
        """Result of the last frequency_response, also when it was stopped by an error."""
        return copy.deepcopy(self._last_response)

    @staticmethod
    def plot_frequency_response(rows):
        """|gm| and phase per amplitude, plus both relative to the largest amplitude measured at
        each frequency (where the amplitude dependence is visible)."""
        from matplotlib import cm
        amps = sorted({r['amplitude'] for r in rows})
        ref = {}
        for r in rows:
            if r['f'] not in ref or r['amplitude'] > ref[r['f']]['amplitude']:
                ref[r['f']] = r
        fig = Figure(figsize=(8, 10))
        axes = [fig.add_subplot(4, 1, k + 1) for k in range(4)]
        for k, a in enumerate(amps):
            c = cm.viridis(k / max(1, len(amps) - 1))
            sel = sorted((r for r in rows if r['amplitude'] == a), key=lambda r: r['f'])
            f = [r['f'] for r in sel]
            axes[0].semilogx(f, [r['gm_abs'] for r in sel], 'o-', ms=3, color=c,
                             label=f'{a * 1e3:g} mA')
            axes[1].semilogx(f, [r['gm_ph_deg'] for r in sel], 'o-', ms=3, color=c)
            axes[2].semilogx(f, [100 * (r['gm_abs'] / ref[r['f']]['gm_abs'] - 1) for r in sel],
                             'o-', ms=3, color=c)
            axes[3].semilogx(f, [r['gm_ph_deg'] - ref[r['f']]['gm_ph_deg'] for r in sel],
                             'o-', ms=3, color=c)
        axes[0].set_ylabel('|I / OUT1| (A/V)')
        axes[1].set_ylabel('phase I vs OUT1 (deg)')
        axes[2].set_ylabel('|gm| vs largest\namplitude (%)')
        axes[3].set_ylabel('phase vs largest\namplitude (deg)')
        axes[3].set_xlabel('frequency (Hz)')
        axes[0].legend(fontsize=7, ncol=4)
        for ax in axes:
            ax.grid(alpha=0.3, which='both')
        fig.tight_layout()
        return fig

    # --- GUI requests (queued to the logic thread) ----------------------------------------------
    def request(self, command, **kwargs):
        """Run a command in the logic thread (GUI): 'power_on', 'power_off', 'set_output',
        'output_off', 'capture', 'set_current_limit', 'configure_burst',
        'set_sync_output', 'trigger', 'save'. Errors are logged and sent with sigError."""
        self._sigRequest.emit(str(command), dict(kwargs))

    @QtCore.Slot(str, dict)
    def _handle_request(self, command, kwargs):
        allowed = ('power_on', 'power_off', 'set_output', 'output_off', 'capture',
                   'set_current_limit', 'configure_burst', 'set_sync_output', 'trigger', 'save')
        if command not in allowed:
            self.sigError.emit(f'unknown command {command}')
            return
        try:
            getattr(self, command)(**kwargs)
        except Exception as e:
            self.log.error(f'{command} failed: {e}')
            self.sigError.emit(f'{command}: {e}')

    # --- live monitor ----------------------------------------------------------------------------
    def _auto_decimation(self):
        if self._scope_decimation:
            return int(self._scope_decimation)
        st = self._source().output_state
        f = st['frequency'] if st['waveform'] != 'dc' else 0.0
        if f <= 0:
            return 1024
        # about 5 periods per trace
        return int(min(65536, max(1, np.ceil(5 * 125e6 / (16384 * f)))))

    def _record(self, m):
        sup = m.get('supply', {}) or {}
        row = dict(t=m['t'], i_amp=m.get('i_amp', np.nan), i_amp_output=m.get('i_amp_output', np.nan),
                   i_mean=m['i_mean'], i_mean_output=m.get('i_mean_output', np.nan), i_rms=m['i_rms'],
                   v_amp=m.get('v_amp', np.nan), v_rms=m['v_rms'],
                   phase_i_minus_v=m.get('phase_i_minus_v', np.nan),
                   supply_power=m.get('supply_power', np.nan),
                   i_supply_pos=sup.get('i1', np.nan), i_supply_neg=sup.get('i4', np.nan))
        with self._lock:
            self._history.append(row)
        self._last_measurement = m
        self.sigMeasurement.emit(m)
        h = self.history()
        h['t'] = h['t'] - self._t0
        self.sigHistory.emit(h)

    @QtCore.Slot()
    def _tick_live(self):
        try:
            src = self._source()
            idle_s = time.time() - self._last_command
            if (self._idle_off_minutes and src.powered
                    and idle_s > 60 * float(self._idle_off_minutes)):
                self.log.warning(f'current source idle for {idle_s / 60:.0f} min: switching it off')
                src.power_off()
            elif self._live and src.powered:
                self._record(src.measure(1))
                self._tick += 1
                if self._scope_every and self._tick % int(self._scope_every) == 0:
                    c = src.capture(self._auto_decimation())
                    self._last_scope = c
                    self.sigScope.emit(c)
        except Exception as e:
            self.log.debug(f'current source live monitor: {e}')
        finally:
            if self._timer is not None:
                self._timer.start(int(1000 * max(self._monitor_interval, 0.2)))

    @QtCore.Slot(str, bool)
    def _on_abort(self, reason, supply_off):
        self.log.error(f'current source abort: {reason}' + (' (supply off)' if supply_off else ''))
        self.sigAbort.emit(reason, supply_off)

    # --- saving ------------------------------------------------------------------------------------
    def save(self, nametag='current_source', notes=''):
        """Save the history (live readings) and the last scope trace to the data directory."""
        storage = TextDataStorage(root_dir=self.module_default_data_dir)
        h = self.history()
        meta = dict(output_state=self.output_state, notes=notes)
        data = np.column_stack([h[k] for k in _HISTORY_KEYS]) if len(h['t']) else np.zeros((0, len(_HISTORY_KEYS)))
        path, _, _ = storage.save_data(data, metadata=meta, nametag=nametag + '_history',
                                       column_headers=list(_HISTORY_KEYS))
        paths = [path]
        if self._last_scope is not None:
            c = self._last_scope
            p2, _, _ = storage.save_data(np.column_stack([c['t'], c['current'], c['output']]),
                                         metadata=dict(meta, sample_rate=c['sample_rate'],
                                                       decimation=c['decimation']),
                                         nametag=nametag + '_scope',
                                         column_headers=['t (s)', 'current (A)', 'output (V)'])
            fig = Figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            ax.plot(c['t'] * 1e3, c['current'], lw=0.8)
            ax.set_xlabel('t (ms)')
            ax.set_ylabel('current (A)')
            storage.save_thumbnail(fig, p2.rsplit('.', 1)[0])
            paths.append(p2)
        self.log.info(f'current source data saved: {paths}')
        return paths
