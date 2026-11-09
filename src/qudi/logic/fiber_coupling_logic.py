# -*- coding: utf-8 -*-
"""
Automatic single-mode fiber coupling with two motorized mirror mounts (open-loop steppers).

Routines (each ends with a result dict: values, checks, scans; saved to the data directory):
    quick      steep-axis peak of every motor pair, repeated while a pair moves by more than its
               peak width (the pairs are mechanically coupled). ~0.5-1 min. Use after drifts.
    fine       quick, then a beam-walk (flat-axis) scan of every pair with all pairs re-peaked at
               every point, then quick. Moves on the flat axis only for a significant gain. ~8 min.
    calibrate  per motor: scans up and down (width, backlash) and after +-excursions (lost steps);
               updates the ridge frame of every pair (steep direction and width). ~5 min, weekly.

Hardware rules encoded here (Laser_alignment/docs/REPORT.md):
- Optimize in the ridge frame of each motor pair (the two motors, one on each mirror, that tilt
  the beam in the same plane): 'steep' = mirrors against each other (narrow coupling peak),
  'flat' = both together (beam walk, broad).
- Backlash (40-80 steps): a motor reverses only via an overshoot of `backlash_approach` steps.
- Replay rule: a position is reproducible to ~1 step only when reached by the identical motion.
  A steep scan starts with a pre-load of `preload_steps` in scan direction; the park position is
  reached by replaying the same pre-load and sub-steps.

Persistent state (status variables, shared by every qudi config that uses the same module name):
the calibrated ridge frame, the power-meter calibration, the alignment history and the best
coupling value. Keep the module name ('fiber_coupling_logic') the same in all configs.

Example config:

    fiber_coupling_logic:
        module.Class: 'fiber_coupling_logic.FiberCouplingLogic'
        connect:
            mount: mirror_mount
            metric: redpitaya_input_monitor
        options:
            metric_channel: 'nirvana_monitor'
            pairs:                      # start values; 'calibrate' refines steep, flat and sigma
                - name: 'M1_M4'
                  motors: [0, 3]
                  steep: [0.7241, -0.6897]
                  flat: [0.6897, 0.7241]
                  sigma_steep: 18.0
                - name: 'M2_M3'
                  motors: [1, 2]
                  steep: [0.7234, -0.6905]
                  flat: [0.6905, 0.7234]
                  sigma_steep: 27.1
            splitter_ratio: 99.0        # power main port / tap port of the fiber splitter
            # optional (defaults):
            # preload_steps: 130
            # backlash_approach: 120
            # steep_range_sigma: 3.0
            # steep_points_per_sigma: 5.0
            # flat_span: 300
            # flat_step: 100
            # metric_noise: 0.015       # relative slow fluctuation of the metric
            # compare_tolerance: 0.02   # relative difference treated as real
            # park_tolerance: 0.06      # warn if parking ends further below the scan maximum
            # min_signal: 0.0           # below: no light
            # calibration_excursion: 300
            # calibration_interval_days: 7
            # trace_interval: 0.25
            # save_runs: True

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
import datetime
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


class _Aborted(Exception):
    pass


def _gauss(x, a, x0, s, c):
    return a * np.exp(-0.5 * ((x - x0) / s) ** 2) + c


def fit_peak(x, y):
    """Gaussian fit of a peak profile. Returns dict(amplitude, center, sigma, offset) or None."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 5 or not np.isfinite(y).all() or y.max() <= y.min():
        return None
    try:
        from scipy.optimize import curve_fit
        k = int(np.argmax(y))
        p0 = [y.max() - y.min(), x[k], max(np.ptp(x) / 6, 1.0), y.min()]
        p, _ = curve_fit(_gauss, x, y, p0=p0, maxfev=4000)
    except Exception:
        return None
    if p[0] <= 0 or p[2] == 0 or not x.min() <= p[1] <= x.max():
        return None
    return dict(amplitude=float(p[0]), center=float(p[1]), sigma=float(abs(p[2])),
                offset=float(p[3]))


def _now():
    return datetime.datetime.now().isoformat(timespec='seconds')


class FiberCouplingLogic(LogicBase):
    """Fiber-coupling optimizer for two stepper mirror mounts."""

    _mount = Connector(name='mount', interface='StepperMirrorMountInterface')
    _metric = Connector(name='metric', interface='ProcessValueInterface')

    _metric_channel = ConfigOption('metric_channel', missing='error')
    _config_pairs = ConfigOption('pairs', missing='error')
    _splitter_ratio = ConfigOption('splitter_ratio', default=99.0, checker=lambda x: x > 0)
    _preload = ConfigOption('preload_steps', default=130)
    _approach = ConfigOption('backlash_approach', default=120)
    _steep_range = ConfigOption('steep_range_sigma', default=3.0, checker=lambda x: x > 1)
    _steep_density = ConfigOption('steep_points_per_sigma', default=5.0, checker=lambda x: x > 0)
    _flat_span = ConfigOption('flat_span', default=300, checker=lambda x: x > 0)
    _flat_step = ConfigOption('flat_step', default=100, checker=lambda x: x > 0)
    _noise = ConfigOption('metric_noise', default=0.015, checker=lambda x: x > 0)
    _tolerance = ConfigOption('compare_tolerance', default=0.02, checker=lambda x: x >= 0)
    _park_tolerance = ConfigOption('park_tolerance', default=0.06, checker=lambda x: x >= 0)
    _min_signal = ConfigOption('min_signal', default=0.0)
    _excursion = ConfigOption('calibration_excursion', default=300, checker=lambda x: x > 0)
    _cal_interval = ConfigOption('calibration_interval_days', default=7.0)
    _trace_interval = ConfigOption('trace_interval', default=0.25, checker=lambda x: x > 0)
    _trace_maxlen = ConfigOption('trace_length', default=20000)
    _save_runs = ConfigOption('save_runs', default=True)

    _calibration = StatusVar('calibration', default=None)
    _calibration_history = StatusVar('calibration_history', default=list())
    _power_calibration = StatusVar('power_calibration', default=None)
    _history = StatusVar('history', default=list())
    _reference = StatusVar('reference_value', default=None)
    _last_result = StatusVar('last_result', default=None)
    _directions = StatusVar('last_directions', default=None)
    _live_trace = StatusVar('live_trace', default=True)

    sigTraceUpdated = QtCore.Signal(object, object)         # t (s), metric
    sigScanUpdated = QtCore.Signal(dict)                    # live and finished scan profiles
    sigPositionsUpdated = QtCore.Signal(object)
    sigBusyChanged = QtCore.Signal(bool, str)
    sigProgress = QtCore.Signal(str)
    sigRoutineFinished = QtCore.Signal(dict)
    sigCalibrationChanged = QtCore.Signal(object)
    sigPowerCalibrationChanged = QtCore.Signal(object)
    sigHistoryChanged = QtCore.Signal(object)

    _sigRun = QtCore.Signal(str)
    _sigJog = QtCore.Signal(int, int)
    _sigPowerCal = QtCore.Signal(float, float)

    ROUTINES = ('quick', 'fine', 'calibrate')
    _HISTORY_LENGTH = 2000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = RecursiveMutex()
        self._abort = False
        self._busy = False
        self._t0 = time.time()
        self._trace_t = deque()
        self._trace_m = deque()
        self._run_log = []
        self._run_scans = {}
        self._run_checks = []
        self._last_emit = 0.0
        self._timer = None
        self._pairs = []

    # --- qudi ---------------------------------------------------------------------------------
    def on_activate(self):
        n = len(self._mount().motor_names)
        for p in self._config_pairs:
            for key in ('name', 'motors', 'steep', 'flat', 'sigma_steep'):
                if key not in p:
                    raise ValueError(f'pair definition {p} lacks "{key}"')
        self._pairs = self._frame_from(self._calibration)
        if not self._directions or len(self._directions) != n:
            self._directions = [0] * n
        self._trace_t = deque(maxlen=int(self._trace_maxlen))
        self._trace_m = deque(maxlen=int(self._trace_maxlen))
        self._sigRun.connect(self._run, QtCore.Qt.QueuedConnection)
        self._sigJog.connect(self._jog, QtCore.Qt.QueuedConnection)
        self._sigPowerCal.connect(self._store_power_calibration, QtCore.Qt.QueuedConnection)
        self._timer = QtCore.QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._idle_tick)
        self._timer.start(int(1000 * self._trace_interval))

    def on_deactivate(self):
        self._abort = True
        if self._timer is not None:
            self._timer.stop()
            self._timer.timeout.disconnect()
            self._timer = None
        self._sigRun.disconnect()
        self._sigJog.disconnect()
        self._sigPowerCal.disconnect()

    def _frame_from(self, calibration):
        """Pair definitions of the config, with steep/flat/sigma from a matching calibration."""
        pairs = [dict(p, motors=[int(m) for m in p['motors']],
                      steep=[float(c) for c in p['steep']], flat=[float(c) for c in p['flat']],
                      sigma_steep=float(p['sigma_steep'])) for p in self._config_pairs]
        cal = (calibration or {}).get('pairs', {})
        for p in pairs:
            c = cal.get(p['name'])
            if c and list(c.get('motors', [])) == p['motors']:
                p.update(steep=list(c['steep']), flat=list(c['flat']),
                         sigma_steep=float(c['sigma_steep']))
        return pairs

    # --- properties --------------------------------------------------------------------------
    @property
    def pairs(self):
        return copy.deepcopy(self._pairs)

    @property
    def motor_names(self):
        return self._mount().motor_names

    @property
    def metric_unit(self):
        return self._metric().constraints.channel_units.get(self._metric_channel, '')

    @property
    def busy(self):
        return self._busy

    @property
    def splitter_ratio(self):
        return float(self._splitter_ratio)

    @property
    def calibration(self):
        return copy.deepcopy(self._calibration)

    @property
    def calibration_due(self):
        """True if there is no calibration or it is older than calibration_interval_days."""
        if not self._calibration:
            return True
        age = datetime.datetime.now() - datetime.datetime.fromisoformat(self._calibration['time'])
        return age.total_seconds() > 86400 * float(self._cal_interval)

    @property
    def power_calibration(self):
        return copy.deepcopy(self._power_calibration)

    @property
    def history(self):
        return copy.deepcopy(self._history)

    @property
    def reference_value(self):
        return self._reference

    @property
    def last_result(self):
        return copy.deepcopy(self._last_result)

    @property
    def live_trace(self):
        return bool(self._live_trace)

    @live_trace.setter
    def live_trace(self, on):
        self._live_trace = bool(on)

    def positions(self):
        return self._mount().get_positions()

    def trace(self):
        with self._lock:
            return np.array(self._trace_t), np.array(self._trace_m)

    def estimate_power(self, value):
        """Power at the main splitter port and coupling efficiency for a metric value, scaled
        from the power-meter calibration (assumes unchanged laser power and a linear metric).
        Returns (power_main in W, efficiency) or None."""
        c = self._power_calibration
        if not c or not c.get('metric'):
            return None
        scale = float(value) / c['metric']
        return c['power_main'] * scale, c['efficiency'] * scale

    # --- public API ----------------------------------------------------------------------------
    def start_routine(self, routine):
        """Start 'quick', 'fine' or 'calibrate' (returns immediately; see sigRoutineFinished)."""
        if routine not in self.ROUTINES:
            raise ValueError(f'unknown routine {routine}; one of {self.ROUTINES}')
        if self._busy:
            self.log.warning('Fiber coupling is busy; stop the running routine first')
            return
        self._busy = True           # set here, so wait_until_idle cannot miss the start
        self._sigRun.emit(routine)

    def stop_routine(self):
        self._abort = True

    def wait_until_idle(self, timeout=None):
        """Block until no routine runs (for scripts; not from the logic thread). Returns the last
        result, or None on timeout."""
        t_end = None if timeout is None else time.time() + timeout
        while self._busy:
            if t_end is not None and time.time() > t_end:
                return None
            time.sleep(0.1)
        return self.last_result

    def jog(self, motor, steps):
        """Plain relative move (no backlash handling), e.g. from the GUI buttons."""
        if self._busy:
            self.log.warning('Fiber coupling is busy; jog ignored')
            return
        self._sigJog.emit(int(motor), int(steps))

    def store_power_calibration(self, power_input, power_tap):
        """Power-meter calibration: power_input (W) in front of the fiber splitter, power_tap (W)
        at its tap port, measured while the fiber is connected. The metric is read when called."""
        if self._busy:
            self.log.warning('Fiber coupling is busy; power calibration not stored')
            return
        self._sigPowerCal.emit(float(power_input), float(power_tap))

    def reset_reference(self):
        """Forget the best value (e.g. after a change of laser power)."""
        self._reference = None
        self.sigHistoryChanged.emit(self.history)

    # --- slots (logic thread) ------------------------------------------------------------------
    @QtCore.Slot(int, int)
    def _jog(self, motor, steps):
        try:
            self._mount().move_steps(motor, steps)
            self._directions[motor] = 0      # direction state unknown after a plain jog
        except Exception as e:
            self.log.error(f'Jog failed: {e}')
        self.sigPositionsUpdated.emit(self.positions())

    @QtCore.Slot(float, float)
    def _store_power_calibration(self, power_input, power_tap):
        if power_input <= 0 or power_tap <= 0:
            self.log.error('Power calibration needs both powers > 0')
            return
        vals = [self._read_metric() for _ in range(25)]
        m = float(np.mean(vals))
        if m <= self._min_signal:
            self.log.error(f'Power calibration: no coupling signal ({m:.4g} {self.metric_unit})')
            return
        r = float(self._splitter_ratio)
        self._power_calibration = dict(
            time=_now(), power_input=power_input, power_tap=power_tap, splitter_ratio=r,
            power_main=power_tap * r, efficiency=power_tap * (1 + r) / power_input,
            metric=m, metric_std=float(np.std(vals)), unit=self.metric_unit)
        c = self._power_calibration
        self.log.info(f'Power calibration: {c["power_main"] * 1e3:.4g} mW at the main port, '
                      f'coupling efficiency {c["efficiency"]:.1%} at {m:.4g} {self.metric_unit}')
        self._dump_status_variables()
        self.sigPowerCalibrationChanged.emit(self.power_calibration)

    @QtCore.Slot(str)
    def _run(self, routine):
        self._busy = True
        self._abort = False
        self._run_log, self._run_scans, self._run_checks = [], {}, []
        self.module_state.lock()
        self.sigBusyChanged.emit(True, routine)
        t_start = time.time()
        result = dict(routine=routine, time=_now(), ok=False, checks=[])
        try:
            result['start'] = self._mean_metric(5)
            if routine == 'quick':
                self._quick()
            elif routine == 'fine':
                self._quick(check=False)
                for p in self._pairs:
                    self._flat_search(p)
                self._quick()
            elif routine == 'calibrate':
                self._calibrate()
            vals = [self._measure() for _ in range(10)]
            result.update(ok=True, final=float(np.mean(vals)), final_std=float(np.std(vals)))
            msg = (f'{routine} done in {time.time() - t_start:.0f} s: '
                   f'{result["start"]:.4g} -> {result["final"]:.4g} {self.metric_unit}')
            self.log.info(msg)
        except _Aborted:
            msg = f'{routine} stopped'
            self.log.info(msg)
        except Exception as e:
            msg = f'{routine} failed: {e}'
            self.log.exception(msg)
            self._check(False, f'{routine} failed: {e}')
            if hasattr(self._mount(), 'reconnect') and 'timeout' in str(e).lower():
                try:
                    self._mount().reconnect()
                    self.log.warning('Mount reconnected after a communication failure; '
                                     'positions may be off by the last move')
                except Exception as e2:
                    self.log.error(f'Reconnect failed: {e2}')
        finally:
            result.update(duration=float(time.time() - t_start), message=msg,
                          scans=copy.deepcopy(self._run_scans))
            try:
                self._finish(result)
            except Exception:
                self.log.exception('Evaluating the routine result failed:')
            self._busy = False
            self.module_state.unlock()
            self.sigPositionsUpdated.emit(self.positions())
            self.sigBusyChanged.emit(False, '')
            self.sigRoutineFinished.emit(result)

    def _finish(self, result):
        """Add the checks, update best value and history, save the run."""
        unit = self.metric_unit
        if result['ok'] and result['routine'] in ('quick', 'fine'):
            start, final = result['start'], result['final']
            ref = self._reference
            # start and final differ by the slow fluctuation of the metric alone: 2 sigma of a
            # difference before calling it a loss
            loss_limit = max(self._tolerance, 2 * np.sqrt(2) * self._noise)
            self._check(final >= start * (1 - loss_limit) or start <= self._min_signal,
                        f'signal {start:.4g} -> {final:.4g} {unit}')
            if ref:
                ok = final >= 0.9 * ref
                self._check(ok, f'{final / ref:.0%} of the best value ({ref:.4g} {unit})'
                            + ('' if ok else ': try fine alignment, or check the laser power'))
            if self._power_calibration:
                p, eta = self.estimate_power(final)
                self._check(True, f'about {p * 1e3:.3g} mW at the main port, efficiency ~{eta:.0%} '
                            f'(power calibration of {self._power_calibration["time"][:10]})')
            if ref is None or final > ref:
                self._reference = final
            self._history.append(dict(time=result['time'], routine=result['routine'],
                                      start=start, final=final,
                                      duration=result['duration'],
                                      warnings=sum(not c['ok'] for c in self._run_checks)))
            del self._history[:-self._HISTORY_LENGTH]
            self.sigHistoryChanged.emit(self.history)
        if self.calibration_due and result['routine'] != 'calibrate':
            self._check(False, 'motor calibration is due (button "Calibrate")')
        result['checks'] = list(self._run_checks)
        result['verdict'] = ('failed' if not result['ok'] else
                             'ok' if all(c['ok'] for c in self._run_checks) else 'warning')
        if self._save_runs and self._run_log:
            try:
                result['file'] = self._save_run(result)
            except Exception:
                self.log.exception('Saving the fiber coupling run failed:')
        self._last_result = {k: v for k, v in result.items() if k != 'scans'}
        # qudi saves status variables only on deactivation; a crashed or killed qudi would lose
        # calibrations and history
        self._dump_status_variables()

    @QtCore.Slot()
    def _idle_tick(self):
        try:
            if not self._busy and self._live_trace:
                self._append(self._read_metric())
        except Exception as e:
            self.log.debug(f'trace read failed: {e}')
        finally:
            if self._timer is not None:
                self._timer.start(int(1000 * self._trace_interval))

    # --- measurement ---------------------------------------------------------------------------
    def _read_metric(self):
        return float(self._metric().get_process_value(self._metric_channel))

    def _append(self, m):
        with self._lock:
            t = time.time() - self._t0
            self._trace_t.append(t)
            self._trace_m.append(m)
        now = time.time()
        if now - self._last_emit > 0.2:
            self._last_emit = now
            self.sigTraceUpdated.emit(*self.trace())
        return t

    def _measure(self):
        if self._abort:
            raise _Aborted()
        m = self._read_metric()
        t = self._append(m)
        self._run_log.append([t] + list(self.positions()) + [m])
        return m

    def _mean_metric(self, n):
        return float(np.mean([self._measure() for _ in range(n)]))

    def _check(self, ok, text):
        self._run_checks.append(dict(ok=bool(ok), text=text))

    def _progress(self, text):
        self.sigProgress.emit(text)

    # --- motion --------------------------------------------------------------------------------
    def _move_to(self, motor, target, direction):
        """Go to target, finishing with travel in direction (+1/-1); reversal via overshoot."""
        if self._abort:
            raise _Aborted()
        mount = self._mount()
        target = int(round(target))
        pos = mount.get_positions()[motor]
        delta = target - pos
        if delta * direction < 0 or (delta == 0 and self._directions[motor] != direction):
            mount.move_steps(motor, target - direction * self._approach - pos)
            delta = target - mount.get_positions()[motor]
        if delta:
            mount.move_steps(motor, delta)
        self._directions[motor] = direction

    def _goto(self, motors, coords, dirs):
        for m, c, d in zip(motors, coords, dirs):
            self._move_to(m, c, d)

    def _pair_pos(self, pair):
        pos = self.positions()
        return np.array([pos[m] for m in pair['motors']], float)

    # --- steep axis ----------------------------------------------------------------------------
    def _steep_peak(self, pair, range_sigma=None, max_shift=3):
        """Steep-axis scan of +-range sigma (pre-load in scan direction), recentred if the peak is
        at the edge; parks on the best grid point by replaying the scan path."""
        motors = pair['motors']
        e = np.array(pair['steep'], float)
        s = float(pair['sigma_steep'])
        dirs = np.array([1 if c >= 0 else -1 for c in e])
        inc = max(1, int(round(s / float(self._steep_density))))
        half = int(float(range_sigma or self._steep_range) * s // inc) * inc
        T = np.arange(-half, half + 1, inc)
        start = self._pair_pos(pair)
        p0 = start.copy()
        V, k, recentred, nolight = [], len(T) // 2, 0, False
        for _ in range(max_shift):
            self._goto(motors, p0 + T[0] * e - dirs * self._preload, dirs)
            V = []
            for t in T:
                self._goto(motors, p0 + t * e, dirs)
                V.append(self._measure())
                self.sigScanUpdated.emit(dict(kind='steep', pair=pair['name'],
                                              x=T[:len(V)].tolist(), y=list(V), done=False))
            k = int(np.argmax(V))
            if V[k] <= self._min_signal:
                k, nolight = len(T) // 2, True      # no light: park on the scan centre
                break
            if 0 < k < len(T) - 1:
                break
            p0 = p0 + T[k] * e                      # peak at the scan edge: recentre, rescan
            recentred += 1
        # replay: identical pre-load and sub-steps up to the best grid point. (Parking on the
        # fitted centre instead was tested: ~3.5 % lower, 2026-10-19.)
        self._goto(motors, p0 + T[0] * e - dirs * self._preload, dirs)
        for t in T[:k + 1]:
            self._goto(motors, p0 + t * e, dirs)
        replay = self._measure()
        fit = fit_peak(T, V)
        scan = dict(kind='steep', pair=pair['name'], x=T.tolist(), y=[float(v) for v in V],
                    best_x=float(T[k]), vmax=float(V[k]), replay=float(replay), fit=fit,
                    sigma_ratio=fit['sigma'] / s if fit else None, recentred=recentred,
                    nolight=nolight, moved=float(np.dot(self._pair_pos(pair) - start, e)),
                    done=True)
        self._run_scans[f'steep {pair["name"]}'] = scan
        self.sigScanUpdated.emit(scan)
        self.sigPositionsUpdated.emit(self.positions())
        return scan

    def _quick(self, max_rounds=3, check=True):
        """Steep peak of every pair; again while a pair moved by more than its peak width."""
        for rnd in range(max_rounds):
            moved = False
            for p in self._pairs:
                self._progress(f'steep peak {p["name"]} (round {rnd + 1})')
                scan = self._steep_peak(p)
                moved |= abs(scan['moved']) > p['sigma_steep']
            if not moved:
                break
        if check:
            self._check_steep_scans()

    def _check_steep_scans(self):
        for p in self._pairs:
            scan = self._run_scans.get(f'steep {p["name"]}')
            if scan is None:
                continue
            name = p['name']
            if scan['nolight']:
                self._check(False, f'{name}: no light in the steep scan (beam far off; use '
                                   f'the manual controls or a cold search)')
                continue
            r = scan['sigma_ratio']
            ok = r is not None and 0.6 <= r <= 1.6
            text = (f'{name}: peak width {r:.2f} x calibration' if r is not None
                    else f'{name}: peak shape not fitted')
            if not ok:
                text += ' (a motor may slip: check its cable and coupler, then calibrate)'
            self._check(ok, text)
            loss = 1 - scan['replay'] / scan['vmax']
            ok = loss <= self._park_tolerance
            self._check(ok, f'{name}: parked at {100 * (1 - loss):.0f} % of the scan maximum'
                        + ('' if ok else ' (lost steps or drift during the scan)'))

    # --- flat axis -----------------------------------------------------------------------------
    def _flat_search(self, pair, max_rounds=2):
        """Beam walk of one pair: scan the flat axis over +-flat_span (all pairs steep-peaked at
        every point), fit a parabola and move to the best point of the fit within the scan
        (vertex or edge) if its predicted gain is significant. If that is the scan edge, scan
        again around it (up to max_rounds)."""
        name = pair['name']
        f = np.array(pair['flat'], float)
        fdirs = np.array([1 if c >= 0 else -1 for c in f])
        total, v = 0.0, None
        for rnd in range(max_rounds):
            X, Y, P = self._flat_scan(pair, rnd)
            target, gain, parabola = self._flat_target(X, Y)
            i0 = int(np.argmin(np.abs(X)))
            if target != 0.0:
                j = int(np.argmin(np.abs(X - target)))
                v = self._go_and_peak(pair, P[j] + (target - X[j]) * f, fdirs)
                if v < Y[i0] * (1 - self._tolerance):
                    self.log.info(f'{name}: beam-walk target {target:+.0f} gave {v:.4g}, '
                                  f'back to the start point')
                    target = 0.0
            if target == 0.0:
                v = self._go_and_peak(pair, P[i0], fdirs)
            total += target
            scan = dict(kind='flat', pair=name, x=X.tolist(), y=Y.tolist(), parabola=parabola,
                        best_x=float(target), predicted_gain=gain, value=float(v), done=True)
            self._run_scans[f'flat {name}'] = scan
            self.sigScanUpdated.emit(scan)
            if abs(target) < X.max():          # not at the scan edge: done
                break
        self._check(True, f'{name}: beam walk {total:+.0f} steps' +
                    ('' if total else ' (no significant gain)'))
        return v

    def _flat_scan(self, pair, rnd):
        """Flat-axis scan around the current position with all pairs steep-peaked per point.
        Returns offsets X, peak values Y and the pair position P after each point."""
        motors = pair['motors']
        f = np.array(pair['flat'], float)
        e = np.array(pair['steep'], float)
        fdirs = np.array([1 if c >= 0 else -1 for c in f])
        step = int(self._flat_step)
        nside = max(1, int(round(float(self._flat_span) / step)))
        offs = step * np.arange(-nside, nside + 1)
        base = self._pair_pos(pair)
        X, Y, P = [], [], []
        self._goto(motors, base + offs[0] * f - fdirs * self._preload, fdirs)
        for off in offs:
            self._progress(f'beam walk {pair["name"]}: {off:+d} steps (round {rnd + 1})')
            # keep the steep coordinate of the previous point (the ridge is straight)
            steep_now = np.dot(self._pair_pos(pair) - base, e) if P else 0.0
            self._goto(motors, base + off * f + steep_now * e, fdirs)
            v = self._steep_peak(pair)['replay']
            P.append(self._pair_pos(pair))
            for q in self._pairs:                   # flat moves detune the other pairs too
                if q['name'] != pair['name']:
                    v = self._steep_peak(q)['replay']
            X.append(float(off))
            Y.append(float(v))
            self.sigScanUpdated.emit(dict(kind='flat', pair=pair['name'], x=list(X), y=list(Y),
                                          done=False))
        return np.array(X), np.array(Y), P

    def _flat_target(self, X, Y):
        """Best point of a parabola fit within the scan and its predicted gain over offset 0.
        Returns (target offset or 0 if the gain is not significant, gain, parabola coefficients).
        The gain must exceed compare_tolerance and twice its standard error, from the larger of
        metric_noise and the scatter about the fit."""
        xm = X.max()
        xs = X / xm
        A = np.vstack([xs ** 2, xs, np.ones_like(xs)]).T
        c = np.linalg.lstsq(A, Y, rcond=None)[0]
        resid = Y - A @ c
        sigma = max(self._noise * Y.mean(), np.sqrt(np.sum(resid ** 2) / max(1, len(Y) - 3)))
        grid = np.linspace(-1, 1, 201)
        model = np.polyval(c, grid)
        xt = float(grid[int(np.argmax(model))])
        y0 = float(np.polyval(c, 0.0))
        g = np.array([xt ** 2, xt, 0.0])             # model(xt) - model(0) = g . c
        err = float(sigma * np.sqrt(g @ np.linalg.inv(A.T @ A) @ g))
        diff = float(np.polyval(c, xt) - y0)
        gain = diff / y0
        parabola = dict(c2=float(c[0] / xm ** 2), c1=float(c[1] / xm), c0=float(c[2]))
        significant = gain > self._tolerance and diff > 2 * err
        return (xt * xm if significant else 0.0), float(gain), parabola

    def _go_and_peak(self, pair, target, fdirs):
        """Approach a pair position along the flat directions (pre-load), steep-peak all pairs."""
        self._goto(pair['motors'], np.asarray(target) - fdirs * self._preload, fdirs)
        self._goto(pair['motors'], target, fdirs)
        v = self._steep_peak(pair)['replay']
        for q in self._pairs:
            if q['name'] != pair['name']:
                v = self._steep_peak(q)['replay']
        return v

    # --- calibration ---------------------------------------------------------------------------
    def _motor_scan(self, motor, home, half, inc, direction, label):
        """Scan one motor over home +-half in direction (pre-load first). Returns (T, V, fit)."""
        T = np.arange(-half, half + 1, inc)[::direction]
        self._move_to(motor, home + T[0] - direction * self._preload, direction)
        V = []
        for t in T:
            self._move_to(motor, home + t, direction)
            V.append(self._measure())
            self.sigScanUpdated.emit(dict(kind='motor', motor=int(motor), label=label,
                                          x=T[:len(V)].tolist(), y=list(V), done=False))
        fit = fit_peak(T, V)
        scan = dict(kind='motor', motor=int(motor), label=label, x=T.tolist(),
                    y=[float(v) for v in V], fit=fit, done=True)
        self._run_scans[f'{self.motor_names[motor]} {label}'] = scan
        self.sigScanUpdated.emit(scan)
        return T, V, fit

    def _calibrate(self):
        """Per motor: width, backlash and lost steps; per pair: steep width and direction.
        The steep direction follows from the single-motor widths (|e_i| = sigma / sigma_i); its
        signs are kept. The new frame is used only if all fits of the pair succeeded."""
        names = self.motor_names
        E = int(self._excursion)
        self._quick(check=False)
        new_pairs, motors_res = {}, {}
        for p in self._pairs:
            widths = []
            for j, m in enumerate(p['motors']):
                self._steep_peak(p)                      # start each motor on the peak
                sig_m = p['sigma_steep'] / max(abs(p['steep'][j]), 0.3)
                half = int(4 * sig_m)
                inc = max(2, int(round(sig_m / 4)))
                home = self.positions()[m]
                self._progress(f'calibrate {names[m]}: up and down')
                _, _, up = self._motor_scan(m, home, half, inc, +1, 'up')
                _, _, down = self._motor_scan(m, home, half, inc, -1, 'down')
                centres = [up['center'] if up else np.nan]
                for exc in (+E, -E):
                    self._progress(f'calibrate {names[m]}: after {exc:+d} steps')
                    self._move_to(m, home + exc, 1 if exc > 0 else -1)
                    self._move_to(m, home, 1)
                    _, _, r = self._motor_scan(m, home, half, inc, +1, f'after {exc:+d}')
                    centres.append(r['center'] if r else np.nan)
                self._move_to(m, home, 1)
                res = dict(sigma=up['sigma'] if up else None,
                           backlash=up['center'] - down['center'] if up and down else None,
                           lost_steps=float(np.nanmax(np.abs(np.array(centres[1:]) - centres[0])))
                           if np.isfinite(centres).all() else None)
                motors_res[names[m]] = res
                widths.append(res['sigma'])
                self._check_motor(names[m], res)
            self._progress(f'calibrate {p["name"]}: steep width')
            scan = self._steep_peak(p, range_sigma=4)
            if scan['fit'] is None or None in widths:
                self._check(False, f'{p["name"]}: fits failed, previous frame kept')
                continue
            mag = 1 / np.array(widths)
            steep = np.sign(p['steep']) * mag / np.linalg.norm(mag)
            flat = np.array([abs(steep[1]), abs(steep[0])]) * np.sign(p['flat'])
            sigma = float(scan['fit']['sigma'])
            turn = np.degrees(np.arccos(np.clip(abs(np.dot(steep, p['steep'])), 0, 1)))
            plausible = 0.5 <= sigma / p['sigma_steep'] <= 2 and turn < 20
            self._check(plausible, f'{p["name"]}: steep sigma {p["sigma_steep"]:.1f} -> '
                        f'{sigma:.1f} steps, direction turned {turn:.1f} deg'
                        + ('' if plausible else ': implausible, previous frame kept'))
            if plausible:
                new_pairs[p['name']] = dict(motors=list(p['motors']), steep=steep.tolist(),
                                            flat=flat.tolist(), sigma_steep=sigma)
        cal = copy.deepcopy(self._calibration) or {}
        cal.setdefault('pairs', {}).update(new_pairs)
        cal.update(time=_now(), motors=motors_res, excursion=E,
                   checks=[c for c in self._run_checks])
        self._calibration = cal
        self._calibration_history.append(dict(time=cal['time'], motors=motors_res,
                                              pairs=copy.deepcopy(new_pairs)))
        del self._calibration_history[:-self._HISTORY_LENGTH]
        self._pairs = self._frame_from(self._calibration)
        self.sigCalibrationChanged.emit(self.calibration)
        self._quick()

    def _check_motor(self, name, res):
        b, lost = res['backlash'], res['lost_steps']
        if b is None or lost is None:
            self._check(False, f'{name}: peak not fitted (no light, or the motor does not move)')
            return
        margin = min(self._approach, self._preload)
        self._check(abs(b) < 0.8 * margin, f'{name}: backlash {b:.0f} steps' +
                    ('' if abs(b) < 0.8 * margin else
                     f' (close to backlash_approach/preload_steps = {margin}: raise them)'))
        self._check(lost <= 3, f'{name}: returns within {lost:.1f} steps after +-{self._excursion}'
                    + ('' if lost <= 3 else ' (loses steps: check cable, driver and coupler)'))

    # --- saving --------------------------------------------------------------------------------
    def _save_run(self, result):
        names = list(self.motor_names)
        unit = self.metric_unit
        storage = TextDataStorage(root_dir=self.module_default_data_dir)
        meta = {k: result[k] for k in ('routine', 'time', 'verdict', 'start', 'final', 'duration')
                if k in result}
        # '%' is the interpolation character of the header's configparser
        checks = [('ok: ' if c['ok'] else 'WARNING: ') + c['text'] for c in result['checks']]
        meta.update(checks=[c.replace('%', 'percent') for c in checks],
                    metric_channel=self._metric_channel, pairs=self._pairs,
                    preload_steps=self._preload, backlash_approach=self._approach)
        path, timestamp, _ = storage.save_data(
            np.array(self._run_log, float), metadata=meta,
            nametag=f'fiber_coupling_{result["routine"]}',
            column_headers=['time (s)'] + [f'{n} (steps)' for n in names] + [f'metric ({unit})'])
        storage.save_thumbnail(self._draw_figure(result), path.rsplit('.', 1)[0])
        self.log.info(f'Fiber coupling run saved to {path}')
        return path

    def _draw_figure(self, result):
        scans = result['scans']
        groups = ('steep', 'flat') if result['routine'] != 'calibrate' else ('motor',)
        panels = [k for k, s in scans.items() if s['kind'] in groups]
        rows = 1 + (len(panels) + 2) // 3
        fig = Figure(figsize=(12, 3.2 * rows))
        log = np.array(self._run_log, float)
        ax = fig.add_subplot(rows, 1, 1)
        ax.plot(log[:, 0] - log[0, 0], log[:, -1], lw=0.8)
        ax.set_xlabel('time (s)')
        ax.set_ylabel(f'signal ({self.metric_unit})')
        ax.set_title(f'{result["time"]}: {result.get("message", "")}', loc='left')
        for i, key in enumerate(panels):
            s = scans[key]
            a = fig.add_subplot(rows, 3, 4 + i)
            a.plot(s['x'], s['y'], 'o', ms=3)
            if s.get('fit'):
                xf = np.linspace(min(s['x']), max(s['x']), 200)
                f = s['fit']
                a.plot(xf, _gauss(xf, f['amplitude'], f['center'], f['sigma'], f['offset']), lw=1)
            if s.get('parabola'):
                xf = np.linspace(min(s['x']), max(s['x']), 200)
                q = s['parabola']
                a.plot(xf, np.polyval([q['c2'], q['c1'], q['c0']], xf), lw=1)
            a.set_title(key, fontsize=9, loc='left')
            a.set_xlabel('steps', fontsize=8)
        fig.tight_layout()
        return fig
