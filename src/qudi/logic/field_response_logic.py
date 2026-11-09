# -*- coding: utf-8 -*-
"""
Response of the ODMR frequency tracking to magnetic test fields: a coil driven by a waveform
current source (CurrentSourceLogic; the source and its accuracy are documented in
hardware/current_source/README.md). Setup, procedure and results: docs/field_response_measurements.md
in qudi-iqo-modules.

Measurements (blocking calls; run them from a script, the qudi console or a notebook while the
tracking engine runs, open or closed loop, see the ODMR Tracking GUI):

    calibrate_field(currents)       DC current steps -> MHz/A of the tracked resonance (and mT/A
                                    along the NV axis); stored here (nv_calibration)
    measure_clock_ratio()           ratio of the two Red Pitaya clocks (frequency of the tracked
                                    response to one long sine burst)
    transfer_function(frequencies)  sine bursts -> H(f) = tracked shift / applied shift
    response_map(freqs, amps_ut)    transfer_function for each field amplitude (uT along the NV
                                    axis): the frequency x amplitude map of the tracking
    step_response(amplitude)        current steps (averaged over repetitions) -> rise time, delay,
                                    overshoot of the tracked frequency
    amplitude_sweep(f, amplitudes)  H and the error signal versus field amplitude at fixed f
                                    (large-signal limit, lock loss)

The NV calibration (MHz of line shift per A of coil current, per tracked resonance slot) belongs
to this experiment, not to the current source: it depends on the coil, its position and the NV
orientation. calibrate_field measures it; set_nv_calibration sets it by hand.

Amplitudes: in A (amplitude), MHz of tracked shift (amplitude_mhz) or uT along the NV axis
(amplitude_ut), the latter two through the NV calibration.

Applied current (current_reference): H(f) needs the phasor of the current at each frequency.
- 'command' (default): the source's own estimate, expected_current(f, amplitude): the set
  amplitude at the phase of the measured I/OUT1(f) of the load. The source's README gives its
  accuracy (<= 1 % and <= 1 deg from 20 Hz to 50 kHz at 0.2 A; a few % at mA currents).
- 'shunt': the current captured by the source at each burst start.
Every row has shunt_over_cmd and shunt_phase_deg (shunt capture / command), so the two can be
compared. At mA currents the shunt reading of the present source has errors of several %
(README, "Accuracy"); quote the uncertainty of the applied current with the results.

Synchronization: every generator start of the coil Red Pitaya (RP2) emits an edge on its DIO0_N
(trigger output, source DAC). Wired to DIO0_N of the tracking Red Pitaya (RP1), the edge is
streamed as a y_position event (set_marker_input('sync'), data streamer ABI 1.5) with the time
of the RP1 clock (8 ns steps, + 24 ns synchronizer delay, subtracted here). RP2 captures the coil
current on the same trigger, so the applied current is known relative to the edge, and the
tracked frequency relative to the same edge: H(f) needs no software timing. The two boards run
on independent crystals (some 10 ppm apart); measure_clock_ratio() measures the ratio from the
frequency of the tracked response to a long sine burst, and the analysis uses f / (1 + eps) as
the frequency on the RP1 time axis. Measured 2026-11-06: one marker per generator trigger (also
for a burst with several repetitions), none without a trigger.

Wiring: RP2 DIO0_N -> RP1 DIO0_N, plus a ground connection (both 3.3 V LVCMOS, extension
connector E1).

Example config:

    field_response_logic:
        module.Class: 'field_response_logic.FieldResponseLogic'
        connect:
            current_source: current_source_logic
            tracking_hw: redpitaya_odmr_lock     # optional; or set_tracker(source) from a script
        options:
            gamma_mhz_per_mt: 28.025     # NV gyromagnetic ratio
            marker_delay: 24e-9          # s, RP1 synchronizer delay of the marker
            current_reference: 'command' # applied current: 'command' or 'shunt'

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

import numpy as np
from matplotlib.figure import Figure

from qudi.core.module import LogicBase
from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.util.datastorage import TextDataStorage
from qudi.util.sine_fit import fit_sine

CLOCK_HZ = 125e6
N_CAPTURE = 16384       # samples of an RP2 capture
CAPTURE_PERIODS = 5     # periods per current capture: time step = 5 / (f * 16384) (0.11 deg)
CURRENT_SKIP = 50e-6    # s not fitted after the burst start (current L/R = 6 us)
Y_SOURCE = 5            # data streamer source id of y_position
SYNC_FLAG = 0x4         # y_position flag bit: event from the sync marker (mode 'both')


def _now():
    return datetime.datetime.now().isoformat(timespec='seconds')


# =============================================================================================
# analysis (pure functions, unit-tested offline)
# =============================================================================================
def sync_marker_ticks(events, mode='sync'):
    """Ticks of the sync-marker events among recorded stream events."""
    ev = np.asarray(events)
    y = ev[ev['source'] == Y_SOURCE]
    if mode == 'both':
        y = y[(y['flags'] & SYNC_FLAG) != 0]
    return y['tick'].astype(np.int64)


def burst_phasors(ticks, values, marker_ticks, f, duration, skip=0.0, eps=0.0,
                  marker_delay=24e-9, clock_hz=CLOCK_HZ, trend=False):
    """For every marker: the phasor of `values` at f / (1 + eps) in the window
    [skip, duration] after the marker, time referenced to the marker edge (marker tick minus
    marker_delay). ticks, marker_ticks: RP1 clock ticks. NaN samples are ignored.
    trend: fit a linear drift too (the resonance drifts by kHz/s while the microwave warms the
    diamond; without it a drift d biases the sine by ~d / (pi f)).
    Returns (phasors, offsets, residuals, number of samples) arrays."""
    ticks = np.asarray(ticks, np.int64)
    values = np.asarray(values, float)
    f_rp1 = f / (1.0 + eps)
    out = []
    for mt in np.asarray(marker_ticks, np.int64):
        t = (ticks - mt) / clock_hz + marker_delay
        sel = (t >= skip) & (t <= duration) & np.isfinite(values)
        if np.count_nonzero(sel) < 8:
            out.append((np.nan + 0j, np.nan, np.nan, int(np.count_nonzero(sel))))
            continue
        fit = fit_sine(values[sel], f_rp1, t=t[sel], trend=trend)
        out.append((fit.phasor, fit.offset, fit.resid_rms, int(np.count_nonzero(sel))))
    if not out:
        return (np.empty(0, complex),) + tuple(np.empty(0) for _ in range(3))
    p, c, r, n = zip(*out)
    return np.array(p), np.array(c), np.array(r), np.array(n)


def current_phasor(capture, f, duration, skip=0.0):
    """Phasor, offset and rms residual of the current in a capture (time relative to the
    generator start)."""
    t = np.asarray(capture['t'], float)
    i = np.asarray(capture['current'], float)
    sel = (t >= skip) & (t <= duration)
    fit = fit_sine(i[sel], f, t=t[sel])
    return fit.phasor, fit.offset, fit.resid_rms


def clock_ratio(marker_ticks, period, clock_hz=CLOCK_HZ):
    """eps with (marker spacing) = period * (1 + eps): RP1 seconds per RP2 second - 1."""
    d = np.diff(np.asarray(marker_ticks, np.int64)) / clock_hz
    k = np.round(d / period)
    good = k >= 1
    if not np.any(good):
        raise ValueError('no marker intervals')
    return float(np.sum(d[good]) / np.sum(k[good] * period) - 1.0)


def fit_frequency(t, x, f0, rel_span=2e-4):
    """Frequency of a sine near f0 (least squares: the linear phasor fit scanned over f,
    then a parabola through the best three points). For the clock ratio: a linear tracker
    reproduces the generator frequency exactly, on the RP1 time axis f0 / (1 + eps)."""
    t, x = np.asarray(t, float), np.asarray(x, float)
    ok = np.isfinite(x)
    t, x = t[ok], x[ok]

    def resid(f):
        return fit_sine(x, f, t=t).resid_rms

    span = f0 * rel_span
    for _ in range(4):
        fs = np.linspace(f0 - span, f0 + span, 41)
        r = np.array([resid(f) for f in fs])
        k = int(np.clip(np.argmin(r), 1, len(fs) - 2))
        y0, y1, y2 = r[k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        f0 = fs[k] + (0.5 * (y0 - y2) / den * (fs[1] - fs[0]) if den > 0 else 0.0)
        span /= 10
    return float(f0)


def fit_calibration(i, f, t, drift=True):
    """f = k i + f0 (+ drift t), least squares. Returns (k, f0, drift); drift 0 if not fitted."""
    i, f, t = (np.asarray(v, float) for v in (i, f, t))
    cols = [i, np.ones_like(i)] + ([t] if drift else [])
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), f, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2]) if drift else 0.0


def step_traces(grid, traces, high_time, min_rel=0.5):
    """Drop traces without a step: the mean over the second half of the high time minus the mean
    before the edge below min_rel x the median of all traces (e.g. the marker of the
    zero-amplitude trigger that applies the burst settings, 7 Nov 2026: 21 markers for 20
    steps). Returns (kept traces, number dropped)."""
    tr = np.asarray(traces, float)
    pre = grid < 0
    late = (grid > 0.5 * high_time) & (grid <= high_time)
    h = np.nanmean(tr[:, late], axis=1) - np.nanmean(tr[:, pre], axis=1)
    med = np.nanmedian(h)
    keep = np.isfinite(h) & (np.abs(h) >= min_rel * abs(med)) & (np.sign(h) == np.sign(med))
    return tr[keep], int(np.count_nonzero(~keep))


def step_metrics(t, y):
    """Rise time 10-90 %, delay to 50 %, overshoot of an averaged step (t from the step
    edge, y baseline-corrected so that the settled value is reached at the end)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    pre = y[t < 0]
    y0 = float(np.nanmean(pre)) if pre.size else 0.0
    post = y[t >= 0]
    tail = post[-max(5, post.size // 10):]
    y1 = float(np.nanmean(tail))
    h = y1 - y0
    if not np.isfinite(h) or h == 0:
        return dict(step=h, rise_10_90=np.nan, delay_50=np.nan, overshoot=np.nan)
    n = (y - y0) / h
    tp, npost = t[t >= 0], n[t >= 0]

    def first(level):
        k = np.flatnonzero(npost >= level)
        return float(tp[k[0]]) if k.size else np.nan

    return dict(step=h, baseline=y0, rise_10_90=first(0.9) - first(0.1), delay_50=first(0.5),
                overshoot=float(np.nanmax(npost) - 1.0))


# =============================================================================================
# logic
# =============================================================================================
class FieldResponseLogic(LogicBase):
    """Tracking response to coil test fields."""

    _current_source = Connector(name='current_source', interface='CurrentSourceLogic')
    _tracker = Connector(name='tracking_hw', interface='OdmrTrackingInterface', optional=True)

    _gamma = ConfigOption('gamma_mhz_per_mt', default=28.025, missing='nothing')
    _marker_delay = ConfigOption('marker_delay', default=24e-9, missing='nothing')
    _current_reference = ConfigOption('current_reference', default='command', missing='nothing',
                                      checker=lambda x: x in ('command', 'shunt'))

    _nv_cal = StatusVar('nv_calibration', default=dict())     # {slot: dict(mhz_per_a, ...)}
    _eps = StatusVar('clock_ratio_eps', default=0.0)
    _eps_time = StatusVar('clock_ratio_time', default=None)
    _results = StatusVar('last_results', default=dict())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._abort = False
        self._source = None             # tracker set with set_tracker (script-driven trackers)

    def set_tracker(self, source):
        """Use `source` instead of the tracking_hw connector, e.g. a Method D recorder from a
        study script (my_software/method_d_study/md_tracking_source.py). It needs
        tracking_engine_active, stream_sample_rate, open_recording, read_recording,
        close_recording, reconstruct_recording(events) -> dict(ticks, corr_hz[slot], err[slot])
        and set_marker_input. None: back to the connector."""
        self._source = source

    def _tr(self):
        if self._source is not None:
            return self._source
        tr = self._tracker()
        if tr is None:
            raise RuntimeError('no tracker: connect tracking_hw or call set_tracker()')
        return tr

    def on_activate(self):
        pass

    def on_deactivate(self):
        self._abort = True

    @property
    def clock_eps(self):
        return float(self._eps)

    @property
    def nv_calibration(self):
        """{slot: dict(mhz_per_a, mt_per_a_along_nv, time, source)}, slot as a string."""
        return copy.deepcopy(self._nv_cal)

    def set_nv_calibration(self, mhz_per_a, slot=0, source='manual'):
        """Line shift of the tracked resonance `slot` per coil current (MHz/A, signed)."""
        cal = dict(self._nv_cal)
        cal[str(int(slot))] = dict(mhz_per_a=float(mhz_per_a),
                                   mt_per_a_along_nv=abs(float(mhz_per_a)) / self._gamma,
                                   time=_now(), source=str(source))
        self._nv_cal = cal
        self.log.info(f'NV calibration slot {slot}: {mhz_per_a:.4f} MHz/A ({source})')

    @property
    def last_results(self):
        return copy.deepcopy(self._results)

    def stop(self):
        """Ask a running measurement to stop after the current step."""
        self._abort = True

    # --- helpers --------------------------------------------------------------------------
    def _check(self):
        if self._abort:
            raise RuntimeError('stopped')

    def _require_tracking(self):
        tr = self._tr()
        if not tr.tracking_engine_active:
            raise RuntimeError('start the tracking engine first (ODMR Tracking GUI: Start Stream '
                               'or Start Tracking)')
        return tr

    def _record(self, seconds, handle, events):
        tr = self._tr()
        t_end = time.time() + seconds
        while time.time() < t_end:
            self._check()
            time.sleep(0.05)
            events.append(np.asarray(tr.read_recording(handle)['events']))
        events.append(np.asarray(tr.read_recording(handle)['events']))

    def _reconstruct(self, events, slot):
        tr = self._tr()
        ev = np.concatenate([e for e in events if len(e)]) if any(len(e) for e in events) else events[0]
        rec = tr.reconstruct_recording(ev)
        return ev, rec['ticks'].astype(np.int64), np.asarray(rec['corr_hz'])[int(slot)], \
            np.asarray(rec['err'])[int(slot)]

    def _with_sync(self, mode='sync'):
        tr = self._tr()
        tr.set_marker_input(mode)
        self._current_source().set_sync_output(True)

    # --- field calibration -----------------------------------------------------------------
    def calibrate_field(self, currents=(0.0, 0.005, 0.01, 0.02, 0.01, 0.0, -0.005, -0.01, -0.02,
                                        -0.01, 0.0), settle=0.3, dwell=0.5,
                        slot=0, store=True, fit_drift=True):
        """DC current steps (default +-20 mA: +-0.23 MHz at ~11.6 MHz/A, well inside the
        tracking range; amperes would move the line across hyperfine features) -> tracked
        resonance shift per current (MHz/A) from a straight-line fit against the measured DC
        current (shunt; at DC it agrees with the output voltage / R within 1-1.5 %), with a
        linear drift in time (fit_drift; the line drifts by several kHz/s while the microwave
        warms the diamond, 7 Nov 2026). mhz_per_a_command: the same fit against the set
        currents. With store, the result becomes nv_calibration[slot]; mt_per_a_along_nv =
        |MHz/A| / gamma is the field component along the NV axis per ampere."""
        self._abort = False
        coil = self._current_source()
        tr = self._require_tracking()
        handle = tr.open_recording()
        rows = []
        try:
            for i_set in currents:
                self._check()
                coil.set_output('dc', 0.0, 0.0, float(i_set))
                time.sleep(settle)
                tr.read_recording(handle)                 # discard the transition
                events = []
                self._record(dwell, handle, events)
                _, _, corr, err = self._reconstruct(events, slot)
                m = coil.measure(1)
                rows.append(dict(i_set=float(i_set), i_meas=m['i_mean'], t=time.time(),
                                 f_mhz=float(np.nanmean(corr)) / 1e6,
                                 f_std_khz=float(np.nanstd(corr)) / 1e3, n=int(np.isfinite(corr).sum())))
        finally:
            coil.output_off()
            tr.close_recording(handle)
        i = np.array([r['i_meas'] for r in rows])
        f = np.array([r['f_mhz'] for r in rows])
        t = np.array([r['t'] for r in rows])
        k, f0, drift = fit_calibration(i, f, t - t[0], drift=fit_drift)
        k_cmd = fit_calibration(np.array([r['i_set'] for r in rows]), f, t - t[0], drift=fit_drift)[0]
        p = (k, f0)
        resid = f - (k * i + f0 + drift * (t - t[0]))
        res = dict(kind='field_calibration', time=_now(), slot=int(slot), mhz_per_a=float(k),
                   f0_mhz=float(f0), drift_khz_per_s=float(drift * 1e3),
                   resid_khz=(resid * 1e3).tolist(), rows=rows, mhz_per_a_command=float(k_cmd),
                   mt_per_a_along_nv=float(abs(p[0]) / self._gamma))
        if store:
            self.set_nv_calibration(res['mhz_per_a'], slot, source='calibrate_field')
        res['file'] = self._save(res, np.array([[r['i_set'], r['i_meas'], r['f_mhz'], r['f_std_khz']]
                                                for r in rows]),
                                 ['i_set (A)', 'i_meas (A)', 'f (MHz)', 'f std (kHz)'], 'field_calibration',
                                 self._plot_calibration(i, f - drift * (t - t[0]), p))
        self._results = dict(self._results, field_calibration=res)
        return res

    # --- clock ratio -----------------------------------------------------------------------
    def measure_clock_ratio(self, frequency=1e3, duration=1.0, amplitude=None,
                            amplitude_mhz=0.1, slot=0, mode='sync'):
        """Ratio of the two Red Pitaya clocks: one long sine burst; the tracked response has
        exactly the generator frequency, seen on the RP1 clock as f / (1 + eps). (The RP2
        trigger output fires once per trigger, not per burst repetition, so marker spacing
        cannot be used.) Needs the NV calibration if amplitude_mhz is given."""
        self._abort = False
        coil = self._current_source()
        tr = self._require_tracking()
        if amplitude is None:
            amplitude = self.amplitude_for(amplitude_mhz=amplitude_mhz, slot=slot)
        self._with_sync(mode)
        coil.output_off()
        n = int(round(duration * frequency))
        coil.configure_burst(n, 1, None)
        # set_output applies the burst settings with one trigger at zero amplitude (that
        # trigger also emits a marker): open the recording only afterwards
        coil.set_output('sine', frequency, amplitude)
        handle = tr.open_recording()
        events = []
        try:
            tr.read_recording(handle)
            coil.trigger()
            self._record(n / frequency + 0.1, handle, events)
        finally:
            coil.output_off()
            coil.configure_burst(None)
            tr.close_recording(handle)
        ev, ticks, corr, err = self._reconstruct(events, slot)
        mt = sync_marker_ticks(ev, mode)
        if mt.size != 1:
            raise RuntimeError(f'{mt.size} sync markers (expected 1): check wiring and marker input')
        t = (ticks - mt[0]) / CLOCK_HZ + self._marker_delay
        sel = (t > 0.05 * n / frequency) & (t < n / frequency)
        f_fit = fit_frequency(t[sel], corr[sel], frequency)
        eps = frequency / f_fit - 1.0
        self._eps, self._eps_time = eps, _now()
        res = dict(kind='clock_ratio', time=_now(), eps=eps, eps_ppm=eps * 1e6, f_fit=f_fit,
                   frequency=frequency, duration=n / frequency, amplitude=amplitude)
        self._results = dict(self._results, clock_ratio=res)
        self.log.info(f'clock ratio RP1/RP2 - 1 = {eps * 1e6:+.3f} ppm')
        return res

    # --- amplitudes ------------------------------------------------------------------------
    def _k_mhz(self, slot):
        cal = self._nv_cal.get(str(int(slot)))
        if cal is None:
            raise RuntimeError(f'no NV calibration for slot {slot}: run calibrate_field first '
                               f'(or set_nv_calibration)')
        return float(cal['mhz_per_a'])

    def amplitude_for(self, amplitude=None, amplitude_mhz=None, amplitude_ut=None, slot=0):
        """Coil current amplitude (A) from one of: A, MHz of tracked shift, uT along the NV
        axis (both through the NV calibration of the slot)."""
        if amplitude is not None:
            return abs(float(amplitude))
        if amplitude_ut is not None:
            amplitude_mhz = float(amplitude_ut) * 1e-3 * self._gamma
        if amplitude_mhz is None:
            raise ValueError('give amplitude (A), amplitude_mhz or amplitude_ut')
        return abs(float(amplitude_mhz) / self._k_mhz(slot))

    # --- transfer function -----------------------------------------------------------------
    def _tf_point(self, f, amplitude, n_cyc, n_bursts, skip, slot, mode, capture_current,
                  current_reference, k_mhz):
        """One frequency: n_bursts sine bursts of n_cyc periods. Returns (row, data)."""
        coil = self._current_source()
        tr = self._tr()
        dur = n_cyc / f
        a_set = min(amplitude, coil.max_current('sine', f))
        coil.configure_burst(n_cyc, 1, None)
        coil.set_output('sine', f, a_set)
        handle = tr.open_recording()
        events, currents = [], []
        try:
            tr.read_recording(handle)
            for _ in range(int(n_bursts)):
                self._check()
                if capture_current:
                    # fine time step: the trigger position is only known to ~1 sample (with
                    # dur / 16000 per sample: 4.9 deg at 10 kHz, 28 deg at 50 kHz, 8 Nov 2026)
                    dec = int(min(65536, max(1, np.ceil(CAPTURE_PERIODS / f * CLOCK_HZ / N_CAPTURE))))
                    coil.arm_capture(dec, 'generator', 200)
                coil.trigger()
                self._record(dur + 0.05, handle, events)
                if capture_current:
                    currents.append(coil.read_capture(5.0))
        finally:
            tr.close_recording(handle)
            coil.output_off()
        ev, ticks, corr, err = self._reconstruct(events, slot)
        mt = sync_marker_ticks(ev, mode)
        pc, oc, rc, nc = burst_phasors(ticks, corr / 1e6, mt, f, dur, skip, self._eps,
                                       self._marker_delay, trend=True)
        p_cmd = coil.expected_current(f, a_set)
        p_shunt = complex(np.nan, np.nan)
        if capture_current and currents:
            p_shunt = complex(np.mean([current_phasor(c, f, min(dur, c['t'][-1]), CURRENT_SKIP)[0]
                                       for c in currents]))
        if current_reference == 'shunt':
            if not np.isfinite(p_shunt):
                raise RuntimeError("current_reference 'shunt' needs capture_current=True")
            p_i = p_shunt
        else:
            p_i = p_cmd
        p_applied = p_i * k_mhz
        p_tracked = complex(np.nanmean(pc)) if np.isfinite(pc).any() else complex(np.nan, np.nan)
        h = p_tracked / p_applied
        row = dict(f=f, cycles=n_cyc, markers=int(mt.size), bursts=int(n_bursts),
                   amplitude_set=float(a_set), i_amp=float(abs(p_i)),
                   applied_mhz=float(abs(p_applied)),
                   applied_ut=float(abs(p_applied) / self._gamma * 1e3),
                   tracked_mhz=float(abs(p_tracked)),
                   tracked_ut=float(abs(p_tracked) / self._gamma * 1e3),
                   h_abs=float(abs(h)), h_phase_deg=float(np.degrees(np.angle(h))),
                   h_phase_std_deg=float(np.degrees(np.nanstd(np.angle(pc / p_tracked))))
                   if np.isfinite(pc).sum() > 1 else np.nan,
                   shunt_over_cmd=float(abs(p_shunt / p_cmd)),
                   shunt_phase_deg=float(np.degrees(np.angle(p_shunt / p_cmd))),
                   resid_khz=float(np.nanmean(rc) * 1e3), err_rms=float(np.nanstd(err)))
        return row, dict(f=f, ticks=ticks, corr=corr, markers=mt)

    def transfer_function(self, frequencies, amplitude_mhz=None, amplitude=None, cycles=None,
                          duration=0.05, n_bursts=5, skip=0.005, slot=0, mode='sync',
                          capture_current=True, amplitude_ut=None, current_reference=None,
                          save=True):
        """Sine bursts at each frequency; H(f) = phasor of the tracked frequency shift /
        phasor of the applied shift (applied current phasor * NV calibration of the slot),
        both relative to the same generator start.

        amplitude: A, or amplitude_mhz / amplitude_ut through the NV calibration. Each burst
        lasts `duration` (or `cycles` periods); the first `skip` s after the start (tracker
        transient) are not fitted. current_reference: 'command' or 'shunt' (None: config
        option, see the module docstring)."""
        self._abort = False
        coil = self._current_source()
        self._require_tracking()
        k_mhz = self._k_mhz(slot)
        amplitude = self.amplitude_for(amplitude, amplitude_mhz, amplitude_ut, slot)
        cref = current_reference or self._current_reference
        if cref not in ('command', 'shunt'):
            raise ValueError("current_reference must be 'command' or 'shunt'")
        self._with_sync(mode)
        rows, data = [], []
        coil.output_off()
        try:
            for f in np.atleast_1d(frequencies).astype(float):
                self._check()
                n_cyc = int(cycles) if cycles else max(1, int(round(duration * f)))
                row, d = self._tf_point(f, amplitude, n_cyc, n_bursts, skip, slot, mode,
                                        capture_current, cref, k_mhz)
                rows.append(row)
                data.append(d)
                self.log.info(f'H({f:.6g} Hz) = {row["h_abs"]:.4f} at {row["h_phase_deg"]:+.1f} deg '
                              f'({row["markers"]} markers, shunt/command {row["shunt_over_cmd"]:.3f})')
        finally:
            coil.output_off()
            coil.configure_burst(None)
        res = dict(kind='transfer_function', time=_now(), slot=int(slot), eps=self._eps,
                   amplitude=amplitude, current_reference=cref, mhz_per_a=k_mhz, rows=rows)
        keys = ['f', 'h_abs', 'h_phase_deg', 'h_phase_std_deg', 'i_amp', 'applied_mhz', 'applied_ut', 'tracked_mhz', 'tracked_ut',
                'shunt_over_cmd', 'shunt_phase_deg', 'resid_khz', 'err_rms', 'markers']
        if save:
            res['file'] = self._save(res, np.array([[r.get(k, np.nan) for k in keys] for r in rows],
                                                   float), keys, 'transfer_function',
                                     self._plot_tf(rows))
        self._results = dict(self._results, transfer_function=res)
        return res

    def response_map(self, frequencies, amplitudes_ut, **kwargs):
        """transfer_function for each field amplitude (uT along the NV axis): the tracking
        response versus frequency and field strength. kwargs go to transfer_function. Saves one
        table (one row per amplitude and frequency)."""
        self._abort = False
        out, table = [], []
        keys = ['amplitude_ut', 'f', 'h_abs', 'h_phase_deg', 'h_phase_std_deg', 'applied_ut',
                'tracked_ut', 'shunt_over_cmd', 'shunt_phase_deg', 'err_rms']
        for a in amplitudes_ut:
            self._check()
            r = self.transfer_function(frequencies, amplitude_ut=float(a), save=False, **kwargs)
            for row in r['rows']:
                row['amplitude_ut'] = float(a)
                table.append([row.get(k, np.nan) for k in keys])
            out.append(r)
        res = dict(kind='response_map', time=_now(), amplitudes_ut=[float(a) for a in amplitudes_ut],
                   frequencies=[float(f) for f in np.atleast_1d(frequencies)],
                   current_reference=out[0]['current_reference'] if out else None,
                   eps=self._eps, maps=out)
        res['file'] = self._save(res, np.array(table, float), keys, 'response_map',
                                 self._plot_map(out))
        self._results = dict(self._results, response_map=res)
        return res

    # --- step response ---------------------------------------------------------------------
    def step_response(self, amplitude=None, amplitude_mhz=None, high_time=0.02, n_steps=20,
                      pre=0.002, slot=0, mode='sync', amplitude_ut=None):
        """Current steps 0 -> amplitude -> 0 (arbitrary waveform, one burst per trigger);
        the tracked frequency averaged over n_steps, aligned to the generator start.
        amplitude in A, or amplitude_mhz / amplitude_ut through the NV calibration. The
        step height in the result (step) is the NV-measured one; expected_mhz is the command
        times the NV calibration. The current itself rises with L/R = 6 us; that is part of the
        result."""
        self._abort = False
        coil = self._current_source()
        tr = self._require_tracking()
        k_mhz = float(self._nv_cal.get(str(int(slot)), {}).get('mhz_per_a', np.nan))
        amplitude = self.amplitude_for(amplitude, amplitude_mhz, amplitude_ut, slot)
        self._with_sync(mode)
        coil.output_off()
        table = np.r_[np.ones(500), np.zeros(500)]
        coil.set_arbitrary_waveform(table)
        f = 1.0 / (2 * high_time)
        coil.configure_burst(1, 1, None)
        events = []
        try:
            coil.set_output('arbitrary', f, amplitude)
            handle = tr.open_recording()
            try:
                tr.read_recording(handle)
                for _ in range(int(n_steps)):
                    self._check()
                    coil.trigger()
                    self._record(2 * high_time + 0.05, handle, events)
            finally:
                tr.close_recording(handle)
        finally:
            coil.output_off()
            coil.configure_burst(None)
        ev, ticks, corr, err = self._reconstruct(events, slot)
        mt = sync_marker_ticks(ev, mode)
        rate = self._tr().stream_sample_rate
        grid = np.arange(-pre, 2 * high_time, 1.0 / rate)
        traces = []
        for m in mt:
            t = (ticks - m) / CLOCK_HZ + self._marker_delay
            sel = (t >= grid[0] - 2 / rate) & (t <= grid[-1] + 2 / rate)
            if np.count_nonzero(sel) > 10:
                traces.append(np.interp(grid, t[sel], corr[sel], left=np.nan, right=np.nan))
        if not traces:
            raise RuntimeError('no sync markers recorded: check the wiring and set_marker_input')
        traces, rejected = step_traces(grid, traces, high_time)
        if rejected:
            self.log.info(f'step_response: {rejected} marker(s) without a step ignored')
        avg = np.nanmean(traces, axis=0)
        up = grid <= high_time
        met = step_metrics(grid[up], avg[up])
        res = dict(kind='step_response', time=_now(), slot=int(slot), amplitude=amplitude,
                   expected_mhz=amplitude * k_mhz if np.isfinite(k_mhz) else None,
                   steps=len(traces), rejected=int(rejected),
                   **{k: float(v) for k, v in met.items()})
        res['file'] = self._save(res, np.column_stack([grid, avg]), ['t (s)', 'tracked frequency (Hz)'],
                                 'step_response', self._plot_step(grid, np.array(traces), avg, res))
        self._results = dict(self._results, step_response=res)
        return res

    # --- large signal ----------------------------------------------------------------------
    def amplitude_sweep(self, frequency, amplitudes, **kwargs):
        """transfer_function at one frequency for each amplitude (A): H and the error rms
        versus amplitude; a collapse of |H| or a jump of the error marks the large-signal
        limit of the tracking."""
        rows = []
        for a in amplitudes:
            r = self.transfer_function([frequency], amplitude=float(a), **kwargs)['rows'][0]
            r['amplitude'] = float(a)
            rows.append(r)
        res = dict(kind='amplitude_sweep', time=_now(), frequency=frequency, rows=rows)
        self._results = dict(self._results, amplitude_sweep=res)
        return res

    # --- saving and plots ------------------------------------------------------------------
    def _save(self, meta, data, headers, tag, fig=None):
        storage = TextDataStorage(root_dir=self.module_default_data_dir)
        meta = {k: v for k, v in meta.items() if k not in ('rows',)}
        path, _, _ = storage.save_data(np.asarray(data, float), metadata=meta, nametag=tag,
                                       column_headers=headers)
        if fig is not None:
            storage.save_thumbnail(fig, path.rsplit('.', 1)[0])
        self.log.info(f'{tag} saved to {path}')
        return path

    @staticmethod
    def _plot_calibration(i, f, p):
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.plot(i, f, 'o')
        x = np.linspace(min(i), max(i), 10)
        ax.plot(x, np.polyval(p, x), 'k-', lw=0.8)
        ax.set_xlabel('coil current (A)')
        ax.set_ylabel('tracked resonance (MHz)')
        ax.set_title(f'{p[0]:.4f} MHz/A')
        fig.tight_layout()
        return fig

    @staticmethod
    def _plot_tf(rows):
        f = np.array([r['f'] for r in rows])
        fig = Figure(figsize=(7, 6))
        a1, a2 = fig.add_subplot(211), fig.add_subplot(212)
        a1.loglog(f, [r['h_abs'] for r in rows], 'o-')
        a2.semilogx(f, [r['h_phase_deg'] for r in rows], 'o-')
        a1.set_ylabel('|H|')
        a2.set_ylabel('phase (deg)')
        a2.set_xlabel('field frequency (Hz)')
        for a in (a1, a2):
            a.grid(alpha=0.3, which='both')
        fig.tight_layout()
        return fig

    @staticmethod
    def _plot_map(maps):
        fig = Figure(figsize=(7, 6))
        a1, a2 = fig.add_subplot(211), fig.add_subplot(212)
        for m in maps:
            rows = m['rows']
            if not rows:
                continue
            f = [r['f'] for r in rows]
            lab = f"{rows[0].get('amplitude_ut', np.nan):.3g} uT"
            a1.semilogx(f, [r['h_abs'] for r in rows], 'o-', label=lab)
            a2.semilogx(f, [r['h_phase_deg'] for r in rows], 'o-')
        a1.set_ylabel('|H|')
        a2.set_ylabel('phase (deg)')
        a2.set_xlabel('field frequency (Hz)')
        a1.legend(fontsize=7)
        for a in (a1, a2):
            a.grid(alpha=0.3, which='both')
        fig.tight_layout()
        return fig

    @staticmethod
    def _plot_step(grid, traces, avg, res):
        fig = Figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        for tr in traces[:20]:
            ax.plot(grid * 1e3, tr / 1e6, color='0.8', lw=0.5)
        ax.plot(grid * 1e3, avg / 1e6, lw=1.2)
        ax.set_xlabel('time after the current step (ms)')
        ax.set_ylabel('tracked frequency (MHz)')
        ax.set_title(f"rise 10-90 % {res['rise_10_90'] * 1e6:.1f} us, delay {res['delay_50'] * 1e6:.1f} us, "
                     f"overshoot {res['overshoot'] * 100:.1f} %", fontsize=9)
        fig.tight_layout()
        return fig
