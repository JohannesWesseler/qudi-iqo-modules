# -*- coding: utf-8 -*-
"""
Multi-Resonance ODMR Frequency Tracking Logic.

Generalises OdmrFrequencyTrackingLogic from one resonance to N (currently two)
resonances tracked simultaneously by hardware-driven LO hopping on a single Red
Pitaya + single Windfreak LO + single IQ mixer:

  - one wide ODMR scan over the whole band, then the user selects two frequency
    windows (left/right outermost features); each is linear-fit for its centre
    zero-crossing + discriminator slope (exactly the single-resonance flow, x2);
  - a constant-IF, 2-point LO JUMP_LIST is built from the two zero-crossings, with
    a per-LO SSB calibration loaded into each FPGA cal slot;
  - the FPGA hops the LO indefinitely (continuous scan loop), freezing/resuming
    each resonance's demod chain + per-slot integrator in lockstep with the hop;
  - the two per-resonance frequency corrections are monitored live (per-slot
    integrator registers) and, optionally, as high-rate reconstructed traces.

Copyright (c) 2021, the qudi developers. See the AUTHORS.md file at the top-level
directory of this distribution and on <https://github.com/Ulm-IQO/qudi-core/>

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with
qudi. If not, see <https://www.gnu.org/licenses/>.
"""

__all__ = ['MultiResonanceOdmrTrackingLogic']

import numpy as np
import os
import json
import time
from collections import deque
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from PySide2 import QtCore
from qudi.core.connector import Connector
from qudi.core.statusvariable import StatusVar
from qudi.core.configoption import ConfigOption
from qudi.logic.odmr_frequency_tracking_logic import OdmrFrequencyTrackingLogic
from qudi.util.datastorage import NpyDataStorage, TextDataStorage


class MultiResonanceOdmrTrackingLogic(OdmrFrequencyTrackingLogic):
    """
    Two-resonance (extensible to N) ODMR frequency-tracking orchestration.

    Inherits the single-resonance ODMR scanning + linear fit + lock infrastructure
    and adds the multi-resonance hop configuration / monitoring.

    Example config:

        multi_resonance_odmr_tracking_logic:
            module.Class: 'multi_resonance_odmr_tracking_logic.MultiResonanceOdmrTrackingLogic'
            options:
                num_resonances: 2
                modulation_frequency: 15258.789   # f_m [Hz]
                demod_phase_deg: 123.0            # bench-tuned after osc enable
                osc_settle_time: 200e-6           # freeze window per hop [s]
                dwell_time: 1.0e-3                # per-resonance live time [s]
                scan_settling_time: 100e-6        # LO-settle before dwell [s]
                trigger_length: 50e-6             # LO-hop trigger pulse [s]
                tracking_power: 13                # dBm -> IF amplitude
                stream_traces: True               # legacy option; GUI controls stream explicitly
                status_poll_interval: 0.2         # s
            connect:
                microwave: 'mw_source_synthnv'
                data_scanner: 'redpitaya_finite_sampling'
                odmr_lock_hw: 'redpitaya_odmr_lock'
                multi_track_hw: 'redpitaya_odmr_lock'
                time_series_logic: 'time_series_reader_logic'
    """

    # ---- additional connector: the multi-resonance hardware surface ----
    _multi_track_hw = Connector(name='multi_track_hw',
                                interface='MultiResonanceTrackingInterface')

    # ---- config options ----
    _num_resonances = ConfigOption('num_resonances', default=2, missing='info')
    _modulation_frequency = ConfigOption('modulation_frequency', default=15258.789, missing='info')
    _osc_settle_time = ConfigOption('osc_settle_time', default=200e-6, missing='info')
    _dwell_time = ConfigOption('dwell_time', default=1.0e-3, missing='info')
    _scan_settling_time = ConfigOption('scan_settling_time', default=100e-6, missing='info')
    _trigger_length = ConfigOption('trigger_length', default=50e-6, missing='info')
    _tracking_power = ConfigOption('tracking_power', default=13.0, missing='info')
    # Kept so existing configurations still load. Streaming is now controlled
    # explicitly with start_multi_streaming()/stop_multi_streaming().
    _stream_traces = ConfigOption('stream_traces', default=True, missing='info')
    _multi_status_poll_interval = ConfigOption('multi_status_poll_interval', default=0.2, missing='nothing')
    _trace_history_points = ConfigOption('trace_history_points', default=100000, missing='nothing')

    # ---- status variables ----
    _demod_phase_deg = StatusVar('demod_phase_deg', default=0.0)
    _fit_ranges = StatusVar('fit_ranges', default=[(2.86e9, 2.88e9), (2.90e9, 2.92e9)])
    _max_correction_hz_sv = StatusVar('max_correction_hz', default=1.0e6)

    # ---- signals ----
    sigResonanceFitCompleted = QtCore.Signal(int, dict)   # (index, fit_result)
    sigMultiTrackingStateChanged = QtCore.Signal(bool)    # tracking active
    sigMultiStreamStateChanged = QtCore.Signal(bool)      # high-rate stream active
    sigMultiConfigurationCompleted = QtCore.Signal(bool, str)  # success, error
    sigSlotStatusUpdated = QtCore.Signal(list)            # [per-slot status dicts]
    sigCorrectionHistoryUpdated = QtCore.Signal(object, object)  # times, (N x T) Hz
    sigErrorHistoryUpdated = QtCore.Signal(object, object)       # times, (N x T) LSB
    sigResonanceTracesUpdated = QtCore.Signal(object, object)    # times, (N x T) Hz (high-rate, legacy)
    # High-rate synchronous stream: times, post-FIR error, correction, pre-FIR CIC.
    # All arrays share the same FPGA sample axis and are decimated together for display.
    sigHighRateTracesUpdated = QtCore.Signal(object, object, object, object)
    sigNumResonancesChanged = QtCore.Signal(int)

    # Max points per curve pushed to the GUI for the high-rate view (decimation cap).
    _HIGH_RATE_MAX_POINTS = 4000

    # ---- continuous per-visit field-trace logging ----
    # Samples retained (never emitted) at the trailing edge of each drain so a
    # hop-visit straddling the drain boundary is finalized whole on the next drain.
    # One visit is ~dwell*rate (~30 samples at 1 ms / 30.5 kHz); 512 samples (~17 ms)
    # is many cycles of headroom while costing a negligible tail buffer.
    _FIELD_GUARD_SAMPLES = 512
    # On-disk record: one row per resonance visit. Same quantity the KDC_HW_SYNC_MULTIRES
    # 2D scan bins (fresh-only corr_hz / err), just per-visit in TIME instead of per-bin
    # in space, reduced with the identical np.nanmean so the two are directly comparable.
    #   t_s     : elapsed seconds since the first logged sample (stream-sample clock)
    #   slot    : resonance index (0/1)
    #   corr_hz : np.nanmean of this visit's fresh correction samples [Hz]
    #   err_lsb : np.nanmean of this visit's fresh demod-error samples [LSB]
    #   n       : number of fresh samples averaged (weight for exact bin-equivalent means)
    _FIELD_DTYPE = np.dtype([('t_s', '<f8'), ('slot', '<i1'),
                             ('corr_hz', '<f4'), ('err_lsb', '<f4'), ('n', '<u2')])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._multi_tracking_active = False
        self._multi_stream_active = False
        # Hardware hopping/acquisition engine. It remains active while either the
        # FPGA lock or the independent trace stream needs it.
        self._multi_engine_active = False
        # per-resonance fit results / targets
        self._resonance_fits: List[Optional[Dict]] = [None, None]
        self._resonance_freqs: List[Optional[float]] = [None, None]
        self._resonance_slopes: List[Optional[float]] = [None, None]
        self._lo_frequencies: List[Optional[float]] = [None, None]
        # low-rate per-slot history (correction + error) from register polling
        self._hist_t0 = None
        self._hist_times = None
        self._hist_corr = None
        self._hist_err = None
        self._tracking_started_at = None
        self._stream_started_at = None
        # Last undecimated high-rate rolling window.  Keeping this in the logic
        # makes it possible to save after tracking has been stopped.
        self._high_rate_times = None
        self._high_rate_err = None
        self._high_rate_corr = None
        self._high_rate_cic = None
        self._high_rate_sample_rate = None
        # status poll timer: created in on_activate (on the logic thread) so it can
        # be started/stopped from the logic thread (a timer created here in __init__
        # would have main-thread affinity -> "Timers cannot be started from another
        # thread" -> never fires -> no values shown).
        self._multi_status_timer = None

        # continuous per-visit field-trace logging (driven by the status poll, which
        # already runs on the logic thread). start/stop only flip request flags so
        # they are safe to call from any thread; the actual drain / file I/O / drain-
        # ownership handoff all happen inside _service_field_logging on the poll.
        self._field_log_open = False
        self._field_log_path = None
        self._field_log_start_request = False
        self._field_log_stop_request = False
        self._field_file = None
        self._field_tail_words = np.empty(0, dtype=np.float64)
        self._field_sample_base = 0        # absolute sample index of tail_words[0]
        self._field_rate = None            # stream sample rate [Hz]
        self._field_words_per_sample = 3   # overwritten from FPGA stream metadata
        self._field_start_wall = None      # wall-clock at the first logged sample
        self._field_written = 0            # visit rows written
        self._field_loss_words = 0         # NaN (transport-loss) words seen
        self._field_total_words = 0

    # =========================================================================
    # Activation
    # =========================================================================
    def on_activate(self):
        super().on_activate()
        # create the poll timer here (runs on the logic thread) so start/stop work
        self._multi_status_timer = QtCore.QTimer(parent=self)
        self._multi_status_timer.setSingleShot(False)
        self._multi_status_timer.timeout.connect(self._poll_multi_status)
        n = int(self._num_resonances)
        self._ensure_resonance_capacity(n)
        self._reset_history()
        self.log.info(f'Multi-Resonance ODMR Tracking Logic activated (N={n})')

    def on_deactivate(self):
        if self._field_log_open:
            try:
                self._field_log_stop_request = True
                self._service_field_logging()
            except Exception as e:
                self.log.warning(f'Error closing field log on deactivate: {e}')
        if self._multi_stream_active:
            try:
                self.stop_multi_streaming()
            except Exception as e:
                self.log.warning(f'Error stopping multi-stream on deactivate: {e}')
        if self._multi_tracking_active:
            try:
                self.stop_multi_tracking()
            except Exception as e:
                self.log.warning(f'Error stopping multi-tracking on deactivate: {e}')
        if self._multi_engine_active:
            try:
                self._stop_engine()
            except Exception as e:
                self.log.warning(f'Error stopping residual tracking engine: {e}')
        if self._multi_status_timer is not None:
            self._multi_status_timer.stop()
        super().on_deactivate()

    # =========================================================================
    # ODMR scan preparation: also tear down hopping before a wide scan
    # =========================================================================
    def _prepare_for_odmr_scan(self):
        if self._multi_tracking_active or self._multi_stream_active:
            self.log.info('Stopping multi-resonance tracking/streaming for ODMR scan')
            try:
                if self._multi_stream_active:
                    self.stop_multi_streaming()
                if self._multi_tracking_active:
                    self.stop_multi_tracking()
            except Exception as e:
                self.log.warning(f'Could not stop multi-tracking/stream before scan: {e}')
        super()._prepare_for_odmr_scan()

    # =========================================================================
    # Per-resonance fitting (manual two-region select; the single-res flow x2)
    # =========================================================================
    def set_scan_region(self, freq_min: float, freq_max: float,
                        points: Optional[int] = None):
        """Set the ODMR scan to a SINGLE range [freq_min, freq_max] for a fast
        EQUIDISTANT sweep of ONE resonance's detail scan.

        Scan each resonance's region in turn (fast sweep) and fit each; both fits are
        remembered independently. We deliberately do NOT use the multi-range path:
        with >1 range the ODMR logic switches the scanner to JUMP_LIST, which on the
        Windfreak means programming the hop table one entry at a time over serial
        (~20 ms each) into a table capped at ~500 points -- far too slow and too
        small for a high-resolution ODMR sweep. The 2-point tracking hop still uses
        JUMP_LIST (only 2 entries), which is fine.
        """
        if points is None:
            try:
                points = int(self.frequency_ranges[0][2])
            except Exception:
                points = 1000
        self.set_frequency_range_count(1)
        self.set_frequency_range(float(freq_min), float(freq_max), int(points), 0)
        self.log.info(
            f'Scan range set to [{freq_min/1e9:.5f}, {freq_max/1e9:.5f}] GHz, '
            f'{points} pts (single-range EQUIDISTANT sweep).')

    def _gather_fit_data(self, index: int, freq_min: Optional[float],
                         freq_max: Optional[float]):
        """Return (freq, signal) for resonance ``index``.

        Uses scan RANGE ``index`` if it exists (the detail-scan case: resonance i =
        range i); otherwise falls back to all ranges concatenated (the single wide
        survey-scan case). Masked to [freq_min, freq_max] when given.
        """
        channel_names = list(self.signal_data.keys())
        if not channel_names:
            raise ValueError('No ODMR scan data available. Run a scan first.')
        ch = channel_names[0]
        freqs = self.frequency_data            # list, one array per range
        sigs = self.signal_data[ch]            # list, one array per range
        if index < len(freqs):
            f = np.asarray(freqs[index], dtype=float)
            s = np.asarray(sigs[index], dtype=float)
        else:
            f = np.concatenate([np.asarray(x, dtype=float) for x in freqs])
            s = np.concatenate([np.asarray(x, dtype=float) for x in sigs])
        if freq_min is not None and freq_max is not None:
            mask = (f >= freq_min) & (f <= freq_max)
            f, s = f[mask], s[mask]
        return f, s

    def fit_resonance_n(self, index: int, freq_min: Optional[float] = None,
                        freq_max: Optional[float] = None) -> Dict:
        """Linear-fit resonance ``index`` and store its centre zero-crossing + slope.

        Range-aware: fits scan RANGE ``index`` (detail-scan case), optionally narrowed
        to the linear window [freq_min, freq_max]; falls back to a window over the
        concatenated survey scan when only one range exists.
        """
        n = int(self._num_resonances)
        if not (0 <= index < n):
            raise ValueError(f'resonance index must be in 0..{n-1}, got {index}')

        f, s = self._gather_fit_data(index, freq_min, freq_max)
        if f.size < 2:
            raise ValueError(f'Resonance {index}: not enough points to fit '
                             f'(got {f.size}). Check the range/window.')

        slope, offset = np.polyfit(f, s, deg=1)
        fit_data = slope * f + offset
        ss_res = np.sum((s - fit_data) ** 2)
        ss_tot = np.sum((s - s.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        if abs(slope) <= 1e-12:
            raise ValueError(f'Resonance {index}: fit slope ~0, no zero-crossing.')
        zc = -offset / slope
        if not (f.min() <= zc <= f.max()):
            self.log.warning(
                f'Resonance {index}: zero-crossing {zc/1e9:.6f} GHz outside the '
                f'fitted data [{f.min()/1e9:.6f}, {f.max()/1e9:.6f}] GHz '
                f'(poor fit / wrong window?).')

        result = {'slope': slope, 'offset': offset, 'zero_crossing_freq': zc,
                  'freq_min': float(f.min()), 'freq_max': float(f.max()),
                  'fit_frequency': f, 'fit_data': fit_data, 'r_squared': r_squared,
                  'timestamp': datetime.now()}

        self._resonance_fits[index] = result
        self._resonance_freqs[index] = float(zc)
        self._resonance_slopes[index] = abs(float(slope))
        ranges = list(self._fit_ranges)
        while len(ranges) <= index:
            ranges.append((float(f.min()), float(f.max())))
        ranges[index] = (float(f.min()), float(f.max()))
        self._fit_ranges = ranges

        self.sigResonanceFitCompleted.emit(index, dict(result))
        self.log.info(
            f'Resonance {index} fit: zero-crossing={zc/1e9:.6f} GHz, '
            f'slope={self._resonance_slopes[index]:.3e} LSB/Hz, R²={r_squared:.4f}')
        return result

    @property
    def resonance_frequencies(self) -> List[Optional[float]]:
        """The tracked centre (zero-crossing) frequencies [Hz], one per resonance."""
        return list(self._resonance_freqs[:int(self._num_resonances)])

    @property
    def lo_frequencies(self) -> List[Optional[float]]:
        """The LO frequencies [Hz] programmed for each resonance (after configure)."""
        return list(self._lo_frequencies[:int(self._num_resonances)])

    @property
    def multi_tracking_active(self) -> bool:
        return self._multi_tracking_active

    @property
    def multi_stream_active(self) -> bool:
        return self._multi_stream_active

    @property
    def num_resonances(self) -> int:
        return int(self._num_resonances)

    def _ensure_resonance_capacity(self, n: int) -> None:
        n = max(1, min(int(n), 2))
        for attr in ('_resonance_fits', '_resonance_freqs',
                     '_resonance_slopes', '_lo_frequencies'):
            values = list(getattr(self, attr, []))
            if len(values) < n:
                values.extend([None] * (n - len(values)))
            setattr(self, attr, values)

    @QtCore.Slot(int)
    def set_num_resonances(self, n: int) -> None:
        """Select active resonance count for the unified tracking GUI."""
        n = max(1, min(int(n), 2))
        if self._multi_engine_active:
            raise RuntimeError('Cannot change resonance count while tracking or streaming is active.')
        if int(self._num_resonances) == n:
            self._ensure_resonance_capacity(n)
            return
        self._num_resonances = n
        self._ensure_resonance_capacity(n)
        self._reset_history()
        self.sigNumResonancesChanged.emit(n)
        self.log.info(f'Active ODMR tracking resonances set to N={n}')

    # =========================================================================
    # Configuration: LO jump-list + per-slot cal + oscillator + integrators
    # =========================================================================
    def configure_multi_tracking(self):
        """Configure all hardware for two-resonance hopping from the two fits.

        Order: set the global IF triplet + FM + slot-0 cal for resonance 0 (set_cw),
        load slot-1 cal for resonance 1, make the cal slot follow the hop, program
        the 2-point LO jump-list (RF->LO, sideband-aware) and arm it, then configure
        the oscillator + per-slot integrator gains. Does NOT enable the loop.
        """
        if self._multi_engine_active:
            raise RuntimeError('Stop tracking and streaming before reconfiguring.')
        n = int(self._num_resonances)
        if any(self._resonance_freqs[i] is None for i in range(n)):
            raise ValueError('Fit all resonances (fit_resonance_n) before configuring tracking.')

        mw = self._microwave()
        lock_hw = self._multi_track_hw()
        rf_freqs = [float(self._resonance_freqs[i]) for i in range(n)]
        # Use the SAME power the (working) detail scans used, so the per-component IF
        # amplitude stays inside the calibrated range. Fall back to the configured
        # tracking_power only if no scan power has been set.
        power = float(self._scan_power)
        if not np.isfinite(power):
            power = float(self._cw_power)
        if not np.isfinite(power):
            power = float(self._tracking_power)
        self.log.info(f'Multi-tracking using power {power} dBm (from scan/cw settings)')

        # 1. error-signal polarity from the sideband (LSB -> invert)
        try:
            sideband = str(getattr(mw, 'sideband', 'upper')).lower()
        except Exception:
            sideband = 'upper'
        lock_hw.set_invert(sideband in ('lower', 'lsb'))

        # 2. global IF triplet + FM + slot-0 cal for resonance 0
        mw.set_cw(rf_freqs[0], power)

        if n == 1:
            if hasattr(mw, 'set_cal_slot_source'):
                mw.set_cal_slot_source(hardware=False)
            lock_hw.set_integrator_source('sw')
            try:
                lo0 = float(mw._rf_to_lo_frequency(rf_freqs[0]))
            except Exception:
                lo0 = None
            self._ensure_resonance_capacity(1)
            self._lo_frequencies[0] = lo0
            osc_source = 'sw'
        else:
            # 3. per-slot cal for the remaining resonances
            for i in range(1, n):
                mw.calibration_load_slot(i, rf_freqs[i], power=power)

            # 4. cal slot + integrator follow the live hop index
            mw.set_cal_slot_source(hardware=True)
            lock_hw.set_integrator_source('current_step')

            # 5. LO jump-list (constant IF, hop only the LO), c1 wrap for indefinite cycling
            hop_rate = 1.0 / float(self._dwell_time)
            self._lo_frequencies = mw.configure_jump_list(
                rf_freqs, hop_rate, continuous=True, enable_output=True)
            osc_source = 'current_step'

        # 6. oscillator + freeze window + global loop gains
        lock_hw.configure_oscillator(
            f_m_hz=float(self._modulation_frequency),
            demod_phase_deg=float(self._demod_phase_deg),
            settle_time_s=float(self._osc_settle_time),
            source=osc_source)
        slopes = [self._resonance_slopes[i] for i in range(n) if self._resonance_slopes[i]]
        rep_slope = float(np.mean(slopes)) if slopes else 1.1
        lock_hw.set_bandwidth(float(self._lock_bandwidth), rep_slope)
        lock_hw.set_max_correction_hz(float(self._max_correction_hz_sv))

        self.log.info(
            f'Multi-tracking configured: RF={[round(f/1e9, 6) for f in rf_freqs]} GHz, '
            # Only format active slots. In N=1 the capacity list intentionally
            # retains an unused second entry (None), which must not turn a fully
            # successful hardware configuration into a GUI-visible failure.
            f'LO={[round(f/1e9, 6) for f in self._lo_frequencies[:n]]} GHz, '
            f'f_m={self._modulation_frequency/1e3:.4f} kHz, dwell={self._dwell_time*1e6:.0f} us, '
            f'BW={self._lock_bandwidth} Hz (rep slope {rep_slope:.3e} LSB/Hz)')

    @QtCore.Slot(dict)
    def configure_multi_tracking_from_params(self, params: Dict) -> None:
        """Apply GUI parameters and report configuration success explicitly.

        This slot belongs to the logic object, so a queued GUI connection executes
        all hardware access on the logic thread.
        """
        try:
            self.set_num_resonances(int(params.get('num_resonances', 2)))
            self.lock_bandwidth = float(params['bandwidth'])
            self.set_modulation_frequency(float(params['modulation_frequency']))
            self.set_dwell_time(float(params['dwell_time']))
            self.set_settle_time(float(params['settle_time']))
            self.set_max_correction_hz(float(params['max_correction_hz']))
            self.set_demod_phase(float(params['demod_phase_deg']))
            self.configure_multi_tracking()
        except Exception as e:
            message = str(e)
            self.log.error('Failed to configure multi-resonance tracking: %s',
                           message, exc_info=True)
            self.sigMultiConfigurationCompleted.emit(False, message)
        else:
            self.sigMultiConfigurationCompleted.emit(True, '')

    # =========================================================================
    # Independent hopping/lock/stream lifecycle
    # =========================================================================
    def _require_tracking_configuration(self) -> int:
        n = int(self._num_resonances)
        if any(self._lo_frequencies[i] is None for i in range(n)):
            raise RuntimeError(
                'Tracking is not configured. Run Configure Tracking successfully first.')
        return n

    def _start_engine(self, stream_traces: bool) -> None:
        """Start oscillator + hopping engine with the FPGA lock still disabled."""
        if self._multi_engine_active:
            if stream_traces and not self._multi_stream_active:
                self._multi_track_hw().start_trace_stream()
            return

        n = self._require_tracking_configuration()
        lock_hw = self._multi_track_hw()
        lock_hw.enable_oscillator(True)
        try:
            lock_hw.enable_lock(False)
            lock_hw.clear_integrators()
            lock_hw.start_tracking(
                nslots=n, dwell_time_s=float(self._dwell_time),
                settling_time_s=float(self._scan_settling_time),
                trigger_length_s=float(self._trigger_length),
                stream_traces=bool(stream_traces))
        except Exception:
            try:
                lock_hw.stop_tracking()
                lock_hw.enable_lock(False)
                lock_hw.enable_oscillator(False)
            except Exception:
                pass
            raise
        self._multi_engine_active = True
        self._reset_history(tracking_started=True)

    def _stop_engine(self) -> None:
        """Stop hopping/oscillator after both lock and stream have released it."""
        if not self._multi_engine_active:
            return
        lock_hw = self._multi_track_hw()
        try:
            lock_hw.stop_tracking()
        finally:
            lock_hw.enable_oscillator(False)
            self._multi_engine_active = False

    def _update_status_timer(self) -> None:
        if self._multi_status_timer is None:
            return
        if self._multi_engine_active:
            if not self._multi_status_timer.isActive():
                self._multi_status_timer.start(
                    int(self._multi_status_poll_interval * 1000))
        else:
            self._multi_status_timer.stop()

    def start_multi_tracking(self):
        """Enable the FPGA lock; trace streaming is left unchanged."""
        if self._multi_tracking_active:
            self.log.warning('Multi-resonance tracking already active')
            return
        lock_hw = self._multi_track_hw()
        engine_was_active = self._multi_engine_active
        try:
            # Starting tracking alone deliberately creates an unstreamed engine.
            # If an open-loop stream is already active, its hopping engine is reused.
            self._start_engine(stream_traces=False)
            lock_hw.clear_integrators()
            lock_hw.enable_lock(True)
            self._multi_tracking_active = True
            self._update_status_timer()
            self.sigMultiTrackingStateChanged.emit(True)
            self.log.info('Multi-resonance FPGA lock enabled (stream unchanged: %s)',
                          self._multi_stream_active)
        except Exception as e:
            self.log.error(f'Failed to start multi-resonance tracking: {e}', exc_info=True)
            try:
                lock_hw.enable_lock(False)
                if not engine_was_active:
                    self._stop_engine()
            except Exception:
                pass
            self._multi_tracking_active = False
            self._update_status_timer()
            self.sigMultiTrackingStateChanged.emit(False)
            raise

    def stop_multi_tracking(self):
        """Disable the FPGA lock; trace streaming is left unchanged."""
        if not self._multi_tracking_active:
            return
        lock_hw = self._multi_track_hw()
        try:
            lock_hw.enable_lock(False)
        except Exception as e:
            self.log.warning(f'Error during multi-tracking stop: {e}')
        finally:
            self._multi_tracking_active = False
            if not self._multi_stream_active:
                try:
                    self._stop_engine()
                except Exception as e:
                    self.log.warning(f'Error stopping idle hopping engine: {e}')
            self._update_status_timer()
            self.sigMultiTrackingStateChanged.emit(False)
            self.log.info('Multi-resonance FPGA lock disabled (stream unchanged: %s)',
                          self._multi_stream_active)

    def start_multi_streaming(self):
        """Start simultaneous demod-error/correction traces without enabling lock."""
        if self._multi_stream_active:
            self.log.warning('Multi-resonance trace stream already active')
            return
        engine_was_active = self._multi_engine_active
        try:
            self._start_engine(stream_traces=True)
            # A newly started stream has its own elapsed-time origin and replaces
            # any cached window from an earlier stream session.
            if engine_was_active:
                self._high_rate_times = None
                self._high_rate_err = None
                self._high_rate_corr = None
                self._high_rate_cic = None
                self._high_rate_sample_rate = None
            self._multi_stream_active = True
            self._stream_started_at = datetime.now()
            self._update_status_timer()
            self.sigMultiStreamStateChanged.emit(True)
            self.log.info('Multi-resonance trace stream started (FPGA lock: %s)',
                          self._multi_tracking_active)
        except Exception as e:
            self.log.error(f'Failed to start multi-resonance trace stream: {e}', exc_info=True)
            if not engine_was_active:
                try:
                    self._stop_engine()
                except Exception:
                    pass
            self._multi_stream_active = False
            self._update_status_timer()
            self.sigMultiStreamStateChanged.emit(False)
            raise

    def stop_multi_streaming(self):
        """Stop traces without changing the FPGA lock state."""
        if not self._multi_stream_active:
            return
        try:
            self._multi_track_hw().stop_trace_stream()
        except Exception as e:
            self.log.warning(f'Error stopping multi-resonance trace stream: {e}')
            # In particular, a mapped motor scan deliberately rejects this operation.
            # Keep the logic/GUI state truthful instead of displaying a stopped stream
            # while the hardware and motor scan are still consuming it.
            self.sigMultiStreamStateChanged.emit(True)
            return

        self._multi_stream_active = False
        if not self._multi_tracking_active:
            try:
                self._stop_engine()
            except Exception as e:
                self.log.warning(f'Error stopping idle hopping engine: {e}')
        self._update_status_timer()
        self.sigMultiStreamStateChanged.emit(False)
        self.log.info('Multi-resonance trace stream stopped (FPGA lock: %s)',
                      self._multi_tracking_active)

    def clear_integrators(self):
        """Clear all per-slot integrators (re-acquire after a disturbance)."""
        self._multi_track_hw().clear_integrators()
        self.log.info('All per-slot integrators cleared')

    # =========================================================================
    # Monitoring
    # =========================================================================
    def _reset_history(self, tracking_started: bool = False):
        n = int(self._num_resonances)
        m = int(self._trace_history_points)
        self._hist_t0 = time.monotonic() if tracking_started else None
        self._tracking_started_at = datetime.now() if tracking_started else None
        self._stream_started_at = None
        self._hist_times = deque(maxlen=m)
        self._hist_corr = [deque(maxlen=m) for _ in range(n)]
        self._hist_err = [deque(maxlen=m) for _ in range(n)]
        self._high_rate_times = None
        self._high_rate_err = None
        self._high_rate_corr = None
        self._high_rate_cic = None
        self._high_rate_sample_rate = None

    @QtCore.Slot()
    def _poll_multi_status(self):
        """Poll per-slot lock status (correction/error/locked) + optional traces."""
        if not self._multi_engine_active:
            return
        lock_hw = self._multi_track_hw()
        try:
            status = lock_hw.get_all_status()
        except Exception as e:
            self.log.debug(f'Multi status poll failed: {e}')
            return

        # accumulate the low-rate correction history
        # A monotonic clock makes elapsed time immune to system-clock/NTP changes.
        now = time.monotonic()
        if self._hist_t0 is None:
            self._hist_t0 = now
        self._hist_times.append(now - self._hist_t0)
        # Append exactly one value per resonance for every timestamp. Missing slot
        # status is represented by NaN rather than creating ragged/misaligned rows.
        for i in range(len(self._hist_corr)):
            st = status[i] if i < len(status) else {}
            self._hist_corr[i].append(float(st.get('correction_hz', np.nan)))
            self._hist_err[i].append(float(st.get('error_lsb', np.nan)))

        t_arr = np.fromiter(self._hist_times, dtype=np.float64)
        self.sigSlotStatusUpdated.emit(list(status))
        self.sigCorrectionHistoryUpdated.emit(
            t_arr, np.asarray([np.fromiter(d, dtype=np.float64) for d in self._hist_corr]))
        self.sigErrorHistoryUpdated.emit(
            t_arr, np.asarray([np.fromiter(d, dtype=np.float64) for d in self._hist_err]))

        # optional high-rate reconstructed traces (dual-quantity: 4 simultaneous
        # traces = both errors + both corrections). Decimated for the GUI.
        if self._multi_stream_active:
            try:
                traces = lock_hw.read_traces()
            except Exception as e:
                self.log.debug(f'read_traces failed: {e}')
                traces = None
            if traces is not None:
                times = np.asarray(traces['times'], dtype=np.float64)
                err = np.asarray(traces['err'], dtype=np.float64)          # (N, T) LSB
                corr_hz = np.asarray(traces['corr_hz'], dtype=np.float64)  # (N, T) Hz
                cic = np.asarray(traces['cic'], dtype=np.float64)          # (N, T) CIC LSB
                # Cache the full rolling window before display decimation.  The
                # same common time vector is used for error and correction.
                self._high_rate_times = times
                self._high_rate_err = err
                self._high_rate_corr = corr_hz
                self._high_rate_cic = cic
                self._high_rate_sample_rate = float(traces.get('sample_rate', np.nan))
                t_n = times.shape[0]
                if t_n > self._HIGH_RATE_MAX_POINTS:
                    idx = np.linspace(0, t_n - 1, self._HIGH_RATE_MAX_POINTS).astype(np.int64)
                    times = times[idx]
                    err = err[:, idx]
                    corr_hz = corr_hz[:, idx]
                    cic = cic[:, idx]
                self.sigHighRateTracesUpdated.emit(times, err, corr_hz, cic)
                # legacy single-quantity signal (corrections) for any older consumer
                self.sigResonanceTracesUpdated.emit(times, corr_hz)

        # continuous per-visit field-trace logging shares this poll's cadence and
        # thread. It owns the physical stream drain while active (read_stream_words),
        # so the read_traces call above intentionally did not drain when logging.
        self._service_field_logging()

    # =========================================================================
    # Tracking data storage
    # =========================================================================
    def _tracking_metadata(self, data_kind: str) -> Dict:
        """Metadata shared by low- and high-rate tracking exports."""
        n = int(self._num_resonances)
        signed_slopes = tuple(
            float(fit['slope']) if fit is not None else np.nan
            for fit in self._resonance_fits[:n])
        demod_hz_per_lsb = tuple(
            (1.0 / slope) if np.isfinite(slope) and slope != 0 else np.nan
            for slope in signed_slopes)
        filters = tuple('' for _ in range(n))
        fir_gains = tuple(np.nan for _ in range(n))
        try:
            calibration = self._multi_track_hw().get_trace_calibration()
            filters = tuple(calibration.get('filters', filters))
            fir_gains = tuple(calibration.get('fir_dc_gain_from_cic_lsb', fir_gains))
        except Exception as e:
            self.log.debug('Could not read CIC/FIR stream calibration: %s', e)
        cic_slopes = tuple(
            (signed_slopes[i] / fir_gains[i])
            if (i < len(fir_gains) and np.isfinite(signed_slopes[i]) and
                np.isfinite(fir_gains[i]) and fir_gains[i] != 0) else np.nan
            for i in range(n))
        cic_hz_per_lsb = tuple(
            (1.0 / slope) if np.isfinite(slope) and slope != 0 else np.nan
            for slope in cic_slopes)
        return {
            'Data kind': data_kind,
            'Number of resonances': n,
            'Engine session started at': (self._tracking_started_at.isoformat()
                                          if self._tracking_started_at is not None else ''),
            'Trace stream started at': (self._stream_started_at.isoformat()
                                        if self._stream_started_at is not None else ''),
            'Tracking active when saved': bool(self._multi_tracking_active),
            'Trace streaming active when saved': bool(self._multi_stream_active),
            'Resonance frequencies (Hz)': tuple(self._resonance_freqs[:n]),
            'LO frequencies (Hz)': tuple(self._lo_frequencies[:n]),
            'Modulation frequency (Hz)': float(self._modulation_frequency),
            'Demodulation phase (deg)': float(self._demod_phase_deg),
            'Dwell time (s)': float(self._dwell_time),
            'Oscillator settle time (s)': float(self._osc_settle_time),
            'Status poll interval (s)': float(self._multi_status_poll_interval),
            'Low-rate history capacity (points)': int(self._trace_history_points),
            'Lock bandwidth (Hz)': float(self._lock_bandwidth),
            'Maximum correction (Hz)': float(self._max_correction_hz_sv),
            'Selected lock-in FIR filters': filters,
            'Demod discriminator slopes (signed LSB/Hz)': signed_slopes,
            'Demod conversion factors (signed Hz/LSB)': demod_hz_per_lsb,
            'FIR DC gain (post-FIR LSB/pre-FIR CIC LSB)': fir_gains,
            'CIC discriminator slopes (signed CIC LSB/Hz)': cic_slopes,
            'CIC conversion factors (signed Hz/CIC LSB)': cic_hz_per_lsb,
        }

    @staticmethod
    def _join_tracking_columns(times, errors, corrections, n, cic=None):
        """Return synchronized tracking columns sharing an identical row axis."""
        times = np.asarray(times, dtype=np.float64)
        errors = np.asarray(errors, dtype=np.float64)
        corrections = np.asarray(corrections, dtype=np.float64)
        if times.ndim != 1 or errors.ndim != 2 or corrections.ndim != 2:
            raise ValueError('Tracking traces must have shapes (T,), (N, T), (N, T).')
        cic_array = None if cic is None else np.asarray(cic, dtype=np.float64)
        if cic_array is not None and cic_array.ndim != 2:
            raise ValueError('CIC trace must have shape (N, T).')
        n = min(int(n), errors.shape[0], corrections.shape[0],
                cic_array.shape[0] if cic_array is not None else int(n))
        length = min(times.size, errors.shape[1], corrections.shape[1])
        if cic_array is not None:
            length = min(length, cic_array.shape[1])
        if n < 1 or length < 1:
            per_res = 3 if cic_array is not None else 2
            return np.empty((0, 1 + per_res * max(n, 0)), dtype=np.float64), tuple()
        columns = [times[:length]]
        headers = ['Elapsed time (s)']
        for i in range(n):
            columns.extend((errors[i, :length], corrections[i, :length]))
            headers.extend((f'Resonance {i} error (LSB)',
                            f'Resonance {i} correction (Hz)'))
            if cic_array is not None:
                columns.append(cic_array[i, :length])
                headers.append(f'Resonance {i} after-CIC error (LSB)')
        return np.column_stack(columns), tuple(headers)

    @QtCore.Slot(str)
    def save_tracking_data(self, tag=None) -> Tuple[str, ...]:
        """Save simultaneous error and correction traces for the current session.

        Two files are produced when high-rate streaming is enabled: the retained
        low-rate register-poll session history and the undecimated current high-rate
        rolling window. Both remain saveable after tracking is stopped.
        """
        # Refresh both products immediately before taking the snapshot. This avoids
        # losing the last poll interval and, because the GUI queues this export
        # before the slower ODMR scan export, drains the high-rate stream promptly.
        if self._multi_engine_active:
            self._poll_multi_status()
        with self._threadlock:
            timestamp = datetime.now()
            tag = f'{tag}_' if tag else ''
            n = int(self._num_resonances)
            storage = TextDataStorage(root_dir=self.module_default_data_dir,
                                      column_formats='.15e')
            saved_paths = []

            low_times = np.fromiter(self._hist_times or (), dtype=np.float64)
            if low_times.size:
                low_err = np.asarray([
                    np.fromiter(values, dtype=np.float64) for values in self._hist_err
                ])
                low_corr = np.asarray([
                    np.fromiter(values, dtype=np.float64) for values in self._hist_corr
                ])
                data, headers = self._join_tracking_columns(
                    low_times, low_err, low_corr, n)
                path, _, _ = storage.save_data(
                    data,
                    metadata=self._tracking_metadata('low-rate retained session history'),
                    nametag=f'{tag}ODMR_tracking_history',
                    timestamp=timestamp,
                    column_headers=headers,
                    column_dtypes=[float] * len(headers),
                    use_timestamp=self._use_timestamp)
                saved_paths.append(path)

            if self._high_rate_times is not None:
                data, headers = self._join_tracking_columns(
                    self._high_rate_times, self._high_rate_err,
                    self._high_rate_corr, n, cic=self._high_rate_cic)
                if data.shape[0]:
                    metadata = self._tracking_metadata('high-rate rolling-window snapshot')
                    metadata['Sample rate (Hz)'] = self._high_rate_sample_rate
                    metadata['Trace alignment'] = (
                        'Post-FIR error, correction, and pre-FIR CIC are captured in one '
                        'FPGA stream record on a common uniform axis; parked resonances '
                        'are zero-order held')
                    metadata['Saved window start elapsed time (s)'] = float(data[0, 0])
                    metadata['Saved window stop elapsed time (s)'] = float(data[-1, 0])
                    # Binary storage avoids blocking the tracking logic for a long
                    # text-formatting pass over hundreds of thousands of samples.
                    high_rate_storage = NpyDataStorage(
                        root_dir=self.module_default_data_dir)
                    path, _, _ = high_rate_storage.save_data(
                        data,
                        metadata=metadata,
                        nametag=f'{tag}ODMR_tracking_high_rate',
                        timestamp=timestamp,
                        column_headers=headers)
                    saved_paths.append(path)

            if saved_paths:
                self.log.info('Saved ODMR tracking traces: %s', ', '.join(saved_paths))
            else:
                self.log.warning('No ODMR tracking trace data available to save.')
            return tuple(saved_paths)

    # =========================================================================
    # Continuous per-visit field-trace logging (single-position B(t))
    # =========================================================================
    # A hours-long, crash-safe record of the SAME quantity the KDC_HW_SYNC_MULTIRES
    # 2D scan maps (fresh-only per-resonance corr_hz / err), reduced per hop-visit with
    # the identical np.nanmean -- so a single-position time trace and a 2D-map pixel are
    # the same operation, one indexed in time and the other in space. Neither the FPGA
    # lock nor the trace stream is started/stopped here (both stay GUI-owned); logging
    # only piggybacks on the running stream. B(t) is derived offline as
    # (res1_corr - res0_corr), exactly as the 2D GUI derives the field map.
    @property
    def field_logging_active(self) -> bool:
        """True once the poll has actually opened the field-trace file (not merely
        requested); poll this after start_field_logging()/stop_field_logging()."""
        return bool(self._field_log_open)

    @QtCore.Slot(str)
    def start_field_logging(self, path_prefix: str) -> None:
        """Request continuous per-visit field logging to ``<path_prefix>.bin`` (+.json).

        Thread-safe: only sets a request flag. The status poll (logic thread) opens the
        file, takes over the stream drain, and appends per-visit rows. Requires the
        multi-resonance MARKED trace stream to be running (Start Stream in the GUI).
        """
        self._field_log_path = str(path_prefix)
        self._field_log_stop_request = False
        self._field_log_start_request = True

    @QtCore.Slot()
    def stop_field_logging(self) -> None:
        """Request a clean stop: flush the trailing partial visit, close the file, and
        return the stream drain to the GUI poll. Thread-safe (flag only)."""
        self._field_log_stop_request = True

    @staticmethod
    def _find_complete_visits(err: np.ndarray, corr: np.ndarray, guard: int):
        """Reduce fresh-only (N, T) traces to one np.nanmean value per completed visit.

        A "visit" is a maximal run of non-NaN samples in ``corr[slot]`` (the interval a
        resonance is actively demodulated between hops). Runs that reach into the last
        ``guard`` samples are NOT emitted -- they may continue past the drain boundary --
        and everything from the earliest such run onward is returned as the retain point
        so the next drain finalizes them whole.

        Returns ``(events, cut_idx)`` where events is a list of
        ``(center_idx, slot, corr_mean, err_mean, count)`` sorted by sample index, and
        ``cut_idx`` is the first sample index to retain (carry forward as the tail).
        """
        corr = np.asarray(corr, dtype=np.float64)
        err = np.asarray(err, dtype=np.float64)
        if corr.ndim != 2 or corr.shape[1] == 0:
            return [], 0
        n_slots, t = corr.shape
        safe_end = max(0, t - int(guard))
        runs = []  # (slot, start, end_inclusive)
        for slot in range(n_slots):
            valid = ~np.isnan(corr[slot])
            if not valid.any():
                continue
            d = np.diff(valid.astype(np.int8))
            starts = list(np.where(d == 1)[0] + 1)
            ends = list(np.where(d == -1)[0])
            if valid[0]:
                starts.insert(0, 0)
            if valid[-1]:
                ends.append(t - 1)
            for s, e in zip(starts, ends):
                runs.append((slot, int(s), int(e)))
        # Retain from the earliest run that reaches the guard zone (or from safe_end,
        # so a run that begins right at the trailing edge is never emitted early).
        cut_idx = safe_end
        for slot, s, e in runs:
            if e >= safe_end:
                cut_idx = min(cut_idx, s)
        events = []
        for slot, s, e in runs:
            if e < cut_idx:  # fully inside the emit region -> finalize
                seg_c = corr[slot, s:e + 1]
                cnt = int(np.count_nonzero(~np.isnan(seg_c)))
                if cnt == 0:
                    continue
                events.append(((s + e) // 2, slot,
                               float(np.nanmean(seg_c)),
                               float(np.nanmean(err[slot, s:e + 1])), cnt))
        events.sort(key=lambda ev: ev[0])
        return events, cut_idx

    def _field_metadata(self) -> Dict:
        meta = self._tracking_metadata('continuous per-visit field trace')
        n = int(self._num_resonances)
        meta.update({
            'record dtype': [[name, str(self._FIELD_DTYPE.fields[name][0])]
                             for name in self._FIELD_DTYPE.names],
            'columns': list(self._FIELD_DTYPE.names),
            'stream sample rate (Hz)': self._field_rate,
            'per-visit reducer': 'np.nanmean of fresh-only samples (matches KDC_HW_SYNC_MULTIRES scan)',
            'field derivation': 'B ~ (res1 corr_hz - res0 corr_hz); f_k = resonance_freqs[k] + corr_hz',
            'absolute resonance frequencies (Hz)': tuple(self._resonance_freqs[:n]),
            'start wall clock (unix s)': self._field_start_wall,
            'start wall clock ISO': (datetime.fromtimestamp(self._field_start_wall).isoformat()
                                     if self._field_start_wall else None),
            'visit rows written': int(self._field_written),
            'transport loss words': int(self._field_loss_words),
            'total stream words': int(self._field_total_words),
        })
        return meta

    def _write_field_metadata(self) -> None:
        try:
            with open(self._field_log_path + '.json', 'w') as f:
                json.dump(self._field_metadata(), f, indent=2, default=str)
        except Exception as e:
            self.log.warning(f'Could not write field-log metadata: {e}')

    def _service_field_logging(self) -> None:
        """Poll-driven state machine: open on request, drain+reduce+append each tick,
        close on request. Runs on the logic thread (called from _poll_multi_status)."""
        hw = self._multi_track_hw()
        if self._field_log_start_request and not self._field_log_open:
            self._field_log_start_request = False
            try:
                self._open_field_log(hw)
            except Exception as e:
                self.log.error(f'Could not start field logging: {e}')
                return
        if not self._field_log_open:
            return
        try:
            self._drain_field_once(hw, final=False)
        except Exception as e:
            self.log.warning(f'Field-log drain error: {e}')
        if self._field_log_stop_request:
            self._field_log_stop_request = False
            try:
                self._drain_field_once(hw, final=True)
            except Exception as e:
                self.log.warning(f'Field-log final drain error: {e}')
            finally:
                self._close_field_log(hw)

    def _open_field_log(self, hw) -> None:
        if hw is None or not bool(getattr(hw, 'trace_stream_active', False)):
            raise RuntimeError('Multi-resonance trace stream is not active; click '
                               'Start Stream before field logging.')
        src = str(getattr(hw, 'stream_source', 'marked'))
        if src != 'marked':
            raise RuntimeError(f"Field logging needs the MARKED stream (per-visit "
                               f"fresh-only); current source is {src!r}. Restart the "
                               f"stream so it uses the marked bitstream.")
        os.makedirs(os.path.dirname(self._field_log_path) or '.', exist_ok=True)
        hw.begin_field_drain()
        self._field_rate = float(hw.stream_sample_rate)
        self._field_words_per_sample = int(getattr(hw, 'stream_words_per_sample', 3))
        self._field_tail_words = np.empty(0, dtype=np.float64)
        self._field_sample_base = 0
        self._field_start_wall = None
        self._field_written = 0
        self._field_loss_words = 0
        self._field_total_words = 0
        self._field_file = open(self._field_log_path + '.bin', 'wb')
        self._field_log_open = True
        self._write_field_metadata()
        self.log.info(f'Field logging started -> {self._field_log_path}.bin '
                      f'(marked stream, {self._field_rate:.1f} Hz aggregate).')

    def _drain_field_once(self, hw, final: bool) -> None:
        if hw is None or not self._field_log_open:
            return
        w = hw.read_stream_words()
        w = (np.asarray(w, dtype=np.float64) if w is not None and len(w)
             else np.empty(0, dtype=np.float64))
        if w.size:
            if self._field_start_wall is None:
                self._field_start_wall = time.time()
            self._field_total_words += int(w.size)
            self._field_loss_words += int(np.count_nonzero(np.isnan(w)))
        words = (np.concatenate([self._field_tail_words, w])
                 if self._field_tail_words.size else w)
        width = int(getattr(self, '_field_words_per_sample', 3))
        n_samp = words.size // width
        if n_samp == 0:
            # No complete triplet yet -- carry the whole (sub-triplet) remainder.
            self._field_tail_words = words
            return
        usable = words[:width * n_samp]
        traces = hw.reconstruct_mapped_traces(usable)
        err = np.asarray(traces['err'], dtype=np.float64)
        corr = np.asarray(traces['corr_hz'], dtype=np.float64)
        t = corr.shape[1] if corr.ndim == 2 else 0
        if t == 0:
            # Reconstruction yielded nothing usable; keep everything for the next drain.
            self._field_tail_words = words
            return
        if t != n_samp:
            # Keep the word<->sample mapping consistent with the reconstruction's own
            # sample count (marked stream is 1 triplet/sample, so this is a safety net).
            n_samp = t
            usable = words[:width * n_samp]
        remainder = words[width * n_samp:]
        guard = 0 if final else int(self._FIELD_GUARD_SAMPLES)
        events, cut_idx = self._find_complete_visits(err, corr, guard)
        cut_idx = max(0, min(int(cut_idx), n_samp))
        if events:
            rows = np.empty(len(events), dtype=self._FIELD_DTYPE)
            if not self._field_rate:
                raise RuntimeError('FPGA stream sample rate is unavailable')
            rate = self._field_rate
            base = self._field_sample_base
            for i, (center, slot, corr_mean, err_mean, cnt) in enumerate(events):
                rows['t_s'][i] = (base + center) / rate
                rows['slot'][i] = slot
                rows['corr_hz'][i] = corr_mean
                rows['err_lsb'][i] = err_mean
                rows['n'][i] = min(int(cnt), 65535)
            rows.tofile(self._field_file)
            self._field_file.flush()
            self._field_written += len(events)
        # Carry the unfinalized samples + any mid-triplet remainder as raw words, and
        # advance the absolute sample base by the number of finalized samples.
        if final:
            self._field_tail_words = np.empty(0, dtype=np.float64)
        else:
            self._field_tail_words = np.concatenate([usable[width * cut_idx:], remainder])
        self._field_sample_base += cut_idx

    def _close_field_log(self, hw) -> None:
        try:
            if self._field_file is not None:
                self._field_file.flush()
                self._field_file.close()
        except Exception as e:
            self.log.warning(f'Error closing field-log file: {e}')
        finally:
            self._field_file = None
        try:
            if hw is not None:
                hw.end_field_drain()
        except Exception as e:
            self.log.warning(f'Error releasing field drain: {e}')
        self._field_log_open = False
        self._write_field_metadata()
        self.log.info(
            f'Field logging stopped: {self._field_written} visit rows, '
            f'transport loss {self._field_loss_words}/{max(1, self._field_total_words)} words.')

    # =========================================================================
    # Convenience setters for bench bring-up
    # =========================================================================
    def set_demod_phase(self, phase_deg: float):
        """Set the demodulation phase (bench-tune after enabling the oscillator)."""
        self._demod_phase_deg = float(phase_deg)
        if self._multi_track_hw().nslots:  # touch HW only if reachable
            self._multi_track_hw().configure_oscillator(
                f_m_hz=float(self._modulation_frequency),
                demod_phase_deg=float(self._demod_phase_deg),
                settle_time_s=float(self._osc_settle_time),
                source='current_step' if int(self._num_resonances) > 1 else 'sw')
        self.log.info(f'Demod phase set to {phase_deg:.1f} deg')

    def get_iq_demod_phase(self) -> float:
        """The iq0 demod phase [deg] you tuned on the ODMR curve. The oscillator's
        demod_phase uses the same sign convention, so this is the value to use for
        tracking (the GUI 'Use iq0 phase' button copies it)."""
        try:
            return float(self._multi_track_hw().get_iq_demod_phase())
        except Exception as e:
            self.log.warning(f'Could not read iq0 demod phase: {e}')
            return float(self._demod_phase_deg)

    def set_modulation_frequency(self, f_m_hz: float):
        """Set the shared modulation frequency f_m [Hz] (applied on reconfigure)."""
        if f_m_hz <= 0:
            raise ValueError('modulation_frequency must be positive')
        self._modulation_frequency = float(f_m_hz)
        self.log.info(f'Modulation frequency set to {f_m_hz/1e3:.4f} kHz. Reconfigure to apply.')

    def set_dwell_time(self, dwell_s: float):
        if dwell_s <= 0:
            raise ValueError('dwell_time must be positive')
        self._dwell_time = float(dwell_s)
        self.log.info(f'Dwell time set to {dwell_s*1e6:.0f} us '
                      f'(hop rate {1.0/dwell_s:.1f} Hz). Reconfigure to apply.')

    def set_settle_time(self, settle_s: float):
        self._osc_settle_time = float(settle_s)
        self.log.info(f'Oscillator settle time set to {settle_s*1e6:.0f} us. Reconfigure to apply.')

    def set_max_correction_hz(self, max_correction_hz: float) -> None:
        """Override: set the (global) saturation limit and remember it."""
        self._max_correction_hz_sv = float(max_correction_hz)
        self._multi_track_hw().set_max_correction_hz(float(max_correction_hz))
        self.log.info(f'Max correction set to {max_correction_hz/1e6:.3f} MHz')

    @QtCore.Slot(float)
    def set_trace_window_seconds(self, seconds: float) -> None:
        """Set the high-rate trace display window (rolling buffer duration) [s]."""
        try:
            self._multi_track_hw().set_trace_window_seconds(float(seconds))
            self.log.debug(f'High-rate trace window set to {float(seconds):.1f} s')
        except Exception as e:
            self.log.warning(f'Could not set trace window: {e}')
