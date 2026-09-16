# -*- coding: utf-8 -*-
"""
Red Pitaya ODMR Frequency Lock Hardware Interface.

Wraps PyRPL's odmr_freq_lock module for Qudi integration.

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

import threading
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from qudi.core.configoption import ConfigOption
from qudi.interface.odmr_freq_lock_interface import OdmrFreqLockInterface
from qudi.interface.multi_resonance_tracking_interface import MultiResonanceTrackingInterface
from qudi.hardware.redpitaya.resource_manager import get_pyrpl_instance, release_pyrpl_instance


class RedPitayaOdmrLockHardware(OdmrFreqLockInterface, MultiResonanceTrackingInterface):
    """
    Hardware interface to Red Pitaya ODMR frequency lock via PyRPL.

    Implements both the single-resonance ``OdmrFreqLockInterface`` (wraps the
    region-8 lock loop ``rp.odmrfreqlock``) and the multi-resonance
    ``MultiResonanceTrackingInterface`` (additionally wraps the region-9
    oscillator/freeze ``rp.odmrmultitrack``, the per-slot integrators, and the
    region-5 ``rp.scan`` continuous-hop loop + push stream + hop markers). The
    overlapping method names (enable_lock, set_invert, set_max_correction_hz,
    set_bandwidth) have identical semantics across the two interfaces, so a single
    implementation satisfies both. The per-resonance SSB cal + LO jump-list live on
    the microwave/IF source module, not here.

    Config example:
        redpitaya_odmr_lock:
            module.Class: 'redpitaya.redpitaya_odmr_lock.RedPitayaOdmrLockHardware'
            options:
                redpitaya_config_name: 'rpy_shared_config'
                redpitaya_hostname: '10.203.129.28'
                lock_in_filter_resonance_1: '2kHz_minphase'
                lock_in_filter_resonance_2: '2kHz_minphase'
                tracking_algorithm: 'conventional'  # or 'smith_linear'
                smith_gain_multiplier: 4.0          # increase toward 10 after validation
                smith_delay_samples: 64
                max_trace_samples: 2000000   # cap on the high-rate trace buffer
    """

    _redpitaya_config_name = ConfigOption('redpitaya_config_name',
                                          default='rpy_shared_config', missing='info')
    _redpitaya_hostname = ConfigOption('redpitaya_hostname', missing='error')
    _redpitaya_fpga_filename = ConfigOption(
        'redpitaya_fpga_filename', default=None, missing='info')
    _lock_in_filter_resonance_1 = ConfigOption(
        'lock_in_filter_resonance_1', default='2kHz_minphase', missing='info')
    _lock_in_filter_resonance_2 = ConfigOption(
        'lock_in_filter_resonance_2', default='2kHz_minphase', missing='info')
    _tracking_algorithm = ConfigOption(
        'tracking_algorithm', default='conventional', missing='info')
    _smith_gain_multiplier = ConfigOption(
        'smith_gain_multiplier', default=4.0, missing='info')
    _smith_delay_samples = ConfigOption(
        'smith_delay_samples', default=None, missing='info')
    # Size (in stream WORDS) of the high-rate display buffer for read_traces. It is a
    # ROLLING window: once full, the oldest samples are dropped so the tracking GUI
    # shows the most recent slice at CONSTANT resolution (like the Time Series GUI),
    # instead of growing unbounded / freezing. In Region-11 mode this is interpreted
    # as an event-record limit. Reduce it for a shorter, finer-resolution window.
    # Indefinite drift is still fully covered by the low-rate
    # per-slot register polling (get_slot_status / correction-history plot).
    _max_trace_samples = ConfigOption('max_trace_samples', default=2_000_000, missing='nothing')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pyrpl = None
        self._lock = None
        self._multitrack = None
        self._fgen3 = None
        self._scan = None
        self._streamer = None
        self._stream_reader = None
        self._fpga_clock_hz = 0
        self._sample_period_cycles = 0
        self._stream_sample_rate_hz = 0.0
        # multi-resonance tracking state
        self._tracking_active = False
        self._streaming_traces = False
        self._single_slot_no_hop = False
        self._nslots = 2
        self._trace_events = None
        self._pending_xmarkers = np.empty(0, dtype=np.int64)
        self._pending_ymarkers = np.empty(0, dtype=np.int64)
        self._marker_x_seen = 0
        self._marker_y_seen = 0
        self._trace_ticks = np.array([], dtype=np.int64)
        self._trace_steps = np.array([], dtype=np.int64)
        # Number of complete stream records removed from the front of the rolling
        # buffer.  This preserves an elapsed-since-stream-start time axis.
        self._trace_sample_offset = 0
        # One structured Region-11 event per returned array element.
        self._stream_words_per_sample = 1
        # Runtime rolling-window size in WORDS for the high-rate display buffer
        # (settable live via set_trace_window_seconds; initialized from the
        # max_trace_samples ConfigOption in on_activate).
        self._trace_window_words = 2_000_000
        self._stream_source = 'events'
        self._configured_lock_in_filters = ['2kHz_minphase', '2kHz_minphase']
        # 2D motor-scan takeover: while True the motor scan owns the physical stream
        # drain (read_stream_words), so read_traces must NOT also drain (word theft).
        self._mapped_scan_active = False
        # Single-position field-trace takeover: while True the tracking logic's
        # continuous field logger owns the physical drain (read_stream_words), so
        # read_traces must NOT also drain (same word-theft guard as the mapped scan).
        self._field_drain_active = False
        # Serializes stream-touching ops (drain + restart) across the two caller
        # threads: the tracking logic's read_traces poll and the motor-scan logic's
        # read_stream_words / enable_position_markers restart. Qudi hardware methods
        # run on the caller's thread (no auto-marshalling), so without this a restart
        # could replace the StreamClient mid-read.
        self._stream_lock = threading.RLock()

    def on_activate(self):
        """Connect to PyRPL and get the lock / multitrack / fgen3 / scan modules."""
        # Get shared PyRPL instance via resource manager
        self._pyrpl, _ = get_pyrpl_instance(
            hostname=self._redpitaya_hostname,
            config_name=self._redpitaya_config_name,
            fpga_filename=self._redpitaya_fpga_filename
        )

        # Runtime high-rate display window (words), seeded from the ConfigOption.
        self._trace_window_words = max(4, int(self._max_trace_samples))

        rp = self._pyrpl.rp
        # Get odmrfreqlock module (PyRPL naming: all lowercase, no underscores)
        self._lock = rp.odmrfreqlock
        # Multi-resonance hardware surface (may be absent on an old bitstream)
        self._multitrack = getattr(rp, 'odmrmultitrack', None)
        self._fgen3 = getattr(rp, 'fgen3', None)
        self._scan = getattr(rp, 'scan', None)
        self._streamer = getattr(rp, 'datastreamer', None)
        if self._streamer is None:
            raise RuntimeError(
                'rp.datastreamer is required; load the matching experiment FPGA image')

        lock0 = rp.lockin
        lock1 = rp.lockin1
        lock0_caps = lock0.require_compatible_firmware()
        lock1_caps = lock1.require_compatible_firmware()
        tracker_caps = self._lock.require_compatible_firmware()
        stream_caps = self._streamer.require_compatible_firmware()
        timing = {(int(c['fpga_clock_hz']), int(c['sample_period_cycles']))
                  for c in (lock0_caps, lock1_caps, tracker_caps)}
        if len(timing) != 1:
            raise RuntimeError(
                'Lock-in/tracker timing capabilities disagree: %r' % sorted(timing))
        self._fpga_clock_hz, self._sample_period_cycles = timing.pop()
        if int(stream_caps['timestamp_clock_hz']) != self._fpga_clock_hz:
            raise RuntimeError('Tracker and data-streamer timestamp clocks disagree')
        self._stream_sample_rate_hz = (
            float(self._fpga_clock_hz) / self._sample_period_cycles)
        if self._tracking_algorithm not in ('conventional', 'smith_linear'):
            raise ValueError(
                f'Invalid tracking_algorithm {self._tracking_algorithm!r}; '
                'use "conventional" or "smith_linear"')
        if (self._tracking_algorithm == 'smith_linear' and
                not tracker_caps.get('algorithms', {}).get('smith_predictor', False)):
            raise RuntimeError(
                'tracking_algorithm="smith_linear" requires the matching Smith FPGA image')
        self._trace_events = np.empty(0, dtype=self._streamer.EVENT_DTYPE)
        self._stream_words_per_sample = 1

        # Multitrack routes resonance 1 through lockin channel 1 and resonance 2
        # through lockin1 channel 1. Configure both instances explicitly.
        self._configure_lock_in_filters(rp)

        # Ensure lock + oscillator are disabled on activation
        self._lock.enable = False
        self._lock.smith_enable = False
        if self._multitrack is not None:
            try:
                self._multitrack.enable = False
            except Exception as e:
                self.log.warning(f'Could not disable multitrack oscillator on activation: {e}')

        if self._multitrack is None:
            self.log.warning('rp.odmrmultitrack not found - multi-resonance tracking '
                             'unavailable (old bitstream?). Single-resonance lock still works.')
        self.log.info(
            'Red Pitaya ODMR Lock connected: %s; rate=%.6f Hz, period=%d clocks, '
            'algorithm=%d', self._redpitaya_hostname, self._stream_sample_rate_hz,
            self._sample_period_cycles, int(tracker_caps['algorithm_id']))

    def _configure_lock_in_filters(self, rp) -> None:
        """Apply the per-resonance FIR phase selection to both lock-in instances."""
        modeled_filter = rp.lockin.smith_filter_name
        requested = [self._lock_in_filter_resonance_1,
                     self._lock_in_filter_resonance_2]
        if self._tracking_algorithm == 'smith_linear':
            if requested[0] != modeled_filter:
                self.log.info(
                    'Smith tracking overrides resonance-1 FIR selection to %s',
                    modeled_filter)
            requested[0] = modeled_filter

        for index, (module_name, filter_name) in enumerate(
                zip(('lockin', 'lockin1'), requested), start=1):
            lock_in = getattr(rp, module_name, None)
            if lock_in is None:
                self.log.warning(
                    f'rp.{module_name} not found; cannot configure resonance {index} FIR')
                continue

            valid_filters = {
                name for name, available in lock_in.filter_capabilities.items()
                if available and name != 'fir_bypass'}
            if '2kHz_minphase' in valid_filters:
                valid_filters.add('2kHz')  # legacy alias
            fallback_filter = ('2kHz_minphase'
                               if '2kHz_minphase' in valid_filters
                               else next(iter(sorted(valid_filters))))

            # Channel 1 is the current multitrack data path. Keep channel 2 in
            # step so it is ready if the unused quadrature lane is enabled later.
            if filter_name not in valid_filters:
                self.log.warning(
                    f'FIR "{filter_name}" is not available in the loaded FPGA '
                    f'profile; using "{fallback_filter}"')
                filter_name = fallback_filter
            lock_in.filter_select_ch1 = filter_name
            lock_in.filter_select_ch2 = filter_name
            self._configured_lock_in_filters[index - 1] = (
                '2kHz_minphase' if filter_name == '2kHz' else filter_name)
            self.log.info(
                f'Resonance {index} FIR ({module_name}): {filter_name}')

    def get_trace_calibration(self) -> Dict[str, Any]:
        """Return the fixed-point scaling needed to calibrate the CIC trace.

        ``fir_dc_gain_from_cic_lsb`` maps the exported CIC word (CIC[39:8]) to
        the selected 32-bit FIR output at DC.  The discriminator fit supplies the
        remaining physical LSB/Hz factor in the tracking logic.
        """
        # Sensitivity sweeps can change the shared FIR after activation.
        # Report the current hardware choice, not the initial config file.
        if self._pyrpl is not None:
            self._configured_lock_in_filters = [
                getattr(self._pyrpl.rp, module).filter_select_ch1
                for module in ('lockin', 'lockin1')]
        minphase_gain = 937716.0 / (2 ** 21)
        linear_gain = (802861.0 / (2 ** 21)) * (299.0 / 256.0)
        fast_linear_gain = 200955.0 / (2 ** 21)
        wide_minphase_gain = 7326.0 / (2 ** 14)
        gains = [fast_linear_gain if name == '20kHz_linear' else
                 wide_minphase_gain if name in ('6kHz_minphase', '10kHz_minphase', '10kHz_minphase_2048') else
                 linear_gain if name == '2kHz_linear' else minphase_gain
                 for name in self._configured_lock_in_filters]
        return {
            'stream_words_per_sample': int(self._stream_words_per_sample),
            'filters': tuple(self._configured_lock_in_filters[:self._nslots]),
            'fir_dc_gain_from_cic_lsb': tuple(gains[:self._nslots]),
        }

    def on_deactivate(self):
        """Disable lock and disconnect."""
        try:
            if self._tracking_active:
                self.stop_tracking()
        except Exception as e:
            self.log.warning(f'Could not stop multi-resonance tracking on deactivation: {e}')
        if self._lock is not None:
            try:
                self._lock.enable = False
            except Exception as e:
                self.log.warning(f'Could not disable lock on deactivation: {e}')
        if self._multitrack is not None:
            try:
                self._multitrack.enable = False
            except Exception as e:
                self.log.warning(f'Could not disable multitrack on deactivation: {e}')

        # Release pyrpl instance
        if self._pyrpl is not None:
            release_pyrpl_instance(
                hostname=self._redpitaya_hostname,
                config_name=self._redpitaya_config_name
            )
            self._pyrpl = None

        self._lock = None
        self.log.info('Red Pitaya ODMR Lock deactivated')

    # =========================================================================
    # OdmrFreqLockInterface Implementation
    # =========================================================================

    def set_bandwidth(self, bandwidth_hz: float, slope_lsb_per_hz: float,
                      pi: bool = False, zero_ratio: float = 3.0) -> None:
        """Configure the loop gains (integral-only, or PI if ``pi=True``).

        Satisfies both OdmrFreqLockInterface (single-res, 2 args) and
        MultiResonanceTrackingInterface (adds ``pi``/``zero_ratio``). The gains are
        global (shared by all slots in hardware).
        """
        if bandwidth_hz <= 0:
            raise ValueError(f'Bandwidth must be positive, got {bandwidth_hz}')
        if slope_lsb_per_hz <= 0:
            raise ValueError(f'Slope must be positive, got {slope_lsb_per_hz}')

        if self._tracking_algorithm == 'smith_linear':
            if pi:
                raise ValueError(
                    'smith_linear currently implements the collaborator I+Smith model; '
                    'set pi=False')
            # Other shared Qudi hardware modules can select the FIR during scans or
            # activation. Reassert the modeled path immediately before configuring
            # the controller, then let the FPGA compatibility input verify it.
            lock_in = self._pyrpl.rp.lockin
            lock_in.fir_bypass_ch1 = False
            lock_in.demod_bypass_ch1 = False
            modeled_filter = lock_in.smith_filter_name
            lock_in.filter_select_ch1 = modeled_filter
            self.set_integrator_source('sw')
            delay_samples = (None if self._smith_delay_samples is None else
                             int(self._smith_delay_samples))
            self._lock.set_bandwidth_smith(
                bandwidth_hz, slope_lsb_per_hz,
                gain_multiplier=float(self._smith_gain_multiplier),
                delay_samples=delay_samples)
            self._configured_lock_in_filters[0] = modeled_filter
            self.log.info(
                f'Lock configured (I+Smith): base BW={bandwidth_hz:.1f} Hz, '
                f'gain=x{float(self._smith_gain_multiplier):.3g}, '
                f'slope={slope_lsb_per_hz:.3e} LSB/Hz, '
                f'delay={int(self._lock.smith_delay_samples)} samples')
        elif pi:
            if not 0.1 <= zero_ratio <= 30.0:
                raise ValueError(f'Zero ratio must be in [0.1, 30.0], got {zero_ratio}')
            self._lock.set_bandwidth_pi(bandwidth_hz, slope_lsb_per_hz, zero_ratio)
            self.log.info(
                f'Lock configured (PI): BW={bandwidth_hz:.1f} Hz, '
                f'slope={slope_lsb_per_hz:.3e} LSB/Hz, alpha={zero_ratio:.2f}'
            )
        else:
            self._lock.set_bandwidth(bandwidth_hz, slope_lsb_per_hz)
            self.log.info(
                f'Lock configured (I-only): BW={bandwidth_hz:.1f} Hz, '
                f'slope={slope_lsb_per_hz:.3e} LSB/Hz'
            )

    def set_tracking_algorithm(self, algorithm: str,
                               smith_gain_multiplier: Optional[float] = None,
                               smith_delay_samples: Optional[int] = None) -> None:
        """Select conventional or single-resonance linear-FIR Smith tracking.

        The loop must be disabled for an algorithm change. Calling
        :meth:`set_bandwidth` afterwards applies the selected controller gains and
        predictor model. This method makes conservative x2/x4/x6/x8/x10 sweeps
        possible from the Qudi namespace without restarting Qudi.
        """
        if self._lock is None:
            raise RuntimeError('Red Pitaya ODMR lock is not active')
        if self._lock.enable:
            raise RuntimeError('disable the lock before changing tracking algorithm')
        if algorithm not in ('conventional', 'smith_linear'):
            raise ValueError('algorithm must be "conventional" or "smith_linear"')
        if algorithm == 'smith_linear' and self._pyrpl.rp.lockin.smith_filter_name is None:
            raise ValueError('This minimum-phase FPGA profile has no Smith-compatible filter')
        if smith_gain_multiplier is not None:
            value = float(smith_gain_multiplier)
            if value <= 0:
                raise ValueError('Smith gain multiplier must be positive')
            self._smith_gain_multiplier = value
        if smith_delay_samples is not None:
            value = int(smith_delay_samples)
            if not 1 <= value < 128:
                raise ValueError('Smith delay must be in [1, 127] samples')
            self._smith_delay_samples = value
        self._tracking_algorithm = algorithm
        self._lock.smith_enable = False
        if algorithm == 'smith_linear':
            lock_in = self._pyrpl.rp.lockin
            modeled_filter = lock_in.smith_filter_name
            lock_in.fir_bypass_ch1 = False
            lock_in.demod_bypass_ch1 = False
            lock_in.filter_select_ch1 = modeled_filter
            self._configured_lock_in_filters[0] = modeled_filter
        self._lock.clear()
        delay_for_log = (self._smith_delay_samples if self._smith_delay_samples is not None
                         else self._pyrpl.rp.lockin.linear_group_delay_samples)
        self.log.info(
            'Tracking algorithm selected: %s (Smith gain x%.3g, delay %d samples)',
            algorithm, float(self._smith_gain_multiplier),
            int(delay_for_log))

    def set_bandwidth_pi(self, bandwidth_hz: float, slope_lsb_per_hz: float,
                         zero_ratio: float = 3.0) -> None:
        """Configure PI frequency lock with zero placement."""
        if bandwidth_hz <= 0:
            raise ValueError(f'Bandwidth must be positive, got {bandwidth_hz}')
        if slope_lsb_per_hz <= 0:
            raise ValueError(f'Slope must be positive, got {slope_lsb_per_hz}')
        if not 0.1 <= zero_ratio <= 30.0:
            raise ValueError(f'Zero ratio must be in [0.1, 30.0], got {zero_ratio}')

        self._lock.set_bandwidth_pi(bandwidth_hz, slope_lsb_per_hz, zero_ratio)

        self.log.info(
            f'Lock configured (PI): BW={bandwidth_hz:.1f} Hz, '
            f'slope={slope_lsb_per_hz:.3e} LSB/Hz, α={zero_ratio:.2f} '
            f'(zero at {bandwidth_hz/zero_ratio:.1f} Hz)'
        )

    def enable_lock(self, enable: bool) -> None:
        """Enable or disable frequency lock."""
        self._lock.enable = bool(enable)
        self.log.info(f'Lock {"enabled" if enable else "disabled"}')

    def get_status(self) -> Dict[str, Any]:
        """Get current lock status."""
        return self._lock.get_status()

    def clear(self) -> None:
        """Clear integrator state."""
        self._lock.clear()
        self.log.debug('Lock integrator cleared')

    def get_constraints(self) -> Dict[str, Any]:
        """Get hardware constraints."""
        # PyRPL constraints (from odmr_freq_lock.py and Verilog)
        return {
            'bandwidth_range': (10.0, 10000.0),  # Hz, practical range
            'slope_range': (1e-6, 1e6),  # LSB/Hz, very wide range
            'damping_range': (0.5, 1.0),  # Dimensionless
            'max_correction_hz': 62.5e6,  # Half of 125 MHz (Nyquist)
        }

    def set_max_correction_hz(self, max_correction_hz: float) -> None:
        """Set maximum frequency correction (FTW saturation limit)."""
        if max_correction_hz <= 0:
            raise ValueError(f'Max correction must be positive, got {max_correction_hz}')
        if max_correction_hz > 62.5e6:
            raise ValueError(f'Max correction cannot exceed 62.5 MHz (Nyquist), got {max_correction_hz}')

        self._lock.max_correction_hz = max_correction_hz

        self.log.info(f'Lock max correction set to {max_correction_hz/1e6:.3f} MHz')

    def get_max_correction_hz(self) -> float:
        """Get current maximum frequency correction setting."""
        return self._lock.max_correction_hz

    def set_invert(self, inverted: bool) -> None:
        """
        Set error signal polarity inversion.

        For LSB (f_RF = f_LO - f_IF): set inverted=True
        For USB (f_RF = f_LO + f_IF): set inverted=False
        """
        self._lock.invert = bool(inverted)
        sideband = 'LSB' if inverted else 'USB'
        self.log.info(f'Error inversion set to {inverted} (for {sideband} operation)')

    def get_invert(self) -> bool:
        """Get current error signal inversion setting."""
        return self._lock.invert

    # =========================================================================
    # MultiResonanceTrackingInterface Implementation
    # =========================================================================

    def _require_multitrack(self):
        if self._multitrack is None:
            raise RuntimeError('Multi-resonance tracking unavailable: rp.odmrmultitrack '
                               'not present (old bitstream?).')

    def _configure_hop_trigger_pin(self):
        """Route the scan LO-hop trigger to DIO7_P as an inverted (idle-high,
        active-low) module output, matching the Windfreak active-low trigger input."""
        try:
            self._pyrpl.rp.hk.configure_pin('P7', direction='output',
                                            source='module', invert=True)
            self.log.debug('DIO7_P configured as inverted scan-trigger output for LO hopping')
        except Exception as e:
            self.log.warning(f'Could not configure DIO7 hop-trigger pin: {e}')

    @property
    def nslots(self) -> int:
        """Number of resonance slots N implemented in hardware."""
        try:
            return int(self._lock.nslots)
        except Exception:
            return int(self._nslots)

    @property
    def tracking_engine_active(self) -> bool:
        """True while the continuous acquisition/hop engine is running."""
        return bool(self._tracking_active)

    @property
    def trace_stream_active(self) -> bool:
        """True while the simultaneous error/correction stream is running."""
        return bool(self._streaming_traces)

    @property
    def position_markers_active(self) -> bool:
        """True while a mapped motor scan owns the physical stream drain."""
        return bool(self._mapped_scan_active)

    @property
    def field_drain_active(self) -> bool:
        """True while a single-position field logger owns the physical stream drain."""
        return bool(self._field_drain_active)

    @property
    def stream_source(self) -> str:
        """Current reconstruction source: 'marked' (fresh-only, per-visit exact) or
        the legacy ZOH-filled 'dual'."""
        return getattr(self, '_stream_source', 'marked')

    @property
    def stream_sample_rate(self) -> float:
        """Aggregate demod stream rate [Hz], reported by the FPGA image."""
        return float(self._stream_sample_rate_hz)

    @property
    def fpga_clock_hz(self) -> int:
        """FPGA DSP clock reported by the active firmware image."""
        return int(self._fpga_clock_hz)

    @property
    def sample_period_cycles(self) -> int:
        """Lock-in/tracker update interval in FPGA clock cycles."""
        return int(self._sample_period_cycles)

    @property
    def ftw_per_hz(self) -> float:
        """Generator frequency-tuning words per hertz for this image."""
        return (2.0 ** 32) / self.fpga_clock_hz

    @property
    def stream_words_per_sample(self) -> int:
        """Number of 32-bit words in one synchronized stream record."""
        return int(self._stream_words_per_sample)

    def begin_field_drain(self) -> None:
        """Hand the physical stream drain to a single-position field logger.

        Mirrors the 2D mapped-scan takeover: while active, ``read_traces`` (the 5 Hz
        GUI poll) stops draining so the field logger's ``read_stream_words`` sees every
        word. The logger still feeds the display buffer (read_stream_words does), so
        the tracking GUI's live traces keep updating. Requires an active trace stream.
        """
        with self._stream_lock:
            if not self._streaming_traces:
                raise RuntimeError('Start the trace stream before field logging.')
            if self._mapped_scan_active:
                raise RuntimeError('A mapped 2D scan already owns the stream drain.')
            self._field_drain_active = True

    def end_field_drain(self) -> None:
        """Return the drain to the GUI poll (idempotent)."""
        with self._stream_lock:
            self._field_drain_active = False

    def get_iq_demod_phase(self) -> float:
        """Current iq0 demod phase [deg] (the legacy/scan reference phase you tune
        visually on the ODMR curve). The oscillator's demod_phase uses the SAME sign
        convention (odmr_multitrack.demod_phase is invert=True like iq0.phase), so
        set the oscillator demod_phase equal to this to reproduce the scan demod."""
        try:
            return float(self._pyrpl.rp.iq0.phase)
        except Exception as e:
            self.log.warning(f'Could not read iq0 phase: {e}')
            return 0.0

    def configure_oscillator(self, f_m_hz: float, demod_phase_deg: float,
                             settle_time_s: float, source: str = 'current_step') -> None:
        """Configure the per-channel modulation oscillator and freeze window."""
        self._require_multitrack()
        if source not in ('current_step', 'sw'):
            raise ValueError(f"source must be 'current_step' or 'sw', got {source}")
        self._multitrack.frequency = float(f_m_hz)
        self._multitrack.demod_phase = float(demod_phase_deg)
        self._multitrack.settle_time = float(settle_time_s)
        self._multitrack.src = source
        if source == 'sw':
            try:
                self._multitrack.sw_channel = 0
            except Exception:
                pass
        self.log.info(
            f'Oscillator configured: f_m={f_m_hz/1e3:.4f} kHz, '
            f'demod_phase={demod_phase_deg:.1f} deg, settle={settle_time_s*1e6:.0f} us, '
            f'src={source}'
        )

    def enable_oscillator(self, enable: bool) -> None:
        """Enable/disable the multi-channel oscillator + freeze gating."""
        self._require_multitrack()
        self._multitrack.enable = bool(enable)
        self.log.info(f'Multitrack oscillator {"enabled" if enable else "disabled"}')

    def set_integrator_source(self, source: str = 'current_step') -> None:
        """Select which slot integrates / which cal slot is active: hardware hop
        index ('current_step') or software ('sw'). Applies to both the per-slot
        integrators (odmrfreqlock) and the cal-slot bank (fgen3)."""
        if source not in ('current_step', 'sw'):
            raise ValueError(f"source must be 'current_step' or 'sw', got {source}")
        hw = (source == 'current_step')
        self._lock.active_slot_src = hw
        if not hw:
            try:
                self._lock.active_slot = 0
            except Exception:
                pass
        if self._fgen3 is not None:
            self._fgen3.active_slot_src = hw
            if not hw:
                try:
                    self._fgen3.active_slot = 0
                except Exception:
                    pass
        self.log.info(f'Integrator + cal-slot source set to {source} '
                      f'(active_slot_src={hw})')

    def clear_integrators(self) -> None:
        """Clear all per-slot integrator states."""
        self._lock.clear()
        self.log.debug('All per-slot integrators cleared')

    def get_slot_status(self, slot: int) -> Dict[str, Any]:
        """Per-slot lock status (enabled/locked/saturated/error_lsb/correction_hz)."""
        n = self.nslots
        if not (0 <= slot < n):
            raise ValueError(f'slot must be in 0..{n-1}, got {slot}')
        status = dict(self._lock.status_slot(slot))
        # Ensure the headline fields are present with consistent names
        status.setdefault('correction_hz', self._lock.correction_hz_slot(slot))
        status.setdefault('error_lsb', self._lock.error_lsb_slot(slot))
        status['slot'] = slot
        return status

    def get_all_status(self) -> List[Dict[str, Any]]:
        """Per-slot status for all slots. Built from get_slot_status so the
        correction_hz / error_lsb / locked / saturated keys are always present
        (the GUI reads these)."""
        return [self.get_slot_status(s) for s in range(self.nslots)]

    def start_tracking(self, nslots: int, dwell_time_s: float,
                       settling_time_s: float, trigger_length_s: float,
                       stream_traces: bool = True) -> None:
        """Start indefinite hardware-driven LO hopping over ``nslots`` resonances."""
        self._require_multitrack()
        if self._scan is None:
            raise RuntimeError('rp.scan not available; cannot start hopping.')
        if self._tracking_active:
            self.log.warning('Multi-resonance tracking already active; ignoring start.')
            return

        self._nslots = int(nslots)
        if self._tracking_algorithm == 'smith_linear' and self._nslots != 1:
            raise RuntimeError(
                'smith_linear is validated only for single-resonance tracking; '
                'select conventional control for multi-resonance hopping')
        # ensure the LO-hop trigger reaches the Windfreak (DIO7_P, inverted)
        if self._nslots > 1:
            self._configure_hop_trigger_pin()
        # reset the high-rate trace buffers
        self._trace_ticks = np.array([], dtype=np.int64)
        self._trace_steps = np.array([], dtype=np.int64)
        self._trace_sample_offset = 0
        self._streaming_traces = False

        self._single_slot_no_hop = self._nslots == 1
        if self._single_slot_no_hop:
            self.set_integrator_source('sw')
            try:
                if self._scan.busy:
                    self._scan.stop()
                self._scan.reset()
                self._scan.num_steps = 1
            except Exception as e:
                self.log.warning(f'Could not reset scan current_step for N=1 tracking: {e}')
            if self._fgen3 is not None:
                try:
                    self._fgen3.active_slot_src = False
                    self._fgen3.active_slot = 0
                except Exception:
                    pass

        # Mark the engine active before starting its two independent components so
        # a caller can reliably use stop_tracking() to clean up a partial failure.
        self._tracking_active = True

        if stream_traces:
            # Region-11 supplies independently timestamped FIR/CIC/correction and
            # step/live-window events. Hopping and observation remain independent.
            self._start_trace_stream_hardware()

        if not self._single_slot_no_hop:
            # Hopping is independent of streaming. Starting it separately is what
            # lets stop_trace_stream() leave N=2 tracking running.
            self._scan.num_steps = self._nslots
            self._scan.dwell_time = dwell_time_s
            self._scan.settling_time = settling_time_s
            self._scan.trigger_length = trigger_length_s
            self._scan.start(continuous=True)

        self._tracking_active = True
        self.log.info(
            f'Multi-resonance tracking started: N={self._nslots}, '
            f'dwell={dwell_time_s*1e6:.0f} us, settle={settling_time_s*1e6:.0f} us, '
            f'stream_traces={stream_traces}'
            + (' (single-slot no-hop: dwell/settle ignored)' if self._single_slot_no_hop else '')
        )

    def stop_tracking(self) -> None:
        """Stop continuous hopping (and the trace stream if running)."""
        if not self._tracking_active:
            return
        try:
            with self._stream_lock:
                if self._streaming_traces:
                    self._streamer.unsubscribe(self._stream_reader)
                    self._stream_reader = None
                    self._streaming_traces = False
                if not self._single_slot_no_hop:
                    self._scan.stop()
        finally:
            self._tracking_active = False
            self._streaming_traces = False
            self._single_slot_no_hop = False
            self._mapped_scan_active = False
        self.log.info('Multi-resonance tracking stopped')

    def _start_trace_stream_hardware(self) -> None:
        """Start/reset only the push-stream side of the tracking engine."""
        self._trace_ticks = np.array([], dtype=np.int64)
        self._trace_steps = np.array([], dtype=np.int64)
        self._trace_sample_offset = 0
        self._pending_xmarkers = np.empty(0, dtype=np.int64)
        self._pending_ymarkers = np.empty(0, dtype=np.int64)
        self._trace_events = np.empty(0, dtype=self._streamer.EVENT_DTYPE)
        self._stream_reader = self._streamer.subscribe(
            ('fir', 'cic', 'correction', 'step', 'x_position',
             'y_position', 'window'), poll_us=50,
            ring_bytes=32 << 20, coalesce_us=500)
        self._stream_source = 'events'
        self._stream_words_per_sample = 1
        self._streaming_traces = True

    def start_trace_stream(self) -> None:
        """Start traces without changing hopping or the FPGA lock state."""
        if not self._tracking_active:
            raise RuntimeError('Start the hopping/acquisition engine before streaming.')
        if self._streaming_traces:
            self.log.warning('Multi-resonance trace stream already active.')
            return
        with self._stream_lock:
            self._start_trace_stream_hardware()
        self.log.info('Multi-resonance trace stream started independently.')

    def stop_trace_stream(self) -> None:
        """Stop traces without changing hopping or the FPGA lock state."""
        with self._stream_lock:
            # Check ownership under the same lock used by enable_position_markers;
            # otherwise a concurrent GUI stop could slip between marker takeover
            # and the stream restart.
            if not self._streaming_traces:
                return
            if self._mapped_scan_active:
                raise RuntimeError(
                    'Cannot stop the trace stream while a KDC_HW_SYNC_MULTIRES motor '
                    'scan owns its position-marker data.')
            self._streamer.unsubscribe(self._stream_reader)
            self._stream_reader = None
            self._streaming_traces = False
            self._mapped_scan_active = False
        self.log.info('Multi-resonance trace stream stopped independently.')

    def read_traces(self) -> Optional[Dict[str, Any]]:
        """Reconstructed per-resonance high-rate traces since session start (capped).

        Aligns timestamped Region-11 events into synchronous per-resonance post-FIR
        error, correction, and pre-FIR CIC traces at the FPGA-reported demodulation
        rate. Indefinite operation is covered by per-slot register polling
        (:meth:`get_slot_status`); this high-rate buffer is a ROLLING window of the
        last ``max_trace_samples`` words (oldest dropped), so it stays bounded at
        constant resolution instead of growing/freezing.

        Returns:
            dict or None: ``{'times': (T,), 'err': (N, T) raw LSB,
            'corr_hz': (N, T) Hz, 'sample_rate': Hz}`` where T is the number of
            complete records received and N is the number of resonances.
        """
        if not (self._tracking_active and self._streaming_traces):
            return None

        if not self._mapped_scan_active and not self._field_drain_active:
            with self._stream_lock:
                new_events = self._stream_reader.read_events()
            self._append_event_trace(new_events)
        if self._trace_events is None or not self._trace_events.size:
            return None
        rec = self._streamer.reconstruct_tracking(
            self._trace_events, nslots=self._nslots,
            interval=self._sample_period_cycles,
            correction_converter=self._scan.ftw_to_hz)
        err = self._ffill_marked_rows(rec['err'], rec['step'])
        corr_hz = self._ffill_marked_rows(rec['corr'], rec['step'])
        cic = self._ffill_marked_rows(rec['cic'], rec['step'])
        if rec['ticks'].size:
            times = ((rec['ticks'] - rec['ticks'][0]).astype(np.float64) /
                     self._fpga_clock_hz + self._trace_sample_offset /
                     self._stream_sample_rate_hz)
        else:
            times = np.empty(0, dtype=np.float64)
        return {'times': times, 'err': err, 'corr_hz': corr_hz,
                'cic': cic, 'dead': rec['dead'],
                'sample_rate': self._stream_sample_rate_hz}

    @staticmethod
    def _ffill_rows(a):
        """Row-wise forward-fill of NaNs (zero-order hold) for (N, T) arrays."""
        a = np.asarray(a, dtype=np.float64)
        if a.ndim != 2 or a.size == 0:
            return a
        out = a.copy()
        for r in range(out.shape[0]):
            row = out[r]
            valid = ~np.isnan(row)
            if not valid.any():
                continue
            idx = np.where(valid, np.arange(row.size), 0)
            np.maximum.accumulate(idx, out=idx)
            out[r] = row[idx]
        return out

    @classmethod
    def _ffill_marked_rows(cls, a, step):
        """Hold parked/dead slots while preserving loss in a resonance's live slot."""
        source = np.asarray(a, dtype=np.float64)
        out = cls._ffill_rows(source)
        step = np.asarray(step, dtype=np.int64)
        for r in range(out.shape[0]):
            live_loss = (step == r) & np.isnan(source[r])
            out[r, live_loss] = np.nan
        return out

    def _append_event_trace(self, events) -> None:
        """Append decoded events and keep a bounded, timestamped rolling window."""
        if events is not None and len(events):
            events = np.asarray(events, dtype=self._streamer.EVENT_DTYPE)
            self._trace_events = np.concatenate((self._trace_events, events))
        if self._trace_events is None:
            return
        # A mapped motor scan consumes marker indices incrementally. Retain its
        # complete event history until the scan ends so trimming cannot move the
        # marker arrays underneath the per-axis ``seen`` cursors.
        if self._mapped_scan_active:
            return
        # The old option counted words. Preserve approximately the same memory
        # bound by interpreting it as a maximum number of event records here.
        keep = max(1, int(self._trace_window_words))
        if self._trace_events.size > keep:
            dropped = self._trace_events.size - keep
            dropped_fir = np.count_nonzero(
                self._trace_events[:dropped]['source'] ==
                self._streamer.SOURCE_IDS['fir'])
            self._trace_sample_offset += int(dropped_fir)
            self._trace_events = self._trace_events[dropped:]

    def set_trace_window_seconds(self, seconds: float) -> None:
        """Set the high-rate display window duration (rolling buffer length).

        Live-settable from the tracking GUI. The event buffer is sized from the
        FPGA-reported sample rate; shrinking trims immediately on the next read.
        Independent of
        the per-slot register-poll drift history (that has its own length).
        """
        s = float(seconds)
        if not np.isfinite(s) or s <= 0:
            self.log.warning('set_trace_window_seconds: ignoring non-positive value %r', seconds)
            return
        self._trace_window_words = max(1, int(round(s * self._stream_sample_rate_hz)))
        # Trim right away if the new window is shorter than the current buffer.
        if self._trace_events is not None and self._trace_events.size:
            self._append_event_trace(np.empty(0, dtype=self._streamer.EVENT_DTYPE))
        self.log.debug('High-rate trace window set to %.2f s (%d events).',
                       s, self._trace_window_words)

    # =========================================================================
    # 2D motor-scan composition: x/y position markers on the running stream
    # =========================================================================
    def enable_position_markers(self, enable: bool) -> None:
        """Enable/disable KDC x/y position-marker capture on the running stream.

        Position events are always available Region-11 sources. Restarting this
        subscriber establishes a fresh local marker origin without resetting the
        shared packetizer or disturbing other subscribers and does not perturb the
        controller or hop sequencer.
        """
        if self._scan is None:
            self.log.warning('enable_position_markers: rp.scan unavailable.')
            return
        with self._stream_lock:
            if enable and not (self._tracking_active and self._streaming_traces):
                raise RuntimeError(
                    'KDC_HW_SYNC_MULTIRES needs an active multi-resonance trace stream. '
                    'Configure tracking and click Start Stream first; Start Tracking is '
                    'optional (open-loop error mapping is supported).')
            if enable:
                # Stop the tracking poll's drain FIRST (so it can't read the
                # StreamClient we are about to replace), then restart with markers.
                self._mapped_scan_active = True
                self._restart_mapped_stream(xy_markers=True)
            else:
                self._restart_mapped_stream(xy_markers=False)
                self._mapped_scan_active = False
        self.log.info('KDC x/y position markers %s (stream restarted, aligned).',
                      'ENABLED (2D mapped scan)' if enable else 'disabled')

    def _restart_mapped_stream(self, xy_markers: bool) -> None:
        """Replace this subscriber to establish a fresh local marker origin."""
        elapsed = 0
        if self._trace_events is not None:
            elapsed = int(np.count_nonzero(
                self._trace_events['source'] == self._streamer.SOURCE_IDS['fir']))
        self._streamer.unsubscribe(self._stream_reader)
        self._trace_events = np.empty(0, dtype=self._streamer.EVENT_DTYPE)
        self._stream_reader = self._streamer.subscribe(
            ('fir', 'cic', 'correction', 'step', 'x_position',
             'y_position', 'window'), poll_us=50,
            ring_bytes=32 << 20, coalesce_us=500)
        self._trace_sample_offset += elapsed
        self._pending_xmarkers = np.empty(0, dtype=np.int64)
        self._pending_ymarkers = np.empty(0, dtype=np.int64)
        self._marker_x_seen = 0
        self._marker_y_seen = 0

    def read_position_markers(self) -> Tuple[np.ndarray, np.ndarray]:
        """New (x, y) position markers since the last call, as sample indices."""
        empty = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        if not self._streaming_traces:
            return empty
        result = (self._pending_xmarkers, self._pending_ymarkers)
        self._pending_xmarkers = np.empty(0, dtype=np.int64)
        self._pending_ymarkers = np.empty(0, dtype=np.int64)
        return result

    def read_stream_words(self) -> np.ndarray:
        """New raw self-describing stream words (destructive drain)."""
        if not self._streaming_traces:
            return np.empty(0, dtype=self._streamer.EVENT_DTYPE)
        try:
            with self._stream_lock:
                events = self._stream_reader.read_events()
        except Exception as e:
            self.log.warning('read_stream_words event drain failed: %s', e)
            return np.empty(0, dtype=self._streamer.EVENT_DTYPE)
        self._append_event_trace(events)
        if self._trace_events.size:
            rec = self._streamer.reconstruct_tracking(
                self._trace_events, nslots=self._nslots,
                interval=self._sample_period_cycles)
            self._pending_xmarkers = rec['x_position'][self._marker_x_seen:]
            self._pending_ymarkers = rec['y_position'][self._marker_y_seen:]
            self._marker_x_seen = rec['x_position'].size
            self._marker_y_seen = rec['y_position'].size
        return events

    def reconstruct_mapped_traces(self, words) -> Dict[str, Any]:
        """Decode accumulated record words -> fresh-only err/corr/CIC traces."""
        events = np.asarray(words, dtype=self._streamer.EVENT_DTYPE)
        rec = self._streamer.reconstruct_tracking(
            events, nslots=self._nslots,
            interval=self._sample_period_cycles,
            correction_converter=self._scan.ftw_to_hz)
        times = ((rec['ticks'] - rec['ticks'][0]).astype(np.float64) /
                 self._fpga_clock_hz
                 if rec['ticks'].size else np.empty(0, dtype=np.float64))
        return {'err': rec['err'], 'corr_hz': rec['corr'], 'cic': rec['cic'],
                'times': times, 'dead': rec['dead'],
                'sample_rate': self._stream_sample_rate_hz}
