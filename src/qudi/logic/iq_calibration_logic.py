"""
IQ mixer calibration logic module for qudi.

This module orchestrates the automated calibration of IQ mixers by:
1. Controlling the microwave source
2. Measuring LO leakage and image rejection via spectrum analyzer
3. Optimizing DC offsets (I, Q) and IQ imbalance (g, phi)
4. Calculating SFDR (Spurious-Free Dynamic Range) from spectrum measurements
5. Generating before/after spectrum comparison PDF plots
6. Saving calibration data to CSV files for later use

The calibration process uses Nelder-Mead optimization to minimize:
- LO leakage (by adjusting DC offsets)
- Image power (by adjusting gain and phase imbalance)

SFDR measurement (optional, enabled by default):
- Captures wideband spectrum before and after calibration
- Calculates SFDR as target_power - max_spur_power
- Identifies harmonic origin of spurs (LO, LO±2IF, LO±3IF)
- Generates annotated PDF comparison plots
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field

from PySide2 import QtCore
from scipy.optimize import minimize

from qudi.core.module import LogicBase
from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.util.mutex import RecursiveMutex


@dataclass
class CalibrationPoint:
    """Data class for a single calibration point result."""
    lo_frequency_ghz: float
    if_amplitude: float
    I_offset: float
    Q_offset: float
    g: float
    phi: float
    lo_leakage_dbm: float
    image_power_dbm: float
    sfdr_db: Optional[float] = None


@dataclass
class CalibrationConfig:
    """Configuration for a calibration run."""
    if_frequency_hz: float
    lo_frequencies_ghz: List[float]
    if_amplitudes: List[float]
    optimization_iterations: int = 2
    xatol: float = 1e-4
    fatol: float = 3.0
    maxiter: int = 50


class IQCalibrationLogic(LogicBase):
    """Logic module for automated IQ mixer calibration.

    This module orchestrates the calibration of IQ mixers by:
    1. Controlling the microwave source
    2. Measuring LO leakage and image rejection via spectrum analyzer
    3. Optimizing DC offsets (I, Q) and IQ imbalance (g, phi)
    4. Saving calibration data to CSV files for later use

    Example configuration:
        logic:
            iq_calibration_logic:
                module.Class: 'iq_calibration_logic.IQCalibrationLogic'
                options:
                    if_frequency_hz: 21.58e6
                    calibration_output_dir: 'C:\\calibration_results'
                    optimization_iterations: 2
                    xatol: 1e-4
                    fatol: 3.0
                    maxiter: 50
                connect:
                    spectrum_analyzer: 'rto6_spectrum'
                    microwave: 'mw_source_rp_windfreak'
    """

    # Connectors to hardware modules
    _spectrum_analyzer = Connector(interface='SpectrumMeasurementInterface', name='spectrum_analyzer')
    _microwave = Connector(interface='MicrowaveInterface', name='microwave')

    # Configuration options
    _if_frequency_hz = ConfigOption(
        name='if_frequency_hz',
        default=21.58e6
    )
    _calibration_output_dir = ConfigOption(
        name='calibration_output_dir',
        default='C:\\calibration_results'
    )
    _optimization_iterations = ConfigOption(
        name='optimization_iterations',
        default=2
    )
    _xatol = ConfigOption(
        name='xatol',
        default=1e-4
    )
    _fatol = ConfigOption(
        name='fatol',
        default=3.0
    )
    _maxiter = ConfigOption(
        name='maxiter',
        default=50
    )
    _sideband = ConfigOption(
        name='sideband',
        default='upper'
    )

    # Robustness / performance options (ported from legacy implementation)
    _warm_start = ConfigOption(
        name='warm_start',
        default=True
    )
    _dc_simplex_step = ConfigOption(
        name='dc_simplex_step',
        default=0.05
    )
    _iq_g_simplex_step = ConfigOption(
        name='iq_g_simplex_step',
        default=0.02
    )
    _iq_phi_simplex_step = ConfigOption(
        name='iq_phi_simplex_step',
        default=0.05
    )
    _point_retry_count = ConfigOption(
        name='point_retry_count',
        default=3
    )
    _point_retry_delay_s = ConfigOption(
        name='point_retry_delay_s',
        default=5.0
    )

    # SFDR measurement options (ported from legacy IQ_calibrator_pyrpl.py)
    _enable_sfdr_measurement = ConfigOption(
        name='enable_sfdr_measurement',
        default=True
    )
    _sfdr_span_multiplier = ConfigOption(
        name='sfdr_span_multiplier',
        default=3.0
    )
    _sfdr_bandwidth_factor = ConfigOption(
        name='sfdr_bandwidth_factor',
        default=8.0  # 8 × IF frequency for search bandwidth
    )
    _save_spectrum_plots = ConfigOption(
        name='save_spectrum_plots',
        default=True
    )
    _sfdr_sweep_rbw_hz = ConfigOption(
        name='sfdr_sweep_rbw_hz',
        default=1e6  # 1 MHz RBW for wideband SFDR sweeps
    )

    # Measurement setup options (Task B - ported from legacy _setup_oscilloscope_measurement)
    _measurement_rbw_hz = ConfigOption(
        name='measurement_rbw_hz',
        default=1e6  # 1 MHz RBW for marker measurements
    )
    _measurement_span_factor = ConfigOption(
        name='measurement_span_factor',
        default=4.1  # Span = factor × IF frequency
    )

    # Signal presence verification options (Task C)
    _signal_presence_threshold_dbm = ConfigOption(
        name='signal_presence_threshold_dbm',
        default=-70.0  # Below this, warn that signal may be too weak
    )
    _abort_on_weak_signal = ConfigOption(
        name='abort_on_weak_signal',
        default=False  # If True, abort calibration when signal is weak
    )

    # Negative SFDR retry options (for low IF amplitudes where USB < LSB can occur)
    _negative_sfdr_retry_count = ConfigOption(
        name='negative_sfdr_retry_count',
        default=2  # Number of retries with perturbed initial conditions if USB < LSB
    )

    # Persisted state
    _calibration_results = StatusVar(name='calibration_results', default={})
    _last_calibration_time = StatusVar(name='last_calibration_time', default=None)
    _current_calibration_file = StatusVar(name='current_calibration_file', default=None)

    # Qt signals for GUI and async updates
    sigCalibrationProgress = QtCore.Signal(float, str)  # (progress_pct, status_msg)
    sigCalibrationPointComplete = QtCore.Signal(dict)   # single point result
    sigCalibrationComplete = QtCore.Signal(str)         # output file path
    sigCalibrationError = QtCore.Signal(str)            # error message
    sigSpectrumUpdated = QtCore.Signal(object, object)  # (freq_array, amp_array)
    sigMeasurementUpdate = QtCore.Signal(float, float)  # (lo_leakage, image_power)
    sigSFDRUpdated = QtCore.Signal(float, dict)         # (sfdr_db, details_dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_lock = RecursiveMutex()
        self._abort_requested = False
        self._current_if_amplitude = 0.0
        self._measurement_counter = 0

    @staticmethod
    def _normalize_sideband(sideband: str) -> str:
        sideband_normalized = str(sideband).strip().lower()
        if sideband_normalized in {'upper', 'usb'}:
            return 'upper'
        if sideband_normalized in {'lower', 'lsb'}:
            return 'lower'
        raise ValueError(f'Invalid sideband "{sideband}". Use "upper"/"usb" or "lower"/"lsb".')

    @staticmethod
    def _initial_simplex_2d(center: Tuple[float, float], step_x: float, step_y: float) -> np.ndarray:
        """Create a 2D Nelder–Mead initial simplex around a center point."""
        cx, cy = float(center[0]), float(center[1])
        sx, sy = abs(float(step_x)), abs(float(step_y))
        if sx == 0.0:
            sx = 1e-6
        if sy == 0.0:
            sy = 1e-6

        simplex = np.zeros((3, 2), dtype=np.float64)
        simplex[0, :] = [cx - sx, cy - sy]
        simplex[1, :] = [cx + sx, cy]
        simplex[2, :] = [cx, cy + sy]
        return simplex

    def on_activate(self) -> None:
        """Initialize the calibration logic."""
        os.makedirs(self._calibration_output_dir, exist_ok=True)
        self.log.info(f'IQ Calibration Logic activated. Output dir: {self._calibration_output_dir}')

    def on_deactivate(self) -> None:
        """Clean up."""
        self._abort_requested = True

    # ─────────────────────────────────────────────────────────────────────────
    # Core Calibration Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _measure_lo_leakage(self, params: np.ndarray) -> float:
        """Objective function for DC offset optimization.

        Sets DC offsets and measures power at LO frequency.

        Args:
            params: [i_offset, q_offset] array

        Returns:
            Power at LO frequency in dBm
        """
        i_offset, q_offset = params
        mw = self._microwave()

        # Apply DC offsets
        mw.calibration_set_dc_offsets(i_offset, q_offset)

        # Measure power at LO frequency (marker 1)
        spectrum = self._spectrum_analyzer()
        spectrum.trigger_single()
        power_dbm = spectrum.get_marker_amplitude(1)

        self._measurement_counter += 1
        if self._measurement_counter % 5 == 0:
            self.sigMeasurementUpdate.emit(power_dbm, float('nan'))

        return power_dbm

    def _measure_image_rejection(self, params: np.ndarray, amplitude: float) -> float:
        """Objective function for IQ imbalance optimization.

        Applies IQ correction and measures power at image frequency.

        Args:
            params: [g, phi] array (gain imbalance, phase imbalance in radians)
            amplitude: IF amplitude

        Returns:
            Power at image frequency in dBm
        """
        g, phi = params
        mw = self._microwave()

        # Apply IQ correction
        mw.calibration_set_iq_correction(g, phi, amplitude, component_index=0)

        # Measure power at image frequency (marker 2)
        spectrum = self._spectrum_analyzer()
        spectrum.trigger_single()
        power_dbm = spectrum.get_marker_amplitude(2)

        self._measurement_counter += 1
        if self._measurement_counter % 5 == 0:
            self.sigMeasurementUpdate.emit(float('nan'), power_dbm)

        return power_dbm

    # ─────────────────────────────────────────────────────────────────────────
    # Spectrum Setup and Signal Verification (Tasks B & C)
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_spectrum_measurement(self, lo_freq_hz: float, if_freq_hz: float) -> None:
        """Configure spectrum analyzer for calibration measurements.

        Sets center frequency, span, and RBW on the spectrum analyzer,
        and ensures markers are enabled and ready to use.

        Ported from legacy _setup_oscilloscope_measurement().

        Args:
            lo_freq_hz: LO frequency in Hz
            if_freq_hz: IF frequency in Hz
        """
        spectrum = self._spectrum_analyzer()

        # Calculate appropriate span
        span_hz = float(self._measurement_span_factor) * if_freq_hz

        # Configure spectrum analyzer
        spectrum.center_frequency = lo_freq_hz
        spectrum.span = span_hz
        spectrum.resolution_bandwidth = float(self._measurement_rbw_hz)

        # Trigger single acquisition and activate markers
        spectrum.trigger_single()
        spectrum.enable_marker(1, True)
        spectrum.enable_marker(2, True)

        self.log.debug(
            f'Spectrum measurement configured: center={lo_freq_hz/1e9:.4f} GHz, '
            f'span={span_hz/1e6:.1f} MHz, RBW={self._measurement_rbw_hz/1e3:.1f} kHz'
        )

    def _verify_signal_present(
            self,
            lo_freq_hz: float,
            if_freq_hz: float,
            sideband: str = 'upper'
    ) -> Tuple[bool, float]:
        """Verify that the calibration signal is present.

        Measures power at the expected desired tone frequency and compares
        against threshold. Logs a warning if signal is weak.

        Args:
            lo_freq_hz: LO frequency in Hz
            if_freq_hz: IF frequency in Hz
            sideband: 'upper' or 'lower'

        Returns:
            Tuple of (signal_ok, measured_power_dbm)
        """
        sideband = self._normalize_sideband(sideband)
        spectrum = self._spectrum_analyzer()

        # Calculate expected signal frequency (desired tone)
        if sideband == 'upper':
            signal_freq_hz = lo_freq_hz + if_freq_hz
        else:
            signal_freq_hz = lo_freq_hz - if_freq_hz

        # Measure power at expected signal location using marker 1.
        # IMPORTANT: marker 1 is also used for LO leakage optimization, so restore it to LO afterwards.
        try:
            spectrum.set_marker_frequency(1, signal_freq_hz)
            spectrum.trigger_single()
            power_dbm = spectrum.get_marker_amplitude(1)
        finally:
            # Best-effort restore; do not mask the original exception.
            try:
                spectrum.set_marker_frequency(1, lo_freq_hz)
            except Exception:
                pass

        threshold = float(self._signal_presence_threshold_dbm)
        signal_ok = power_dbm > threshold

        if not signal_ok:
            self.log.warning(
                f'Weak signal detected: {power_dbm:.1f} dBm at {signal_freq_hz/1e9:.4f} GHz '
                f'(threshold: {threshold:.1f} dBm)'
            )
        else:
            self.log.info(f'Signal verified: {power_dbm:.1f} dBm at {signal_freq_hz/1e9:.4f} GHz')

        return signal_ok, power_dbm

    # ─────────────────────────────────────────────────────────────────────────
    # SFDR Measurement and Spectrum Plotting Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _acquire_sfdr_spectrum(
            self,
            lo_frequency_hz: float,
            if_frequency_hz: float,
            span_multiplier: Optional[float] = None,
            restore_settings: bool = False,
            saved_settings: Optional[dict] = None
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Acquire wideband spectrum for SFDR calculation.

        Configures the spectrum analyzer with appropriate span and resolution,
        triggers single acquisition, and returns the spectrum data.

        Args:
            lo_frequency_hz: LO frequency in Hz
            if_frequency_hz: IF frequency in Hz
            span_multiplier: Multiplier for span (uses config default if None)
            restore_settings: If True, restore SA settings after acquisition
            saved_settings: Previously saved SA settings to restore (if None, saves current)

        Returns:
            Tuple of (frequency_array_hz, amplitude_array_dbm, saved_settings_dict)
            The saved_settings_dict can be passed to subsequent calls for restoration.
        """
        if span_multiplier is None:
            span_multiplier = float(self._sfdr_span_multiplier)

        spectrum = self._spectrum_analyzer()

        # Save current SA settings before modifying (for later restoration)
        if saved_settings is None:
            saved_settings = {
                'center_frequency': spectrum.center_frequency,
                'span': spectrum.span,
                'resolution_bandwidth': spectrum.resolution_bandwidth
            }

        # Calculate span: 4.1 × IF × span_multiplier (matches original)
        base_span = 4.1 * if_frequency_hz
        span_hz = base_span * span_multiplier

        # Configure spectrum analyzer for wideband SFDR measurement
        spectrum.center_frequency = lo_frequency_hz
        spectrum.span = span_hz
        # Set explicit RBW for SFDR sweeps (Task D - matches legacy sweepBW)
        spectrum.resolution_bandwidth = float(self._sfdr_sweep_rbw_hz)

        # Trigger acquisition and get spectrum
        spectrum.trigger_single()
        freq_hz, amp_dbm = spectrum.get_spectrum()

        # Restore SA settings if requested
        if restore_settings:
            spectrum.center_frequency = saved_settings['center_frequency']
            spectrum.span = saved_settings['span']
            spectrum.resolution_bandwidth = saved_settings['resolution_bandwidth']

        return np.array(freq_hz), np.array(amp_dbm), saved_settings

    def _calculate_sfdr(
            self,
            freq_hz: np.ndarray,
            amp_dbm: np.ndarray,
            lo_frequency_hz: float,
            if_frequency_hz: float,
            sideband: str = 'upper'
    ) -> Tuple[float, dict]:
        """Calculate SFDR (Spurious-Free Dynamic Range) from spectrum data.

        Ported from legacy IQ_calibrator_pyrpl.py:296-415.

        Args:
            freq_hz: Frequency array in Hz
            amp_dbm: Amplitude array in dBm
            lo_frequency_hz: LO frequency in Hz
            if_frequency_hz: IF frequency in Hz
            sideband: 'upper' or 'lower' sideband selection

        Returns:
            Tuple of (sfdr_db, details_dict) where details_dict contains:
            - target_freq_hz: Actual target frequency found
            - target_power_dbm: Target signal power
            - spur_freq_hz: Frequency of maximum spur
            - spur_power_dbm: Power of maximum spur
            - spur_label: Harmonic identification (e.g., 'LO', 'LO+2IF')
        """
        sideband = self._normalize_sideband(sideband)

        # Calculate target frequency based on sideband
        if sideband == 'upper':
            target_freq = lo_frequency_hz + if_frequency_hz
        else:
            target_freq = lo_frequency_hz - if_frequency_hz

        # Search bandwidth: sfdr_bandwidth_factor × IF (default 8 × IF)
        bandwidth_factor = float(self._sfdr_bandwidth_factor)
        bandwidth_hz = bandwidth_factor * if_frequency_hz

        # Extract band-limited data around target
        freq_min = target_freq - bandwidth_hz / 2
        freq_max = target_freq + bandwidth_hz / 2
        band_mask = (freq_hz >= freq_min) & (freq_hz <= freq_max)

        if not np.any(band_mask):
            self.log.warning('SFDR calculation: No data points in search band')
            return float('nan'), {}

        freq_band = freq_hz[band_mask]
        amp_band = amp_dbm[band_mask]

        # Mask NaN/Inf values for robust peak finding
        finite_mask = np.isfinite(amp_band)
        if not np.any(finite_mask):
            self.log.warning('SFDR calculation: No finite amplitude values in band')
            return float('nan'), {}

        # Find target signal peak within search range of expected frequency
        # Scale search window: use min(IF, 5 MHz) to handle small IF values
        # and exclude LO/image from being picked as target
        search_range_hz = min(if_frequency_hz * 0.8, 5e6)
        search_mask = (np.abs(freq_band - target_freq) <= search_range_hz) & finite_mask

        if not np.any(search_mask):
            self.log.warning('SFDR calculation: No data points near target frequency')
            return float('nan'), {}

        search_indices = np.where(search_mask)[0]
        max_idx_in_search = np.nanargmax(amp_band[search_mask])
        target_idx = search_indices[max_idx_in_search]

        target_power = amp_band[target_idx]
        actual_target_freq = freq_band[target_idx]

        # Exclude region around target peak (±IF/2)
        target_exclusion = if_frequency_hz / 2
        spur_mask = (np.abs(freq_band - actual_target_freq) > target_exclusion) & finite_mask

        if not np.any(spur_mask):
            # No spurs found outside exclusion zone
            self.log.info('SFDR calculation: No spurs found outside exclusion zone')
            return float('inf'), {
                'target_freq_hz': actual_target_freq,
                'target_power_dbm': target_power,
                'spur_freq_hz': None,
                'spur_power_dbm': None,
                'spur_label': 'none'
            }

        # Find maximum spur (use nanargmax for robustness)
        spur_amps = amp_band[spur_mask]
        spur_freqs = freq_band[spur_mask]
        max_spur_idx = np.nanargmax(spur_amps)
        max_spur_power = spur_amps[max_spur_idx]
        max_spur_freq = spur_freqs[max_spur_idx]

        # Calculate SFDR
        sfdr_db = target_power - max_spur_power

        # Identify harmonic origin of maximum spur
        harmonic_freqs = {
            'LO': lo_frequency_hz,
            'LO+IF': lo_frequency_hz + if_frequency_hz,
            'LO-IF': lo_frequency_hz - if_frequency_hz,
            'LO+2IF': lo_frequency_hz + 2 * if_frequency_hz,
            'LO-2IF': lo_frequency_hz - 2 * if_frequency_hz,
            'LO+3IF': lo_frequency_hz + 3 * if_frequency_hz,
            'LO-3IF': lo_frequency_hz - 3 * if_frequency_hz,
        }

        # Find closest harmonic within 5 MHz
        closest_harmonic = 'unknown'
        min_distance = float('inf')
        for label, freq in harmonic_freqs.items():
            distance = abs(max_spur_freq - freq)
            if distance < min_distance and distance < 5e6:
                min_distance = distance
                closest_harmonic = label

        self.log.debug(
            f'SFDR: {sfdr_db:.1f} dB | Target: {actual_target_freq/1e9:.6f} GHz @ {target_power:.1f} dBm | '
            f'Max spur: {max_spur_freq/1e9:.6f} GHz @ {max_spur_power:.1f} dBm ({closest_harmonic})'
        )

        return sfdr_db, {
            'target_freq_hz': actual_target_freq,
            'target_power_dbm': target_power,
            'spur_freq_hz': max_spur_freq,
            'spur_power_dbm': max_spur_power,
            'spur_label': closest_harmonic
        }

    def _plot_calibration_spectrum(
            self,
            freq_before_hz: np.ndarray,
            amp_before_dbm: np.ndarray,
            freq_after_hz: np.ndarray,
            amp_after_dbm: np.ndarray,
            sfdr_result: dict,
            lo_frequency_hz: float,
            if_frequency_hz: float,
            if_amplitude: float,
            output_path: str
    ) -> None:
        """Generate calibration spectrum PDF with SFDR annotations.

        Creates a comparison plot showing spectrum before and after calibration,
        with markers for target signal, maximum spur, and SFDR measurement.

        Uses OO matplotlib API for thread safety (no pyplot global state).

        Ported from legacy IQ_calibrator_pyrpl.py:417-485.

        Args:
            freq_before_hz: Frequency array for before spectrum (Hz)
            amp_before_dbm: Amplitude before calibration in dBm
            freq_after_hz: Frequency array for after spectrum (Hz)
            amp_after_dbm: Amplitude after calibration in dBm
            sfdr_result: Dict from _calculate_sfdr() with target/spur info
            lo_frequency_hz: LO frequency in Hz
            if_frequency_hz: IF frequency in Hz
            if_amplitude: IF amplitude used for calibration
            output_path: Full path for output PDF file
        """
        # Lazy import matplotlib to avoid backend conflicts
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = None
        try:
            # Validate input arrays
            if (freq_before_hz is None or len(freq_before_hz) == 0 or
                    freq_after_hz is None or len(freq_after_hz) == 0):
                self.log.warning('Cannot generate spectrum plot: empty frequency arrays')
                return

            # Create figure using OO API (thread-safe, no pyplot global state)
            fig = Figure(figsize=(12, 8))
            FigureCanvasAgg(fig)  # Required for rendering
            ax = fig.add_subplot(111)

            # Convert to GHz for plotting
            freq_before_ghz = freq_before_hz / 1e9
            freq_after_ghz = freq_after_hz / 1e9

            # Plot before/after spectra (on their respective frequency grids)
            ax.plot(freq_before_ghz, amp_before_dbm, alpha=0.5, linewidth=1, label='Before calibration')
            ax.plot(freq_after_ghz, amp_after_dbm, alpha=0.7, linewidth=1, label='After calibration')

            # Extract SFDR details (with guards for missing fields)
            target_freq = sfdr_result.get('target_freq_hz') if sfdr_result else None
            target_power = sfdr_result.get('target_power_dbm') if sfdr_result else None
            spur_freq = sfdr_result.get('spur_freq_hz') if sfdr_result else None
            spur_power = sfdr_result.get('spur_power_dbm') if sfdr_result else None
            spur_label = sfdr_result.get('spur_label', 'unknown') if sfdr_result else 'unknown'

            # Mark target signal (green)
            if target_freq is not None and target_power is not None:
                ax.axvline(target_freq / 1e9, color='green', linestyle='--', alpha=0.5, linewidth=1)
                ax.plot(target_freq / 1e9, target_power, 'go', markersize=10, label='Target signal')

            # Mark maximum spur (red)
            if spur_freq is not None and spur_power is not None:
                ax.plot(spur_freq / 1e9, spur_power, 'ro', markersize=8, label=f'Max spur ({spur_label})')

                # Draw SFDR measurement line
                if target_freq is not None and target_power is not None:
                    ax.plot(
                        [target_freq / 1e9, spur_freq / 1e9],
                        [target_power, spur_power],
                        'r--', alpha=0.5, linewidth=1
                    )

            # Shade SFDR bandwidth region
            sfdr_bandwidth_mhz = (float(self._sfdr_bandwidth_factor) * if_frequency_hz) / 1e6
            if target_freq is not None:
                bw_half_ghz = sfdr_bandwidth_mhz / 2 / 1e3
                ax.axvspan(
                    target_freq / 1e9 - bw_half_ghz,
                    target_freq / 1e9 + bw_half_ghz,
                    alpha=0.1, color='gray', label=f'SFDR bandwidth (±{sfdr_bandwidth_mhz/2:.1f} MHz)'
                )

            # Mark expected harmonic locations (orange dotted lines)
            harmonic_freqs = [
                (lo_frequency_hz, 'LO'),
                (lo_frequency_hz + if_frequency_hz, 'LO+IF'),
                (lo_frequency_hz - if_frequency_hz, 'LO-IF'),
                (lo_frequency_hz + 2 * if_frequency_hz, 'LO+2IF'),
                (lo_frequency_hz - 2 * if_frequency_hz, 'LO-2IF'),
                (lo_frequency_hz + 3 * if_frequency_hz, 'LO+3IF'),
                (lo_frequency_hz - 3 * if_frequency_hz, 'LO-3IF'),
            ]

            # Get frequency range from after spectrum for harmonic line visibility
            freq_min_ghz = min(freq_after_ghz.min(), freq_before_ghz.min())
            freq_max_ghz = max(freq_after_ghz.max(), freq_before_ghz.max())

            for freq, label in harmonic_freqs:
                freq_ghz_val = freq / 1e9
                if freq_min_ghz <= freq_ghz_val <= freq_max_ghz:
                    ax.axvline(freq_ghz_val, color='orange', linestyle=':', alpha=0.3, linewidth=0.8)

            # Calculate SFDR for title (with guards)
            if target_power is not None and spur_power is not None:
                sfdr_db = target_power - spur_power
            else:
                sfdr_db = float('nan')

            # Build title with guards for None values
            if target_freq is not None:
                title_target = f'{target_freq/1e9:.4f} GHz'
            else:
                title_target = 'N/A'

            ax.set_title(
                f'IQ-Calibration Spectrum | Target: {title_target}\n'
                f'SFDR: {sfdr_db:.1f} dB | IF amplitude: {if_amplitude:.3f} | LO: {lo_frequency_hz/1e9:.3f} GHz'
            )
            ax.set_xlabel('Frequency (GHz)')
            ax.set_ylabel('Amplitude (dBm)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

            fig.tight_layout()

            # Save to PDF
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            fig.savefig(output_path, format='pdf', dpi=150, bbox_inches='tight')
            self.log.info(f'Saved calibration spectrum plot to {output_path}')

        except Exception as e:
            self.log.warning(f'Failed to generate spectrum plot: {e}')
        finally:
            # Clean up figure (OO API doesn't need plt.close())
            if fig is not None:
                fig.clear()

    def calibrate_single_point(
            self,
            lo_freq_ghz: float,
            if_amplitude: float,
            if_frequency_hz: Optional[float] = None,
            initial_dc_offsets: Optional[Tuple[float, float]] = None,
            initial_iq_correction: Optional[Tuple[float, float]] = None,
            output_dir: Optional[str] = None
    ) -> CalibrationPoint:
        """Run full calibration for a single (LO, amplitude) point.

        This performs:
        1. Captures "before" spectrum (if SFDR measurement enabled)
        2. DC offset optimization (minimize LO leakage)
        3. IQ imbalance optimization (minimize image power)
        4. Captures "after" spectrum and calculates SFDR (if enabled)
        5. Generates comparison PDF plot (if enabled and output_dir provided)

        Args:
            lo_freq_ghz: LO frequency in GHz
            if_amplitude: IF signal amplitude (0-1)
            if_frequency_hz: IF frequency in Hz (uses default if None)
            initial_dc_offsets: Optional initial guess for (I_offset, Q_offset)
            initial_iq_correction: Optional initial guess for (g, phi)
            output_dir: Directory for saving spectrum plots (if None, plots not saved)

        Returns:
            CalibrationPoint with optimized parameters and SFDR measurement
        """
        if if_frequency_hz is None:
            if_frequency_hz = self._if_frequency_hz

        with self._thread_lock:
            mw = self._microwave()
            spectrum = self._spectrum_analyzer()

            sideband = self._normalize_sideband(self._sideband)

            # Ensure calibration is performed with a single IF tone.
            #
            # The RedPitaya ifsynth supports 3 simultaneous components; calibration must use only one.
            # We enforce and verify the microwave module's "single" mode so its `set_cw()` call
            # configures only one IF component on the RedPitaya.
            set_multi_frequency_mode = getattr(mw, 'set_multi_frequency_mode', None)
            if not callable(set_multi_frequency_mode):
                raise RuntimeError(
                    'Microwave module does not implement set_multi_frequency_mode(); '
                    'cannot guarantee single-tone IQ calibration.'
                )

            get_multi_frequency_info = getattr(mw, 'get_multi_frequency_info', None)
            if not callable(get_multi_frequency_info):
                raise RuntimeError(
                    'Microwave module does not implement get_multi_frequency_info(); '
                    'cannot verify single-tone IQ calibration.'
                )

            previous_mode = None
            try:
                previous_mode = str(get_multi_frequency_info().get('mode', '')).strip().lower()
            except Exception:
                previous_mode = None

            mode_forced = False
            try:
                set_multi_frequency_mode('single')
                mode_forced = True

                multi_freq_info = get_multi_frequency_info()
                if not isinstance(multi_freq_info, dict):
                    raise RuntimeError(
                        f'get_multi_frequency_info() returned {type(multi_freq_info).__name__}, expected dict.'
                    )

                actual_mode = str(multi_freq_info.get('mode', '')).strip().lower()
                active_if_frequencies = multi_freq_info.get('active_if_frequencies', None)
                if actual_mode != 'single' or not isinstance(active_if_frequencies, list) or len(active_if_frequencies) != 1:
                    raise RuntimeError(
                        'IQ calibration requires multi_frequency_mode="single" with exactly one active IF frequency, '
                        f'but got mode="{actual_mode}", active_if_frequencies={active_if_frequencies!r}.'
                    )

                # Calculate frequencies
                lo_freq_hz = lo_freq_ghz * 1e9
                if sideband == 'upper':
                    image_freq_hz = lo_freq_hz - if_frequency_hz
                else:
                    image_freq_hz = lo_freq_hz + if_frequency_hz

                self.log.info(
                    f'Calibrating: LO={lo_freq_ghz:.4f} GHz, IF amp={if_amplitude:.3f}, sideband={sideband.upper()}'
                )

                # Configure sideband selection (if supported by microwave module)
                set_sideband = getattr(mw, 'calibration_set_sideband', None)
                if callable(set_sideband):
                    set_sideband(sideband)

                # Configure microwave source for CW at LO frequency.
                #
                # Note: mw.set_cw() expects an RF frequency. The microwave module computes its LO frequency
                # from RF and its internal average IF frequency. During calibration, we want to set the LO to
                # the exact `lo_freq_hz` given here. We therefore compute the RF frequency argument such that
                # the microwave module will set LO=lo_freq_hz.
                avg_if_hz = float(multi_freq_info.get('average_if_frequency', if_frequency_hz))

                rf_set_hz = lo_freq_hz + avg_if_hz if sideband == 'upper' else lo_freq_hz - avg_if_hz

                # The `power` argument here does not control Windfreak LO power (that is configured in the
                # microwave hardware module via `lo_power`). It only affects the microwave module's IF amplitude
                # calculation. We set it consistently with the requested `if_amplitude` and clamp it to the
                # microwave module's constraints.
                if if_amplitude <= 0:
                    power_dbm = float(mw.constraints.power_limits[0])
                else:
                    power_dbm = float(20.0 * np.log10(if_amplitude) + 10.0)
                    min_p, max_p = mw.constraints.power_limits
                    power_dbm = float(np.clip(power_dbm, min_p, max_p))

                mw.set_cw(frequency=rf_set_hz, power=power_dbm)

                # Set IF frequency and amplitude
                mw.calibration_set_if_frequency(if_frequency_hz, component_index=0)
                mw.calibration_set_if_amplitude(if_amplitude, component_index=0)

                # Enable outputs
                mw.calibration_enable_output(True)
                mw.cw_on()

                # Configure spectrum analyzer with explicit RBW and span (Task B)
                self._setup_spectrum_measurement(lo_freq_hz, if_frequency_hz)

                # Set up markers for LO leakage and image
                spectrum.set_marker_frequency(1, lo_freq_hz)        # LO leakage
                spectrum.set_marker_frequency(2, image_freq_hz)     # Image

                # Verify signal is present before starting optimization (Task C)
                signal_ok, initial_signal_power = self._verify_signal_present(
                    lo_freq_hz, if_frequency_hz, sideband
                )
                if not signal_ok and self._abort_on_weak_signal:
                    raise RuntimeError(
                        f'Signal too weak ({initial_signal_power:.1f} dBm) at expected tone frequency - '
                        f'aborting calibration. Check hardware connections and output enable.'
                    )

                # Capture "before" spectrum for SFDR comparison (if enabled)
                # Save SA settings before SFDR acquisition to restore after
                sfdr_freq_before = None
                amp_before = None
                sa_settings_before_sfdr = None
                if self._enable_sfdr_measurement:
                    self.log.debug('Capturing pre-calibration spectrum for SFDR measurement')
                    sfdr_freq_before, amp_before, sa_settings_before_sfdr = self._acquire_sfdr_spectrum(
                        lo_freq_hz, if_frequency_hz, restore_settings=True
                    )

                self._current_if_amplitude = if_amplitude
                self._measurement_counter = 0

                best_result = None
                best_reward = float('inf')

                if initial_dc_offsets is None:
                    i_offset, q_offset = 0.0, 0.0
                else:
                    i_offset, q_offset = float(initial_dc_offsets[0]), float(initial_dc_offsets[1])

                if initial_iq_correction is None:
                    g, phi = 0.0, 0.0
                else:
                    g, phi = float(initial_iq_correction[0]), float(initial_iq_correction[1])

                mw.calibration_set_dc_offsets(i_offset, q_offset)
                mw.calibration_set_iq_correction(g, phi, if_amplitude, component_index=0)

                for iteration in range(self._optimization_iterations):
                    if self._abort_requested:
                        self.log.warning('Calibration aborted')
                        return None

                    self.log.debug(f'Optimization iteration {iteration + 1}/{self._optimization_iterations}')

                    # Step 1: Optimize DC offsets (minimize LO leakage)
                    mw.calibration_set_iq_correction(g, phi, if_amplitude, component_index=0)
                    dc_simplex = self._initial_simplex_2d(
                        (i_offset, q_offset),
                        step_x=self._dc_simplex_step,
                        step_y=self._dc_simplex_step
                    )
                    dc_result = minimize(
                        self._measure_lo_leakage,
                        x0=np.array([i_offset, q_offset]),
                        method='Nelder-Mead',
                        options={
                            'xatol': self._xatol,
                            'fatol': self._fatol,
                            'maxiter': self._maxiter,
                            'initial_simplex': dc_simplex
                        }
                    )

                    self.log.debug(f'DC optimization: I={dc_result.x[0]:.5f}, Q={dc_result.x[1]:.5f}, '
                                   f'LO leakage={dc_result.fun:.2f} dBm')

                    # Apply best DC offsets
                    i_offset, q_offset = float(dc_result.x[0]), float(dc_result.x[1])
                    mw.calibration_set_dc_offsets(i_offset, q_offset)

                    # Step 2: Optimize g/phi (minimize image)
                    iq_simplex = self._initial_simplex_2d(
                        (g, phi),
                        step_x=self._iq_g_simplex_step,
                        step_y=self._iq_phi_simplex_step
                    )
                    iq_result = minimize(
                        lambda x: self._measure_image_rejection(x, if_amplitude),
                        x0=np.array([g, phi]),
                        method='Nelder-Mead',
                        options={
                            'xatol': self._xatol,
                            'fatol': self._fatol,
                            'maxiter': self._maxiter,
                            'initial_simplex': iq_simplex
                        }
                    )

                    self.log.debug(f'IQ optimization: g={iq_result.x[0]:.5f}, phi={iq_result.x[1]:.5f}, '
                                   f'Image power={iq_result.fun:.2f} dBm')

                    # Apply best IQ correction (important for the next DC iteration)
                    g, phi = float(iq_result.x[0]), float(iq_result.x[1])
                    mw.calibration_set_iq_correction(g, phi, if_amplitude, component_index=0)

                    # Evaluate combined result (lower is better).
                    #
                    # The individual optimizers minimize marker power in dB (typically dBm / dBFS, often negative).
                    # To select the best outer iteration, we combine LO leakage + image in *linear* power and
                    # minimize their sum.
                    lo_leakage_db = float(dc_result.fun)
                    image_power_db = float(iq_result.fun)
                    if np.isfinite(lo_leakage_db) and np.isfinite(image_power_db):
                        reward = (10.0 ** (lo_leakage_db / 10.0)) + (10.0 ** (image_power_db / 10.0))
                    else:
                        reward = float('inf')

                    if best_result is None or reward < best_reward:
                        best_reward = reward
                        best_result = CalibrationPoint(
                            lo_frequency_ghz=lo_freq_ghz,
                            if_amplitude=if_amplitude,
                            I_offset=i_offset,
                            Q_offset=q_offset,
                            g=g,
                            phi=phi,
                            lo_leakage_dbm=lo_leakage_db,
                            image_power_dbm=image_power_db
                        )

                # Turn off microwave output
                mw.off()

                self.log.info(f'Calibration complete: LO leakage={best_result.lo_leakage_dbm:.2f} dBm, '
                              f'Image={best_result.image_power_dbm:.2f} dBm')

                # Ensure the final hardware state matches the returned result
                mw.calibration_set_dc_offsets(best_result.I_offset, best_result.Q_offset)
                mw.calibration_set_iq_correction(best_result.g, best_result.phi, if_amplitude, component_index=0)

                # SFDR measurement (if enabled)
                # Wrapped in try/except so SFDR failures don't fail the whole calibration
                sfdr_db = None
                sfdr_details = {}
                sfdr_freq_after = None
                amp_after = None
                if self._enable_sfdr_measurement:
                    try:
                        self.log.debug('Capturing post-calibration spectrum for SFDR measurement')

                        # Re-enable output briefly for SFDR measurement
                        mw.calibration_enable_output(True)
                        mw.cw_on()

                        # Capture "after" spectrum with optimal calibration applied
                        # Use restore_settings=True to restore SA to optimization settings
                        sfdr_freq_after, amp_after, _ = self._acquire_sfdr_spectrum(
                            lo_freq_hz, if_frequency_hz,
                            restore_settings=True,
                            saved_settings=sa_settings_before_sfdr
                        )

                        # Calculate SFDR from "after" spectrum using its own frequency array
                        sfdr_db, sfdr_details = self._calculate_sfdr(
                            sfdr_freq_after, amp_after, lo_freq_hz, if_frequency_hz, sideband
                        )

                        if np.isfinite(sfdr_db):
                            self.log.info(f'SFDR: {sfdr_db:.1f} dB')
                        else:
                            self.log.warning('SFDR calculation returned non-finite value')

                        # Update best_result with SFDR
                        best_result = CalibrationPoint(
                            lo_frequency_ghz=best_result.lo_frequency_ghz,
                            if_amplitude=best_result.if_amplitude,
                            I_offset=best_result.I_offset,
                            Q_offset=best_result.Q_offset,
                            g=best_result.g,
                            phi=best_result.phi,
                            lo_leakage_dbm=best_result.lo_leakage_dbm,
                            image_power_dbm=best_result.image_power_dbm,
                            sfdr_db=sfdr_db
                        )

                        # Generate comparison plot (if enabled and output_dir provided)
                        # Plot uses before/after frequency arrays independently
                        if (self._save_spectrum_plots and
                                amp_before is not None and sfdr_freq_before is not None and
                                amp_after is not None and sfdr_freq_after is not None and
                                output_dir is not None):
                            plot_filename = f'spectrum_if_{if_amplitude:.3f}_lo_{lo_freq_ghz:.3f}_GHz.pdf'
                            plot_path = os.path.join(output_dir, plot_filename)
                            self._plot_calibration_spectrum(
                                sfdr_freq_before, amp_before,
                                sfdr_freq_after, amp_after,
                                sfdr_details,
                                lo_freq_hz, if_frequency_hz, if_amplitude, plot_path
                            )

                        # Emit SFDR signal
                        if sfdr_db is not None:
                            self.sigSFDRUpdated.emit(sfdr_db, sfdr_details)

                    except Exception as e:
                        self.log.warning(f'SFDR measurement failed (calibration result still valid): {e}')
                        sfdr_db = None
                        sfdr_details = {}
                    finally:
                        # Turn off output after SFDR measurement
                        try:
                            mw.off()
                        except Exception:
                            pass

                # Emit signal with result (including SFDR)
                result_dict = {
                    'sideband': sideband,
                    'lo_frequency_ghz': best_result.lo_frequency_ghz,
                    'if_amplitude': best_result.if_amplitude,
                    'I_offset': best_result.I_offset,
                    'Q_offset': best_result.Q_offset,
                    'g': best_result.g,
                    'phi': best_result.phi,
                    'lo_leakage_dbm': best_result.lo_leakage_dbm,
                    'image_power_dbm': best_result.image_power_dbm,
                    'sfdr_db': best_result.sfdr_db
                }
                if sfdr_details:
                    result_dict['spur_freq_hz'] = sfdr_details.get('spur_freq_hz')
                    result_dict['spur_power_dbm'] = sfdr_details.get('spur_power_dbm')
                    result_dict['spur_label'] = sfdr_details.get('spur_label')

                self.sigCalibrationPointComplete.emit(result_dict)

                return best_result
            finally:
                if mode_forced:
                    try:
                        mw.off()
                    except Exception:
                        pass

                if mode_forced and previous_mode and previous_mode != 'single':
                    try:
                        set_multi_frequency_mode(previous_mode)
                    except Exception as e:
                        self.log.debug(f'Could not restore multi-frequency mode to "{previous_mode}": {e}')

    def run_full_calibration(
            self,
            lo_frequencies_ghz: List[float],
            if_amplitudes: List[float],
            if_frequency_hz: Optional[float] = None
    ) -> str:
        """Run calibration sweep over LO frequencies and IF amplitudes.

        Args:
            lo_frequencies_ghz: List of LO frequencies to calibrate
            if_amplitudes: List of IF amplitudes to calibrate
            if_frequency_hz: IF frequency in Hz (uses default if None)

        Returns:
            Path to the output CSV file
        """
        if if_frequency_hz is None:
            if_frequency_hz = self._if_frequency_hz

        self._abort_requested = False
        results = []
        total_points = len(lo_frequencies_ghz) * len(if_amplitudes)
        current_point = 0

        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if_mhz = if_frequency_hz / 1e6
        output_dir = os.path.join(
            self._calibration_output_dir,
            timestamp,
            f'IF_{if_mhz:.3f}MHz'
        )
        os.makedirs(output_dir, exist_ok=True)

        self.log.info(f'Starting calibration: {total_points} points, output: {output_dir}')

        # Warm-start across the calibration grid (ported from legacy implementation).
        last_dc_offsets: Optional[Tuple[float, float]] = None
        last_iq_correction: Optional[Tuple[float, float]] = None
        failed_points: List[Dict[str, object]] = []

        try:
            for lo_ghz in lo_frequencies_ghz:
                for amplitude in if_amplitudes:
                    if self._abort_requested:
                        self.log.warning('Calibration aborted by user')
                        break

                    current_point += 1
                    progress = current_point / total_points * 100

                    status_msg = f'Calibrating LO={lo_ghz:.3f} GHz, amp={amplitude:.3f} ({current_point}/{total_points})'
                    self.sigCalibrationProgress.emit(progress, status_msg)

                    initial_dc = last_dc_offsets if self._warm_start else None
                    initial_iq = last_iq_correction if self._warm_start else None

                    result = None
                    for attempt in range(int(self._point_retry_count) + 1):
                        if self._abort_requested:
                            break
                        try:
                            result = self.calibrate_single_point(
                                lo_ghz,
                                amplitude,
                                if_frequency_hz,
                                initial_dc_offsets=initial_dc,
                                initial_iq_correction=initial_iq,
                                output_dir=output_dir  # For SFDR spectrum plots
                            )
                            break
                        except Exception as e:
                            if attempt >= int(self._point_retry_count):
                                msg = (
                                    f'Calibration failed for LO={lo_ghz:.6f} GHz, amp={amplitude:.6f} '
                                    f'(attempt {attempt + 1}/{int(self._point_retry_count) + 1}): {e}'
                                )
                                self.log.error(msg)
                                self.sigCalibrationError.emit(msg)
                                failed_points.append({
                                    'lo_frequency_ghz': lo_ghz,
                                    'if_amplitude': amplitude,
                                    'if_frequency_hz': if_frequency_hz,
                                    'sideband': self._normalize_sideband(self._sideband),
                                    'error': str(e)
                                })
                                result = None
                            else:
                                self.log.warning(
                                    f'Point failed (attempt {attempt + 1}/{int(self._point_retry_count) + 1}): {e}'
                                )
                                time.sleep(float(self._point_retry_delay_s))

                    if result:
                        results.append({
                            'sideband': self._normalize_sideband(self._sideband),
                            'base_if_frequency_mhz': if_mhz,
                            'lo_frequency_ghz': result.lo_frequency_ghz,
                            'if_amplitude': result.if_amplitude,
                            'g': result.g,
                            'phi': result.phi,
                            'I_offset': result.I_offset,
                            'Q_offset': result.Q_offset,
                            'lo_leakage_dbm': result.lo_leakage_dbm,
                            'image_power_dbm': result.image_power_dbm,
                            'sfdr_db': result.sfdr_db  # SFDR from spectrum measurement
                        })

                        if self._warm_start:
                            last_dc_offsets = (float(result.I_offset), float(result.Q_offset))
                            last_iq_correction = (float(result.g), float(result.phi))

                if self._abort_requested:
                    break

            # Save results to CSV
            output_file = os.path.join(
                output_dir,
                f'calibration_redpitaya_all_results_IF_{if_mhz:.3f}MHz.csv'
            )
            df = pd.DataFrame(results)
            df.to_csv(output_file, sep='\t', index=True)

            # Save a separate failure log (does not affect the calibration file format).
            if failed_points:
                failed_points_file = os.path.join(output_dir, 'calibration_failed_points.tsv')
                pd.DataFrame(failed_points).to_csv(failed_points_file, sep='\t', index=False)

            # Update status variables
            self._calibration_results[if_frequency_hz] = results
            self._last_calibration_time = timestamp
            self._current_calibration_file = output_file

            self.sigCalibrationComplete.emit(output_file)
            self.log.info(f'Calibration complete. Results saved to: {output_file}')

            return output_file

        except Exception as e:
            error_msg = f'Calibration failed: {e}'
            self.log.error(error_msg)
            self.sigCalibrationError.emit(error_msg)
            raise

    def abort_calibration(self) -> None:
        """Request calibration abort."""
        self._abort_requested = True
        self.log.info('Calibration abort requested')

    # ─────────────────────────────────────────────────────────────────────────
    # Incremental Saving and Resume Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _append_result_to_csv(
            self,
            result_dict: Dict,
            csv_path: str,
            write_header: bool = False
    ) -> None:
        """Append a single calibration result to CSV file.

        This method enables incremental saving of calibration results,
        writing each point to disk immediately after completion to prevent
        data loss in case of crashes during long calibration runs.

        Args:
            result_dict: Single calibration point result dictionary
            csv_path: Path to CSV file
            write_header: If True, write header row first (for new files)
        """
        df_row = pd.DataFrame([result_dict])

        # Write with header if new file, append without header otherwise
        mode = 'w' if write_header else 'a'
        header = write_header

        df_row.to_csv(csv_path, sep='\t', index=True, mode=mode, header=header)

    def _load_checkpoint(
            self,
            csv_path: str
    ) -> Tuple[Set[Tuple[float, float]], Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Load checkpoint from existing partial CSV for resume functionality.

        Reads an existing calibration CSV file and extracts:
        - Set of completed (lo_frequency_ghz, if_amplitude) tuples
        - Last point's DC offsets for warm-start initialization
        - Last point's IQ correction parameters for warm-start initialization

        Args:
            csv_path: Path to partial calibration CSV file

        Returns:
            Tuple of:
            - completed_points: Set of (lo_ghz, amplitude) tuples already done
            - last_dc_offsets: (I_offset, Q_offset) from last point, or None
            - last_iq_correction: (g, phi) from last point, or None

        Raises:
            ValueError: If CSV exists but is corrupted/unreadable
        """
        completed_points: Set[Tuple[float, float]] = set()
        last_dc: Optional[Tuple[float, float]] = None
        last_iq: Optional[Tuple[float, float]] = None

        if not os.path.exists(csv_path):
            return completed_points, last_dc, last_iq

        try:
            df = pd.read_csv(csv_path, sep='\t', index_col=0)
            if df.empty:
                return completed_points, last_dc, last_iq

            # Build set of completed (lo_ghz, amplitude) tuples
            # Use rounding to avoid floating-point comparison issues
            for _, row in df.iterrows():
                lo = round(float(row['lo_frequency_ghz']), 6)
                amp = round(float(row['if_amplitude']), 6)
                completed_points.add((lo, amp))

            # Get warm-start values from last row
            last_row = df.iloc[-1]
            last_dc = (float(last_row['I_offset']), float(last_row['Q_offset']))
            last_iq = (float(last_row['g']), float(last_row['phi']))

            self.log.info(f'Loaded checkpoint: {len(completed_points)} completed points from {csv_path}')
            return completed_points, last_dc, last_iq

        except Exception as e:
            raise ValueError(f'Failed to load checkpoint from {csv_path}: {e}')

    def run_multi_if_calibration(
            self,
            if_frequencies_hz: List[float],
            lo_frequencies_ghz: List[float],
            if_amplitudes: List[float],
            resume_from_checkpoint: bool = False,
            output_dir: Optional[str] = None
    ) -> str:
        """Run calibration for multiple IF frequencies in a single invocation.

        This is the main entry point for overnight calibration runs. It creates
        a timestamped output directory with subdirectories for each IF frequency,
        runs the full calibration grid for each IF, and produces a grand summary
        CSV aggregating results across all IFs.

        **Incremental Saving:** Each calibration point is saved to disk immediately
        after completion, preventing data loss in case of crashes during long runs.

        **Resume Capability:** If a calibration was interrupted, you can resume from
        the checkpoint by passing resume_from_checkpoint=True and the output_dir
        of the interrupted run.

        Typical IF frequencies for NV-ODMR experiments:
            - 19.322 MHz
            - 21.580 MHz
            - 23.738 MHz

        Args:
            if_frequencies_hz: List of IF frequencies to calibrate (in Hz)
            lo_frequencies_ghz: List of LO frequencies to sweep (in GHz)
            if_amplitudes: List of IF amplitudes to sweep (0-1)
            resume_from_checkpoint: If True, resume from existing partial results.
                Requires output_dir to be specified.
            output_dir: Directory for results. If None (default), creates new
                timestamped directory. Required when resume_from_checkpoint=True.

        Returns:
            Path to the main output directory containing all results

        Raises:
            ValueError: If resume_from_checkpoint=True but output_dir doesn't exist
            RuntimeError: If user aborts calibration (partial results are still saved)

        Example:
            >>> cal = qudi.module_manager.get('iq_calibration_logic').instance
            >>> if_freqs = [19.322e6, 21.580e6, 23.738e6]
            >>> lo_freqs = cal.get_default_lo_frequencies(center_ghz=2.87, span_ghz=0.3, num_points=61)
            >>> if_amps = cal.get_default_if_amplitudes(min_amp=0.01, max_amp=0.3, num_points=10)
            >>> # Normal calibration (with incremental saving):
            >>> output_dir = cal.run_multi_if_calibration(if_freqs, lo_freqs, if_amps)
            >>> # Resume after crash:
            >>> output_dir = cal.run_multi_if_calibration(
            ...     if_freqs, lo_freqs, if_amps,
            ...     resume_from_checkpoint=True,
            ...     output_dir='C:/calibration_results/2026-01-20-15-30-00'
            ... )
        """
        self._abort_requested = False

        # Handle output directory - either resume from existing or create new
        if resume_from_checkpoint and output_dir is not None:
            main_output_dir = output_dir
            if not os.path.exists(main_output_dir):
                raise ValueError(f'Resume requested but output_dir does not exist: {output_dir}')
            self.log.info(f'Resuming calibration in existing directory: {main_output_dir}')
        else:
            # Create new timestamped directory (default behavior)
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            main_output_dir = os.path.join(self._calibration_output_dir, timestamp)
            os.makedirs(main_output_dir, exist_ok=True)

        self.log.info(
            f'Starting multi-IF calibration: {len(if_frequencies_hz)} IF frequencies, '
            f'{len(lo_frequencies_ghz)} LO points × {len(if_amplitudes)} amplitudes each. '
            f'Output: {main_output_dir}'
        )

        # Track results and failures across all IFs
        all_if_results: List[Dict] = []
        per_if_csv_paths: List[str] = []
        failed_ifs: List[Dict] = []

        total_ifs = len(if_frequencies_hz)

        for if_idx, if_freq_hz in enumerate(if_frequencies_hz):
            if self._abort_requested:
                self.log.warning('Multi-IF calibration aborted by user')
                break

            if_mhz = if_freq_hz / 1e6

            # Emit progress with IF context
            progress_pct = (if_idx / total_ifs) * 100
            status_msg = f'IF {if_idx + 1}/{total_ifs}: {if_mhz:.3f} MHz - Starting calibration'
            self.sigCalibrationProgress.emit(progress_pct, status_msg)

            self.log.info(f'=== Starting IF frequency {if_idx + 1}/{total_ifs}: {if_mhz:.3f} MHz ===')

            # Create subdirectory for this IF
            if_subdir = os.path.join(main_output_dir, f'IF_{if_mhz:.3f}MHz')
            os.makedirs(if_subdir, exist_ok=True)

            # Define CSV path for incremental saving
            per_if_csv_path = os.path.join(
                if_subdir,
                f'calibration_redpitaya_all_results_IF_{if_mhz:.3f}MHz.csv'
            )

            # Reset warm-start parameters for each IF (like legacy does per IF)
            # This ensures each IF calibration starts fresh without carrying over
            # parameters from a different IF that may not be appropriate
            original_calibration_output_dir = self._calibration_output_dir

            try:
                self._abort_requested = False
                results = []
                total_points = len(lo_frequencies_ghz) * len(if_amplitudes)
                current_point = 0

                # Reset warm-start for this IF
                last_dc_offsets: Optional[Tuple[float, float]] = None
                last_iq_correction: Optional[Tuple[float, float]] = None
                failed_points: List[Dict] = []

                # Load checkpoint if resuming
                completed_points: Set[Tuple[float, float]] = set()
                if resume_from_checkpoint:
                    try:
                        completed_points, checkpoint_dc, checkpoint_iq = self._load_checkpoint(per_if_csv_path)
                        if completed_points:
                            self.log.info(f'Resuming IF {if_mhz:.3f} MHz: {len(completed_points)} points already done')
                            # Initialize warm-start from checkpoint
                            if self._warm_start and checkpoint_dc is not None:
                                last_dc_offsets = checkpoint_dc
                                last_iq_correction = checkpoint_iq
                    except ValueError as e:
                        self.log.warning(f'Checkpoint loading failed, starting fresh: {e}')
                        completed_points = set()

                # Track if this is the first new point (for CSV header)
                first_new_point = True

                for lo_ghz in lo_frequencies_ghz:
                    for amplitude in if_amplitudes:
                        if self._abort_requested:
                            self.log.warning(f'Calibration aborted during IF {if_mhz:.3f} MHz')
                            break

                        current_point += 1

                        # Skip if already completed (resume mode)
                        point_key = (round(lo_ghz, 6), round(amplitude, 6))
                        if point_key in completed_points:
                            self.log.debug(f'Skipping completed point: LO={lo_ghz:.4f} GHz, amp={amplitude:.4f}')
                            continue

                        overall_progress = ((if_idx + current_point / total_points) / total_ifs) * 100

                        status_msg = (
                            f'IF {if_mhz:.3f} MHz | LO={lo_ghz:.3f} GHz, amp={amplitude:.3f} '
                            f'({current_point}/{total_points})'
                        )
                        self.sigCalibrationProgress.emit(overall_progress, status_msg)

                        initial_dc = last_dc_offsets if self._warm_start else None
                        initial_iq = last_iq_correction if self._warm_start else None

                        result = None
                        for attempt in range(int(self._point_retry_count) + 1):
                            if self._abort_requested:
                                break
                            try:
                                result = self.calibrate_single_point(
                                    lo_ghz,
                                    amplitude,
                                    if_freq_hz,
                                    initial_dc_offsets=initial_dc,
                                    initial_iq_correction=initial_iq,
                                    output_dir=if_subdir
                                )
                                break
                            except Exception as e:
                                if attempt >= int(self._point_retry_count):
                                    msg = (
                                        f'Point failed for IF={if_mhz:.3f} MHz, LO={lo_ghz:.6f} GHz, '
                                        f'amp={amplitude:.6f}: {e}'
                                    )
                                    self.log.error(msg)
                                    self.sigCalibrationError.emit(msg)
                                    failed_points.append({
                                        'if_frequency_hz': if_freq_hz,
                                        'lo_frequency_ghz': lo_ghz,
                                        'if_amplitude': amplitude,
                                        'sideband': self._normalize_sideband(self._sideband),
                                        'error': str(e)
                                    })
                                    result = None
                                else:
                                    self.log.warning(
                                        f'Point failed (attempt {attempt + 1}/{int(self._point_retry_count) + 1}): {e}'
                                    )
                                    time.sleep(float(self._point_retry_delay_s))

                        if result:
                            result_dict = {
                                'sideband': self._normalize_sideband(self._sideband),
                                'if_frequency_hz': if_freq_hz,
                                'base_if_frequency_mhz': if_mhz,
                                'lo_frequency_ghz': result.lo_frequency_ghz,
                                'if_amplitude': result.if_amplitude,
                                'g': result.g,
                                'phi': result.phi,
                                'I_offset': result.I_offset,
                                'Q_offset': result.Q_offset,
                                'lo_leakage_dbm': result.lo_leakage_dbm,
                                'image_power_dbm': result.image_power_dbm,
                                'sfdr_db': result.sfdr_db
                            }
                            results.append(result_dict)
                            all_if_results.append(result_dict)

                            # Incremental save - append to CSV immediately
                            # Write header only for first new point (not resuming with existing data)
                            write_header = first_new_point and not os.path.exists(per_if_csv_path)
                            self._append_result_to_csv(result_dict, per_if_csv_path, write_header=write_header)
                            first_new_point = False

                            if self._warm_start:
                                last_dc_offsets = (float(result.I_offset), float(result.Q_offset))
                                last_iq_correction = (float(result.g), float(result.phi))

                    if self._abort_requested:
                        break

                # Per-IF results already saved incrementally to per_if_csv_path
                # Just track the path and update status variables
                if os.path.exists(per_if_csv_path):
                    per_if_csv_paths.append(per_if_csv_path)
                    self.log.info(f'Completed per-IF results: {per_if_csv_path} ({len(results)} new points)')

                # Update status variables for this IF
                if results:
                    self._calibration_results[if_freq_hz] = results

                # Save per-IF failure log
                if failed_points:
                    failed_file = os.path.join(if_subdir, 'calibration_failed_points.tsv')
                    pd.DataFrame(failed_points).to_csv(failed_file, sep='\t', index=False)

            except Exception as e:
                self.log.error(f'IF {if_mhz:.3f} MHz calibration failed: {e}')
                failed_ifs.append({
                    'if_frequency_hz': if_freq_hz,
                    'if_frequency_mhz': if_mhz,
                    'error': str(e)
                })
                # Continue to next IF (don't abort entire run)
                continue

        # Generate grand summary CSV across all IFs
        if all_if_results:
            grand_summary_path = os.path.join(
                main_output_dir,
                'calibration_redpitaya_GRAND_SUMMARY_all_IFs.csv'
            )
            grand_df = pd.DataFrame(all_if_results)
            grand_df.to_csv(grand_summary_path, sep='\t', index=True)
            self.log.info(f'Saved grand summary: {grand_summary_path} ({len(all_if_results)} total points)')

        # Save list of per-IF CSV paths for reference
        if per_if_csv_paths:
            index_path = os.path.join(main_output_dir, 'per_if_calibration_files.txt')
            with open(index_path, 'w') as f:
                for path in per_if_csv_paths:
                    f.write(f'{path}\n')

        # Save failed IFs log if any
        if failed_ifs:
            failed_ifs_path = os.path.join(main_output_dir, 'failed_if_frequencies.tsv')
            pd.DataFrame(failed_ifs).to_csv(failed_ifs_path, sep='\t', index=False)
            self.log.warning(f'{len(failed_ifs)} IF frequency calibrations failed. See {failed_ifs_path}')

        # Update status
        self._last_calibration_time = timestamp

        # Emit completion
        final_progress = 100.0 if not self._abort_requested else (if_idx / total_ifs) * 100
        self.sigCalibrationProgress.emit(final_progress, f'Multi-IF calibration complete: {main_output_dir}')
        self.sigCalibrationComplete.emit(main_output_dir)

        self.log.info(
            f'Multi-IF calibration complete. {len(all_if_results)} total points across '
            f'{total_ifs - len(failed_ifs)} IFs. Output: {main_output_dir}'
        )

        return main_output_dir

    # ─────────────────────────────────────────────────────────────────────────
    # Measurement Methods
    # ─────────────────────────────────────────────────────────────────────────

    def get_spectrum_snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current spectrum for display.

        Returns:
            tuple: (frequency_array, amplitude_array)
        """
        spectrum = self._spectrum_analyzer()
        freq_array, amp_array = spectrum.get_spectrum()
        self.sigSpectrumUpdated.emit(freq_array, amp_array)
        return freq_array, amp_array

    def measure_calibration_powers(self) -> Tuple[float, float]:
        """Measure LO leakage and image power with current settings.

        Returns:
            tuple: (lo_leakage_dbm, image_power_dbm)
        """
        spectrum = self._spectrum_analyzer()
        spectrum.trigger_single()
        lo_power = spectrum.get_marker_amplitude(1)
        image_power = spectrum.get_marker_amplitude(2)
        self.sigMeasurementUpdate.emit(lo_power, image_power)
        return lo_power, image_power

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration Data Management
    # ─────────────────────────────────────────────────────────────────────────

    def load_calibration_file(self, filepath: str) -> pd.DataFrame:
        """Load existing calibration data.

        Args:
            filepath: Path to calibration CSV file

        Returns:
            DataFrame with calibration data
        """
        df = pd.read_csv(filepath, sep='\t', index_col=0)
        self.log.info(f'Loaded calibration data: {len(df)} points from {filepath}')
        return df

    def get_calibration_parameters(
            self,
            filepath: str,
            lo_ghz: float,
            amplitude: float
    ) -> Tuple[float, float, float, float]:
        """Interpolate calibration parameters for given LO and amplitude.

        Args:
            filepath: Path to calibration CSV file
            lo_ghz: LO frequency in GHz
            amplitude: IF amplitude

        Returns:
            tuple: (g, phi, i_offset, q_offset)
        """
        from scipy.interpolate import griddata

        df = self.load_calibration_file(filepath)
        points = df[['lo_frequency_ghz', 'if_amplitude']].values
        query = np.array([[lo_ghz, amplitude]])

        g = griddata(points, df['g'].values, query, method='cubic')[0]
        phi = griddata(points, df['phi'].values, query, method='cubic')[0]
        i_off = griddata(points, df['I_offset'].values, query, method='cubic')[0]
        q_off = griddata(points, df['Q_offset'].values, query, method='cubic')[0]

        return g, phi, i_off, q_off

    def check_interpolation_bounds(
            self,
            filepath: str,
            lo_ghz: float,
            amplitude: float
    ) -> Tuple[bool, str]:
        """Check if a point is within the calibration data bounds.

        Args:
            filepath: Path to calibration CSV file
            lo_ghz: LO frequency in GHz
            amplitude: IF amplitude

        Returns:
            tuple: (in_bounds, message)
        """
        from scipy.spatial import Delaunay

        df = self.load_calibration_file(filepath)
        points = df[['lo_frequency_ghz', 'if_amplitude']].values

        try:
            hull = Delaunay(points)
            in_hull = hull.find_simplex([lo_ghz, amplitude]) >= 0

            if in_hull:
                return True, 'Point is within calibration bounds'
            else:
                return False, f'EXTRAPOLATING: Point ({lo_ghz}, {amplitude}) is outside calibration bounds'
        except Exception as e:
            return False, f'Could not check bounds: {e}'

    # ─────────────────────────────────────────────────────────────────────────
    # Configuration Methods
    # ─────────────────────────────────────────────────────────────────────────

    def get_default_lo_frequencies(
            self,
            center_ghz: float = 2.87,
            span_ghz: float = 0.3,
            num_points: int = 61
    ) -> List[float]:
        """Generate a default list of LO frequencies for calibration.

        Args:
            center_ghz: Center frequency in GHz
            span_ghz: Frequency span in GHz
            num_points: Number of frequency points

        Returns:
            List of LO frequencies in GHz
        """
        start = center_ghz - span_ghz / 2
        stop = center_ghz + span_ghz / 2
        return np.linspace(start, stop, num_points).tolist()

    def get_default_if_amplitudes(
            self,
            min_amp: float = 0.01,
            max_amp: float = 0.3,
            num_points: int = 10
    ) -> List[float]:
        """Generate a default list of IF amplitudes for calibration.

        Args:
            min_amp: Minimum amplitude
            max_amp: Maximum amplitude
            num_points: Number of amplitude points

        Returns:
            List of IF amplitudes
        """
        return np.linspace(min_amp, max_amp, num_points).tolist()

    @property
    def if_frequency_hz(self) -> float:
        """Get the current IF frequency."""
        return self._if_frequency_hz

    @if_frequency_hz.setter
    def if_frequency_hz(self, value: float) -> None:
        """Set the IF frequency."""
        self._if_frequency_hz = value

    @property
    def calibration_output_dir(self) -> str:
        """Get the calibration output directory."""
        return self._calibration_output_dir

    @calibration_output_dir.setter
    def calibration_output_dir(self, value: str) -> None:
        """Set the calibration output directory."""
        self._calibration_output_dir = value
        os.makedirs(value, exist_ok=True)

    @property
    def last_calibration_file(self) -> Optional[str]:
        """Get the path to the last calibration file."""
        return self._current_calibration_file

    @property
    def is_calibrating(self) -> bool:
        """Check if a calibration is currently running."""
        return self.module_state() == 'locked' and not self._abort_requested
