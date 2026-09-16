# -*- coding: utf-8 -*-
"""
Sensitivity Measurement Parameter Sweep Logic Module

This module orchestrates automated ODMR-based magnetic field sensitivity measurements
with systematic parameter sweeps over microwave power, FM modulation frequency,
and FM deviation.

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

__all__ = ['SensitivitySweepLogic']

import numpy as np
import pandas as pd
import time
import os
import json
from datetime import datetime
from itertools import product
from typing import Tuple, Dict, List, Optional, Any
from PySide2 import QtCore

from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.core.module import LogicBase
from qudi.util.mutex import RecursiveMutex
from qudi.util.datastorage import TextDataStorage, ImageFormat

# Import analysis functions from existing code
# Add parent directories to path to find my_software
import sys
import os
# Get path to qudi-core root (3 levels up from this file)
qudi_core_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if qudi_core_root not in sys.path:
    sys.path.insert(0, qudi_core_root)

from my_software.tools.fitting import fit_hyperfine
from my_software.sensitivity_msmt.auswertung.sensitivity_auswertung_modular import (
    magnetic_field_from_voltages,
    plot_asds,
    plot_magnetic_field_time_traces
)

# Import visualization module for summary plots
from qudi.logic.sensitivity_sweep_visualizer import SensitivitySweepVisualizer
from qudi.logic.tracking_sensitivity_tools import (
    build_tracking_conditions,
    json_safe,
    reconstruct_field_traces,
)


class SensitivitySweepLogic(LogicBase):
    """
    Logic module for automated ODMR-based magnetic field sensitivity parameter sweeps.

    This module orchestrates the full measurement workflow:
    1. Configure microwave source (power, FM frequency, FM deviation)
    2. Run ODMR scan and fit resonance to find zero-crossing
    3. Set CW microwave to zero-crossing frequency
    4. Record time series data for sensitivity measurement
    5. Calculate amplitude spectral density and magnetic field sensitivity
    6. Repeat for all parameter combinations

    Supports pausable/resumable operation and real-time progress reporting.

    Example config:

        sensitivity_sweep_logic:
            module.Class: 'sensitivity_sweep_logic.SensitivitySweepLogic'
            options:
                thermal_stabilization_time: 180
                sweep_loop_order: ['power', 'f_mod', 'f_dev']
                which_zero_crossing: 2
                n_most_prominent_peaks: 5
                min_fit_amplitude: 0.005
                min_feature_height: 0.003
                include_off_resonant_measurement: False
            connectors:
                odmr_logic: 'odmr_logic'
                time_series_logic: 'time_series_reader_logic'
    """

    # =========================================================================
    # Connectors
    # =========================================================================

    _odmr_logic = Connector(name='odmr_logic', interface='OdmrLogic')
    _time_series_logic = Connector(name='time_series_logic', interface='TimeSeriesReaderLogic')
    _odmr_lock_hw = Connector(
        name='odmr_lock_hw', interface='OdmrFreqLockInterface', optional=True)

    # =========================================================================
    # Config Options
    # =========================================================================

    _thermal_stabilization_time = ConfigOption(
        name='thermal_stabilization_time',
        default=180,
        missing='info'
    )

    _sweep_loop_order = ConfigOption(
        name='sweep_loop_order',
        default=['power', 'f_mod', 'f_dev'],
        missing='info'
    )

    _which_zero_crossing = ConfigOption(
        name='which_zero_crossing',
        default=2,
        missing='info'
    )

    _n_most_prominent_peaks = ConfigOption(
        name='n_most_prominent_peaks',
        default=5,
        missing='info'
    )

    _min_fit_amplitude = ConfigOption(
        name='min_fit_amplitude',
        default=0.005,
        missing='info'
    )

    _min_feature_height = ConfigOption(
        name='min_feature_height',
        default=0.003,
        missing='info'
    )

    _include_off_resonant_measurement = ConfigOption(
        name='include_off_resonant_measurement',
        default=False,
        missing='info'
    )

    _off_resonant_offset_hz = ConfigOption(
        name='off_resonant_offset_hz',
        default=30e6,
        missing='info'
    )

    _default_f_enbw = ConfigOption(
        name='default_f_enbw',
        default=500.0,
        missing='info'
    )  # Default Equivalent Noise Bandwidth of lock-in filter in Hz

    # =========================================================================
    # Status Variables (Persistent State)
    # =========================================================================

    _sweep_state = StatusVar(name='sweep_state', default='idle')  # idle, running, paused, cancelled
    _current_sweep_index = StatusVar(name='current_sweep_index', default=0)
    _sweep_parameters = StatusVar(name='sweep_parameters', default={})
    _odmr_parameters = StatusVar(name='odmr_parameters', default={})
    _stream_parameters = StatusVar(name='stream_parameters', default={})
    _results_list = StatusVar(name='results_list', default=[])
    _best_sensitivity = StatusVar(name='best_sensitivity', default=np.inf)
    _best_parameters = StatusVar(name='best_parameters', default={})
    _current_folder = StatusVar(name='current_folder', default='')

    # =========================================================================
    # Signals
    # =========================================================================

    sigSweepStarted = QtCore.Signal(int)  # total_points
    sigPointStarted = QtCore.Signal(int, dict)  # index, params
    sigPointCompleted = QtCore.Signal(int, dict)  # index, results
    sigSweepProgress = QtCore.Signal(dict)  # {current_idx, total, best_sens, best_params, ...}
    sigSweepPaused = QtCore.Signal()
    sigSweepResumed = QtCore.Signal()
    sigSweepFinished = QtCore.Signal(str)  # results_folder_path
    sigSweepCancelled = QtCore.Signal()
    sigError = QtCore.Signal(str)  # error_message

    # Real-time data for GUI plotting
    sigOdmrDataReady = QtCore.Signal(object, object)  # frequencies, signal
    sigFitDataReady = QtCore.Signal(dict)  # fit_result
    sigASDDataReady = QtCore.Signal(object, object, float)  # frequencies, asd_data, sensitivity
    sigTimeTraceReady = QtCore.Signal(object, object)  # times, b_field_data

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._thread_lock = RecursiveMutex()

        # Sweep execution control
        self._pause_requested = False
        self._cancel_requested = False

        # Current measurement data (temporary)
        self._parameter_combinations = []
        self._total_combinations = 0

    def on_activate(self):
        """Initialize the logic module."""
        try:
            # Verify connectors are valid
            odmr = self._odmr_logic()
            ts_logic = self._time_series_logic()

            self.log.info(f'Connected to ODMR logic: {odmr}')
            self.log.info(f'Connected to Time Series logic: {ts_logic}')

            # Initialize state
            if self._sweep_state not in ['idle', 'paused']:
                self._sweep_state = 'idle'

            self.log.info('Sensitivity Sweep Logic activated')

        except Exception as e:
            self.log.error(f'Failed to activate Sensitivity Sweep Logic: {e}')
            raise

    def on_deactivate(self):
        """Clean up resources."""
        try:
            # If sweep is running, stop it
            if self._sweep_state == 'running':
                self.cancel_sweep()
                time.sleep(0.5)  # Allow cleanup

            self.log.info('Sensitivity Sweep Logic deactivated')

        except Exception as e:
            self.log.error(f'Error during deactivation: {e}')

    # =========================================================================
    # Internal Methods - Event Processing
    # =========================================================================

    def _sleep_interruptible(self, seconds: float, check_interval: float = 0.5) -> bool:
        """
        Sleep for the specified duration while remaining responsive to pause/cancel.

        This method processes Qt events periodically during the sleep, allowing
        queued signals (like pause_sweep, cancel_sweep) to be processed.

        Args:
            seconds: Total sleep duration in seconds
            check_interval: How often to check for interrupts (default 0.5s)

        Returns:
            True if sleep completed normally, False if interrupted by pause/cancel
        """
        elapsed = 0.0
        while elapsed < seconds:
            # Process pending events (including pause/cancel signals)
            QtCore.QCoreApplication.processEvents()

            # Check if we should abort
            if self._pause_requested or self._cancel_requested:
                return False

            # Sleep for a short interval
            sleep_time = min(check_interval, seconds - elapsed)
            time.sleep(sleep_time)
            elapsed += sleep_time

        # Final event processing
        QtCore.QCoreApplication.processEvents()
        return not (self._pause_requested or self._cancel_requested)

    def _process_events_and_check_abort(self) -> bool:
        """
        Process pending Qt events and check if sweep should abort.

        Returns:
            True if sweep should continue, False if pause/cancel requested
        """
        QtCore.QCoreApplication.processEvents()
        return not (self._pause_requested or self._cancel_requested)

    # =========================================================================
    # Public Methods - Sweep Configuration
    # =========================================================================

    @QtCore.Slot(dict, dict, dict)
    def configure_sweep(self,
                       sweep_params: Dict[str, np.ndarray],
                       odmr_params: Dict[str, Any],
                       stream_params: Dict[str, Any]) -> None:
        """
        Configure parameter sweep settings.

        Args:
            sweep_params: Dictionary with keys 'power', 'f_mod', 'f_dev'
                         Each value is a numpy array of parameter values to sweep
                         Example: {'power': np.array([-25, -20, -15]),
                                  'f_mod': np.array([16e3, 20e3, 25e3]),
                                  'f_dev': np.array([500, 550, 600])}

            odmr_params: ODMR scan configuration
                        {'frequency_start': 2.86e9,
                         'frequency_stop': 2.88e9,
                         'frequency_points': 1001,
                         'run_time': 60,
                         'data_rate': 1000,
                         'multi_freq_mode': 'triple'}

            stream_params: Time series recording configuration
                          {'n_time_traces': 32,
                           'trace_duration': 1.0,  # seconds per trace
                           'data_rate': 1000}
        """
        with self._thread_lock:
            if self._sweep_state == 'running':
                raise RuntimeError('Cannot configure sweep while it is running')

            # Validate sweep parameters
            required_keys = ['power', 'f_mod', 'f_dev']
            for key in required_keys:
                if key not in sweep_params:
                    raise ValueError(f'Missing required sweep parameter: {key}')
                if not isinstance(sweep_params[key], (np.ndarray, list)):
                    raise ValueError(f'Sweep parameter {key} must be array-like')

            profiles = stream_params.get('filter_sweep', [])
            if stream_params.get('fir_filter_bandwidth') == '10kHz_minphase_2048':
                capabilities = self._time_series_logic()._streamer()._pyrpl.rp.lockin.filter_capabilities
                if not capabilities.get('10kHz_minphase_2048'):
                    raise ValueError('The selected 10 kHz filter requires the /2048 N=3 M=1 FPGA image')
                if any(mode in ('smith_linear', 'closed_loop_smith')
                       for mode in stream_params.get('tracking_modes', [])):
                    raise ValueError('The /2048 minimum-phase profile does not support Smith compensation')
                stream_params = dict(stream_params, f_enbw=10640.924904046242)
            if profiles:
                if any(mode in ('smith_linear', 'closed_loop_smith')
                       for mode in stream_params.get('tracking_modes', [])):
                    raise ValueError('Minimum-phase filter comparison requires Smith disabled')
                capabilities = self._time_series_logic()._streamer()._pyrpl.rp.lockin.filter_capabilities
                for profile in profiles:
                    name = profile['fir_filter_bandwidth']
                    if name not in ('6kHz_minphase', '10kHz_minphase') or not capabilities.get(name):
                        raise ValueError(f'Comparison filter {name} is unavailable on this FPGA')
                    bandwidths = np.asarray(profile['controller_bandwidths_hz'], dtype=float)
                    if bandwidths.ndim != 1 or not bandwidths.size or not np.all(
                            np.isfinite(bandwidths) & (bandwidths > 0)):
                        raise ValueError(f'Invalid controller bandwidths for {name}')
                    if not np.isfinite(profile['f_enbw']) or profile['f_enbw'] <= 0:
                        raise ValueError(f'Invalid ENBW for {name}')

            # Store configuration
            self._sweep_parameters = {
                'power': np.array(sweep_params['power']),
                'f_mod': np.array(sweep_params['f_mod']),
                'f_dev': np.array(sweep_params['f_dev'])
            }
            self._odmr_parameters = odmr_params.copy()
            self._stream_parameters = stream_params.copy()

            # Generate parameter combinations
            self._generate_parameter_combinations()

            self.log.info(
                f'Sweep configured: {self._total_combinations} total measurements'
            )
            self.log.info(
                f'Loop order: {self._sweep_loop_order}'
            )

    def _generate_parameter_combinations(self):
        """Generate all parameter combinations in the specified loop order."""
        param_dict = {
            'power': self._sweep_parameters['power'],
            'f_mod': self._sweep_parameters['f_mod'],
            'f_dev': self._sweep_parameters['f_dev']
        }

        # Validate loop order
        for param in self._sweep_loop_order:
            if param not in param_dict:
                raise ValueError(
                    f'Invalid loop order parameter: {param}. '
                    f'Must be one of {list(param_dict.keys())}'
                )

        # Create combinations in specified order
        ordered_arrays = [param_dict[p] for p in self._sweep_loop_order]
        self._parameter_combinations = list(product(*ordered_arrays))
        self._total_combinations = len(self._parameter_combinations)

        self.log.debug(f'Generated {self._total_combinations} parameter combinations')

    # =========================================================================
    # Public Methods - Sweep Control
    # =========================================================================

    @QtCore.Slot(dict, dict, dict)
    def configure_and_start_sweep(self, sweep_params: Dict[str, Any],
                                   odmr_params: Dict[str, Any],
                                   stream_params: Dict[str, Any]) -> None:
        """
        Configure and start sweep - single slot for GUI signal connection.

        This method ensures both configure and start happen on the logic thread,
        which is critical for proper pause/cancel functionality.

        Args:
            sweep_params: Parameter arrays for sweep
            odmr_params: ODMR scan configuration
            stream_params: Time series recording configuration
        """
        try:
            self.configure_sweep(sweep_params, odmr_params, stream_params)
            self.start_sweep()
        except Exception as e:
            self.log.error(f'Failed to start sweep: {e}', exc_info=True)
            self.sigError.emit(f'Failed to start sweep: {str(e)}')

    @QtCore.Slot()
    def start_sweep(self) -> None:
        """Start the parameter sweep from the beginning."""
        with self._thread_lock:
            if self._sweep_state == 'running':
                self.log.warning('Sweep already running')
                return

            if not self._parameter_combinations:
                raise RuntimeError('No sweep configured. Call configure_sweep() first.')

            # Reset state
            self._sweep_state = 'running'
            self._current_sweep_index = 0
            self._results_list = []
            self._best_sensitivity = np.inf
            self._best_parameters = {}
            self._pause_requested = False
            self._cancel_requested = False

            # Create data folder
            self._create_measurement_folder()

            # Emit start signal
            self.sigSweepStarted.emit(self._total_combinations)

            self.log.info(f'Starting sweep: {self._total_combinations} measurements')

            # Lock module and start sweep loop in thread
            self.module_state.lock()
            QtCore.QTimer.singleShot(0, self._run_sweep_loop)

    @QtCore.Slot()
    def pause_sweep(self) -> None:
        """Pause the running sweep (can be resumed later)."""
        with self._thread_lock:
            if self._sweep_state != 'running':
                self.log.warning('No sweep running to pause')
                return

            self._pause_requested = True
            self.log.info('Pause requested - will pause after current measurement')

    @QtCore.Slot()
    def resume_sweep(self) -> None:
        """Resume a paused sweep."""
        with self._thread_lock:
            if self._sweep_state != 'paused':
                self.log.warning('No paused sweep to resume')
                return

            self._sweep_state = 'running'
            self._pause_requested = False

            self.sigSweepResumed.emit()
            self.log.info(f'Resuming sweep from point {self._current_sweep_index + 1}')

            # Continue sweep loop
            self.module_state.lock()
            QtCore.QTimer.singleShot(0, self._run_sweep_loop)

    @QtCore.Slot()
    def cancel_sweep(self) -> None:
        """Cancel the running or paused sweep."""
        with self._thread_lock:
            if self._sweep_state not in ['running', 'paused']:
                self.log.warning('No sweep to cancel')
                return

            self._cancel_requested = True
            self.log.info('Cancel requested - will stop after current measurement')

    def get_sweep_status(self) -> Dict[str, Any]:
        """
        Get current sweep status.

        Returns:
            Dictionary with status information:
            {
                'state': 'idle'/'running'/'paused'/'cancelled',
                'current_index': int,
                'total_points': int,
                'best_sensitivity': float,
                'best_parameters': dict,
                'current_folder': str
            }
        """
        with self._thread_lock:
            return {
                'state': self._sweep_state,
                'current_index': self._current_sweep_index,
                'total_points': self._total_combinations,
                'best_sensitivity': self._best_sensitivity,
                'best_parameters': self._best_parameters.copy(),
                'current_folder': self._current_folder
            }

    def get_current_results(self) -> pd.DataFrame:
        """
        Get current results as pandas DataFrame.

        Returns:
            DataFrame with columns for parameters and sensitivity results
        """
        with self._thread_lock:
            if not self._results_list:
                return pd.DataFrame()
            return pd.DataFrame(self._results_list)

    # =========================================================================
    # Internal Methods - Sweep Loop
    # =========================================================================

    @QtCore.Slot()
    def _run_sweep_loop(self):
        """Main sweep loop - runs in logic thread."""
        try:
            # Process measurements from current_index to end
            while self._current_sweep_index < self._total_combinations:
                # CRITICAL: Process pending Qt events to allow pause/cancel signals through
                # Without this, queued signals cannot be processed while the loop is running
                QtCore.QCoreApplication.processEvents()

                # Check for pause/cancel requests
                if self._pause_requested:
                    self._handle_pause()
                    return

                if self._cancel_requested:
                    self._handle_cancel()
                    return

                # Get current parameter combination
                idx = self._current_sweep_index
                combination = self._parameter_combinations[idx]
                param_values = dict(zip(self._sweep_loop_order, combination))

                # Emit point started signal
                self.sigPointStarted.emit(idx, param_values)

                # Perform measurement
                try:
                    point_results = self._measure_single_point(idx, param_values)
                    if isinstance(point_results, dict):
                        point_results = [point_results]
                    for result in point_results:
                        self._results_list.append(result)

                        # Update best result across all open/closed-loop conditions.
                        if not np.isnan(result.get('sensitivity_nT_rtHz', np.nan)):
                            if result['sensitivity_nT_rtHz'] < self._best_sensitivity:
                                self._best_sensitivity = result['sensitivity_nT_rtHz']
                                self._best_parameters = param_values.copy()
                                self._best_parameters.update({
                                    'fir_filter_bandwidth': result.get('fir_filter_bandwidth'),
                                    'measurement_mode': result.get('measurement_mode', 'open_loop'),
                                    'controller_algorithm': result.get(
                                        'controller_algorithm', 'disabled'),
                                    'controller_bandwidth_hz': result.get(
                                        'controller_bandwidth_hz'),
                                    'smith_gain_multiplier': result.get(
                                        'smith_gain_multiplier'),
                                    'smith_delay_samples': result.get(
                                        'smith_delay_samples'),
                                })
                                self.log.info(
                                    f'New best sensitivity: {self._best_sensitivity:.3f} nT/sqrtHz'
                                )

                        # One table row per controller condition, all sharing the
                        # same point index and associated hyperfine scan.
                        self.sigPointCompleted.emit(idx, result)

                except InterruptedError:
                    # Sweep was interrupted by pause/cancel - don't treat as error
                    # The pause/cancel will be handled at the top of the next loop iteration
                    self.log.debug('Measurement interrupted, checking pause/cancel state')
                    continue

                except Exception as e:
                    self.log.error(f'Error measuring point {idx + 1}: {e}', exc_info=True)
                    # Add failed result
                    result = {
                        'power_dbm': param_values['power'],
                        'f_mod_hz': param_values['f_mod'],
                        'f_dev_khz': param_values['f_dev'],
                        'sensitivity_nT_rtHz': np.nan,
                        'error': str(e)
                    }
                    self._results_list.append(result)
                    self.sigError.emit(f'Point {idx + 1} failed: {str(e)}')

                # Emit progress update
                progress = {
                    'current_idx': idx + 1,
                    'total': self._total_combinations,
                    'best_sens': self._best_sensitivity,
                    'best_params': self._best_parameters.copy(),
                    'completion_percent': (idx + 1) / self._total_combinations * 100
                }
                self.sigSweepProgress.emit(progress)

                # Save intermediate results
                self._save_intermediate_results()

                # Move to next point
                self._current_sweep_index += 1

            # Sweep completed
            self._handle_completion()

        except Exception as e:
            self.log.error(f'Fatal error in sweep loop: {e}', exc_info=True)
            self.sigError.emit(f'Sweep failed: {str(e)}')
            self._sweep_state = 'idle'
            if self.module_state() == 'locked':
                self.module_state.unlock()

    def _measure_single_point(self, idx: int, params: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Perform a single sensitivity measurement for given parameters.

        Args:
            idx: Index of current measurement
            params: Parameter dictionary with 'power', 'f_mod', 'f_dev'

        Returns:
            One result dictionary per filter/controller condition. Controller
            conditions share a scan and fit only within the same filter.
        """
        power_dbm = params['power']
        f_mod_hz = params['f_mod']
        f_dev_khz = params['f_dev']

        self.log.info(
            f'\n--- Measurement {idx + 1}/{self._total_combinations} ---\n'
            f'Power: {power_dbm:.2f} dBm, '
            f'f_mod: {f_mod_hz / 1e3:.1f} kHz, '
            f'f_dev: {f_dev_khz:.1f} kHz'
        )

        # Apply the point's FM and scan settings before thermal stabilization.
        self._configure_mw_source(power_dbm, f_mod_hz, f_dev_khz)

        # Check if power changed (thermal stabilization needed)
        if idx > 0:
            prev_combination = self._parameter_combinations[idx - 1]
            prev_params = dict(zip(self._sweep_loop_order, prev_combination))
            if prev_params['power'] != power_dbm:
                self.log.info(
                    f'Power changed ({prev_params["power"]:.2f} -> {power_dbm:.2f} dBm). '
                    f'Waiting {self._thermal_stabilization_time}s for thermal stabilization...'
                )
                # Scan power is only a stored setting until output is enabled.
                # Heat at the new power near the scan centre before acquisition.
                settling_frequency = 0.5 * (
                    self._odmr_parameters['frequency_start'] +
                    self._odmr_parameters['frequency_stop'])
                try:
                    self._set_cw_frequency(settling_frequency, power_dbm)
                    if not self._sleep_interruptible(self._thermal_stabilization_time):
                        self.log.info('Thermal stabilization interrupted by pause/cancel request')
                        raise InterruptedError('Sweep interrupted during thermal stabilization')
                finally:
                    self._odmr_logic().toggle_cw_output(False)

        profiles = self._stream_parameters.get('filter_sweep')
        if not profiles:
            return self._measure_filter_point(idx, params)
        original_parameters = self._stream_parameters
        results = []
        try:
            for profile in profiles:
                if self._pause_requested or self._cancel_requested:
                    raise InterruptedError('Filter comparison interrupted')
                self._stream_parameters = dict(original_parameters, **profile)
                self._stream_parameters['fir_bypass'] = False
                try:
                    results.extend(self._measure_filter_point(idx, params))
                except InterruptedError:
                    raise
                except Exception as error:
                    self.log.exception('Filter measurement failed: %s', profile['fir_filter_bandwidth'])
                    results.append({
                        'power_dbm': power_dbm, 'f_mod_hz': f_mod_hz,
                        'f_dev_khz': f_dev_khz,
                        'fir_filter_bandwidth': profile['fir_filter_bandwidth'],
                        'sensitivity_nT_rtHz': np.nan, 'error': str(error)})
                finally:
                    self._disable_tracking_safely()
                    self._odmr_logic().toggle_cw_output(False)
        finally:
            self._stream_parameters = original_parameters
        return results

    def _measure_filter_point(self, idx, params):
        """Acquire a fresh scan and slope for one filter, then its conditions."""
        power_dbm, f_mod_hz, f_dev_khz = (
            params['power'], params['f_mod'], params['f_dev'])
        # Configure before naming/saving so labels describe the actual filter.
        requested_filter = self._stream_parameters.get('fir_filter_bandwidth')
        self._configure_lock_in_filters()
        filter_name = self._stream_parameters.get('fir_filter_bandwidth', 'unknown')
        if self._stream_parameters.get('filter_sweep') and (
                filter_name != requested_filter or self._stream_parameters.get('fir_bypass')):
            raise RuntimeError('Requested comparison filter was not applied')

        # Create filename nametag for this measurement (just the tag, not full path)
        # Use 'n' prefix for negative power values to avoid '-' in filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        power_str = f'n{abs(power_dbm):.2f}' if power_dbm < 0 else f'{power_dbm:.2f}'
        nametag = f'P_{power_str}dBm_fmod_{f_mod_hz / 1e3:.1f}k_fdev_{f_dev_khz:.1f}k_{filter_name}_{timestamp}'

        # Keep every stream, parameter snapshot and the associated hyperfine scan
        # in one point directory.  The independent Qudi ODMR save remains in its
        # normal data directory for compatibility with the ODMR GUI.
        point_folder = os.path.join(self._current_folder, nametag)
        os.makedirs(point_folder, exist_ok=True)
        filename_prefix = os.path.join(point_folder, nametag)

        # Step 2: Run ODMR scan
        frequencies, odmr_signal = self._run_odmr_scan(nametag)

        # Emit ODMR data for GUI
        self.sigOdmrDataReady.emit(frequencies, odmr_signal)

        # Step 3: Fit resonance
        fit_result = self._fit_odmr_data(frequencies, odmr_signal, filename_prefix)

        if fit_result is None:
            self.log.warning('Fit failed - skipping point')
            return [{
                'fir_filter_bandwidth': filter_name,
                'power_dbm': power_dbm,
                'f_mod_hz': f_mod_hz,
                'f_dev_khz': f_dev_khz,
                'sensitivity_nT_rtHz': np.nan,
                'sensitivity_rms_nT_rtHz': np.nan
            }]

        # Emit fit data for GUI (add fit curve data for plotting)
        # Note: fit_hyperfine doesn't return fit_frequency and fit_data arrays
        # We'll construct them from the fitted parabola parameters if needed
        fit_result_with_curves = fit_result.copy()
        # Add empty arrays for now - GUI can plot the original ODMR data
        fit_result_with_curves['fit_frequency'] = frequencies
        fit_result_with_curves['fit_data'] = odmr_signal  # Could compute parabola fit here
        self.sigFitDataReady.emit(fit_result_with_curves)

        # Calculate ODMR center frequency from peak and dip positions
        # Center = mean(mean(peak_positions), mean(dip_positions))
        odmr_center_hz = self._calculate_odmr_center(fit_result)

        # Store a local, lossless copy next to the traces.  This is the exact
        # normalized scan that produced the slope used below, not merely a link
        # to the latest scan in another Qudi directory.
        hyperfine_path = os.path.join(point_folder, 'associated_hyperfine_scan.npz')
        np.savez_compressed(
            hyperfine_path,
            frequency_hz=np.asarray(frequencies, dtype=np.float64),
            demod_signal=np.asarray(odmr_signal, dtype=np.float64),
            selected_zero_crossing_index=int(self._which_zero_crossing))
        with open(os.path.join(point_folder, 'associated_hyperfine_fit.json'), 'w') as handle:
            json.dump(json_safe(fit_result), handle, indent=2, allow_nan=False)

        # Step 4: Set CW to zero crossing
        zc_freq = fit_result['zero_crossing_frequencies [Hz]'][self._which_zero_crossing]
        slope = fit_result['zero_crossing_slopes [V/Hz]'][self._which_zero_crossing]
        self._set_cw_frequency(zc_freq, power_dbm)

        raw_slope, finite_sampler_scale = self._raw_discriminator_slope(slope)
        common_result = {
            'fir_filter_bandwidth': filter_name,
            'power_dbm': power_dbm,
            'f_mod_hz': f_mod_hz,
            'f_dev_khz': f_dev_khz,
            'odmr_center_hz': odmr_center_hz,
            'zero_crossing_hz': float(zc_freq),
            'linewidth_hz': fit_result['linewidths [Hz]'][self._which_zero_crossing],
            'zc_slope_V_per_Hz': float(slope),
            'controller_slope_raw_lsb_per_hz': float(raw_slope),
            'finite_sampler_scale': float(finite_sampler_scale),
            'associated_hyperfine_scan': os.path.relpath(
                hyperfine_path, self._current_folder),
        }
        point_manifest = {
            'schema_version': 1,
            'created': datetime.now().isoformat(),
            'microwave_parameters': {
                'power_dbm': power_dbm,
                'fm_modulation_frequency_hz': f_mod_hz,
                'fm_deviation_khz': f_dev_khz,
            },
            'odmr_parameters': self._odmr_parameters,
            'stream_parameters': self._stream_parameters,
            'fit_summary': common_result,
            'fit_result_file': 'associated_hyperfine_fit.json',
            'scan_file': 'associated_hyperfine_scan.npz',
        }
        with open(os.path.join(point_folder, 'measurement_point.json'), 'w') as handle:
            json.dump(json_safe(point_manifest), handle, indent=2, allow_nan=False)

        comparison_enabled = bool(
            self._stream_parameters.get('tracking_comparison_enabled', False))
        results = []
        try:
            if comparison_enabled:
                conditions = build_tracking_conditions(
                    self._stream_parameters.get('tracking_modes', ['open_loop']),
                    self._stream_parameters.get('controller_bandwidths_hz', [300.0]),
                    self._stream_parameters.get('smith_gain_multipliers', [4.0]),
                    self._stream_parameters.get('smith_delay_samples', [None]))
                if self._stream_parameters.get(
                        'include_off_resonant', self._include_off_resonant_measurement):
                    conditions.append({
                        'measurement_mode': 'off_resonant_reference',
                        'controller_algorithm': 'disabled',
                        'controller_bandwidth_hz': None,
                        'smith_gain_multiplier': None,
                        'smith_delay_samples': None,
                    })
                self.log.info('Measuring %d tracking conditions from one hyperfine scan.',
                              len(conditions))
                for condition_index, condition in enumerate(conditions):
                    if self._pause_requested or self._cancel_requested:
                        raise InterruptedError('comparison interrupted')
                    condition = dict(condition)
                    condition['microwave_frequency_hz'] = float(zc_freq)
                    if condition['measurement_mode'] == 'off_resonant_reference':
                        off_resonant_offset = float(self._stream_parameters.get(
                            'off_resonant_offset_hz', self._off_resonant_offset_hz))
                        condition['microwave_frequency_hz'] += off_resonant_offset
                        self._set_cw_frequency(
                            condition['microwave_frequency_hz'], power_dbm)
                    condition_result, asd_frequency, asd_data = \
                        self._measure_tracking_condition(
                            point_folder=point_folder,
                            condition_index=condition_index,
                            condition=condition,
                            signed_slope=float(raw_slope))
                    condition_result.update(common_result)
                    results.append(condition_result)
                    self.sigASDDataReady.emit(
                        asd_frequency, asd_data,
                        condition_result['sensitivity_nT_rtHz'])
            else:
                # Preserve the established open-loop workflow for old configs.
                sensitivity_on, sensitivity_on_rms, asd_freq, asd_on = \
                    self._measure_sensitivity(filename_prefix + '_ON-resonant', slope)
                self.sigASDDataReady.emit(asd_freq, asd_on, sensitivity_on)
                result = dict(common_result)
                result.update({
                    'measurement_mode': 'open_loop',
                    'controller_algorithm': 'disabled',
                    'controller_bandwidth_hz': None,
                    'smith_gain_multiplier': None,
                    'smith_delay_samples': None,
                    'sensitivity_nT_rtHz': sensitivity_on,
                    'sensitivity_rms_nT_rtHz': sensitivity_on_rms,
                })
                results.append(result)
        finally:
            self._disable_tracking_safely()
            self._odmr_logic().toggle_cw_output(False)

        return results

    def _configure_mw_source(self, power_dbm: float, f_mod_hz: float, f_dev_khz: float):
        """Configure microwave source via ODMR logic."""
        odmr = self._odmr_logic()

        # Turn off CW output first before changing multi-frequency mode
        # The microwave module does not allow mode changes while output is active
        odmr.toggle_cw_output(False)
        time.sleep(0.1)  # Allow settling

        # Set scan power
        odmr.set_scan_power(power_dbm)

        # Get microwave module
        mw = odmr._microwave()

        # Configure multi-frequency mode
        multi_freq_mode = self._odmr_parameters.get('multi_freq_mode', 'triple')
        mw.set_multi_frequency_mode(multi_freq_mode, None)

        # Configure FM parameters
        mw.set_fm_parameters(
            enable=True,
            deviation_khz=f_dev_khz,
            modulation_frequency=f_mod_hz
        )

        self.log.debug(
            f'MW source configured: P={power_dbm} dBm, '
            f'f_mod={f_mod_hz} Hz, f_dev={f_dev_khz} kHz'
        )

    def _run_odmr_scan(self, nametag: str) -> Tuple[np.ndarray, np.ndarray]:
        """Run ODMR scan and return data."""
        odmr = self._odmr_logic()

        # Configure ODMR scan
        odmr.set_runtime(self._odmr_parameters['run_time'])
        odmr.set_frequency_range(
            self._odmr_parameters['frequency_start'],
            self._odmr_parameters['frequency_stop'],
            self._odmr_parameters['frequency_points'],
            0  # no oversampling
        )
        odmr.set_data_rate(self._odmr_parameters.get('data_rate', 1000))

        # Start scan
        self.log.debug('Starting ODMR scan...')
        odmr.start_odmr_scan()

        # Wait for completion with interrupt checking
        run_time = self._odmr_parameters['run_time']
        sleep_interval = min(0.5, run_time / 10) if run_time > 1 else 0.1
        while odmr.module_state() == 'locked':
            # Process events and check for abort
            QtCore.QCoreApplication.processEvents()
            if self._pause_requested or self._cancel_requested:
                self.log.info('ODMR scan interrupted by pause/cancel request')
                odmr.stop_odmr_scan()
                raise InterruptedError('Sweep interrupted during ODMR scan')
            time.sleep(sleep_interval)

        self.log.debug('ODMR scan complete')

        # Save data - pass just the nametag (ODMR logic adds its own path and prefix)
        odmr._save_thumbnails = True
        odmr._use_timestamp = False  # We already include timestamp in nametag
        odmr.save_odmr_data(nametag)

        # Get data
        joined_data = odmr._join_signal_data()
        frequencies = np.array(joined_data.T[0])
        signal = np.array(joined_data.T[1])

        # Normalize (subtract mean)
        signal = signal - np.mean(signal)

        return frequencies, signal

    def _fit_odmr_data(self, frequencies: np.ndarray, signal: np.ndarray,
                      filename_prefix: str) -> Optional[Dict]:
        """Fit ODMR data to extract zero crossings."""
        self.log.debug('Fitting ODMR data...')

        try:
            fit_result = fit_hyperfine(
                frequencies,
                signal,
                feature_prominence=self._min_fit_amplitude,
                n_most_prominent_peaks=self._n_most_prominent_peaks,
                plot_result=False,
                save_result_plot=True,
                min_feature_height=self._min_feature_height,
                filename=filename_prefix
            )

            if fit_result is None:
                return None

            # Check if we have the required zero crossing
            zc_freqs = fit_result.get('zero_crossing_frequencies [Hz]', [])
            if self._which_zero_crossing >= len(zc_freqs):
                self.log.error(
                    f'Fit did not find required zero crossing {self._which_zero_crossing}. '
                    f'Only found {len(zc_freqs)} zero crossings.'
                )
                return None

            self.log.debug(
                f'Fit successful: ZC freq = {zc_freqs[self._which_zero_crossing] / 1e9:.6f} GHz'
            )

            return fit_result

        except Exception as e:
            self.log.error(f'Fitting failed: {e}', exc_info=True)
            return None

    def _calculate_odmr_center(self, fit_result: Dict) -> float:
        """
        Calculate the overall ODMR center frequency from fitted peak and dip positions.

        Args:
            fit_result: Dictionary from fit_hyperfine containing 'peak_positions [Hz]'
                       and 'dip_positions [Hz]' arrays.

        Returns:
            Center frequency in Hz, or np.nan if calculation fails.
        """
        try:
            peak_positions = fit_result.get('peak_positions [Hz]', np.array([]))
            dip_positions = fit_result.get('dip_positions [Hz]', np.array([]))

            # Filter out NaN values
            valid_peaks = peak_positions[~np.isnan(peak_positions)]
            valid_dips = dip_positions[~np.isnan(dip_positions)]

            if len(valid_peaks) == 0 or len(valid_dips) == 0:
                self.log.warning('Cannot calculate ODMR center: no valid peak or dip positions')
                return np.nan

            mean_peak = np.mean(valid_peaks)
            mean_dip = np.mean(valid_dips)
            center = (mean_peak + mean_dip) / 2.0

            self.log.debug(
                f'ODMR center calculation: mean_peak={mean_peak/1e9:.6f} GHz, '
                f'mean_dip={mean_dip/1e9:.6f} GHz, center={center/1e9:.6f} GHz'
            )

            return center

        except Exception as e:
            self.log.error(f'Error calculating ODMR center: {e}')
            return np.nan

    def _set_cw_frequency(self, frequency: float, power: float):
        """Set CW microwave output.

        Note: Must turn CW OFF before changing parameters, because toggle_cw_output(True)
        returns early if CW is already on, without applying the new frequency.
        """
        odmr = self._odmr_logic()

        # Turn off CW first to ensure new parameters are applied
        odmr.toggle_cw_output(False)
        time.sleep(0.1)  # Allow settling

        # Set new parameters
        odmr.set_cw_parameters(float(frequency), float(power))

        # Turn CW back on (this now applies the new frequency)
        odmr.toggle_cw_output(True)
        time.sleep(0.1)  # Allow settling

        self.log.debug(f'CW set to {frequency/1e9:.6f} GHz at {power:.2f} dBm')

    def _configure_lock_in_filters(self):
        """
        Configure lock-in FIR filter settings based on stream_parameters.

        Reads 'fir_bypass' and 'fir_filter_bandwidth' from stream_parameters
        and applies them to the lock-in module via the time_series_logic's streamer.
        """
        fir_bypass = self._stream_parameters.get('fir_bypass', False)
        fir_filter_bw = self._stream_parameters.get('fir_filter_bandwidth', '2kHz_minphase')
        smith_requested = (
            'smith_linear' in self._stream_parameters.get('tracking_modes', []))

        self.log.info(f'Configuring lock-in filters: bypass={fir_bypass}, bandwidth={fir_filter_bw}')

        try:
            # Access the streamer hardware through time_series_logic
            ts_logic = self._time_series_logic()
            streamer = ts_logic._streamer()

            # CRITICAL: Ensure stream input is set to 'demod' for sensitivity measurements
            # This protects against mode changes from other modules (e.g., ODMR tracking
            # might have switched to 'ftw_corr' mode)
            if hasattr(streamer, 'set_stream_input') and hasattr(streamer, 'stream_input'):
                current_mode = streamer.stream_input
                if current_mode != 'demod':
                    self.log.warning(
                        f'Stream input was "{current_mode}", switching to "demod" for sensitivity measurement'
                    )
                    # TSR must be stopped to change mode
                    if ts_logic.module_state() == 'locked':
                        ts_logic.stop_reading()
                    streamer.set_stream_input('demod')
                    self.log.info('Stream input set to "demod"')

            # Access pyrpl instance from streamer
            if not hasattr(streamer, '_pyrpl') or streamer._pyrpl is None:
                if self._stream_parameters.get('filter_sweep'):
                    raise RuntimeError('Cannot apply comparison filter: PyRPL unavailable')
                self.log.warning('Cannot access pyrpl instance from streamer - lock-in filters not configured')
                return

            pyrpl_instance = streamer._pyrpl

            # Access lockin module (named 'lockin' not 'lock_in' per PyRPL naming convention)
            if not hasattr(pyrpl_instance.rp, 'lockin'):
                if self._stream_parameters.get('filter_sweep'):
                    raise RuntimeError('Cannot apply comparison filter: lock-in unavailable')
                self.log.warning('lockin module not available in pyrpl - filters not configured')
                return

            lock_in = pyrpl_instance.rp.lockin

            # The implemented Smith model explicitly represents the compiled
            # linear-phase FIR delay.  A bypassed or minimum-phase lane is a
            # different plant and would make a Smith/non-Smith comparison
            # misleading.  This also protects against a GUI StatusVar retaining
            # the 20 kHz selection when switching back to the /4096 image.
            if smith_requested:
                smith_filter = lock_in.smith_filter_name
                if fir_bypass:
                    self.log.warning(
                        'FIR bypass is incompatible with the Smith comparison; '
                        f'enabling the advertised Smith filter "{smith_filter}"')
                    fir_bypass = False
                if fir_filter_bw != smith_filter:
                    self.log.warning(
                        f'Filter "{fir_filter_bw}" does not match the Smith plant; '
                        f'using advertised filter "{smith_filter}" for all conditions')
                    fir_filter_bw = smith_filter

            # Configure FIR bypass
            lock_in.fir_bypass_ch1 = fir_bypass
            self.log.debug(f'Set fir_bypass_ch1 = {fir_bypass}')

            # Configure filter bandwidth (only effective when bypass is False)
            if not fir_bypass:
                # Validate against the FIRs actually compiled into this image.
                valid_filters = {
                    name for name, available in lock_in.filter_capabilities.items()
                    if available and name != 'fir_bypass'}
                if '2kHz_minphase' in valid_filters:
                    valid_filters.add('2kHz')
                fallback_filter = ('2kHz_minphase'
                                   if '2kHz_minphase' in valid_filters
                                   else next(iter(sorted(valid_filters))))
                if fir_filter_bw not in valid_filters:
                    if self._stream_parameters.get('filter_sweep'):
                        raise RuntimeError(f'Comparison filter {fir_filter_bw} is unavailable')
                    self.log.warning(
                        f'Invalid filter selection "{fir_filter_bw}", '
                        f'using "{fallback_filter}"')
                    fir_filter_bw = fallback_filter

                lock_in.filter_select_ch1 = fir_filter_bw
                self.log.debug(f'Set filter_select_ch1 = {fir_filter_bw}')

            # Verify settings were applied
            actual_bypass = lock_in.fir_bypass_ch1
            actual_filter = lock_in.filter_select_ch1
            # Persist the effective hardware selection, rather than a stale GUI
            # choice, into the run metadata and subsequent condition records.
            self._stream_parameters['fir_bypass'] = bool(actual_bypass)
            self._stream_parameters['fir_filter_bandwidth'] = actual_filter
            self.log.info(f'Lock-in filter configured: bypass={actual_bypass}, filter={actual_filter}')

            if actual_bypass != fir_bypass:
                if self._stream_parameters.get('filter_sweep'):
                    raise RuntimeError('FIR bypass readback mismatch')
                self.log.error(f'FIR bypass setting mismatch! Requested {fir_bypass}, got {actual_bypass}')
            if not fir_bypass and actual_filter != fir_filter_bw:
                self.log.error(f'Filter bandwidth mismatch! Requested {fir_filter_bw}, got {actual_filter}')

        except AttributeError as e:
            if self._stream_parameters.get('filter_sweep'):
                raise
            self.log.error(f'Failed to access lock-in module: {e}')
        except Exception as e:
            if self._stream_parameters.get('filter_sweep'):
                raise
            self.log.error(f'Error configuring lock-in filters: {e}', exc_info=True)

    def _disable_tracking_safely(self) -> None:
        """Best-effort safe state used after every comparison condition."""
        try:
            lock_hw = self._odmr_lock_hw()
            if lock_hw is not None:
                lock_hw.enable_lock(False)
                lock_hw.clear()
        except Exception as e:
            self.log.warning('Could not return frequency tracker to its disabled state: %s', e)

    def _controller_snapshot(self, lock_hw) -> Dict[str, Any]:
        """Collect the controller state without making a particular FPGA ABI mandatory."""
        snapshot = {}
        for method_name, key in (
                ('get_status', 'status'),
                ('get_constraints', 'constraints')):
            try:
                snapshot[key] = getattr(lock_hw, method_name)()
            except Exception as e:
                snapshot[key + '_error'] = str(e)
        try:
            snapshot['max_correction_hz'] = lock_hw.get_max_correction_hz()
        except Exception as e:
            snapshot['max_correction_error'] = str(e)
        try:
            pyrpl_instance = getattr(lock_hw, '_pyrpl', None)
            if pyrpl_instance is not None:
                lockin = pyrpl_instance.rp.lockin
                snapshot['fpga_profile'] = lockin.capabilities
                snapshot['filter_capabilities'] = lockin.filter_capabilities
                snapshot['filter_select_ch1'] = lockin.filter_select_ch1
                snapshot['fir_bypass_ch1'] = bool(lockin.fir_bypass_ch1)
        except Exception as e:
            snapshot['fpga_profile_error'] = str(e)
        return json_safe(snapshot)

    def _raw_discriminator_slope(self, fitted_slope: float) -> Tuple[float, float]:
        """Undo finite-sampler display calibration for the raw FPGA controller."""
        scale = 1.0
        try:
            scanner = self._odmr_logic()._data_scanner()
            scale = (float(getattr(scanner, '_calibration_factor', 1.0)) *
                     float(getattr(scanner, '_signal_scale', 1.0)))
        except Exception as e:
            self.log.warning('Could not inspect finite-sampler calibration: %s', e)
        if not np.isfinite(scale) or scale == 0:
            raise ValueError(f'invalid finite-sampler discriminator scale {scale!r}')
        return float(fitted_slope) / scale, scale

    def _configure_tracking_condition(self, condition: Dict[str, Any],
                                      signed_slope: float):
        """Apply one open/conventional/Smith condition and return its hardware."""
        lock_hw = self._odmr_lock_hw()
        if lock_hw is None:
            raise RuntimeError(
                'tracking comparison requested but odmr_lock_hw is not connected')

        lock_hw.enable_lock(False)
        if hasattr(lock_hw, 'prepare_legacy_single_resonance'):
            lock_hw.prepare_legacy_single_resonance()
        lock_hw.clear()

        algorithm = condition['controller_algorithm']
        if algorithm == 'disabled':
            return lock_hw

        kwargs = {}
        if algorithm == 'smith_linear':
            kwargs['smith_gain_multiplier'] = condition['smith_gain_multiplier']
            if condition['smith_delay_samples'] is not None:
                kwargs['smith_delay_samples'] = condition['smith_delay_samples']
        lock_hw.set_tracking_algorithm(algorithm, **kwargs)

        max_correction = float(
            self._stream_parameters.get('tracking_max_correction_hz', 50e6))
        lock_hw.set_max_correction_hz(max_correction)
        lock_hw.set_bandwidth(
            float(condition['controller_bandwidth_hz']), abs(float(signed_slope)))
        lock_hw.clear()
        lock_hw.enable_lock(True)
        return lock_hw

    def _acquire_timestamped_tracking_events(self, closed_loop: bool,
                                              lock_hw=None):
        """Acquire raw Region-11 events without passing through the display buffer."""
        ts_logic = self._time_series_logic()
        if ts_logic.module_state() == 'locked':
            ts_logic.stop_reading()
            time.sleep(0.2)
        streamer = ts_logic._streamer()
        data_streamer = getattr(streamer, '_data_streamer', None)
        if data_streamer is None:
            raise RuntimeError('the configured streamer has no Region-11 data streamer')

        sources = ('fir', 'correction') if closed_loop else ('fir',)
        duration = (float(self._stream_parameters.get('n_time_traces', 32)) *
                    float(self._stream_parameters.get('trace_duration', 1.0)))
        reader = data_streamer.subscribe(
            sources=sources,
            ring_bytes=int(self._stream_parameters.get('stream_ring_bytes', 0)),
            coalesce_us=int(self._stream_parameters.get('stream_coalesce_us', 0)))
        chunks = []
        transport_stats = {}
        fpga_stats = {}
        controller_status_history = []
        started = time.monotonic()
        status_poll_interval = float(
            self._stream_parameters.get('tracking_status_poll_interval_s', 0.2))
        next_status_poll = started
        try:
            deadline = started + duration
            while time.monotonic() < deadline:
                if not self._sleep_interruptible(
                        min(0.05, max(0.0, deadline - time.monotonic())),
                        check_interval=0.05):
                    raise InterruptedError('tracking stream acquisition interrupted')
                events = reader.read_events()
                if events.size:
                    chunks.append(events)
                if reader.error is not None:
                    raise RuntimeError(f'stream receiver failed: {reader.error}')
                now = time.monotonic()
                if closed_loop and lock_hw is not None and now >= next_status_poll:
                    try:
                        status = dict(lock_hw.get_status())
                        status['acquisition_time_s'] = now - started
                        controller_status_history.append(json_safe(status))
                    except Exception as e:
                        controller_status_history.append({
                            'acquisition_time_s': now - started,
                            'read_error': str(e),
                        })
                    next_status_poll = now + max(0.02, status_poll_interval)
            tail = reader.read_events()
            if tail.size:
                chunks.append(tail)
            transport_stats = reader.stats()
            fpga_stats = data_streamer.stats()
        finally:
            data_streamer.unsubscribe(reader)

        events = (np.concatenate(chunks)
                  if chunks else np.empty(0, dtype=data_streamer.EVENT_DTYPE))
        if not events.size:
            raise RuntimeError('no FPGA stream events were received')

        reconstruction = data_streamer.reconstruct_tracking(
            events, nslots=1, interval=int(streamer._demod_decimation),
            correction_converter=streamer.ftw_to_hz)
        error_signal = reconstruction['err'][0]
        correction_hz = reconstruction['corr'][0] if closed_loop else None

        # Single-resonance operation should advertise a live acquisition window.
        # Retain a diagnostic fallback for images predating that flag so the raw
        # FIR samples are not discarded merely because a metadata bit is absent.
        if error_signal.size and not np.any(np.isfinite(error_signal)):
            fir_events = data_streamer.select(events, 'fir')
            fir_ticks = data_streamer.unwrap_counter(fir_events['timestamp'])
            origin = int(reconstruction['ticks'][0])
            fir_index = np.rint(
                (fir_ticks.astype(np.int64) - origin) /
                float(streamer._demod_decimation)).astype(np.int64)
            error_signal = np.full(reconstruction['ticks'].size, np.nan)
            keep = (fir_index >= 0) & (fir_index < error_signal.size)
            error_signal[fir_index[keep]] = fir_events['data'][keep]
            self.log.warning(
                'FIR live-window flag was absent; used the timestamp grid directly.')

        ticks = reconstruction['ticks']
        time_s = ((ticks - ticks[0]).astype(np.float64) /
                  float(data_streamer.timestamp_clock_hz))
        diagnostics = {
            'requested_duration_s': duration,
            'wall_duration_s': time.monotonic() - started,
            'event_count': int(events.size),
            'observed_event_rate_hz': float(events.size / max(duration, 1e-12)),
            'event_payload_rate_MBps': float(
                events.size * data_streamer.EVENT_DTYPE.itemsize /
                max(duration, 1e-12) / 1e6),
            'fir_event_count': int(data_streamer.select(events, 'fir').size),
            'correction_event_count': int(
                data_streamer.select(events, 'correction').size),
            'transport': json_safe(transport_stats),
            'fpga': json_safe(fpga_stats),
            'controller_status_history': controller_status_history,
        }
        return time_s, error_signal, correction_hz, events, diagnostics

    def _analyse_field_trace(self, field_nt: np.ndarray, sample_rate_hz: float,
                             filename_prefix: str) -> Dict[str, Any]:
        """Save/plot a gap-free field trace and return its ASD noise floor."""
        field_nt = np.asarray(field_nt, dtype=np.float64)
        finite = np.isfinite(field_nt)
        if not np.any(finite):
            return {'sensitivity': np.nan, 'sensitivity_rms': np.nan,
                    'frequencies': np.empty(0), 'asd_hanning': np.empty(0),
                    'internal_missing_samples': int(field_nt.size)}

        first, last = np.flatnonzero(finite)[[0, -1]]
        trimmed = field_nt[first:last + 1]
        internal_missing = int(np.count_nonzero(~np.isfinite(trimmed)))
        if internal_missing:
            self.log.error(
                'Refusing to calculate a lossless ASD for %s: %d internal samples missing.',
                filename_prefix, internal_missing)
            return {'sensitivity': np.nan, 'sensitivity_rms': np.nan,
                    'frequencies': np.empty(0), 'asd_hanning': np.empty(0),
                    'internal_missing_samples': internal_missing}

        duration = trimmed.size / float(sample_rate_hz)
        result = plot_asds(
            trimmed, float(sample_rate_hz), duration,
            float(self._stream_parameters.get('f_enbw', self._default_f_enbw)),
            save_fig=True, save_data=True, filename_prefix=filename_prefix,
            sensitivity_f_min=float(
                self._stream_parameters.get('sensitivity_f_min', 200.0)),
            sensitivity_f_max=float(
                self._stream_parameters.get('sensitivity_f_max', 1400.0)),
            exclude_50hz_harmonics=bool(
                self._stream_parameters.get('exclude_50hz_harmonics', True)))
        result['internal_missing_samples'] = 0
        return result

    def _measure_tracking_condition(self, point_folder: str, condition_index: int,
                                    condition: Dict[str, Any], signed_slope: float):
        """Configure, acquire, analyse and save one controller comparison run."""
        algorithm = condition['controller_algorithm']
        tag_parts = [f'{condition_index:02d}', condition['measurement_mode'], algorithm]
        if condition['controller_bandwidth_hz'] is not None:
            tag_parts.append(f'bw{condition["controller_bandwidth_hz"]:g}Hz')
        if condition['smith_gain_multiplier'] is not None:
            tag_parts.append(f'x{condition["smith_gain_multiplier"]:g}')
        if condition['smith_delay_samples'] is not None:
            tag_parts.append(f'd{int(condition["smith_delay_samples"])}')
        condition_tag = '_'.join(tag_parts)
        condition_folder = os.path.join(point_folder, condition_tag)
        os.makedirs(condition_folder, exist_ok=True)

        lock_hw = self._configure_tracking_condition(condition, signed_slope)
        before = self._controller_snapshot(lock_hw)
        if algorithm != 'disabled':
            settling = float(
                self._stream_parameters.get('tracking_settling_time_s', 1.0))
            if settling > 0 and not self._sleep_interruptible(settling):
                raise InterruptedError('controller settling interrupted')

        try:
            time_s, error_signal, correction_hz, events, diagnostics = \
                self._acquire_timestamped_tracking_events(
                    closed_loop=(algorithm != 'disabled'), lock_hw=lock_hw)
            after = self._controller_snapshot(lock_hw)
        finally:
            self._disable_tracking_safely()

        inverted = False
        try:
            inverted = bool(lock_hw.get_invert())
        except Exception:
            pass
        traces = reconstruct_field_traces(
            error_signal, correction_hz, signed_slope,
            correction_inverted=inverted)
        sample_rate = float(self._time_series_logic().sampling_rate)
        finite_correction = np.abs(
            traces['correction_frequency_hz'][
                np.isfinite(traces['correction_frequency_hz'])])
        correction_peak_hz = (float(np.max(finite_correction))
                              if finite_correction.size else 0.0)
        correction_limit_hz = before.get('max_correction_hz')
        correction_limit_hit = bool(
            correction_limit_hz is not None and correction_limit_hz > 0 and
            correction_peak_hz >= 0.999 * float(correction_limit_hz))

        trace_path = os.path.join(condition_folder, 'tracking_time_trace.npz')
        np.savez_compressed(
            trace_path,
            time_s=time_s,
            discriminator_signal=error_signal,
            sample_rate_hz=sample_rate,
            **traces)
        if bool(self._stream_parameters.get('save_raw_stream_events', True)):
            np.save(os.path.join(condition_folder, 'raw_stream_events.npy'), events)

        estimate_asd = self._analyse_field_trace(
            traces['estimated_field_nt'], sample_rate,
            os.path.join(condition_folder, 'estimated_field'))
        residual_asd = self._analyse_field_trace(
            traces['residual_field_nt'], sample_rate,
            os.path.join(condition_folder, 'residual_field'))
        if algorithm == 'disabled':
            correction_asd = {'sensitivity': 0.0, 'sensitivity_rms': 0.0}
        else:
            correction_asd = self._analyse_field_trace(
                traces['correction_field_nt'], sample_rate,
                os.path.join(condition_folder, 'correction_field'))

        relevant_sources = ('fir', 'correction') if algorithm != 'disabled' else ('fir',)
        source_overflows = diagnostics['fpga'].get('source_overflows', {})
        transport_lossless = (
            int(diagnostics['transport'].get('n_gap', 0)) == 0 and
            int(diagnostics['transport'].get('n_seq_skips', 0)) == 0 and
            int(diagnostics['fpga'].get('fpga_overflows', 0)) == 0 and
            all(int(source_overflows.get(source, 0)) == 0
                for source in relevant_sources) and
            int(estimate_asd.get('internal_missing_samples', 0)) == 0)
        measurement_valid = transport_lossless and not correction_limit_hit
        if not measurement_valid:
            self.log.error(
                '%s marked invalid (transport_lossless=%s, correction_limit_hit=%s).',
                condition_tag, transport_lossless, correction_limit_hit)

        metadata = {
            'schema_version': 1,
            'created': datetime.now().isoformat(),
            'condition': condition,
            'signed_discriminator_slope_per_hz': signed_slope,
            'correction_inverted_for_lower_sideband': inverted,
            'sample_rate_hz': sample_rate,
            'stream_diagnostics': diagnostics,
            'controller_before_acquisition': before,
            'controller_after_acquisition': after,
            'trace_file': os.path.basename(trace_path),
            'raw_event_file': ('raw_stream_events.npy' if bool(
                self._stream_parameters.get('save_raw_stream_events', True)) else None),
            'sensitivity_nT_rtHz': estimate_asd['sensitivity'],
            'residual_sensitivity_nT_rtHz': residual_asd['sensitivity'],
            'correction_sensitivity_nT_rtHz': correction_asd['sensitivity'],
            'correction_peak_hz': correction_peak_hz,
            'correction_limit_hit': correction_limit_hit,
            'transport_lossless': transport_lossless,
            'measurement_valid': measurement_valid,
        }
        with open(os.path.join(condition_folder, 'measurement_metadata.json'), 'w') as handle:
            json.dump(json_safe(metadata), handle, indent=2, allow_nan=False)

        result = dict(condition)
        status_history = diagnostics['controller_status_history']
        valid_status = [status for status in status_history
                        if 'read_error' not in status]
        actual_smith_delay = None
        if algorithm == 'smith_linear':
            actual_smith_delay = before.get('status', {}).get(
                'smith_delay_samples', condition['smith_delay_samples'])
        result.update({
            'condition_folder': os.path.relpath(condition_folder, self._current_folder),
            'sensitivity_nT_rtHz': float(
                estimate_asd['sensitivity']) if measurement_valid else np.nan,
            'sensitivity_rms_nT_rtHz': float(
                estimate_asd.get('sensitivity_rms', np.nan))
                if measurement_valid else np.nan,
            'residual_sensitivity_nT_rtHz': float(
                residual_asd['sensitivity']) if measurement_valid else np.nan,
            'correction_sensitivity_nT_rtHz': float(
                correction_asd['sensitivity']) if measurement_valid else np.nan,
            'measurement_valid': measurement_valid,
            'transport_lossless': transport_lossless,
            'missing_samples': int(
                estimate_asd.get('internal_missing_samples', 0)),
            'stream_event_count': diagnostics['event_count'],
            'stream_fir_event_count': diagnostics['fir_event_count'],
            'stream_correction_event_count': diagnostics['correction_event_count'],
            'stream_event_rate_hz': diagnostics['observed_event_rate_hz'],
            'stream_event_payload_MBps': diagnostics['event_payload_rate_MBps'],
            'stream_transport_gap_words': diagnostics['transport'].get('n_gap', 0),
            'stream_sequence_skips': diagnostics['transport'].get('n_seq_skips', 0),
            'stream_fpga_overflows': diagnostics['fpga'].get('fpga_overflows', 0),
            'smith_delay_samples_actual': actual_smith_delay,
            'controller_mu_hz_per_lsb': before.get('status', {}).get(
                'mu_hz_per_lsb'),
            'controller_any_saturated': any(
                bool(status.get('saturated', False)) for status in valid_status) or
                correction_limit_hit,
            'controller_correction_peak_hz': correction_peak_hz,
            'controller_correction_limit_hit': correction_limit_hit,
            'controller_locked_fraction': (
                float(np.mean([bool(status.get('locked', False))
                               for status in valid_status]))
                if valid_status else np.nan),
        })
        self.log.info(
            '%s sensitivity: total %.3f, residual %.3f, correction %.3f nT/sqrtHz',
            condition_tag, result['sensitivity_nT_rtHz'],
            result['residual_sensitivity_nT_rtHz'],
            result['correction_sensitivity_nT_rtHz'])
        return result, estimate_asd['frequencies'], estimate_asd['asd_hanning']

    def _measure_sensitivity(self, filename_prefix: str, slope: float
                            ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Measure magnetic field sensitivity using time series data.

        Args:
            filename_prefix: Base filename for saving data
            slope: Zero-crossing slope in V/Hz for B-field conversion

        Returns:
            Tuple of (sensitivity_nT_rtHz, sensitivity_rms_nT_rtHz, asd_frequencies, asd_data)
        """
        ts_logic = self._time_series_logic()

        # Get streaming parameters
        n_traces = self._stream_parameters.get('n_time_traces', 32)
        trace_duration = self._stream_parameters.get('trace_duration', 1.0)

        # Get the actual hardware sample rate (Red Pitaya FPGA has fixed ~30.5 kHz)
        actual_sample_rate = ts_logic.sampling_rate
        self.log.info(f'Hardware sample rate: {actual_sample_rate:.1f} Hz')

        # Calculate total samples needed
        # Use hardware sample rate directly (no oversampling)
        samples_per_trace = int(actual_sample_rate * trace_duration)
        total_samples = samples_per_trace * n_traces
        total_duration = n_traces * trace_duration

        self.log.info(
            f'Acquiring sensitivity data: {n_traces} traces × {trace_duration}s = {total_duration}s total, '
            f'{total_samples} samples at {actual_sample_rate:.1f} Hz'
        )

        # Configure time series logic for maximum window
        # Set window size to cover the entire acquisition
        ts_logic.set_trace_settings(
            oversampling_factor=1,
            moving_average_width=1,
            trace_window_size=total_duration,
            data_rate=actual_sample_rate  # Use actual hardware rate
        )

        # IMPORTANT: Stop any existing streaming before starting fresh
        # This ensures a clean state and prevents buffer overflow issues
        if ts_logic.module_state() == 'locked':
            ts_logic.stop_reading()
            time.sleep(0.3)  # Allow stop to complete

        # Start fresh data acquisition
        ts_logic.start_reading()
        time.sleep(1.0)  # Allow buffer to fill initially

        # Wait for the full acquisition time with interrupt checking
        self.log.debug(f'Waiting {total_duration}s for data acquisition...')
        if not self._sleep_interruptible(total_duration + 0.5):
            self.log.info('Data acquisition interrupted by pause/cancel request')
            ts_logic.stop_reading()
            raise InterruptedError('Sweep interrupted during data acquisition')

        # Get trace data from time series logic
        # trace_data returns (times_array, {channel_name: data_array})
        trace_times, trace_data_dict = ts_logic.trace_data

        # Get actual data rate after configuration
        actual_data_rate = ts_logic.data_rate
        self.log.debug(f'Effective data rate: {actual_data_rate:.1f} Hz')

        # Get channel data (get first available channel)
        if not trace_data_dict:
            raise RuntimeError('No channel data available from time_series_logic')

        channel_name = list(trace_data_dict.keys())[0]
        voltage_trace = trace_data_dict[channel_name]

        self.log.info(
            f'Retrieved trace data: {len(voltage_trace)} samples from channel "{channel_name}"'
        )

        # Validate we have enough data
        if len(voltage_trace) < actual_data_rate:
            self.log.warning(
                f'Insufficient data: got {len(voltage_trace)} samples, '
                f'expected at least {actual_data_rate} (1 second worth)'
            )

        # Convert to magnetic field
        # scaling_factor accounts for any analog output scaling
        scaling_factor = 1.0  # Adjust if your hardware has output scaling

        b_field_trace = magnetic_field_from_voltages(
            voltage_trace,
            slope,
            scaling_factor=scaling_factor
        )

        # Calculate actual trace duration from data
        actual_trace_duration = len(voltage_trace) / actual_data_rate
        self.log.debug(f'Actual trace duration: {actual_trace_duration:.3f} s')

        # Ensure trace_times matches the data length
        # time_series_logic may return None or mismatched times
        if trace_times is None or len(trace_times) != len(b_field_trace):
            self.log.debug(f'Reconstructing time array (trace_times was '
                          f'{type(trace_times).__name__}, len={len(trace_times) if trace_times is not None else 0})')
            trace_times = np.arange(len(b_field_trace)) / actual_data_rate

        # Emit time trace for GUI
        self.sigTimeTraceReady.emit(trace_times, b_field_trace)

        # Save raw voltage time trace data as numpy file
        # (conversion factor slope is saved in the summary CSV as zc_slope_V_per_Hz)
        try:
            time_trace_data_path = filename_prefix + '_voltage_time_trace.npz'
            np.savez(
                time_trace_data_path,
                time_s=trace_times,
                voltage_V=voltage_trace,
                sample_rate_Hz=actual_data_rate
            )
            self.log.info(f'Voltage time trace saved: {time_trace_data_path}')
        except Exception as e:
            self.log.error(f'Failed to save time trace data: {e}')

        # Plot and save time traces
        try:
            n_samples = len(b_field_trace)
            self.log.debug(f'Plotting time traces: {n_samples} samples, '
                          f'{len(trace_times)} time points, rate={actual_data_rate:.1f} Hz, '
                          f'duration={actual_trace_duration:.2f} s')
            self.log.debug(f'Time trace filename prefix: {filename_prefix}')

            # Validate we have enough data for plotting
            # The plot function uses int(duration) which would be 0 if duration < 1
            if actual_trace_duration < 1.0:
                self.log.warning(f'Trace duration {actual_trace_duration:.3f}s < 1s, '
                               f'time trace plot may be empty')

            plot_magnetic_field_time_traces(
                b_field_trace,
                trace_times,
                actual_data_rate,
                actual_trace_duration,
                save_fig=True,
                filename_prefix=filename_prefix,
                voltage_trace=voltage_trace
            )

            # Verify file was created
            expected_file = filename_prefix + "_magnetic_field_time_traces.pdf"
            if os.path.exists(expected_file):
                self.log.info(f'Time trace plot saved: {expected_file}')
            else:
                self.log.warning(f'Time trace plot NOT found at expected path: {expected_file}')

        except Exception as e:
            self.log.error(f'Failed to plot time traces: {e}', exc_info=True)

        # Calculate ASD
        # f_ENBW is the Equivalent Noise Bandwidth of the lock-in filter
        # For Red Pitaya IQ demodulation with typical settings, ENBW ~ bandwidth * 1.06
        # Use the config option as default, allow override from stream_params
        f_enbw = self._stream_parameters.get('f_enbw', self._default_f_enbw)
        self.log.debug(f'Using f_ENBW = {f_enbw:.1f} Hz for ASD calculation')

        # Get sensitivity bandwidth parameters from stream_params
        sensitivity_f_min = self._stream_parameters.get('sensitivity_f_min', 200.0)
        sensitivity_f_max = self._stream_parameters.get('sensitivity_f_max', 1400.0)
        exclude_50hz = self._stream_parameters.get('exclude_50hz_harmonics', True)
        self.log.debug(f'Sensitivity bandwidth: {sensitivity_f_min:.0f}-{sensitivity_f_max:.0f} Hz, '
                      f'exclude 50Hz harmonics: {exclude_50hz}')

        try:
            asd_result = plot_asds(
                b_field_trace,
                actual_data_rate,
                actual_trace_duration,
                f_enbw,  # 4th positional argument: f_ENBW
                save_fig=True,
                save_data=True,
                filename_prefix=filename_prefix,
                sensitivity_f_min=sensitivity_f_min,
                sensitivity_f_max=sensitivity_f_max,
                exclude_50hz_harmonics=exclude_50hz
            )

            sensitivity = asd_result['sensitivity']
            sensitivity_rms = asd_result.get('sensitivity_rms', np.nan)
            asd_frequencies = asd_result.get('frequencies', np.array([]))
            asd_data = asd_result.get('asd_hanning', np.array([]))  # Use Hanning window ASD

            self.log.info(f'Sensitivity: {sensitivity:.3f} nT/sqrt(Hz) (RMS: {sensitivity_rms:.3f})')

        except Exception as e:
            self.log.error(f'ASD calculation failed: {e}', exc_info=True)
            sensitivity = np.nan
            sensitivity_rms = np.nan
            asd_frequencies = np.array([])
            asd_data = np.array([])

        # CRITICAL: Stop streaming before returning to avoid concurrent Red Pitaya access
        # when other operations (like toggle_cw_output) try to communicate with the FPGA
        try:
            if ts_logic.module_state() == 'locked':
                ts_logic.stop_reading()
                time.sleep(0.5)  # Allow streaming thread to fully stop
                self.log.debug('Stopped time series streaming')
        except Exception as e:
            self.log.warning(f'Failed to stop streaming cleanly: {e}')

        return sensitivity, sensitivity_rms, asd_frequencies, asd_data

    # =========================================================================
    # Internal Methods - State Management
    # =========================================================================

    def _handle_pause(self):
        """Handle pause request."""
        with self._thread_lock:
            self._sweep_state = 'paused'
            self._pause_requested = False

            self.sigSweepPaused.emit()
            self.log.info(f'Sweep paused at point {self._current_sweep_index + 1}')

            if self.module_state() == 'locked':
                self.module_state.unlock()

    def _handle_cancel(self):
        """Handle cancel request."""
        with self._thread_lock:
            self._sweep_state = 'cancelled'
            self._cancel_requested = False

            # Save results collected so far
            self._save_final_results()

            # Generate summary visualization plots (even for cancelled sweeps)
            self._generate_summary_plots()

            self.sigSweepCancelled.emit()
            self.log.info(f'Sweep cancelled after {self._current_sweep_index} measurements')

            if self.module_state() == 'locked':
                self.module_state.unlock()

    def _handle_completion(self):
        """Handle sweep completion."""
        with self._thread_lock:
            self._sweep_state = 'idle'

            # Save final results
            self._save_final_results()

            # Generate summary visualization plots
            self._generate_summary_plots()

            # Emit completion signal
            self.sigSweepFinished.emit(self._current_folder)

            self.log.info(
                f'\n=== Sweep Complete ===\n'
                f'Total measurements: {len(self._results_list)}\n'
                f'Best sensitivity: {self._best_sensitivity:.3f} nT/sqrtHz\n'
                f'Best parameters: {self._best_parameters}\n'
                f'Results saved to: {self._current_folder}'
            )

            if self.module_state() == 'locked':
                self.module_state.unlock()

    # =========================================================================
    # Internal Methods - Data Management
    # =========================================================================

    def _create_measurement_folder(self):
        """Create timestamped folder for measurement data."""
        base_folder = self.module_default_data_dir
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        folder_name = os.path.join(base_folder, 'SensitivitySweep', timestamp)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        self._current_folder = folder_name
        self.log.info(f'Data folder: {folder_name}')

    def _save_intermediate_results(self):
        """Save intermediate results (called after each point)."""
        if not self._results_list:
            return

        try:
            df = pd.DataFrame(self._results_list)
            csv_path = os.path.join(self._current_folder, 'parameter_sweep_summary.csv')
            df.to_csv(csv_path, sep='\t', index=False, na_rep='NaN')
        except Exception as e:
            self.log.error(f'Error saving intermediate results: {e}')

    def _save_final_results(self):
        """Save final results and metadata."""
        try:
            # Save results DataFrame
            if self._results_list:
                df = pd.DataFrame(self._results_list)
                csv_path = os.path.join(self._current_folder, 'parameter_sweep_summary.csv')
                df.to_csv(csv_path, sep='\t', index=False, na_rep='NaN')

            # Save metadata
            # Determine the effective off-resonant settings (from stream_params or ConfigOptions)
            include_off_resonant = self._stream_parameters.get(
                'include_off_resonant', self._include_off_resonant_measurement
            )
            off_resonant_offset = self._stream_parameters.get(
                'off_resonant_offset_hz', self._off_resonant_offset_hz
            )

            # Get sensitivity bandwidth parameters
            sensitivity_f_min = self._stream_parameters.get('sensitivity_f_min', 200.0)
            sensitivity_f_max = self._stream_parameters.get('sensitivity_f_max', 1400.0)
            exclude_50hz = self._stream_parameters.get('exclude_50hz_harmonics', True)

            conditions_per_point = 1
            if self._stream_parameters.get('tracking_comparison_enabled', False):
                conditions_per_point = len(build_tracking_conditions(
                    self._stream_parameters.get('tracking_modes', ['open_loop']),
                    self._stream_parameters.get('controller_bandwidths_hz', [300.0]),
                    self._stream_parameters.get('smith_gain_multipliers', [4.0]),
                    self._stream_parameters.get('smith_delay_samples', [None])))
                if include_off_resonant:
                    conditions_per_point += 1
            profiles = self._stream_parameters.get('filter_sweep') or [{}]
            conditions_per_filter = {}
            for profile in profiles:
                parameters = dict(self._stream_parameters, **profile)
                count = 1
                if parameters.get('tracking_comparison_enabled', False):
                    count = len(build_tracking_conditions(
                        parameters.get('tracking_modes', ['open_loop']),
                        parameters.get('controller_bandwidths_hz', [300.0]),
                        parameters.get('smith_gain_multipliers', [4.0]),
                        parameters.get('smith_delay_samples', [None]))) + int(include_off_resonant)
                conditions_per_filter[parameters.get('fir_filter_bandwidth', 'unknown')] = count
            metadata = {
                'total_measurements': len(self._results_list),
                'total_planned_hyperfine_scans': self._total_combinations * len(profiles),
                'conditions_per_hyperfine_scan': (conditions_per_point if len(profiles) == 1 else None),
                'conditions_per_filter': conditions_per_filter,
                'total_planned_measurements': (
                    self._total_combinations * sum(conditions_per_filter.values())),
                'best_sensitivity_nT_rtHz': float(self._best_sensitivity),
                'best_parameters': self._best_parameters,
                'sweep_loop_order': self._sweep_loop_order,
                'thermal_stabilization_time_s': self._thermal_stabilization_time,
                'which_zero_crossing': self._which_zero_crossing,
                'odmr_parameters': self._odmr_parameters,
                'stream_parameters': self._stream_parameters,
                'include_off_resonant_measurement': include_off_resonant,
                'off_resonant_offset_hz': off_resonant_offset,
                'sensitivity_bandwidth_hz': [sensitivity_f_min, sensitivity_f_max],
                'exclude_50hz_harmonics': exclude_50hz,
                'timestamp': datetime.now().isoformat()
            }

            json_path = os.path.join(self._current_folder, 'sweep_metadata.json')
            with open(json_path, 'w') as f:
                json.dump(json_safe(metadata), f, indent=4, allow_nan=False)

            self.log.info(f'Results saved to {self._current_folder}')

        except Exception as e:
            self.log.error(f'Error saving final results: {e}', exc_info=True)

    def _generate_summary_plots(self):
        """
        Generate summary visualization plots for the sweep results.

        Creates plots showing how sensitivity and ODMR parameters vary with
        the swept parameters. Plots are saved to a 'summary_plots' subfolder.
        """
        if not self._results_list or len(self._results_list) < 2:
            self.log.info('Not enough data points for summary visualization (need >= 2)')
            return

        try:
            # Prepare data and metadata
            df = pd.DataFrame(self._results_list)

            # Build metadata dict with current state
            metadata = {
                'best_sensitivity_nT_rtHz': float(self._best_sensitivity),
                'best_parameters': self._best_parameters.copy(),
                'sweep_loop_order': self._sweep_loop_order,
                'total_measurements': len(self._results_list),
                'total_planned': self._total_combinations
            }

            # Create visualizer and generate plots
            visualizer = SensitivitySweepVisualizer(
                results_df=df,
                metadata=metadata,
                output_folder=self._current_folder,
                logger=self.log
            )

            generated_files = visualizer.generate_all_plots()

            if generated_files:
                self.log.info(
                    f'Generated {len(generated_files)} summary plots in '
                    f'{os.path.join(self._current_folder, "summary_plots")}'
                )
            else:
                self.log.warning('No summary plots were generated')

        except Exception as e:
            self.log.error(f'Error generating summary plots: {e}', exc_info=True)
