# -*- coding: utf-8 -*-
"""
Data processing mixin for MotorScanLogic.

Handles data acquisition for different scan modes: ODMR, streaming, frequency tracking.
"""

import os
import numpy as np
from typing import Dict

from PySide2 import QtCore

from .data_structures import ScanMode

# FTW (Frequency Tuning Word) conversion constant for Red Pitaya @ 125 MHz
# FTW = frequency_hz * FTW_PER_HZ; frequency_hz = ftw / FTW_PER_HZ
FTW_PER_HZ = (2**32) / 125e6  # ≈ 34.359738


class DataProcessingMixin:
    """
    Mixin class providing data acquisition and processing for MotorScanLogic.
    
    This mixin provides:
    - Time series data handling (CONTINUOUS_STREAM mode)
    - Frequency tracking data processing (CONTINUOUS_FREQ_TRACK mode)
    - ODMR scan processing and fitting (STEP_ODMR mode)
    - Lock status monitoring
    
    The parent class must provide:
    - _time_series_logic: Connector to time series logic
    - _odmr_logic: Connector to ODMR logic
    - _tracking_logic: Connector to the ODMR tracking logic
    - _scan_data: MotorScanData instance
    - _scan_state: Current ScanState
    - _active_scan_mode: Current scan mode value
    - _ts_raw_data_buffer: Dict for buffering raw data
    - _ts_connected: bool flag for signal connection
    - _tracking_zero_crossing: float, baseline frequency
    - _fit_function: callable for ODMR fitting
    - Various ConfigOptions for fitting parameters
    - log: Logger instance
    """

    def _connect_time_series_signals(self):
        """Connect to time series logic signals for non-blocking data reception."""
        if self._ts_connected:
            return
        
        ts_logic = self._time_series_logic()
        if ts_logic is not None and hasattr(ts_logic, 'sigNewRawData'):
            try:
                ts_logic.sigNewRawData.connect(
                    self._on_new_raw_data,
                    QtCore.Qt.QueuedConnection
                )
                self._ts_connected = True
                self.log.debug("Connected to time_series_logic.sigNewRawData")
            except (TypeError, RuntimeError) as e:
                self.log.warning(f"Could not connect to sigNewRawData: {e}")
    
    def _disconnect_time_series_signals(self):
        """Disconnect from time series logic signals."""
        if not self._ts_connected:
            return
        
        ts_logic = self._time_series_logic()
        if ts_logic is not None and hasattr(ts_logic, 'sigNewRawData'):
            try:
                ts_logic.sigNewRawData.disconnect(self._on_new_raw_data)
                self.log.debug("Disconnected from time_series_logic.sigNewRawData")
            except (TypeError, RuntimeError):
                pass
        self._ts_connected = False
    
    @QtCore.Slot(object, object)
    def _on_new_raw_data(self, data_buffer, times_buffer):
        """
        Handle new raw data from time series logic.
        
        This slot receives data continuously from the time_series_logic's sigNewRawData
        signal, allowing non-blocking data collection while the motor is moving.
        The data is buffered and associated with the current scan position.
        
        Args:
            data_buffer: Raw data array from streamer (flattened, all channels)
            times_buffer: Optional timestamp array
        """
        from .data_structures import ScanState
        
        if self._scan_state != ScanState.RUNNING:
            return

        if self._scan_data is None:
            return

        mode = self._scan_data.scan_mode

        # Only process data for continuous streaming modes
        if mode not in (ScanMode.CONTINUOUS_STREAM, ScanMode.CONTINUOUS_FREQ_TRACK):
            return

        # Get channel information from time series logic
        ts_logic = self._time_series_logic()
        if ts_logic is None:
            return

        try:
            if mode == ScanMode.CONTINUOUS_FREQ_TRACK:
                # For frequency tracking, we receive FTW values and convert to absolute frequency
                self._process_freq_track_data(data_buffer)
            else:
                # Standard CONTINUOUS_STREAM processing
                channel_names = ts_logic.active_channel_names
                if not channel_names:
                    return

                # Parse the flattened data buffer into per-channel data
                # data_buffer is flattened: [ch1_s1, ch2_s1, ch1_s2, ch2_s2, ...]
                n_channels = len(channel_names)
                if data_buffer is None or len(data_buffer) == 0:
                    return

                n_samples = len(data_buffer) // n_channels
                if n_samples == 0:
                    return

                # Reshape to (n_samples, n_channels) then transpose to (n_channels, n_samples)
                data_reshaped = data_buffer[:n_samples * n_channels].reshape(n_samples, n_channels).T

                # Buffer the data for each channel
                for i, ch_name in enumerate(channel_names):
                    if ch_name not in self._ts_raw_data_buffer:
                        self._ts_raw_data_buffer[ch_name] = []
                    self._ts_raw_data_buffer[ch_name].extend(data_reshaped[i].tolist())

        except Exception as e:
            self.log.debug(f"Error processing raw data: {e}")

    def _process_freq_track_data(self, data_buffer):
        """
        Process FTW data for CONTINUOUS_FREQ_TRACK mode.

        Converts raw FTW (Frequency Tuning Word) values from the FPGA to
        absolute frequency by adding the zero-crossing baseline.

        Args:
            data_buffer: Raw data array from streamer (FTW values, signed 32-bit)
        """
        if data_buffer is None or len(data_buffer) == 0:
            return

        if self._tracking_zero_crossing is None:
            self.log.warning("No zero-crossing baseline set for frequency tracking")
            return

        # Get time series logic for channel info
        ts_logic = self._time_series_logic()
        if ts_logic is None:
            return

        try:
            # Time series may have multiple channels, but we only use the first (FTW correction)
            channel_names = ts_logic.active_channel_names
            n_channels = len(channel_names) if channel_names else 1
            n_samples = len(data_buffer) // n_channels
            if n_samples == 0:
                return

            # Extract first channel data (FTW values)
            if n_channels > 1:
                ftw_data = data_buffer[:n_samples * n_channels].reshape(n_samples, n_channels)[:, 0]
            else:
                ftw_data = data_buffer[:n_samples]

            # Convert FTW to frequency correction in Hz
            # FTW is a signed 32-bit value representing frequency in FPGA units
            ftw_array = np.asarray(ftw_data, dtype=np.float64)
            correction_hz = ftw_array / FTW_PER_HZ

            # Calculate absolute frequency = zero-crossing + correction
            absolute_freq_hz = self._tracking_zero_crossing + correction_hz

            # Buffer for the 'absolute_frequency' channel
            if 'absolute_frequency' not in self._ts_raw_data_buffer:
                self._ts_raw_data_buffer['absolute_frequency'] = []
            self._ts_raw_data_buffer['absolute_frequency'].extend(absolute_freq_hz.tolist())

        except Exception as e:
            self.log.debug(f"Error processing frequency tracking data: {e}")

    @QtCore.Slot()
    def _on_lock_status_poll_timeout(self):
        """
        Poll lock status during CONTINUOUS_FREQ_TRACK mode.

        Auto-pauses the scan if lock is lost, emitting sigLockLostDuringScan.
        Emits sigLockStatusUpdated for GUI indicator updates.
        """
        from .data_structures import ScanState
        
        if self._scan_state != ScanState.RUNNING:
            return

        if self._active_scan_mode != ScanMode.CONTINUOUS_FREQ_TRACK.value:
            return

        tracking_logic = self._tracking_logic()
        if tracking_logic is None:
            return

        # Get current lock status
        try:
            lock_enabled = tracking_logic.tracking_active
        except Exception as e:
            self.log.warning(f"Could not read lock status: {e}")
            return

        # Emit status update for GUI
        self.sigLockStatusUpdated.emit(lock_enabled)

        # Check for lock loss
        if not lock_enabled:
            self.log.warning("Lock lost during frequency tracking scan - pausing scan")
            self._paused_due_to_lock_loss = True
            self._scan_state = ScanState.PAUSED
            self._lock_status_timer.stop()
            self.sigScanStateChanged.emit(self._scan_state.name)
            self.sigLockLostDuringScan.emit()

    def _start_odmr_scan_async(self):
        """Start ODMR scan asynchronously (non-blocking)."""
        odmr = self._odmr_logic()
        if odmr is None:
            self.log.error("ODMR logic not available")
            self._advance_to_next_point({})
            return

        # Set position hint for ODMR dummy
        if hasattr(odmr, 'set_position_hint'):
            odmr.set_position_hint(self._current_pos_dict)

        # Connect to ODMR completion signal
        self._odmr_scan_complete = False
        self._waiting_for_odmr = True

        try:
            odmr.sigScanStateUpdated.connect(
                self._on_odmr_scan_state_changed,
                QtCore.Qt.QueuedConnection
            )
        except (TypeError, RuntimeError):
            pass  # Already connected

        # Start ODMR scan
        odmr.start_odmr_scan()

    @QtCore.Slot(bool)
    def _on_odmr_scan_state_changed(self, is_running: bool):
        """Called when ODMR scan state changes."""
        if not self._waiting_for_odmr:
            return

        if not is_running:
            # ODMR scan completed
            self._waiting_for_odmr = False
            self._odmr_scan_complete = True

            # Disconnect signal to avoid multiple calls
            odmr = self._odmr_logic()
            if odmr is not None:
                try:
                    odmr.sigScanStateUpdated.disconnect(self._on_odmr_scan_state_changed)
                except (TypeError, RuntimeError):
                    pass

            # Process ODMR results and advance
            result = self._process_odmr_results()
            self._advance_to_next_point(result)

    def _process_odmr_results(self) -> Dict:
        """Process ODMR scan results and extract data using fit_hyperfine."""
        odmr = self._odmr_logic()
        motor = self._motor_hardware()
        point_idx = self._current_point_idx
        position = self._current_pos_dict

        # Get actual position
        actual_pos = motor.get_pos()

        # Get ODMR data
        frequency_data = odmr.frequency_data if odmr else []
        signal_data = odmr.signal_data if odmr else {}

        # Initialize result
        result = {
            'point_index': point_idx,
            'target_position': position,
            'actual_position': actual_pos,
            'center_frequency': np.nan,
            'linewidth': np.nan,
            'splitting': np.nan,
            'n_features_found': 0,
            'fit_result': None,
        }

        # Get grid index for filename
        grid_idx = self._scan_data.point_index_to_grid_index(point_idx)
        if len(grid_idx) == 2:
            pixel_tag = f'pixel_x{grid_idx[0]:03d}_y{grid_idx[1]:03d}'
        else:
            pixel_tag = f'pixel_{point_idx:04d}'

        # Perform fitting if we have data and fit function
        if self._fit_function is not None and len(frequency_data) > 0:
            try:
                if signal_data:
                    channel_name = list(signal_data.keys())[0]
                    channel_data = signal_data[channel_name]

                    if isinstance(channel_data, list) and len(channel_data) > 0:
                        voltage_data = np.array(channel_data[0])
                        freq_data = np.array(frequency_data[0]) if isinstance(frequency_data, list) else np.array(frequency_data)
                    else:
                        voltage_data = np.array(channel_data)
                        freq_data = np.array(frequency_data)

                    # Determine if we should save fit plots
                    save_plot = self._save_odmr_fit_plots and self._current_scan_folder is not None
                    if save_plot:
                        # Create odmr_fits subfolder if it doesn't exist
                        fits_folder = os.path.join(self._current_scan_folder, 'odmr_fits')
                        os.makedirs(fits_folder, exist_ok=True)
                        fit_filename = os.path.join(fits_folder, pixel_tag)
                    else:
                        fit_filename = None

                    # Calculate max_pair_distance_hz (use default if not set)
                    max_pair_dist = self._max_pair_distance_hz
                    if max_pair_dist is None:
                        max_pair_dist = self._hyperfine_spacing_hz * 0.75  # ~1.6 MHz for 2.158 MHz spacing

                    # Use robust fitting if enabled, otherwise standard fit_hyperfine
                    if self._use_robust_fitting and hasattr(self, '_fit_function_robust') and self._fit_function_robust is not None:
                        # Use fit_odmr_robust with automatic quality assessment
                        fit_result = self._fit_function_robust(
                            freq_data,
                            voltage_data,
                            n_expected_features=self._fit_n_most_prominent_peaks,
                            hyperfine_spacing_hz=self._hyperfine_spacing_hz,
                            max_pair_distance_hz=max_pair_dist,
                            feature_prominence=self._fit_feature_prominence,
                            min_feature_height=self._fit_min_feature_height
                        )
                        # Log quality assessment info
                        if fit_result:
                            quality = fit_result.get('quality_grade', 'unknown')
                            method = fit_result.get('method_used', 'unknown')
                            self.log.debug(f"Point {point_idx}: quality={quality}, method={method}")
                    else:
                        # Call fit_hyperfine with proximity-based pairing
                        fit_result = self._fit_function(
                            freq_data,  # positional argument (like sensitivity_sweep)
                            voltage_data,  # positional argument (like sensitivity_sweep)
                            feature_prominence=self._fit_feature_prominence,
                            n_most_prominent_peaks=self._fit_n_most_prominent_peaks,
                            min_feature_height=self._fit_min_feature_height,
                            max_pair_distance_hz=max_pair_dist,  # NEW: proximity-based pairing
                            plot_result=False,
                            save_result_plot=save_plot,
                            filename=fit_filename
                        )

                    if fit_result is not None:
                        result['fit_result'] = fit_result
                        result['n_features_found'] = fit_result.get('n_features_found', fit_result.get('n_features', 0))

                        # Check if fit_odmr_robust provided center_frequency directly
                        # (from constrained model or COM fallback)
                        if 'center_frequency' in fit_result and fit_result['center_frequency'] is not None:
                            center_freq = fit_result['center_frequency']
                            if not np.isnan(center_freq):
                                result['center_frequency'] = center_freq
                                result['fit_method'] = fit_result.get('method_used', 'unknown')
                                result['quality_grade'] = fit_result.get('quality_grade', 'unknown')

                        # Also try standard fit_hyperfine output format (zero_crossing_frequencies)
                        if np.isnan(result['center_frequency']):
                            zc_freqs = fit_result.get('zero_crossing_frequencies [Hz]', [])
                            if isinstance(zc_freqs, np.ndarray) and len(zc_freqs) > 0:
                                valid_zc = zc_freqs[~np.isnan(zc_freqs)]
                                if len(valid_zc) > 0:
                                    result['center_frequency'] = np.mean(valid_zc)
                                    if len(valid_zc) >= 2:
                                        result['splitting'] = valid_zc[-1] - valid_zc[0]

                        # Extract linewidth from either format
                        linewidths = fit_result.get('linewidths [Hz]', [])
                        if isinstance(linewidths, np.ndarray) and len(linewidths) > 0:
                            valid_lw = linewidths[~np.isnan(linewidths)]
                            if len(valid_lw) > 0:
                                result['linewidth'] = np.mean(valid_lw)
                        elif 'linewidth' in fit_result and fit_result['linewidth'] is not None:
                            result['linewidth'] = fit_result['linewidth']

                        self.log.debug(f"Point {point_idx}: center_freq={result['center_frequency']/1e9:.6f} GHz, "
                                      f"linewidth={result['linewidth']/1e3:.1f} kHz, "
                                      f"n_features={result['n_features_found']}")
                    else:
                        self.log.warning(f"Fit returned None at point {point_idx}")

            except Exception as e:
                self.log.warning(f"Fitting failed at point {point_idx}: {e}")

        # Fallback: extract basic stats from raw ODMR data if no fitting available
        if np.isnan(result['center_frequency']) and len(frequency_data) > 0 and signal_data:
            self.log.debug(f"Using fallback extraction for point {point_idx}")
            try:
                channel_name = list(signal_data.keys())[0]
                channel_data = signal_data[channel_name]

                if isinstance(channel_data, list) and len(channel_data) > 0:
                    voltage_data = np.array(channel_data[0])
                    freq_data = frequency_data[0] if isinstance(frequency_data, list) else frequency_data
                else:
                    voltage_data = np.array(channel_data)
                    freq_data = frequency_data

                freq_data = np.array(freq_data)

                min_idx = np.argmin(voltage_data)
                result['center_frequency'] = freq_data[min_idx]
                result['n_features_found'] = 1

                min_val = voltage_data[min_idx]
                max_val = np.max(voltage_data)
                half_max = (min_val + max_val) / 2

                above_half = voltage_data > half_max
                transitions = np.where(np.diff(above_half.astype(int)))[0]
                if len(transitions) >= 2:
                    left_idx = transitions[0]
                    right_idx = transitions[-1]
                    result['linewidth'] = abs(freq_data[right_idx] - freq_data[left_idx])

            except Exception as e:
                self.log.debug(f"Fallback extraction failed: {e}")

        # Update grid data
        if self._scan_data.is_2d:
            self._scan_data.center_frequency[grid_idx] = result['center_frequency']
            self._scan_data.linewidth[grid_idx] = result['linewidth']
            self._scan_data.splitting[grid_idx] = result['splitting']
            self._scan_data.fit_quality[grid_idx] = result['n_features_found']
            self._scan_data.odmr_fit_results[grid_idx[0]][grid_idx[1]] = result['fit_result']
        else:
            self._scan_data.center_frequency[grid_idx[0]] = result['center_frequency']
            self._scan_data.linewidth[grid_idx[0]] = result['linewidth']
            self._scan_data.splitting[grid_idx[0]] = result['splitting']
            self._scan_data.fit_quality[grid_idx[0]] = result['n_features_found']

        # Store raw ODMR data for this pixel (for later saving)
        if self._scan_data.odmr_raw_per_pixel is not None:
            try:
                # Extract and store raw ODMR frequency and signal data
                raw_odmr_dict = {
                    'target_position': position.copy(),
                    'actual_position': actual_pos.copy(),
                    'grid_index': grid_idx,
                    'frequency_data': None,
                    'signal_data': {},
                }
                
                if len(frequency_data) > 0:
                    # Store frequency data (handle list vs array)
                    if isinstance(frequency_data, list) and len(frequency_data) > 0:
                        raw_odmr_dict['frequency_data'] = np.array(frequency_data[0])
                    else:
                        raw_odmr_dict['frequency_data'] = np.array(frequency_data)
                
                if signal_data:
                    for ch_name, ch_data in signal_data.items():
                        if isinstance(ch_data, list) and len(ch_data) > 0:
                            raw_odmr_dict['signal_data'][ch_name] = np.array(ch_data[0])
                        else:
                            raw_odmr_dict['signal_data'][ch_name] = np.array(ch_data)
                
                self._scan_data.odmr_raw_per_pixel[point_idx] = raw_odmr_dict
                
            except Exception as e:
                self.log.debug(f"Failed to store raw ODMR data for point {point_idx}: {e}")

        return result

    def _collect_continuous_data_and_advance(self):
        """
        Collect streaming data at current position and advance to next point.
        
        Uses data buffered from the sigNewRawData signal, which allows non-blocking
        data collection. The time_series_logic continues to stream data to its GUI
        while we receive a copy via the signal.
        """
        motor = self._motor_hardware()
        point_idx = self._current_point_idx

        actual_pos = motor.get_pos()

        # Collect buffered data from the raw data buffer (populated by _on_new_raw_data)
        samples_collected = 0
        collected_data = {}
        
        for channel in self._scan_data.stream_data_mean.keys():
            if channel in self._ts_raw_data_buffer:
                # Get all buffered data for this channel and clear the buffer
                data_list = self._ts_raw_data_buffer[channel]
                collected_data[channel] = data_list.copy()
                samples_collected += len(data_list)
                # Clear the buffer for next position
                self._ts_raw_data_buffer[channel] = []
            else:
                collected_data[channel] = []

        # Update grid data
        grid_idx = self._scan_data.point_index_to_grid_index(point_idx)

        for channel, data_list in collected_data.items():
            if channel in self._scan_data.stream_data_raw:
                self._scan_data.stream_data_raw[channel][point_idx].extend(data_list)

                all_data = self._scan_data.stream_data_raw[channel][point_idx]
                if len(all_data) > 0:
                    mean_val = np.mean(all_data)
                    if self._scan_data.is_2d:
                        self._scan_data.stream_data_mean[channel][grid_idx] = mean_val
                    else:
                        self._scan_data.stream_data_mean[channel][grid_idx[0]] = mean_val

        result = {
            'point_index': point_idx,
            'target_position': self._current_pos_dict,
            'actual_position': actual_pos,
            'samples_collected': samples_collected,
        }

        self._advance_to_next_point(result)

    # =========================================================================
    # Continuous Line Data Binning
    # =========================================================================

    def _bin_line_data(
        self,
        line_index: int,
        position_time_buffer: list,
        raw_data_buffer: dict,
        data_start_time: float
    ) -> dict:
        """
        Bin collected time-series data to grid points for a completed line.
        
        Uses interpolated position data to determine bin boundaries at d/2
        distances between grid points. Data recorded at times between 
        boundary crossings is assigned to the corresponding grid point.
        
        Args:
            line_index: Zero-based line index.
            position_time_buffer: List of (timestamp, {axis: position}) tuples
                                  from position sampling.
            raw_data_buffer: Dict of channel_name -> list of raw data samples.
            data_start_time: Absolute timestamp when time-series started for
                            aligning raw data timestamps.
        
        Returns:
            Dict with binning results and statistics.
        """
        from scipy import interpolate
        
        if self._scan_data is None:
            self.log.error("No scan data available for binning")
            return {'success': False, 'error': 'No scan data'}
        
        # Get line metadata
        point_indices = self._scan_data.get_line_point_indices(line_index)
        n_points = len(point_indices)
        bin_boundaries = self._scan_data.get_bin_boundaries(line_index)
        fast_axis = self._scan_data.get_fast_axis()
        
        if n_points == 0 or len(bin_boundaries) == 0:
            self.log.warning(f"No points or boundaries for line {line_index}")
            return {'success': False, 'error': 'Empty line'}
        
        # Extract position vs time data along fast axis
        if len(position_time_buffer) < 2:
            self.log.warning(f"Insufficient position samples for line {line_index}: "
                           f"{len(position_time_buffer)} samples")
            return {'success': False, 'error': 'Insufficient position samples'}
        
        pos_times = np.array([t for t, _ in position_time_buffer])
        pos_values = np.array([p.get(fast_axis, 0) for _, p in position_time_buffer])

        # Trim stationary samples from start and end of position buffer.
        # The motor is briefly stationary after the move command (USB latency)
        # and after arriving at the end position. These constant-position
        # regions cause divide-by-zero in scipy interp1d (position→time is
        # not a function when multiple times map to the same position).
        pos_diff = np.diff(pos_values)
        moving_mask = np.abs(pos_diff) > 1e-7  # ~0.1 µm threshold
        moving_indices = np.where(moving_mask)[0]

        if len(moving_indices) < 2:
            self.log.warning(f"Motor did not move during line {line_index}: "
                           f"position stayed at {pos_values[0]*1000:.3f}mm")
            return {'success': False, 'error': 'Motor did not move'}

        # Keep one sample before first movement and one after last movement
        trim_start = max(0, moving_indices[0])
        trim_end = min(len(pos_values) - 1, moving_indices[-1] + 2)
        pos_times = pos_times[trim_start:trim_end]
        pos_values = pos_values[trim_start:trim_end]

        self.log.debug(f"Line {line_index}: trimmed position buffer from "
                      f"{len(position_time_buffer)} to {len(pos_values)} samples "
                      f"(removed {trim_start} leading + "
                      f"{len(position_time_buffer) - trim_end} trailing stationary)")

        # Determine scan direction
        scanning_positive = pos_values[-1] > pos_values[0]

        # Sort boundaries in scan order
        if scanning_positive:
            sorted_boundaries = np.sort(bin_boundaries)
        else:
            sorted_boundaries = np.sort(bin_boundaries)[::-1]

        # Note: We create position->time interpolation below, not time->position
        # since we need to find the times when motor crossed position boundaries

        # Find times when motor crossed each bin boundary
        # We need time -> position inverse, so we interpolate position -> time
        try:
            time_interp = interpolate.interp1d(
                pos_values, pos_times,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            boundary_times = time_interp(sorted_boundaries)
        except Exception as e:
            self.log.warning(f"Failed to interpolate boundary times: {e}")
            # Fallback: linearly divide the time range
            boundary_times = np.linspace(pos_times[0], pos_times[-1], len(sorted_boundaries))

        # Ensure boundary times are monotonically increasing
        boundary_times = np.sort(boundary_times)
        
        self.log.debug(f"Line {line_index}: {n_points} bins, boundary times: "
                      f"{boundary_times[0]:.3f}s to {boundary_times[-1]:.3f}s")
        
        # Get time-series data timing info
        ts_logic = self._time_series_logic()
        if ts_logic is None:
            self.log.error("Time series logic not available")
            return {'success': False, 'error': 'No time series logic'}
        
        # Estimate sample rate from time series logic
        try:
            sample_rate = ts_logic.data_rate if hasattr(ts_logic, 'data_rate') else 30000.0
        except Exception:
            sample_rate = 30000.0  # Default to 30 kHz
        
        # Bin the raw data for each channel
        samples_per_bin = []
        
        for channel, raw_data in raw_data_buffer.items():
            if channel not in self._scan_data.stream_data_mean:
                continue
            
            raw_array = np.array(raw_data) if not isinstance(raw_data, np.ndarray) else raw_data
            n_samples = len(raw_array)
            
            if n_samples == 0:
                self.log.warning(f"No data in channel {channel} for line {line_index}")
                continue
            
            # Calculate sample times relative to line start
            # FIX Finding #2: Align sample times with position buffer time reference
            # Position timestamps start from pos_times[0], so we align sample times similarly
            sample_times_raw = np.arange(n_samples) / sample_rate
            
            # Calculate time offset between data start and position sampling start
            # Both data_start_time and pos_times[0] are relative to _line_scan_start_time
            # If there's latency in data arrival, we need to account for it
            if pos_times[0] != 0:
                # Align sample times to same reference as position times
                # Assume first data sample arrived at data_start_time (passed as 0 if aligned)
                time_offset = 0  # data_start_time - line_scan_start_time offset (usually 0)
                sample_times = sample_times_raw + time_offset
            else:
                sample_times = sample_times_raw
            
            # Validate time alignment - warn if position and sample time ranges don't overlap
            pos_time_range = (pos_times[0], pos_times[-1])
            sample_time_range = (sample_times[0], sample_times[-1])
            if sample_time_range[1] < pos_time_range[0] or sample_time_range[0] > pos_time_range[1]:
                self.log.warning(
                    f"Time ranges don't overlap! Position: {pos_time_range}, Samples: {sample_time_range}. "
                    f"Check timestamp alignment."
                )
            
            # Find sample indices for each bin boundary
            boundary_sample_indices = np.searchsorted(sample_times, boundary_times)
            boundary_sample_indices = np.clip(boundary_sample_indices, 0, n_samples)
            
            # FIX Finding #3: Add assertion for clarity
            assert len(boundary_sample_indices) == n_points + 1, (
                f"Unexpected boundary count: got {len(boundary_sample_indices)}, expected {n_points + 1}"
            )
            
            # Bin data between consecutive boundaries
            for i, point_idx in enumerate(point_indices):
                # Get grid index for this point
                grid_idx = self._scan_data.point_index_to_grid_index(point_idx)
                
                # Sample range for this bin
                start_sample = boundary_sample_indices[i] if i < len(boundary_sample_indices) else 0
                end_sample = boundary_sample_indices[i + 1] if i + 1 < len(boundary_sample_indices) else n_samples
                
                # Extract bin data
                bin_data = raw_array[start_sample:end_sample]
                n_bin_samples = len(bin_data)
                
                if n_bin_samples > 0:
                    mean_val = np.mean(bin_data)
                    
                    # Store in scan_data
                    if self._scan_data.is_2d:
                        self._scan_data.stream_data_mean[channel][grid_idx] = mean_val
                    else:
                        self._scan_data.stream_data_mean[channel][grid_idx[0]] = mean_val
                    
                    # Store raw data for this bin
                    if channel in self._scan_data.stream_data_raw:
                        self._scan_data.stream_data_raw[channel][point_idx] = bin_data.tolist()
                    
                    samples_per_bin.append(n_bin_samples)
                else:
                    self.log.debug(f"Empty bin at point {point_idx}, grid {grid_idx}")
                    samples_per_bin.append(0)
        
        # Store actual positions from interpolation
        if self._scan_data.actual_positions is not None:
            grid_positions = self._scan_data.get_line_grid_positions(line_index)
            for i, point_idx in enumerate(point_indices):
                for j, axis in enumerate(self._scan_data.scan_axes):
                    self._scan_data.actual_positions[point_idx, j] = grid_positions[i, j]
        
        # Calculate statistics
        result = {
            'success': True,
            'line_index': line_index,
            'n_points': n_points,
            'samples_per_bin': samples_per_bin,
            'avg_samples_per_bin': np.mean(samples_per_bin) if samples_per_bin else 0,
            'min_samples_per_bin': min(samples_per_bin) if samples_per_bin else 0,
            'max_samples_per_bin': max(samples_per_bin) if samples_per_bin else 0,
        }
        
        self.log.info(f"Line {line_index} binned: {n_points} points, "
                     f"avg {result['avg_samples_per_bin']:.0f} samples/bin "
                     f"(range: {result['min_samples_per_bin']}-{result['max_samples_per_bin']})")
        
        return result

    def _get_line_raw_data_buffer(self) -> dict:
        """
        Get a copy of the current raw data buffer for the line being scanned.
        
        Returns:
            Dict of channel_name -> list of raw data samples.
        """
        return {ch: data.copy() for ch, data in self._ts_raw_data_buffer.items()}

    def _clear_line_raw_data_buffer(self):
        """Clear the raw data buffer after processing a line."""
        for ch in self._ts_raw_data_buffer:
            self._ts_raw_data_buffer[ch] = []
