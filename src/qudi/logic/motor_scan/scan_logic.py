# -*- coding: utf-8 -*-
"""
Motor-based XY scanning logic module for qudi.

This module provides scanning functionality using motor-controlled stages (e.g., Thorlabs KDC101)
with three data acquisition modes:
1. CONTINUOUS_STREAM: Motors move continuously while streaming data is binned to grid positions
2. STEP_ODMR: Motors stop at each grid position, execute ODMR scans, then proceed
3. CONTINUOUS_FREQ_TRACK: Motors move continuously, absolute frequency from tracking lock

Example config:

    motor_scan_logic:
        module.Class: 'motor_scan.scan_logic.MotorScanLogic'
        connect:
            motor_hardware: thorlabs_xy_stage
            odmr_logic: odmr_logic  # optional, for STEP_ODMR mode
            time_series_logic: time_series_reader_logic  # optional, for CONTINUOUS_STREAM mode
        options:
            default_scan_mode: 'STEP_ODMR'  # or 'CONTINUOUS_STREAM'
            position_poll_interval: 0.05  # seconds, for position feedback during continuous scan
            odmr_fit_function: 'fit_hyperfine'  # fitting function to use for ODMR analysis
"""

import time
import os
import sys
import datetime
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from PySide2 import QtCore

from qudi.core.module import LogicBase
from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.util.mutex import RecursiveMutex

from .data_structures import ScanMode, ScanPattern, ScanState, MotorScanData
from .motor_control import MotorControlMixin
from .data_processing import DataProcessingMixin
from .data_saving import DataSavingMixin
from .data_loading import DataLoadingMixin
from .continuous_line_scan import ContinuousLineScanMixin
from .hw_sync_scan import HwSyncScanMixin

# Add qudi-core root to path to find my_software (same as sensitivity_sweep_logic)
# Get path to qudi-core root (4 levels up from this file in the package)
qudi_core_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
if qudi_core_root not in sys.path:
    sys.path.insert(0, qudi_core_root)


class MotorScanLogic(HwSyncScanMixin, ContinuousLineScanMixin, MotorControlMixin, DataProcessingMixin, DataSavingMixin, DataLoadingMixin, LogicBase):
    """
    Logic module for motor-based XY scanning.
    
    Orchestrates motor movement and data acquisition for three modes:
    - CONTINUOUS_STREAM: Motors move continuously, streaming data binned to grid
    - STEP_ODMR: Stop at each point, execute ODMR scan, fit and store results
    - CONTINUOUS_FREQ_TRACK: Motors move continuously, absolute frequency from lock
    
    This class combines functionality from several mixins:
    - ContinuousLineScanMixin: Continuous line-by-line scanning for CONTINUOUS_* modes
    - MotorControlMixin: Movement, homing, position polling
    - DataProcessingMixin: Data acquisition, ODMR fitting, streaming
    - DataSavingMixin: Saving data to files, figure generation
    """

    _POINT_POSITION_TOLERANCE = 100e-6  # 100 um
    _POINT_MOVE_TIMEOUT = 120.0
    _POINT_MOVE_MAX_RETRIES = 3
    
    # Connectors
    _motor_hardware = Connector(interface='MotorInterface', name='motor_hardware')
    _odmr_logic = Connector(interface='LogicBase', name='odmr_logic', optional=True)
    _time_series_logic = Connector(interface='LogicBase',
                                   name='time_series_logic', optional=True)
    # ODMR tracking logic (odmr_tracking_logic), for CONTINUOUS_FREQ_TRACK.
    _tracking_logic = Connector(interface='LogicBase', name='tracking_logic',
                                optional=True)
    # Shared Red Pitaya stream owner. When connected, KDC_HW_SYNC routes the board's
    # single push stream through this owner (begin/end_scan_stream) so a marker-mode
    # scan cleanly takes over from / restores the monitor stream instead of starting
    # a competing second stream. Optional: without it the scan falls back to grabbing
    # the pyrpl scan module directly (headless tests).
    _streamer = Connector(interface='DataInStreamInterface', name='streamer',
                          optional=True)
    # Tracking hardware (redpitaya_odmr_lock). Required only for the
    # KDC_HW_SYNC_MULTIRES mode: the tracker owns the acquisition engine and the
    # trace stream, and this logic asks it to add x/y markers and to drain and
    # reconstruct the per-resonance traces for spatial binning. The user configures
    # and starts the trace stream in the ODMR Tracking GUI; the FPGA lock may
    # independently be on (closed loop) or off (open loop).
    _tracking_hw = Connector(interface='OdmrTrackingInterface',
                             name='tracking_hw', optional=True)

    # Config options
    _default_scan_mode = ConfigOption(
        name='default_scan_mode',
        default='STEP_ODMR',
        constructor=lambda x: ScanMode[x] if isinstance(x, str) else ScanMode(x)
    )
    _position_poll_interval = ConfigOption(
        name='position_poll_interval',
        default=0.05  # 50 ms
    )
    _odmr_fit_function = ConfigOption(
        name='odmr_fit_function',
        default='fit_hyperfine'
    )
    _require_fit_function = ConfigOption(
        name='require_fit_function',
        default=True,
        missing='info'
    )
    _save_thumbnails = ConfigOption(
        name='save_thumbnails',
        default=True
    )
    # Fit parameters (matching sensitivity_sweep_logic defaults)
    _fit_feature_prominence = ConfigOption(
        name='fit_feature_prominence',
        default=0.02,
        missing='info'
    )
    _fit_n_most_prominent_peaks = ConfigOption(
        name='fit_n_most_prominent_peaks',
        default=5,
        missing='info'
    )
    _fit_min_feature_height = ConfigOption(
        name='fit_min_feature_height',
        default=0.02,
        missing='info'
    )
    _save_odmr_fit_plots = ConfigOption(
        name='save_odmr_fit_plots',
        default=True,
        missing='info'
    )
    # Robust fitting parameters
    _hyperfine_spacing_hz = ConfigOption(
        name='hyperfine_spacing_hz',
        default=2.158e6,  # N14 hyperfine splitting
        missing='info'
    )
    _max_pair_distance_hz = ConfigOption(
        name='max_pair_distance_hz',
        default=None,  # If None, defaults to 0.75 * hyperfine_spacing_hz
        missing='info'
    )
    _use_robust_fitting = ConfigOption(
        name='use_robust_fitting',
        default=False,  # If True, use fit_odmr_robust with auto quality assessment
        missing='info'
    )
    _home_before_scan = ConfigOption(
        name='home_before_scan',
        default=False,
        missing='info'
    )
    _require_homed_before_scan = ConfigOption(
        name='require_homed_before_scan',
        default=True,
        missing='info'
    )
    _lock_status_poll_interval = ConfigOption(
        name='lock_status_poll_interval',
        default=0.5,  # 500 ms - poll lock status during CONTINUOUS_FREQ_TRACK mode
        missing='info'
    )
    # Continuous line scanning options (for CONTINUOUS_* modes)
    _continuous_line_mode_enabled = ConfigOption(
        name='continuous_line_mode',
        default=True,  # Enable continuous line scanning (no stopping at grid points)
        missing='info'
    )
    _position_sample_interval = ConfigOption(
        name='position_sample_interval',
        default=0.05,  # 50ms = 20 Hz position sampling during continuous line scan
        missing='info'
    )
    _auto_save_on_completion = ConfigOption(
        name='auto_save_on_completion',
        default=True,  # Automatically save scan data when scan completes
        missing='info'
    )
    # Unused since KDC_HW_SYNC reads through the 'streamer' connector; kept so
    # existing config files that still set them load without errors.
    _redpitaya_hostname = ConfigOption(
        name='redpitaya_hostname',
        default=None,
        missing='nothing'
    )
    _redpitaya_config_name = ConfigOption(
        name='redpitaya_config_name',
        default=None,
        missing='nothing'
    )
    _hw_sync_channel = ConfigOption(
        name='hw_sync_channel',
        default='demod',  # FPGA stream input: 'demod' or 'ftw_corr'
        missing='nothing'
    )
    _hw_sync_pulse_width = ConfigOption(
        name='hw_sync_pulse_width',
        default=1e-4,  # 100 us KDC trigger pulse width
        missing='nothing'
    )
    _hw_sync_trig_port = ConfigOption(
        name='hw_sync_trig_port',
        default=1,  # KDC101 TRIG SMA port (1 or 2)
        missing='nothing'
    )
    _hw_sync_margin_frac = ConfigOption(
        name='hw_sync_margin_frac',
        default=0.5,  # fast-axis over-travel past outer bin boundaries, in bins
        missing='nothing'
    )
    _hw_sync_runup = ConfigOption(
        name='hw_sync_runup',
        default=0.5e-3,  # absolute fast-axis run-up (m) before the first (throwaway)
                         # position-step pulse so the stage is at constant velocity.
        missing='nothing'
    )

    # Status variables (persistent across sessions)
    _scan_ranges = StatusVar(
        name='scan_ranges',
        default={'x': (0.0, 0.01), 'y': (0.0, 0.01)}  # 10mm default
    )
    _scan_resolution = StatusVar(
        name='scan_resolution',
        default={'x': 10, 'y': 10}
    )
    _active_scan_mode = StatusVar(name='active_scan_mode', default=None)
    _active_scan_pattern = StatusVar(name='active_scan_pattern', default='SNAKE_X')
    # KDC_HW_SYNC: whether to keep+save the full per-bin time-traces (and the faithful
    # demod trace). Default False -> only the per-bin MEAN map (what the GUI plots) is
    # built and saved; this also avoids holding every bin's samples in memory on large
    # scans. Toggled from the GUI ("Save full traces").
    _save_full_traces = StatusVar(name='save_full_traces', default=False)
    
    # Signals
    sigScanStateChanged = QtCore.Signal(object)  # ScanState
    sigScanDataUpdated = QtCore.Signal()
    sigScanPointCompleted = QtCore.Signal(int, dict)  # point_index, result_dict
    sigScanCompleted = QtCore.Signal(object)  # MotorScanData
    sigPositionUpdated = QtCore.Signal(dict)  # current position dict
    sigScanSettingsChanged = QtCore.Signal(dict)
    sigSaveStateChanged = QtCore.Signal(bool)  # True when saving, False when done
    sigHomingStateChanged = QtCore.Signal(bool)  # True when homing, False when done
    sigMovementStateChanged = QtCore.Signal(bool)  # True when moving to position, False when done
    sigLockLostDuringScan = QtCore.Signal()  # Emitted when lock is lost during CONTINUOUS_FREQ_TRACK
    sigLockStatusUpdated = QtCore.Signal(bool)  # Emitted with current lock status (for GUI indicator)
    sigLoadedDataChanged = QtCore.Signal(bool)  # True when data loaded, False when cleared
    sigScanStatusMessage = QtCore.Signal(str)  # Descriptive status text for GUI status bar

    # Internal signals for async operations (run on logic thread via QueuedConnection)
    _sigNextPoint = QtCore.Signal()
    _sigDoHoming = QtCore.Signal(object)  # axes list, invokes homing on logic thread
    _sigDoStartScan = QtCore.Signal(object)  # axes list, invokes scan start on logic thread
    _sigDoMove = QtCore.Signal(dict)  # position dict, invokes move on logic thread

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._thread_lock = RecursiveMutex()

        # Scan state
        self._scan_state = ScanState.IDLE
        self._scan_data: Optional[MotorScanData] = None
        self._stop_requested = False

        # Timing
        self._scan_start_time = 0.0

        # Non-blocking operation state
        self._current_point_idx = 0
        self._current_pos_dict = {}
        self._waiting_for_motor = False
        self._waiting_for_odmr = False
        self._odmr_scan_complete = False

        # Timers for non-blocking polling
        self._motor_poll_timer = None
        self._position_timer = None

        # Fit function reference (loaded dynamically)
        self._fit_function = None
        
        # Current scan folder for saving fit plots
        self._current_scan_folder = None
        
        # Time series streaming state
        self._ts_we_started = False  # True if we started the time series reader
        self._ts_raw_data_buffer = {}  # Buffer for incoming raw data per channel
        self._ts_connected = False  # Track if we're connected to sigNewRawData

        # CONTINUOUS_FREQ_TRACK mode state
        self._tracking_zero_crossing = None  # Hz, baseline from tracking logic at scan start
        self._tracking_zero_crossing_history = []  # List of (point_idx, zero_crossing) for segments
        self._paused_due_to_lock_loss = False  # Distinguish auto-pause from user pause
        self._lock_status_timer = None  # Timer for polling lock status

        # Non-blocking homing state
        self._homing_poll_timer = None
        self._homing_in_progress = False
        self._homing_axes = []
        self._homing_start_time = 0.0
        self._homing_timeout = 120.0  # seconds

        # Non-blocking manual movement state
        self._moving_in_progress = False
        self._point_move_start_time = 0.0
        self._point_move_retries = 0

        # Initialize mixin state
        self._init_position_sampling_state()  # From MotorControlMixin
        self._init_continuous_line_state()    # From ContinuousLineScanMixin
        self._init_hw_sync_state()            # From HwSyncScanMixin
        self._init_data_loading()             # From DataLoadingMixin

    def on_activate(self):
        """Initialize the module."""
        # Validate connectors
        motor = self._motor_hardware()
        if motor is None:
            self.log.error("Motor hardware connector not available!")
            return
        
        # Get motor constraints
        self._motor_constraints = motor.get_constraints()
        self.log.info(f"Motor axes available: {list(self._motor_constraints.keys())}")
        
        # Set default scan mode if not set
        if self._active_scan_mode is None:
            self._active_scan_mode = self._default_scan_mode.value
        
        # Try to load fit function
        self._load_fit_function()
        
        # Connect internal signals
        self._sigNextPoint.connect(self._process_next_point, QtCore.Qt.QueuedConnection)
        self._sigDoHoming.connect(self._do_homing_async, QtCore.Qt.QueuedConnection)
        self._sigDoStartScan.connect(self._do_start_scan_async, QtCore.Qt.QueuedConnection)
        self._sigDoMove.connect(self._do_move_async, QtCore.Qt.QueuedConnection)

        # Initialize motor poll timer (non-blocking motor movement check)
        self._motor_poll_timer = QtCore.QTimer()
        self._motor_poll_timer.setSingleShot(True)
        self._motor_poll_timer.timeout.connect(self._on_motor_poll_timeout, QtCore.Qt.QueuedConnection)

        # Initialize position timer (for continuous mode data collection)
        self._position_timer = QtCore.QTimer()
        self._position_timer.setSingleShot(True)
        self._position_timer.timeout.connect(self._on_position_poll_timeout, QtCore.Qt.QueuedConnection)

        # Initialize lock status polling timer (for CONTINUOUS_FREQ_TRACK mode)
        self._lock_status_timer = QtCore.QTimer()
        self._lock_status_timer.setSingleShot(False)  # Repeating timer
        self._lock_status_timer.timeout.connect(
            self._on_lock_status_poll_timeout, QtCore.Qt.QueuedConnection
        )

        # Initialize homing poll timer (non-blocking homing check)
        self._homing_poll_timer = QtCore.QTimer()
        self._homing_poll_timer.setSingleShot(True)
        self._homing_poll_timer.timeout.connect(
            self._on_homing_poll_timeout, QtCore.Qt.QueuedConnection
        )

        self.log.info("MotorScanLogic activated.")
    
    def _set_monitor_stream_input(self, input_mode):
        """Select what the monitor stream (redpitaya_stream) carries: 'demod'
        (demodulated error) or 'ftw_corr' (frequency correction)."""
        streamer = None
        try:
            streamer = self._streamer()
        except Exception:
            pass
        if streamer is None:
            try:
                streamer = self._time_series_logic()._streamer()
            except Exception:
                streamer = None
        if streamer is None or not hasattr(streamer, 'set_stream_input'):
            self.log.warning('No stream module to set the input to %s', input_mode)
            return
        try:
            streamer.set_stream_input(input_mode)
        except Exception as e:
            self.log.warning(f'Could not set the stream input to {input_mode}: {e}')

    def on_deactivate(self):
        """Clean up on deactivation."""
        # Stop any running scan
        if self._scan_state != ScanState.IDLE:
            self.stop_scan()

        # Stop timers
        if self._motor_poll_timer is not None:
            self._motor_poll_timer.stop()
        if self._position_timer is not None:
            self._position_timer.stop()
        if self._lock_status_timer is not None:
            self._lock_status_timer.stop()
        if self._homing_poll_timer is not None:
            self._homing_poll_timer.stop()

        # Disconnect ODMR signals if connected
        odmr = self._odmr_logic()
        if odmr is not None and hasattr(odmr, 'sigScanStateUpdated'):
            try:
                odmr.sigScanStateUpdated.disconnect(self._on_odmr_scan_state_changed)
            except (TypeError, RuntimeError):
                pass

        # Disconnect time series signals if connected
        self._disconnect_time_series_signals()

        # Release the shared pyrpl instance held for KDC_HW_SYNC (if any)
        self._release_scan_module()

        # Disconnect signals
        try:
            self._sigNextPoint.disconnect()
        except:
            pass
        try:
            self._sigDoHoming.disconnect()
        except:
            pass
        try:
            self._sigDoStartScan.disconnect()
        except:
            pass
        try:
            self._sigDoMove.disconnect()
        except:
            pass

        self.log.info("MotorScanLogic deactivated.")
    
    def _load_fit_function(self):
        """Load the ODMR fit function from my_software.tools.fitting."""
        try:
            from my_software.tools.fitting import (
                fit_hyperfine,
                fit_odmr_robust,
                fit_odmr_zero_crossing
            )
            self._fit_function = fit_hyperfine
            self._fit_function_robust = fit_odmr_robust
            self._fit_function_zero_crossing = fit_odmr_zero_crossing
            self.log.info("Loaded ODMR fitting functions from my_software.tools.fitting")
        except ImportError as e:
            if self._require_fit_function:
                self.log.warning(f"Could not load fit function: {e}. ODMR fitting will be unavailable.")
            else:
                self.log.debug(f"Fit function not loaded (optional): {e}")
            self._fit_function = None
            self._fit_function_robust = None
            self._fit_function_zero_crossing = None
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def scan_state(self) -> ScanState:
        """Current scan state."""
        return self._scan_state
    
    @property
    def is_scanning(self) -> bool:
        """True if a scan is in progress (including initialization phase)."""
        return self._scan_state in (ScanState.INITIALIZING, ScanState.RUNNING, ScanState.PAUSED)
    
    @property
    def scan_data(self) -> Optional[MotorScanData]:
        """Current scan data."""
        with self._thread_lock:
            return self._scan_data
    
    @property
    def motor_constraints(self) -> Dict:
        """Motor hardware constraints."""
        return self._motor_constraints.copy()
    
    @property
    def available_axes(self) -> List[str]:
        """List of available motor axes."""
        return list(self._motor_constraints.keys())
    
    @property
    def scan_settings(self) -> Dict:
        """Current scan settings."""
        return {
            'scan_mode': self._active_scan_mode,
            'scan_pattern': self._active_scan_pattern,
            'scan_ranges': self._scan_ranges.copy(),
            'scan_resolution': self._scan_resolution.copy(),
        }
    
    @property
    def current_position(self) -> Dict[str, float]:
        """Get current motor position."""
        try:
            return self._motor_hardware().get_pos()
        except Exception as e:
            self.log.warning(f"Failed to get motor position: {e}")
            return {}

    def _motor_is_homed_for_axes(self, axes: List[str]) -> bool:
        """Return True if the motor driver reports a valid home for all axes."""
        motor = self._motor_hardware()
        if motor is None or not hasattr(motor, 'is_homed'):
            return True

        try:
            return bool(motor.is_homed(list(axes)))
        except TypeError:
            try:
                return bool(motor.is_homed())
            except Exception as e:
                self.log.warning(f"Could not query motor homed state: {e}")
                return False
        except Exception as e:
            self.log.warning(f"Could not query motor homed state: {e}")
            return False
    
    @property
    def scan_pattern(self) -> ScanPattern:
        """Current scan pattern."""
        if isinstance(self._active_scan_pattern, str):
            return ScanPattern[self._active_scan_pattern]
        return ScanPattern(self._active_scan_pattern)
    
    # =========================================================================
    # Settings Methods
    # =========================================================================
    
    @property
    def save_full_traces(self) -> bool:
        """Whether KDC_HW_SYNC scans keep+save the full per-bin time-traces (True) or
        only the per-bin mean map (False, default)."""
        return bool(self._save_full_traces)

    @QtCore.Slot(bool)
    def set_save_full_traces(self, enabled: bool):
        """Enable/disable keeping+saving the full per-bin time-traces for KDC_HW_SYNC
        scans (GUI toggle). When off, only the mean map is built and saved (lower
        memory + disk). Cannot change mid-scan."""
        with self._thread_lock:
            if self.is_scanning:
                self.log.error("Cannot change full-trace saving while scanning.")
                return
            self._save_full_traces = bool(enabled)
            self.log.info("KDC_HW_SYNC full per-bin time-trace saving: %s",
                          "ON" if self._save_full_traces else "OFF (mean only)")
            self.sigScanSettingsChanged.emit(self.scan_settings)

    @QtCore.Slot(str)
    def set_scan_mode(self, mode: Union[str, ScanMode]):
        """Set the scan mode."""
        with self._thread_lock:
            if self.is_scanning:
                self.log.error("Cannot change scan mode while scanning.")
                return
            
            if isinstance(mode, str):
                mode = ScanMode[mode]
            
            self._active_scan_mode = mode.value
            self.sigScanSettingsChanged.emit(self.scan_settings)
            self.log.info(f"Scan mode set to: {mode.name}")
    
    @QtCore.Slot(str)
    def set_scan_pattern(self, pattern: Union[str, ScanPattern]):
        """
        Set the scan pattern.
        
        Args:
            pattern: Scan pattern (LINE_BY_LINE_X, SNAKE_X, LINE_BY_LINE_Y, SNAKE_Y)
        """
        with self._thread_lock:
            if self.is_scanning:
                self.log.error("Cannot change scan pattern while scanning.")
                return
            
            if isinstance(pattern, str):
                pattern = ScanPattern[pattern]
            
            self._active_scan_pattern = pattern.name
            self.sigScanSettingsChanged.emit(self.scan_settings)
            self.log.info(f"Scan pattern set to: {pattern.name}")
    
    @QtCore.Slot(dict)
    def set_scan_ranges(self, ranges: Dict[str, Tuple[float, float]]):
        """
        Set scan ranges for each axis.
        
        Args:
            ranges: Dict mapping axis name to (start, stop) tuple.
        """
        with self._thread_lock:
            if self.is_scanning:
                self.log.error("Cannot change scan ranges while scanning.")
                return
            
            # Validate against motor constraints
            for axis, (start, stop) in ranges.items():
                if axis not in self._motor_constraints:
                    self.log.warning(f"Axis '{axis}' not in motor constraints, ignoring.")
                    continue
                
                constraints = self._motor_constraints[axis]
                pos_min = constraints.get('pos_min', -float('inf'))
                pos_max = constraints.get('pos_max', float('inf'))
                
                # Clamp to valid range
                start = max(pos_min, min(pos_max, start))
                stop = max(pos_min, min(pos_max, stop))
                
                self._scan_ranges[axis] = (start, stop)
            
            self.sigScanSettingsChanged.emit(self.scan_settings)
    
    @QtCore.Slot(dict)
    def set_scan_resolution(self, resolution: Dict[str, int]):
        """
        Set scan resolution (number of points) for each axis.
        
        Args:
            resolution: Dict mapping axis name to number of points.
        """
        with self._thread_lock:
            if self.is_scanning:
                self.log.error("Cannot change scan resolution while scanning.")
                return
            
            for axis, points in resolution.items():
                if axis in self._scan_ranges:
                    self._scan_resolution[axis] = max(1, int(points))
            
            self.sigScanSettingsChanged.emit(self.scan_settings)
    
    # =========================================================================
    # Scan Control Methods  
    # =========================================================================
    
    @QtCore.Slot(list)
    def start_scan(self, axes: List[str] = None):
        """
        Start a new scan (non-blocking).
        
        This method returns immediately. The actual scan setup and execution
        runs asynchronously on the logic thread. If home_before_scan is enabled,
        homes the stages before starting the scan.
        
        Args:
            axes: List of axes to scan. Defaults to ['x', 'y'] or ['x'] based on config.
        """
        with self._thread_lock:
            if self.is_scanning:
                self.log.error("Scan already in progress.")
                return
            
            if self._homing_in_progress:
                self.log.error("Cannot start scan while homing is in progress.")
                return

            if self._moving_in_progress:
                self.log.error("Cannot start scan while manual movement is in progress.")
                return

            motor = self._motor_hardware()
            if motor is not None and hasattr(motor, 'is_moving'):
                try:
                    if motor.is_moving():
                        self.log.error("Cannot start scan while motor hardware is moving.")
                        return
                except Exception as e:
                    self.log.warning(f"Could not query motor movement state: {e}")
            
            # Default to available axes
            if axes is None:
                axes = [a for a in ['x', 'y'] if a in self._scan_ranges]
            
            # Quick validation of axes
            for axis in axes:
                if axis not in self._scan_ranges:
                    self.log.error(f"Axis '{axis}' not configured. Available: {list(self._scan_ranges.keys())}")
                    return

            if self._require_homed_before_scan and not self._home_before_scan:
                if not self._motor_is_homed_for_axes(axes):
                    self.log.error(
                        f"Cannot start scan: axes {axes} are not homed. "
                        f"Home the stages first or enable home_before_scan."
                    )
                    self.sigScanStatusMessage.emit('Scan aborted: stages not homed')
                    return

            # Clear any loaded data when starting a new scan
            if self._viewing_loaded_data:
                self._loaded_scan_data = None
                self._viewing_loaded_data = False
                self._loaded_data_folder = None
                self.sigLoadedDataChanged.emit(False)

            # Mark as initializing (prevents double-start, disables UI)
            self._scan_state = ScanState.INITIALIZING
            self.sigScanStateChanged.emit(self._scan_state)
            
            # Emit signal to run scan setup asynchronously on logic thread
            self._sigDoStartScan.emit(list(axes))

    @QtCore.Slot(object)
    def _do_start_scan_async(self, axes: List[str]):
        """
        Perform the actual scan setup and start (runs on logic thread via signal).
        
        This method is invoked via _sigDoStartScan with QueuedConnection to ensure
        it runs on the logic's thread, not blocking the GUI thread.
        
        Args:
            axes: List of axes to scan.
        """
        try:
            motor = self._motor_hardware()
            if self._moving_in_progress:
                self.log.error("Scan start aborted: manual movement is still in progress.")
                self._scan_state = ScanState.IDLE
                self.sigScanStateChanged.emit(self._scan_state)
                return
            if motor is not None and hasattr(motor, 'is_moving'):
                try:
                    if motor.is_moving():
                        self.log.error("Scan start aborted: motor hardware is still moving.")
                        self._scan_state = ScanState.IDLE
                        self.sigScanStateChanged.emit(self._scan_state)
                        return
                except Exception as e:
                    self.log.warning(f"Could not query motor movement state: {e}")

            # Home stages before scan if configured
            if self._home_before_scan:
                self.log.info("Homing stages before scan...")
                if motor is not None:
                    try:
                        result = motor.calibrate(axes)
                        if result != 0:
                            self.log.error(
                                f"Homing failed (error code {result}). "
                                f"Aborting scan — encoder zero reference may be "
                                f"incorrect, positions would be wrong."
                            )
                            self.sigScanStatusMessage.emit(
                                'Scan aborted: homing failed'
                            )
                            self._scan_state = ScanState.IDLE
                            self.sigScanStateChanged.emit(self._scan_state)
                            return
                    except Exception as e:
                        self.log.error(f"Homing failed: {e}. Aborting scan.")
                        self._scan_state = ScanState.IDLE
                        self.sigScanStateChanged.emit(self._scan_state)
                        return
                else:
                    self.log.error("Motor hardware not available. Aborting scan.")
                    self._scan_state = ScanState.IDLE
                    self.sigScanStateChanged.emit(self._scan_state)
                    return

            if self._require_homed_before_scan and not self._motor_is_homed_for_axes(axes):
                self.log.error(
                    f"Scan start aborted: axes {axes} do not report a valid "
                    f"home reference."
                )
                self.sigScanStatusMessage.emit('Scan aborted: stages not homed')
                self._scan_state = ScanState.IDLE
                self.sigScanStateChanged.emit(self._scan_state)
                return
            
            # Determine scan mode
            mode = ScanMode(self._active_scan_mode) if isinstance(self._active_scan_mode, int) else self._active_scan_mode
            
            # Check required connectors for the mode
            if mode == ScanMode.STEP_ODMR:
                if self._odmr_logic() is None:
                    self.log.error("ODMR logic not connected. Cannot perform STEP_ODMR scan.")
                    self._scan_state = ScanState.IDLE
                    self.sigScanStateChanged.emit(self._scan_state)
                    return
            elif mode == ScanMode.CONTINUOUS_STREAM:
                if self._time_series_logic() is None:
                    self.log.error("Time series logic not connected. Cannot perform CONTINUOUS_STREAM scan.")
                    self._scan_state = ScanState.IDLE
                    self.sigScanStateChanged.emit(self._scan_state)
                    return
            elif mode == ScanMode.CONTINUOUS_FREQ_TRACK:
                # Validate all required components for frequency tracking
                tracking_logic = self._tracking_logic()
                if tracking_logic is None:
                    self.log.error("ODMR frequency tracking logic not connected. "
                                   "Cannot perform CONTINUOUS_FREQ_TRACK scan.")
                    self._scan_state = ScanState.IDLE
                    self.sigScanStateChanged.emit(self._scan_state)
                    return
                if self._time_series_logic() is None:
                    self.log.error("Time series logic not connected. "
                                   "Cannot perform CONTINUOUS_FREQ_TRACK scan.")
                    self._scan_state = ScanState.IDLE
                    self.sigScanStateChanged.emit(self._scan_state)
                    return
                # The correction stream belongs to the live resonance, so this
                # mode needs single-resonance tracking.
                if int(tracking_logic.num_resonances) != 1:
                    self.log.error("CONTINUOUS_FREQ_TRACK needs N=1 tracking; use "
                                   "KDC_HW_SYNC_MULTIRES for two resonances.")
                    self._scan_state = ScanState.IDLE
                    self.sigScanStateChanged.emit(self._scan_state)
                    return
                # Check lock is enabled
                if not tracking_logic.tracking_active:
                    self.log.error("Lock not enabled. Enable frequency tracking in the "
                                   "ODMR Tracking GUI before starting scan.")
                    self._scan_state = ScanState.IDLE
                    self.sigScanStateChanged.emit(self._scan_state)
                    return
                # Check zero-crossing is available
                zero_crossing = tracking_logic.resonance_frequencies[0]
                if zero_crossing is None:
                    self.log.error("No zero-crossing frequency available. Perform ODMR fit "
                                   "in Tracking GUI first.")
                    self._scan_state = ScanState.IDLE
                    self.sigScanStateChanged.emit(self._scan_state)
                    return
                # Store baseline for this scan
                self._tracking_zero_crossing = zero_crossing
                self._tracking_zero_crossing_history = [(0, zero_crossing)]
                self._paused_due_to_lock_loss = False
                self.log.info(f"Frequency tracking scan with zero-crossing: "
                              f"{zero_crossing / 1e9:.6f} GHz")
            elif mode == ScanMode.POSITION_ONLY:
                # POSITION_ONLY mode requires no measurement connectors
                self.log.info("POSITION_ONLY mode - stage movement only, no data acquisition")
            elif mode == ScanMode.KDC_HW_SYNC:
                # Needs the Red Pitaya data stream module ('streamer' connector)
                # and a motor driver exposing setup_position_trigger.
                if self._get_scan_module() is None:
                    self.log.error("Red Pitaya data stream not available. "
                                   "Cannot perform KDC_HW_SYNC scan (connect "
                                   "'streamer' to redpitaya_stream).")
                    self._scan_state = ScanState.IDLE
                    self.sigScanStateChanged.emit(self._scan_state)
                    return
                if not hasattr(motor, 'setup_position_trigger'):
                    self.log.error("Motor hardware does not support setup_position_trigger. "
                                   "Cannot perform KDC_HW_SYNC scan.")
                    self._scan_state = ScanState.IDLE
                    self.sigScanStateChanged.emit(self._scan_state)
                    return
            elif mode == ScanMode.KDC_HW_SYNC_MULTIRES:
                # Needs the tracker (which owns the acquisition engine and the
                # trace stream) with STREAMING already running, and a motor
                # driver exposing setup_position_trigger.
                hw = self._get_tracking_hw()
                if hw is None:
                    self.log.error("KDC_HW_SYNC_MULTIRES needs the 'tracking_hw' "
                                   "connector (tracking hardware).")
                    self._scan_state = ScanState.IDLE
                    self.sigScanStateChanged.emit(self._scan_state)
                    return
                if not getattr(hw, 'nslots', 0):
                    self.log.error("KDC_HW_SYNC_MULTIRES: tracker reports 0 slots.")
                    self._scan_state = ScanState.IDLE
                    self.sigScanStateChanged.emit(self._scan_state)
                    return
                if not bool(getattr(hw, 'trace_stream_active', False)):
                    message = ('KDC_HW_SYNC_MULTIRES needs Start Stream in the '
                               'ODMR Tracking GUI; Start Tracking '
                               'is optional.')
                    self.log.error(message)
                    self.sigScanStatusMessage.emit(message)
                    self._scan_state = ScanState.IDLE
                    self.sigScanStateChanged.emit(self._scan_state)
                    return
                if not hasattr(motor, 'setup_position_trigger'):
                    self.log.error("Motor hardware does not support setup_position_trigger. "
                                   "Cannot perform KDC_HW_SYNC_MULTIRES scan.")
                    self._scan_state = ScanState.IDLE
                    self.sigScanStateChanged.emit(self._scan_state)
                    return

            # Create scan configuration
            scan_axes = tuple(axes)
            scan_range = tuple(self._scan_ranges[a] for a in axes)
            scan_resolution = tuple(self._scan_resolution[a] for a in axes)
            
            # Get current scan pattern
            pattern = self.scan_pattern
            
            # Create scan data container
            self._scan_data = MotorScanData(
                scan_axes=scan_axes,
                scan_range=scan_range,
                scan_resolution=scan_resolution,
                scan_mode=mode,
                scan_pattern=pattern,
                timestamp_start=datetime.datetime.now()
            )
            
            # Initialize data arrays and set up data acquisition
            if mode == ScanMode.CONTINUOUS_STREAM:
                ts_logic = self._time_series_logic()
                channel_names = ts_logic.active_channel_names if ts_logic else []
                self._scan_data.initialize_data_arrays(channel_names)

                # Clear raw data buffer
                self._ts_raw_data_buffer = {ch: [] for ch in channel_names}

                # Check if time series is already running (e.g., from time_series GUI)
                ts_already_running = ts_logic.module_state() == 'locked' if ts_logic else False

                if ts_already_running:
                    # Time series already running - just connect to its signals
                    self._ts_we_started = False
                else:
                    # Start the time series reader ourselves
                    self._ts_we_started = True
                    if ts_logic is not None:
                        ts_logic.start_reading()

                # Connect to sigNewRawData for non-blocking data reception
                self._connect_time_series_signals()
                self._current_scan_folder = None  # Will be set on save
            elif mode == ScanMode.CONTINUOUS_FREQ_TRACK:
                # Initialize with absolute frequency channel
                channel_names = ['absolute_frequency']
                self._scan_data.initialize_data_arrays(channel_names)

                # Clear raw data buffer (will receive FTW values, convert to frequency)
                self._ts_raw_data_buffer = {ch: [] for ch in channel_names}

                # Stream the frequency correction instead of the demodulated error
                self._set_monitor_stream_input('ftw_corr')

                # Connect to time series for data (same as CONTINUOUS_STREAM)
                ts_logic = self._time_series_logic()
                ts_already_running = ts_logic.module_state() == 'locked' if ts_logic else False

                if ts_already_running:
                    self._ts_we_started = False
                else:
                    self._ts_we_started = True
                    if ts_logic is not None:
                        ts_logic.start_reading()

                self._connect_time_series_signals()
                self._current_scan_folder = None

                # Start lock status polling
                poll_interval_ms = int(self._lock_status_poll_interval * 1000)
                self._lock_status_timer.start(poll_interval_ms)
            elif mode == ScanMode.POSITION_ONLY:
                # Initialize with empty channels - only track positions
                self._scan_data.initialize_data_arrays(channel_names=[])
                self._current_scan_folder = None
            elif mode == ScanMode.KDC_HW_SYNC:
                # Hardware-marker streaming: one channel (demod or ftw_corr).
                channel_names = [self._hw_sync_channel_name()]
                self._scan_data.initialize_data_arrays(channel_names)
                # Configure the slow-axis trigger and start the FPGA marker stream.
                if not self._hw_sync_scan_setup():
                    self.log.error("KDC_HW_SYNC stream setup failed. Aborting scan.")
                    self._scan_state = ScanState.IDLE
                    self.sigScanStateChanged.emit(self._scan_state)
                    return
                self._current_scan_folder = None
            elif mode == ScanMode.KDC_HW_SYNC_MULTIRES:
                # Multi-resonance mapped streaming: 2N channels (res{k}_err/res{k}_corr).
                from .hw_sync_scan import multires_channel_names
                hw = self._get_tracking_hw()
                nslots = int(getattr(hw, 'nslots', 2)) if hw is not None else 2
                channel_names = multires_channel_names(nslots)
                self._scan_data.initialize_data_arrays(channel_names)
                # Add x/y markers to the running stream + slow-axis trigger.
                if not self._hw_sync_scan_setup():
                    self.log.error("KDC_HW_SYNC_MULTIRES setup failed. Aborting scan. "
                                   "Configure multi-resonance tracking and click Start "
                                   "Stream in its GUI (Start Tracking is optional).")
                    self._scan_state = ScanState.IDLE
                    self.sigScanStateChanged.emit(self._scan_state)
                    return
                self._current_scan_folder = None
            else:
                self._scan_data.initialize_data_arrays()
                # Create scan folder now for STEP_ODMR mode so fit plots can be saved during scan
                if self._save_odmr_fit_plots:
                    timestamp = datetime.datetime.now()
                    timestamp_str = timestamp.strftime('%Y%m%d-%H%M-%S')
                    nametag = f'motor_scan_{mode.name}'
                    scan_folder_name = f'{timestamp_str}_{nametag}'
                    self._current_scan_folder = os.path.join(self.module_default_data_dir, scan_folder_name)
                    os.makedirs(self._current_scan_folder, exist_ok=True)
                else:
                    self._current_scan_folder = None
            
            # Set state to RUNNING now that initialization is complete
            self._stop_requested = False
            self._scan_start_time = time.time()
            self._scan_state = ScanState.RUNNING
            
            self.module_state.lock()
            self.sigScanStateChanged.emit(self._scan_state)
            
            self.log.info(f"Starting {mode.name} scan on axes {axes} "
                         f"with pattern {pattern.name}, resolution {scan_resolution}")
            
            # Start the scan loop
            # Use continuous line mode for CONTINUOUS_* modes if enabled
            if self._should_use_continuous_line_mode():
                self.log.info("Using continuous line scanning mode")
                self._start_continuous_line_scan(line_index=0)
            else:
                # Use point-by-point mode (original behavior)
                self._sigNextPoint.emit()
            
        except Exception as e:
            self.log.error(f"Failed to start scan: {e}", exc_info=True)
            self._scan_state = ScanState.IDLE
            self.sigScanStateChanged.emit(self._scan_state)
    
    @QtCore.Slot()
    def stop_scan(self):
        """Stop the current scan."""
        with self._thread_lock:
            if not self.is_scanning:
                return
            
            self._stop_requested = True
            self._scan_state = ScanState.STOPPING
            self.sigScanStateChanged.emit(self._scan_state)
            self.log.info("Scan stop requested.")
    
    @QtCore.Slot()
    def pause_scan(self):
        """Pause the current scan."""
        with self._thread_lock:
            if self._scan_state != ScanState.RUNNING:
                return
            
            self._scan_state = ScanState.PAUSED
            self.sigScanStateChanged.emit(self._scan_state)
            self.log.info("Scan paused.")
    
    @QtCore.Slot()
    def resume_scan(self):
        """Resume a paused scan."""
        with self._thread_lock:
            if self._scan_state != ScanState.PAUSED:
                return

            mode = ScanMode(self._active_scan_mode) if isinstance(self._active_scan_mode, int) else self._active_scan_mode

            # For CONTINUOUS_FREQ_TRACK mode, validate lock and update zero-crossing
            if mode == ScanMode.CONTINUOUS_FREQ_TRACK:
                tracking_logic = self._tracking_logic()
                if tracking_logic is None:
                    self.log.error("Tracking logic not available. Cannot resume.")
                    return

                # Validate lock is re-enabled
                if not tracking_logic.tracking_active:
                    self.log.error("Lock not enabled. Re-enable tracking in ODMR Tracking GUI before resuming.")
                    return

                # Get current zero-crossing (may have changed after re-locking)
                new_zero_crossing = tracking_logic.resonance_frequencies[0]
                if new_zero_crossing is None:
                    self.log.error("No zero-crossing frequency. Perform ODMR fit before resuming.")
                    return

                # Check if zero-crossing changed and record in history
                if new_zero_crossing != self._tracking_zero_crossing:
                    current_point = self._scan_data.current_point_index if self._scan_data else 0
                    old_zc = self._tracking_zero_crossing
                    self.log.info(f"Zero-crossing updated: {old_zc / 1e9:.6f} GHz → "
                                  f"{new_zero_crossing / 1e9:.6f} GHz")
                    self._tracking_zero_crossing_history.append((current_point, new_zero_crossing))
                    self._tracking_zero_crossing = new_zero_crossing

                # Reset lock loss flag and restart lock status polling
                self._paused_due_to_lock_loss = False
                poll_interval_ms = int(self._lock_status_poll_interval * 1000)
                self._lock_status_timer.start(poll_interval_ms)

            self._scan_state = ScanState.RUNNING
            self.sigScanStateChanged.emit(self._scan_state)
            self.log.info("Scan resumed.")
            
            # Use appropriate resume method based on mode
            if self._should_use_continuous_line_mode():
                # Try to resume continuous line if paused mid-line
                if not self._resume_continuous_line():
                    # Not paused mid-line, start next line
                    next_line = self._current_line_index + 1
                    if next_line < self._scan_data.get_num_lines():
                        self._start_continuous_line_scan(next_line)
                    else:
                        self._finalize_scan(completed=True)
            else:
                # Point-by-point mode
                self._sigNextPoint.emit()
    
    @QtCore.Slot()
    def _process_next_point(self):
        """Process the next point in the scan (non-blocking, called via signal queue)."""
        with self._thread_lock:
            # Check if we should stop
            if self._stop_requested or self._scan_state == ScanState.STOPPING:
                # Stop the motor when aborting scan
                motor = self._motor_hardware()
                if motor is not None:
                    motor.abort()
                self._finalize_scan(completed=False)
                return

            # Check if paused
            if self._scan_state == ScanState.PAUSED:
                return  # Will resume via resume_scan()

            # Check if scan is complete
            if self._scan_data.current_point_index >= self._scan_data.total_points:
                self._finalize_scan(completed=True)
                return

            # Get current point
            self._current_point_idx = self._scan_data.current_point_index
            target_pos = self._scan_data.target_positions[self._current_point_idx]

            # Build position dict
            self._current_pos_dict = {axis: pos for axis, pos in
                                      zip(self._scan_data.scan_axes, target_pos)}

            # Start motor movement (non-blocking)
            motor = self._motor_hardware()
            result = motor.move_abs(self._current_pos_dict)
            if result != 0:
                self.log.error(
                    f"Failed to issue move for scan point {self._current_point_idx}: "
                    f"{self._current_pos_dict}"
                )
                self._finalize_scan(completed=False)
                return

            # Set state and start polling timer
            self._waiting_for_motor = True
            self._point_move_start_time = time.time()
            self._point_move_retries = 0
            # Give KDC101 controllers time to process the USB command before
            # trusting is_moving(); otherwise the first poll can see a false idle.
            self._motor_poll_timer.start(200)

    @QtCore.Slot()
    def _on_motor_poll_timeout(self):
        """Called periodically to check if motor has finished moving."""
        with self._thread_lock:
            if self._stop_requested or self._scan_state == ScanState.STOPPING:
                self._waiting_for_motor = False
                # Stop the motor when aborting scan to prevent issues with subsequent homing
                motor = self._motor_hardware()
                if motor is not None:
                    motor.abort()
                self._finalize_scan(completed=False)
                return

            motor = self._motor_hardware()

            # Check if motor is still moving
            is_moving = False
            if hasattr(motor, 'is_moving'):
                is_moving = motor.is_moving()
            else:
                status = motor.get_status()
                is_moving = any(v != 0 for v in status.values())

            if is_moving:
                self._motor_poll_timer.start(50)
                return

            # Verify position is within tolerance of target
            if hasattr(motor, 'get_pos'):
                actual_pos = motor.get_pos()

                position_ok = all(
                    abs(actual_pos.get(axis, 0) - target) <= self._POINT_POSITION_TOLERANCE
                    for axis, target in self._current_pos_dict.items()
                )

                if not position_ok:
                    elapsed = time.time() - self._point_move_start_time

                    if elapsed > self._POINT_MOVE_TIMEOUT:
                        for axis, target in self._current_pos_dict.items():
                            error = abs(actual_pos.get(axis, 0) - target)
                            if error > self._POINT_POSITION_TOLERANCE:
                                self.log.error(
                                    f"Scan point {self._current_point_idx} failed to reach "
                                    f"{axis} target after {elapsed:.1f}s: "
                                    f"target={target*1000:.3f}mm, "
                                    f"actual={actual_pos.get(axis, 0)*1000:.3f}mm, "
                                    f"error={error*1e6:.0f}um"
                                )
                        self._waiting_for_motor = False
                        motor.abort()
                        self._finalize_scan(completed=False)
                        return

                    if not is_moving and self._point_move_retries < self._POINT_MOVE_MAX_RETRIES:
                        self._point_move_retries += 1
                        self.log.warning(
                            f"Scan point {self._current_point_idx}: motor idle before "
                            f"target was reached. Re-issuing move "
                            f"({self._point_move_retries}/{self._POINT_MOVE_MAX_RETRIES})."
                        )
                        motor.move_abs(self._current_pos_dict)

                    self._motor_poll_timer.start(500 if not is_moving else 100)
                    return

                # Log position at measurement point
                point_idx = self._scan_data.current_point_index
                total_points = len(self._scan_data.target_positions)
                pos_str = ", ".join(f"{a}={actual_pos.get(a, 0)*1000:.2f}" for a in self._current_pos_dict)
                self.log.info(f"Point {point_idx+1}/{total_points} at ({pos_str})mm")
            
            self._waiting_for_motor = False

            if self._scan_data.scan_mode == ScanMode.STEP_ODMR:
                self._start_odmr_scan_async()
            else:
                self._collect_continuous_data_and_advance()

    def _advance_to_next_point(self, result: Dict):
        """Advance to the next scan point after completing current one."""
        with self._thread_lock:
            # Store actual position in the scan data array
            actual_pos = result.get('actual_position', {})
            if actual_pos and self._scan_data.actual_positions is not None:
                point_idx = self._current_point_idx
                for i, axis in enumerate(self._scan_data.scan_axes):
                    if axis in actual_pos:
                        self._scan_data.actual_positions[point_idx, i] = actual_pos[axis]
            
            # Update progress
            self._scan_data.current_point_index += 1

            # Emit signals
            self.sigScanPointCompleted.emit(self._current_point_idx, result)
            self.sigScanDataUpdated.emit()

            # Schedule next point (via signal for non-blocking)
            self._sigNextPoint.emit()

    def _position_to_grid_index(self, position: Dict[str, float]) -> Optional[Tuple[int, ...]]:
        """
        Convert a motor position to grid indices.
        
        Args:
            position: Current motor position dict.
            
        Returns:
            Tuple of grid indices or None if outside scan range.
        """
        if self._scan_data is None:
            return None
        
        indices = []
        for i, (axis, (start, stop), resolution) in enumerate(
            zip(self._scan_data.scan_axes, self._scan_data.scan_range, 
                self._scan_data.scan_resolution)
        ):
            pos = position.get(axis, 0)
            
            # Check if within range
            if pos < min(start, stop) or pos > max(start, stop):
                return None
            
            # Calculate index (linear interpolation)
            if resolution > 1:
                idx = int((pos - start) / (stop - start) * (resolution - 1) + 0.5)
                idx = max(0, min(resolution - 1, idx))
            else:
                idx = 0
            
            indices.append(idx)
        
        return tuple(indices) if len(indices) == 2 else (indices[0],)
    
    def _finalize_scan(self, completed: bool):
        """
        Finalize the scan.

        Args:
            completed: True if scan finished normally, False if stopped.
        """
        # Clean up time series connection (CONTINUOUS_STREAM and CONTINUOUS_FREQ_TRACK modes)
        if self._scan_data.scan_mode in (ScanMode.CONTINUOUS_STREAM, ScanMode.CONTINUOUS_FREQ_TRACK):
            self._disconnect_time_series_signals()

            # Only stop time series reader if WE started it
            if self._ts_we_started:
                ts_logic = self._time_series_logic()
                if ts_logic is not None and hasattr(ts_logic, 'stop_reading'):
                    try:
                        ts_logic.stop_reading()
                    except Exception:
                        pass

            self._ts_raw_data_buffer = {}
            self._ts_we_started = False

        # Additional cleanup for CONTINUOUS_FREQ_TRACK mode
        if self._scan_data.scan_mode == ScanMode.CONTINUOUS_FREQ_TRACK:
            # Stop lock status polling
            if self._lock_status_timer is not None:
                self._lock_status_timer.stop()

            # Back to streaming the demodulated error
            self._set_monitor_stream_input('demod')

            # Reset tracking state
            self._paused_due_to_lock_loss = False

        # KDC_HW_SYNC: stop the FPGA marker stream and disable the KDC triggers,
        # then re-bin the whole demod trace from the hardware markers (both axes
        # hardware-anchored) and stash the faithful raw trace+markers for saving.
        if self._scan_data.scan_mode in (ScanMode.KDC_HW_SYNC,
                                         ScanMode.KDC_HW_SYNC_MULTIRES):
            self._hw_sync_scan_teardown()
            try:
                self._hw_sync_finalize()
            except Exception as e:
                self.log.error("%s hardware finalize/re-bin failed: %s",
                               self._scan_data.scan_mode.name, e)

        # Update scan data
        self._scan_data.completed = completed
        self._scan_data.timestamp_end = datetime.datetime.now()
        self._scan_data.scan_duration = time.time() - self._scan_start_time

        # Reset state
        self._scan_state = ScanState.IDLE
        self._stop_requested = False
        
        if self.module_state() == 'locked':
            self.module_state.unlock()
        
        # Emit signals
        self.sigScanStateChanged.emit(self._scan_state)
        self.sigScanCompleted.emit(self._scan_data)

        status = "completed" if completed else "stopped"
        self.log.info(f"Scan {status}. Duration: {self._scan_data.scan_duration:.1f}s, "
                     f"Points: {self._scan_data.current_point_index}/{self._scan_data.total_points}")

        # Auto-save scan data if enabled
        if self._auto_save_on_completion and completed:
            self.log.info("Auto-saving scan data...")
            try:
                self.save_scan_data()
            except Exception as e:
                self.log.error(f"Auto-save failed: {e}")
