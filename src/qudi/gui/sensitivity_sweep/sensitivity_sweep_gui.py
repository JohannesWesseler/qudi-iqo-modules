# -*- coding: utf-8 -*-
"""
Sensitivity Measurement Parameter Sweep GUI Module

This GUI module provides an interface for configuring and monitoring automated
ODMR-based magnetic field sensitivity measurements with parameter sweeps.

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

__all__ = ['SensitivitySweepGui']

import numpy as np
import pyqtgraph as pg
from PySide2 import QtCore, QtWidgets, QtGui

from qudi.core.connector import Connector
from qudi.core.statusvariable import StatusVar
from qudi.core.module import GuiBase
from qudi.util.colordefs import QudiPalettePale as palette
from qudi.util.widgets.scientific_spinbox import ScienDSpinBox


class SensitivitySweepGui(GuiBase):
    """
    GUI for ODMR-based magnetic field sensitivity parameter sweeps.

    Provides controls for:
    - Configuring parameter sweep ranges (power, FM freq, FM deviation)
    - Starting/pausing/resuming/cancelling sweeps
    - Real-time monitoring of sweep progress
    - Live plotting of ODMR scans and ASD data
    - Results table with all completed measurements

    Example config:

        sensitivity_sweep_gui:
            module.Class: 'sensitivity_sweep.sensitivity_sweep_gui.SensitivitySweepGui'
            connectors:
                sensitivity_logic: 'sensitivity_sweep_logic'
    """

    # =========================================================================
    # Connectors
    # =========================================================================

    _sensitivity_logic = Connector(
        name='sensitivity_logic',
        interface='SensitivitySweepLogic'
    )

    # =========================================================================
    # Status Variables (Persistent GUI Settings)
    # =========================================================================

    # Power sweep parameters
    _power_min = StatusVar('power_min', default=-25.0)
    _power_max = StatusVar('power_max', default=-18.0)
    _power_points = StatusVar('power_points', default=15)

    # FM frequency sweep parameters
    _f_mod_min = StatusVar('f_mod_min', default=16e3)
    _f_mod_max = StatusVar('f_mod_max', default=25e3)
    _f_mod_points = StatusVar('f_mod_points', default=5)
    _f_mod_values_khz = StatusVar('f_mod_values_khz', default='')

    # FM deviation sweep parameters
    _f_dev_min = StatusVar('f_dev_min', default=500)
    _f_dev_max = StatusVar('f_dev_max', default=600)
    _f_dev_points = StatusVar('f_dev_points', default=3)

    # ODMR scan parameters
    _odmr_freq_start = StatusVar('odmr_freq_start', default=2.86e9)
    _odmr_freq_stop = StatusVar('odmr_freq_stop', default=2.88e9)
    _odmr_freq_points = StatusVar('odmr_freq_points', default=1001)
    _odmr_run_time = StatusVar('odmr_run_time', default=60)
    _odmr_data_rate = StatusVar('odmr_data_rate', default=1000)

    # Streaming parameters
    _stream_n_traces = StatusVar('stream_n_traces', default=32)
    _stream_trace_duration = StatusVar('stream_trace_duration', default=1.0)
    _stream_data_rate = StatusVar('stream_data_rate', default=1000)
    _stream_f_enbw = StatusVar('stream_f_enbw', default=500.0)  # Lock-in filter ENBW in Hz

    # Lock-in filter parameters
    _fir_bypass = StatusVar('fir_bypass', default=False)
    _fir_filter_bandwidth = StatusVar('fir_filter_bandwidth', default='2kHz_minphase')
    _compare_wide_filters = StatusVar('compare_wide_filters', default=False)
    _bandwidths_6k = StatusVar('bandwidths_6k', default='100')
    _bandwidths_10k = StatusVar('bandwidths_10k', default='100')

    # Open/closed-loop tracking comparison matrix
    _measure_open_loop = StatusVar('measure_open_loop', default=True)
    _measure_conventional = StatusVar('measure_conventional', default=True)
    _measure_smith = StatusVar('measure_smith', default=True)
    _controller_bandwidths = StatusVar('controller_bandwidths', default='300, 600, 1000')
    _smith_gain_multipliers = StatusVar('smith_gain_multipliers', default='4')
    _smith_delay_samples = StatusVar('smith_delay_samples', default='auto')
    _tracking_settling_time = StatusVar('tracking_settling_time', default=1.0)
    _tracking_max_correction_mhz = StatusVar('tracking_max_correction_mhz', default=50.0)
    _save_raw_stream_events = StatusVar('save_raw_stream_events', default=True)

    # Off-resonant measurement parameters
    _include_off_resonant = StatusVar('include_off_resonant', default=False)
    _off_resonant_offset_mhz = StatusVar('off_resonant_offset_mhz', default=30.0)  # MHz

    # Sensitivity calculation bandwidth parameters
    _sensitivity_f_min = StatusVar('sensitivity_f_min', default=200.0)  # Hz
    _sensitivity_f_max = StatusVar('sensitivity_f_max', default=1400.0)  # Hz
    _exclude_50hz_harmonics = StatusVar('exclude_50hz_harmonics', default=True)

    # =========================================================================
    # Signals
    # =========================================================================

    sigStartSweep = QtCore.Signal(dict, dict, dict)  # sweep_params, odmr_params, stream_params
    sigPauseSweep = QtCore.Signal()
    sigResumeSweep = QtCore.Signal()
    sigCancelSweep = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # GUI widgets
        self._mw = None  # Main window
        self._config_dock = None
        self._odmr_dock = None
        self._asd_dock = None
        self._results_dock = None

        # Plot widgets
        self._odmr_plot = None
        self._odmr_curve = None
        self._fit_curve = None
        self._asd_plot = None
        self._asd_curve = None
        self._sensitivity_line = None

        # Control widgets
        self._start_button = None
        self._pause_button = None
        self._resume_button = None
        self._cancel_button = None
        self._progress_bar = None
        self._status_label = None

        # Results table
        self._results_table = None

    def on_activate(self):
        """Initialize GUI on activation."""
        try:
            # Create main window
            self._create_main_window()

            # Create dock widgets
            self._create_config_dock()
            self._create_odmr_dock()
            self._create_asd_dock()
            self._create_results_dock()

            # Connect signals
            self._connect_signals()

            # Restore window geometry
            self._restore_window_geometry(self._mw)

            # Show window
            self._mw.show()

            self.log.info('Sensitivity Sweep GUI activated')

        except Exception as e:
            self.log.error(f'Failed to activate GUI: {e}', exc_info=True)
            raise

    def on_deactivate(self):
        """Clean up GUI on deactivation."""
        try:
            # Disconnect signals
            self._disconnect_signals()

            # Save window geometry
            self._save_window_geometry(self._mw)

            # Close window
            self._mw.close()

            self.log.info('Sensitivity Sweep GUI deactivated')

        except Exception as e:
            self.log.error(f'Error during deactivation: {e}')

    def show(self):
        """Show the GUI window."""
        if self._mw is not None:
            self._mw.show()
            self._mw.raise_()
            self._mw.activateWindow()

    # =========================================================================
    # GUI Construction
    # =========================================================================

    def _create_main_window(self):
        """Create main window."""
        self._mw = QtWidgets.QMainWindow()
        self._mw.setWindowTitle('ODMR Sensitivity Parameter Sweep')
        self._mw.setDockNestingEnabled(True)

        # Create menu bar
        menubar = self._mw.menuBar()

        # File menu
        file_menu = menubar.addMenu('&File')

        export_action = QtWidgets.QAction('Export Results...', self._mw)
        export_action.triggered.connect(self._export_results)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QtWidgets.QAction('&Quit', self._mw)
        quit_action.triggered.connect(self._mw.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu('&View')

        restore_view_action = QtWidgets.QAction('Restore Default View', self._mw)
        restore_view_action.triggered.connect(self._restore_default_view)
        view_menu.addAction(restore_view_action)

        # Help menu
        help_menu = menubar.addMenu('&Help')

        about_action = QtWidgets.QAction('About', self._mw)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        # Create status bar
        self._status_label = QtWidgets.QLabel('Ready')
        self._mw.statusBar().addWidget(self._status_label, 1)

    def _create_config_dock(self):
        """Create sweep configuration dock widget."""
        from qudi.util.widgets.advanced_dockwidget import AdvancedDockWidget

        self._config_dock = AdvancedDockWidget('Sweep Configuration', parent=self._mw)
        self._config_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable
        )

        # Create scrollable widget for config
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 6, 0)  # Add small right margin for scroll bar
        widget.setLayout(layout)

        # Parameter configuration group
        param_group = QtWidgets.QGroupBox('Sweep Parameters')
        param_layout = QtWidgets.QFormLayout()

        # Power parameters
        power_label = QtWidgets.QLabel('<b>Microwave Power [dBm]</b>')
        param_layout.addRow(power_label)

        self._power_min_spinbox = QtWidgets.QDoubleSpinBox()
        self._power_min_spinbox.setRange(-50, 20)
        self._power_min_spinbox.setValue(self._power_min)
        self._power_min_spinbox.setSingleStep(0.5)
        self._power_min_spinbox.setDecimals(2)
        param_layout.addRow('  Min:', self._power_min_spinbox)

        self._power_max_spinbox = QtWidgets.QDoubleSpinBox()
        self._power_max_spinbox.setRange(-50, 20)
        self._power_max_spinbox.setValue(self._power_max)
        self._power_max_spinbox.setSingleStep(0.5)
        self._power_max_spinbox.setDecimals(2)
        param_layout.addRow('  Max:', self._power_max_spinbox)

        self._power_points_spinbox = QtWidgets.QSpinBox()
        self._power_points_spinbox.setRange(1, 100)
        self._power_points_spinbox.setValue(self._power_points)
        param_layout.addRow('  Points:', self._power_points_spinbox)

        # FM frequency parameters
        f_mod_label = QtWidgets.QLabel('<b>FM Frequency [kHz]</b>')
        param_layout.addRow(f_mod_label)

        self._f_mod_min_spinbox = QtWidgets.QDoubleSpinBox()
        self._f_mod_min_spinbox.setRange(0.1, 100)
        self._f_mod_min_spinbox.setValue(self._f_mod_min / 1e3)
        self._f_mod_min_spinbox.setSingleStep(1.0)
        self._f_mod_min_spinbox.setDecimals(5)
        param_layout.addRow('  Min:', self._f_mod_min_spinbox)

        self._f_mod_max_spinbox = QtWidgets.QDoubleSpinBox()
        self._f_mod_max_spinbox.setRange(0.1, 100)
        self._f_mod_max_spinbox.setValue(self._f_mod_max / 1e3)
        self._f_mod_max_spinbox.setSingleStep(1.0)
        self._f_mod_max_spinbox.setDecimals(5)
        param_layout.addRow('  Max:', self._f_mod_max_spinbox)

        self._f_mod_points_spinbox = QtWidgets.QSpinBox()
        self._f_mod_points_spinbox.setRange(1, 100)
        self._f_mod_points_spinbox.setValue(self._f_mod_points)
        param_layout.addRow('  Points:', self._f_mod_points_spinbox)
        self._f_mod_values_edit = QtWidgets.QLineEdit(str(self._f_mod_values_khz))
        self._f_mod_values_edit.setToolTip(
            'Optional exact frequencies in kHz, e.g. 15.26, 20, 30. '
            'Overrides Min/Max/Points; leave empty for a linear sweep.')
        param_layout.addRow('  Exact values [kHz]:', self._f_mod_values_edit)

        # FM deviation parameters
        f_dev_label = QtWidgets.QLabel('<b>FM Deviation [kHz]</b>')
        param_layout.addRow(f_dev_label)

        self._f_dev_min_spinbox = QtWidgets.QDoubleSpinBox()
        self._f_dev_min_spinbox.setRange(1, 1000)
        self._f_dev_min_spinbox.setValue(self._f_dev_min)
        self._f_dev_min_spinbox.setSingleStep(10.0)
        self._f_dev_min_spinbox.setDecimals(1)
        param_layout.addRow('  Min:', self._f_dev_min_spinbox)

        self._f_dev_max_spinbox = QtWidgets.QDoubleSpinBox()
        self._f_dev_max_spinbox.setRange(1, 1000)
        self._f_dev_max_spinbox.setValue(self._f_dev_max)
        self._f_dev_max_spinbox.setSingleStep(10.0)
        self._f_dev_max_spinbox.setDecimals(1)
        param_layout.addRow('  Max:', self._f_dev_max_spinbox)

        self._f_dev_points_spinbox = QtWidgets.QSpinBox()
        self._f_dev_points_spinbox.setRange(1, 100)
        self._f_dev_points_spinbox.setValue(self._f_dev_points)
        param_layout.addRow('  Points:', self._f_dev_points_spinbox)

        # Total points display
        param_layout.addRow(QtWidgets.QLabel(''))  # Spacer
        self._total_points_label = QtWidgets.QLabel('Total: 0 measurements')
        self._total_points_label.setStyleSheet('QLabel { font-weight: bold; color: blue; }')
        param_layout.addRow(self._total_points_label)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # Update total points when any parameter changes
        for spinbox in [self._power_points_spinbox, self._f_mod_points_spinbox, self._f_dev_points_spinbox]:
            spinbox.valueChanged.connect(self._update_total_points)
        self._f_mod_values_edit.textChanged.connect(self._update_total_points)

        # ODMR scan settings group
        odmr_group = QtWidgets.QGroupBox('ODMR Scan Settings')
        odmr_layout = QtWidgets.QFormLayout()

        self._odmr_freq_start_spinbox = ScienDSpinBox()
        self._odmr_freq_start_spinbox.setRange(0, 10e9)
        self._odmr_freq_start_spinbox.setSuffix('Hz')
        self._odmr_freq_start_spinbox.setValue(self._odmr_freq_start)
        self._odmr_freq_start_spinbox.setDecimals(3)
        odmr_layout.addRow('Start Freq:', self._odmr_freq_start_spinbox)

        self._odmr_freq_stop_spinbox = ScienDSpinBox()
        self._odmr_freq_stop_spinbox.setRange(0, 10e9)
        self._odmr_freq_stop_spinbox.setSuffix('Hz')
        self._odmr_freq_stop_spinbox.setValue(self._odmr_freq_stop)
        self._odmr_freq_stop_spinbox.setDecimals(3)
        odmr_layout.addRow('Stop Freq:', self._odmr_freq_stop_spinbox)

        self._odmr_freq_points_spinbox = QtWidgets.QSpinBox()
        self._odmr_freq_points_spinbox.setRange(10, 10000)
        self._odmr_freq_points_spinbox.setValue(self._odmr_freq_points)
        odmr_layout.addRow('Points:', self._odmr_freq_points_spinbox)

        self._odmr_run_time_spinbox = QtWidgets.QSpinBox()
        self._odmr_run_time_spinbox.setRange(1, 3600)
        self._odmr_run_time_spinbox.setSuffix(' s')
        self._odmr_run_time_spinbox.setValue(self._odmr_run_time)
        odmr_layout.addRow('Run Time:', self._odmr_run_time_spinbox)

        self._odmr_data_rate_spinbox = ScienDSpinBox()
        self._odmr_data_rate_spinbox.setRange(1, 1e6)
        self._odmr_data_rate_spinbox.setSuffix('Hz')
        self._odmr_data_rate_spinbox.setValue(self._odmr_data_rate)
        self._odmr_data_rate_spinbox.setToolTip(
            'Data rate for ODMR scan acquisition.\n'
            'This sets the rate at which data points are acquired from the hardware.'
        )
        odmr_layout.addRow('Data Rate:', self._odmr_data_rate_spinbox)

        odmr_group.setLayout(odmr_layout)
        layout.addWidget(odmr_group)

        # Streaming settings group
        stream_group = QtWidgets.QGroupBox('Time Series Settings')
        stream_layout = QtWidgets.QFormLayout()

        self._stream_n_traces_spinbox = QtWidgets.QSpinBox()
        self._stream_n_traces_spinbox.setRange(1, 1000)
        self._stream_n_traces_spinbox.setValue(self._stream_n_traces)
        stream_layout.addRow('Num Traces:', self._stream_n_traces_spinbox)

        self._stream_trace_duration_spinbox = QtWidgets.QDoubleSpinBox()
        self._stream_trace_duration_spinbox.setRange(0.1, 100)
        self._stream_trace_duration_spinbox.setSuffix(' s')
        self._stream_trace_duration_spinbox.setValue(self._stream_trace_duration)
        self._stream_trace_duration_spinbox.setDecimals(1)
        stream_layout.addRow('Trace Duration:', self._stream_trace_duration_spinbox)

        self._stream_f_enbw_spinbox = QtWidgets.QDoubleSpinBox()
        self._stream_f_enbw_spinbox.setRange(1, 100000)
        self._stream_f_enbw_spinbox.setSuffix(' Hz')
        self._stream_f_enbw_spinbox.setValue(self._stream_f_enbw)
        self._stream_f_enbw_spinbox.setDecimals(1)
        self._stream_f_enbw_spinbox.setToolTip(
            'Equivalent Noise Bandwidth of the lock-in filter.\n'
            'CIC + FIR design estimates: 6 kHz minimum phase ≈ 6392.1 Hz;\n'
            '10 kHz minimum phase ≈ 10418.6 Hz.\n'
            'Used by the time-domain standard-deviation estimate;\n'
            'the Welch ASD uses its own spectral-density normalization.'
        )
        stream_layout.addRow('Filter ENBW:', self._stream_f_enbw_spinbox)

        # Lock-in filter settings
        stream_layout.addRow(QtWidgets.QLabel(''))  # Spacer
        fir_label = QtWidgets.QLabel('<b>Lock-in Filter Settings</b>')
        stream_layout.addRow(fir_label)

        self._fir_bypass_checkbox = QtWidgets.QCheckBox('Bypass filter (CIC only)')
        self._fir_bypass_checkbox.setChecked(self._fir_bypass)
        self._fir_bypass_checkbox.setToolTip(
            'Bypass the lowpass filter and use only CIC decimation.\n'
            'When bypassed: ~15 kHz bandwidth, ~160 µs latency.\n'
            'When not bypassed: bandwidth set by filter selection below.'
        )
        stream_layout.addRow(self._fir_bypass_checkbox)

        self._fir_filter_combobox = QtWidgets.QComboBox()
        self._fir_filter_combobox.addItems([
            '2 kHz minimum phase',
            '2 kHz linear phase',
            '20 kHz linear phase (fast /1024 image)',
            '6 kHz minimum phase',
            '10 kHz minimum phase',
            '10 kHz minimum phase (CIC /2048, N=3, M=1)',
        ])
        # Map stored value to combobox index
        _filter_index_map = {
            '2kHz_minphase': 0,
            '2kHz_linear': 1,
            '20kHz_linear': 2,
            '6kHz_minphase': 3,
            '10kHz_minphase': 4,
            '10kHz_minphase_2048': 5,
            '2kHz': 0,  # migrate the legacy minimum-phase spelling
        }
        filter_index = _filter_index_map.get(self._fir_filter_bandwidth, 0)
        self._fir_filter_combobox.setCurrentIndex(filter_index)
        self._fir_filter_combobox.setToolTip(
            'Select lock-in lowpass filter.\n'
            'Only filters compiled into the loaded image can be selected.\n'
            'Minimum phase: established low-latency response.\n'
            'Linear phase: constant group delay, required by the current Smith model.'
        )
        stream_layout.addRow('  Filter:', self._fir_filter_combobox)

        # Enable/disable filter selection based on bypass checkbox
        self._fir_filter_combobox.setEnabled(not self._fir_bypass)
        self._fir_bypass_checkbox.toggled.connect(
            lambda checked: self._fir_filter_combobox.setEnabled(not checked)
        )

        # Tracking comparison settings
        stream_layout.addRow(QtWidgets.QLabel(''))
        comparison_label = QtWidgets.QLabel('<b>Open/Closed-Loop Comparison</b>')
        stream_layout.addRow(comparison_label)

        self._measure_open_loop_checkbox = QtWidgets.QCheckBox('Open loop')
        self._measure_open_loop_checkbox.setChecked(self._measure_open_loop)
        stream_layout.addRow(self._measure_open_loop_checkbox)
        self._measure_conventional_checkbox = QtWidgets.QCheckBox(
            'Closed loop: conventional controller')
        self._measure_conventional_checkbox.setChecked(self._measure_conventional)
        stream_layout.addRow(self._measure_conventional_checkbox)
        self._measure_smith_checkbox = QtWidgets.QCheckBox(
            'Closed loop: Smith compensator')
        self._measure_smith_checkbox.setChecked(self._measure_smith)
        stream_layout.addRow(self._measure_smith_checkbox)

        self._controller_bandwidths_edit = QtWidgets.QLineEdit(
            str(self._controller_bandwidths))
        self._controller_bandwidths_edit.setToolTip(
            'Comma-separated base controller bandwidths in Hz. Each value is measured '
            'for conventional and selected Smith settings.')
        stream_layout.addRow('  Bandwidths [Hz]:', self._controller_bandwidths_edit)

        self._smith_gain_multipliers_edit = QtWidgets.QLineEdit(
            str(self._smith_gain_multipliers))
        self._smith_gain_multipliers_edit.setToolTip(
            'Comma-separated Smith gain multipliers, for example 2, 4, 6.')
        stream_layout.addRow('  Smith gain factors:', self._smith_gain_multipliers_edit)

        self._smith_delay_samples_edit = QtWidgets.QLineEdit(
            str(self._smith_delay_samples))
        self._smith_delay_samples_edit.setToolTip(
            'Comma-separated predictor delays in demodulated samples. Use "auto" '
            'for the FIR group delay advertised by the FPGA image.')
        stream_layout.addRow('  Smith delays:', self._smith_delay_samples_edit)

        self._compare_filters_checkbox = QtWidgets.QCheckBox(
            'Compare 6 kHz and 10 kHz minimum-phase filters')
        self._compare_filters_checkbox.setChecked(self._compare_wide_filters)
        self._compare_filters_checkbox.setToolTip(
            'At each microwave setting, acquire a separate ODMR scan and all selected '
            'conditions for each filter. Microwave power remains the outer loop. '
            'Uses the calculated ENBW for each filter; Smith compensation is disabled.')
        stream_layout.addRow(self._compare_filters_checkbox)
        self._bandwidths_6k_edit = QtWidgets.QLineEdit(str(self._bandwidths_6k))
        self._bandwidths_10k_edit = QtWidgets.QLineEdit(str(self._bandwidths_10k))
        for label, edit in (('6 kHz filter: loop BW [Hz]', self._bandwidths_6k_edit),
                            ('10 kHz filter: loop BW [Hz]', self._bandwidths_10k_edit)):
            edit.setToolTip('Comma-separated integral controller bandwidths, e.g. 100, 300.')
            stream_layout.addRow(label, edit)
            edit.textChanged.connect(self._update_total_points)
        self._compare_filters_checkbox.toggled.connect(self._update_filter_comparison)
        self._fir_filter_combobox.currentIndexChanged.connect(self._update_filter_comparison)
        self._fir_bypass_checkbox.toggled.connect(self._update_filter_comparison)
        self._update_filter_comparison()

        self._tracking_settling_spinbox = QtWidgets.QDoubleSpinBox()
        self._tracking_settling_spinbox.setRange(0.0, 60.0)
        self._tracking_settling_spinbox.setDecimals(3)
        self._tracking_settling_spinbox.setSuffix(' s')
        self._tracking_settling_spinbox.setValue(self._tracking_settling_time)
        stream_layout.addRow('  Lock settling:', self._tracking_settling_spinbox)

        self._tracking_max_correction_spinbox = QtWidgets.QDoubleSpinBox()
        self._tracking_max_correction_spinbox.setRange(0.001, 100.0)
        self._tracking_max_correction_spinbox.setDecimals(3)
        self._tracking_max_correction_spinbox.setSuffix(' MHz')
        self._tracking_max_correction_spinbox.setValue(
            self._tracking_max_correction_mhz)
        stream_layout.addRow('  Max correction:', self._tracking_max_correction_spinbox)

        self._save_raw_events_checkbox = QtWidgets.QCheckBox(
            'Save timestamped raw FPGA events')
        self._save_raw_events_checkbox.setChecked(self._save_raw_stream_events)
        self._save_raw_events_checkbox.setToolTip(
            'Recommended for paper measurements. Preserves source IDs, sequence numbers, '
            'timestamps and raw integer payloads in addition to reconstructed traces.')
        stream_layout.addRow(self._save_raw_events_checkbox)
        for checkbox in (self._measure_open_loop_checkbox,
                         self._measure_conventional_checkbox,
                         self._measure_smith_checkbox):
            checkbox.toggled.connect(self._update_total_points)
        for edit in (self._controller_bandwidths_edit,
                     self._smith_gain_multipliers_edit,
                     self._smith_delay_samples_edit):
            edit.textChanged.connect(self._update_total_points)
        self._update_total_points()

        # Off-resonant measurement option
        stream_layout.addRow(QtWidgets.QLabel(''))  # Spacer
        off_res_label = QtWidgets.QLabel('<b>Off-Resonant Reference</b>')
        stream_layout.addRow(off_res_label)

        self._off_resonant_checkbox = QtWidgets.QCheckBox('Include off-resonant measurement')
        self._off_resonant_checkbox.setChecked(self._include_off_resonant)
        self._off_resonant_checkbox.setToolTip(
            'Additionally measure sensitivity at an off-resonant frequency.\n'
            'This provides a reference for comparison with the on-resonant measurement.'
        )
        stream_layout.addRow(self._off_resonant_checkbox)

        self._off_resonant_offset_spinbox = QtWidgets.QDoubleSpinBox()
        self._off_resonant_offset_spinbox.setRange(1, 100)
        self._off_resonant_offset_spinbox.setSuffix(' MHz')
        self._off_resonant_offset_spinbox.setValue(self._off_resonant_offset_mhz)
        self._off_resonant_offset_spinbox.setDecimals(1)
        self._off_resonant_offset_spinbox.setToolTip(
            'Frequency offset from zero-crossing for off-resonant measurement.\n'
            'The microwave will be set to (zero_crossing_freq + offset) for the reference.'
        )
        stream_layout.addRow('  Offset:', self._off_resonant_offset_spinbox)

        # Enable/disable offset spinbox based on checkbox
        self._off_resonant_offset_spinbox.setEnabled(self._include_off_resonant)
        self._off_resonant_checkbox.toggled.connect(self._off_resonant_offset_spinbox.setEnabled)
        self._off_resonant_checkbox.toggled.connect(self._update_total_points)
        self._update_total_points()

        # Sensitivity calculation bandwidth settings
        stream_layout.addRow(QtWidgets.QLabel(''))  # Spacer
        sens_bw_label = QtWidgets.QLabel('<b>Sensitivity Bandwidth</b>')
        stream_layout.addRow(sens_bw_label)

        self._sensitivity_f_min_spinbox = QtWidgets.QDoubleSpinBox()
        self._sensitivity_f_min_spinbox.setRange(1, 10000)
        self._sensitivity_f_min_spinbox.setSuffix(' Hz')
        self._sensitivity_f_min_spinbox.setValue(self._sensitivity_f_min)
        self._sensitivity_f_min_spinbox.setDecimals(0)
        self._sensitivity_f_min_spinbox.setToolTip(
            'Lower frequency bound for sensitivity calculation.\n'
            'ASD values below this frequency will be excluded.'
        )
        stream_layout.addRow('  f_min:', self._sensitivity_f_min_spinbox)

        self._sensitivity_f_max_spinbox = QtWidgets.QDoubleSpinBox()
        self._sensitivity_f_max_spinbox.setRange(1, 10000)
        self._sensitivity_f_max_spinbox.setSuffix(' Hz')
        self._sensitivity_f_max_spinbox.setValue(self._sensitivity_f_max)
        self._sensitivity_f_max_spinbox.setDecimals(0)
        self._sensitivity_f_max_spinbox.setToolTip(
            'Upper frequency bound for sensitivity calculation.\n'
            'ASD values above this frequency will be excluded.'
        )
        stream_layout.addRow('  f_max:', self._sensitivity_f_max_spinbox)

        self._exclude_50hz_checkbox = QtWidgets.QCheckBox('Exclude 50 Hz harmonics')
        self._exclude_50hz_checkbox.setChecked(self._exclude_50hz_harmonics)
        self._exclude_50hz_checkbox.setToolTip(
            'Exclude 50 Hz line noise harmonics from sensitivity calculation.\n'
            'Harmonics at 50, 100, 150, 200, ... Hz will be filtered out.'
        )
        stream_layout.addRow(self._exclude_50hz_checkbox)

        stream_group.setLayout(stream_layout)
        layout.addWidget(stream_group)

        # Control buttons
        button_layout = QtWidgets.QVBoxLayout()

        self._start_button = QtWidgets.QPushButton('START SWEEP')
        self._start_button.setStyleSheet('QPushButton { font-weight: bold; background-color: #4CAF50; color: white; min-height: 40px; }')
        self._start_button.clicked.connect(self._on_start_sweep)
        button_layout.addWidget(self._start_button)

        button_row = QtWidgets.QHBoxLayout()

        self._pause_button = QtWidgets.QPushButton('PAUSE')
        self._pause_button.setEnabled(False)
        self._pause_button.clicked.connect(self._on_pause_sweep)
        button_row.addWidget(self._pause_button)

        self._resume_button = QtWidgets.QPushButton('RESUME')
        self._resume_button.setEnabled(False)
        self._resume_button.clicked.connect(self._on_resume_sweep)
        button_row.addWidget(self._resume_button)

        self._cancel_button = QtWidgets.QPushButton('CANCEL')
        self._cancel_button.setEnabled(False)
        self._cancel_button.setStyleSheet('QPushButton { background-color: #f44336; color: white; }')
        self._cancel_button.clicked.connect(self._on_cancel_sweep)
        button_row.addWidget(self._cancel_button)

        button_layout.addLayout(button_row)
        layout.addLayout(button_layout)

        # Progress section
        progress_group = QtWidgets.QGroupBox('Progress')
        progress_layout = QtWidgets.QVBoxLayout()

        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        progress_layout.addWidget(self._progress_bar)

        self._progress_label = QtWidgets.QLabel('Not started')
        progress_layout.addWidget(self._progress_label)

        self._best_result_label = QtWidgets.QLabel('Best sensitivity: --')
        self._best_result_label.setStyleSheet('QLabel { font-weight: bold; color: green; }')
        progress_layout.addWidget(self._best_result_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        layout.addStretch()

        scroll_area.setWidget(widget)
        self._config_dock.setWidget(scroll_area)
        self._mw.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._config_dock)

        # Initialize total points
        self._update_total_points()

    def _create_odmr_dock(self):
        """Create ODMR plot dock widget."""
        from qudi.util.widgets.advanced_dockwidget import AdvancedDockWidget

        self._odmr_dock = AdvancedDockWidget('ODMR Scan', parent=self._mw)
        self._odmr_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable
        )

        # Create plot widget
        self._odmr_plot = pg.PlotWidget()
        self._odmr_plot.setLabel('bottom', 'Frequency', units='Hz')
        self._odmr_plot.setLabel('left', 'Signal', units='V')
        self._odmr_plot.showGrid(x=True, y=True)
        self._odmr_plot.setMinimumHeight(150)

        # Create plot curves
        self._odmr_curve = pg.PlotDataItem(
            pen=pg.mkPen(palette.c1, width=2),
            name='ODMR Signal'
        )
        self._odmr_plot.addItem(self._odmr_curve)

        self._fit_curve = pg.PlotDataItem(
            pen=pg.mkPen(palette.c3, width=3, style=QtCore.Qt.DashLine),
            name='Fit'
        )
        self._odmr_plot.addItem(self._fit_curve)

        # Add legend
        self._odmr_plot.addLegend()

        self._odmr_dock.setWidget(self._odmr_plot)
        self._mw.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._odmr_dock)

    def _create_asd_dock(self):
        """Create ASD plot dock widget."""
        from qudi.util.widgets.advanced_dockwidget import AdvancedDockWidget

        self._asd_dock = AdvancedDockWidget('Amplitude Spectral Density', parent=self._mw)
        self._asd_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable
        )

        # Create plot widget
        self._asd_plot = pg.PlotWidget()
        self._asd_plot.setLabel('bottom', 'Frequency', units='Hz')
        self._asd_plot.setLabel('left', 'Magnetic Field ASD', units='nT/√Hz')
        self._asd_plot.setLogMode(x=True, y=True)
        self._asd_plot.showGrid(x=True, y=True)
        self._asd_plot.setMinimumHeight(150)

        # Create plot curve
        self._asd_curve = pg.PlotDataItem(
            pen=pg.mkPen(palette.c2, width=2),
            name='ASD'
        )
        self._asd_plot.addItem(self._asd_curve)

        # Create horizontal line for sensitivity value
        self._sensitivity_line = pg.InfiniteLine(
            pos=0,  # Will be set in log10 space
            angle=0,  # horizontal
            pen=pg.mkPen(color='r', width=2, style=QtCore.Qt.DashLine),
            label='Sensitivity',
            labelOpts={'position': 0.05, 'color': 'r', 'fill': (200, 200, 200, 100)}
        )
        self._sensitivity_line.setVisible(False)  # Hide until we have data
        self._asd_plot.addItem(self._sensitivity_line)

        # Create vertical lines for bandwidth bounds (in log10 space since plot is log mode)
        self._bandwidth_line_min = pg.InfiniteLine(
            pos=0,  # Will be set in log10 space
            angle=90,  # vertical
            pen=pg.mkPen(color='r', width=1, style=QtCore.Qt.DotLine),
        )
        self._bandwidth_line_min.setVisible(False)
        self._asd_plot.addItem(self._bandwidth_line_min)

        self._bandwidth_line_max = pg.InfiniteLine(
            pos=0,  # Will be set in log10 space
            angle=90,  # vertical
            pen=pg.mkPen(color='r', width=1, style=QtCore.Qt.DotLine),
        )
        self._bandwidth_line_max.setVisible(False)
        self._asd_plot.addItem(self._bandwidth_line_max)

        self._asd_dock.setWidget(self._asd_plot)
        self._mw.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._asd_dock)

    def _create_results_dock(self):
        """Create results table dock widget."""
        from qudi.util.widgets.advanced_dockwidget import AdvancedDockWidget

        self._results_dock = AdvancedDockWidget('Results', parent=self._mw)
        self._results_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable
        )

        # Create widget contents
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        widget.setLayout(layout)

        # Create table
        self._results_table = QtWidgets.QTableWidget()
        self._results_table.setColumnCount(13)
        self._results_table.setHorizontalHeaderLabels([
            'Index',
            'Mode',
            'Controller',
            'BW [Hz]',
            'Smith gain',
            'Smith delay',
            'Power [dBm]',
            'f_mod [kHz]',
            'f_dev [kHz]',
            'Linewidth [Hz]',
            'Total sensitivity [nT/√Hz]',
            'Residual sensitivity [nT/√Hz]',
            'FIR filter'
        ])
        self._results_table.setSortingEnabled(True)
        self._results_table.horizontalHeader().setStretchLastSection(True)
        self._results_table.setAlternatingRowColors(True)
        self._results_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        layout.addWidget(self._results_table)

        # Export button
        export_button = QtWidgets.QPushButton('Export to CSV')
        export_button.clicked.connect(self._export_results)
        layout.addWidget(export_button)

        self._results_dock.setWidget(widget)
        self._mw.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._results_dock)

    # =========================================================================
    # Signal Connections
    # =========================================================================

    def _connect_signals(self):
        """Connect signals between GUI and logic."""
        logic = self._sensitivity_logic()

        # Logic → GUI signals
        logic.sigSweepStarted.connect(
            self._on_sweep_started,
            QtCore.Qt.QueuedConnection
        )
        logic.sigPointStarted.connect(
            self._on_point_started,
            QtCore.Qt.QueuedConnection
        )
        logic.sigPointCompleted.connect(
            self._on_point_completed,
            QtCore.Qt.QueuedConnection
        )
        logic.sigSweepProgress.connect(
            self._on_sweep_progress,
            QtCore.Qt.QueuedConnection
        )
        logic.sigSweepPaused.connect(
            self._on_sweep_paused,
            QtCore.Qt.QueuedConnection
        )
        logic.sigSweepResumed.connect(
            self._on_sweep_resumed,
            QtCore.Qt.QueuedConnection
        )
        logic.sigSweepFinished.connect(
            self._on_sweep_finished,
            QtCore.Qt.QueuedConnection
        )
        logic.sigSweepCancelled.connect(
            self._on_sweep_cancelled,
            QtCore.Qt.QueuedConnection
        )
        logic.sigError.connect(
            self._on_error,
            QtCore.Qt.QueuedConnection
        )
        logic.sigOdmrDataReady.connect(
            self._update_odmr_plot,
            QtCore.Qt.QueuedConnection
        )
        logic.sigFitDataReady.connect(
            self._update_fit_plot,
            QtCore.Qt.QueuedConnection
        )
        logic.sigASDDataReady.connect(
            self._update_asd_plot,
            QtCore.Qt.QueuedConnection
        )

        # GUI → Logic signals
        # IMPORTANT: Connect directly to logic method to ensure sweep runs on logic thread
        # This is critical for pause/cancel to work properly
        self.sigStartSweep.connect(
            logic.configure_and_start_sweep,
            QtCore.Qt.QueuedConnection
        )
        self.sigPauseSweep.connect(
            logic.pause_sweep,
            QtCore.Qt.QueuedConnection
        )
        self.sigResumeSweep.connect(
            logic.resume_sweep,
            QtCore.Qt.QueuedConnection
        )
        self.sigCancelSweep.connect(
            logic.cancel_sweep,
            QtCore.Qt.QueuedConnection
        )

    def _disconnect_signals(self):
        """Disconnect all signals."""
        try:
            logic = self._sensitivity_logic()
            logic.sigSweepStarted.disconnect(self._on_sweep_started)
            logic.sigPointStarted.disconnect(self._on_point_started)
            logic.sigPointCompleted.disconnect(self._on_point_completed)
            logic.sigSweepProgress.disconnect(self._on_sweep_progress)
            logic.sigSweepPaused.disconnect(self._on_sweep_paused)
            logic.sigSweepResumed.disconnect(self._on_sweep_resumed)
            logic.sigSweepFinished.disconnect(self._on_sweep_finished)
            logic.sigSweepCancelled.disconnect(self._on_sweep_cancelled)
            logic.sigError.disconnect(self._on_error)
            logic.sigOdmrDataReady.disconnect(self._update_odmr_plot)
            logic.sigFitDataReady.disconnect(self._update_fit_plot)
            logic.sigASDDataReady.disconnect(self._update_asd_plot)

            self.sigStartSweep.disconnect()
            self.sigPauseSweep.disconnect()
            self.sigResumeSweep.disconnect()
            self.sigCancelSweep.disconnect()
        except (TypeError, RuntimeError):
            pass

    # =========================================================================
    # GUI Slots - User Actions
    # =========================================================================

    @staticmethod
    def _parse_positive_list(text, label):
        try:
            values = [float(item.strip()) for item in str(text).split(',') if item.strip()]
        except ValueError as e:
            raise ValueError(f'{label} must be a comma-separated number list') from e
        if not values or any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f'{label} must contain finite positive values')
        return values

    @staticmethod
    def _parse_smith_delays(text):
        values = []
        for item in (part.strip().lower() for part in str(text).split(',')):
            if not item:
                continue
            if item == 'auto':
                value = None
            else:
                try:
                    value = int(item)
                except ValueError as e:
                    raise ValueError(
                        'Smith delays must be comma-separated integers or "auto"') from e
                if not 1 <= value < 128:
                    raise ValueError('Smith delays must lie in [1, 127] samples')
            if value not in values:
                values.append(value)
        return values or [None]

    @QtCore.Slot()
    def _on_start_sweep(self):
        """Handle start button click."""
        # Store settings
        self._power_min = self._power_min_spinbox.value()
        self._power_max = self._power_max_spinbox.value()
        self._power_points = self._power_points_spinbox.value()
        self._f_mod_min = self._f_mod_min_spinbox.value() * 1e3
        self._f_mod_max = self._f_mod_max_spinbox.value() * 1e3
        self._f_mod_points = self._f_mod_points_spinbox.value()
        self._f_dev_min = self._f_dev_min_spinbox.value()
        self._f_dev_max = self._f_dev_max_spinbox.value()
        self._f_dev_points = self._f_dev_points_spinbox.value()

        self._odmr_freq_start = self._odmr_freq_start_spinbox.value()
        self._odmr_freq_stop = self._odmr_freq_stop_spinbox.value()
        self._odmr_freq_points = self._odmr_freq_points_spinbox.value()
        self._odmr_run_time = self._odmr_run_time_spinbox.value()
        self._odmr_data_rate = self._odmr_data_rate_spinbox.value()

        self._stream_n_traces = self._stream_n_traces_spinbox.value()
        self._stream_trace_duration = self._stream_trace_duration_spinbox.value()
        self._stream_f_enbw = self._stream_f_enbw_spinbox.value()

        # Lock-in filter settings
        self._fir_bypass = self._fir_bypass_checkbox.isChecked()
        # Map combobox index to PyRPL register key
        _filter_key_from_index = {
            0: '2kHz_minphase', 1: '2kHz_linear', 2: '20kHz_linear',
            3: '6kHz_minphase', 4: '10kHz_minphase', 5: '10kHz_minphase_2048'}
        self._fir_filter_bandwidth = _filter_key_from_index.get(
            self._fir_filter_combobox.currentIndex(), '2kHz_minphase'
        )
        if self._fir_filter_bandwidth == '10kHz_minphase_2048':
            self._stream_f_enbw = 10640.924904046242

        self._measure_open_loop = self._measure_open_loop_checkbox.isChecked()
        self._measure_conventional = self._measure_conventional_checkbox.isChecked()
        self._measure_smith = self._measure_smith_checkbox.isChecked()
        tracking_modes = []
        if self._measure_open_loop:
            tracking_modes.append('open_loop')
        if self._measure_conventional:
            tracking_modes.append('closed_loop_conventional')
        if self._measure_smith:
            tracking_modes.append('closed_loop_smith')
        if not tracking_modes:
            QtWidgets.QMessageBox.warning(
                self._mw, 'No measurement selected',
                'Select at least one open- or closed-loop comparison mode.')
            return
        try:
            f_mod_array = self._f_mod_array_from_widgets()
            filter_sweep = self._filter_sweep_from_widgets()
            controller_bandwidths = (
                self._parse_positive_list(
                    self._controller_bandwidths_edit.text(), 'Controller bandwidths')
                if (self._measure_conventional or self._measure_smith) and not filter_sweep
                else [300.0])
            smith_gains = (
                self._parse_positive_list(
                    self._smith_gain_multipliers_edit.text(), 'Smith gain factors')
                if self._measure_smith else [4.0])
            smith_delays = (
                self._parse_smith_delays(self._smith_delay_samples_edit.text())
                if self._measure_smith else [None])
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self._mw, 'Invalid tracking matrix', str(e))
            return
        self._controller_bandwidths = self._controller_bandwidths_edit.text()
        self._f_mod_values_khz = self._f_mod_values_edit.text()
        self._compare_wide_filters = self._compare_filters_checkbox.isChecked()
        self._bandwidths_6k = self._bandwidths_6k_edit.text()
        self._bandwidths_10k = self._bandwidths_10k_edit.text()
        self._smith_gain_multipliers = self._smith_gain_multipliers_edit.text()
        self._smith_delay_samples = self._smith_delay_samples_edit.text()
        self._tracking_settling_time = self._tracking_settling_spinbox.value()
        self._tracking_max_correction_mhz = \
            self._tracking_max_correction_spinbox.value()
        self._save_raw_stream_events = self._save_raw_events_checkbox.isChecked()

        # Off-resonant measurement settings
        self._include_off_resonant = self._off_resonant_checkbox.isChecked()
        self._off_resonant_offset_mhz = self._off_resonant_offset_spinbox.value()

        # Sensitivity bandwidth settings
        self._sensitivity_f_min = self._sensitivity_f_min_spinbox.value()
        self._sensitivity_f_max = self._sensitivity_f_max_spinbox.value()
        self._exclude_50hz_harmonics = self._exclude_50hz_checkbox.isChecked()

        # Generate parameter arrays
        if self._power_points == 1:
            power_array = np.array([self._power_min])
        else:
            # Linear spacing in power (not dBm)
            power_linear = np.linspace(
                10 ** (self._power_min / 10),
                10 ** (self._power_max / 10),
                self._power_points
            )
            power_array = 10 * np.log10(power_linear)

        f_dev_array = np.linspace(self._f_dev_min, self._f_dev_max, self._f_dev_points)

        sweep_params = {
            'power': power_array,
            'f_mod': f_mod_array,
            'f_dev': f_dev_array
        }

        odmr_params = {
            'frequency_start': self._odmr_freq_start,
            'frequency_stop': self._odmr_freq_stop,
            'frequency_points': self._odmr_freq_points,
            'run_time': self._odmr_run_time,
            'data_rate': self._odmr_data_rate,
            'multi_freq_mode': 'triple'
        }

        stream_params = {
            'n_time_traces': self._stream_n_traces,
            'trace_duration': self._stream_trace_duration,
            'data_rate': self._stream_data_rate,
            'f_enbw': self._stream_f_enbw,
            'fir_bypass': self._fir_bypass,
            'fir_filter_bandwidth': self._fir_filter_bandwidth,
            'include_off_resonant': self._include_off_resonant,
            'off_resonant_offset_hz': self._off_resonant_offset_mhz * 1e6,  # Convert MHz to Hz
            'sensitivity_f_min': self._sensitivity_f_min,
            'sensitivity_f_max': self._sensitivity_f_max,
            'exclude_50hz_harmonics': self._exclude_50hz_harmonics,
            'tracking_comparison_enabled': True,
            'tracking_modes': tracking_modes,
            'controller_bandwidths_hz': controller_bandwidths,
            'smith_gain_multipliers': smith_gains,
            'smith_delay_samples': smith_delays,
            'tracking_settling_time_s': self._tracking_settling_time,
            'tracking_max_correction_hz': self._tracking_max_correction_mhz * 1e6,
            'save_raw_stream_events': self._save_raw_stream_events,
        }

        if filter_sweep:
            stream_params['filter_sweep'] = filter_sweep

        # Clear results table
        self._results_table.setRowCount(0)

        # Emit signal to start sweep
        self.sigStartSweep.emit(sweep_params, odmr_params, stream_params)

    @QtCore.Slot()
    def _on_pause_sweep(self):
        """Handle pause button click."""
        self.sigPauseSweep.emit()

    @QtCore.Slot()
    def _on_resume_sweep(self):
        """Handle resume button click."""
        self.sigResumeSweep.emit()

    @QtCore.Slot()
    def _on_cancel_sweep(self):
        """Handle cancel button click."""
        reply = QtWidgets.QMessageBox.question(
            self._mw,
            'Confirm Cancel',
            'Are you sure you want to cancel the sweep?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            self.sigCancelSweep.emit()

    def _filter_sweep_from_widgets(self):
        if not self._compare_filters_checkbox.isChecked():
            return []
        return [
            {'fir_filter_bandwidth': name,
             'controller_bandwidths_hz': self._parse_positive_list(edit.text(), name + ' loop bandwidths'),
             'f_enbw': enbw}
            for name, edit, enbw in (
                ('6kHz_minphase', self._bandwidths_6k_edit, 6392.1428),
                ('10kHz_minphase', self._bandwidths_10k_edit, 10418.5833))]

    def _update_filter_comparison(self, *args):
        compare = self._compare_filters_checkbox.isChecked()
        new_profile = self._fir_filter_combobox.currentIndex() == 5
        if new_profile:
            self._measure_smith_checkbox.setChecked(False)
            self._stream_f_enbw_spinbox.setValue(10640.924904046242)
        if compare:
            self._fir_bypass_checkbox.setChecked(False)
            self._measure_smith_checkbox.setChecked(False)
        self._fir_bypass_checkbox.setEnabled(not compare)
        self._fir_filter_combobox.setEnabled(not compare and not self._fir_bypass_checkbox.isChecked())
        self._stream_f_enbw_spinbox.setEnabled(not compare and not new_profile)
        self._measure_smith_checkbox.setEnabled(not compare and not new_profile)
        self._controller_bandwidths_edit.setEnabled(not compare)
        self._bandwidths_6k_edit.setEnabled(compare)
        self._bandwidths_10k_edit.setEnabled(compare)
        self._update_total_points()

    def _f_mod_array_from_widgets(self):
        text = self._f_mod_values_edit.text().strip()
        if text:
            values = np.asarray(self._parse_positive_list(text, 'FM frequencies'))
            if np.any((values < 0.1) | (values > 100)):
                raise ValueError('FM frequencies must be between 0.1 and 100 kHz')
            return values * 1000.
        return np.linspace(self._f_mod_min_spinbox.value(),
                           self._f_mod_max_spinbox.value(),
                           self._f_mod_points_spinbox.value()) * 1000.

    @QtCore.Slot()
    def _update_total_points(self):
        """Update total points display."""
        try:
            f_mod_points = len(self._f_mod_array_from_widgets())
        except ValueError:
            self._total_points_label.setText('Invalid exact FM frequency list')
            return
        scan_points = (self._power_points_spinbox.value() *
                       f_mod_points *
                       self._f_dev_points_spinbox.value())
        conditions = 1
        if hasattr(self, '_bandwidths_10k_edit') and self._compare_filters_checkbox.isChecked():
            try:
                profiles = self._filter_sweep_from_widgets()
                conditions = sum(
                    int(self._measure_open_loop_checkbox.isChecked()) +
                    (len(p['controller_bandwidths_hz']) if self._measure_conventional_checkbox.isChecked() else 0) +
                    int(hasattr(self, '_off_resonant_checkbox') and self._off_resonant_checkbox.isChecked())
                    for p in profiles)
                self._total_points_label.setText(
                    f'Total: {scan_points * 2} hyperfine scans; {scan_points * conditions} streams '
                    '(separate bandwidths per filter)')
            except ValueError:
                self._total_points_label.setText('Invalid per-filter bandwidth list')
            return
        if hasattr(self, '_measure_open_loop_checkbox'):
            try:
                conventional = self._measure_conventional_checkbox.isChecked()
                smith = self._measure_smith_checkbox.isChecked()
                bandwidths = (self._parse_positive_list(
                    self._controller_bandwidths_edit.text(), 'bandwidths')
                    if conventional or smith else [])
                gains = (self._parse_positive_list(
                    self._smith_gain_multipliers_edit.text(), 'Smith gains')
                    if smith else [])
                delays = (self._parse_smith_delays(
                    self._smith_delay_samples_edit.text()) if smith else [])
                conditions = int(self._measure_open_loop_checkbox.isChecked())
                if conventional:
                    conditions += len(bandwidths)
                if smith:
                    conditions += len(bandwidths) * len(gains) * len(delays)
                if (hasattr(self, '_off_resonant_checkbox') and
                        self._off_resonant_checkbox.isChecked()):
                    conditions += 1
            except ValueError:
                self._total_points_label.setText(
                    f'Total: {scan_points} hyperfine scans × invalid comparison matrix')
                return
        total = scan_points * conditions
        self._total_points_label.setText(
            f'Total: {scan_points} hyperfine scans × {conditions} conditions = '
            f'{total} streams')

    # =========================================================================
    # Logic Slots - Updates from Logic
    # =========================================================================

    @QtCore.Slot(int)
    def _on_sweep_started(self, total_points):
        """Handle sweep started signal."""
        self._start_button.setEnabled(False)
        self._start_button.setStyleSheet('QPushButton { font-weight: bold; background-color: #9E9E9E; color: white; min-height: 40px; }')
        self._pause_button.setEnabled(True)
        self._resume_button.setEnabled(False)
        self._cancel_button.setEnabled(True)
        self._progress_bar.setMaximum(total_points)
        self._progress_bar.setValue(0)
        self._progress_label.setText(f'Running: 0 / {total_points}')
        self._status_label.setText('Sweep running...')

    @QtCore.Slot(int, dict)
    def _on_point_started(self, index, params):
        """Handle point started signal."""
        self.log.debug(f'Point {index + 1} started: {params}')
        self._status_label.setText(
            f'Measuring point {index + 1}: '
            f'P={params["power"]:.2f}dBm, '
            f'f_mod={params["f_mod"]/1e3:.1f}kHz, '
            f'f_dev={params["f_dev"]:.1f}kHz'
        )

    @QtCore.Slot(int, dict)
    def _on_point_completed(self, index, result):
        """Handle point completed signal."""
        # Add result to table
        sorting_enabled = self._results_table.isSortingEnabled()
        self._results_table.setSortingEnabled(False)
        row = self._results_table.rowCount()
        self._results_table.insertRow(row)

        self._results_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(index + 1)))
        self._results_table.setItem(row, 1, QtWidgets.QTableWidgetItem(
            str(result.get('measurement_mode', 'open_loop'))))
        self._results_table.setItem(row, 2, QtWidgets.QTableWidgetItem(
            str(result.get('controller_algorithm', 'disabled'))))
        bandwidth = result.get('controller_bandwidth_hz')
        smith_gain = result.get('smith_gain_multiplier')
        smith_delay = result.get('smith_delay_samples')
        self._results_table.setItem(row, 3, QtWidgets.QTableWidgetItem(
            '--' if bandwidth is None else f'{bandwidth:g}'))
        self._results_table.setItem(row, 4, QtWidgets.QTableWidgetItem(
            '--' if smith_gain is None else f'{smith_gain:g}'))
        self._results_table.setItem(row, 5, QtWidgets.QTableWidgetItem(
            'auto' if (result.get('controller_algorithm') == 'smith_linear' and
                       smith_delay is None) else
            '--' if smith_delay is None else str(smith_delay)))
        self._results_table.setItem(row, 6, QtWidgets.QTableWidgetItem(f'{result["power_dbm"]:.2f}'))
        self._results_table.setItem(row, 7, QtWidgets.QTableWidgetItem(f'{result["f_mod_hz"]/1e3:.1f}'))
        self._results_table.setItem(row, 8, QtWidgets.QTableWidgetItem(f'{result["f_dev_khz"]:.1f}'))

        linewidth = result.get('linewidth_hz', np.nan)
        sensitivity = result.get('sensitivity_nT_rtHz', np.nan)
        residual = result.get('residual_sensitivity_nT_rtHz', np.nan)

        self._results_table.setItem(row, 9, QtWidgets.QTableWidgetItem(f'{linewidth:.1e}' if not np.isnan(linewidth) else 'N/A'))
        self._results_table.setItem(row, 10, QtWidgets.QTableWidgetItem(f'{sensitivity:.3f}' if not np.isnan(sensitivity) else 'N/A'))
        self._results_table.setItem(row, 11, QtWidgets.QTableWidgetItem(f'{residual:.3f}' if not np.isnan(residual) else 'N/A'))
        self._results_table.setItem(row, 12, QtWidgets.QTableWidgetItem(
            str(result.get('fir_filter_bandwidth', '--'))))

        # Color code sensitivity (green for good, red for bad)
        if not np.isnan(sensitivity):
            sens_item = self._results_table.item(row, 10)
            if sensitivity < 20:  # Arbitrary threshold
                sens_item.setBackground(QtGui.QColor(200, 255, 200))
            elif sensitivity > 50:
                sens_item.setBackground(QtGui.QColor(255, 200, 200))
        self._results_table.setSortingEnabled(sorting_enabled)

    @QtCore.Slot(dict)
    def _on_sweep_progress(self, progress):
        """Handle sweep progress signal."""
        current = progress['current_idx']
        total = progress['total']
        best_sens = progress['best_sens']
        best_params = progress['best_params']
        percent = progress['completion_percent']

        self._progress_bar.setValue(current)
        self._progress_label.setText(f'Running: {current} / {total} ({percent:.1f}%)')

        if not np.isinf(best_sens):
            controller = best_params.get('controller_algorithm', 'disabled')
            bandwidth = best_params.get('controller_bandwidth_hz')
            controller_text = controller
            if best_params.get('fir_filter_bandwidth'):
                controller_text += ', ' + best_params['fir_filter_bandwidth']
            if bandwidth is not None:
                controller_text += f', {bandwidth:g} Hz'
            if best_params.get('smith_gain_multiplier') is not None:
                controller_text += f', x{best_params["smith_gain_multiplier"]:g}'
            self._best_result_label.setText(
                f'Best sensitivity: {best_sens:.3f} nT/√Hz\n'
                f'(P={best_params.get("power", 0):.2f}dBm, '
                f'f_mod={best_params.get("f_mod", 0)/1e3:.1f}kHz, '
                f'f_dev={best_params.get("f_dev", 0):.1f}kHz; '
                f'{controller_text})'
            )

    @QtCore.Slot()
    def _on_sweep_paused(self):
        """Handle sweep paused signal."""
        self._pause_button.setEnabled(False)
        self._resume_button.setEnabled(True)
        self._cancel_button.setEnabled(True)
        self._status_label.setText('Sweep paused')
        QtWidgets.QMessageBox.information(
            self._mw,
            'Sweep Paused',
            'Sweep has been paused. Click RESUME to continue.'
        )

    @QtCore.Slot()
    def _on_sweep_resumed(self):
        """Handle sweep resumed signal."""
        self._pause_button.setEnabled(True)
        self._resume_button.setEnabled(False)
        self._cancel_button.setEnabled(True)
        self._status_label.setText('Sweep resumed')

    @QtCore.Slot(str)
    def _on_sweep_finished(self, results_folder):
        """Handle sweep finished signal."""
        self._start_button.setEnabled(True)
        self._start_button.setStyleSheet('QPushButton { font-weight: bold; background-color: #4CAF50; color: white; min-height: 40px; }')
        self._pause_button.setEnabled(False)
        self._resume_button.setEnabled(False)
        self._cancel_button.setEnabled(False)
        self._status_label.setText(f'Sweep complete! Results: {results_folder}')

        QtWidgets.QMessageBox.information(
            self._mw,
            'Sweep Complete',
            f'Parameter sweep completed successfully!\n\nResults saved to:\n{results_folder}'
        )

    @QtCore.Slot()
    def _on_sweep_cancelled(self):
        """Handle sweep cancelled signal."""
        self._start_button.setEnabled(True)
        self._start_button.setStyleSheet('QPushButton { font-weight: bold; background-color: #4CAF50; color: white; min-height: 40px; }')
        self._pause_button.setEnabled(False)
        self._resume_button.setEnabled(False)
        self._cancel_button.setEnabled(False)
        self._status_label.setText('Sweep cancelled')

        QtWidgets.QMessageBox.warning(
            self._mw,
            'Sweep Cancelled',
            'Parameter sweep was cancelled by user.'
        )

    @QtCore.Slot(str)
    def _on_error(self, error_msg):
        """Handle error signal."""
        self.log.error(f'Error from logic: {error_msg}')
        QtWidgets.QMessageBox.critical(
            self._mw,
            'Error',
            f'An error occurred:\n\n{error_msg}'
        )

    @QtCore.Slot(object, object)
    def _update_odmr_plot(self, frequencies, signal):
        """Update ODMR plot with new data."""
        if frequencies is not None and signal is not None:
            self._odmr_curve.setData(x=frequencies, y=signal)

    @QtCore.Slot(dict)
    def _update_fit_plot(self, fit_result):
        """Update fit curve on ODMR plot."""
        fit_freq = fit_result.get('fit_frequency')
        fit_data = fit_result.get('fit_data')

        if fit_freq is not None and fit_data is not None:
            self._fit_curve.setData(x=fit_freq, y=fit_data)

    @QtCore.Slot(object, object, float)
    def _update_asd_plot(self, frequencies, asd_data, sensitivity):
        """Update ASD plot with new data and sensitivity line."""
        if frequencies is not None and asd_data is not None:
            self._asd_curve.setData(x=frequencies, y=asd_data)

            # Calculate sensitivity using current GUI bandwidth settings
            f_min = self._sensitivity_f_min_spinbox.value()
            f_max = self._sensitivity_f_max_spinbox.value()
            exclude_harmonics = self._exclude_50hz_checkbox.isChecked()
            calculated_sensitivity = self._calculate_asd_noise_floor(
                frequencies, asd_data, f1=f_min, f2=f_max,
                exclude_50hz_harmonics=exclude_harmonics
            )

            # Update sensitivity line (use log10 because plot is in log mode)
            if not np.isnan(calculated_sensitivity) and calculated_sensitivity > 0:
                self._sensitivity_line.setValue(np.log10(calculated_sensitivity))
                # Include bandwidth info in label
                excl_str = ', excl. 50Hz' if exclude_harmonics else ''
                self._sensitivity_line.label.setFormat(
                    f'{calculated_sensitivity:.2f} nT/√Hz ({f_min:.0f}-{f_max:.0f} Hz{excl_str})'
                )
                self._sensitivity_line.setVisible(True)

            # Update bandwidth indicator lines (use log10 because plot is in log mode)
            if f_min > 0:
                self._bandwidth_line_min.setValue(np.log10(f_min))
                self._bandwidth_line_min.setVisible(True)
            if f_max > 0:
                self._bandwidth_line_max.setValue(np.log10(f_max))
                self._bandwidth_line_max.setVisible(True)

    def _calculate_asd_noise_floor(self, frequencies, asd_data, f1=200, f2=1400,
                                   exclude_50hz_harmonics=True):
        """
        Calculate noise floor from ASD data in specified frequency range.

        Args:
            frequencies: Frequency array in Hz
            asd_data: ASD values array
            f1: Lower frequency bound (default 200 Hz)
            f2: Upper frequency bound (default 1400 Hz)
            exclude_50hz_harmonics: If True, exclude 50 Hz harmonics and their neighbors

        Returns:
            Mean ASD value in the specified range (noise floor / sensitivity)
        """
        if frequencies is None or asd_data is None or len(frequencies) == 0:
            return np.nan

        frequencies = np.asarray(frequencies)
        asd_data = np.asarray(asd_data)

        # Start with frequency range mask
        mask = (frequencies >= f1) & (frequencies <= f2)

        # Exclude 50 Hz harmonics and neighboring bins if requested
        if exclude_50hz_harmonics:
            # Generate all 50 Hz harmonics within the frequency range
            max_harmonic = int(f2 / 50) + 1
            harmonics_50hz = [50 * n for n in range(1, max_harmonic + 1) if f1 <= 50 * n <= f2]
            for harmonic in harmonics_50hz:
                # Exclude frequencies within ±1.5 Hz of each harmonic
                mask &= (np.abs(frequencies - harmonic) > 1.5)

        if not np.any(mask):
            self.log.warning('No frequency bins remaining after filtering for noise floor calculation')
            return np.nan

        return np.mean(asd_data[mask])

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @QtCore.Slot()
    def _export_results(self):
        """Export results table to CSV file."""
        logic = self._sensitivity_logic()
        df = logic.get_current_results()

        if df.empty:
            QtWidgets.QMessageBox.warning(
                self._mw,
                'No Data',
                'No results to export yet.'
            )
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._mw,
            'Export Results',
            'sensitivity_results.csv',
            'CSV Files (*.csv);;All Files (*)'
        )

        if filename:
            try:
                df.to_csv(filename, sep='\t', index=False, na_rep='NaN')
                QtWidgets.QMessageBox.information(
                    self._mw,
                    'Export Successful',
                    f'Results exported to:\n{filename}'
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self._mw,
                    'Export Failed',
                    f'Failed to export results:\n{str(e)}'
                )

    @QtCore.Slot()
    def _restore_default_view(self):
        """Restore default dock widget layout."""
        self._config_dock.setFloating(False)
        self._odmr_dock.setFloating(False)
        self._asd_dock.setFloating(False)
        self._results_dock.setFloating(False)

        self._mw.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._config_dock)
        self._mw.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._odmr_dock)
        self._mw.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._asd_dock)
        self._mw.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._results_dock)

        # Stack docks vertically on right: ODMR on top, ASD in middle, Results at bottom
        self._mw.splitDockWidget(self._odmr_dock, self._asd_dock, QtCore.Qt.Vertical)
        self._mw.splitDockWidget(self._asd_dock, self._results_dock, QtCore.Qt.Vertical)

        # Show all docks
        self._config_dock.show()
        self._odmr_dock.show()
        self._asd_dock.show()
        self._results_dock.show()

    @QtCore.Slot()
    def _show_about(self):
        """Show about dialog."""
        QtWidgets.QMessageBox.about(
            self._mw,
            'About Sensitivity Sweep',
            'ODMR-Based Magnetic Field Sensitivity Parameter Sweep\n\n'
            'This module automates systematic exploration of microwave '
            'parameters to optimize magnetic field sensitivity.\n\n'
            'Part of the Qudi measurement suite.'
        )
