# -*- coding: utf-8 -*-
"""
ODMR frequency-tracking GUI (N = 1 or 2 resonances).

Extends the standard ODMR GUI with tracking controls:
- ODMR scan (inherited), then select a frequency region per resonance directly
  on the plot and linear-fit its zero crossing and slope ("Fit Res i");
- configure and start tracking (N = 1: continuous on resonance 0; N = 2: hardware
  LO hopping), and watch the per-resonance errors and corrections live.

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

__all__ = ['OdmrTrackingGui']

import numpy as np
import pyqtgraph as pg
from PySide2 import QtCore, QtWidgets

from qudi.core.statusvariable import StatusVar
from qudi.util.colordefs import QudiPalettePale as palette
from qudi.util.widgets.scientific_spinbox import ScienDSpinBox
from qudi.util.widgets.advanced_dockwidget import AdvancedDockWidget
from qudi.gui.odmr.odmrgui import OdmrGui

# per-resonance display colours (region tint + trace pen)
_RES_COLORS = [palette.c1, palette.c3]
_N_RES = 2
_RIGHT_SIDEBAR_WIDTH = 380
_FIT_SUBWINDOW_FRACTION = 0.1


class OdmrTrackingGui(OdmrGui):
    """
    ODMR GUI extended with frequency tracking of one or two resonances.

    Example config:

        odmr_tracking_gui:
            module.Class: 'odmr_tracking.odmr_tracking_gui.OdmrTrackingGui'
            connect:
                odmr_logic: 'odmr_tracking_logic'
    """

    # ---- persisted GUI state ----
    # Historical name kept for compatibility: these are the outer scan windows.
    _fit_ranges = StatusVar('fit_ranges', default=[(2.86e9, 2.88e9), (2.90e9, 2.92e9)])
    _fit_sub_ranges = StatusVar('fit_sub_ranges', default=[])
    _gui_num_resonances = StatusVar('num_resonances', default=2)
    _gui_modulation_frequency = StatusVar('modulation_frequency', default=15258.789)
    _gui_demod_phase = StatusVar('demod_phase_deg', default=0.0)
    _gui_dwell_time = StatusVar('dwell_time', default=1.0e-3)
    _gui_settle_time = StatusVar('settle_time', default=200e-6)
    _gui_bandwidth = StatusVar('lock_bandwidth', default=150.0)
    _gui_max_correction = StatusVar('max_correction_hz', default=1.0e6)
    _gui_controller = StatusVar('controller', default='integral')
    _gui_zero_ratio = StatusVar('zero_ratio', default=0.2)
    _gui_detail_points = StatusVar('detail_points', default=101)
    _gui_trace_window_s = StatusVar('trace_window_s', default=20.0)

    # ---- GUI -> logic signals (QueuedConnection: run on the logic thread) ----
    sigFitResonance = QtCore.Signal(int, float, float)
    sigConfigureTracking = QtCore.Signal(dict)
    sigStartTracking = QtCore.Signal()
    sigStopTracking = QtCore.Signal()
    sigStartStreaming = QtCore.Signal()
    sigStopStreaming = QtCore.Signal()
    sigClearIntegrators = QtCore.Signal()
    sigSetDemodPhase = QtCore.Signal(float)
    sigSetScanRegion = QtCore.Signal(float, float, int)  # lo, hi, points (single range)
    sigSetNumResonances = QtCore.Signal(int)
    sigSetTraceWindow = QtCore.Signal(float)  # high-rate display window [s]
    sigSaveTrackingData = QtCore.Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fit_regions = []     # outer scan LinearRegionItem per resonance
        self._fit_sub_regions = [] # inner fit LinearRegionItem per resonance
        self._fit_curves = []      # pg.PlotDataItem per resonance (fit overlay)
        self._res_min_spin = []    # freq-min spinbox per resonance
        self._res_max_spin = []
        self._fit_min_spin = []    # fit-window min spinbox per resonance
        self._fit_max_spin = []
        self._res_groups = []
        self._res_labels = []      # dict of result/status labels per resonance
        self._last_scan_widths = [None] * _N_RES
        self._region_update_guard = False
        self._control_dock = None
        self._trace_dock = None
        self._corr_plot = None                # bottom plot: Correction [Hz]
        self._err_plot = None                 # top plot: Error [LSB]
        self._cic_plot = None                 # middle plot: post-CIC/pre-FIR [LSB]
        self._corr_curves = []                # one per resonance (Hz)
        self._err_curves = []                 # one per resonance (LSB)
        self._cic_curves = []                 # one per resonance (CIC LSB)
        # When the high-rate dual-quantity stream is live it drives both plots; the
        # low-rate register-poll history is the fallback until the first high-rate
        # frame arrives (or when stream_traces is disabled).
        self._got_high_rate = False
        self._tracking_is_active = False
        self._stream_is_active = False
        self._controls_configured = False
        self._corr_hist = None                # (times, (N x T) Hz)  low-rate fallback
        self._err_hist = None                 # (times, (N x T) LSB) low-rate fallback

    # =========================================================================
    # Activation
    # =========================================================================
    def on_activate(self):
        super().on_activate()
        self._hide_unused_parent_elements()
        self._add_fit_regions_to_plot()
        self._create_control_dock()
        self._create_trace_dock()
        self._connect_signals()
        self._restore_settings()
        self._mw.setWindowTitle('ODMR Tracking')
        self._mw.action_save_measurement.setToolTip(
            'Save the current ODMR scan and synchronized post-FIR error, frequency '
            'correction, and post-CIC/pre-FIR traces for the selected Window.')
        self.restore_default_view()
        self.log.info('ODMR tracking GUI activated')

    def on_deactivate(self):
        self._disconnect_signals()
        super().on_deactivate()
        self.log.info('ODMR tracking GUI deactivated')

    def _hide_unused_parent_elements(self):
        if getattr(self, '_fit_dockwidget', None) is not None:
            self._fit_dockwidget.hide()
            self._mw.removeDockWidget(self._fit_dockwidget)
        if hasattr(self._mw, 'action_show_fit_configuration'):
            self._mw.action_show_fit_configuration.setVisible(False)
        pw = self._plot_widget
        if hasattr(pw, '_image_widget'):
            pw._image_widget.hide()
        if hasattr(pw, '_colorbar'):
            pw._colorbar.hide()

    # =========================================================================
    # Plot overlays: outer scan regions + inner fit regions + fit curves
    # =========================================================================
    def _add_fit_regions_to_plot(self):
        plot_item = self._plot_widget._plot_widget.getPlotItem()
        for i in range(_N_RES):
            col = _RES_COLORS[i]
            region = pg.LinearRegionItem(
                brush=pg.mkBrush(col.getRgb()[0], col.getRgb()[1], col.getRgb()[2], 35),
                movable=True)
            region.setZValue(5)
            region.sigRegionChanged.connect(lambda *a, idx=i: self._on_region_changed(idx))
            plot_item.addItem(region, ignoreBounds=True)

            sub_region = pg.LinearRegionItem(
                brush=pg.mkBrush(col.getRgb()[0], col.getRgb()[1], col.getRgb()[2], 120),
                movable=True)
            sub_region.setZValue(15)
            sub_region.sigRegionChanged.connect(
                lambda *a, idx=i: self._on_fit_sub_region_changed(idx))
            plot_item.addItem(sub_region, ignoreBounds=True)

            curve = pg.PlotDataItem(pen=pg.mkPen(col, width=3, style=QtCore.Qt.DashLine))
            curve.setZValue(20)
            plot_item.addItem(curve)
            self._fit_regions.append(region)
            self._fit_sub_regions.append(sub_region)
            self._fit_curves.append(curve)

    # =========================================================================
    # Control dock: per-resonance fit + global tracking params + status
    # =========================================================================
    def _create_control_dock(self):
        self._control_dock = AdvancedDockWidget('Tracking', parent=self._mw)
        self._control_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        widget = QtWidgets.QWidget()
        main = QtWidgets.QVBoxLayout()
        widget.setLayout(main)

        # --- N=1/N=2 mode ---
        mode_grp = QtWidgets.QGroupBox('Tracking Mode')
        mode_form = QtWidgets.QFormLayout()
        self._mode_combo = QtWidgets.QComboBox()
        self._mode_combo.addItem('N=1 single resonance', 1)
        self._mode_combo.addItem('N=2 dual resonance', 2)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._mode_combo.setToolTip(
            'Select how many resonances are actively fitted, configured and tracked.')
        mode_form.addRow('Mode:', self._mode_combo)
        mode_grp.setLayout(mode_form)
        main.addWidget(mode_grp)

        # --- detail-scan setup: scan each resonance's region as a fast single-range
        #     EQUIDISTANT sweep, one at a time (no JUMP_LIST: too slow/small on the
        #     Windfreak for high-res sweeps). ---
        detail_grp = QtWidgets.QGroupBox('Detail Scan (fast single-range sweep)')
        detail_form = QtWidgets.QFormLayout()
        self._detail_points_spin = QtWidgets.QSpinBox()
        self._detail_points_spin.setRange(2, 100001)
        self._detail_points_spin.setValue(int(self._gui_detail_points))
        self._detail_points_spin.setToolTip('Points for each per-resonance detail sweep.')
        info = QtWidgets.QLabel(
            '<i>Drag the pale outer region onto a feature, click "Use Region i as '
            'scan range", run Start Scan, then fit the darker centered sub-window. '
            'Repeat for the active resonances.</i>')
        info.setWordWrap(True)
        info.setStyleSheet('color: gray; font-size: 10px;')
        detail_form.addRow('Scan points:', self._detail_points_spin)
        detail_form.addRow(info)
        detail_grp.setLayout(detail_form)
        main.addWidget(detail_grp)

        # --- per-resonance fit groups ---
        for i in range(_N_RES):
            col = _RES_COLORS[i]
            grp = QtWidgets.QGroupBox(f'Resonance {i}')
            grp.setStyleSheet(f'QGroupBox {{ color: {col.name()}; font-weight: bold; }}')
            form = QtWidgets.QFormLayout()

            fmin = ScienDSpinBox(); fmin.setRange(0, 1e12); fmin.setSuffix('Hz')
            fmin.setDecimals(6); fmin.setMinimumWidth(120)
            fmax = ScienDSpinBox(); fmax.setRange(0, 1e12); fmax.setSuffix('Hz')
            fmax.setDecimals(6); fmax.setMinimumWidth(120)
            fmin.valueChanged.connect(lambda _v, idx=i: self._update_region_from_spin(idx))
            fmax.valueChanged.connect(lambda _v, idx=i: self._update_region_from_spin(idx))

            fit_min = ScienDSpinBox(); fit_min.setRange(0, 1e12); fit_min.setSuffix('Hz')
            fit_min.setDecimals(6); fit_min.setMinimumWidth(120)
            fit_max = ScienDSpinBox(); fit_max.setRange(0, 1e12); fit_max.setSuffix('Hz')
            fit_max.setDecimals(6); fit_max.setMinimumWidth(120)
            fit_min.valueChanged.connect(
                lambda _v, idx=i: self._update_fit_sub_region_from_spin(idx))
            fit_max.valueChanged.connect(
                lambda _v, idx=i: self._update_fit_sub_region_from_spin(idx))

            use_range_btn = QtWidgets.QPushButton(f'Use Region {i} as scan range')
            use_range_btn.setToolTip('Set the ODMR scan to a single range = this '
                                     'region (fast sweep), then click Start Scan.')
            use_range_btn.clicked.connect(lambda _c=False, idx=i: self._use_as_scan_range(idx))

            fit_btn = QtWidgets.QPushButton(f'Fit Res {i}')
            fit_btn.clicked.connect(lambda _c=False, idx=i: self._do_fit(idx))

            slope_lbl = QtWidgets.QLabel('--')
            r2_lbl = QtWidgets.QLabel('--')
            zc_lbl = QtWidgets.QLabel('--')
            locked_lbl = QtWidgets.QLabel('OFF')
            corr_lbl = QtWidgets.QLabel('--')
            err_lbl = QtWidgets.QLabel('--')

            form.addRow('Scan Min:', fmin)
            form.addRow('Scan Max:', fmax)
            form.addRow('Fit Min:', fit_min)
            form.addRow('Fit Max:', fit_max)
            form.addRow(use_range_btn)
            form.addRow(fit_btn)
            form.addRow('Slope [LSB/Hz]:', slope_lbl)
            form.addRow('R²:', r2_lbl)
            form.addRow('Zero-Crossing:', zc_lbl)
            form.addRow('Locked:', locked_lbl)
            form.addRow('Correction [Hz]:', corr_lbl)
            form.addRow('Error [LSB]:', err_lbl)
            grp.setLayout(form)
            main.addWidget(grp)

            self._res_min_spin.append(fmin)
            self._res_max_spin.append(fmax)
            self._fit_min_spin.append(fit_min)
            self._fit_max_spin.append(fit_max)
            self._res_groups.append(grp)
            self._res_labels.append({'slope': slope_lbl, 'r2': r2_lbl, 'zc': zc_lbl,
                                     'locked': locked_lbl, 'correction': corr_lbl,
                                     'error': err_lbl})

        # --- global tracking parameters ---
        pgrp = QtWidgets.QGroupBox('Tracking Parameters')
        pform = QtWidgets.QFormLayout()
        self._fm_spin = ScienDSpinBox(); self._fm_spin.setRange(1e3, 1e6)
        self._fm_spin.setSuffix('Hz'); self._fm_spin.setDecimals(3)
        self._demod_phase_spin = QtWidgets.QDoubleSpinBox()
        self._demod_phase_spin.setRange(0.0, 360.0); self._demod_phase_spin.setDecimals(1)
        self._demod_phase_spin.setSuffix(' deg')
        self._demod_phase_spin.setToolTip('Demod phase; can be tuned live while tracking.')
        self._demod_phase_spin.valueChanged.connect(self._on_demod_phase_changed)
        self._dwell_spin = ScienDSpinBox(); self._dwell_spin.setRange(10e-6, 1.0)
        self._dwell_spin.setSuffix('s'); self._dwell_spin.setDecimals(6)
        self._settle_spin = ScienDSpinBox(); self._settle_spin.setRange(0.0, 0.1)
        self._settle_spin.setSuffix('s'); self._settle_spin.setDecimals(6)
        self._bw_spin = ScienDSpinBox(); self._bw_spin.setRange(0.1, 1e6)
        self._bw_spin.setSuffix('Hz'); self._bw_spin.setDecimals(1)
        self._maxcorr_spin = ScienDSpinBox(); self._maxcorr_spin.setRange(1e3, 62.5e6)
        self._maxcorr_spin.setSuffix('Hz'); self._maxcorr_spin.setDecimals(0)
        self._controller_combo = QtWidgets.QComboBox()
        try:
            controllers = self._odmr_logic().available_controllers
        except Exception:
            controllers = {'integral': 'integral only', 'pi': 'PI'}
        for name, description in controllers.items():
            label = {'integral': 'Integral', 'pi': 'PI'}.get(name, name.upper())
            self._controller_combo.addItem(label, name)
            self._controller_combo.setItemData(
                self._controller_combo.count() - 1, description, QtCore.Qt.ToolTipRole)
        self._controller_combo.setToolTip(
            'Integral: zero steady-state error; a drifting resonance is followed '
            'with a lag proportional to the drift rate. PI: adds a proportional term '
            '(zero at bandwidth / zero ratio) that damps the loop; with ratio 0.2 it '
            'allows about 20 % more bandwidth at the same stability margins. '
            'Further entries are controllers of the FPGA image (PII: a second '
            'integrator, follows a drift without lag).')
        self._zero_ratio_spin = QtWidgets.QDoubleSpinBox()
        self._zero_ratio_spin.setRange(0.1, 30.0); self._zero_ratio_spin.setDecimals(2)
        self._zero_ratio_spin.setToolTip(
            'PI zero at bandwidth / zero ratio. The proportional loop gain is '
            'ratio / sqrt(1 + ratio^2); 0.2 is robust, 1 and above ring '
            '(the predicted margins are in the log).')
        self._controller_combo.currentIndexChanged.connect(
            lambda _i: self._zero_ratio_spin.setEnabled(
                self._controller_combo.currentData() != 'integral'))
        self._read_phase_btn = QtWidgets.QPushButton('← HW')
        self._read_phase_btn.setToolTip(
            'Read the demodulation phase currently set in hardware (shared with ODMR '
            'scans, e.g. after a phase calibration).')
        self._read_phase_btn.clicked.connect(self._read_hw_phase)
        _dp_row = QtWidgets.QHBoxLayout()
        _dp_row.addWidget(self._demod_phase_spin)
        _dp_row.addWidget(self._read_phase_btn)
        self._dwell_label = QtWidgets.QLabel('Dwell time:')
        self._settle_label = QtWidgets.QLabel('Settle time:')
        timing_tip = ('Hop timing (N=2 only). With N=1 the lock runs continuously '
                      'on resonance 0, so dwell/settle timing is not used.')
        self._dwell_spin.setToolTip(timing_tip)
        self._settle_spin.setToolTip(timing_tip)
        self._dwell_label.setToolTip(timing_tip)
        self._settle_label.setToolTip(timing_tip)
        pform.addRow('f_m (modulation):', self._fm_spin)
        pform.addRow('Demod phase:', _dp_row)
        pform.addRow(self._dwell_label, self._dwell_spin)
        pform.addRow(self._settle_label, self._settle_spin)
        pform.addRow('Controller:', self._controller_combo)
        pform.addRow('PI zero ratio:', self._zero_ratio_spin)
        pform.addRow('Bandwidth:', self._bw_spin)
        pform.addRow('Max correction:', self._maxcorr_spin)
        pgrp.setLayout(pform)
        main.addWidget(pgrp)

        # --- action buttons ---
        self._configure_btn = QtWidgets.QPushButton('Configure Tracking')
        self._configure_btn.clicked.connect(self._configure_tracking)
        self._start_btn = QtWidgets.QPushButton('Start Tracking')
        self._start_btn.clicked.connect(lambda: self.sigStartTracking.emit())
        self._start_btn.setEnabled(False)
        self._stop_btn = QtWidgets.QPushButton('Stop Tracking')
        self._stop_btn.clicked.connect(lambda: self.sigStopTracking.emit())
        self._stop_btn.setEnabled(False)
        self._start_stream_btn = QtWidgets.QPushButton('Start Stream')
        self._start_stream_btn.setToolTip(
            'Start simultaneous demod-error/correction acquisition without enabling the lock.')
        self._start_stream_btn.clicked.connect(lambda: self.sigStartStreaming.emit())
        self._start_stream_btn.setEnabled(False)
        self._stop_stream_btn = QtWidgets.QPushButton('Stop Stream')
        self._stop_stream_btn.setToolTip(
            'Stop trace acquisition without changing the FPGA lock state.')
        self._stop_stream_btn.clicked.connect(lambda: self.sigStopStreaming.emit())
        self._stop_stream_btn.setEnabled(False)
        self._stream_state_label = QtWidgets.QLabel('Stream: OFF')
        self._clear_btn = QtWidgets.QPushButton('Clear Integrators')
        self._clear_btn.clicked.connect(lambda: self.sigClearIntegrators.emit())
        brow = QtWidgets.QHBoxLayout()
        brow.addWidget(self._start_btn); brow.addWidget(self._stop_btn)
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(self._start_stream_btn); srow.addWidget(self._stop_stream_btn)
        main.addWidget(self._configure_btn)
        main.addLayout(brow)
        main.addLayout(srow)
        main.addWidget(self._stream_state_label)
        main.addWidget(self._clear_btn)
        main.addStretch()

        # Wrap in a scroll area so all controls remain reachable even when the dock
        # is short (otherwise the bottom buttons get clipped).
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        scroll.setMinimumWidth(widget.sizeHint().width() + 24)
        self._control_dock.setWidget(scroll)
        self._mw.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._control_dock)
        # Give the right dock priority in the bottom-right corner so the (bottom)
        # trace dock does not overlap the control column.
        self._mw.setCorner(QtCore.Qt.BottomRightCorner, QtCore.Qt.RightDockWidgetArea)

    def _create_trace_dock(self):
        self._trace_dock = AdvancedDockWidget('Resonance Time-Traces', parent=self._mw)
        self._trace_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(2, 2, 2, 2)
        container.setLayout(vbox)

        # Three stacked plots on the same FPGA time axis:
        #   top    = post-FIR error [LSB]
        #   middle = post-CIC/pre-FIR error [LSB]
        #   bottom = frequency correction [Hz]
        # Fed by the dual-quantity stream; falls back to the ~5 Hz register-poll
        # history when the high-rate stream is not (yet) running.
        header_row = QtWidgets.QHBoxLayout()
        header_row.addWidget(QtWidgets.QLabel('High-rate traces (both resonances)'))
        header_row.addStretch()
        header_row.addWidget(QtWidgets.QLabel('Window:'))
        self._trace_window_spin = QtWidgets.QDoubleSpinBox()
        # Capped at 60 s: this is the FULL-rate (~30.5 kHz) buffer, so longer windows
        # cost significant memory/CPU per poll. For minute+ drift, watch the per-slot
        # correction status instead.
        self._trace_window_spin.setRange(0.5, 60.0)
        self._trace_window_spin.setDecimals(1)
        self._trace_window_spin.setSingleStep(1.0)
        self._trace_window_spin.setSuffix(' s')
        self._trace_window_spin.setValue(float(self._gui_trace_window_s))
        self._trace_window_spin.setToolTip(
            'Duration of the shown rolling high-rate window. The display uses a fixed '
            'point budget, so a SHORTER window shows finer detail (hop-scale), a LONGER '
            'window shows more drift history. This also sets the high-rate snapshot '
            'available to Save; the low-rate long-duration history is unaffected. '
            'The plot is continuously rebased to 0..Window, while saved time remains '
            'the monotonic elapsed time since stream start.')
        self._trace_window_spin.valueChanged.connect(self._on_trace_window_changed)
        header_row.addWidget(self._trace_window_spin)
        vbox.addLayout(header_row)

        self._err_plot = pg.PlotWidget()
        self._err_plot.setLabel('left', 'Error', units='LSB')
        self._err_plot.setLabel('bottom', 'Time', units='s')
        self._err_plot.setMinimumHeight(150)
        self._err_plot.showGrid(x=True, y=True)
        self._err_plot.disableAutoRange(axis='x')
        self._err_plot.setMouseEnabled(x=False, y=True)
        self._err_plot.addLegend()
        for i in range(_N_RES):
            self._err_curves.append(
                self._err_plot.plot(pen=pg.mkPen(_RES_COLORS[i], width=2),
                                    name=f'Resonance {i}'))
        vbox.addWidget(self._err_plot)

        self._cic_plot = pg.PlotWidget()
        self._cic_plot.setLabel('left', 'After CIC', units='LSB')
        self._cic_plot.setLabel('bottom', 'Time', units='s')
        self._cic_plot.setMinimumHeight(130)
        self._cic_plot.showGrid(x=True, y=True)
        self._cic_plot.disableAutoRange(axis='x')
        self._cic_plot.setMouseEnabled(x=False, y=True)
        self._cic_plot.addLegend()
        for i in range(_N_RES):
            self._cic_curves.append(
                self._cic_plot.plot(pen=pg.mkPen(_RES_COLORS[i], width=2),
                                    name=f'Resonance {i}'))
        self._cic_plot.setXLink(self._err_plot)
        vbox.addWidget(self._cic_plot)

        self._corr_plot = pg.PlotWidget()
        self._corr_plot.setLabel('left', 'Correction', units='Hz')
        self._corr_plot.setLabel('bottom', 'Time', units='s')
        self._corr_plot.setMinimumHeight(150)
        self._corr_plot.showGrid(x=True, y=True)
        self._corr_plot.disableAutoRange(axis='x')
        self._corr_plot.setMouseEnabled(x=False, y=True)
        self._corr_plot.addLegend()
        for i in range(_N_RES):
            self._corr_curves.append(
                self._corr_plot.plot(pen=pg.mkPen(_RES_COLORS[i], width=2),
                                     name=f'Resonance {i}'))
        # share the time axis so panning/zooming stays aligned
        self._corr_plot.setXLink(self._err_plot)
        self._set_trace_x_range()
        vbox.addWidget(self._corr_plot)

        self._trace_dock.setWidget(container)
        self._mw.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self._trace_dock)

    # =========================================================================
    # Signal wiring
    # =========================================================================
    def _connect_signals(self):
        logic = self._odmr_logic()
        self.sigFitResonance.connect(logic.fit_resonance, QtCore.Qt.QueuedConnection)
        self.sigConfigureTracking.connect(
            logic.configure_tracking_from_params, QtCore.Qt.QueuedConnection)
        self.sigStartTracking.connect(logic.start_tracking, QtCore.Qt.QueuedConnection)
        self.sigStopTracking.connect(logic.stop_tracking, QtCore.Qt.QueuedConnection)
        self.sigStartStreaming.connect(logic.start_streaming, QtCore.Qt.QueuedConnection)
        self.sigStopStreaming.connect(logic.stop_streaming, QtCore.Qt.QueuedConnection)
        self.sigClearIntegrators.connect(logic.clear_integrators, QtCore.Qt.QueuedConnection)
        self.sigSetDemodPhase.connect(logic.set_demod_phase, QtCore.Qt.QueuedConnection)
        self.sigSetScanRegion.connect(logic.set_scan_region, QtCore.Qt.QueuedConnection)
        if hasattr(logic, 'set_num_resonances'):
            self.sigSetNumResonances.connect(logic.set_num_resonances, QtCore.Qt.QueuedConnection)
        if hasattr(logic, 'set_trace_window_seconds'):
            self.sigSetTraceWindow.connect(logic.set_trace_window_seconds, QtCore.Qt.QueuedConnection)
        if hasattr(logic, 'save_tracking_data'):
            self.sigSaveTrackingData.connect(
                logic.save_tracking_data, QtCore.Qt.QueuedConnection)

        logic.sigResonanceFitCompleted.connect(self._update_fit_display, QtCore.Qt.QueuedConnection)
        logic.sigTrackingStateChanged.connect(self._update_tracking_state, QtCore.Qt.QueuedConnection)
        logic.sigStreamStateChanged.connect(self._update_stream_state, QtCore.Qt.QueuedConnection)
        logic.sigConfigurationCompleted.connect(
            self._on_configuration_completed, QtCore.Qt.QueuedConnection)
        logic.sigSlotStatusUpdated.connect(self._update_slot_status, QtCore.Qt.QueuedConnection)
        logic.sigCorrectionHistoryUpdated.connect(self._on_corr_history, QtCore.Qt.QueuedConnection)
        logic.sigErrorHistoryUpdated.connect(self._on_err_history, QtCore.Qt.QueuedConnection)
        logic.sigHighRateTracesUpdated.connect(self._on_high_rate_traces, QtCore.Qt.QueuedConnection)

    def _disconnect_signals(self):
        try:
            logic = self._odmr_logic()
            logic.sigResonanceFitCompleted.disconnect(self._update_fit_display)
            logic.sigTrackingStateChanged.disconnect(self._update_tracking_state)
            logic.sigStreamStateChanged.disconnect(self._update_stream_state)
            logic.sigConfigurationCompleted.disconnect(self._on_configuration_completed)
            logic.sigSlotStatusUpdated.disconnect(self._update_slot_status)
            logic.sigCorrectionHistoryUpdated.disconnect(self._on_corr_history)
            logic.sigErrorHistoryUpdated.disconnect(self._on_err_history)
            logic.sigHighRateTracesUpdated.disconnect(self._on_high_rate_traces)
            self.sigFitResonance.disconnect()
            self.sigConfigureTracking.disconnect()
            self.sigStartTracking.disconnect()
            self.sigStopTracking.disconnect()
            self.sigStartStreaming.disconnect()
            self.sigStopStreaming.disconnect()
            self.sigClearIntegrators.disconnect()
            self.sigSetDemodPhase.disconnect()
            self.sigSetScanRegion.disconnect()
            self.sigSetNumResonances.disconnect()
            self.sigSetTraceWindow.disconnect()
            self.sigSaveTrackingData.disconnect()
        except (TypeError, RuntimeError):
            pass

    # =========================================================================
    # GUI slots
    # =========================================================================
    def save_data(self):
        """Save tracking snapshot first, then the inherited ODMR scan data."""
        tag = self._mw.save_nametag_lineedit.text()
        self.sigSaveTrackingData.emit(tag)
        self.sigSaveData.emit(tag)

    def _active_resonance_count(self):
        try:
            return int(self._gui_num_resonances)
        except Exception:
            return _N_RES

    def _persist_region(self, name, index, lo, hi):
        ranges = list(getattr(self, name))
        while len(ranges) <= index:
            ranges.append((lo, hi))
        ranges[index] = (float(lo), float(hi))
        setattr(self, name, ranges)

    def _set_fit_sub_region(self, index, center, width, update_spins=True):
        scan_lo, scan_hi = sorted(self._fit_regions[index].getRegion())
        scan_width = max(scan_hi - scan_lo, 0.0)
        if scan_width <= 0:
            return
        width = max(0.0, min(float(width), scan_width))
        if width <= 0:
            width = scan_width * _FIT_SUBWINDOW_FRACTION
        lo = center - width / 2.0
        hi = center + width / 2.0
        if lo < scan_lo:
            lo, hi = scan_lo, scan_lo + width
        if hi > scan_hi:
            hi, lo = scan_hi, scan_hi - width

        self._fit_sub_regions[index].blockSignals(True)
        self._fit_sub_regions[index].setRegion([lo, hi])
        self._fit_sub_regions[index].blockSignals(False)

        if update_spins:
            self._fit_min_spin[index].blockSignals(True)
            self._fit_max_spin[index].blockSignals(True)
            self._fit_min_spin[index].setValue(lo)
            self._fit_max_spin[index].setValue(hi)
            self._fit_min_spin[index].blockSignals(False)
            self._fit_max_spin[index].blockSignals(False)
        self._persist_region('_fit_sub_ranges', index, lo, hi)

    @QtCore.Slot()
    def _on_region_changed(self, index):
        if self._region_update_guard:
            return
        lo, hi = self._fit_regions[index].getRegion()
        lo, hi = sorted((lo, hi))
        self._res_min_spin[index].blockSignals(True)
        self._res_max_spin[index].blockSignals(True)
        self._res_min_spin[index].setValue(lo)
        self._res_max_spin[index].setValue(hi)
        self._res_min_spin[index].blockSignals(False)
        self._res_max_spin[index].blockSignals(False)
        self._persist_region('_fit_ranges', index, lo, hi)

        scan_width = hi - lo
        center = (lo + hi) / 2.0
        old_width = self._last_scan_widths[index]
        sub_lo, sub_hi = sorted(self._fit_sub_regions[index].getRegion())
        sub_width = sub_hi - sub_lo
        if old_width is None or abs(scan_width - old_width) > 1.0:
            sub_width = scan_width * _FIT_SUBWINDOW_FRACTION
        self._last_scan_widths[index] = scan_width
        self._set_fit_sub_region(index, center, sub_width, update_spins=True)

    @QtCore.Slot()
    def _on_fit_sub_region_changed(self, index):
        if self._region_update_guard:
            return
        scan_lo, scan_hi = sorted(self._fit_regions[index].getRegion())
        scan_width = scan_hi - scan_lo
        if scan_width <= 0:
            return
        lo, hi = sorted(self._fit_sub_regions[index].getRegion())
        width = min(max(hi - lo, 0.0), scan_width)
        if width <= 0:
            width = scan_width * _FIT_SUBWINDOW_FRACTION
        center = (scan_lo + scan_hi) / 2.0
        self._set_fit_sub_region(index, center, width, update_spins=True)

    @QtCore.Slot()
    def _update_region_from_spin(self, index):
        lo = self._res_min_spin[index].value()
        hi = self._res_max_spin[index].value()
        if lo < hi:
            self._fit_regions[index].blockSignals(True)
            self._fit_regions[index].setRegion([lo, hi])
            self._fit_regions[index].blockSignals(False)
            self._on_region_changed(index)

    @QtCore.Slot()
    def _update_fit_sub_region_from_spin(self, index):
        lo = self._fit_min_spin[index].value()
        hi = self._fit_max_spin[index].value()
        if lo < hi:
            scan_lo, scan_hi = sorted(self._fit_regions[index].getRegion())
            scan_width = scan_hi - scan_lo
            width = min(hi - lo, scan_width)
            self._set_fit_sub_region(index, (scan_lo + scan_hi) / 2.0, width,
                                     update_spins=True)

    @QtCore.Slot()
    def _use_as_scan_range(self, index):
        """Set the ODMR scan to a single range = region ``index`` (fast sweep)."""
        lo = self._res_min_spin[index].value()
        hi = self._res_max_spin[index].value()
        if lo >= hi:
            QtWidgets.QMessageBox.warning(self._mw, 'Invalid Region',
                                          f'Resonance {index}: min must be < max.')
            return
        points = int(self._detail_points_spin.value())
        self._gui_detail_points = points
        self.sigSetScanRegion.emit(lo, hi, points)
        self.log.info(f'Scan range set to region {index} '
                      f'[{lo/1e9:.5f}, {hi/1e9:.5f}] GHz, {points} pts. '
                      'Click Start Scan, then Fit Res {0}.'.format(index))

    @QtCore.Slot()
    def _do_fit(self, index):
        lo, hi = sorted(self._fit_sub_regions[index].getRegion())
        if lo >= hi:
            QtWidgets.QMessageBox.warning(self._mw, 'Invalid Range',
                                          f'Resonance {index}: min must be < max.')
            return
        self._persist_region('_fit_sub_ranges', index, lo, hi)
        self.sigFitResonance.emit(index, lo, hi)

    @QtCore.Slot()
    def _configure_tracking(self):
        n = self._active_resonance_count()
        params = {
            'num_resonances': n,
            'modulation_frequency': self._fm_spin.value(),
            'demod_phase_deg': self._demod_phase_spin.value(),
            'dwell_time': self._dwell_spin.value(),
            'settle_time': self._settle_spin.value(),
            'bandwidth': self._bw_spin.value(),
            'max_correction_hz': self._maxcorr_spin.value(),
            'controller': self._controller_combo.currentData(),
            'zero_ratio': self._zero_ratio_spin.value(),
        }
        # persist
        self._gui_modulation_frequency = params['modulation_frequency']
        self._gui_demod_phase = params['demod_phase_deg']
        self._gui_dwell_time = params['dwell_time']
        self._gui_settle_time = params['settle_time']
        self._gui_bandwidth = params['bandwidth']
        self._gui_max_correction = params['max_correction_hz']
        self._gui_controller = params['controller']
        self._gui_zero_ratio = params['zero_ratio']
        self._gui_num_resonances = n
        self._controls_configured = False
        self._update_action_buttons()
        self.sigConfigureTracking.emit(params)

    @QtCore.Slot(bool, str)
    def _on_configuration_completed(self, success, error_message):
        """Enable run controls only after logic-side configuration completed."""
        self._controls_configured = bool(success)
        self._update_action_buttons()
        if success:
            self.log.info('Tracking configured')
        else:
            self.log.error('Failed to configure tracking: %s',
                           error_message)

    @QtCore.Slot()
    def _read_hw_phase(self):
        """Show the demodulation phase currently set in hardware."""
        try:
            ph = float(self._odmr_logic().get_demod_phase()) % 360.0
        except Exception as e:
            self.log.warning(f'Could not read the demodulation phase: {e}')
            return
        self._demod_phase_spin.setValue(ph)  # re-applies it via _on_demod_phase_changed
        self.log.info(f'Demodulation phase read from hardware: {ph:.1f} deg.')

    @QtCore.Slot()
    def _on_demod_phase_changed(self):
        # live demod-phase tuning (takes effect immediately)
        self.sigSetDemodPhase.emit(self._demod_phase_spin.value())

    @QtCore.Slot(float)
    def _on_trace_window_changed(self, value):
        # live rolling-window duration for the high-rate trace plots
        self._gui_trace_window_s = float(value)
        self._set_trace_x_range()
        self.sigSetTraceWindow.emit(float(value))

    @QtCore.Slot()
    def _on_mode_changed(self):
        n = int(self._mode_combo.currentData())
        self._apply_num_resonances(n, emit=True)

    def _apply_num_resonances(self, n, emit=False):
        n = max(1, min(int(n), _N_RES))
        self._gui_num_resonances = n
        if hasattr(self, '_mode_combo'):
            idx = self._mode_combo.findData(n)
            if idx >= 0 and self._mode_combo.currentIndex() != idx:
                self._mode_combo.blockSignals(True)
                self._mode_combo.setCurrentIndex(idx)
                self._mode_combo.blockSignals(False)
        for i in range(_N_RES):
            visible = i < n
            if i < len(self._res_groups):
                self._res_groups[i].setVisible(visible)
            for items in (self._fit_regions, self._fit_sub_regions, self._fit_curves,
                          self._err_curves, self._corr_curves):
                if i < len(items):
                    items[i].setVisible(visible)
        hop_timing_enabled = n > 1
        for widget in (self._dwell_spin, self._settle_spin,
                       self._dwell_label, self._settle_label):
            widget.setEnabled(hop_timing_enabled)
        self._mw.setWindowTitle(f'ODMR Tracking (N={n})')
        if emit:
            self.sigSetNumResonances.emit(n)

    # =========================================================================
    # Logic update slots
    # =========================================================================
    @QtCore.Slot(int, dict)
    def _update_fit_display(self, index, fit_result):
        if not (0 <= index < _N_RES):
            return
        lbl = self._res_labels[index]
        slope = fit_result.get('slope')
        r2 = fit_result.get('r_squared')
        zc = fit_result.get('zero_crossing_freq')
        if slope is not None:
            lbl['slope'].setText(f'{slope:.3e}')
        if r2 is not None:
            lbl['r2'].setText(f'{r2:.4f}')
        lbl['zc'].setText(f'{zc/1e9:.6f} GHz' if zc else 'N/A')
        ff = fit_result.get('fit_frequency')
        fd = fit_result.get('fit_data')
        if ff is not None and fd is not None:
            self._fit_curves[index].setData(x=ff, y=fd)

    @QtCore.Slot(bool)
    def _update_tracking_state(self, active):
        self._tracking_is_active = bool(active)
        self._update_action_buttons()
        for i in range(self._active_resonance_count()):
            text = 'ON' if active else ('OPEN LOOP' if self._stream_is_active else 'OFF')
            self._res_labels[i]['locked'].setText(text)
            self._res_labels[i]['locked'].setStyleSheet(
                'QLabel { color: green; font-weight: bold; }' if active
                else ('QLabel { color: cyan; }' if self._stream_is_active
                      else 'QLabel { color: red; }'))

    @QtCore.Slot(bool)
    def _update_stream_state(self, active):
        self._stream_is_active = bool(active)
        # Once the stream stops, low-rate register polling should immediately take
        # over the plots if tracking continues.
        self._got_high_rate = bool(active) and self._got_high_rate
        self._stream_state_label.setText('Stream: ON' if active else 'Stream: OFF')
        self._stream_state_label.setStyleSheet(
            'QLabel { color: green; font-weight: bold; }' if active else '')
        self._update_action_buttons()
        if not self._tracking_is_active:
            for i in range(self._active_resonance_count()):
                self._res_labels[i]['locked'].setText('OPEN LOOP' if active else 'OFF')
                self._res_labels[i]['locked'].setStyleSheet(
                    'QLabel { color: cyan; }' if active else 'QLabel { color: red; }')

    def _update_action_buttons(self):
        engine_active = self._tracking_is_active or self._stream_is_active
        self._start_btn.setEnabled(
            self._controls_configured and not self._tracking_is_active)
        self._stop_btn.setEnabled(self._tracking_is_active)
        self._start_stream_btn.setEnabled(
            self._controls_configured and not self._stream_is_active)
        self._stop_stream_btn.setEnabled(self._stream_is_active)
        self._configure_btn.setEnabled(not engine_active)
        self._mode_combo.setEnabled(not engine_active)

    @QtCore.Slot(list)
    def _update_slot_status(self, status_list):
        for i, st in enumerate(status_list):
            if i >= self._active_resonance_count():
                break
            lbl = self._res_labels[i]
            locked = bool(st.get('locked', False))
            sat = bool(st.get('saturated', False))
            if self._tracking_is_active:
                lbl['locked'].setText('LOCKED' if locked else ('SAT' if sat else 'tracking'))
                lbl['locked'].setStyleSheet(
                    'QLabel { color: orange; font-weight: bold; }' if sat
                    else ('QLabel { color: green; font-weight: bold; }' if locked
                          else 'QLabel { color: yellow; }'))
            else:
                lbl['locked'].setText('OPEN LOOP' if self._stream_is_active else 'OFF')
                lbl['locked'].setStyleSheet(
                    'QLabel { color: cyan; }' if self._stream_is_active
                    else 'QLabel { color: red; }')
            lbl['correction'].setText(f"{st.get('correction_hz', float('nan')):.3e}")
            lbl['error'].setText(f"{st.get('error_lsb', float('nan')):.1f}")

    @staticmethod
    def _rolling_display_data(times, data, window_s):
        """Return the newest ``window_s`` of data on a display-local 0-based axis.

        Logic/hardware timestamps intentionally retain their absolute elapsed-since-
        stream-start meaning for saving. Only this GUI representation is rebased,
        matching the standard Time Series GUI and preventing traces from running out
        of a fixed x-axis range.
        """
        times = np.asarray(times, dtype=float)
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or times.size == 0:
            return np.empty(0, dtype=float), data
        n = min(times.size, data.shape[1])
        times = times[:n]
        data = data[:, :n]
        if n == 0:
            return times, data

        # Both FPGA and low-rate history clocks are monotonic. searchsorted avoids a
        # full boolean mask on every high-rate GUI frame.
        cutoff = times[-1] - max(0.0, float(window_s))
        first = int(np.searchsorted(times, cutoff, side='left'))
        times = times[first:]
        data = data[:, first:]
        if times.size:
            times = times - times[0]
        return times, data

    def _set_trace_x_range(self):
        """Keep both linked trace plots on the configured rolling 0..window axis."""
        if self._err_plot is None:
            return
        self._err_plot.setXRange(0.0, float(self._gui_trace_window_s),
                                 padding=0.0, update=True)

    def _set_curves(self, curves, times, data):
        """Update per-resonance curves on the fixed rolling display-time axis."""
        times, data = self._rolling_display_data(
            times, data, float(self._gui_trace_window_s))
        if data.ndim != 2 or times.size == 0:
            return
        for i in range(min(_N_RES, data.shape[0], len(curves))):
            curves[i].setData(x=times, y=data[i])

    @QtCore.Slot(object, object, object, object)
    def _on_high_rate_traces(self, times, err, corr_hz, cic):
        """Update all three quantities from one synchronized FPGA record stream."""
        self._got_high_rate = True
        self._set_curves(self._err_curves, times, err)
        self._set_curves(self._cic_curves, times, cic)
        self._set_curves(self._corr_curves, times, corr_hz)
        self._set_trace_x_range()

    @QtCore.Slot(object, object)
    def _on_corr_history(self, times, data):
        # low-rate register-poll fallback (only until the high-rate stream arrives)
        self._corr_hist = (np.asarray(times), np.asarray(data, dtype=float))
        if not self._got_high_rate:
            self._set_curves(self._corr_curves, *self._corr_hist)
            self._set_trace_x_range()

    @QtCore.Slot(object, object)
    def _on_err_history(self, times, data):
        self._err_hist = (np.asarray(times), np.asarray(data, dtype=float))
        if not self._got_high_rate:
            self._set_curves(self._err_curves, *self._err_hist)
            self._set_trace_x_range()

    # =========================================================================
    # View / settings
    # =========================================================================
    @QtCore.Slot()
    def restore_default_view(self):
        self._restore_parent_controls_to_right_sidebar()
        if self._control_dock is not None:
            self._control_dock.setFloating(False)
            self._mw.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._control_dock)
            self._control_dock.setMinimumWidth(_RIGHT_SIDEBAR_WIDTH)
            self._mw.splitDockWidget(
                self._scan_control_dockwidget,
                self._control_dock,
                QtCore.Qt.Vertical
            )
            self._control_dock.show()
        if self._trace_dock is not None:
            self._trace_dock.setFloating(False)
            self._mw.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self._trace_dock)
            self._trace_dock.show()
        self._resize_right_sidebar_docks()

    def _restore_parent_controls_to_right_sidebar(self):
        """Move inherited ODMR controls out of the top dock area.

        The tracking view needs vertical room for the ODMR scan and
        resonance time-trace plots. Keeping scan/CW controls in the right dock
        column preserves their behavior while preventing them from spanning the
        whole main window.
        """
        cw_available = getattr(self, '_OdmrGui__cw_control_available', True)

        self._scan_control_dockwidget.setFloating(False)
        self._scan_control_dockwidget.setMinimumWidth(_RIGHT_SIDEBAR_WIDTH)
        self._mw.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._scan_control_dockwidget)
        self._scan_control_dockwidget.show()

        self._mw.action_show_cw_controls.setChecked(True)
        self._cw_control_dockwidget.setFloating(False)
        self._cw_control_dockwidget.setMinimumWidth(_RIGHT_SIDEBAR_WIDTH)
        self._cw_control_dockwidget.setVisible(cw_available)
        self._mw.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._cw_control_dockwidget)
        if cw_available:
            self._mw.splitDockWidget(
                self._cw_control_dockwidget,
                self._scan_control_dockwidget,
                QtCore.Qt.Vertical
            )

        if self._fit_dockwidget is not None:
            self._fit_dockwidget.setFloating(False)
            self._fit_dockwidget.hide()

    def _resize_right_sidebar_docks(self):
        right_docks = [
            dock for dock in (
                self._cw_control_dockwidget,
                self._scan_control_dockwidget,
                self._control_dock,
            )
            if dock is not None and dock.isVisible()
        ]
        if right_docks:
            self._mw.resizeDocks(
                right_docks,
                [_RIGHT_SIDEBAR_WIDTH] * len(right_docks),
                QtCore.Qt.Horizontal
            )

    def _restore_settings(self):
        ranges = list(self._fit_ranges)
        fit_ranges = list(self._fit_sub_ranges)
        for i in range(_N_RES):
            lo, hi = ranges[i] if i < len(ranges) else (2.86e9 + i * 0.04e9, 2.88e9 + i * 0.04e9)
            self._res_min_spin[i].setValue(lo)
            self._res_max_spin[i].setValue(hi)
            self._fit_regions[i].blockSignals(True)
            self._fit_regions[i].setRegion([lo, hi])
            self._fit_regions[i].blockSignals(False)
            scan_width = hi - lo
            scan_center = (lo + hi) / 2.0
            self._last_scan_widths[i] = scan_width
            if i < len(fit_ranges):
                fit_lo, fit_hi = fit_ranges[i]
                fit_width = min(max(float(fit_hi) - float(fit_lo), 0.0), scan_width)
            else:
                fit_width = scan_width * _FIT_SUBWINDOW_FRACTION
            self._set_fit_sub_region(i, scan_center, fit_width, update_spins=True)
        self._fm_spin.setValue(self._gui_modulation_frequency)
        self._demod_phase_spin.blockSignals(True)
        self._demod_phase_spin.setValue(self._gui_demod_phase)
        self._demod_phase_spin.blockSignals(False)
        self._dwell_spin.setValue(self._gui_dwell_time)
        self._settle_spin.setValue(self._gui_settle_time)
        self._bw_spin.setValue(self._gui_bandwidth)
        self._maxcorr_spin.setValue(self._gui_max_correction)
        idx = self._controller_combo.findData(str(self._gui_controller))
        self._controller_combo.setCurrentIndex(max(idx, 0))
        self._zero_ratio_spin.setValue(float(self._gui_zero_ratio))
        self._zero_ratio_spin.setEnabled(self._controller_combo.currentData() != 'integral')
        self._detail_points_spin.setValue(int(self._gui_detail_points))
        self._apply_num_resonances(int(self._gui_num_resonances), emit=False)
        # Restore the high-rate trace window and push it to the hardware buffer.
        self._trace_window_spin.blockSignals(True)
        self._trace_window_spin.setValue(float(self._gui_trace_window_s))
        self._trace_window_spin.blockSignals(False)
        self.sigSetTraceWindow.emit(float(self._gui_trace_window_s))
