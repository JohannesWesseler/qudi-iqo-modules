# -*- coding: utf-8 -*-
"""
GUI for the automatic fiber coupling (FiberCouplingLogic): quick and fine alignment, the result
of the last run with its checks, power-meter calibration, motor calibration and history.

Example config:

    fiber_coupling_gui:
        module.Class: 'fiber_coupling.fiber_coupling_gui.FiberCouplingGui'
        connect:
            fiber_coupling_logic: fiber_coupling_logic

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

import datetime
import os

import numpy as np
import pyqtgraph as pg
from PySide2 import QtCore, QtWidgets

from qudi.core.module import GuiBase
from qudi.core.connector import Connector
from qudi.core.statusvariable import StatusVar
from qudi.util.colordefs import QudiPalettePale as palette
from qudi.util.widgets.scientific_spinbox import ScienDSpinBox

_COLORS = [palette.c1, palette.c2, palette.c3, palette.c4, palette.c5, palette.c6]
_OK, _WARN, _BAD = '#66bb6a', '#ffa726', '#ef5350'
_MOTOR_SCANS = ('up', 'down', 'after +', 'after -')


def _plot(title, xlabel, ylabel):
    w = pg.PlotWidget(title=title)
    w.setLabel('bottom', xlabel)
    w.setLabel('left', ylabel)
    w.showGrid(x=True, y=True, alpha=0.3)
    return w


def _when(iso):
    """'2026-10-19T21:46:53' -> '2026-10-19 21:46'"""
    return iso[:16].replace('T', ' ')


def _age(iso):
    days = (datetime.datetime.now() - datetime.datetime.fromisoformat(iso)).total_seconds() / 86400
    return 'today' if days < 1 else f'{days:.0f} days ago'


class FiberCouplingMainWindow(QtWidgets.QMainWindow):
    def __init__(self, motor_names, pair_names, unit):
        super().__init__()
        self.setWindowTitle('qudi: Fiber Coupling')
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        hbox = QtWidgets.QHBoxLayout(central)
        self.tabs = QtWidgets.QTabWidget()
        hbox.addWidget(self.tabs, 1)

        # --- tab: alignment -------------------------------------------------------------
        tab = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(tab)
        self.trace_plot = _plot(None, 'time (s)', f'coupling signal ({unit})')
        self.trace_curve = self.trace_plot.plot(pen=pg.mkPen(palette.c1, width=1.5))
        vbox.addWidget(self.trace_plot, 3)
        scans = QtWidgets.QHBoxLayout()
        vbox.addLayout(scans, 2)
        self.steep_plot = _plot('peak scans (steep axis)', 'steep offset (steps)',
                                f'signal ({unit})')
        self.flat_plot = _plot('beam walk (flat axis, fine alignment)', 'flat offset (steps)',
                               f'peak signal ({unit})')
        for p in (self.steep_plot, self.flat_plot):
            p.addLegend(offset=(5, 5))
            scans.addWidget(p)
        self.steep_curves, self.steep_fits, self.flat_curves, self.flat_fits = {}, {}, {}, {}
        for i, name in enumerate(pair_names):
            col = _COLORS[i % len(_COLORS)]
            dashed = pg.mkPen(col, width=1, style=QtCore.Qt.DashLine)
            self.steep_curves[name] = self.steep_plot.plot(
                pen=pg.mkPen(col, width=1), symbol='o', symbolSize=5, symbolBrush=col, name=name)
            self.steep_fits[name] = self.steep_plot.plot(pen=dashed)
            self.flat_curves[name] = self.flat_plot.plot(
                pen=pg.mkPen(col, width=1), symbol='o', symbolSize=6, symbolBrush=col, name=name)
            self.flat_fits[name] = self.flat_plot.plot(pen=dashed)
        self.tabs.addTab(tab, 'Alignment')

        # --- tab: calibration -------------------------------------------------------------
        tab = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(tab)
        self.motor_plots, self.motor_curves = [], []
        for i, name in enumerate(motor_names):
            p = _plot(name, f'{name} position rel. to start (steps)', f'signal ({unit})')
            p.addLegend(offset=(5, 5))
            curves = {lab: p.plot(pen=pg.mkPen(_COLORS[j], width=1), symbol='o', symbolSize=4,
                                  symbolBrush=_COLORS[j], name=lab)
                      for j, lab in enumerate(_MOTOR_SCANS)}
            grid.addWidget(p, i // 2, i % 2)
            self.motor_plots.append(p)
            self.motor_curves.append(curves)
        self.calibration_text = QtWidgets.QLabel('no calibration yet')
        self.calibration_text.setWordWrap(True)
        self.calibration_text.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        grid.addWidget(self.calibration_text, (len(motor_names) + 1) // 2, 0, 1, 2)
        self.tabs.addTab(tab, 'Calibration')

        # --- tab: history -------------------------------------------------------------------
        self.history_plot = pg.PlotWidget(axisItems={'bottom': pg.DateAxisItem()})
        self.history_plot.setLabel('left', f'signal after alignment ({unit})')
        self.history_plot.showGrid(x=True, y=True, alpha=0.3)
        self.history_plot.addLegend(offset=(5, 5))
        self.history_curves = {
            r: self.history_plot.plot(pen=None, symbol='o', symbolSize=7, symbolBrush=c, name=r)
            for r, c in (('quick', palette.c1), ('fine', palette.c2))}
        self.tabs.addTab(self.history_plot, 'History')

        # --- control panel --------------------------------------------------------------------
        side = QtWidgets.QWidget()
        side.setFixedWidth(375)      # labels wrap at this width; the scroll bar takes the rest
        panel = QtWidgets.QVBoxLayout(side)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setFixedWidth(400)
        hbox.addWidget(scroll)

        self.value_label = QtWidgets.QLabel('-')
        font = self.value_label.font()
        font.setPointSize(22)
        font.setBold(True)
        self.value_label.setFont(font)
        self.value_label.setAlignment(QtCore.Qt.AlignCenter)
        self.info_label = QtWidgets.QLabel('')
        self.info_label.setAlignment(QtCore.Qt.AlignCenter)
        self.info_label.setWordWrap(True)
        panel.addWidget(self.value_label)
        panel.addWidget(self.info_label)

        buttons = QtWidgets.QGridLayout()
        self.quick_button = QtWidgets.QPushButton('Quick align')
        self.quick_button.setToolTip('Peak scans of both motor pairs (~1 min). Use after drifts.')
        self.fine_button = QtWidgets.QPushButton('Fine align')
        self.fine_button.setToolTip('Quick align plus beam-walk scans of both pairs (~8 min). '
                                    'Use if quick align does not reach the usual value.')
        for b in (self.quick_button, self.fine_button):
            b.setMinimumHeight(48)
            f = b.font()
            f.setPointSize(f.pointSize() + 2)
            b.setFont(f)
        self.stop_button = QtWidgets.QPushButton('Stop')
        self.stop_button.setEnabled(False)
        buttons.addWidget(self.quick_button, 0, 0)
        buttons.addWidget(self.fine_button, 0, 1)
        buttons.addWidget(self.stop_button, 1, 0, 1, 2)
        panel.addLayout(buttons)
        self.progress_label = QtWidgets.QLabel('idle')
        panel.addWidget(self.progress_label)

        box = QtWidgets.QGroupBox('Last run')
        lay = QtWidgets.QVBoxLayout(box)
        self.result_label = QtWidgets.QLabel('-')
        self.result_label.setWordWrap(True)
        self.result_label.setTextFormat(QtCore.Qt.RichText)
        lay.addWidget(self.result_label)
        panel.addWidget(box)

        box = QtWidgets.QGroupBox('Power meter')
        form = QtWidgets.QFormLayout(box)
        hint = QtWidgets.QLabel(
            '1. Measure in front of the fiber splitter, reconnect.<br>'
            '2. Measure at the 1 % port (Nirvana reference fiber off), enter both and press '
            'Store: the coupling signal is read at that moment.')
        hint.setWordWrap(True)
        form.addRow(hint)
        self.p_input_spin = ScienDSpinBox()
        self.p_tap_spin = ScienDSpinBox()
        for s in (self.p_input_spin, self.p_tap_spin):
            s.setSuffix('W')
            s.setMinimum(0)
            s.setDecimals(3)
        form.addRow('in front of splitter', self.p_input_spin)
        form.addRow('at 1 % port', self.p_tap_spin)
        self.power_store_button = QtWidgets.QPushButton('Store')
        form.addRow(self.power_store_button)
        self.power_label = QtWidgets.QLabel('not calibrated')
        self.power_label.setWordWrap(True)
        form.addRow(self.power_label)
        panel.addWidget(box)

        box = QtWidgets.QGroupBox('Motor calibration')
        lay = QtWidgets.QVBoxLayout(box)
        self.calibration_label = QtWidgets.QLabel('-')
        self.calibration_label.setWordWrap(True)
        self.calibrate_button = QtWidgets.QPushButton('Calibrate (~5 min, weekly)')
        self.calibrate_button.setToolTip(
            'Scans every motor up, down and after excursions: backlash, lost steps and the peak '
            'width and direction of each pair. Updates the frame used by the alignment.')
        lay.addWidget(self.calibration_label)
        lay.addWidget(self.calibrate_button)
        panel.addWidget(box)

        self.manual_box = QtWidgets.QGroupBox('Manual control')
        self.manual_box.setCheckable(True)
        self.manual_box.setChecked(False)
        lay = QtWidgets.QVBoxLayout(self.manual_box)
        manual = QtWidgets.QWidget()
        lay.addWidget(manual)
        grid = QtWidgets.QGridLayout(manual)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(QtWidgets.QLabel('jog (no backlash handling)'), 0, 0, 1, 2)
        self.jog_step = QtWidgets.QSpinBox()
        self.jog_step.setRange(1, 2000)
        self.jog_step.setValue(10)
        self.jog_step.setSuffix(' steps')
        grid.addWidget(self.jog_step, 0, 2, 1, 2)
        self.pos_labels, self.jog_buttons = [], []
        for i, name in enumerate(motor_names):
            grid.addWidget(QtWidgets.QLabel(name), i + 1, 0)
            lab = QtWidgets.QLabel('-')
            lab.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.pos_labels.append(lab)
            grid.addWidget(lab, i + 1, 1)
            minus, plus = QtWidgets.QPushButton('-'), QtWidgets.QPushButton('+')
            grid.addWidget(minus, i + 1, 2)
            grid.addWidget(plus, i + 1, 3)
            self.jog_buttons.append((minus, plus))
        row = len(motor_names) + 1
        self.live_check = QtWidgets.QCheckBox('live signal when idle')
        self.reset_button = QtWidgets.QPushButton('Reset best value')
        self.reset_button.setToolTip('Use after a change of laser power')
        grid.addWidget(self.live_check, row, 0, 1, 2)
        grid.addWidget(self.reset_button, row, 2, 1, 2)
        self.manual_box.toggled.connect(manual.setVisible)
        manual.setVisible(False)
        panel.addWidget(self.manual_box)
        panel.addStretch(1)
        self.resize(1350, 900)


class FiberCouplingGui(GuiBase):
    """GUI for FiberCouplingLogic."""

    _logic = Connector(name='fiber_coupling_logic', interface='FiberCouplingLogic')
    _geometry = StatusVar('window_geometry', default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mw = None
        self._unit = ''

    def on_activate(self):
        logic = self._logic()
        self._unit = logic.metric_unit
        self._mw = mw = FiberCouplingMainWindow(logic.motor_names,
                                                [p['name'] for p in logic.pairs], self._unit)
        mw.live_check.setChecked(logic.live_trace)

        mw.quick_button.clicked.connect(lambda: logic.start_routine('quick'))
        mw.fine_button.clicked.connect(lambda: logic.start_routine('fine'))
        mw.calibrate_button.clicked.connect(lambda: logic.start_routine('calibrate'))
        mw.stop_button.clicked.connect(logic.stop_routine)
        mw.power_store_button.clicked.connect(
            lambda: logic.store_power_calibration(mw.p_input_spin.value(), mw.p_tap_spin.value()))
        for i, (minus, plus) in enumerate(mw.jog_buttons):
            minus.clicked.connect(lambda _=False, i=i: logic.jog(i, -mw.jog_step.value()))
            plus.clicked.connect(lambda _=False, i=i: logic.jog(i, mw.jog_step.value()))
        mw.live_check.toggled.connect(lambda on: setattr(logic, 'live_trace', on))
        mw.reset_button.clicked.connect(logic.reset_reference)

        q = QtCore.Qt.QueuedConnection
        logic.sigTraceUpdated.connect(self._update_trace, q)
        logic.sigScanUpdated.connect(self._update_scan, q)
        logic.sigPositionsUpdated.connect(self._update_positions, q)
        logic.sigBusyChanged.connect(self._update_busy, q)
        logic.sigProgress.connect(mw.progress_label.setText, q)
        logic.sigRoutineFinished.connect(self._update_result, q)
        logic.sigCalibrationChanged.connect(self._update_calibration, q)
        logic.sigPowerCalibrationChanged.connect(self._update_power_calibration, q)
        logic.sigHistoryChanged.connect(self._update_history, q)

        self._update_positions(logic.positions())
        self._update_busy(logic.busy, '')
        self._update_result(logic.last_result)
        self._update_calibration(logic.calibration)
        self._update_power_calibration(logic.power_calibration)
        self._update_history(logic.history)
        if self._geometry:
            try:
                mw.restoreGeometry(QtCore.QByteArray.fromHex(self._geometry.encode()))
            except Exception:
                pass
        self.show()

    def on_deactivate(self):
        logic = self._logic()
        for sig in (logic.sigTraceUpdated, logic.sigScanUpdated, logic.sigPositionsUpdated,
                    logic.sigBusyChanged, logic.sigProgress, logic.sigRoutineFinished,
                    logic.sigCalibrationChanged, logic.sigPowerCalibrationChanged,
                    logic.sigHistoryChanged):
            try:
                sig.disconnect()
            except (RuntimeError, TypeError):
                pass
        self._geometry = bytes(self._mw.saveGeometry().toHex()).decode()
        self._mw.close()
        self._mw = None

    def show(self):
        self._mw.show()
        self._mw.raise_()
        self._mw.activateWindow()

    # --- updates ------------------------------------------------------------------------------
    @QtCore.Slot(object, object)
    def _update_trace(self, t, m):
        mw = self._mw
        if mw is None or len(t) == 0:
            return
        mw.trace_curve.setData(t, m)
        logic = self._logic()
        value = float(m[-1])
        mw.value_label.setText(f'{value:.4g} {self._unit}')
        parts = []
        ref = logic.reference_value
        if ref:
            parts.append(f'{value / ref:.0%} of best')
        est = logic.estimate_power(value)
        if est:
            parts.append(f'~{est[0] * 1e3:.3g} mW at NVs, {est[1]:.0%} coupled')
        mw.info_label.setText('  |  '.join(parts))
        low = ref is not None and value < 0.9 * ref
        mw.info_label.setStyleSheet(f'color: {_WARN}' if low else '')

    @QtCore.Slot(dict)
    def _update_scan(self, d):
        mw = self._mw
        if mw is None:
            return
        x = np.asarray(d['x'], float)
        y = np.asarray(d['y'], float)
        if d['kind'] == 'motor':
            label = d['label'] if d['label'] in ('up', 'down') else d['label'][:7]
            curves = mw.motor_curves[d['motor']]
            if label in curves:
                curves[label].setData(x, y)
            return
        name = d['pair']
        curve, fit_curve = ((mw.steep_curves[name], mw.steep_fits[name]) if d['kind'] == 'steep'
                            else (mw.flat_curves[name], mw.flat_fits[name]))
        curve.setData(x, y)
        if not d.get('done'):
            fit_curve.setData([], [])
            return
        xf = np.linspace(x.min(), x.max(), 300)
        if d.get('fit'):
            f = d['fit']
            g = np.exp(-0.5 * ((xf - f['center']) / f['sigma']) ** 2)
            fit_curve.setData(xf, f['amplitude'] * g + f['offset'])
        elif d.get('parabola'):
            c = d['parabola']
            fit_curve.setData(xf, np.polyval([c['c2'], c['c1'], c['c0']], xf))

    @QtCore.Slot(object)
    def _update_positions(self, pos):
        if self._mw is None:
            return
        for lab, p in zip(self._mw.pos_labels, pos):
            lab.setText(f'{int(p):d}')

    @QtCore.Slot(bool, str)
    def _update_busy(self, busy, routine):
        mw = self._mw
        if mw is None:
            return
        for w in [mw.quick_button, mw.fine_button, mw.calibrate_button, mw.power_store_button,
                  mw.reset_button] + [b for pair in mw.jog_buttons for b in pair]:
            w.setEnabled(not busy)
        mw.stop_button.setEnabled(busy)
        if busy:
            mw.progress_label.setText(f'running: {routine}')
            if routine == 'calibrate':
                mw.tabs.setCurrentIndex(1)
                for curves in mw.motor_curves:
                    for c in curves.values():
                        c.setData([], [])
            else:
                mw.tabs.setCurrentIndex(0)
                if routine == 'fine':
                    for c in list(mw.flat_curves.values()) + list(mw.flat_fits.values()):
                        c.setData([], [])
        else:
            mw.progress_label.setText('idle')

    @QtCore.Slot(object)
    def _update_result(self, r):
        mw = self._mw
        if mw is None or not r:
            return
        color = {'ok': _OK, 'warning': _WARN}.get(r.get('verdict'), _BAD)
        head = r.get('verdict', 'failed').upper()
        lines = [f'<b style="color:{color}">{head}</b> &nbsp; {r["routine"]}, {_when(r["time"])} '
                 f'({r["duration"]:.0f} s)']
        if 'final' in r:
            lines.append(f'{r["start"]:.4g} &rarr; <b>{r["final"]:.4g} {self._unit}</b>')
        else:
            lines.append(r.get('message', ''))
        for c in r.get('checks', []):
            sym = f'<span style="color:{_OK}">&#10003;</span>' if c['ok'] else \
                f'<span style="color:{_WARN}">&#9888;</span>'
            lines.append(f'{sym} {c["text"]}')
        if r.get('file'):
            lines.append(f'<small>saved: {os.path.basename(r["file"])}</small>')
        mw.result_label.setText('<br>'.join(lines))

    @QtCore.Slot(object)
    def _update_calibration(self, cal):
        mw = self._mw
        if mw is None:
            return
        due = self._logic().calibration_due
        if not cal:
            mw.calibration_label.setText('Not calibrated yet: frame from the config file.')
        else:
            t = cal['time']
            mw.calibration_label.setText(f'Last calibration {_when(t)} ({_age(t)})'
                                         + (': due' if due else ''))
        mw.calibration_label.setStyleSheet(f'color: {_WARN}' if due else '')
        if not cal:
            return
        rows = ['<table cellspacing="6"><tr><th align="left">motor</th><th>width (steps)</th>'
                '<th>backlash (steps)</th><th>returns within (steps)</th></tr>']
        for name, m in cal.get('motors', {}).items():
            fmt = lambda v, f='.1f': '-' if v is None else format(v, f)
            rows.append(f'<tr><td>{name}</td><td align="right">{fmt(m["sigma"])}</td>'
                        f'<td align="right">{fmt(m["backlash"], ".0f")}</td>'
                        f'<td align="right">{fmt(m["lost_steps"])}</td></tr>')
        rows.append('</table><br><table cellspacing="6"><tr><th align="left">pair</th>'
                    '<th>steep sigma (steps)</th><th>steep direction</th></tr>')
        for name, p in cal.get('pairs', {}).items():
            rows.append(f'<tr><td>{name}</td><td align="right">{p["sigma_steep"]:.1f}</td>'
                        f'<td align="right">[{p["steep"][0]:+.3f}, {p["steep"][1]:+.3f}]</td></tr>')
        rows.append('</table>')
        mw.calibration_text.setText(''.join(rows))

    @QtCore.Slot(object)
    def _update_power_calibration(self, c):
        mw = self._mw
        if mw is None or not c:
            return
        mw.p_input_spin.setValue(c['power_input'])
        mw.p_tap_spin.setValue(c['power_tap'])
        mw.power_label.setText(
            f'{_when(c["time"])}: {c["efficiency"]:.1%} coupled, {c["power_main"] * 1e3:.3g} mW '
            f'at the main port, signal {c["metric"]:.4g} {c["unit"]} '
            f'(splitter {c["splitter_ratio"]:g}:1)')

    @QtCore.Slot(object)
    def _update_history(self, history):
        mw = self._mw
        if mw is None:
            return
        for routine, curve in mw.history_curves.items():
            h = [e for e in history if e['routine'] == routine]
            t = [datetime.datetime.fromisoformat(e['time']).timestamp() for e in h]
            curve.setData(t, [e['final'] for e in h])
        for routine, button, text in (('quick', mw.quick_button, 'Quick align'),
                                      ('fine', mw.fine_button, 'Fine align')):
            d = [e['duration'] for e in history if e['routine'] == routine][-5:]
            button.setText(f'{text}\n(~{np.median(d) / 60:.0f} min)' if d and np.median(d) > 90
                           else f'{text}\n(~{np.median(d):.0f} s)' if d else text)
