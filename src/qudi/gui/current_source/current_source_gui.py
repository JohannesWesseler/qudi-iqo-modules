# -*- coding: utf-8 -*-
"""
GUI for a waveform current source (CurrentSourceLogic): supply on/off, waveform, frequency,
amplitude and offset in A, bursts and sync marker, live scope of current and output voltage,
history of the current readings and supply power, aborts.

For the Red Pitaya + LM1875 source this replaces the Red Pitaya web oscilloscope: the web
applications and the SCPI server must not run at the same time (both drive the FPGA).

Example config:

    current_source_gui:
        module.Class: 'current_source.current_source_gui.CurrentSourceGui'
        connect:
            source_logic: current_source_logic

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

import numpy as np
import pyqtgraph as pg
from PySide2 import QtCore, QtWidgets

from qudi.core.module import GuiBase
from qudi.core.connector import Connector
from qudi.util.colordefs import QudiPalettePale as palette
from qudi.util.widgets.scientific_spinbox import ScienDSpinBox

_OK, _WARN, _BAD = '#66bb6a', '#ffa726', '#ef5350'
_WAVEFORMS = ('sine', 'dc', 'square', 'triangle', 'arbitrary')


def _plot(xlabel, ylabel):
    w = pg.PlotWidget()
    w.setLabel('bottom', xlabel)
    w.setLabel('left', ylabel)
    w.showGrid(x=True, y=True, alpha=0.3)
    return w


class CurrentSourceMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('qudi: Current source')
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # ---------------- controls ----------------
        ctrl = QtWidgets.QVBoxLayout()
        layout.addLayout(ctrl, 0)

        power = QtWidgets.QGroupBox('Supply')
        pl = QtWidgets.QGridLayout(power)
        self.power_on_button = QtWidgets.QPushButton('Power on')
        self.power_off_button = QtWidgets.QPushButton('Power off')
        self.power_label = QtWidgets.QLabel('off')
        self.supply_label = QtWidgets.QLabel('')
        pl.addWidget(self.power_on_button, 0, 0)
        pl.addWidget(self.power_off_button, 0, 1)
        pl.addWidget(self.power_label, 1, 0, 1, 2)
        pl.addWidget(self.supply_label, 2, 0, 1, 2)
        ctrl.addWidget(power)

        out = QtWidgets.QGroupBox('Output')
        ol = QtWidgets.QFormLayout(out)
        self.waveform_combo = QtWidgets.QComboBox()
        self.waveform_combo.addItems(_WAVEFORMS)
        self.frequency_spin = ScienDSpinBox()
        self.frequency_spin.setSuffix('Hz')
        self.frequency_spin.setRange(0.1, 1e6)
        self.frequency_spin.setValue(1e3)
        self.amplitude_spin = ScienDSpinBox()
        self.amplitude_spin.setSuffix('A')
        self.amplitude_spin.setRange(0, 2)
        self.amplitude_spin.setValue(0.1)
        self.offset_spin = ScienDSpinBox()
        self.offset_spin.setSuffix('A')
        self.offset_spin.setRange(-2, 2)
        self.offset_spin.setValue(0)
        self.trim_check = QtWidgets.QCheckBox('trim to measured current (sine)')
        self.max_label = QtWidgets.QLabel('')
        self.apply_button = QtWidgets.QPushButton('Apply')
        self.off_button = QtWidgets.QPushButton('Output off')
        ol.addRow('waveform', self.waveform_combo)
        ol.addRow('frequency', self.frequency_spin)
        ol.addRow('amplitude (peak)', self.amplitude_spin)
        ol.addRow('offset', self.offset_spin)
        ol.addRow(self.trim_check)
        ol.addRow('max. amplitude', self.max_label)
        bl = QtWidgets.QHBoxLayout()
        bl.addWidget(self.apply_button)
        bl.addWidget(self.off_button)
        ol.addRow(bl)
        ctrl.addWidget(out)

        burst = QtWidgets.QGroupBox('Burst and sync marker')
        bfl = QtWidgets.QFormLayout(burst)
        self.burst_check = QtWidgets.QCheckBox('burst mode')
        self.cycles_spin = QtWidgets.QSpinBox()
        self.cycles_spin.setRange(1, 50000)
        self.cycles_spin.setValue(10)
        self.reps_spin = QtWidgets.QSpinBox()
        self.reps_spin.setRange(0, 50000)
        self.reps_spin.setValue(1)
        self.reps_spin.setSpecialValueText('endless')
        self.period_spin = ScienDSpinBox()
        self.period_spin.setSuffix('s')
        self.period_spin.setRange(1e-6, 100)
        self.period_spin.setValue(0.01)
        self.burst_apply = QtWidgets.QPushButton('Set burst')
        self.sync_check = QtWidgets.QCheckBox('sync marker out (DIO0_N)')
        self.trigger_button = QtWidgets.QPushButton('Trigger')
        bfl.addRow(self.burst_check)
        bfl.addRow('cycles', self.cycles_spin)
        bfl.addRow('repetitions', self.reps_spin)
        bfl.addRow('period', self.period_spin)
        bfl.addRow(self.burst_apply)
        bfl.addRow(self.sync_check)
        bfl.addRow(self.trigger_button)
        ctrl.addWidget(burst)

        misc = QtWidgets.QGroupBox('Monitor')
        ml = QtWidgets.QFormLayout(misc)
        self.live_check = QtWidgets.QCheckBox('live readings and scope')
        self.scope_dec_spin = QtWidgets.QSpinBox()
        self.scope_dec_spin.setRange(0, 65536)
        self.scope_dec_spin.setSpecialValueText('auto')
        self.save_button = QtWidgets.QPushButton('Save history + scope')
        ml.addRow(self.live_check)
        ml.addRow('scope decimation', self.scope_dec_spin)
        ml.addRow(self.save_button)
        ctrl.addWidget(misc)
        ctrl.addStretch(1)

        # ---------------- plots and readings ----------------
        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 1)
        self.reading_label = QtWidgets.QLabel('')
        self.reading_label.setStyleSheet('font-size: 13pt')
        right.addWidget(self.reading_label)
        self.scope_plot = _plot('time (ms)', 'current, shunt (A)')
        self.scope_curve = self.scope_plot.plot(pen=pg.mkPen(palette.c1, width=1))
        self.scope_out_plot = _plot('time (ms)', 'amplifier output (V)')
        self.scope_out_curve = self.scope_out_plot.plot(pen=pg.mkPen(palette.c2, width=1))
        self.scope_out_plot.setXLink(self.scope_plot)
        self.history_plot = _plot('time (s)', 'current (A)')
        self.history_plot.addLegend()
        self.hist_amp = self.history_plot.plot(pen=pg.mkPen(palette.c1, width=1.5),
                                               name='amplitude (shunt)')
        self.hist_amp_out = self.history_plot.plot(pen=pg.mkPen(palette.c2, width=1.5),
                                                   name='amplitude (output / Z)')
        self.hist_mean = self.history_plot.plot(pen=pg.mkPen(palette.c3, width=1.5),
                                                name='mean (shunt)')
        self.power_plot = _plot('time (s)', 'supply power (W)')
        self.power_curve = self.power_plot.plot(pen=pg.mkPen(palette.c4, width=1.5))
        self.power_plot.setXLink(self.history_plot)
        right.addWidget(self.scope_plot, 2)
        right.addWidget(self.scope_out_plot, 1)
        right.addWidget(self.history_plot, 1)
        right.addWidget(self.power_plot, 1)
        self.statusBar().showMessage('')
        self.resize(1300, 900)


class CurrentSourceGui(GuiBase):
    """Current source control window."""

    _logic = Connector(name='source_logic', interface='CurrentSourceLogic')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mw = None

    def on_activate(self):
        self._mw = mw = CurrentSourceMainWindow()
        logic = self._logic()
        mw.power_on_button.clicked.connect(lambda: logic.request('power_on'))
        mw.power_off_button.clicked.connect(lambda: logic.request('power_off'))
        mw.apply_button.clicked.connect(self._apply)
        mw.off_button.clicked.connect(lambda: logic.request('output_off'))
        mw.burst_apply.clicked.connect(self._apply_burst)
        mw.sync_check.toggled.connect(lambda on: logic.request('set_sync_output', enabled=on))
        mw.trigger_button.clicked.connect(lambda: logic.request('trigger'))
        mw.live_check.setChecked(logic.live_monitor)
        mw.live_check.toggled.connect(lambda on: setattr(logic, 'live_monitor', on))
        mw.scope_dec_spin.setValue(logic.scope_decimation)
        mw.scope_dec_spin.valueChanged.connect(lambda v: setattr(logic, 'scope_decimation', v))
        mw.save_button.clicked.connect(lambda: logic.request('save'))
        mw.waveform_combo.currentIndexChanged.connect(self._update_max)
        mw.frequency_spin.valueChanged.connect(self._update_max)
        q = QtCore.Qt.QueuedConnection
        logic.sigMeasurement.connect(self._update_reading, q)
        logic.sigHistory.connect(self._update_history, q)
        logic.sigScope.connect(self._update_scope, q)
        logic.sigStateChanged.connect(self._update_state, q)
        logic.sigError.connect(self._show_error, q)
        logic.sigAbort.connect(self._show_abort, q)
        self._update_state(logic.output_state)
        self._update_max()
        self.show()

    def on_deactivate(self):
        logic = self._logic()
        for sig in (logic.sigMeasurement, logic.sigHistory, logic.sigScope, logic.sigStateChanged,
                    logic.sigError, logic.sigAbort):
            try:
                sig.disconnect()
            except (RuntimeError, TypeError):
                pass
        self._mw.close()
        self._mw = None

    def show(self):
        self._mw.show()
        self._mw.activateWindow()
        self._mw.raise_()

    # --- actions ---------------------------------------------------------------------------
    def _apply(self):
        mw = self._mw
        self._logic().request('set_output', waveform=mw.waveform_combo.currentText(),
                              frequency=mw.frequency_spin.value(),
                              amplitude=mw.amplitude_spin.value(),
                              offset=mw.offset_spin.value(),
                              trim=mw.trim_check.isChecked())

    def _apply_burst(self):
        mw = self._mw
        if mw.burst_check.isChecked():
            self._logic().request('configure_burst', n_cycles=mw.cycles_spin.value(),
                                  n_repetitions=mw.reps_spin.value(),
                                  period=mw.period_spin.value())
        else:
            self._logic().request('configure_burst', n_cycles=None)

    # --- updates ---------------------------------------------------------------------------
    def _update_max(self, *_):
        mw = self._mw
        try:
            i = self._logic().max_current(mw.waveform_combo.currentText(), mw.frequency_spin.value())
        except Exception:
            return
        mw.max_label.setText(f'{i:.3f} A')

    def _update_state(self, st):
        mw = self._mw
        if mw is None:
            return
        powered = st.get('powered', False)
        mw.power_label.setText(('<b>ON</b>' if powered else 'off') + (
            f"  | output {'on' if st['on'] else 'off'}: {st['waveform']}"
            + (f" {st['frequency']:.6g} Hz" if st['waveform'] != 'dc' else '')
            + f", {st['amplitude']:.4g} A pk, offset {st['offset']:+.4g} A"
            + (f", burst {st['burst']}" if st.get('burst') else '')))
        mw.power_label.setStyleSheet(f'color: {_OK if powered else "#999999"}')

    def _update_reading(self, m):
        mw = self._mw
        if mw is None:
            return
        if 'i_amp' in m:
            txt = (f"I = {m['i_amp']:.4f} A pk (shunt), {m['i_amp_output']:.4f} A (output / Z), "
                   f"set {m['i_amp_command']:.4f} A\n"
                   f"phase I-V {np.degrees(m['phase_i_minus_v']):+.1f}°, "
                   f"|Z| {np.hypot(m['z_re'], m['z_im']):.3f} Ω, THD {m['i_thd'] * 100:.2f} %")
        else:
            txt = (f"I mean {m['i_mean']:+.4f} A (shunt), {m['i_mean_output']:+.4f} A (output / R), "
                   f"set {m['i_mean_command']:+.4f} A\n"
                   f"rms {m['i_rms']:.4f} A, peak {m['i_peak']:.4f} A")
        mw.reading_label.setText(txt)
        sup = m.get('supply') or {}
        if sup:
            mw.supply_label.setText(f"+{sup.get('v1', 0):.2f} V {sup.get('i1', 0) * 1e3:.0f} mA | "
                                    f"-{sup.get('v4', 0):.2f} V {sup.get('i4', 0) * 1e3:.0f} mA | "
                                    f"{m.get('supply_power', float('nan')):.2f} W")

    def _update_history(self, h):
        mw = self._mw
        if mw is None or not len(h['t']):
            return
        mw.hist_amp.setData(h['t'], np.nan_to_num(h['i_amp'], nan=0.0))
        mw.hist_amp_out.setData(h['t'], np.nan_to_num(h['i_amp_output'], nan=0.0))
        mw.hist_mean.setData(h['t'], h['i_mean'])
        mw.power_curve.setData(h['t'], np.nan_to_num(h['supply_power'], nan=0.0))

    def _update_scope(self, c):
        mw = self._mw
        if mw is None:
            return
        mw.scope_curve.setData(c['t'] * 1e3, c['current'])
        mw.scope_out_curve.setData(c['t'] * 1e3, c['output'])

    def _show_error(self, text):
        self._mw.statusBar().setStyleSheet(f'color: {_WARN}')
        self._mw.statusBar().showMessage(text, 20000)

    def _show_abort(self, reason, supply_off):
        self._mw.statusBar().setStyleSheet(f'color: {_BAD}')
        self._mw.statusBar().showMessage('ABORT: ' + reason + (' (supply off)' if supply_off else ''))
