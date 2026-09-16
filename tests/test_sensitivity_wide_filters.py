"""Headless checks of wide-filter selection from GUI to sensitivity logic."""
import copy
import os
import unittest
import tempfile
import threading
from types import SimpleNamespace, MethodType
from unittest.mock import Mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QT_API', 'pyside2')
from PySide2 import QtWidgets, QtCore
from qudi.gui.sensitivity_sweep.sensitivity_sweep_gui import SensitivitySweepGui
from qudi.logic.sensitivity_sweep_logic import SensitivitySweepLogic
from qudi.hardware.redpitaya.redpitaya_odmr_lock import RedPitayaOdmrLockHardware
from pyrpl.hardware_modules.lock_in import _filter_select_options


class TestWideFilters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_old_firmware_does_not_offer_new_register_values(self):
        old = SimpleNamespace(filter_profile_id=0, filter_capabilities={})
        self.assertNotIn('6kHz_minphase', _filter_select_options(old, 6))
        new = SimpleNamespace(filter_profile_id=0, filter_capabilities={
            '6kHz_minphase': True, '10kHz_minphase': True})
        self.assertEqual(_filter_select_options(new, 6)['6kHz_minphase'], 2 << 6)
        self.assertEqual(_filter_select_options(new, 8)['10kHz_minphase'], 3 << 8)
        fast = SimpleNamespace(filter_profile_id=1)
        self.assertEqual(list(_filter_select_options(fast, 6)), ['20kHz_linear'])
        profile2048 = SimpleNamespace(filter_profile_id=2)
        self.assertEqual(dict(_filter_select_options(profile2048, 6)),
                         {'10kHz_minphase_2048': 3 << 6})

    def test_gui_emits_each_filter_and_logic_applies_it(self):
        for name in ('2kHz_minphase', '2kHz_linear', '20kHz_linear',
                     '6kHz_minphase', '10kHz_minphase', '10kHz_minphase_2048'):
            with self.subTest(filter=name):
                gui = SensitivitySweepGui(
                    qudi_main_weakref=lambda: None, name='wide_filter_test', config={})
                for attr, status in gui._meta['status_variables'].items():
                    setattr(gui, attr, copy.deepcopy(status.default))
                gui._fir_filter_bandwidth = name
                gui._measure_smith = False
                gui._mw = QtWidgets.QMainWindow()
                gui._results_table = QtWidgets.QTableWidget()
                gui._create_config_dock()
                gui._stream_f_enbw_spinbox.setValue(10418.6)
                self.assertAlmostEqual(gui._stream_f_enbw_spinbox.value(), 10418.6)
                emitted = []
                gui.sigStartSweep.connect(lambda sweep, odmr, stream: emitted.append(stream))
                gui._on_start_sweep()
                self.assertEqual(len(emitted), 1)
                self.assertEqual(emitted[0]['fir_filter_bandwidth'], name)
                self.assertNotIn('closed_loop_smith', emitted[0]['tracking_modes'])

                lockin = SimpleNamespace(filter_capabilities={name: True},
                                         smith_filter_name='2kHz_linear')
                streamer = SimpleNamespace(_pyrpl=SimpleNamespace(
                    rp=SimpleNamespace(lockin=lockin)))
                logic = Mock()
                logic._stream_parameters = emitted[0]
                logic._time_series_logic.return_value._streamer.return_value = streamer
                SensitivitySweepLogic._configure_lock_in_filters(logic)
                self.assertEqual(lockin.filter_select_ch1, name)
                self.assertFalse(lockin.fir_bypass_ch1)
                self.assertEqual(logic._stream_parameters['fir_filter_bandwidth'], name)
                gui._mw.close()
                gui._results_table.close()

    def test_exact_fm_frequencies_for_2048_profile(self):
        gui = SensitivitySweepGui(qudi_main_weakref=lambda: None, name='fm_test', config={})
        for attr, status in gui._meta['status_variables'].items():
            setattr(gui, attr, copy.deepcopy(status.default))
        gui._fir_filter_bandwidth = '10kHz_minphase_2048'
        gui._mw = QtWidgets.QMainWindow()
        gui._results_table = QtWidgets.QTableWidget()
        gui._create_config_dock()
        self.addCleanup(gui._mw.close)
        self.addCleanup(gui._results_table.close)
        gui._f_mod_values_edit.setText('15.26, 20, 30')
        emitted = []
        gui.sigStartSweep.connect(lambda sweep, odmr, stream: emitted.append((sweep, stream)))
        gui._on_start_sweep()
        self.assertEqual(list(emitted[0][0]['f_mod']), [15260., 20000., 30000.])
        self.assertNotIn('closed_loop_smith', emitted[0][1]['tracking_modes'])
        self.assertFalse(gui._measure_smith_checkbox.isEnabled())
        self.assertEqual(gui._f_mod_values_khz, '15.26, 20, 30')
        gui._f_mod_values_edit.setText('nan')
        with self.assertRaises(ValueError):
            gui._f_mod_array_from_widgets()

    def test_2048_preflight_requires_matching_image_and_no_smith(self):
        logic = Mock()
        logic._thread_lock = threading.RLock()
        logic._sweep_state = 'idle'
        logic._sweep_loop_order = ['power', 'f_mod', 'f_dev']
        logic._generate_parameter_combinations = MethodType(
            SensitivitySweepLogic._generate_parameter_combinations, logic)
        lockin = logic._time_series_logic.return_value._streamer.return_value._pyrpl.rp.lockin
        sweep = {'power': [-10], 'f_mod': [15260, 20000, 30000], 'f_dev': [100]}
        stream = {'fir_filter_bandwidth': '10kHz_minphase_2048', 'tracking_modes': ['open_loop']}
        lockin.filter_capabilities = {'10kHz_minphase': True}
        with self.assertRaises(ValueError):
            SensitivitySweepLogic.configure_sweep(logic, sweep, {}, stream)
        lockin.filter_capabilities = {'10kHz_minphase_2048': True}
        with self.assertRaises(ValueError):
            SensitivitySweepLogic.configure_sweep(
                logic, sweep, {}, dict(stream, tracking_modes=['closed_loop_smith']))
        SensitivitySweepLogic.configure_sweep(logic, sweep, {}, stream)
        self.assertEqual(logic._total_combinations, 3)
        self.assertAlmostEqual(logic._stream_parameters['f_enbw'], 10640.924904046242)

    def test_trace_calibration_follows_live_filter_selection(self):
        hw = SimpleNamespace(
            _pyrpl=SimpleNamespace(rp=SimpleNamespace(
                lockin=SimpleNamespace(filter_select_ch1='6kHz_minphase'),
                lockin1=SimpleNamespace(filter_select_ch1='10kHz_minphase'))),
            _configured_lock_in_filters=['2kHz_minphase', '2kHz_minphase'],
            _stream_words_per_sample=4, _nslots=2)
        result = RedPitayaOdmrLockHardware.get_trace_calibration(hw)
        self.assertEqual(result['filters'], ('6kHz_minphase', '10kHz_minphase'))
        self.assertEqual(result['fir_dc_gain_from_cic_lsb'], (7326 / 2**14,) * 2)

    def test_gui_filter_matrix_and_separate_bandwidths(self):
        gui = SensitivitySweepGui(qudi_main_weakref=lambda: None, name='matrix_test', config={})
        for attr, status in gui._meta['status_variables'].items():
            setattr(gui, attr, copy.deepcopy(status.default))
        gui._mw = QtWidgets.QMainWindow()
        gui._results_table = QtWidgets.QTableWidget()
        gui._create_config_dock()
        self.addCleanup(gui._mw.close)
        self.addCleanup(gui._results_table.close)
        gui._measure_open_loop_checkbox.setChecked(True)
        gui._measure_conventional_checkbox.setChecked(True)
        gui._measure_smith_checkbox.setChecked(True)
        gui._fir_bypass_checkbox.setChecked(True)
        gui._compare_filters_checkbox.setChecked(True)
        gui._bandwidths_6k_edit.setText('100, 200')
        gui._bandwidths_10k_edit.setText('300')
        gui._controller_bandwidths_edit.setText('ignored')
        gui._off_resonant_checkbox.setChecked(False)
        self.assertFalse(gui._measure_smith_checkbox.isChecked())
        self.assertFalse(gui._measure_smith_checkbox.isEnabled())
        self.assertFalse(gui._fir_bypass_checkbox.isChecked())
        emitted = []
        gui.sigStartSweep.connect(lambda sweep, odmr, stream: emitted.append(stream))
        gui._on_start_sweep()
        profiles = emitted[0]['filter_sweep']
        self.assertEqual([p['fir_filter_bandwidth'] for p in profiles],
                         ['6kHz_minphase', '10kHz_minphase'])
        self.assertEqual([p['controller_bandwidths_hz'] for p in profiles], [[100., 200.], [300.]])
        self.assertEqual(emitted[0]['tracking_modes'], ['open_loop', 'closed_loop_conventional'])
        self.assertTrue(gui._compare_wide_filters)
        gui._bandwidths_6k_edit.setText('nan')
        with self.assertRaises(ValueError):
            gui._filter_sweep_from_widgets()

    def make_matrix_logic(self, folder):
        logic = Mock()
        logic._current_folder = folder
        logic._pause_requested = logic._cancel_requested = False
        logic._sweep_loop_order = ['power', 'f_mod', 'f_dev']
        logic._parameter_combinations = [(-10, 15000, 100), (-10, 15000, 200),
                                         (-5, 15000, 100), (-5, 15000, 200)]
        logic._total_combinations = 4
        logic._thermal_stabilization_time = 180
        logic._odmr_parameters = {'frequency_start': 2.8e9, 'frequency_stop': 2.9e9}
        logic._stream_parameters = {'filter_sweep': [
            {'fir_filter_bandwidth': '6kHz_minphase', 'controller_bandwidths_hz': [100, 200], 'f_enbw': 6392.1428},
            {'fir_filter_bandwidth': '10kHz_minphase', 'controller_bandwidths_hz': [300], 'f_enbw': 10418.5833}]}
        logic._measure_filter_point = MethodType(SensitivitySweepLogic._measure_filter_point, logic)
        logic._fit_odmr_data.return_value = None
        logic._run_odmr_scan.return_value = ([1, 2], [3, 4])
        return logic

    def test_power_outermost_wait_once_and_fresh_scan_per_filter(self):
        with tempfile.TemporaryDirectory() as folder:
            logic = self.make_matrix_logic(folder)
            original = logic._stream_parameters
            applied = []
            logic._configure_lock_in_filters.side_effect = lambda: applied.append(
                (logic._stream_parameters['fir_filter_bandwidth'],
                 logic._stream_parameters['controller_bandwidths_hz']))
            results = []
            for idx, combination in enumerate(logic._parameter_combinations):
                results.extend(SensitivitySweepLogic._measure_single_point(
                    logic, idx, dict(zip(logic._sweep_loop_order, combination))))
            self.assertEqual(applied, [('6kHz_minphase', [100, 200]), ('10kHz_minphase', [300])] * 4)
            self.assertEqual(logic._run_odmr_scan.call_count, 8)
            self.assertEqual(logic._fit_odmr_data.call_count, 8)
            logic._sleep_interruptible.assert_called_once_with(180)
            self.assertEqual([r['power_dbm'] for r in results], [-10] * 4 + [-5] * 4)
            self.assertEqual(len({c.args[0] for c in logic._run_odmr_scan.call_args_list}), 8)
            self.assertIs(logic._stream_parameters, original)

    def test_cancel_between_filters_restores_parameters_and_disables_output(self):
        with tempfile.TemporaryDirectory() as folder:
            logic = self.make_matrix_logic(folder)
            original = logic._stream_parameters
            def first_fit(*args):
                logic._cancel_requested = True
                return None
            logic._fit_odmr_data.side_effect = first_fit
            with self.assertRaises(InterruptedError):
                SensitivitySweepLogic._measure_single_point(
                    logic, 0, {'power': -10, 'f_mod': 15000, 'f_dev': 100})
            self.assertEqual(logic._run_odmr_scan.call_count, 1)
            self.assertIs(logic._stream_parameters, original)
            logic._odmr_logic.return_value.toggle_cw_output.assert_called_with(False)

    def test_each_filter_uses_its_own_slope_and_integral_bandwidths(self):
        with tempfile.TemporaryDirectory() as folder:
            logic = self.make_matrix_logic(folder)
            logic._stream_parameters.update(
                tracking_comparison_enabled=True,
                tracking_modes=['open_loop', 'closed_loop_conventional'])
            logic._include_off_resonant_measurement = False
            logic._which_zero_crossing = 0
            logic._calculate_odmr_center.return_value = 2.85e9
            logic._raw_discriminator_slope.side_effect = lambda slope: (slope, 1.)
            logic._fit_odmr_data.side_effect = [
                {'zero_crossing_frequencies [Hz]': [2.85e9],
                 'zero_crossing_slopes [V/Hz]': [slope], 'linewidths [Hz]': [1e6]}
                for slope in (3., 6.)]
            def measure(**kwargs):
                return (dict(kwargs['condition'], sensitivity_nT_rtHz=1.), [1., 2.], [1., 1.])
            logic._measure_tracking_condition.side_effect = measure
            results = SensitivitySweepLogic._measure_single_point(
                logic, 0, {'power': -10, 'f_mod': 15000, 'f_dev': 100})
            calls = logic._measure_tracking_condition.call_args_list
            self.assertEqual([c.kwargs['signed_slope'] for c in calls], [3., 3., 3., 6., 6.])
            self.assertEqual([r['controller_bandwidth_hz'] for r in results], [None, 100., 200., None, 300.])
            self.assertEqual([r['controller_algorithm'] for r in results],
                             ['disabled', 'conventional', 'conventional', 'disabled', 'conventional'])
            self.assertEqual([r['fir_filter_bandwidth'] for r in results],
                             ['6kHz_minphase'] * 3 + ['10kHz_minphase'] * 2)

    def test_sorted_results_keep_filter_and_sensitivity_in_same_row(self):
        table = QtWidgets.QTableWidget(0, 13)
        self.addCleanup(table.close)
        table.setSortingEnabled(True)
        table.sortItems(10, QtCore.Qt.AscendingOrder)
        gui = SimpleNamespace(_results_table=table)
        for name, sensitivity in [('6kHz_minphase', 9.), ('10kHz_minphase', 1.)]:
            SensitivitySweepGui._on_point_completed(gui, 0, {
                'power_dbm': -10, 'f_mod_hz': 15000, 'f_dev_khz': 100,
                'fir_filter_bandwidth': name, 'sensitivity_nT_rtHz': sensitivity})
        self.assertEqual(table.item(0, 12).text(), '10kHz_minphase')
        self.assertEqual(table.item(0, 10).text(), '1.000')
        self.assertEqual(table.item(1, 12).text(), '6kHz_minphase')


if __name__ == '__main__':
    unittest.main()
