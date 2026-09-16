"""Thermal waiting must happen with the new microwave power applied."""
import tempfile
import unittest
from unittest.mock import Mock
from types import MethodType

from qudi.logic.sensitivity_sweep_logic import SensitivitySweepLogic


class ScanReached(Exception):
    pass


class TestThermalSettling(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.logic = Mock()
        self.logic._stream_parameters = {'fir_filter_bandwidth': '2kHz_minphase'}
        self.logic._measure_filter_point = MethodType(SensitivitySweepLogic._measure_filter_point, self.logic)
        self.logic._current_folder = self.folder.name
        self.logic._sweep_loop_order = ['power', 'f_mod', 'f_dev']
        self.logic._parameter_combinations = [(-10, 15000, 300), (-5, 15000, 300)]
        self.logic._odmr_parameters = {'frequency_start': 2.8e9, 'frequency_stop': 2.9e9}
        self.logic._thermal_stabilization_time = 180
        self.logic._total_combinations = 2
        self.events = []
        self.logic._configure_mw_source.side_effect = lambda *args: self.events.append('configure')
        self.logic._set_cw_frequency.side_effect = lambda *args: self.events.append(('cw', args))
        self.logic._odmr_logic.return_value.toggle_cw_output.side_effect = (
            lambda enabled: self.events.append(('output', enabled)))
        self.logic._sleep_interruptible.side_effect = self.wait
        self.logic._run_odmr_scan.side_effect = self.scan

    def wait(self, duration):
        self.events.append(('wait', duration))
        return True

    def scan(self, nametag):
        self.events.append('scan')
        raise ScanReached()

    def measure(self, power=-5):
        SensitivitySweepLogic._measure_single_point(
            self.logic, 1, {'power': power, 'f_mod': 15000, 'f_dev': 300})

    def test_new_power_is_on_before_wait_and_scan_follows_wait(self):
        with self.assertRaises(ScanReached):
            self.measure()
        self.assertEqual(self.events, [
            'configure', ('cw', (2.85e9, -5)), ('wait', 180),
            ('output', False), 'scan'])

    def test_same_power_does_not_wait(self):
        with self.assertRaises(ScanReached):
            self.measure(power=-10)
        self.assertEqual(self.events, ['configure', 'scan'])

    def test_interruption_turns_output_off_and_does_not_scan(self):
        self.logic._sleep_interruptible.side_effect = lambda duration: False
        with self.assertRaises(InterruptedError):
            self.measure()
        self.assertEqual(self.events, [
            'configure', ('cw', (2.85e9, -5)), ('output', False)])
        self.logic._run_odmr_scan.assert_not_called()


if __name__ == '__main__':
    unittest.main()
