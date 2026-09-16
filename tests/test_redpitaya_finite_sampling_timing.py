"""Regression tests for FPGA-valid-aware finite ODMR scan timing."""
import unittest
from contextlib import nullcontext
from unittest.mock import Mock

import numpy as np

from qudi.hardware.redpitaya.redpitaya_finite_sampling_input import (
    _calculate_scan_timing,
)
from qudi.logic.odmr_logic import OdmrLogic
from qudi.util.enums import SamplingOutputMode


class TestRedPitayaFiniteScanTiming(unittest.TestCase):
    def test_long_settling_explains_four_second_scan(self):
        timing = _calculate_scan_timing(
            1000, 50e-6, 0.0035, 125_000_000, 4096, 16)
        self.assertTrue(timing['rate_limited'])
        self.assertAlmostEqual(1000 / timing['actual_rate_hz'], 4.074288)

    def test_odmr_reports_accepted_rate_with_oversampling(self):
        for oversampling in (1, 5):
            with self.subTest(oversampling=oversampling):
                logic = Mock()
                logic._threadlock = nullcontext()
                logic.module_state.return_value = 'idle'
                logic._oversampling_factor = oversampling
                logic._data_rate = 1000.0
                logic._scan_frequency_ranges = [(2.8e9, 2.9e9, 100)]
                logic._frequency_data = [np.linspace(2.8e9, 2.9e9, 100)]
                logic._default_scan_mode = SamplingOutputMode.JUMP_LIST
                scanner = logic._data_scanner.return_value
                scanner.sample_rate = 245.44166804014935

                OdmrLogic.start_odmr_scan(logic)

                scanner.set_sample_rate.assert_called_once_with(1000.0 * oversampling)
                scanner.set_frame_size.assert_called_once_with(100 * oversampling)
                self.assertEqual(logic._data_rate, scanner.sample_rate / oversampling)
                logic.sigScanParametersUpdated.emit.assert_called_once_with(
                    {'data_rate': scanner.sample_rate / oversampling})
                self.assertEqual(
                    logic._microwave.return_value.configure_scan.call_args.args[-1],
                    scanner.sample_rate)
                logic._sigNextLine.emit.assert_called_once()

    def test_fast_profile_fits_42_valid_samples_in_one_ms_point(self):
        timing = _calculate_scan_timing(
            requested_rate_hz=1000.0,
            trigger_time_s=50e-6,
            settling_time_s=600e-6,
            fpga_clock_hz=125_000_000,
            valid_period_cycles=1024,
            minimum_valid_samples=16)
        self.assertFalse(timing['rate_limited'])
        self.assertEqual(timing['dwell_cycles'], 43_750)
        self.assertEqual(timing['guaranteed_valid_samples'], 42)
        self.assertEqual(timing['actual_rate_hz'], 1000.0)

    def test_fast_profile_guarantees_valid_samples_when_rate_is_impossible(self):
        timing = _calculate_scan_timing(
            requested_rate_hz=1000.0,
            trigger_time_s=50e-6,
            settling_time_s=1e-3,
            fpga_clock_hz=125_000_000,
            valid_period_cycles=1024,
            minimum_valid_samples=16)
        self.assertTrue(timing['rate_limited'])
        self.assertEqual(timing['dwell_cycles'], 16 * 1024)
        self.assertEqual(timing['guaranteed_valid_samples'], 16)
        self.assertAlmostEqual(timing['actual_rate_hz'], 846.6884322039639)

    def test_stable_profile_keeps_requested_rate_when_window_fits(self):
        timing = _calculate_scan_timing(
            requested_rate_hz=1000.0,
            trigger_time_s=50e-6,
            settling_time_s=100e-6,
            fpga_clock_hz=125_000_000,
            valid_period_cycles=4096,
            minimum_valid_samples=16)
        self.assertFalse(timing['rate_limited'])
        self.assertEqual(timing['actual_rate_hz'], 1000.0)
        self.assertGreaterEqual(timing['guaranteed_valid_samples'], 16)

    def test_non_demodulated_input_still_gets_a_positive_dwell(self):
        timing = _calculate_scan_timing(
            requested_rate_hz=1000.0,
            trigger_time_s=50e-6,
            settling_time_s=1e-3,
            fpga_clock_hz=125_000_000)
        self.assertEqual(timing['dwell_cycles'], 1)
        self.assertGreater(timing['dwell_time_s'], 0.0)


if __name__ == '__main__':
    unittest.main()
