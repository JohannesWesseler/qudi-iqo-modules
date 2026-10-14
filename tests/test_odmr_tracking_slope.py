"""Offline checks of the measured-slope check before the loop gains are set."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from qudi.logic.odmr_tracking_logic import OdmrTrackingLogic


def _check(measured, fitted):
    logic = SimpleNamespace(log=Mock())
    return OdmrTrackingLogic._checked_slope(logic, measured, fitted), logic.log


class CheckedSlopeTest(unittest.TestCase):

    def test_measured_slope_is_used(self):
        slope, log = _check(50.0, 47.0)
        self.assertEqual(slope, 50.0)
        log.warning.assert_not_called()

    def test_large_difference_warns(self):
        slope, log = _check(50.0, 5.8)
        self.assertEqual(slope, 50.0)
        log.warning.assert_called_once()

    def test_much_smaller_slope_is_refused(self):
        # Seen on the board: the resonance 140 kHz from the configured frequency
        # gave 14 instead of 40 LSB/Hz, and gains 2.9 times too high oscillated
        with self.assertRaises(RuntimeError):
            _check(13.8, 39.6)
        slope, log = _check(25.0, 39.6)
        self.assertEqual(slope, 25.0)
        log.warning.assert_called_once()

    def test_falling_error_signal_is_refused(self):
        with self.assertRaises(RuntimeError):
            _check(-47.0, 47.0)


if __name__ == '__main__':
    unittest.main()
