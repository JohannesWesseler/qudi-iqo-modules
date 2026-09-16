import json
import unittest

import numpy as np

from qudi.logic.tracking_sensitivity_tools import (
    build_tracking_conditions,
    json_safe,
    reconstruct_field_traces,
)


class TrackingSensitivityToolsTest(unittest.TestCase):

    def test_comparison_matrix_reuses_one_open_loop_condition(self):
        conditions = build_tracking_conditions(
            ['open_loop', 'closed_loop_conventional', 'closed_loop_smith'],
            [300, 1000], [2, 4], [None, 64])
        self.assertEqual(len(conditions), 1 + 2 + 2 * 2 * 2)
        self.assertEqual(conditions[0]['controller_algorithm'], 'disabled')
        self.assertEqual(
            [item['controller_bandwidth_hz'] for item in conditions[1:3]],
            [300.0, 1000.0])
        self.assertEqual(conditions[-1]['smith_gain_multiplier'], 4.0)
        self.assertEqual(conditions[-1]['smith_delay_samples'], 64)

    def test_open_loop_does_not_require_controller_values(self):
        conditions = build_tracking_conditions(['open_loop'], [], [], [])
        self.assertEqual(len(conditions), 1)

    def test_field_reconstruction_upper_sideband(self):
        # slope*(drive-resonance): error -2 means resonance is +1 Hz away.
        result = reconstruct_field_traces(
            error_signal=np.array([-2.0, 2.0]),
            correction_hz=np.array([10.0, 20.0]),
            signed_slope_per_hz=2.0,
            gyromagnetic_ratio_hz_per_nt=1.0)
        np.testing.assert_allclose(result['residual_frequency_hz'], [1.0, -1.0])
        np.testing.assert_allclose(result['estimated_frequency_hz'], [11.0, 19.0])

    def test_field_reconstruction_lower_sideband_flips_ftw_only(self):
        result = reconstruct_field_traces(
            error_signal=np.array([-2.0]), correction_hz=np.array([10.0]),
            signed_slope_per_hz=2.0, correction_inverted=True,
            gyromagnetic_ratio_hz_per_nt=1.0)
        np.testing.assert_allclose(result['correction_frequency_hz'], [-10.0])
        np.testing.assert_allclose(result['estimated_frequency_hz'], [-9.0])

    def test_json_safe_is_strict_json_serializable(self):
        value = json_safe({
            'array': np.array([1, np.nan]),
            'scalar': np.float64(3.5),
            'invalid': np.float64(np.inf),
        })
        self.assertEqual(value['array'], [1.0, None])
        self.assertIsNone(value['invalid'])
        json.dumps(value, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
