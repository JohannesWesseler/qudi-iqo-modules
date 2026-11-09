"""Offline checks of the field-response analysis (sync markers, burst phasors, clock ratio,
step metrics) on synthetic data with known answers."""
import unittest

import numpy as np

from qudi.logic.field_response_logic import (sync_marker_ticks, burst_phasors, clock_ratio,
                                             step_metrics, current_phasor, fit_frequency,
                                             fit_calibration, step_traces, CLOCK_HZ)
from pyrpl.hardware_modules.data_streamer import DataStreamer


def _events(rows):
    out = np.zeros(len(rows), dtype=DataStreamer.EVENT_DTYPE)
    for k, (src, tick, flags) in enumerate(rows):
        out[k]['source'], out[k]['tick'], out[k]['flags'] = src, tick, flags
    return out


class TestAnalysis(unittest.TestCase):
    def test_sync_markers_modes(self):
        ev = _events([(0, 10, 1), (5, 100, 1), (5, 200, 5), (4, 300, 0), (5, 400, 4)])
        self.assertEqual(sync_marker_ticks(ev, 'sync').tolist(), [100, 200, 400])
        self.assertEqual(sync_marker_ticks(ev, 'both').tolist(), [200, 400])

    def test_burst_phasor_with_delay_and_clock_offset(self):
        f, eps, delay, phase = 12345.0, 23e-6, 24e-9, 0.7
        rate_cycles = 250                    # 500 kS/s stream
        ticks = np.arange(0, int(0.2 * CLOCK_HZ), rate_cycles, dtype=np.int64) + 1_000_000
        markers = np.array([1_000_000 + 3_000_017, 1_000_000 + 13_000_011], np.int64)
        # true signal: starts at each marker edge (marker tick - delay), RP2 frequency f,
        # i.e. f / (1 + eps) on the RP1 time axis
        y = np.zeros(ticks.size)
        for m in markers:
            t = (ticks - m) / CLOCK_HZ + delay
            on = (t >= 0) & (t < 0.04)
            y[on] += 2.0 * np.sin(2 * np.pi * f / (1 + eps) * t[on] + phase)
        y += np.random.default_rng(1).normal(0, 0.05, y.size)
        p, c, r, n = burst_phasors(ticks, y, markers, f, 0.04, skip=0.001, eps=eps,
                                   marker_delay=delay)
        self.assertTrue(np.allclose(np.abs(p), 2.0, rtol=2e-3), np.abs(p))
        self.assertTrue(np.allclose(np.angle(p), phase, atol=3e-3), np.angle(p))
        # ignoring a 23 ppm clock offset costs 360 * f * eps * T/2 ~ 2 deg here
        p0, *_ = burst_phasors(ticks, y, markers, f, 0.04, skip=0.001, eps=0.0, marker_delay=delay)
        self.assertGreater(abs(np.angle(p0[0]) - phase), 0.02)

    def test_clock_ratio(self):
        eps, period = -17.3e-6, 0.01
        mt = (np.arange(40) * period * (1 + eps) * CLOCK_HZ).round().astype(np.int64) + 5
        mt = np.delete(mt, [7, 21])          # lost markers must not matter
        self.assertAlmostEqual(clock_ratio(mt, period) * 1e6, eps * 1e6, delta=0.01)

    def test_fit_frequency_gives_clock_ratio(self):
        f, eps = 1000.0, 13.7e-6
        t = np.arange(0, 1.0, 2e-6)
        x = 3.0 * np.sin(2 * np.pi * f / (1 + eps) * t + 0.4) + np.random.default_rng(2).normal(0, 0.3, t.size)
        f_fit = fit_frequency(t, x, f)
        # Cramer-Rao: sigma_f = sqrt(12) sigma / (2 pi A T sqrt(N)) ~ 0.08 ppm here
        self.assertAlmostEqual((f / f_fit - 1) * 1e6, eps * 1e6, delta=0.35)

    def test_calibration_with_drift(self):
        # up-down current sequence while the line drifts 5 kHz/s (7 Nov): without the drift
        # term the slope is biased, with it exact
        i = np.array([0, 5, 10, 20, 10, 0, -5, -10, -20, -10, 0]) * 1e-3
        t = np.arange(i.size) * 1.1
        f = 2736.0 + 17.0 * i - 5e-3 * t
        k, f0, d = fit_calibration(i, f, t)
        self.assertAlmostEqual(k, 17.0, places=7)
        self.assertAlmostEqual(d, -5e-3, places=10)
        k0, _, _ = fit_calibration(i, f, t, drift=False)
        self.assertGreater(abs(k0 - 17.0), 0.5)

    def test_step_traces_drop_markers_without_step(self):
        grid = np.arange(-0.002, 0.04, 1e-5)
        step = np.where((grid >= 0) & (grid < 0.02), -1e5, 0.0)
        rng = np.random.default_rng(4)
        tr = [step + rng.normal(0, 1e4, grid.size) for _ in range(20)] + [rng.normal(0, 1e4, grid.size)]
        kept, dropped = step_traces(grid, tr, 0.02)
        self.assertEqual((len(kept), dropped), (20, 1))

    def test_step_metrics(self):
        t = np.arange(-0.002, 0.02, 2e-6)
        tau = 50e-6
        y = np.where(t >= 10e-6, 1 - np.exp(-(t - 10e-6) / tau), 0.0) * 5e6 + 2.87e9
        m = step_metrics(t, y)
        self.assertAlmostEqual(m['rise_10_90'], tau * np.log(9), delta=3e-6)
        self.assertAlmostEqual(m['delay_50'], 10e-6 + tau * np.log(2), delta=3e-6)
        self.assertAlmostEqual(m['step'], 5e6, delta=1e3)

    def test_current_phasor(self):
        fs, f = 125e6 / 64, 10e3
        t = (np.arange(16384) - 200) / fs
        i = np.where(t >= 0, 0.5 * np.sin(2 * np.pi * f * t - 0.3), 0.0)
        p, c, r = current_phasor(dict(t=t, current=i), f, t[-1])
        self.assertAlmostEqual(abs(p), 0.5, delta=1e-3)
        self.assertAlmostEqual(np.angle(p), -0.3, delta=1e-3)

    def test_burst_trend_removes_drift_bias(self):
        # 5 periods at 20 Hz while the line drifts -5 kHz/s: an unfitted drift leaks
        # ~d / (pi f) = 80 Hz into the sine; burst_phasors(trend=True) removes it
        f, amp, d = 20.0, 28e3, -5e3
        ticks = np.arange(0, int(0.3 * CLOCK_HZ), 400, dtype=np.int64)
        t = ticks / CLOCK_HZ
        x = amp * np.sin(2 * np.pi * f * t + 0.3) + d * t
        p0 = burst_phasors(ticks, x, [0], f, 5 / f, marker_delay=0.0)[0][0]
        p1 = burst_phasors(ticks, x, [0], f, 5 / f, marker_delay=0.0, trend=True)[0][0]
        self.assertGreater(abs(p0 - amp * np.exp(0.3j)), 50)
        self.assertLess(abs(p1 - amp * np.exp(0.3j)), 1e-6 * amp)

if __name__ == '__main__':
    unittest.main()
