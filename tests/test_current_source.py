"""Offline checks of the current source parts that need no hardware: the sine fit (qudi.util),
the load model and calibration files of hardware/current_source, and the SCPI client's lock."""
import json
import os
import tempfile
import threading
import time
import unittest

import numpy as np

from qudi.util.sine_fit import fit_sine, harmonic_distortion
from qudi.hardware.current_source.load_model import LoadModel
from qudi.hardware.current_source.redpitaya_scpi import RedPitayaScpi
from qudi.hardware.current_source.redpitaya_lm1875_source import (calibration_path,
                                                                  load_amplifier_calibration,
                                                                  CALIBRATION_DIR)

LOAD_FILE = calibration_path('load_helmholtz_r17_5mm.json')


class TestSineFit(unittest.TestCase):
    def test_uniform_and_irregular_sampling_agree(self):
        fs, f = 1e6, 1234.5
        rng = np.random.default_rng(1)
        t = np.arange(20000) / fs
        x = 0.7 * np.sin(2 * np.pi * f * t - 1.1) + 0.05 + rng.normal(0, 1e-3, t.size)
        a = fit_sine(x, f, sample_rate=fs)
        b = fit_sine(x, f, t=t)
        self.assertAlmostEqual(a.amplitude, 0.7, delta=1e-4)
        self.assertAlmostEqual(a.phase, -1.1, delta=1e-4)
        self.assertAlmostEqual(a.offset, 0.05, delta=1e-4)
        self.assertAlmostEqual(a.phasor, b.phasor)
        self.assertAlmostEqual(a.resid_rms, 1e-3, delta=1e-4)

    def test_phase_follows_time_origin(self):
        fs, f = 1e5, 100.0
        x = np.sin(2 * np.pi * f * np.arange(5000) / fs)
        self.assertAlmostEqual(fit_sine(x, f, sample_rate=fs, t0=-0.25 / f).phase, np.pi / 2,
                               delta=1e-9)

    def test_trend(self):
        f, amp, d = 20.0, 28e3, -5e3
        t = np.arange(0, 5 / f, 3.2e-6)
        x = amp * np.sin(2 * np.pi * f * t + 0.3) + d * t
        self.assertGreater(abs(fit_sine(x, f, t=t).phasor - amp * np.exp(0.3j)), 50)
        fit = fit_sine(x, f, t=t, trend=True)
        self.assertLess(abs(fit.phasor - amp * np.exp(0.3j)), 1e-6 * amp)
        self.assertAlmostEqual(fit.slope, d, delta=1e-6 * abs(d))

    def test_needs_time_axis(self):
        with self.assertRaises(ValueError):
            fit_sine(np.zeros(10), 1.0)

    def test_harmonic_distortion(self):
        fs, f = 1e6, 1e3
        t = np.arange(100000) / fs
        x = np.sin(2 * np.pi * f * t) + 0.03 * np.sin(2 * np.pi * 2 * f * t) \
            + 0.04 * np.cos(2 * np.pi * 3 * f * t)
        self.assertAlmostEqual(harmonic_distortion(x, f, fs), 0.05, delta=1e-6)


class TestCalibrationFiles(unittest.TestCase):
    def test_amplifier_file(self):
        d = load_amplifier_calibration(calibration_path('amplifier.json'))
        self.assertEqual(d['offset_null'], -0.000794)
        self.assertEqual(d['adc_offsets'], [3.4e-4, -6e-5])      # [IN1, IN2]
        self.assertLess(d['out1_clean'], d['out1_max'])

    def test_amplifier_file_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'bad.json')
            with open(p, 'w') as fh:
                json.dump(dict(offset_null=0, adc_offsets=[0], out1_clean=0.7, out1_max=0.8), fh)
            with self.assertRaises(ValueError):
                load_amplifier_calibration(p)

    def test_paths(self):
        self.assertEqual(os.path.dirname(LOAD_FILE), CALIBRATION_DIR)
        self.assertEqual(calibration_path(os.path.abspath('x.json')), os.path.abspath('x.json'))


class TestLoadModel(unittest.TestCase):
    def setUp(self):
        self.m = LoadModel.from_file(LOAD_FILE)

    def test_file_values(self):
        self.assertAlmostEqual(abs(self.m.transconductance(1e3)[0]), 4.734, delta=0.01)
        # table down to 1 Hz (2026-11-09): joins the DC value, phase close to zero
        g1 = self.m.transconductance(1.0)[0]
        self.assertAlmostEqual(abs(g1), self.m.gm_dc, delta=0.002 * self.m.gm_dc)
        self.assertLess(abs(np.degrees(np.angle(g1))), 0.3)
        self.assertAlmostEqual(self.m.f[0], 1.0)
        self.assertAlmostEqual(abs(self.m.impedance(80e3)[0]), 11.30, delta=0.1)
        self.assertEqual(abs(self.m.transconductance(0)[0]), 4.756)
        self.assertEqual(self.m.r_dc, 3.433)
        info = self.m.info
        self.assertEqual(info['rated_current'], 1.66)
        self.assertEqual(info['heating']['dc_continuous_current'], 0.8)
        self.assertNotIn('table', info)

    def test_extrapolation_is_continuous(self):
        f_lo, f_hi = self.m.f[0], self.m.f[-1]
        for f in (f_lo, f_hi):
            a, b = self.m.transconductance([f * 0.999, f * 1.001])
            self.assertLess(abs(a / b - 1), 0.01)

    def test_output_ratio_healthy_and_faults(self):
        fs = 125e6 / 64
        t = np.arange(16384) / fs
        for i in (np.sin(2 * np.pi * 10e3 * t) + 0.2, np.sign(np.sin(2 * np.pi * 1e3 * t)),
                  0.5 + 0 * t):
            v = self.m.predicted_output(i, fs)           # healthy: output = Z * I
            ratio, _ = self.m.output_ratio(i, v, fs)
            self.assertAlmostEqual(ratio, 1.0, delta=0.02)
            self.assertGreater(self.m.output_ratio(0.1 * i, v, fs)[0], 3)    # open-ish load
            self.assertLess(self.m.output_ratio(i, 0.1 * v, fs)[0], 0.33)    # short

    def test_output_ratio_ignores_adc_noise_at_high_rate(self):
        fs = 125e6 / 16
        rng = np.random.default_rng(3)
        i = 0.0035 + rng.normal(0, 0.6e-3, 16384)          # idle: offset + white ADC noise
        v = 0.0035 * 3.43 + rng.normal(0, 16 * 0.4e-3, 16384)
        _, v_expected = self.m.output_ratio(i, v, fs)
        self.assertLess(v_expected, 0.1)                     # below the check threshold


class TestBurstResponse(unittest.TestCase):
    def test_rl_load(self):
        from qudi.logic.current_source_logic import burst_response
        f, v1, r, l, g = 2e3, 0.05, 3.43, 20.5e-6, 16.0
        z = r + 2j * np.pi * f * l
        gm = g / z
        fs = 125e6 / 64
        t = (np.arange(16384) - 200) / fs
        on = t >= 0
        i = np.where(on, abs(gm) * v1 * np.sin(2 * np.pi * f * t + np.angle(gm)), 0.0) + 1e-3
        v = np.where(on, abs(gm * z) * v1 * np.sin(2 * np.pi * f * t + np.angle(gm * z)), 0.0)
        res = burst_response(dict(t=t, current=i, output=v), f, v1, 5 / f)
        self.assertAlmostEqual(abs(res['gm'] / gm), 1.0, delta=1e-9)
        self.assertAlmostEqual(np.angle(res['gm'] / gm), 0.0, delta=1e-9)
        self.assertAlmostEqual(abs(res['z'] / z), 1.0, delta=1e-9)
        self.assertAlmostEqual(abs(res['gain']), g, delta=1e-9)
        self.assertAlmostEqual(res['i_pre_mean'], 1e-3, delta=1e-12)


class _EchoSocket:
    """Replies to every command line with the command itself, after a short delay, so that two
    unsynchronized queries would swap or mix their answers."""

    def __init__(self):
        self.lines = []
        self.cv = threading.Condition()
        self.rng = np.random.default_rng(0)

    def sendall(self, data):
        with self.cv:
            self.lines.append(data)
            self.cv.notify_all()
            delay = self.rng.uniform(0, 2e-4)
        time.sleep(delay)          # the sender is slow to get to its readline

    def readline(self):
        with self.cv:
            self.cv.wait_for(lambda: self.lines, timeout=2)
            return self.lines.pop(0) if self.lines else b''

    def settimeout(self, t):
        pass


class TestScpiLock(unittest.TestCase):
    def test_concurrent_queries_get_their_own_replies(self):
        rp = RedPitayaScpi.__new__(RedPitayaScpi)        # no connection
        rp._io_lock = threading.RLock()
        rp.log = __import__('logging').getLogger('test')
        rp.timeout = 1.0
        sock = _EchoSocket()
        rp.sock, rp._rf = sock, sock
        errors = []

        def worker(name):
            for k in range(200):
                cmd = f'{name}:{k}'
                if rp.query(cmd) != cmd:
                    errors.append(cmd)

        th = [threading.Thread(target=worker, args=(n,)) for n in ('A', 'B', 'C')]
        for t in th:
            t.start()
        for t in th:
            t.join(10)
        self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
