"""End-to-end check of FieldResponseLogic.transfer_function / response_map on a simulated current
source and tracker (no hardware, no qudi runtime): a first-order tracker T(f) = 1 / (1 + i f / fc),
a drifting resonance, sync markers, and a current readback with a known error. The analysis must
return H = T(f) with the command reference, H = T(f) / (readback error) with the shunt reference,
and report the readback error as shunt_over_cmd."""
import logging
import unittest

import numpy as np

from qudi.logic.field_response_logic import FieldResponseLogic, CLOCK_HZ
from pyrpl.hardware_modules.data_streamer import DataStreamer

K_MHZ = -11.6            # MHz/A
FC = 8e3                 # Hz, tracker bandwidth of the simulation
GM = 4.75 * np.exp(-0.15j)   # I/OUT1 at every frequency (A/V)
SHUNT_ERR = 1.05 * np.exp(1j * np.radians(1.0))   # readback error of the simulated shunt
DRIFT = -5e3             # Hz/s
SPACING = 400            # clocks between stream samples
DELAY = 24e-9


def tracker_t(f):
    return 1.0 / (1.0 + 1j * f / FC)


class FakeTracker:
    tracking_engine_active = True
    stream_sample_rate = CLOCK_HZ / SPACING

    def __init__(self):
        self.tick = 10_000_000
        self.values = {}
        self.pending = []
        self.marker_input = 'kdc'

    def set_marker_input(self, mode):
        self.marker_input = mode

    def open_recording(self):
        self.pending = []
        return 1

    def read_recording(self, handle):
        ev = np.concatenate(self.pending) if self.pending else np.zeros(0, DataStreamer.EVENT_DTYPE)
        self.pending = []
        return {'events': ev}

    def close_recording(self, handle):
        return {}

    def burst(self, f, amp, n_cyc, rng):
        """Tracked shift (Hz) for a sine burst of current amplitude amp (A) starting now."""
        start = self.tick
        dur = n_cyc / f
        ticks = np.arange(start - 2000 * SPACING, start + int((dur + 0.04) * CLOCK_HZ), SPACING,
                          dtype=np.int64)
        t = (ticks - start) / CLOCK_HZ + DELAY
        p = K_MHZ * 1e6 * amp * np.exp(1j * np.angle(GM)) * tracker_t(f)
        y = np.where((t >= 0) & (t < dur), abs(p) * np.sin(2 * np.pi * f * t + np.angle(p)), 0.0)
        y += DRIFT * ticks / CLOCK_HZ + rng.normal(0, 2e3, ticks.size)
        self.values.update(zip(ticks.tolist(), y.tolist()))
        ev = np.zeros(ticks.size + 1, DataStreamer.EVENT_DTYPE)
        ev['source'][:-1], ev['tick'][:-1] = 0, ticks
        ev['source'][-1], ev['tick'][-1] = 5, start
        self.pending.append(ev)
        self.tick = int(ticks[-1]) + 10 * SPACING

    def reconstruct_recording(self, ev):
        ev = np.asarray(ev)
        ticks = np.sort(ev[ev['source'] == 0]['tick']).astype(np.int64)
        corr = np.array([self.values[int(k)] for k in ticks])
        return {'ticks': ticks, 'corr_hz': corr[None, :], 'err': np.zeros((1, ticks.size))}


class FakeSource:
    """CurrentSourceLogic as far as FieldResponseLogic uses it."""

    def __init__(self, tracker):
        self.tr = tracker
        self.rng = np.random.default_rng(7)
        self.n_cyc, self.f, self.amp = 1, 1.0, 0.0
        self.sync = False

    def max_current(self, waveform, f):
        return 1.0

    def configure_burst(self, n, rep=1, period=None):
        self.n_cyc = n or 1

    def set_output(self, waveform, f, amp):
        self.f, self.amp = f, amp

    def output_off(self):
        pass

    def set_sync_output(self, on):
        self.sync = on

    def expected_current(self, f, amplitude):
        return amplitude * GM / abs(GM)

    def arm_capture(self, dec, trigger, pre):
        self.dec = dec

    def trigger(self):
        self.tr.burst(self.f, self.amp, self.n_cyc, self.rng)

    def read_capture(self, timeout):
        fs = CLOCK_HZ / self.dec
        t = (np.arange(16384) - 200) / fs
        p = self.amp * np.exp(1j * np.angle(GM)) * SHUNT_ERR
        i = np.where(t >= 0, abs(p) * np.sin(2 * np.pi * self.f * t + np.angle(p)), 0.0)
        return dict(t=t, current=i)


class SimLogic:
    """FieldResponseLogic's measurement methods on the fakes (ConfigOptions as plain values)."""
    _gamma = 28.025
    _eps = 0.0
    _marker_delay = DELAY
    _current_reference = 'command'
    log = logging.getLogger('sim_field_response')

    def __init__(self):
        self._abort = False
        self._results = {}
        self._nv_cal = {'0': dict(mhz_per_a=K_MHZ)}
        self._source = FakeTracker()
        self.coil = FakeSource(self._source)

    def _current_source(self):
        return self.coil

    def _save(self, *args, **kwargs):
        return 'not saved'


for _name in ('transfer_function', '_tf_point', 'response_map', 'amplitude_for', '_k_mhz',
              '_require_tracking', '_with_sync', '_reconstruct', '_record', '_check', '_tr',
              '_plot_tf', '_plot_map'):
    setattr(SimLogic, _name, FieldResponseLogic.__dict__[_name])


class TestTransferFunctionSim(unittest.TestCase):
    def test_command_reference(self):
        lg = SimLogic()
        freqs = [20.0, 1e3, 8e3]
        res = lg.transfer_function(freqs, amplitude_ut=2.0, n_bursts=2)
        a = 2.0e-3 * 28.025 / abs(K_MHZ)
        self.assertAlmostEqual(res['amplitude'], a)
        self.assertEqual(lg._source.marker_input, 'sync')
        self.assertNotIn('reference', res)
        for row, f in zip(res['rows'], freqs):
            h = tracker_t(f)
            self.assertAlmostEqual(row['h_abs'], abs(h), delta=2e-3)
            self.assertAlmostEqual(row['h_phase_deg'], np.degrees(np.angle(h)), delta=0.2)
            self.assertNotIn('h_norm_abs', row)
            self.assertAlmostEqual(row['applied_ut'], 2.0, delta=1e-9)
            self.assertAlmostEqual(row['shunt_over_cmd'], 1.05, delta=1e-3)
            self.assertAlmostEqual(row['shunt_phase_deg'], 1.0, delta=0.05)
            self.assertEqual(row['markers'], 2)

    def test_shunt_reference(self):
        lg = SimLogic()
        res = lg.transfer_function([2e3], amplitude=0.005, n_bursts=1, current_reference='shunt')
        h = tracker_t(2e3) / SHUNT_ERR
        self.assertAlmostEqual(res['rows'][0]['h_abs'], abs(h), delta=2e-3)
        self.assertAlmostEqual(res['rows'][0]['h_phase_deg'], np.degrees(np.angle(h)), delta=0.2)

    def test_response_map(self):
        lg = SimLogic()
        res = lg.response_map([5e3], [1.0, 4.0], n_bursts=1)
        self.assertEqual(len(res['maps']), 2)
        for m, a in zip(res['maps'], (1.0, 4.0)):
            self.assertEqual(m['rows'][0]['amplitude_ut'], a)
            self.assertAlmostEqual(m['rows'][0]['tracked_ut'], a * abs(tracker_t(5e3)), delta=3e-3)

    def test_missing_nv_calibration(self):
        lg = SimLogic()
        lg._nv_cal = {}
        with self.assertRaises(RuntimeError):
            lg.transfer_function([1e3], amplitude_ut=1.0)


if __name__ == '__main__':
    unittest.main()
