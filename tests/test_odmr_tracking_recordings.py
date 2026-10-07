"""Offline checks of the tracking-hardware recordings (independent stream
subscriptions with incrementally reported x/y markers)."""
import threading
import unittest
from types import SimpleNamespace, MethodType
from unittest.mock import Mock

import numpy as np

from qudi.hardware.redpitaya.redpitaya_odmr_lock import RedPitayaOdmrLockHardware
from pyrpl.hardware_modules.data_streamer import DataStreamer

INTERVAL = 4096
ORIGIN = 1_000_000


class _Reader:
    """Returns pre-recorded event chunks, one per read_events call."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.error = None

    def read_events(self):
        if self._chunks:
            return self._chunks.pop(0)
        return np.empty(0, dtype=DataStreamer.EVENT_DTYPE)

    def stats(self):
        return {'n_gap': 0}


def _events(rows):
    """rows: (source, tick, data, flags) tuples, in arrival order."""
    out = np.zeros(len(rows), dtype=DataStreamer.EVENT_DTYPE)
    for i, (source, tick, data, flags) in enumerate(rows):
        out[i]['source'] = DataStreamer.SOURCE_IDS[source]
        out[i]['tick'] = tick
        out[i]['timestamp'] = tick & 0xFFFFFFFF
        out[i]['data'] = data
        out[i]['flags'] = flags
    return out


def _hardware(chunks):
    streamer = SimpleNamespace(
        SOURCE_IDS=DataStreamer.SOURCE_IDS, EVENT_DTYPE=DataStreamer.EVENT_DTYPE,
        reconstruct_tracking=DataStreamer.reconstruct_tracking,
        subscribe=Mock(side_effect=lambda *a, **k: _Reader(chunks)),
        unsubscribe=Mock(), stats=Mock(return_value={'fpga_overflows': 0}))
    hw = SimpleNamespace(
        _TRACE_SOURCES=RedPitayaOdmrLockHardware._TRACE_SOURCES,
        _streamer=streamer, _recordings={}, _next_recording=1,
        _recording_lock=threading.RLock(), _sample_period_cycles=INTERVAL,
        _nslots=2, _fpga_clock_hz=125_000_000,
        _stream_sample_rate_hz=125e6 / INTERVAL,
        _odmr=SimpleNamespace(ftw_to_hz=lambda v: np.asarray(v, dtype=float) / 34.36),
        _require_active=lambda: None, log=Mock())
    for name in ('_subscribe_traces', 'open_recording', '_recording', 'read_recording',
                 '_recording_markers', 'close_recording', 'reconstruct_recording'):
        setattr(hw, name, MethodType(getattr(RedPitayaOdmrLockHardware, name), hw))
    return hw


def _fir(k, slot):
    # flags bit 0: live window, bit 1: slot
    return ('fir', ORIGIN + k * INTERVAL, 100 + k, 1 | (slot << 1))


class TestRecordings(unittest.TestCase):
    def setUp(self):
        # Chunk 0 holds only markers (one before the first FIR sample); later
        # chunks interleave FIR samples of two hopping resonances with markers.
        self.chunks = [
            _events([('x_position', ORIGIN - 3 * INTERVAL + 7, 0, 0),
                     ('y_position', ORIGIN - 10, 0, 0)]),
            _events([_fir(0, 0), _fir(1, 0),
                     ('x_position', ORIGIN + INTERVAL + 1, 0, 0),
                     _fir(2, 1), ('x_position', ORIGIN + 2 * INTERVAL, 0, 0)]),
            _events([_fir(3, 1), ('correction', ORIGIN + 3 * INTERVAL + 5, 3436, 1 << 1),
                     ('x_position', ORIGIN + 4 * INTERVAL - 1, 0, 0),
                     _fir(4, 0), ('y_position', ORIGIN + 4 * INTERVAL + 100, 0, 0)]),
        ]
        self.hw = _hardware(self.chunks)

    def test_incremental_markers_match_full_reconstruction(self):
        handle = self.hw.open_recording()
        events, xs, ys = [], [], []
        for _ in range(4):
            new = self.hw.read_recording(handle)
            events.append(new['events'])
            xs.append(new['x_markers'])
            ys.append(new['y_markers'])
        full = self.hw.reconstruct_recording(np.concatenate(events))
        np.testing.assert_array_equal(np.concatenate(xs), full['x_markers'])
        np.testing.assert_array_equal(np.concatenate(ys), full['y_markers'])
        # first sample at or after the pulse (negative before the first sample)
        np.testing.assert_array_equal(full['x_markers'], [-2, 2, 2, 4])
        np.testing.assert_array_equal(full['y_markers'], [0, 5])

    def test_markers_held_until_the_first_sample(self):
        handle = self.hw.open_recording()
        first = self.hw.read_recording(handle)
        self.assertEqual(first['x_markers'].size + first['y_markers'].size, 0)
        second = self.hw.read_recording(handle)
        np.testing.assert_array_equal(second['x_markers'], [-2, 2, 2])
        np.testing.assert_array_equal(second['y_markers'], [0])

    def test_reconstruction_places_slots_and_counts_sources(self):
        handle = self.hw.open_recording()
        events = np.concatenate([self.hw.read_recording(handle)['events']
                                 for _ in range(3)])
        rec = self.hw.reconstruct_recording(events)
        self.assertEqual(rec['err'].shape, (2, 5))
        np.testing.assert_array_equal(rec['err'][0], [100, 101, np.nan, np.nan, 104])
        np.testing.assert_array_equal(rec['err'][1], [np.nan, np.nan, 102, 103, np.nan])
        self.assertAlmostEqual(rec['corr_hz'][1, 3], 100.0)
        self.assertEqual(rec['event_counts']['fir'], 5)
        self.assertEqual(rec['event_counts']['x_position'], 4)
        np.testing.assert_allclose(rec['times'], np.arange(5) * INTERVAL / 125e6)

    def test_recordings_are_independent_and_closed(self):
        a = self.hw.open_recording()
        b = self.hw.open_recording()
        self.assertNotEqual(a, b)
        self.assertEqual(self.hw._streamer.subscribe.call_count, 2)
        stats = self.hw.close_recording(a)
        self.assertEqual(stats['transport'], {'n_gap': 0})
        self.assertEqual(self.hw.close_recording(a), {})      # idempotent
        self.assertEqual(self.hw._streamer.unsubscribe.call_count, 1)
        with self.assertRaises(ValueError):
            self.hw.read_recording(a)
        self.hw.read_recording(b)

    def test_receiver_error_raises(self):
        handle = self.hw.open_recording()
        self.hw._recordings[handle]['reader'].error = 'socket closed'
        with self.assertRaises(RuntimeError):
            self.hw.read_recording(handle)


if __name__ == '__main__':
    unittest.main()
