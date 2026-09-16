# -*- coding: utf-8 -*-
"""Tests for battery protocol validation and dummy control orchestration."""

import json
import tempfile
import unittest
import weakref
from pathlib import Path

from PySide2 import QtCore

from qudi.hardware.battery_tester.hrt_battery_tester import HrtBatteryTester
from qudi.hardware.battery_tester.playwright_transport import PlaywrightHrtTransport
from qudi.hardware.battery_tester.socketio_transport import SocketIoHrtTransport
from qudi.hardware.dummy.battery_tester_dummy import BatteryTesterDummy
from qudi.interface.battery_tester_interface import (
    BatteryTesterControlUnavailableError,
    BatteryTesterProtocolError,
)
from qudi.interface.battery_tester_models import ChannelOperatingState, SafetyLimits
from qudi.interface.battery_tester_task import (
    BatteryProtocol,
    BatteryTaskSpec,
    BatteryTaskValidationError,
    ProtocolGlobalOverride,
)
from qudi.logic.battery_tester_logic import BatteryTesterLogic


class _FakeQudi:
    pass


class _FakeTransport:

    def __init__(self, rows=None, start_error=None, read_error=None):
        self.rows = rows or []
        self.start_error = start_error
        self.read_error = read_error
        self.closed = False

    def start(self):
        if self.start_error is not None:
            raise self.start_error

    def get_live_channels(self):
        if self.read_error is not None:
            raise self.read_error
        return self.rows

    def close(self):
        self.closed = True


class _FakePage:
    url = 'http://tester/system'

    def screenshot(self, path, full_page):
        Path(path).write_bytes(b'png')

    def content(self):
        return '<html><body>diagnostic</body></html>'


def _protocol() -> BatteryProtocol:
    return BatteryProtocol(json.dumps({
        'version': 2.1,
        'globals': [{'name': 'I_TEST', 'value': 0.1, 'unit': 'A'}],
        'steps': [
            {
                'id': 1,
                'type': 'LOOP_START',
                'repetitions': 1,
                'steps': [{'id': 2, 'type': 'PAUSE', 'content': {}}],
            },
            {'id': 3, 'type': 'LOOP_END'},
        ],
    }), source_name='test.json')


def _task(**changes) -> BatteryTaskSpec:
    values = {
        'channel_id': 1,
        'test_name': 'dummy-cycle',
        'protocol': _protocol(),
        'battery_model': 'dummy-cell',
        'battery_number': '001',
        'battery_capacity_ah': 1.0,
        'safety_limits': SafetyLimits(
            minimum_voltage_v=2.5,
            maximum_voltage_v=4.2,
            maximum_discharge_current_a=0.5,
            maximum_charge_current_a=0.5,
            maximum_temperature_c=45.0,
        ),
        'global_overrides': (ProtocolGlobalOverride('I_TEST', 0.2, 'A'),),
    }
    values.update(changes)
    return BatteryTaskSpec(**values)


class BatteryProtocolTest(unittest.TestCase):

    def test_checksum_is_independent_of_json_formatting(self):
        first = _protocol()
        reordered = BatteryProtocol(
            '{"steps":[{"type":"LOOP_START","steps":[{"content":{},"type":"PAUSE",'
            '"id":2}],"repetitions":1,"id":1},{"type":"LOOP_END","id":3}],'
            '"globals":[{"unit":"A","value":0.1,"name":"I_TEST"}],"version":2.1}'
        )
        self.assertEqual(first.checksum_sha256, reordered.checksum_sha256)
        self.assertEqual(first.step_ids, (1, 2, 3))

    def test_all_bundled_protocol_templates_validate(self):
        root = Path(__file__).parents[1] / 'src/qudi/hardware/battery_tester/Testprotokolle'
        protocols = [BatteryProtocol.from_path(path) for path in sorted(root.glob('*.json'))]
        self.assertEqual(len(protocols), 5)
        self.assertTrue(all(protocol.step_ids for protocol in protocols))
        self.assertEqual({protocol.version for protocol in protocols}, {'2.0', '2.1'})

    def test_unknown_global_override_is_rejected(self):
        with self.assertRaisesRegex(BatteryTaskValidationError, 'Unknown protocol globals'):
            _task(global_overrides=(ProtocolGlobalOverride('DOES_NOT_EXIST', 1),))

    def test_incomplete_safety_envelope_is_rejected(self):
        with self.assertRaisesRegex(BatteryTaskValidationError, 'explicitly specified'):
            _task(safety_limits=SafetyLimits(2.5, 4.2, 0.5, 0.5, None))


class BatteryTesterPassiveReliabilityTest(unittest.TestCase):

    def test_live_transport_is_rebuilt_after_a_read_failure(self):
        qudi = _FakeQudi()
        qudi.configuration = {'default_data_dir': None, 'daily_data_dirs': True}
        hardware = HrtBatteryTester(
            weakref.ref(qudi),
            'hrt_battery_tester',
            config={
                'base_url': 'http://127.0.0.1:1841/',
                'reconnect_attempts': 1,
                'reconnect_backoff_s': 0,
                'capture_diagnostics_on_error': False,
            },
        )
        failed_current = _FakeTransport(read_error=RuntimeError('live transport crashed'))
        failed_replacement = _FakeTransport(start_error=RuntimeError('launch failed'))
        good_replacement = _FakeTransport(rows=[{'channelID': 13}])
        replacements = iter((failed_replacement, good_replacement))
        hardware._live_data_transport = failed_current
        hardware._new_live_transport = lambda: next(replacements)

        rows = hardware._read_live_channels_with_reconnect()

        self.assertEqual(rows, [{'channelID': 13}])
        self.assertTrue(failed_current.closed)
        self.assertTrue(failed_replacement.closed)
        self.assertIs(hardware._live_data_transport, good_replacement)

    def test_socketio_payload_is_normalized_and_staleness_is_detected(self):
        class FakeClient:
            connected = True

            def emit(self, event):
                pass

            def disconnect(self):
                self.connected = False

        transport = SocketIoHrtTransport(
            'http://127.0.0.1:1841/', max_live_age_s=5
        )
        transport._client = FakeClient()
        transport._connected.set()
        transport._on_new_system_data({
            'ch_id': 13,
            'controller_id': 13,
            'ch_status': 120,
            'exp_id': 42,
            'time_s': 12.5,
            'protocol_step': 3,
            'stepTime_s': 2.5,
            'loop_counter': 1,
            'voltage': 3700,
            'voltage_aux': 10,
            'voltage_power_terminals': 3690,
            'current': 125,
            'temp': 24,
            'temp2': 25,
        })

        row = transport.get_live_channels()[0]
        self.assertEqual(row['channelID'], 13)
        self.assertEqual(row['lastLiveExpID'], 42)
        self.assertTrue(row['running'])
        self.assertEqual(row['live_voltage'], 3700)

        transport._updated_monotonic[13] -= 10
        with self.assertRaisesRegex(Exception, 'stale'):
            transport.get_live_channels()
        transport.close()

    def test_socketio_idle_payload_clears_experiment_counters(self):
        transport = SocketIoHrtTransport('http://127.0.0.1:1841/')
        transport._on_new_system_data({
            'ch_id': 13,
            'controller_id': 13,
            'ch_status': 0,
            'exp_id': 0,
            'time_s': 948151107,
            'protocol_step': 99,
            'stepTime_s': 123,
            'loop_counter': 7,
        })
        row = transport._cache[13]

        self.assertFalse(row['running'])
        self.assertIsNone(row['live_uptime'])
        self.assertIsNone(row['protocolStep'])
        self.assertIsNone(row['protocolStepTime'])
        self.assertIsNone(row['loopCounter'])

    def test_browser_diagnostics_are_bounded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            transport = PlaywrightHrtTransport(
                'http://127.0.0.1:1841/',
                diagnostics_dir=temp_dir,
                max_diagnostic_sets=2,
            )
            page = _FakePage()
            for index in range(3):
                path = transport._capture_diagnostics(
                    page, 'test', RuntimeError(f'failure {index}')
                )
                self.assertIsNotNone(path)

            diagnostics = sorted(Path(temp_dir).glob('hrt_diag_*'))
            self.assertEqual(len(diagnostics), 2)
            for directory in diagnostics:
                self.assertTrue((directory / 'page.png').is_file())
                self.assertTrue((directory / 'page.html').is_file())
                self.assertTrue((directory / 'metadata.json').is_file())


class BatteryTesterDummyLogicTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.qt_app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.qudi = _FakeQudi()
        self.qudi.configuration = {
            'default_data_dir': self.temp_dir.name,
            'daily_data_dirs': False,
        }
        self.tester = BatteryTesterDummy(
            weakref.ref(self.qudi),
            'battery_tester_dummy',
            config={
                'channel_ids': [1, 2],
                'initial_voltage_v': 3.7,
                'temperature_c': 25.0,
                'start_delay_s': 0.01,
                'experiment_duration_s': 10.0,
            },
        )
        self.logic = BatteryTesterLogic(
            weakref.ref(self.qudi),
            'battery_tester_logic',
            config={'poll_interval_s': 60, 'require_temperature_sensor': True},
        )
        self.tester.module_state.activate()
        self.logic._battery_tester.connect(self.tester)
        self.logic.module_state.activate()

    def tearDown(self):
        snapshot = self.tester.get_channel_snapshot(1)
        if snapshot.running:
            self.tester.stop_task(1)
        if self.logic.module_state() == 'locked':
            self.logic.refresh_now()
        if self.logic.module_state() in ('idle', 'locked'):
            self.logic.module_state.deactivate()
        if self.tester.module_state() in ('idle', 'locked'):
            self.tester.module_state.deactivate()
        self.temp_dir.cleanup()

    def test_submit_is_idempotent_and_control_lifecycle_is_owned(self):
        spec = _task()
        self.logic.prepare_task(spec)
        first = self.logic.submit_task(spec)
        second = self.logic.submit_task(spec)
        self.assertEqual(first, second)
        self.assertEqual(len(self.logic.get_channel_snapshot(1).queue), 1)

        experiment = self.logic.start_task(first)
        self.assertEqual(experiment.task_id, first.task_id)
        self.assertTrue(self.logic.get_channel_snapshot(1).running)
        self.assertEqual(self.logic.module_state(), 'locked')

        self.logic.pause_task(1)
        self.logic.pause_task(1)
        self.assertIs(
            self.logic.get_channel_snapshot(1).state,
            ChannelOperatingState.PAUSED,
        )
        self.logic.resume_task(1)
        self.logic.resume_task(1)
        self.assertIsNot(
            self.logic.get_channel_snapshot(1).state,
            ChannelOperatingState.PAUSED,
        )
        self.logic.stop_task(1)
        self.assertFalse(self.logic.get_channel_snapshot(1).running)
        self.assertEqual(self.logic.module_state(), 'idle')

    def test_duplicate_uuid_with_different_task_is_rejected(self):
        original = _task()
        self.logic.submit_task(original)
        changed = _task(run_uuid=original.run_uuid, test_name='changed')
        with self.assertRaises(BatteryTesterProtocolError):
            self.tester.submit_task(changed)

    def test_live_hardware_style_capability_gate_fails_before_submission(self):
        capabilities = self.tester._capabilities
        self.tester._capabilities = type(capabilities)(
            **{**capabilities.__dict__, 'task_control_available': False}
        )
        self.logic._capabilities = self.tester.capabilities
        with self.assertRaises(BatteryTesterControlUnavailableError):
            self.logic.submit_task(_task())

    def test_missing_temperature_sensor_is_rejected(self):
        self.tester._temperature_c = None
        snapshot = self.tester.get_channel_snapshot(1)
        self.assertIsNone(snapshot.temperature_c)
        with self.assertRaisesRegex(BatteryTaskValidationError, 'sensor is unavailable'):
            self.logic.prepare_task(_task())

    def test_passive_monitoring_is_buffered_and_saved_with_qudi_storage(self):
        session = self.logic.start_monitoring(channel_ids=[1], nametag='unit test')
        self.assertTrue(session.active)
        self.assertTrue(Path(session.file_path).is_file())

        self.logic.refresh_now()
        self.logic.refresh_now()
        stopped = self.logic.stop_monitoring()

        self.assertFalse(stopped.active)
        self.assertEqual(stopped.rows_written, 3)
        self.assertEqual(stopped.buffered_rows, 0)
        self.assertTrue(stopped.plot_file_path)
        self.assertTrue(Path(stopped.plot_file_path).is_file())
        contents = Path(stopped.file_path).read_text()
        self.assertIn("schema='qudi-battery-monitor-v1'", contents)
        self.assertIn("live_transport='dummy'", contents)
        data_lines = [
            line for line in contents.splitlines()
            if line and not line.startswith('#')
        ]
        self.assertEqual(len(data_lines), 3)
        self.assertTrue(all(len(line.split('\t')) == 21 for line in data_lines))

    def test_polling_failure_is_recorded_without_stopping_monitoring(self):
        self.logic.start_monitoring(channel_ids=[1])

        def fail_poll():
            raise RuntimeError('simulated connection loss')

        self.tester.get_channel_snapshots = fail_poll
        self.logic.refresh()
        stopped = self.logic.stop_monitoring()

        self.assertEqual(stopped.connection_error_rows, 1)
        self.assertTrue(stopped.file_path)
        contents = Path(stopped.file_path).read_text()
        self.assertIn('connection_error', contents)


if __name__ == '__main__':
    unittest.main()
