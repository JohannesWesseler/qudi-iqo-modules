# -*- coding: utf-8 -*-
"""Unit tests for the HRT battery tester read-only integration."""

import json
import threading
import unittest
import weakref
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PySide2 import QtCore

from qudi.hardware.battery_tester.hrt_rest_client import HrtRestClient
from qudi.interface.battery_tester_interface import BatteryTesterInterface
from qudi.interface.battery_tester_models import (
    BatteryTesterCapabilities,
    ChannelOperatingState,
    TerminationReason,
    channel_snapshot_from_hrt_payloads,
)
from qudi.logic.battery_tester_logic import BatteryTesterLogic


class ChannelStateMappingTest(unittest.TestCase):

    def test_hrt_state_codes(self):
        expected = {
            None: ChannelOperatingState.UNKNOWN,
            0: ChannelOperatingState.STANDBY,
            1: ChannelOperatingState.TRANSITIONAL,
            2: ChannelOperatingState.PAUSED,
            7: ChannelOperatingState.TRANSITIONAL,
            8: ChannelOperatingState.TRANSITIONAL,
            12: ChannelOperatingState.CHARGING,
            20: ChannelOperatingState.DISCHARGING,
            40: ChannelOperatingState.STARTING,
            41: ChannelOperatingState.PAUSED,
            51: ChannelOperatingState.TRANSITIONAL,
            82: ChannelOperatingState.PAUSED,
            89: ChannelOperatingState.PAUSED,
            99: ChannelOperatingState.ERROR,
            120: ChannelOperatingState.CHARGING,
            159: ChannelOperatingState.CHARGING,
            220: ChannelOperatingState.DISCHARGING,
            259: ChannelOperatingState.DISCHARGING,
            999: ChannelOperatingState.UNKNOWN,
        }
        for code, state in expected.items():
            with self.subTest(code=code):
                self.assertIs(ChannelOperatingState.from_hrt_code(code), state)

    def test_terminal_event_codes(self):
        self.assertIs(TerminationReason.from_hrt_event(30), TerminationReason.COMPLETED)
        self.assertIs(TerminationReason.from_hrt_event(104), TerminationReason.SAFETY_LIMIT)
        self.assertIs(TerminationReason.from_hrt_event(140), TerminationReason.VOLTAGE_MISMATCH)
        self.assertIs(TerminationReason.from_hrt_event(150), TerminationReason.STEP_ERROR)
        self.assertIs(TerminationReason.from_hrt_event(190), TerminationReason.STOPPED)
        self.assertIs(TerminationReason.from_hrt_event(199), TerminationReason.GENERAL_ERROR)
        self.assertIs(TerminationReason.from_hrt_event(191), TerminationReason.UNKNOWN)


class ChannelSnapshotTest(unittest.TestCase):

    def setUp(self):
        self.database = {
            'channelID': 13,
            'experimentID': 48,
            'taskID': 47,
            'taskName': 'test-task',
            'protocolName': 'protocol.json',
            'protocolVersion': '2.2',
            'batteryModel': 'cell-model',
            'batteryNumber': '007',
            'queue': [{
                'id': 47,
                'channelID': 13,
                'position': 1,
                'taskName': 'test-task',
                'project': 'project',
                'batteryID': 6,
                'batteryModel': 'cell-model',
                'batteryNumber': '007',
                'batteryC_Ah': 0.01,
                'batteryMass_g': 2.5,
                'batteryVMin_V': 2.7,
                'batteryVMax_V': 3.7,
                'batteryIMin_A': 0.01,
                'batteryIMax_A': 0.02,
                'batteryTMax_dC': 55,
                'protocolID': 8,
                'protocolName': 'protocol.json',
                'protocolVersion': '2.2',
                'protocolGlobals': '{}',
                'protocolSteps': '[]',
                'optionBits': 0,
                'comment': 'qudi-run:123',
            }],
        }

    def test_live_values_are_converted_to_si(self):
        live = {
            'channelID': 13,
            'controllerID': 13,
            'live_status': 120,
            'lastLiveExpID': 51,
            'running': True,
            'live_uptime': 12.5,
            'protocolStep': 2,
            'protocolStepTime': 3.5,
            'loopCounter': 4,
            'live_voltage': 3712.5,
            'live_voltage_aux': -20.0,
            'live_voltage_terminals': 3699.0,
            'live_current': 125.0,
            'live_temp': 24.25,
            'live_temp2': 25.5,
            'taskID': 47,
            'taskName': 'test-task',
        }
        snapshot = channel_snapshot_from_hrt_payloads(self.database, live)

        self.assertIs(snapshot.state, ChannelOperatingState.CHARGING)
        self.assertTrue(snapshot.running)
        self.assertTrue(snapshot.live_data_available)
        self.assertEqual(snapshot.experiment_id, 51)
        self.assertAlmostEqual(snapshot.voltage_v, 3.7125)
        self.assertAlmostEqual(snapshot.voltage_aux_v, -0.020)
        self.assertAlmostEqual(snapshot.voltage_terminals_v, 3.699)
        self.assertAlmostEqual(snapshot.current_a, 0.125)
        self.assertAlmostEqual(snapshot.temperature_c, 24.25)
        self.assertEqual(len(snapshot.queue), 1)
        self.assertEqual(snapshot.queue[0].safety_limits.maximum_temperature_c, 55.0)

    def test_stale_experiment_ids_are_not_reported_as_running(self):
        live = {
            'channelID': 13,
            'controllerID': 13,
            'live_status': 0,
            'lastLiveExpID': 51,
            'running': False,
            'taskID': 47,
            'taskName': 'old-task',
        }
        snapshot = channel_snapshot_from_hrt_payloads(self.database, live)

        self.assertFalse(snapshot.running)
        self.assertIsNone(snapshot.experiment_id)
        self.assertIsNone(snapshot.task_id)
        self.assertIsNone(snapshot.task_name)

    def test_overrange_values_are_not_returned_as_measurements(self):
        live = {
            'channelID': 13,
            'controllerID': 13,
            'live_status': 0,
            'lastLiveExpID': 0,
            'live_voltage': 99999.999,
            'live_current': 99999999000,
            'live_temp': -99999.999,
        }
        snapshot = channel_snapshot_from_hrt_payloads(self.database, live)

        self.assertIsNone(snapshot.voltage_v)
        self.assertIsNone(snapshot.current_a)
        self.assertIsNone(snapshot.temperature_c)
        self.assertEqual(
            snapshot.overrange_fields,
            ('voltage', 'current', 'temperature'),
        )

    def test_disconnected_temperature_sensor_sentinel_is_rejected(self):
        live = {
            'channelID': 13,
            'controllerID': 13,
            'live_status': 0,
            'lastLiveExpID': 0,
            'running': False,
            'live_temp': -99.0,
        }
        snapshot = channel_snapshot_from_hrt_payloads(self.database, live)

        self.assertIsNone(snapshot.temperature_c)
        self.assertIn('temperature', snapshot.overrange_fields)

    def test_missing_controller_means_unknown_live_state(self):
        live = {'channelID': 13, 'controllerID': None, 'live_status': 0}
        snapshot = channel_snapshot_from_hrt_payloads(self.database, live)

        self.assertFalse(snapshot.live_data_available)
        self.assertIs(snapshot.state, ChannelOperatingState.UNKNOWN)


class _JsonHandler(BaseHTTPRequestHandler):
    responses = {
        '/package.json': {'version': '3.1.8'},
        '/api/version': {'version': '1.3.6'},
        '/btdm/api/v1/system': {
            'system': {'systemID': 24116, 'systemName': 'BD24116'}
        },
        '/btdm/api/v1/channels': {
            'channels': [{'channelID': 13, 'queue': []}, {'channelID': 14, 'queue': []}]
        },
        '/btdm/api/v1/channels/13': {
            'channels': [{'channelID': 13, 'queue': []}]
        },
    }

    def do_GET(self):
        payload = self.responses.get(self.path)
        if payload is None:
            self.send_error(404)
            return
        encoded = json.dumps(payload).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format_, *args):
        pass


class HrtRestClientTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), _JsonHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f'http://127.0.0.1:{cls.server.server_port}/'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_read_only_endpoints(self):
        client = HrtRestClient(self.base_url, api_url=self.base_url, timeout_s=2)

        self.assertEqual(client.get_frontend_version(), '3.1.8')
        self.assertEqual(client.get_api_version(), '1.3.6')
        self.assertEqual(client.get_system()['systemID'], 24116)
        self.assertEqual([row['channelID'] for row in client.get_channels()], [13, 14])
        self.assertEqual(client.get_channel(13)['channelID'], 13)


class _FakeQudi:
    pass


class _BatteryTesterStub(BatteryTesterInterface):

    def __init__(self, *args, snapshots, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshots = tuple(snapshots)

    def on_activate(self):
        pass

    def on_deactivate(self):
        pass

    @property
    def capabilities(self):
        return BatteryTesterCapabilities(
            system_id=1,
            system_name='test',
            frontend_version='test',
            api_version='test',
            serial_handler_version=None,
            serial_handler_commit=None,
            channel_ids=tuple(snapshot.channel_id for snapshot in self.snapshots),
        )

    def get_channel_snapshots(self):
        return self.snapshots

    def get_channel_snapshot(self, channel_id):
        return next(snapshot for snapshot in self.snapshots if snapshot.channel_id == channel_id)


class BatteryTesterLogicTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.qt_app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    def test_polling_updates_aggregate_module_state(self):
        qudi = _FakeQudi()
        snapshot = channel_snapshot_from_hrt_payloads(
            {'channelID': 13, 'queue': []},
            {
                'channelID': 13,
                'controllerID': 13,
                'live_status': 0,
                'lastLiveExpID': 0,
                'running': False,
            },
        )
        tester = _BatteryTesterStub(
            weakref.ref(qudi), 'battery_tester_stub', snapshots=(snapshot,)
        )
        logic = BatteryTesterLogic(
            weakref.ref(qudi), 'battery_tester_logic', config={'poll_interval_s': 60}
        )
        try:
            tester.module_state.activate()
            logic._battery_tester.connect(tester)
            logic.module_state.activate()

            self.assertTrue(logic.connected)
            self.assertEqual(logic.module_state(), 'idle')
            self.assertEqual(logic.get_channel_snapshot(13).channel_id, 13)

            tester.snapshots = (
                replace(
                    snapshot,
                    state=ChannelOperatingState.CHARGING,
                    device_state_code=120,
                    running=True,
                    experiment_id=1,
                ),
            )
            logic.refresh_now()
            self.assertEqual(logic.module_state(), 'locked')

            tester.snapshots = (snapshot,)
            logic.refresh_now()
            self.assertEqual(logic.module_state(), 'idle')
        finally:
            if logic.module_state() in ('idle', 'locked'):
                logic.module_state.deactivate()
            if tester.module_state() in ('idle', 'locked'):
                tester.module_state.deactivate()


if __name__ == '__main__':
    unittest.main()
