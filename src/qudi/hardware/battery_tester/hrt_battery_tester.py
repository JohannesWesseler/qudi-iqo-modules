# -*- coding: utf-8 -*-
"""Read-only Qudi hardware module for Battery Dynamics HRT battery testers."""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from qudi.core.configoption import ConfigOption
from qudi.interface.battery_tester_interface import (
    BatteryTesterConnectionError,
    BatteryTesterInterface,
)
from qudi.interface.battery_tester_models import (
    BatteryTesterCapabilities,
    ChannelSnapshot,
    channel_snapshot_from_hrt_payloads,
)
from qudi.util.mutex import Mutex

from .hrt_rest_client import HrtRestClient
from .playwright_transport import PlaywrightHrtTransport
from .socketio_transport import SocketIoHrtTransport


_SUPPORTED_SOCKETIO_FRONTENDS = ('3.1.8',)


class HrtBatteryTester(BatteryTesterInterface):
    """Battery Dynamics HRT adapter using its internal REST and live-data services.

    This first implementation phase is deliberately read-only. It discovers the device, queued
    tasks, and live channel values. Mutating control operations inherited from
    :class:`BatteryTesterInterface` raise ``BatteryTesterControlUnavailableError``.

    Example configuration::

        hardware:
            hrt_battery_tester:
                module.Class: 'battery_tester.hrt_battery_tester.HrtBatteryTester'
                options:
                    base_url: 'http://10.1.24.116:1841/'
                    api_url: 'http://10.1.24.116:8000/'
                    live_transport: 'socketio'
                    allowed_channels: [13, 14, 15, 16]
    """

    _base_url = ConfigOption(name='base_url', missing='error')
    _api_url = ConfigOption(name='api_url', default=None, missing='nothing')
    _request_timeout_s = ConfigOption(name='request_timeout_s', default=10.0, missing='nothing')
    _live_transport = ConfigOption(name='live_transport', default='socketio', missing='nothing')
    _socketio_url = ConfigOption(name='socketio_url', default=None, missing='nothing')
    _socketio_path = ConfigOption(name='socketio_path', default='socket.io', missing='nothing')
    _max_live_age_s = ConfigOption(name='max_live_age_s', default=5.0, missing='nothing')
    _browser_channel = ConfigOption(name='browser_channel', default='chrome', missing='nothing')
    _headless = ConfigOption(name='headless', default=True, missing='nothing')
    _navigation_timeout_s = ConfigOption(
        name='navigation_timeout_s', default=30.0, missing='nothing'
    )
    _command_timeout_s = ConfigOption(name='command_timeout_s', default=10.0, missing='nothing')
    _allowed_channels = ConfigOption(name='allowed_channels', default=None, missing='nothing')
    _require_live_data = ConfigOption(name='require_live_data', default=True, missing='nothing')
    _reconnect_attempts = ConfigOption(
        name='reconnect_attempts', default=2, checker=lambda value: int(value) >= 0,
        missing='nothing'
    )
    _reconnect_backoff_s = ConfigOption(
        name='reconnect_backoff_s', default=0.5, checker=lambda value: float(value) >= 0,
        missing='nothing'
    )
    _reconnect_cooldown_s = ConfigOption(
        name='reconnect_cooldown_s', default=10.0, checker=lambda value: float(value) >= 0,
        missing='nothing'
    )
    _capture_diagnostics_on_error = ConfigOption(
        name='capture_diagnostics_on_error', default=True, missing='nothing'
    )
    _diagnostics_dir = ConfigOption(name='diagnostics_dir', default=None, missing='nothing')
    _max_diagnostic_sets = ConfigOption(
        name='max_diagnostic_sets', default=10, checker=lambda value: int(value) > 0,
        missing='nothing'
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_lock = Mutex()
        self._rest_client: Optional[HrtRestClient] = None
        self._live_data_transport = None
        self._capabilities: Optional[BatteryTesterCapabilities] = None
        self._live_retry_after_monotonic = 0.0

    def on_activate(self) -> None:
        """Connect to the read API and start the configured passive live-data transport."""
        rest_client = HrtRestClient(
            ui_url=self._base_url,
            api_url=self._api_url,
            timeout_s=float(self._request_timeout_s),
        )
        system = rest_client.get_system()
        database_channels = rest_client.get_channels()
        channel_ids = tuple(sorted(int(channel['channelID']) for channel in database_channels))
        if not channel_ids:
            raise BatteryTesterConnectionError('HRT returned no channels during activation.')

        configured_channels = self._validate_allowed_channels(channel_ids)
        frontend_version = rest_client.get_frontend_version()
        api_version = rest_client.get_api_version()
        transport_name = str(self._live_transport).strip().lower()
        if transport_name == 'socketio' and frontend_version not in _SUPPORTED_SOCKETIO_FRONTENDS:
            raise BatteryTesterConnectionError(
                f'HRT frontend {frontend_version!r} is not validated for native Socket.IO; '
                f'supported versions are {_SUPPORTED_SOCKETIO_FRONTENDS}. Use an explicitly '
                f'validated transport profile after an upgrade.'
            )
        capabilities = BatteryTesterCapabilities(
            system_id=_optional_int(system.get('systemID')),
            system_name=_optional_str(system.get('systemName')),
            frontend_version=frontend_version,
            api_version=api_version,
            serial_handler_version=_optional_str(system.get('serialHandlerVersion')),
            serial_handler_commit=_optional_str(system.get('serialHandlerCommitHash')),
            channel_ids=configured_channels,
            task_control_available=False,
            live_transport=transport_name,
        )

        live_transport = self._new_live_transport()
        try:
            live_transport.start()
            live_ids = {
                int(payload['channelID']) for payload in live_transport.get_live_channels()
            }
            missing_live_records = set(configured_channels).difference(live_ids)
            if missing_live_records and self._require_live_data:
                raise BatteryTesterConnectionError(
                    'HRT frontend did not provide live records for configured channels: '
                    f'{sorted(missing_live_records)}'
                )
        except Exception:
            live_transport.close()
            if self._require_live_data:
                raise
            self.log.warning(
                'HRT live-data transport is unavailable. Channel state and values will be '
                'reported as unknown.',
                exc_info=True,
            )
            live_transport = None

        reported_count = _optional_int(system.get('channelCount'))
        if reported_count is not None and reported_count != len(channel_ids):
            self.log.warning(
                'HRT system metadata reports %d channels, but the channel API returned %d (%s). '
                'Using the explicit channel records.',
                reported_count,
                len(channel_ids),
                ', '.join(str(channel_id) for channel_id in channel_ids),
            )

        self._rest_client = rest_client
        self._live_data_transport = live_transport
        self._live_retry_after_monotonic = 0.0
        self._capabilities = capabilities
        self.log.info(
            'Connected read-only to HRT %s (system ID %s), frontend %s, API %s, '
            'live transport %s, channels %s.',
            capabilities.system_name or '<unknown>',
            capabilities.system_id,
            capabilities.frontend_version or '<unknown>',
            capabilities.api_version or '<unknown>',
            transport_name,
            ', '.join(str(channel_id) for channel_id in capabilities.channel_ids),
        )

    def on_deactivate(self) -> None:
        """Close live subscriptions without changing any tester task or channel state."""
        live_transport = self._live_data_transport
        self._live_data_transport = None
        self._rest_client = None
        self._capabilities = None
        self._live_retry_after_monotonic = 0.0
        if live_transport is not None:
            live_transport.close()

    @property
    def capabilities(self) -> BatteryTesterCapabilities:
        capabilities = self._capabilities
        if capabilities is None:
            raise BatteryTesterConnectionError('HRT hardware module is not active.')
        return capabilities

    def get_channel_snapshots(self) -> Tuple[ChannelSnapshot, ...]:
        """Read queue metadata over REST and live measurements from the live-data service."""
        with self._thread_lock:
            rest_client = self._rest_client
            if rest_client is None:
                raise BatteryTesterConnectionError('HRT hardware module is not active.')

            database_by_id = {
                int(payload['channelID']): payload
                for payload in self._read_with_retries(
                    rest_client.get_channels, 'read HRT channel records'
                )
            }
            live_by_id: Dict[int, dict] = {}
            try:
                live_rows = self._read_live_channels_with_reconnect()
                live_by_id = {
                    int(payload['channelID']): dict(payload)
                    for payload in live_rows
                }
            except Exception:
                if self._require_live_data:
                    raise
                self.log.warning(
                    'HRT live-data transport remains unavailable; reporting unknown live values.',
                    exc_info=True,
                )

            snapshots = []
            for channel_id in self.capabilities.channel_ids:
                try:
                    database = database_by_id[channel_id]
                except KeyError as exc:
                    raise BatteryTesterConnectionError(
                        f'Configured HRT channel {channel_id} disappeared from the channel API.'
                    ) from exc
                snapshots.append(
                    channel_snapshot_from_hrt_payloads(database, live_by_id.get(channel_id))
                )
            return tuple(snapshots)

    def get_channel_snapshot(self, channel_id: int) -> ChannelSnapshot:
        channel_id = int(channel_id)
        if channel_id not in self.capabilities.channel_ids:
            raise ValueError(
                f'HRT channel {channel_id} is not configured. Available channels: '
                f'{self.capabilities.channel_ids}'
            )
        for snapshot in self.get_channel_snapshots():
            if snapshot.channel_id == channel_id:
                return snapshot
        raise BatteryTesterConnectionError(f'HRT channel {channel_id} was not returned.')

    def _validate_allowed_channels(self, discovered: Tuple[int, ...]) -> Tuple[int, ...]:
        configured = self._allowed_channels
        if configured is None:
            return discovered
        if isinstance(configured, (str, bytes)):
            raise ValueError('allowed_channels must be a sequence of integer channel IDs.')
        requested = tuple(dict.fromkeys(int(channel_id) for channel_id in configured))
        if not requested:
            raise ValueError('allowed_channels must not be empty when configured.')
        unknown = set(requested).difference(discovered)
        if unknown:
            raise ValueError(
                f'Unknown configured HRT channels {sorted(unknown)}; discovered {discovered}.'
            )
        return requested

    def _new_live_transport(self):
        transport_name = str(self._live_transport).strip().lower()
        if transport_name == 'socketio':
            return SocketIoHrtTransport(
                ui_url=self._base_url,
                socketio_url=self._socketio_url,
                socketio_path=self._socketio_path,
                connection_timeout_s=float(self._command_timeout_s),
                max_live_age_s=float(self._max_live_age_s),
            )
        if transport_name != 'playwright':
            raise ValueError(
                f'Unknown HRT live_transport {self._live_transport!r}; expected '
                f'"socketio" or "playwright".'
            )
        diagnostics_dir = None
        if self._capture_diagnostics_on_error:
            configured = self._diagnostics_dir
            if configured:
                diagnostics_dir = str(Path(str(configured)).expanduser().resolve())
            else:
                try:
                    diagnostics_dir = str(
                        (Path(self.module_default_data_dir) / 'diagnostics').resolve()
                    )
                except Exception:
                    self.log.warning(
                        'Unable to resolve the HRT diagnostics directory; browser diagnostics '
                        'are disabled for this module instance.'
                    )
        return PlaywrightHrtTransport(
            ui_url=self._base_url,
            browser_channel=self._browser_channel,
            headless=bool(self._headless),
            navigation_timeout_s=float(self._navigation_timeout_s),
            command_timeout_s=float(self._command_timeout_s),
            diagnostics_dir=diagnostics_dir,
            max_diagnostic_sets=int(self._max_diagnostic_sets),
        )

    def _read_live_channels_with_reconnect(self):
        transport = self._live_data_transport
        if transport is not None:
            try:
                return transport.get_live_channels()
            except Exception:
                self.log.warning(
                    'HRT live-data read failed; rebuilding the transport.', exc_info=True
                )
                transport.close()
                self._live_data_transport = None

        now = time.monotonic()
        if now < self._live_retry_after_monotonic:
            raise BatteryTesterConnectionError(
                'HRT live-data reconnect is in its cooldown interval.'
            )

        last_error = None
        attempts = int(self._reconnect_attempts) + 1
        for attempt in range(attempts):
            replacement = self._new_live_transport()
            try:
                replacement.start()
                rows = replacement.get_live_channels()
            except Exception as exc:
                last_error = exc
                replacement.close()
                if attempt + 1 < attempts and float(self._reconnect_backoff_s) > 0:
                    time.sleep(float(self._reconnect_backoff_s))
            else:
                self._live_data_transport = replacement
                self._live_retry_after_monotonic = 0.0
                self.log.info('Reconnected the HRT live-data transport successfully.')
                return rows

        self._live_retry_after_monotonic = (
            time.monotonic() + float(self._reconnect_cooldown_s)
        )
        raise BatteryTesterConnectionError(
            f'Unable to reconnect the HRT live-data transport after {attempts} attempt(s): '
            f'{last_error}'
        ) from last_error

    def _read_with_retries(self, operation, description: str):
        last_error = None
        attempts = int(self._reconnect_attempts) + 1
        for attempt in range(attempts):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                if attempt + 1 < attempts and float(self._reconnect_backoff_s) > 0:
                    time.sleep(float(self._reconnect_backoff_s))
        raise BatteryTesterConnectionError(
            f'Unable to {description} after {attempts} attempt(s): {last_error}'
        ) from last_error


def _optional_int(value):
    return None if value is None or value == '' else int(value)


def _optional_str(value):
    return None if value is None else str(value)
