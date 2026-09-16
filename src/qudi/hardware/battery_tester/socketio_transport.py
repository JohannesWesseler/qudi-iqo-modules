# -*- coding: utf-8 -*-
"""Native Socket.IO live-data transport for Battery Dynamics HRT testers."""

import time
from threading import Event, RLock
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse, urlunparse

from qudi.interface.battery_tester_interface import (
    BatteryTesterConnectionError,
    BatteryTesterProtocolError,
)

try:
    import socketio
except ImportError:
    socketio = None


class SocketIoHrtTransport:
    """Cache HRT ``new_system_data`` events without embedding a browser.

    The installed HRT frontend 3.1.8 uses Socket.IO 2.2.0 / Engine.IO 3. The matching Python
    client line is python-socketio 4.x with python-engineio 3.x. Emitting ``start`` and ``stop``
    only subscribes/unsubscribes this client from the live feed; it does not control a measurement.
    """

    def __init__(
            self,
            ui_url: str,
            socketio_url: Optional[str] = None,
            socketio_path: str = 'socket.io',
            connection_timeout_s: float = 10.0,
            max_live_age_s: float = 5.0,
            initial_settle_s: float = 0.2,
    ) -> None:
        self._service_url = _normalize_service_url(
            socketio_url or _derive_service_url(ui_url, 3000)
        )
        self._socketio_path = str(socketio_path).strip('/') or 'socket.io'
        self._connection_timeout_s = float(connection_timeout_s)
        self._max_live_age_s = float(max_live_age_s)
        self._initial_settle_s = float(initial_settle_s)
        if (
                self._connection_timeout_s <= 0
                or self._max_live_age_s <= 0
                or self._initial_settle_s < 0
        ):
            raise ValueError('Socket.IO timeouts must be positive and settle time non-negative.')

        self._lock = RLock()
        self._first_data = Event()
        self._connected = Event()
        self._client = None
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._updated_monotonic: Dict[int, float] = {}
        self._protocol_error: Optional[str] = None

    @property
    def is_running(self) -> bool:
        client = self._client
        return client is not None and bool(client.connected) and self._connected.is_set()

    def start(self) -> None:
        if socketio is None:
            raise ImportError(
                'Native HRT live data requires python-socketio 4.x and python-engineio 3.x. '
                'Install the battery-tester extra.'
            )
        if self._client is not None:
            raise RuntimeError('HRT Socket.IO transport has already been started.')

        client = socketio.Client(
            reconnection=True,
            reconnection_attempts=0,
            reconnection_delay=1,
            reconnection_delay_max=5,
            logger=False,
            engineio_logger=False,
            request_timeout=self._connection_timeout_s,
        )
        self._client = client

        @client.event
        def connect():
            self._connected.set()
            client.emit('start')

        @client.event
        def disconnect():
            self._connected.clear()

        @client.on('new_system_data')
        def new_system_data(payload):
            self._on_new_system_data(payload)

        try:
            client.connect(
                self._service_url,
                transports=['websocket'],
                socketio_path=self._socketio_path,
            )
            if not self._first_data.wait(timeout=self._connection_timeout_s):
                raise BatteryTesterConnectionError(
                    'Connected to HRT Socket.IO but received no new_system_data event.'
                )
            if self._initial_settle_s:
                time.sleep(self._initial_settle_s)
        except Exception as exc:
            self.close()
            if isinstance(exc, BatteryTesterConnectionError):
                raise
            raise BatteryTesterConnectionError(
                f'Unable to connect to HRT Socket.IO at {self._service_url}: {exc}'
            ) from exc

    def get_live_channels(self) -> List[Mapping[str, Any]]:
        if not self.is_running:
            raise BatteryTesterConnectionError('HRT Socket.IO live feed is disconnected.')
        with self._lock:
            if self._protocol_error is not None:
                raise BatteryTesterProtocolError(self._protocol_error)
            if not self._cache:
                raise BatteryTesterConnectionError('HRT Socket.IO live cache is empty.')
            now = time.monotonic()
            stale = [
                channel_id
                for channel_id, updated in self._updated_monotonic.items()
                if now - updated > self._max_live_age_s
            ]
            if stale:
                raise BatteryTesterConnectionError(
                    f'HRT Socket.IO live data is stale for channels {sorted(stale)}.'
                )
            return [dict(self._cache[channel_id]) for channel_id in sorted(self._cache)]

    def close(self) -> None:
        client = self._client
        self._client = None
        self._connected.clear()
        if client is None:
            return
        if client.connected:
            try:
                client.emit('stop')
            except Exception:
                pass
            try:
                client.disconnect()
            except Exception:
                pass

    def _on_new_system_data(self, payload: Any) -> None:
        rows = payload if isinstance(payload, list) else [payload]
        normalized = []
        try:
            for row in rows:
                if not isinstance(row, Mapping):
                    raise TypeError('event row is not an object')
                normalized.append(_normalize_live_row(row))
        except Exception as exc:
            with self._lock:
                self._protocol_error = f'Invalid HRT new_system_data payload: {exc}'
            self._first_data.set()
            return

        updated = time.monotonic()
        with self._lock:
            for row in normalized:
                channel_id = int(row['channelID'])
                self._cache[channel_id] = row
                self._updated_monotonic[channel_id] = updated
            self._protocol_error = None
        self._first_data.set()


def _normalize_live_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    channel_id = int(row['ch_id'])
    experiment_id = _optional_int(row.get('exp_id'))
    running = bool(experiment_id is not None and experiment_id > 0)
    return {
        'channelID': channel_id,
        'controllerID': _optional_int(row.get('controller_id')),
        'live_status': _optional_int(row.get('ch_status')),
        'lastLiveExpID': experiment_id,
        # Mirror ChannelsModel.updateLiveData(): controller uptime/step counters are not an
        # experiment runtime and must be cleared whenever no experiment is active.
        'live_uptime': row.get('time_s') if running else None,
        'protocolStep': row.get('protocol_step') if running else None,
        'protocolStepTime': row.get('stepTime_s') if running else None,
        'loopCounter': row.get('loop_counter') if running else None,
        'live_voltage': row.get('voltage'),
        'live_voltage_aux': row.get('voltage_aux'),
        'live_voltage_terminals': row.get('voltage_power_terminals'),
        'live_current': row.get('current'),
        'live_temp': row.get('temp'),
        'live_temp2': row.get('temp2'),
        'experimentID': experiment_id,
        'running': running,
    }


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None or value == '' else int(value)


def _normalize_service_url(url: str) -> str:
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError(f'Invalid Socket.IO HTTP(S) URL: {url!r}')
    return urlunparse((parsed.scheme, parsed.netloc, '/', '', '', ''))


def _derive_service_url(ui_url: str, port: int) -> str:
    parsed = urlparse(ui_url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError(f'Invalid HRT UI URL: {ui_url!r}')
    host = f'[{parsed.hostname}]' if ':' in parsed.hostname else parsed.hostname
    return urlunparse((parsed.scheme, f'{host}:{int(port)}', '/', '', '', ''))
