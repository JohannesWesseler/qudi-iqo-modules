# -*- coding: utf-8 -*-
"""Read-only client for the REST services used by the HRT web frontend."""

import json
import socket
from typing import Any, Dict, List, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

from qudi.interface.battery_tester_interface import (
    BatteryTesterConnectionError,
    BatteryTesterProtocolError,
)


class HrtRestClient:
    """Small, dependency-free reader for the HRT's unsupported internal REST API.

    The API is used by the vendor's own frontend. It is deliberately kept behind this private
    transport class because it is not a documented public device API and may change on upgrades.
    """

    def __init__(
            self,
            ui_url: str,
            api_url: Optional[str] = None,
            timeout_s: float = 10.0,
    ) -> None:
        self.ui_url = _normalize_base_url(ui_url)
        self.api_url = _normalize_base_url(api_url or _derive_service_url(ui_url, 8000))
        self.timeout_s = float(timeout_s)
        if self.timeout_s <= 0:
            raise ValueError('timeout_s must be positive')

    def get_frontend_version(self) -> Optional[str]:
        payload = self._get_json(self.ui_url, 'package.json')
        return _optional_version(payload)

    def get_api_version(self) -> Optional[str]:
        payload = self._get_json(self.api_url, 'api/version')
        return _optional_version(payload)

    def get_system(self) -> Mapping[str, Any]:
        payload = self._get_json(self.api_url, 'btdm/api/v1/system')
        system = payload.get('system')
        if not isinstance(system, Mapping):
            raise BatteryTesterProtocolError(
                'HRT system response does not contain an object named "system".'
            )
        return dict(system)

    def get_channels(self) -> List[Mapping[str, Any]]:
        payload = self._get_json(self.api_url, 'btdm/api/v1/channels')
        channels = payload.get('channels')
        if not isinstance(channels, list) or not all(isinstance(ch, Mapping) for ch in channels):
            raise BatteryTesterProtocolError(
                'HRT channels response does not contain a list named "channels".'
            )
        return [dict(channel) for channel in channels]

    def get_channel(self, channel_id: int) -> Mapping[str, Any]:
        channel_id = int(channel_id)
        payload = self._get_json(self.api_url, f'btdm/api/v1/channels/{channel_id}')
        channels = payload.get('channels')
        if not isinstance(channels, list) or len(channels) != 1:
            raise BatteryTesterProtocolError(
                f'HRT response for channel {channel_id} did not contain exactly one channel.'
            )
        channel = channels[0]
        if not isinstance(channel, Mapping) or int(channel.get('channelID', -1)) != channel_id:
            raise BatteryTesterProtocolError(
                f'HRT returned an unexpected record for requested channel {channel_id}.'
            )
        return dict(channel)

    def _get_json(self, base_url: str, path: str) -> Dict[str, Any]:
        url = urljoin(base_url, path)
        request = Request(
            url,
            headers={
                'Accept': 'application/json',
                'User-Agent': 'Qudi-HRT-BatteryTester/0.1',
                'Cache-Control': 'no-cache',
            },
            method='GET',
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                body = response.read()
        except (HTTPError, URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise BatteryTesterConnectionError(f'GET {url} failed: {exc}') from exc

        try:
            payload = json.loads(body.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BatteryTesterProtocolError(f'GET {url} returned invalid JSON.') from exc
        if not isinstance(payload, dict):
            raise BatteryTesterProtocolError(f'GET {url} did not return a JSON object.')
        return payload


def _normalize_base_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError('A non-empty absolute URL is required.')
    parsed = urlparse(url.strip())
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError(f'Invalid HTTP(S) URL: {url!r}')
    normalized = urlunparse((parsed.scheme, parsed.netloc, '/', '', '', ''))
    return normalized


def _derive_service_url(ui_url: str, port: int) -> str:
    parsed = urlparse(ui_url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError(f'Invalid HRT UI URL: {ui_url!r}')
    host = f'[{parsed.hostname}]' if ':' in parsed.hostname else parsed.hostname
    return urlunparse((parsed.scheme, f'{host}:{int(port)}', '/', '', '', ''))


def _optional_version(payload: Mapping[str, Any]) -> Optional[str]:
    version = payload.get('version')
    return None if version is None else str(version)

