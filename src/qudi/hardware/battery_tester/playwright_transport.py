# -*- coding: utf-8 -*-
"""Legacy Playwright discovery/UI fallback for HRT channel information.

This transport is not the production live-data path. Its synchronous worker can trigger a native
access violation while the Windows Qudi process exits; use ``SocketIoHrtTransport`` for unattended
monitoring and isolate any future browser-only control workflow in a subprocess.
"""

import json
import shutil
from concurrent.futures import Future
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any, Dict, List, Mapping, Optional
from uuid import uuid4

from qudi.interface.battery_tester_interface import (
    BatteryTesterConnectionError,
    BatteryTesterProtocolError,
)

try:
    from playwright.sync_api import Browser, Page, Playwright, sync_playwright
except ImportError:
    Browser = Page = Playwright = Any
    sync_playwright = None


_CHANNEL_SNAPSHOT_SCRIPT = """() => {
    const store = Ext.getStore('ChannelsStore');
    if (!store) throw new Error('ChannelsStore is not available');
    return store.getData().items.map(record => {
        const d = record.data;
        return {
            channelID: d.channelID,
            controllerID: d.controllerID,
            live_status: d.live_status,
            lastLiveExpID: d.lastLiveExpID,
            live_uptime: d.live_uptime,
            protocolStep: d.protocolStep,
            protocolStepTime: d.protocolStepTime,
            loopCounter: d.loopCounter,
            live_voltage: d.live_voltage,
            live_voltage_aux: d.live_voltage_aux,
            live_voltage_terminals: d.live_voltage_terminals,
            live_current: d.live_current,
            live_temp: d.live_temp,
            live_temp2: d.live_temp2,
            taskID: d.taskID,
            taskName: d.taskName,
            experimentID: d.experimentID,
            running: d.running,
            queueCount: record.tasks().count()
        };
    });
}"""


@dataclass
class _Command:
    operation: str
    future: Future


class PlaywrightHrtTransport:
    """Own all Playwright objects in one worker thread for explicit discovery use.

    Qudi hardware methods can be invoked from a logic-module thread. This command queue ensures
    that Playwright's thread-affine browser objects never leave the worker that created them.
    It does not provide process isolation and is therefore not suitable for unattended Windows
    Qudi operation on the commissioned deployment.
    """

    def __init__(
            self,
            ui_url: str,
            browser_channel: Optional[str] = 'chrome',
            headless: bool = True,
            navigation_timeout_s: float = 30.0,
            command_timeout_s: float = 10.0,
            diagnostics_dir: Optional[str] = None,
            max_diagnostic_sets: int = 10,
    ) -> None:
        self._ui_url = ui_url
        self._browser_channel = browser_channel or None
        self._headless = bool(headless)
        self._navigation_timeout_s = float(navigation_timeout_s)
        self._command_timeout_s = float(command_timeout_s)
        self._diagnostics_dir = (
            None if diagnostics_dir is None else Path(diagnostics_dir).expanduser().resolve()
        )
        self._max_diagnostic_sets = int(max_diagnostic_sets)
        if self._navigation_timeout_s <= 0 or self._command_timeout_s <= 0:
            raise ValueError('Playwright timeouts must be positive.')
        if self._max_diagnostic_sets <= 0:
            raise ValueError('max_diagnostic_sets must be positive.')

        self._commands: Queue = Queue()
        self._startup: Future = Future()
        self._thread: Optional[Thread] = None
        self._closed = Event()
        self._console_messages = deque(maxlen=100)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._closed.is_set()

    def start(self) -> None:
        if sync_playwright is None:
            raise ImportError(
                'Playwright is required for HRT live data. Install the battery-tester extra or '
                'run: python -m pip install "playwright>=1.40,<2"'
            )
        if self._thread is not None:
            raise RuntimeError('Playwright HRT transport has already been started.')
        self._thread = Thread(target=self._worker_main, name='HrtPlaywrightWorker', daemon=True)
        self._thread.start()
        try:
            self._startup.result(timeout=self._navigation_timeout_s + 5.0)
        except Exception:
            self.close()
            raise

    def get_live_channels(self) -> List[Mapping[str, Any]]:
        result = self._call('get_live_channels')
        if not isinstance(result, list) or not all(isinstance(row, Mapping) for row in result):
            raise BatteryTesterProtocolError('HRT frontend returned invalid live channel data.')
        return [dict(row) for row in result]

    def close(self) -> None:
        thread = self._thread
        if thread is None:
            return
        if thread.is_alive():
            future = Future()
            self._commands.put(_Command('close', future))
            try:
                future.result(timeout=self._command_timeout_s)
            except Exception:
                pass
            thread.join(timeout=self._command_timeout_s)
        self._closed.set()
        self._thread = None

    def _call(self, operation: str) -> Any:
        if not self.is_running:
            raise BatteryTesterConnectionError('HRT Playwright worker is not running.')
        future = Future()
        self._commands.put(_Command(operation, future))
        try:
            return future.result(timeout=self._command_timeout_s)
        except TimeoutError as exc:
            raise BatteryTesterConnectionError(
                f'HRT Playwright operation {operation!r} timed out.'
            ) from exc

    def _worker_main(self) -> None:
        browser = None
        close_future = None
        try:
            with sync_playwright() as playwright:
                try:
                    browser, page = self._open_browser(playwright)
                    self._startup.set_result(True)
                    while True:
                        command = self._commands.get()
                        if command.operation == 'close':
                            close_future = command.future
                            break
                        try:
                            if command.operation == 'get_live_channels':
                                result = page.evaluate(_CHANNEL_SNAPSHOT_SCRIPT)
                            else:
                                raise ValueError(
                                    f'Unknown HRT browser operation: {command.operation}'
                                )
                        except Exception as exc:
                            diagnostic_path = self._capture_diagnostics(
                                page, command.operation, exc
                            )
                            suffix = (
                                f' Diagnostics: {diagnostic_path}'
                                if diagnostic_path is not None else ''
                            )
                            command.future.set_exception(
                                BatteryTesterConnectionError(
                                    f'HRT browser operation {command.operation!r} failed: '
                                    f'{exc}{suffix}'
                                )
                            )
                        else:
                            command.future.set_result(result)
                finally:
                    if browser is not None:
                        try:
                            browser.close()
                        except Exception:
                            pass
            if close_future is not None and not close_future.done():
                close_future.set_result(True)
        except Exception as exc:
            if not self._startup.done():
                self._startup.set_exception(
                    BatteryTesterConnectionError(f'Unable to start HRT browser worker: {exc}')
                )
            if close_future is not None and not close_future.done():
                close_future.set_exception(
                    BatteryTesterConnectionError(f'Unable to close HRT browser worker: {exc}')
                )
            self._fail_pending_commands(exc)
        finally:
            self._closed.set()

    def _open_browser(self, playwright: Playwright):
        launch_kwargs = {'headless': self._headless}
        if self._browser_channel is not None:
            launch_kwargs['channel'] = self._browser_channel
        browser: Browser = playwright.chromium.launch(**launch_kwargs)
        page: Page = browser.new_page()
        page.on('console', self._on_console_message)
        page.on('pageerror', lambda error: self._console_messages.append(f'pageerror: {error}'))
        timeout_ms = int(round(self._navigation_timeout_s * 1000))
        page.set_default_timeout(timeout_ms)
        try:
            page.goto(self._ui_url, wait_until='domcontentloaded', timeout=timeout_ms)
            page.wait_for_function(
                """() => window.Ext && Ext.getStore && Ext.getStore('ChannelsStore') &&
                          Ext.getStore('ChannelsStore').getCount() > 0""",
                timeout=timeout_ms,
            )
            page.wait_for_function(
                """() => Ext.getStore('ChannelsStore').getData().items.some(
                          record => record.data.controllerID !== undefined &&
                                    record.data.controllerID !== null)""",
                timeout=timeout_ms,
            )
            # A live update can arrive as multiple channel events. Allow one complete update
            # interval before exposing the initial snapshot; subsequent reads do not wait.
            page.wait_for_timeout(1200)
        except Exception as exc:
            self._capture_diagnostics(page, 'startup', exc)
            raise
        return browser, page

    def _fail_pending_commands(self, cause: Exception) -> None:
        while not self._commands.empty():
            command = self._commands.get_nowait()
            if not command.future.done():
                command.future.set_exception(
                    BatteryTesterConnectionError(f'HRT browser worker stopped: {cause}')
                )

    def _on_console_message(self, message) -> None:
        if message.type in ('error', 'warning'):
            self._console_messages.append(f'{message.type}: {message.text}')

    def _capture_diagnostics(
            self, page: Page, operation: str, cause: Exception
    ) -> Optional[str]:
        root = self._diagnostics_dir
        if root is None:
            return None
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        diagnostic_dir = root / f'hrt_diag_{timestamp}_{uuid4().hex[:8]}'
        try:
            diagnostic_dir.mkdir(parents=True, exist_ok=False)
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'operation': operation,
                'cause': repr(cause),
                'ui_url': self._ui_url,
                'current_url': page.url,
                'console_messages': tuple(self._console_messages),
            }
            try:
                page.screenshot(path=str(diagnostic_dir / 'page.png'), full_page=True)
            except Exception as screenshot_error:
                metadata['screenshot_error'] = repr(screenshot_error)
            try:
                (diagnostic_dir / 'page.html').write_text(
                    page.content(), encoding='utf-8'
                )
            except Exception as html_error:
                metadata['html_error'] = repr(html_error)
            (diagnostic_dir / 'metadata.json').write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            self._prune_diagnostics(root)
            return str(diagnostic_dir)
        except Exception:
            return None

    def _prune_diagnostics(self, root: Path) -> None:
        candidates = sorted(
            (path for path in root.glob('hrt_diag_*') if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
        )
        for path in candidates[:-self._max_diagnostic_sets]:
            if path.parent.resolve() == root.resolve():
                shutil.rmtree(str(path))
