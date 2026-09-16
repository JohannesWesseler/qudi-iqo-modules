# -*- coding: utf-8 -*-
"""Qudi orchestration, passive monitoring, and safety logic for battery testers."""

import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from uuid import uuid4

from PySide2 import QtCore

from qudi.core.configoption import ConfigOption
from qudi.core.connector import Connector
from qudi.core.module import LogicBase
from qudi.interface.battery_tester_interface import BatteryTesterControlUnavailableError
from qudi.interface.battery_tester_models import (
    BatteryTesterCapabilities,
    BatteryMonitoringSession,
    ChannelOperatingState,
    ChannelSnapshot,
)
from qudi.interface.battery_tester_task import (
    BatteryTaskSpec,
    ExperimentReference,
    TaskReference,
    validate_task_for_channel,
)
from qudi.util.mutex import Mutex
from qudi.util.datastorage import TextDataStorage


_MONITOR_COLUMN_HEADERS = (
    'unix_time (s)',
    'elapsed_time (s)',
    'channel_id',
    'state',
    'device_state_code',
    'running',
    'live_data_available',
    'experiment_id',
    'task_id',
    'runtime (s)',
    'protocol_step',
    'protocol_step_time (s)',
    'loop_counter',
    'voltage (V)',
    'aux_voltage (V)',
    'terminal_voltage (V)',
    'current (A)',
    'temperature (degC)',
    'temperature_2 (degC)',
    'overrange_mask',
    'queue_length',
)
_MONITOR_COLUMN_DTYPES = (
    float, float, int, str, int, int, int, int, int, float, int, float, int,
    float, float, float, float, float, float, int, int,
)
_MONITOR_COLUMN_FORMATS = (
    '.6f', '.6f', 'd', 's', 'd', 'd', 'd', 'd', 'd', '.6f', 'd', '.6f', 'd',
    '.12g', '.12g', '.12g', '.12g', '.9g', '.9g', 'd', 'd',
)
_OVERRANGE_BITS = {
    'voltage': 1,
    'voltage_aux': 2,
    'voltage_terminals': 4,
    'current': 8,
    'temperature': 16,
    'temperature_2': 32,
}


class BatteryTesterLogic(LogicBase):
    """Poll and expose battery tester state without coupling clients to web automation.

    The logic validates live channel state and battery safety limits before delegating mutations.
    The HRT hardware adapter still advertises control as unavailable; these methods are therefore
    executable with the dummy tester but fail closed with the live tester.
    """

    _battery_tester = Connector(interface='BatteryTesterInterface', name='battery_tester')
    _poll_interval_s = ConfigOption(
        name='poll_interval_s',
        default=1.0,
        checker=lambda value: float(value) > 0,
        missing='nothing',
    )
    _require_temperature_sensor = ConfigOption(
        name='require_temperature_sensor',
        default=True,
        missing='nothing',
    )
    _auto_start_monitoring = ConfigOption(
        name='auto_start_monitoring', default=False, missing='nothing'
    )
    _monitor_channel_ids = ConfigOption(
        name='monitor_channel_ids', default=None, missing='nothing'
    )
    _monitor_flush_rows = ConfigOption(
        name='monitor_flush_rows',
        default=100,
        checker=lambda value: int(value) > 0,
        missing='nothing',
    )
    _monitor_flush_interval_s = ConfigOption(
        name='monitor_flush_interval_s',
        default=5.0,
        checker=lambda value: float(value) > 0,
        missing='nothing',
    )
    _monitor_max_buffer_rows = ConfigOption(
        name='monitor_max_buffer_rows',
        default=10000,
        checker=lambda value: int(value) > 0,
        missing='nothing',
    )
    _monitor_save_thumbnail = ConfigOption(
        name='monitor_save_thumbnail', default=True, missing='nothing'
    )

    sigConnectionStateChanged = QtCore.Signal(bool, str)
    sigChannelStateChanged = QtCore.Signal(int, object)
    sigChannelsChanged = QtCore.Signal(object)
    sigTaskSubmitted = QtCore.Signal(object)
    sigExperimentStarted = QtCore.Signal(object)
    sigExperimentFinished = QtCore.Signal(object)
    sigMonitoringStateChanged = QtCore.Signal(bool, str)
    sigMonitoringDataAppended = QtCore.Signal(int)
    sigMonitoringPlotSaved = QtCore.Signal(str)
    sigError = QtCore.Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_lock = Mutex()
        self._poll_timer: Optional[QtCore.QTimer] = None
        self._capabilities: Optional[BatteryTesterCapabilities] = None
        self._snapshots: Tuple[ChannelSnapshot, ...] = ()
        self._connected = False
        self._connection_message = 'Not activated'
        self._owned_tasks: Dict[Tuple[int, int], TaskReference] = {}
        self._owned_experiments: Dict[int, ExperimentReference] = {}
        self._monitor_storage: Optional[TextDataStorage] = None
        self._monitor_file_path: Optional[str] = None
        self._monitor_plot_file_path: Optional[str] = None
        self._monitor_session_id: Optional[str] = None
        self._monitor_started_at: Optional[datetime] = None
        self._monitor_stopped_at: Optional[datetime] = None
        self._monitor_started_monotonic = 0.0
        self._monitor_channel_selection: Tuple[int, ...] = ()
        self._monitor_buffer = []
        self._monitor_rows_written = 0
        self._monitor_dropped_rows = 0
        self._monitor_connection_error_rows = 0
        self._monitor_last_flush_monotonic = 0.0
        self._monitoring_active = False

    def on_activate(self) -> None:
        tester = self._battery_tester()
        self._capabilities = tester.capabilities

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setSingleShot(False)
        self._poll_timer.setInterval(max(1, int(round(float(self._poll_interval_s) * 1000))))
        self._poll_timer.timeout.connect(self.refresh, QtCore.Qt.QueuedConnection)

        self._refresh(raise_on_error=True)
        self._poll_timer.start()
        if self._auto_start_monitoring:
            self.start_monitoring(channel_ids=self._monitor_channel_ids)

    def on_deactivate(self) -> None:
        timer = self._poll_timer
        self._poll_timer = None
        if timer is not None:
            timer.stop()
            timer.timeout.disconnect()
        if self._monitoring_active:
            try:
                self.stop_monitoring()
            except Exception:
                self.log.exception('Unable to flush battery monitoring data during deactivation:')
        self._snapshots = ()
        self._capabilities = None
        self._owned_tasks = {}
        self._owned_experiments = {}
        self._set_connection_state(False, 'Deactivated')

    @property
    def capabilities(self) -> BatteryTesterCapabilities:
        capabilities = self._capabilities
        if capabilities is None:
            raise RuntimeError('Battery tester logic is not active.')
        return capabilities

    @property
    def channel_snapshots(self) -> Tuple[ChannelSnapshot, ...]:
        with self._thread_lock:
            return self._snapshots

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def connection_message(self) -> str:
        return self._connection_message

    @property
    def monitoring_active(self) -> bool:
        return self._monitoring_active

    @property
    def monitoring_session(self) -> Optional[BatteryMonitoringSession]:
        if self._monitor_session_id is None:
            return None
        return BatteryMonitoringSession(
            session_id=self._monitor_session_id,
            started_at=self._monitor_started_at,
            stopped_at=self._monitor_stopped_at,
            channel_ids=self._monitor_channel_selection,
            file_path=self._monitor_file_path,
            rows_written=self._monitor_rows_written,
            buffered_rows=len(self._monitor_buffer),
            dropped_rows=self._monitor_dropped_rows,
            connection_error_rows=self._monitor_connection_error_rows,
            active=self._monitoring_active,
            plot_file_path=self._monitor_plot_file_path,
        )

    def get_channel_snapshot(self, channel_id: int) -> ChannelSnapshot:
        channel_id = int(channel_id)
        with self._thread_lock:
            for snapshot in self._snapshots:
                if snapshot.channel_id == channel_id:
                    return snapshot
        raise ValueError(f'No snapshot is available for battery tester channel {channel_id}.')

    @QtCore.Slot()
    def refresh(self) -> None:
        """Poll all channels once; errors are signalled and retried on the next timer tick."""
        self._refresh(raise_on_error=False)

    def refresh_now(self) -> Tuple[ChannelSnapshot, ...]:
        """Synchronously refresh and return all channel snapshots.

        This is intended for scripts and tests. GUI clients should normally consume the signals.
        """
        self._refresh(raise_on_error=True)
        return self.channel_snapshots

    def start_monitoring(
            self,
            channel_ids=None,
            nametag: Optional[str] = None,
            notes: Optional[str] = None,
            metadata: Optional[dict] = None,
    ) -> BatteryMonitoringSession:
        """Create a Qudi data file and begin appending passive channel snapshots."""
        if self._monitoring_active:
            raise RuntimeError('Battery monitoring is already active.')
        if self._monitor_buffer:
            raise RuntimeError(
                'Buffered rows from the previous monitoring session remain unsaved. '
                'Call flush_monitoring() before starting a new session.'
            )
        available = self.capabilities.channel_ids
        selected = available if channel_ids is None else tuple(
            dict.fromkeys(int(channel_id) for channel_id in channel_ids)
        )
        if not selected:
            raise ValueError('At least one monitoring channel must be selected.')
        unknown = set(selected).difference(available)
        if unknown:
            raise ValueError(
                f'Unknown monitoring channels {sorted(unknown)}; available channels are '
                f'{available}.'
            )

        started_at = datetime.now()
        session_id = str(uuid4())
        file_metadata = dict(metadata or {})
        file_metadata.update({
            'schema': 'qudi-battery-monitor-v1',
            'session_id': session_id,
            'system_id': self.capabilities.system_id,
            'system_name': self.capabilities.system_name,
            'frontend_version': self.capabilities.frontend_version,
            'api_version': self.capabilities.api_version,
            'serial_handler_version': self.capabilities.serial_handler_version,
            'serial_handler_commit': self.capabilities.serial_handler_commit,
            'live_transport': self.capabilities.live_transport,
            'channel_ids': selected,
            'poll_interval_s': float(self._poll_interval_s),
            'missing_float_value': 'nan',
            'missing_integer_value': -1,
            'overrange_bits': tuple(sorted(_OVERRANGE_BITS.items())),
            'read_only_hardware': not self.capabilities.task_control_available,
        })
        storage = TextDataStorage(
            root_dir=self.module_default_data_dir,
            delimiter='\t',
            file_extension='.dat',
            column_formats=_MONITOR_COLUMN_FORMATS,
        )
        tag = 'battery_monitor'
        if nametag:
            suffix = re.sub(r'[^A-Za-z0-9_-]+', '_', str(nametag)).strip('_')
            if suffix:
                tag = f'{tag}_{suffix}'
        file_path, _ = storage.new_file(
            timestamp=started_at,
            metadata=file_metadata,
            notes=notes,
            nametag=tag,
            column_headers=_MONITOR_COLUMN_HEADERS,
            column_dtypes=_MONITOR_COLUMN_DTYPES,
        )

        self._monitor_storage = storage
        self._monitor_file_path = file_path
        self._monitor_plot_file_path = None
        self._monitor_session_id = session_id
        self._monitor_started_at = started_at
        self._monitor_stopped_at = None
        self._monitor_started_monotonic = time.monotonic()
        self._monitor_channel_selection = selected
        self._monitor_rows_written = 0
        self._monitor_dropped_rows = 0
        self._monitor_connection_error_rows = 0
        self._monitor_last_flush_monotonic = self._monitor_started_monotonic
        self._monitoring_active = True

        current = self.channel_snapshots
        if current:
            self._record_monitor_snapshots(current)
        self.sigMonitoringStateChanged.emit(True, file_path)
        self.log.info(
            'Started passive battery monitoring session %s for channels %s: %s',
            session_id,
            selected,
            file_path,
        )
        return self.monitoring_session

    def flush_monitoring(self) -> int:
        """Synchronously append all buffered monitoring rows and return the written count."""
        if self._monitor_storage is None or self._monitor_file_path is None:
            return 0
        if not self._monitor_buffer:
            self._monitor_last_flush_monotonic = time.monotonic()
            return 0
        rows = tuple(self._monitor_buffer)
        result = self._monitor_storage.append_file(rows, self._monitor_file_path)
        rows_written = int(result[0])
        if rows_written != len(rows):
            raise IOError(
                f'Battery monitor expected to append {len(rows)} rows, wrote {rows_written}.'
            )
        del self._monitor_buffer[:rows_written]
        self._monitor_rows_written += rows_written
        self._monitor_last_flush_monotonic = time.monotonic()
        self.sigMonitoringDataAppended.emit(rows_written)
        return rows_written

    def stop_monitoring(self) -> BatteryMonitoringSession:
        """Stop passive monitoring, flush the data, and optionally save a time-trace plot."""
        if not self._monitoring_active:
            session = self.monitoring_session
            if session is None:
                raise RuntimeError('No battery monitoring session has been started.')
            return session
        self._monitoring_active = False
        self._monitor_stopped_at = datetime.now()
        try:
            self.flush_monitoring()
            if self._monitor_save_thumbnail:
                try:
                    self.save_monitoring_plot()
                except Exception as exc:
                    message = f'Unable to save battery monitoring plot: {exc}'
                    self.log.warning(message, exc_info=True)
                    self.sigError.emit(message)
        finally:
            file_path = self._monitor_file_path or ''
            self.sigMonitoringStateChanged.emit(False, file_path)
            self.log.info('Stopped passive battery monitoring: %s', file_path)
        return self.monitoring_session

    def save_monitoring_plot(self) -> str:
        """Save voltage/current traces in an isolated, non-Qt plotting process."""
        if self._monitor_storage is None or self._monitor_file_path is None:
            raise RuntimeError('No battery monitoring session has been started.')
        if self._monitoring_active:
            self.flush_monitoring()

        plot_script = (
            Path(__file__).parents[1]
            / 'hardware'
            / 'battery_tester'
            / 'monitor_plot.py'
        )
        environment = os.environ.copy()
        environment['MPLBACKEND'] = 'Agg'
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        completed = subprocess.run(
            [
                sys.executable,
                str(plot_script),
                self._monitor_file_path,
                '--title',
                self.capabilities.system_name or 'Battery tester',
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
            creationflags=creation_flags,
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f'isolated plot process exited with code {completed.returncode}: {detail}'
            )
        plot_path = completed.stdout.strip().splitlines()[-1]
        if not Path(plot_path).is_file():
            raise RuntimeError(f'isolated plot process did not create {plot_path!r}.')
        self._monitor_plot_file_path = plot_path
        self.sigMonitoringPlotSaved.emit(plot_path)
        self.log.info('Saved battery monitoring plot: %s', plot_path)
        return plot_path

    def prepare_task(self, spec: BatteryTaskSpec) -> ChannelSnapshot:
        """Validate a task and current live channel state without changing the tester."""
        snapshots = self.refresh_now()
        try:
            snapshot = next(item for item in snapshots if item.channel_id == spec.channel_id)
        except StopIteration as exc:
            raise ValueError(f'No configured battery tester channel {spec.channel_id}.') from exc
        validate_task_for_channel(
            spec,
            self.capabilities,
            snapshot,
            require_temperature_sensor=bool(self._require_temperature_sensor),
        )
        return snapshot

    def submit_task(self, spec: BatteryTaskSpec) -> TaskReference:
        """Validate and idempotently submit a task to the configured hardware."""
        self._require_task_control()
        self.prepare_task(spec)
        reference = self._battery_tester().submit_task(spec)
        with self._thread_lock:
            self._owned_tasks[(reference.channel_id, reference.task_id)] = reference
        self._refresh(raise_on_error=True)
        self.sigTaskSubmitted.emit(reference)
        return reference

    def start_task(self, task: TaskReference) -> ExperimentReference:
        """Start a Qudi-owned task only when it is first in an idle channel queue."""
        self._require_task_control()
        snapshot = self._refresh_and_get_channel(task.channel_id)
        with self._thread_lock:
            owned = self._owned_tasks.get((task.channel_id, task.task_id))
            active = self._owned_experiments.get(task.channel_id)
        if owned != task:
            raise BatteryTesterControlUnavailableError(
                f'Task {task.task_id} on channel {task.channel_id} is not owned by this Qudi run.'
            )
        if (
                snapshot.running
                and snapshot.task_id == task.task_id
                and active is not None
                and active.task_id == task.task_id
        ):
            return active
        if snapshot.running or snapshot.state is not ChannelOperatingState.STANDBY:
            raise BatteryTesterControlUnavailableError(
                f'Channel {task.channel_id} is not in standby ({snapshot.state.value}).'
            )
        if not snapshot.queue or snapshot.queue[0].task_id != task.task_id:
            raise BatteryTesterControlUnavailableError(
                f'Task {task.task_id} is not at the head of channel {task.channel_id} queue.'
            )
        experiment = self._battery_tester().start_task(task)
        with self._thread_lock:
            self._owned_experiments[experiment.channel_id] = experiment
        self._refresh(raise_on_error=True)
        self.sigExperimentStarted.emit(experiment)
        return experiment

    def pause_task(self, channel_id: int) -> None:
        self._require_owned_experiment(channel_id)
        snapshot = self._refresh_and_get_channel(channel_id)
        if snapshot.running and snapshot.state is ChannelOperatingState.PAUSED:
            return
        if not snapshot.running:
            raise BatteryTesterControlUnavailableError(
                f'Channel {channel_id} does not have a running, unpaused Qudi experiment.'
            )
        self._battery_tester().pause_task(int(channel_id))
        self._refresh(raise_on_error=True)

    def resume_task(self, channel_id: int) -> None:
        self._require_owned_experiment(channel_id)
        snapshot = self._refresh_and_get_channel(channel_id)
        if snapshot.running and snapshot.state is not ChannelOperatingState.PAUSED:
            return
        if not snapshot.running:
            raise BatteryTesterControlUnavailableError(
                f'Channel {channel_id} does not have a paused Qudi experiment.'
            )
        self._battery_tester().resume_task(int(channel_id))
        self._refresh(raise_on_error=True)

    def stop_task(self, channel_id: int) -> None:
        """Explicitly abort a Qudi-owned experiment; never touches unmanaged activity."""
        self._require_owned_experiment(channel_id)
        snapshot = self._refresh_and_get_channel(channel_id)
        if not snapshot.running:
            raise BatteryTesterControlUnavailableError(
                f'Channel {channel_id} does not have a running Qudi experiment.'
            )
        self._battery_tester().stop_task(int(channel_id))
        self._refresh(raise_on_error=True)

    def _refresh(self, raise_on_error: bool) -> None:
        try:
            new_snapshots = self._battery_tester().get_channel_snapshots()
            new_snapshots = tuple(new_snapshots)
        except Exception as exc:
            message = f'Battery tester polling failed: {exc}'
            self._set_connection_state(False, message)
            self.log.warning(message, exc_info=True)
            self.sigError.emit(message)
            self._record_monitor_error()
            if raise_on_error:
                raise
            return

        finished = []
        with self._thread_lock:
            old_by_id = {snapshot.channel_id: snapshot for snapshot in self._snapshots}
            self._snapshots = new_snapshots
            for snapshot in new_snapshots:
                old = old_by_id.get(snapshot.channel_id)
                if old is not None and old.running and not snapshot.running:
                    experiment = self._owned_experiments.pop(snapshot.channel_id, None)
                    if experiment is not None:
                        finished.append(experiment)

        self._set_connection_state(True, 'Connected')
        for snapshot in new_snapshots:
            if old_by_id.get(snapshot.channel_id) != snapshot:
                self.sigChannelStateChanged.emit(snapshot.channel_id, snapshot)
        self.sigChannelsChanged.emit(new_snapshots)
        self._update_module_state(new_snapshots)
        self._record_monitor_snapshots(new_snapshots)
        for experiment in finished:
            self.sigExperimentFinished.emit(experiment)

    def _set_connection_state(self, connected: bool, message: str) -> None:
        changed = connected != self._connected or message != self._connection_message
        self._connected = connected
        self._connection_message = message
        if changed:
            self.sigConnectionStateChanged.emit(connected, message)

    def _update_module_state(self, snapshots: Tuple[ChannelSnapshot, ...]) -> None:
        current_state = self.module_state()
        if current_state not in ('idle', 'locked'):
            return
        any_running = any(snapshot.running for snapshot in snapshots)
        if any_running and current_state == 'idle':
            self.module_state.lock()
        elif not any_running and current_state == 'locked':
            self.module_state.unlock()

    def _require_task_control(self) -> None:
        if not self.capabilities.task_control_available:
            raise BatteryTesterControlUnavailableError(
                'The connected battery tester hardware has task control disabled.'
            )

    def _require_owned_experiment(self, channel_id: int) -> ExperimentReference:
        self._require_task_control()
        channel_id = int(channel_id)
        with self._thread_lock:
            experiment = self._owned_experiments.get(channel_id)
        if experiment is None:
            raise BatteryTesterControlUnavailableError(
                f'Channel {channel_id} has no experiment owned by this Qudi logic instance.'
            )
        return experiment

    def _refresh_and_get_channel(self, channel_id: int) -> ChannelSnapshot:
        channel_id = int(channel_id)
        snapshots = self.refresh_now()
        try:
            return next(snapshot for snapshot in snapshots if snapshot.channel_id == channel_id)
        except StopIteration as exc:
            raise ValueError(f'No configured battery tester channel {channel_id}.') from exc

    def _record_monitor_snapshots(self, snapshots: Tuple[ChannelSnapshot, ...]) -> None:
        if not self._monitoring_active:
            return
        unix_time = time.time()
        elapsed = time.monotonic() - self._monitor_started_monotonic
        selected = set(self._monitor_channel_selection)
        for snapshot in snapshots:
            if snapshot.channel_id not in selected:
                continue
            overrange_mask = sum(
                _OVERRANGE_BITS.get(field, 0) for field in snapshot.overrange_fields
            )
            self._monitor_buffer.append((
                unix_time,
                elapsed,
                snapshot.channel_id,
                snapshot.state.value,
                _int_or_missing(snapshot.device_state_code),
                int(snapshot.running),
                int(snapshot.live_data_available),
                _int_or_missing(snapshot.experiment_id),
                _int_or_missing(snapshot.task_id),
                _float_or_nan(snapshot.runtime_s),
                _int_or_missing(snapshot.protocol_step),
                _float_or_nan(snapshot.protocol_step_time_s),
                _int_or_missing(snapshot.loop_counter),
                _float_or_nan(snapshot.voltage_v),
                _float_or_nan(snapshot.voltage_aux_v),
                _float_or_nan(snapshot.voltage_terminals_v),
                _float_or_nan(snapshot.current_a),
                _float_or_nan(snapshot.temperature_c),
                _float_or_nan(snapshot.temperature_2_c),
                overrange_mask,
                len(snapshot.queue),
            ))
        self._bound_monitor_buffer()
        self._maybe_flush_monitoring()

    def _record_monitor_error(self) -> None:
        if not self._monitoring_active:
            return
        unix_time = time.time()
        elapsed = time.monotonic() - self._monitor_started_monotonic
        self._monitor_buffer.append((
            unix_time, elapsed, -1, 'connection_error', -1, 0, 0, -1, -1,
            math.nan, -1, math.nan, -1, math.nan, math.nan, math.nan, math.nan,
            math.nan, math.nan, 0, 0,
        ))
        self._monitor_connection_error_rows += 1
        self._bound_monitor_buffer()
        self._maybe_flush_monitoring()

    def _bound_monitor_buffer(self) -> None:
        maximum = int(self._monitor_max_buffer_rows)
        overflow = len(self._monitor_buffer) - maximum
        if overflow > 0:
            del self._monitor_buffer[:overflow]
            self._monitor_dropped_rows += overflow
            message = f'Dropped {overflow} buffered battery monitoring rows.'
            self.log.error(message)
            self.sigError.emit(message)

    def _maybe_flush_monitoring(self) -> None:
        if not self._monitor_buffer:
            return
        due_by_rows = len(self._monitor_buffer) >= int(self._monitor_flush_rows)
        due_by_time = (
            time.monotonic() - self._monitor_last_flush_monotonic
            >= float(self._monitor_flush_interval_s)
        )
        if not due_by_rows and not due_by_time:
            return
        try:
            self.flush_monitoring()
        except Exception as exc:
            message = f'Unable to append passive battery monitoring data: {exc}'
            self.log.error(message, exc_info=True)
            self.sigError.emit(message)


def _float_or_nan(value) -> float:
    return math.nan if value is None else float(value)


def _int_or_missing(value) -> int:
    return -1 if value is None else int(value)
