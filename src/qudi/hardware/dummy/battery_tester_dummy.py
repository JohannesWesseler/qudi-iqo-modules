# -*- coding: utf-8 -*-
"""Deterministic multichannel dummy for battery tester orchestration development."""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from qudi.core.configoption import ConfigOption
from qudi.interface.battery_tester_interface import (
    BatteryTesterConnectionError,
    BatteryTesterInterface,
    BatteryTesterProtocolError,
)
from qudi.interface.battery_tester_models import (
    BatteryTesterCapabilities,
    ChannelOperatingState,
    ChannelSnapshot,
    QueuedTaskSnapshot,
)
from qudi.interface.battery_tester_task import (
    BatteryTaskSpec,
    ExperimentReference,
    TaskReference,
)
from qudi.util.mutex import RecursiveMutex


@dataclass
class _DummyTask:
    spec: BatteryTaskSpec
    reference: TaskReference


@dataclass
class _DummyRun:
    task: _DummyTask
    reference: ExperimentReference
    started_at: float
    paused_at: Optional[float] = None
    paused_duration_s: float = 0.0

    def elapsed(self, now: float) -> float:
        end = self.paused_at if self.paused_at is not None else now
        return max(0.0, end - self.started_at - self.paused_duration_s)


class BatteryTesterDummy(BatteryTesterInterface):
    """In-memory tester with queue idempotency and simulated charge/discharge states."""

    _channel_ids = ConfigOption(name='channel_ids', default=(1, 2, 3, 4), missing='nothing')
    _initial_voltage_v = ConfigOption(
        name='initial_voltage_v', default=3.7, missing='nothing'
    )
    _temperature_c = ConfigOption(name='temperature_c', default=25.0, missing='nothing')
    _start_delay_s = ConfigOption(name='start_delay_s', default=0.25, missing='nothing')
    _experiment_duration_s = ConfigOption(
        name='experiment_duration_s', default=10.0, missing='nothing'
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_lock = RecursiveMutex()
        self._capabilities: Optional[BatteryTesterCapabilities] = None
        self._queues: Dict[int, List[_DummyTask]] = {}
        self._tasks_by_run_uuid: Dict[str, _DummyTask] = {}
        self._active: Dict[int, _DummyRun] = {}
        self._next_task_id = 1
        self._next_experiment_id = 1

    def on_activate(self) -> None:
        channel_ids = tuple(dict.fromkeys(int(value) for value in self._channel_ids))
        if not channel_ids or any(channel_id <= 0 for channel_id in channel_ids):
            raise ValueError('channel_ids must contain unique positive integers.')
        if float(self._experiment_duration_s) <= 0 or float(self._start_delay_s) < 0:
            raise ValueError('Dummy experiment duration must be positive and delay non-negative.')
        self._queues = {channel_id: [] for channel_id in channel_ids}
        self._tasks_by_run_uuid = {}
        self._active = {}
        self._next_task_id = 1
        self._next_experiment_id = 1
        self._capabilities = BatteryTesterCapabilities(
            system_id=0,
            system_name='BatteryTesterDummy',
            frontend_version=None,
            api_version=None,
            serial_handler_version=None,
            serial_handler_commit=None,
            channel_ids=channel_ids,
            task_control_available=True,
            live_transport='dummy',
        )

    def on_deactivate(self) -> None:
        with self._thread_lock:
            self._active = {}
            self._queues = {}
            self._tasks_by_run_uuid = {}
            self._capabilities = None

    @property
    def capabilities(self) -> BatteryTesterCapabilities:
        if self._capabilities is None:
            raise BatteryTesterConnectionError('Battery tester dummy is not active.')
        return self._capabilities

    def get_channel_snapshots(self) -> Tuple[ChannelSnapshot, ...]:
        with self._thread_lock:
            now = time.monotonic()
            self._advance(now)
            return tuple(self._make_snapshot(channel_id, now) for channel_id in self.capabilities.channel_ids)

    def get_channel_snapshot(self, channel_id: int) -> ChannelSnapshot:
        with self._thread_lock:
            channel_id = self._validate_channel(channel_id)
            now = time.monotonic()
            self._advance(now)
            return self._make_snapshot(channel_id, now)

    def submit_task(self, spec: BatteryTaskSpec) -> TaskReference:
        with self._thread_lock:
            channel_id = self._validate_channel(spec.channel_id)
            queued = self._tasks_by_run_uuid.get(spec.run_uuid)
            if queued is not None:
                if queued.spec != spec:
                    raise BatteryTesterProtocolError(
                        f'Run UUID {spec.run_uuid} already identifies a different task.'
                    )
                return queued.reference
            reference = TaskReference(
                channel_id=channel_id,
                task_id=self._next_task_id,
                run_uuid=spec.run_uuid,
                protocol_checksum_sha256=spec.protocol.checksum_sha256,
            )
            self._next_task_id += 1
            queued = _DummyTask(spec=spec, reference=reference)
            self._queues[channel_id].append(queued)
            self._tasks_by_run_uuid[spec.run_uuid] = queued
            return reference

    def start_task(self, task: TaskReference) -> ExperimentReference:
        with self._thread_lock:
            channel_id = self._validate_channel(task.channel_id)
            self._advance(time.monotonic())
            if channel_id in self._active:
                active = self._active[channel_id]
                if active.task.reference == task:
                    return active.reference
                raise BatteryTesterProtocolError(f'Channel {channel_id} is already running a task.')
            queue = self._queues[channel_id]
            if not queue or queue[0].reference != task:
                raise BatteryTesterProtocolError(
                    f'Task {task.task_id} is not at the head of channel {channel_id} queue.'
                )
            reference = ExperimentReference(
                channel_id=channel_id,
                task_id=task.task_id,
                experiment_id=self._next_experiment_id,
                run_uuid=task.run_uuid,
            )
            self._next_experiment_id += 1
            self._active[channel_id] = _DummyRun(
                task=queue[0], reference=reference, started_at=time.monotonic()
            )
            if self.module_state() == 'idle':
                self.module_state.lock()
            return reference

    def pause_task(self, channel_id: int) -> None:
        with self._thread_lock:
            channel_id = self._validate_channel(channel_id)
            run = self._require_active(channel_id)
            if run.paused_at is None:
                run.paused_at = time.monotonic()

    def resume_task(self, channel_id: int) -> None:
        with self._thread_lock:
            channel_id = self._validate_channel(channel_id)
            run = self._require_active(channel_id)
            if run.paused_at is not None:
                now = time.monotonic()
                run.paused_duration_s += now - run.paused_at
                run.paused_at = None

    def stop_task(self, channel_id: int) -> None:
        with self._thread_lock:
            channel_id = self._validate_channel(channel_id)
            self._require_active(channel_id)
            del self._active[channel_id]
            self._update_module_state()

    def _advance(self, now: float) -> None:
        duration = float(self._start_delay_s) + float(self._experiment_duration_s)
        completed = [
            channel_id
            for channel_id, run in self._active.items()
            if run.paused_at is None and run.elapsed(now) >= duration
        ]
        for channel_id in completed:
            run = self._active.pop(channel_id)
            queue = self._queues[channel_id]
            if queue and queue[0].reference == run.task.reference:
                queue.pop(0)
        if completed:
            self._update_module_state()

    def _make_snapshot(self, channel_id: int, now: float) -> ChannelSnapshot:
        run = self._active.get(channel_id)
        state = ChannelOperatingState.STANDBY
        status_code = 0
        running = False
        experiment_id = task_id = None
        task_name = protocol_name = protocol_version = battery_model = battery_number = None
        runtime_s = protocol_step_time_s = None
        protocol_step = loop_counter = None
        current_a = 0.0
        voltage_v = float(self._initial_voltage_v)

        if run is not None:
            elapsed = run.elapsed(now)
            spec = run.task.spec
            running = True
            experiment_id = run.reference.experiment_id
            task_id = run.reference.task_id
            task_name = spec.test_name
            protocol_name = spec.protocol.source_name or 'protocol.json'
            protocol_version = spec.protocol.version
            battery_model = spec.battery_model
            battery_number = spec.battery_number
            runtime_s = elapsed
            protocol_step_time_s = elapsed
            loop_counter = 0
            protocol_step = spec.protocol.step_ids[0]
            if run.paused_at is not None:
                state, status_code = ChannelOperatingState.PAUSED, 82
            elif elapsed < float(self._start_delay_s):
                state, status_code = ChannelOperatingState.STARTING, 40
            else:
                progress = min(
                    1.0,
                    (elapsed - float(self._start_delay_s)) / float(self._experiment_duration_s),
                )
                if progress < 0.5:
                    state, status_code = ChannelOperatingState.CHARGING, 120
                    current_a = min(0.1, spec.safety_limits.maximum_charge_current_a * 0.5)
                    peak_voltage = min(
                        voltage_v + 0.2,
                        spec.safety_limits.maximum_voltage_v,
                    )
                    voltage_v += (peak_voltage - voltage_v) * (progress / 0.5)
                else:
                    state, status_code = ChannelOperatingState.DISCHARGING, 220
                    current_a = -min(
                        0.1, spec.safety_limits.maximum_discharge_current_a * 0.5
                    )
                    peak_voltage = min(
                        voltage_v + 0.2,
                        spec.safety_limits.maximum_voltage_v,
                    )
                    voltage_v += (peak_voltage - voltage_v) * (1.0 - progress) / 0.5
                step_index = min(
                    len(spec.protocol.step_ids) - 1,
                    int(progress * len(spec.protocol.step_ids)),
                )
                protocol_step = spec.protocol.step_ids[step_index]

        return ChannelSnapshot(
            channel_id=channel_id,
            controller_id=channel_id,
            state=state,
            device_state_code=status_code,
            live_data_available=True,
            running=running,
            experiment_id=experiment_id,
            task_id=task_id,
            task_name=task_name,
            protocol_name=protocol_name,
            protocol_version=protocol_version,
            battery_model=battery_model,
            battery_number=battery_number,
            runtime_s=runtime_s,
            protocol_step=protocol_step,
            protocol_step_time_s=protocol_step_time_s,
            loop_counter=loop_counter,
            voltage_v=voltage_v,
            voltage_aux_v=voltage_v,
            voltage_terminals_v=voltage_v,
            current_a=current_a,
            temperature_c=(
                None if self._temperature_c is None else float(self._temperature_c)
            ),
            temperature_2_c=(
                None if self._temperature_c is None else float(self._temperature_c) + 0.2
            ),
            overrange_fields=(),
            queue=tuple(
                self._queued_snapshot(item, position)
                for position, item in enumerate(self._queues[channel_id], start=1)
            ),
        )

    @staticmethod
    def _queued_snapshot(item: _DummyTask, position: int) -> QueuedTaskSnapshot:
        spec = item.spec
        document = spec.protocol.document
        return QueuedTaskSnapshot(
            task_id=item.reference.task_id,
            channel_id=spec.channel_id,
            position=float(position),
            task_name=spec.test_name,
            project=spec.project,
            battery_id=None,
            battery_model=spec.battery_model,
            battery_number=spec.battery_number,
            battery_capacity_ah=spec.battery_capacity_ah,
            battery_mass_g=spec.battery_mass_g,
            protocol_id=None,
            protocol_name=spec.protocol.source_name or 'protocol.json',
            protocol_version=spec.protocol.version,
            protocol_globals_json=json.dumps(document['globals'], separators=(',', ':')),
            protocol_steps_json=json.dumps(document['steps'], separators=(',', ':')),
            safety_limits=spec.safety_limits,
            option_bits=0,
            comment=spec.device_comment,
        )

    def _validate_channel(self, channel_id: int) -> int:
        channel_id = int(channel_id)
        if channel_id not in self.capabilities.channel_ids:
            raise ValueError(
                f'Unknown dummy channel {channel_id}; available channels are '
                f'{self.capabilities.channel_ids}.'
            )
        return channel_id

    def _require_active(self, channel_id: int) -> _DummyRun:
        try:
            return self._active[channel_id]
        except KeyError as exc:
            raise BatteryTesterProtocolError(f'Channel {channel_id} has no active task.') from exc

    def _update_module_state(self) -> None:
        if self._active and self.module_state() == 'idle':
            self.module_state.lock()
        elif not self._active and self.module_state() == 'locked':
            self.module_state.unlock()
