# -*- coding: utf-8 -*-
"""Shared data types for battery tester hardware and logic modules."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional, Tuple


class ChannelOperatingState(Enum):
    """Normalized operating states used independently of a tester implementation."""

    UNKNOWN = 'unknown'
    STANDBY = 'standby'
    TRANSITIONAL = 'transitional'
    STARTING = 'starting'
    PAUSED = 'paused'
    ERROR = 'error'
    CHARGING = 'charging'
    DISCHARGING = 'discharging'

    @classmethod
    def from_hrt_code(cls, code: Optional[int]) -> 'ChannelOperatingState':
        """Map an HRT channel-state code to a normalized state."""
        if code is None:
            return cls.UNKNOWN
        if code == 0:
            return cls.STANDBY
        if code in (1, 7, 8, 51, 52):
            return cls.TRANSITIONAL
        if code == 2:
            return cls.PAUSED
        if 12 <= code <= 19:
            return cls.CHARGING
        if 20 <= code <= 29:
            return cls.DISCHARGING
        if code == 40:
            return cls.STARTING
        if code == 41 or 80 <= code <= 89:
            return cls.PAUSED
        if code == 99:
            return cls.ERROR
        if 100 <= code <= 159:
            return cls.CHARGING
        if 200 <= code <= 259:
            return cls.DISCHARGING
        return cls.UNKNOWN


class TerminationReason(Enum):
    """Normalized meanings of terminal HRT data event codes."""

    UNKNOWN = 'unknown'
    COMPLETED = 'completed'
    SAFETY_LIMIT = 'safety_limit'
    VOLTAGE_MISMATCH = 'voltage_mismatch'
    STEP_ERROR = 'step_error'
    STOPPED = 'stopped'
    GENERAL_ERROR = 'general_error'

    @classmethod
    def from_hrt_event(cls, event: Optional[int]) -> 'TerminationReason':
        if event == 30:
            return cls.COMPLETED
        if event is not None and 100 <= event <= 109:
            return cls.SAFETY_LIMIT
        if event == 140:
            return cls.VOLTAGE_MISMATCH
        if event == 150:
            return cls.STEP_ERROR
        if event == 190:
            return cls.STOPPED
        if event == 199:
            return cls.GENERAL_ERROR
        return cls.UNKNOWN


@dataclass(frozen=True)
class SafetyLimits:
    """Per-task safety envelope, expressed in SI units."""

    minimum_voltage_v: Optional[float]
    maximum_voltage_v: Optional[float]
    maximum_discharge_current_a: Optional[float]
    maximum_charge_current_a: Optional[float]
    maximum_temperature_c: Optional[float]


@dataclass(frozen=True)
class QueuedTaskSnapshot:
    """Immutable representation of a task in an HRT channel queue."""

    task_id: int
    channel_id: int
    position: float
    task_name: str
    project: Optional[str]
    battery_id: Optional[int]
    battery_model: Optional[str]
    battery_number: Optional[str]
    battery_capacity_ah: Optional[float]
    battery_mass_g: Optional[float]
    protocol_id: Optional[int]
    protocol_name: Optional[str]
    protocol_version: Optional[str]
    protocol_globals_json: Optional[str]
    protocol_steps_json: Optional[str]
    safety_limits: SafetyLimits
    option_bits: int
    comment: Optional[str]

    @classmethod
    def from_hrt_payload(cls, payload: Mapping[str, Any]) -> 'QueuedTaskSnapshot':
        return cls(
            task_id=int(payload['id']),
            channel_id=int(payload['channelID']),
            position=float(payload.get('position') or 0),
            task_name=str(payload.get('taskName') or ''),
            project=_optional_str(payload.get('project')),
            battery_id=_optional_int(payload.get('batteryID')),
            battery_model=_optional_str(payload.get('batteryModel')),
            battery_number=_optional_str(payload.get('batteryNumber')),
            battery_capacity_ah=_optional_float(payload.get('batteryC_Ah')),
            battery_mass_g=_optional_float(payload.get('batteryMass_g')),
            protocol_id=_optional_int(payload.get('protocolID')),
            protocol_name=_optional_str(payload.get('protocolName')),
            protocol_version=_optional_str(payload.get('protocolVersion')),
            protocol_globals_json=_optional_str(payload.get('protocolGlobals')),
            protocol_steps_json=_optional_str(payload.get('protocolSteps')),
            safety_limits=SafetyLimits(
                minimum_voltage_v=_optional_float(payload.get('batteryVMin_V')),
                maximum_voltage_v=_optional_float(payload.get('batteryVMax_V')),
                maximum_discharge_current_a=_optional_float(payload.get('batteryIMin_A')),
                maximum_charge_current_a=_optional_float(payload.get('batteryIMax_A')),
                maximum_temperature_c=_optional_float(payload.get('batteryTMax_dC')),
            ),
            option_bits=int(payload.get('optionBits') or 0),
            comment=_optional_str(payload.get('comment')),
        )


@dataclass(frozen=True)
class ChannelSnapshot:
    """Combined database and live snapshot for one tester channel.

    Live voltage and current values are converted from the frontend's mV/mA representation to
    SI units. Temperature values are already supplied in degrees Celsius.
    """

    channel_id: int
    controller_id: Optional[int]
    state: ChannelOperatingState
    device_state_code: Optional[int]
    live_data_available: bool
    running: bool
    experiment_id: Optional[int]
    task_id: Optional[int]
    task_name: Optional[str]
    protocol_name: Optional[str]
    protocol_version: Optional[str]
    battery_model: Optional[str]
    battery_number: Optional[str]
    runtime_s: Optional[float]
    protocol_step: Optional[int]
    protocol_step_time_s: Optional[float]
    loop_counter: Optional[int]
    voltage_v: Optional[float]
    voltage_aux_v: Optional[float]
    voltage_terminals_v: Optional[float]
    current_a: Optional[float]
    temperature_c: Optional[float]
    temperature_2_c: Optional[float]
    overrange_fields: Tuple[str, ...]
    queue: Tuple[QueuedTaskSnapshot, ...]

    @property
    def externally_controlled(self) -> bool:
        return self.controller_id is not None and self.controller_id != self.channel_id


@dataclass(frozen=True)
class BatteryTesterCapabilities:
    """Capabilities and identity discovered from the HRT web services."""

    system_id: Optional[int]
    system_name: Optional[str]
    frontend_version: Optional[str]
    api_version: Optional[str]
    serial_handler_version: Optional[str]
    serial_handler_commit: Optional[str]
    channel_ids: Tuple[int, ...]
    task_control_available: bool = False
    live_transport: Optional[str] = None


@dataclass(frozen=True)
class BatteryMonitoringSession:
    """Immutable public snapshot of one passive monitoring file/session."""

    session_id: str
    started_at: datetime
    stopped_at: Optional[datetime]
    channel_ids: Tuple[int, ...]
    file_path: str
    rows_written: int
    buffered_rows: int
    dropped_rows: int
    connection_error_rows: int
    active: bool
    plot_file_path: Optional[str] = None


def channel_snapshot_from_hrt_payloads(
        database: Mapping[str, Any],
        live: Optional[Mapping[str, Any]],
) -> ChannelSnapshot:
    """Build a normalized channel snapshot from REST and frontend-live payloads."""
    channel_id = int(database['channelID'])
    queue_payload = database.get('queue') or ()
    queue = tuple(QueuedTaskSnapshot.from_hrt_payload(task) for task in queue_payload)

    if live is None:
        return ChannelSnapshot(
            channel_id=channel_id,
            controller_id=None,
            state=ChannelOperatingState.UNKNOWN,
            device_state_code=None,
            live_data_available=False,
            running=False,
            experiment_id=None,
            task_id=None,
            task_name=None,
            protocol_name=None,
            protocol_version=None,
            battery_model=None,
            battery_number=None,
            runtime_s=None,
            protocol_step=None,
            protocol_step_time_s=None,
            loop_counter=None,
            voltage_v=None,
            voltage_aux_v=None,
            voltage_terminals_v=None,
            current_a=None,
            temperature_c=None,
            temperature_2_c=None,
            overrange_fields=(),
            queue=queue,
        )

    live_data_available = live.get('controllerID') is not None
    device_state = _optional_int(live.get('live_status')) if live_data_available else None
    live_experiment_id = _optional_int(live.get('lastLiveExpID'))
    running = bool(live.get('running'))
    overrange = []
    voltage_v = _live_milli_value(live.get('live_voltage'), 'voltage', overrange)
    voltage_aux_v = _live_milli_value(live.get('live_voltage_aux'), 'voltage_aux', overrange)
    voltage_terminals_v = _live_milli_value(
        live.get('live_voltage_terminals'), 'voltage_terminals', overrange
    )
    current_a = _live_milli_value(live.get('live_current'), 'current', overrange)
    temperature_c = _live_temperature(live.get('live_temp'), 'temperature', overrange)
    temperature_2_c = _live_temperature(live.get('live_temp2'), 'temperature_2', overrange)

    return ChannelSnapshot(
        channel_id=channel_id,
        controller_id=_optional_int(live.get('controllerID')),
        state=ChannelOperatingState.from_hrt_code(device_state),
        device_state_code=device_state,
        live_data_available=live_data_available,
        running=running,
        experiment_id=live_experiment_id if running else None,
        task_id=(
            _optional_int(live.get('taskID')) or _optional_int(database.get('taskID'))
            if running else None
        ),
        task_name=(
            _optional_str(live.get('taskName')) or _optional_str(database.get('taskName'))
            if running else None
        ),
        protocol_name=_optional_str(database.get('protocolName')) if running else None,
        protocol_version=_optional_str(database.get('protocolVersion')) if running else None,
        battery_model=_optional_str(database.get('batteryModel')) if running else None,
        battery_number=_optional_str(database.get('batteryNumber')) if running else None,
        runtime_s=_optional_float(live.get('live_uptime')),
        protocol_step=_optional_int(live.get('protocolStep')),
        protocol_step_time_s=_optional_float(live.get('protocolStepTime')),
        loop_counter=_optional_int(live.get('loopCounter')),
        voltage_v=voltage_v,
        voltage_aux_v=voltage_aux_v,
        voltage_terminals_v=voltage_terminals_v,
        current_a=current_a,
        temperature_c=temperature_c,
        temperature_2_c=temperature_2_c,
        overrange_fields=tuple(overrange),
        queue=queue,
    )


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None or value == '' else int(value)


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None or value == '' else float(value)


def _optional_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _live_milli_value(value: Any, field: str, overrange: list) -> Optional[float]:
    parsed = _optional_float(value)
    if parsed is None:
        return None
    threshold = 99_000.0 if field.startswith('voltage') else 99_000_000_000.0
    if abs(parsed) >= threshold:
        overrange.append(field)
        return None
    return parsed / 1000.0


def _live_temperature(value: Any, field: str, overrange: list) -> Optional[float]:
    parsed = _optional_float(value)
    if parsed is None:
        return None
    # Depending on firmware/controller state, a disconnected sensor is reported either near
    # -99 °C or as a large +/-99999 sentinel. Neither value is a usable temperature measurement.
    if parsed <= -90.0 or abs(parsed) >= 99_000.0:
        overrange.append(field)
        return None
    return parsed
