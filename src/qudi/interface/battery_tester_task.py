# -*- coding: utf-8 -*-
"""Immutable protocol, task, and experiment types for battery tester orchestration."""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union
from uuid import UUID, uuid4

from qudi.interface.battery_tester_models import (
    BatteryTesterCapabilities,
    ChannelOperatingState,
    ChannelSnapshot,
    SafetyLimits,
)


SUPPORTED_PROTOCOL_VERSIONS = ('2.0', '2.1', '2.2')
QUDI_RUN_MARKER_PREFIX = 'qudi-run:'


class BatteryTaskValidationError(ValueError):
    """A protocol or task specification is incomplete or unsafe."""


@dataclass(frozen=True)
class BatteryProtocol:
    """Validated, canonically serialized HRT protocol document.

    The checksum is independent of JSON whitespace and object-key ordering. Array ordering remains
    significant because it defines protocol execution order.
    """

    canonical_json: str
    source_name: Optional[str] = field(default=None, compare=False)
    version: str = field(init=False)
    checksum_sha256: str = field(init=False)
    global_names: Tuple[str, ...] = field(init=False)
    step_ids: Tuple[int, ...] = field(init=False)
    step_types: Tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        try:
            document = json.loads(self.canonical_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise BatteryTaskValidationError(f'Protocol is not valid JSON: {exc}') from exc
        normalized, version, global_names, step_ids, step_types = _validate_protocol(document)
        object.__setattr__(self, 'canonical_json', normalized)
        object.__setattr__(self, 'version', version)
        object.__setattr__(
            self,
            'checksum_sha256',
            hashlib.sha256(normalized.encode('utf-8')).hexdigest(),
        )
        object.__setattr__(self, 'global_names', global_names)
        object.__setattr__(self, 'step_ids', step_ids)
        object.__setattr__(self, 'step_types', step_types)

    @classmethod
    def from_json_bytes(
            cls, payload: bytes, source_name: Optional[str] = None
    ) -> 'BatteryProtocol':
        try:
            text = bytes(payload).decode('utf-8-sig')
        except UnicodeDecodeError as exc:
            raise BatteryTaskValidationError('Protocol JSON must be UTF-8 encoded.') from exc
        return cls(text, source_name=source_name)

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> 'BatteryProtocol':
        path = Path(path)
        return cls.from_json_bytes(path.read_bytes(), source_name=path.name)

    @property
    def json_bytes(self) -> bytes:
        return self.canonical_json.encode('utf-8')

    @property
    def document(self) -> Dict[str, Any]:
        """Return a fresh mutable copy suitable for a transport serializer."""
        return json.loads(self.canonical_json)


@dataclass(frozen=True)
class ProtocolGlobalOverride:
    """One requested override of a named protocol global."""

    name: str
    value: Union[float, int, str]
    unit: Optional[str] = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise BatteryTaskValidationError('Protocol override name must not be empty.')
        if isinstance(self.value, bool):
            raise BatteryTaskValidationError(f'Override {name!r} must not be boolean.')
        if isinstance(self.value, (float, int)):
            if not math.isfinite(float(self.value)):
                raise BatteryTaskValidationError(f'Override {name!r} must be finite.')
        elif not isinstance(self.value, str) or not self.value.strip():
            raise BatteryTaskValidationError(
                f'Override {name!r} must be a finite number or non-empty expression.'
            )
        object.__setattr__(self, 'name', name)
        if self.unit is not None:
            object.__setattr__(self, 'unit', str(self.unit).strip() or None)


@dataclass(frozen=True)
class BatteryTaskSpec:
    """Complete, immutable request to place one battery test in a channel queue."""

    channel_id: int
    test_name: str
    protocol: BatteryProtocol
    battery_model: str
    battery_number: str
    battery_capacity_ah: float
    safety_limits: SafetyLimits
    project: Optional[str] = None
    battery_mass_g: Optional[float] = None
    global_overrides: Tuple[ProtocolGlobalOverride, ...] = ()
    comment: Optional[str] = None
    run_uuid: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        object.__setattr__(self, 'channel_id', int(self.channel_id))
        if self.channel_id <= 0:
            raise BatteryTaskValidationError('channel_id must be a positive integer.')
        for field_name in ('test_name', 'battery_model', 'battery_number'):
            normalized = str(getattr(self, field_name)).strip()
            if not normalized:
                raise BatteryTaskValidationError(f'{field_name} must not be empty.')
            object.__setattr__(self, field_name, normalized)
        try:
            normalized_uuid = str(UUID(str(self.run_uuid)))
        except (ValueError, AttributeError) as exc:
            raise BatteryTaskValidationError('run_uuid must be a valid UUID.') from exc
        object.__setattr__(self, 'run_uuid', normalized_uuid)
        object.__setattr__(self, 'battery_capacity_ah', float(self.battery_capacity_ah))
        if not math.isfinite(self.battery_capacity_ah) or self.battery_capacity_ah <= 0:
            raise BatteryTaskValidationError('battery_capacity_ah must be finite and positive.')
        if self.battery_mass_g is not None:
            mass = float(self.battery_mass_g)
            if not math.isfinite(mass) or mass <= 0:
                raise BatteryTaskValidationError('battery_mass_g must be finite and positive.')
            object.__setattr__(self, 'battery_mass_g', mass)
        overrides = tuple(self.global_overrides)
        object.__setattr__(self, 'global_overrides', overrides)
        names = [override.name for override in overrides]
        if len(names) != len(set(names)):
            raise BatteryTaskValidationError('Each protocol global may only be overridden once.')
        unknown = set(names).difference(self.protocol.global_names)
        if unknown:
            raise BatteryTaskValidationError(
                f'Unknown protocol globals requested for override: {sorted(unknown)}.'
            )
        _validate_safety_limits(self.safety_limits)
        if self.project is not None:
            object.__setattr__(self, 'project', str(self.project).strip() or None)
        if self.comment is not None:
            comment = str(self.comment).strip() or None
            if comment and QUDI_RUN_MARKER_PREFIX in comment:
                raise BatteryTaskValidationError(
                    f'Comments must not contain the reserved marker {QUDI_RUN_MARKER_PREFIX!r}.'
                )
            object.__setattr__(self, 'comment', comment)

    @property
    def ownership_marker(self) -> str:
        return f'{QUDI_RUN_MARKER_PREFIX}{self.run_uuid}'

    @property
    def device_comment(self) -> str:
        if self.comment:
            return f'{self.comment}\n{self.ownership_marker}'
        return self.ownership_marker


@dataclass(frozen=True)
class TaskReference:
    """Tester-side identity returned after idempotent queue submission."""

    channel_id: int
    task_id: int
    run_uuid: str
    protocol_checksum_sha256: str


@dataclass(frozen=True)
class ExperimentReference:
    """Tester-side identity returned after starting a queued task."""

    channel_id: int
    task_id: int
    experiment_id: int
    run_uuid: str


def validate_task_for_channel(
        spec: BatteryTaskSpec,
        capabilities: BatteryTesterCapabilities,
        snapshot: ChannelSnapshot,
        require_temperature_sensor: bool = True,
) -> None:
    """Validate a task against current channel state before any mutating action."""
    if spec.channel_id not in capabilities.channel_ids:
        raise BatteryTaskValidationError(
            f'Channel {spec.channel_id} is not configured; available channels are '
            f'{capabilities.channel_ids}.'
        )
    if snapshot.channel_id != spec.channel_id:
        raise BatteryTaskValidationError(
            f'Task requests channel {spec.channel_id}, but snapshot is for '
            f'channel {snapshot.channel_id}.'
        )
    if not snapshot.live_data_available:
        raise BatteryTaskValidationError(
            f'Channel {spec.channel_id} has no live controller data.'
        )
    if snapshot.externally_controlled:
        raise BatteryTaskValidationError(
            f'Channel {spec.channel_id} is controlled by controller {snapshot.controller_id}.'
        )
    if snapshot.running or snapshot.state is not ChannelOperatingState.STANDBY:
        raise BatteryTaskValidationError(
            f'Channel {spec.channel_id} must be in standby; current state is '
            f'{snapshot.state.value}.'
        )

    limits = spec.safety_limits
    if snapshot.voltage_v is None:
        raise BatteryTaskValidationError(
            f'Channel {spec.channel_id} cell voltage is unavailable or overrange.'
        )
    if not limits.minimum_voltage_v <= snapshot.voltage_v <= limits.maximum_voltage_v:
        raise BatteryTaskValidationError(
            f'Channel {spec.channel_id} voltage {snapshot.voltage_v:g} V is outside the task '
            f'safety range [{limits.minimum_voltage_v:g}, {limits.maximum_voltage_v:g}] V.'
        )
    if require_temperature_sensor and snapshot.temperature_c is None:
        raise BatteryTaskValidationError(
            f'Channel {spec.channel_id} temperature sensor is unavailable or overrange.'
        )
    for label, temperature in (
            ('temperature', snapshot.temperature_c),
            ('temperature_2', snapshot.temperature_2_c),
    ):
        if temperature is not None and temperature > limits.maximum_temperature_c:
            raise BatteryTaskValidationError(
                f'Channel {spec.channel_id} {label} {temperature:g} °C exceeds the task limit '
                f'{limits.maximum_temperature_c:g} °C.'
            )


def _validate_protocol(
        document: Any,
) -> Tuple[str, str, Tuple[str, ...], Tuple[int, ...], Tuple[str, ...]]:
    if not isinstance(document, Mapping):
        raise BatteryTaskValidationError('Protocol root must be a JSON object.')
    raw_version = document.get('version')
    if raw_version is None:
        raise BatteryTaskValidationError('Protocol is missing its version.')
    version = str(raw_version)
    if version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise BatteryTaskValidationError(
            f'Unsupported protocol version {version!r}; supported versions are '
            f'{SUPPORTED_PROTOCOL_VERSIONS}.'
        )
    global_names = _extract_global_names(document.get('globals'))
    steps = document.get('steps')
    if not isinstance(steps, list) or not steps:
        raise BatteryTaskValidationError('Protocol must contain a non-empty steps array.')
    step_ids = []
    step_types = []
    _collect_steps(steps, step_ids, step_types)
    if len(step_ids) != len(set(step_ids)):
        raise BatteryTaskValidationError('Protocol step IDs must be unique, including nested steps.')
    try:
        normalized = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise BatteryTaskValidationError(
            f'Protocol contains a value that cannot be represented as strict JSON: {exc}'
        ) from exc
    return normalized, version, global_names, tuple(step_ids), tuple(step_types)


def _extract_global_names(globals_payload: Any) -> Tuple[str, ...]:
    if not isinstance(globals_payload, (list, Mapping)):
        raise BatteryTaskValidationError('Protocol globals must be an array or object.')
    entries = globals_payload if isinstance(globals_payload, list) else [
        entry
        for group in globals_payload.values()
        if isinstance(group, list)
        for entry in group
    ]
    names = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not str(entry.get('name') or '').strip():
            raise BatteryTaskValidationError('Every protocol global must have a non-empty name.')
        names.append(str(entry['name']).strip())
    if len(names) != len(set(names)):
        raise BatteryTaskValidationError('Protocol global names must be unique.')
    return tuple(names)


def _collect_steps(steps: list, step_ids: list, step_types: list) -> None:
    for step in steps:
        if not isinstance(step, Mapping):
            raise BatteryTaskValidationError('Every protocol step must be a JSON object.')
        try:
            step_id = int(step['id'])
        except (KeyError, TypeError, ValueError) as exc:
            raise BatteryTaskValidationError('Every protocol step requires an integer ID.') from exc
        step_type = str(step.get('type') or '').strip()
        if step_id <= 0 or not step_type:
            raise BatteryTaskValidationError('Protocol step IDs and types must be non-empty.')
        step_ids.append(step_id)
        step_types.append(step_type)
        nested = step.get('steps')
        if nested is not None:
            if not isinstance(nested, list) or not nested:
                raise BatteryTaskValidationError(
                    f'Nested steps for protocol step {step_id} must be a non-empty array.'
                )
            _collect_steps(nested, step_ids, step_types)


def _validate_safety_limits(limits: SafetyLimits) -> None:
    values = {
        'minimum_voltage_v': limits.minimum_voltage_v,
        'maximum_voltage_v': limits.maximum_voltage_v,
        'maximum_discharge_current_a': limits.maximum_discharge_current_a,
        'maximum_charge_current_a': limits.maximum_charge_current_a,
        'maximum_temperature_c': limits.maximum_temperature_c,
    }
    if any(value is None for value in values.values()):
        raise BatteryTaskValidationError('All task safety limits must be explicitly specified.')
    normalized = {name: float(value) for name, value in values.items()}
    if not all(math.isfinite(value) for value in normalized.values()):
        raise BatteryTaskValidationError('All task safety limits must be finite.')
    if normalized['minimum_voltage_v'] < 0:
        raise BatteryTaskValidationError('minimum_voltage_v must not be negative.')
    if normalized['maximum_voltage_v'] <= normalized['minimum_voltage_v']:
        raise BatteryTaskValidationError(
            'maximum_voltage_v must be greater than minimum_voltage_v.'
        )
    if normalized['maximum_discharge_current_a'] <= 0:
        raise BatteryTaskValidationError('maximum_discharge_current_a must be positive.')
    if normalized['maximum_charge_current_a'] <= 0:
        raise BatteryTaskValidationError('maximum_charge_current_a must be positive.')
    if normalized['maximum_temperature_c'] <= -273.15:
        raise BatteryTaskValidationError('maximum_temperature_c must exceed absolute zero.')
