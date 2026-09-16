# -*- coding: utf-8 -*-
"""Hardware-independent interface for multichannel battery testers."""

from abc import abstractmethod
from typing import Tuple

from qudi.core.module import Base
from qudi.interface.battery_tester_models import BatteryTesterCapabilities, ChannelSnapshot
from qudi.interface.battery_tester_task import (
    BatteryTaskSpec,
    ExperimentReference,
    TaskReference,
)


class BatteryTesterError(RuntimeError):
    """Base exception for battery tester communication and control errors."""


class BatteryTesterConnectionError(BatteryTesterError):
    """The tester web services or browser frontend could not be reached."""


class BatteryTesterProtocolError(BatteryTesterError):
    """The tester returned data that did not satisfy the expected protocol."""


class BatteryTesterControlUnavailableError(BatteryTesterError):
    """A requested mutating operation is not implemented or not safely enabled."""


class BatteryTesterInterface(Base):
    """Abstract interface consumed by Qudi battery measurement logic."""

    @property
    @abstractmethod
    def capabilities(self) -> BatteryTesterCapabilities:
        """Return the capabilities discovered during activation."""

    @abstractmethod
    def get_channel_snapshots(self) -> Tuple[ChannelSnapshot, ...]:
        """Return a current immutable snapshot for every configured channel."""

    @abstractmethod
    def get_channel_snapshot(self, channel_id: int) -> ChannelSnapshot:
        """Return a current immutable snapshot for one channel."""

    def submit_task(self, spec: BatteryTaskSpec) -> TaskReference:
        raise BatteryTesterControlUnavailableError(
            'Task submission is not available in this implementation phase.'
        )

    def start_task(self, task: TaskReference) -> ExperimentReference:
        raise BatteryTesterControlUnavailableError(
            'Task start is not available in this implementation phase.'
        )

    def pause_task(self, channel_id: int) -> None:
        raise BatteryTesterControlUnavailableError(
            'Task pause is not available in this implementation phase.'
        )

    def resume_task(self, channel_id: int) -> None:
        raise BatteryTesterControlUnavailableError(
            'Task resume is not available in this implementation phase.'
        )

    def stop_task(self, channel_id: int) -> None:
        raise BatteryTesterControlUnavailableError(
            'Task stop is not available in this implementation phase.'
        )
