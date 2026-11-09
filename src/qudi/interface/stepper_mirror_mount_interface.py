# -*- coding: utf-8 -*-
"""
Interface for open-loop stepper-motor mirror mounts (no encoders, no homing).

Positions are logical step counters kept by the hardware module; only differences are
meaningful. Backlash and the replay rule are handled by the logic, not by the hardware.

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

from abc import abstractmethod
from typing import Tuple

from qudi.core.module import Base


class StepperMirrorMountInterface(Base):
    """Relative stepper moves on a set of motors with logical position counters."""

    @property
    @abstractmethod
    def motor_names(self) -> Tuple[str, ...]:
        """Names of the motors; index i of every method refers to motor_names[i]."""
        pass

    @property
    @abstractmethod
    def step_limit(self) -> int:
        """Largest allowed |logical position| per motor."""
        pass

    @abstractmethod
    def move_steps(self, motor: int, steps: int) -> None:
        """Move motor by a relative number of steps; blocks until the move is complete.
        The logical position is updated by the hardware module."""
        pass

    @abstractmethod
    def get_positions(self) -> Tuple[int, ...]:
        """Logical positions of all motors in steps."""
        pass

    @abstractmethod
    def set_positions(self, positions) -> None:
        """Redefine the logical positions (no motion), e.g. to zero the counters."""
        pass

    @abstractmethod
    def release(self) -> None:
        """De-energize all motor coils."""
        pass
