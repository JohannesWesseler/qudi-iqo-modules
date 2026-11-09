# -*- coding: utf-8 -*-
"""
Interface for a waveform current source: drives a programmable current (DC, sine, square,
triangle, arbitrary; continuous or in triggered bursts) through a load and measures what it
outputs. What the load is (a coil, a resistor) and what the current does there is not part of
this interface.

Units: current in A (peak value of the waveform), voltage in V, frequency in Hz, time in s.

Waveforms:
    'dc'         constant current `offset`
    'sine'       offset + amplitude * sin(2 pi f t)
    'square'     offset +- amplitude
    'triangle'   offset + amplitude * triangle(f t)
    'arbitrary'  offset + amplitude * table(f t), table set by set_arbitrary_waveform(),
                 normalized to max |table| = 1

Readings (measure): a source can know its output current in more than one way, and they need
not agree. Each is reported under its own key; the documentation of the hardware module says
how accurate each one is.
    i_*            current measured directly (e.g. a shunt)
    v_*            output voltage at the load
    i_*_output     current calculated from the output voltage and the load impedance
    i_*_command    current the source expects from its setting (expected_current)
Sine output: *_amp (peak amplitude at the set frequency) and phase_i_minus_v; all waveforms:
*_mean, *_rms, *_peak.

Bursts (configure_burst) and the sync output (set_sync_output) let an experiment align the
current with data recorded elsewhere: every generator start emits one marker edge, and a capture
armed with trigger='generator' starts on the same edge.

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
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from PySide2 import QtCore

from qudi.core.module import Base

WAVEFORMS = ('dc', 'sine', 'square', 'triangle', 'arbitrary')


class CurrentSourceAbort(RuntimeError):
    """An abort condition was met; the output is off (and the supply, if `supply_off`)."""

    def __init__(self, reason: str, supply_off: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.supply_off = supply_off


class CommandRefused(ValueError):
    """A requested setting violates a limit. Nothing was changed."""


@dataclass(frozen=True)
class CurrentSourceConstraints:
    waveforms: Tuple[str, ...]
    frequency_range: Tuple[float, float]       # Hz, generator
    current_limit: float                        # A peak, present limit
    current_limit_absolute: float               # A peak, the limit may be raised up to this
    dc_continuous_current: float                # A, DC above this only for dc_high_max_time
    dc_high_max_time: float                     # s
    dc_high_rest_time: float                    # s at <= dc_continuous_current in between
    ac_continuous_current: float                # A peak, AC above this only for ac_high_max_time
    ac_high_max_time: float                     # s
    ac_high_rest_time: float                    # s
    capture_length: int                         # samples per capture
    capture_sample_rate: float                  # Hz, at decimation 1
    arbitrary_length: int                       # points of the arbitrary table
    burst_max_cycles: int
    burst_max_repetitions: int


class CurrentSourceInterface(Base):
    """Waveform current source with current and voltage measurement."""

    sigStateChanged = QtCore.Signal(dict)       # output_state after every change
    sigSafetyAbort = QtCore.Signal(str, bool)   # reason, supply switched off

    # --- description -----------------------------------------------------------------------
    @property
    @abstractmethod
    def constraints(self) -> CurrentSourceConstraints:
        pass

    @abstractmethod
    def max_current(self, waveform: str, frequency: float) -> float:
        """Largest peak current (A) allowed for this waveform at this frequency: the current
        limit or the output-voltage limit, whichever is lower."""
        pass

    @property
    @abstractmethod
    def current_limit(self) -> float:
        pass

    @current_limit.setter
    def current_limit(self, value: float) -> None:
        pass

    @abstractmethod
    def expected_current(self, frequency: float, amplitude: float) -> complex:
        """Current the source expects for a sine of this amplitude (A, as passed to
        set_output) at this frequency: amplitude * exp(i phase), the phase of
        amplitude * sin(2 pi f t + phase) relative to the generator start (t = 0 of a capture
        armed with trigger='generator')."""
        pass

    # --- power -----------------------------------------------------------------------------
    @abstractmethod
    def power_on(self) -> Dict[str, Any]:
        """Switch the amplifier supply on (output off), check the idle state. Returns the
        supply readings."""
        pass

    @abstractmethod
    def power_off(self) -> None:
        """Output off first, then the amplifier supply."""
        pass

    @property
    @abstractmethod
    def powered(self) -> bool:
        pass

    # --- output ----------------------------------------------------------------------------
    @abstractmethod
    def set_output(self, waveform: str, frequency: float = 0.0, amplitude: float = 0.0,
                   offset: float = 0.0, ramp: bool = True) -> Dict[str, Any]:
        """Set (and switch on) the output. amplitude, offset in A. With ramp, a running output
        is ramped down and the new one up in limited steps with a check after each step.
        Returns the output state (including the generator voltages and predicted current)."""
        pass

    @abstractmethod
    def output_off(self) -> None:
        """Ramp to zero and switch the output off (supply stays on)."""
        pass

    @property
    @abstractmethod
    def output_state(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def set_arbitrary_waveform(self, samples: Sequence[float]) -> None:
        """Table for waveform 'arbitrary' (one period, any length up to arbitrary_length;
        resampled and normalized to max |sample| = 1). Only while the output is off or not
        'arbitrary'."""
        pass

    @abstractmethod
    def configure_burst(self, n_cycles: Optional[int], n_repetitions: int = 1,
                        period: Optional[float] = None) -> None:
        """n_cycles None: continuous output. Otherwise the next trigger() plays n_cycles
        periods, n_repetitions times (0: endless), one burst every `period` s. Only while
        the output is off."""
        pass

    @abstractmethod
    def set_sync_output(self, enabled: bool) -> None:
        """Marker output (one edge at every generator start)."""
        pass

    @abstractmethod
    def trigger(self) -> None:
        """Start the generator now (a burst, or restart of a continuous waveform)."""
        pass

    # --- measurement -----------------------------------------------------------------------
    @abstractmethod
    def measure(self, n_average: int = 1) -> Dict[str, Any]:
        """Output now (keys: see the module docstring), plus the supply readings. Runs the
        safety checks."""
        pass

    @abstractmethod
    def capture(self, decimation: int = 64) -> Dict[str, Any]:
        """Immediate capture: dict(t, current (A), output (V), sample_rate, decimation)."""
        pass

    @abstractmethod
    def arm_capture(self, decimation: int = 64, trigger: str = 'generator',
                    pre_trigger: int = 0) -> None:
        """Arm a capture that starts on the next generator start ('generator') or on the
        external trigger input ('external'). pre_trigger samples before the trigger."""
        pass

    @abstractmethod
    def read_capture(self, timeout: float = 5.0) -> Dict[str, Any]:
        """Wait for the armed capture and return it like capture(), plus trigger_index."""
        pass

    @abstractmethod
    def read_supply(self) -> Dict[str, Any]:
        """Supply voltages, currents and fuse states."""
        pass
