# -*- coding: utf-8 -*-
"""
Interface for ODMR frequency lock hardware.

Copyright (c) 2021, the qudi developers. See the AUTHORS.md file at the top-level
directory of this distribution and on <https://github.com/Ulm-IQO/qudi-core/>

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with
qudi. If not, see <https://www.gnu.org/licenses/>.
"""

from abc import abstractmethod
from typing import Dict, Any, Optional
from qudi.core.module import Base


class OdmrFreqLockInterface(Base):
    """
    Interface for hardware-based ODMR frequency locking.

    Provides integral and PI control modes for tracking ODMR resonances
    using demodulated error signals.
    """

    @abstractmethod
    def set_bandwidth(self, bandwidth_hz: float, slope_lsb_per_hz: float) -> None:
        """
        Configure integral-only frequency lock.

        Args:
            bandwidth_hz: Lock bandwidth in Hz (typical: 100-1000 Hz)
            slope_lsb_per_hz: Error signal slope from linear fit (LSB/Hz)
                             Must be positive (absolute value).

        Raises:
            ValueError: If parameters out of valid range
        """
        pass

    @abstractmethod
    def set_bandwidth_pi(self, bandwidth_hz: float, slope_lsb_per_hz: float,
                         zero_ratio: float = 3.0) -> None:
        """
        Configure PI (proportional-integral) frequency lock.

        Args:
            bandwidth_hz: Lock bandwidth in Hz (typical: 100-1000 Hz)
            slope_lsb_per_hz: Error signal slope from linear fit (LSB/Hz)
            zero_ratio: PI zero placement ratio α (default: 3.0)
                       Zero frequency = bandwidth_hz / zero_ratio
                       Valid range: [0.1, 30.0]
                       - α < 2: Very aggressive (fast, may oscillate)
                       - α = 2.0: Aggressive (zero at BW/2, faster but may overshoot)
                       - α = 3.0: Balanced (zero at BW/3, good damping, recommended)
                       - α = 4.0: Conservative (zero at BW/4, slower but very stable)
                       - α > 4: Very conservative (P term becomes negligible)

        Raises:
            ValueError: If parameters out of valid range
        """
        pass

    @abstractmethod
    def enable_lock(self, enable: bool) -> None:
        """
        Enable or disable frequency lock.

        Args:
            enable: True to enable lock, False to disable
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get current lock status.

        Returns:
            dict with keys:
                - enabled (bool): Lock enabled
                - locked (bool): Lock acquired (error < threshold)
                - saturated (bool): Correction saturated (any saturation)
                - error_lsb (float): Current error signal (LSB)
                - correction_hz (float): Current frequency correction (Hz)
                - mu_hz_per_lsb (float): Integral gain (Hz/LSB)
                - kp_hz_per_lsb (float): Proportional gain (Hz/LSB, 0 if I-only)
        """
        pass

    def get_filter_capabilities(self) -> Dict[str, bool]:
        """
        Demodulation filters of the loaded firmware.

        Returns:
            dict {filter name: available}. Hardware without selectable
            filters returns an empty dict.
        """
        return {}

    def get_demod_filter(self) -> Dict[str, Any]:
        """
        Current demodulation filter of the tracked channel.

        Returns:
            dict with keys 'filter' (str), 'bypass' (bool), 'smith_filter'
            (filter the Smith model assumes, or None) and 'fpga_profile'
            (dict). Empty for hardware without selectable filters.
        """
        return {}

    def set_demod_filter(self, filter_name: Optional[str], bypass: bool) -> Dict[str, Any]:
        """
        Select the demodulation filter of the tracked channel.

        Args:
            filter_name: one of get_filter_capabilities(); None keeps the
                current selection.
            bypass: True passes the decimator output without the FIR.

        Returns:
            The applied setting, as get_demod_filter().
        """
        raise NotImplementedError('This hardware has no selectable demodulation filter')

    @abstractmethod
    def clear(self) -> None:
        """
        Clear integrator state (reset correction to zero).

        Does not disable lock. Use to re-acquire lock after disturbance.
        """
        pass

    @abstractmethod
    def get_constraints(self) -> Dict[str, Any]:
        """
        Get hardware constraints.

        Returns:
            dict with keys:
                - bandwidth_range (tuple): (min_hz, max_hz)
                - slope_range (tuple): (min_lsb_per_hz, max_lsb_per_hz)
                - damping_range (tuple): (min, max) for PI mode
                - max_correction_hz (float): Maximum frequency correction
        """
        pass

    @abstractmethod
    def set_max_correction_hz(self, max_correction_hz: float) -> None:
        """
        Set maximum frequency correction (FTW saturation limit).

        The lock integrator output is clamped to ±max_correction_hz.
        When the correction hits this limit, the 'saturated' status flag is set.

        Args:
            max_correction_hz: Maximum correction magnitude in Hz (default: 1e6)
                              Typical range: 100 kHz to 10 MHz depending on
                              expected drift and tuning range.

        Raises:
            ValueError: If value out of hardware-supported range
        """
        pass

    @abstractmethod
    def get_max_correction_hz(self) -> float:
        """
        Get current maximum frequency correction setting.

        Returns:
            float: Maximum correction magnitude in Hz
        """
        pass

    @abstractmethod
    def set_invert(self, inverted: bool) -> None:
        """
        Set error signal polarity inversion for correct lock operation.

        This setting is critical for correct feedback polarity depending on
        which sideband is used in IQ mixing:

        - Lower Sideband (LSB): f_RF = f_LO - f_IF
          Set inverted=True because increasing f_IF decreases f_RF.

        - Upper Sideband (USB): f_RF = f_LO + f_IF
          Set inverted=False because increasing f_IF increases f_RF.

        With wrong polarity, the loop will have positive feedback and either
        oscillate or push the frequency away from resonance.

        Args:
            inverted: True for LSB operation, False for USB operation
        """
        pass

    @abstractmethod
    def get_invert(self) -> bool:
        """
        Get current error signal inversion setting.

        Returns:
            bool: True if error is inverted (LSB mode), False otherwise (USB mode)
        """
        pass
