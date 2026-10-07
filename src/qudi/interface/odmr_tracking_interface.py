# -*- coding: utf-8 -*-
"""
Interface for ODMR resonance-tracking hardware (one or more resonances).

N >= 1 resonances share one LO, one IQ mixer and one modulation frequency. The
hardware provides:
  - a modulation/demodulation reference (frequency f_m, demodulation phase),
  - channel control: which resonance is live, and the freeze window after each
    change of the live resonance,
  - a frequency controller with one integrator slot per resonance, fed by the
    live resonance's demodulated error and correcting the IF synthesizer,
  - an acquisition engine: for N > 1 a hardware sequencer hops the LO between
    the resonances; an optional trace stream delivers error, correction and
    pre-FIR (CIC) traces of every resonance on one time axis.
N = 1 is the single-resonance case of the same hardware (no hopping).

The per-resonance SSB calibration and the LO jump list live on the
microwave/IF-source side, so they are not part of this interface.

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
from typing import Dict, Any, List, Optional, Tuple
from qudi.core.module import Base


class OdmrTrackingInterface(Base):
    """
    Hardware-based ODMR frequency tracking of N >= 1 resonances.

    Loop gains, deadband and saturation limit are global (shared by all
    resonances); each resonance has its own integrator state.
    """

    # =========================================================================
    # Controller
    # =========================================================================

    @abstractmethod
    def set_bandwidth(self, bandwidth_hz: float, slope_lsb_per_hz: float,
                      pi: bool = False, zero_ratio: float = 1.0) -> None:
        """
        Set the (global) loop gains for a target bandwidth.

        Args:
            bandwidth_hz: Target closed-loop bandwidth (Hz).
            slope_lsb_per_hz: Discriminator slope (LSB/Hz, > 0). With N > 1 a
                representative slope; each resonance's bandwidth then scales
                with its own slope.
            pi: True for PI control, False for integral-only.
            zero_ratio: PI zero at bandwidth_hz / zero_ratio, in [0.1, 30]
                (PI only). The proportional loop gain is
                zero_ratio / sqrt(1 + zero_ratio^2); with the filter delay,
                values above 1 ring.
        """
        pass

    def set_tracking_algorithm(self, algorithm: str,
                               smith_gain_multiplier: Optional[float] = None,
                               smith_delay_samples: Optional[int] = None) -> None:
        """Select the controller algorithm (hardware-specific names, e.g.
        'conventional'). The lock must be disabled; set_bandwidth applies it."""
        raise NotImplementedError('This hardware has one controller algorithm')

    @abstractmethod
    def enable_lock(self, enable: bool) -> None:
        """Enable or disable the frequency controller (all resonances)."""
        pass

    @abstractmethod
    def clear_integrators(self) -> None:
        """Reset every resonance's integrator (correction) to zero."""
        pass

    @abstractmethod
    def set_max_correction_hz(self, max_correction_hz: float) -> None:
        """Set the correction limit +/- max_correction_hz (all resonances)."""
        pass

    @abstractmethod
    def get_max_correction_hz(self) -> float:
        """Current correction limit (Hz)."""
        pass

    @abstractmethod
    def set_invert(self, inverted: bool) -> None:
        """
        Set the error-signal polarity: True for the lower sideband
        (f_RF = f_LO - f_IF, a higher IF lowers f_RF), False for the upper one.
        With the wrong polarity the loop pushes away from the resonance.
        """
        pass

    @abstractmethod
    def get_invert(self) -> bool:
        """Current error-signal polarity (True = inverted)."""
        pass

    @abstractmethod
    def get_constraints(self) -> Dict[str, Any]:
        """
        Returns:
            dict with keys bandwidth_range (min_hz, max_hz), slope_range
            (min, max LSB/Hz), max_correction_hz (largest allowed limit).
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Controller status of the live resonance: enabled, locked, saturated,
        error_lsb, correction_hz, gains.
        """
        pass

    @abstractmethod
    def get_slot_status(self, slot: int) -> Dict[str, Any]:
        """Status of resonance ``slot``: at least enabled, locked, saturated,
        error_lsb, correction_hz."""
        pass

    @abstractmethod
    def get_all_status(self) -> List[Dict[str, Any]]:
        """get_slot_status for every resonance slot."""
        pass

    # =========================================================================
    # Demodulation filters
    # =========================================================================

    def get_filter_capabilities(self) -> Dict[str, bool]:
        """{filter name: available} of the loaded firmware; empty if fixed."""
        return {}

    def get_demod_filter(self) -> Dict[str, Any]:
        """
        Filter of resonance 0's tracked lane: dict with 'filter' (str),
        'bypass' (bool), 'smith_filter' (filter a predictor model assumes, or
        None), 'fpga_profile' (dict). Empty if the filter is fixed.
        """
        return {}

    def set_demod_filter(self, filter_name: Optional[str], bypass: bool) -> Dict[str, Any]:
        """Select resonance 0's filter (None keeps it) and FIR bypass; returns
        get_demod_filter()."""
        raise NotImplementedError('This hardware has no selectable demodulation filter')

    # =========================================================================
    # Reference and channel control
    # =========================================================================

    @abstractmethod
    def configure_reference(self, f_m_hz: float, demod_phase_deg: float) -> None:
        """Set the modulation frequency f_m (Hz) and the demodulation phase
        2*pi*f_m*tau (degrees)."""
        pass

    @abstractmethod
    def get_demod_phase(self) -> float:
        """Current demodulation phase (degrees)."""
        pass

    @abstractmethod
    def configure_channels(self, settle_time_s: float,
                           source: str = 'current_step') -> None:
        """
        Args:
            settle_time_s: Freeze window after each change of the live resonance
                (s): the LO transient and the filter tail are gated out.
            source: 'current_step' (the hardware hop index selects the live
                resonance) or 'sw' (resonance 0 stays live).
        """
        pass

    @abstractmethod
    def enable_channel_selection(self, enable: bool) -> None:
        """Enable channel selection and freeze gating. Disabled, resonance 0 is
        live and nothing is frozen."""
        pass

    # =========================================================================
    # Acquisition engine and trace stream
    # =========================================================================

    @property
    @abstractmethod
    def nslots(self) -> int:
        """Number of resonance slots implemented in hardware."""
        pass

    @property
    @abstractmethod
    def tracking_engine_active(self) -> bool:
        """Whether the acquisition engine runs (independent of the lock)."""
        pass

    @property
    @abstractmethod
    def trace_stream_active(self) -> bool:
        """Whether the error/correction trace stream runs."""
        pass

    @abstractmethod
    def start_engine(self, nslots: int, dwell_time_s: float,
                     settling_time_s: float, trigger_length_s: float,
                     stream_traces: bool = True) -> None:
        """
        Start the acquisition engine for ``nslots`` resonances.

        For nslots > 1 the hardware hops the LO through the resonances, one
        trigger per resonance, until stop_engine. For nslots == 1 resonance 0
        stays live and the timing arguments are ignored.

        Args:
            nslots: Number of resonances N (= LO jump-list length).
            dwell_time_s: Live time per resonance (s).
            settling_time_s: LO settle delay before the dwell (s).
            trigger_length_s: LO-hop trigger pulse length (s).
            stream_traces: Also start the trace stream (read_traces).
        """
        pass

    @abstractmethod
    def stop_engine(self) -> None:
        """Stop hopping and the trace stream."""
        pass

    @abstractmethod
    def start_trace_stream(self) -> None:
        """Start the trace stream on a running engine (lock state unchanged)."""
        pass

    @abstractmethod
    def stop_trace_stream(self) -> None:
        """Stop only the trace stream (engine and lock unchanged)."""
        pass

    @abstractmethod
    def set_trace_window_seconds(self, seconds: float) -> None:
        """Length (s) of the rolling high-rate window returned by read_traces."""
        pass

    @abstractmethod
    def read_traces(self) -> Optional[Dict[str, Any]]:
        """
        Per-resonance traces of the current rolling window.

        Returns:
            None if not streaming, else a dict with
              - 'times' (T,): elapsed time since the stream started (s),
              - 'err' (N, T): error (LSB); parked resonances hold their last
                value, NaN = transport loss in the live resonance,
              - 'corr_hz' (N, T): correction (Hz), same semantics,
              - 'cic' (N, T): demodulated signal before the FIR (LSB),
              - 'sample_rate': samples per second.
        """
        pass

    @abstractmethod
    def get_trace_calibration(self) -> Dict[str, Any]:
        """Stream format, selected filters and FIR DC gains (CIC LSB to FIR LSB)."""
        pass

    # =========================================================================
    # Direct stream drain: 2D motor scans and continuous field logging
    # =========================================================================
    # While a mapped scan or a field logger owns the drain, read_traces does
    # not drain; the owner is the only caller of read_stream_words.

    @property
    @abstractmethod
    def position_markers_active(self) -> bool:
        """Whether a mapped motor scan owns the trace stream and its markers."""
        pass

    @property
    @abstractmethod
    def stream_sample_rate(self) -> float:
        """Trace stream sample rate (Hz)."""
        pass

    @property
    @abstractmethod
    def stream_words_per_sample(self) -> int:
        """Elements of read_stream_words per sample."""
        pass

    @abstractmethod
    def enable_position_markers(self, enable: bool) -> None:
        """Record x/y position markers on the running trace stream (raises
        without an active stream, so a scan cannot acquire empty maps)."""
        pass

    @abstractmethod
    def read_position_markers(self) -> Tuple[Any, Any]:
        """New (x, y) markers since the last call, as sample indices into the
        reconstructed traces (two int64 arrays)."""
        pass

    @abstractmethod
    def read_stream_words(self) -> Any:
        """New raw stream records since the last call (destructive drain)."""
        pass

    @abstractmethod
    def reconstruct_mapped_traces(self, words: Any) -> Dict[str, Any]:
        """
        Decode accumulated stream records into per-resonance traces without
        hold: NaN while a resonance is parked, frozen or lost, so a per-bin
        nanmean sees only its own live samples.

        Returns:
            dict with 'err', 'cic' (N, T) LSB, 'corr_hz' (N, T) Hz,
            'times' (T,) s and 'sample_rate' (Hz).
        """
        pass

    @abstractmethod
    def begin_field_drain(self) -> None:
        """Hand the stream drain to a continuous field logger (needs an active
        stream; refused while a mapped scan owns it)."""
        pass

    @abstractmethod
    def end_field_drain(self) -> None:
        """Return the drain to read_traces (idempotent)."""
        pass
