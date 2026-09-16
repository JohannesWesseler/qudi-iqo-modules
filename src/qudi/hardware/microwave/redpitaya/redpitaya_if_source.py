"""
Red Pitaya IF Source Module - Multi-Frequency IQ Signal Generation with FM Support
==================================================================================

This module implements the IFSourceBase interface for Red Pitaya devices using the
PyRPL library. It provides calibrated IQ signal generation with support for up to
3 simultaneous frequency components using the fgen3 FPGA module, with optional
frequency modulation (FM) capability.

Hardware:
---------
- Device: Red Pitaya (STEMlab 125-14 or compatible)
- FPGA Module: fgen3 (3-component frequency generator with FM)
- Outputs: 2 channels (Out1 = I, Out2 = Q)

Key Features:
-------------
- Simultaneous generation of up to 3 frequency components
- Independent amplitude and phase control for each component's I/Q contributions
- Real-time IQ imbalance correction using calibration data
- Automatic DC offset compensation (averaged across frequencies)
- Interpolation of calibration parameters for any LO frequency and amplitude
- Frequency modulation (FM) support with independent deviation for each component
- Shared FM modulation frequency controlled via IQ module
- Backward compatibility with single-frequency operation

Calibration File Format:
-----------------------
 Tab-separated CSV files with columns:
 - sideband: 'upper' (USB) or 'lower' (LSB) (optional, for sanity checks)
 - lo_frequency_ghz: LO frequency in GHz
 - if_amplitude: IF amplitude (0-1)
 - g: Gain imbalance parameter
 - phi: Phase imbalance in radians
 - I_offset: DC offset for I channel (V)
 - Q_offset: DC offset for Q channel (V)

 IQ Correction Formulas:
 ----------------------
 For each frequency component with calibration parameters (g, phi):
 - I amplitude = base_amplitude * (1 + g)
 - Q amplitude = base_amplitude * (1 - g)
 - I phase = 0°
 - Q phase = base_q + phi (converted to degrees), where base_q depends on sideband:
   - Upper sideband (USB): base_q = 270° (default)
   - Lower sideband (LSB): base_q = 90°

DC offsets are averaged across all active frequencies since they represent
physical voltage offsets of the output channels.

FM Implementation:
-----------------
- FM modulation frequency is set via iq0 module (shared for all components)
- FM deviation can be set independently for each component
- FM can be enabled/disabled independently for each component

Usage Example:
-------------
```python
# Connect to Red Pitaya
rp_source = RedPitayaIFSource("192.168.1.100")
# The config_name is important for sharing the connection with other modules
rp_source.connect(config_name='my_shared_rp_config')

# Simple single frequency (backward compatible)
rp_source.set_amplitude(0.5)  # Sets frequency 0 to 0.5 amplitude

# Multi-frequency with automatic calibration and FM
frequencies = [19.422e6, 21.580e6, 23.738e6]  # Hz
amplitudes = [0.3, 0.4, 0.3]  # 0-1 normalized
lo_frequency = 2.6e9  # Hz
fm_enables = [True, True, False]  # Enable FM on first two components
fm_deviations_khz = [10.0, 5.0, 0.0]  # FM deviation in kHz
fm_modulation_frequency = 5e3  # 5 kHz modulation

calibration_files = {
    19.422e6: "path/to/calibration_IF_19.422MHz.csv",
    21.580e6: "path/to/calibration_IF_21.580MHz.csv",
    23.738e6: "path/to/calibration_IF_23.738MHz.csv"
}

rp_source.set_multi_frequency_signal(
    frequencies, amplitudes, lo_frequency, calibration_files,
    fm_enables=fm_enables,
    fm_deviations_khz=fm_deviations_khz,
    fm_modulation_frequency=fm_modulation_frequency
)
rp_source.enable_output(True)

# Verify the configuration
status = rp_source.verify_signal_generation()

# Cleanup
rp_source.disconnect()
```

Implementation Notes:
--------------------
1. The fgen3 module must be properly configured in the FPGA bitstream
2. Signal routing is configured to route fgen3 outputs to Out1 (I) and Out2 (Q)
3. Each frequency component can be independently enabled/disabled
4. Phase accumulator is 32-bit for high frequency resolution
5. Calibration data is cached after loading for efficiency
6. FM modulation frequency is controlled via iq0 module

"""

import numpy as np
import pyrpl
from typing import Tuple, Optional, Dict, List
from .if_source_base import IFSourceBase, IQDeviceConfig, IQComponentConfig
from qudi.hardware.redpitaya.resource_manager import get_pyrpl_instance, release_pyrpl_instance
import logging
import pandas as pd
from scipy.interpolate import interp2d, griddata


class RedPitayaIFSource(IFSourceBase):
    """Red Pitaya implementation of IF source using PyRPL with multi-frequency and FM support."""

    def __init__(self, hostname: str, port: int = 2222, name: str = "RedPitaya"):
        super().__init__(name)
        self.hostname = hostname
        self.port = port
        self.pyrpl = None
        self.fgen3 = None
        self._config_name = None
        self._current_config = None
        self._calibration_data = {}
        self._max_components = 3

    def connect(self, **kwargs) -> None:
        """Connect to the Red Pitaya."""
        try:
            self.logger.info(f"Connecting to Red Pitaya at {self.hostname}")

            # Get config name from kwargs, with a reasonable default
            self._config_name = kwargs.get('config_name', 'rp_default_config')
            fpga_filename = kwargs.get('fpga_filename')
            gui = kwargs.get('gui', True)

            # Use the shared factory to get a pyrpl instance
            self.pyrpl, _ = get_pyrpl_instance(
                hostname=self.hostname,
                config_name=self._config_name,
                gui=gui,
                fpga_filename=fpga_filename
            )
            self.logger.info(f"Acquired shared pyrpl instance for {self.hostname} with config '{self._config_name}'")

            # Initialize fgen3 module
            self.fgen3 = self.pyrpl.rp.fgen3

            # Configure signal routing
            self.fgen3.output_to_dsp_enable_o = True
            self.pyrpl.rp.asg0.output_direct = "out1"
            self.pyrpl.rp.asg1.output_direct = "out2"

            self._is_connected = True
            self.logger.info("Red Pitaya connected successfully")

        except Exception as e:
            self.logger.error(f"Failed to connect to Red Pitaya: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from the Red Pitaya."""
        try:
            if self.fgen3:
                self.fgen3.gen_enable = False
                self.fgen3.output_zero = True
                # Disable all components and their FM
                for i in range(self._max_components):
                    setattr(self.fgen3, f'enable{i}', False)
                    setattr(self.fgen3, f'fm_enable{i}', False)
                self.logger.debug("Red Pitaya output disabled")

            if self.pyrpl:
                # Release our use of the shared instance
                # using the config name we stored during connection
                release_pyrpl_instance(
                    hostname=self.hostname,
                    config_name=self._config_name
                )
                self.pyrpl = None
                self._config_name = None

            self._is_connected = False
            self.logger.info("Red Pitaya disconnected")

        except Exception as e:
            self.logger.error(f"Error disconnecting Red Pitaya: {e}")

    def load_calibration_data(self, frequency: float, path: str) -> pd.DataFrame:
        """Load IQ calibration settings for a specific IF frequency from a CSV file."""
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        try:
            calibration_df = pd.read_csv(path, sep="\t", index_col=0)
            if "sideband" in calibration_df.columns:
                raw_sidebands = calibration_df["sideband"].dropna().unique()
                try:
                    normalized = {self._normalize_sideband(s) for s in raw_sidebands}
                except ValueError as e:
                    self.logger.warning(f'Could not parse sideband column in "{path}": {e}')
                    normalized = set()
                if len(normalized) > 1:
                    self.logger.warning(
                        f'Calibration file "{path}" contains multiple sidebands: {sorted(normalized)}'
                    )
                elif len(normalized) == 1 and next(iter(normalized)) != self.sideband:
                    self.logger.warning(
                        f'Calibration file "{path}" sideband={next(iter(normalized))} '
                        f"does not match current sideband={self.sideband}"
                    )
            self._calibration_data[frequency] = calibration_df
            self.logger.info(f"IQ calibration data for {frequency / 1e6:.3f} MHz loaded from {path}")
            return calibration_df
        except Exception as e:
            self.logger.error(f"Failed to load IQ calibration data: {e}")
            raise RuntimeError(f"Could not load IQ calibration data from {path}") from e

    def _interpolate_calibration_parameters(self, frequency: float, lo_frequency_ghz: float,
                                            if_amplitude: float) -> Tuple[float, float, float, float]:
        """Interpolate calibration parameters for a given frequency and amplitude."""
        if frequency not in self._calibration_data:
            raise RuntimeError(f"Calibration data for {frequency / 1e6:.3f} MHz not loaded")

        calibration_data = self._calibration_data[frequency]

        # Extract the parameter ranges from calibration data
        lo_freq_min = calibration_data['lo_frequency_ghz'].min()
        lo_freq_max = calibration_data['lo_frequency_ghz'].max()
        if_amp_min = calibration_data['if_amplitude'].min()
        if_amp_max = calibration_data['if_amplitude'].max()

        # Check if requested parameters are within bounds
        if not (lo_freq_min <= lo_frequency_ghz <= lo_freq_max):
            raise ValueError(
                f"lo_frequency_ghz ({lo_frequency_ghz}) is outside the calibrated range [{lo_freq_min}, {lo_freq_max}]")

        if not (if_amp_min <= if_amplitude <= if_amp_max):
            raise ValueError(f"if_amplitude ({if_amplitude}) is outside the calibrated range [{if_amp_min}, {if_amp_max}]")

        # Log the interpolation request
        self.logger.debug(
            f"Interpolating calibration for frequency={frequency / 1e6:.3f} MHz, "
            f"lo_frequency_ghz={lo_frequency_ghz}, if_amplitude={if_amplitude}")

        # Prepare the input points for interpolation
        points = calibration_data[['lo_frequency_ghz', 'if_amplitude']].values

        # Parameters to interpolate
        parameters = ['g', 'phi', 'I_offset', 'Q_offset']
        interpolated_values = {}

        # Perform interpolation for each parameter
        for param in parameters:
            values = calibration_data[param].values

            try:
                # Using griddata for more robust interpolation
                try:
                    interpolated_value = griddata(points, values,
                                                  (lo_frequency_ghz, if_amplitude),
                                                  method='cubic')
                except:
                    # Fallback to linear interpolation if cubic fails
                    interpolated_value = griddata(points, values,
                                                  (lo_frequency_ghz, if_amplitude),
                                                  method='linear')

                # Check for NaN results
                if np.isnan(interpolated_value):
                    self.logger.warning(f"Interpolation resulted in NaN for {param}, using nearest neighbor")
                    interpolated_value = griddata(points, values,
                                                  (lo_frequency_ghz, if_amplitude),
                                                  method='nearest')

                interpolated_values[param] = float(interpolated_value)

            except Exception as e:
                self.logger.error(f"Failed to interpolate {param}: {e}")
                raise RuntimeError(f"Could not interpolate {param}") from e

        return (interpolated_values['g'], interpolated_values['phi'],
                interpolated_values['I_offset'], interpolated_values['Q_offset'])

    def load_cal_into_slot(self, slot: int, lo_frequency: float,
                           frequencies: List[float], amplitudes: List[float],
                           calibration_files: Optional[Dict[float, str]] = None) -> None:
        """Interpolate the SSB calibration for ``lo_frequency`` and load it into the
        FPGA cal-slot bank slot ``slot`` (multi-resonance tracking).

        This is the per-slot analogue of :meth:`set_multi_frequency_signal`'s cal
        step: it does NOT change the IF frequencies / FM / component enables (those
        are global, set once via the normal single-LO configuration). It only writes
        the per-LO SSB correction (amplitudes + DAC-B phase + carrier-null DC) into
        the chosen cal slot, which the FPGA selects in lockstep with the LO hop when
        ``fgen3.active_slot_src`` follows ``current_step``.

        Args:
            slot: cal-slot index (0..nslots-1) = resonance index.
            lo_frequency: LO frequency (Hz) for this resonance.
            frequencies: active IF frequencies (Hz), one per component.
            amplitudes: per-component IF amplitudes (0..1), same length.
            calibration_files: optional {if_freq: csv_path}; loaded if not cached.
        """
        if len(frequencies) != len(amplitudes):
            raise ValueError('frequencies and amplitudes must have the same length')
        if not (0 <= slot < self.fgen3.nslots):
            raise ValueError(f'slot must be in 0..{self.fgen3.nslots - 1}, got {slot}')

        # ensure calibration data is loaded for each IF frequency
        for freq in frequencies:
            if freq in self._calibration_data:
                continue
            cal_file = None
            if calibration_files is None:
                cal_file = f'calibration_redpitaya_all_results_IF_{freq / 1e6:.3f}MHz.csv'
            elif freq in calibration_files:
                cal_file = calibration_files[freq]
            if cal_file:
                try:
                    self.load_calibration_data(freq, cal_file)
                except Exception as e:
                    self.logger.warning(
                        f'Could not load calibration for {freq / 1e6:.3f} MHz: {e}')

        comps = []
        dc_offsets_i, dc_offsets_q = [], []
        for freq, amp in zip(frequencies, amplitudes):
            if freq not in self._calibration_data:
                # no cal for this component: identity correction (uncalibrated)
                comps.append((amp, amp, self._q_base_phase_deg()))
                continue
            g, phi, i_offset, q_offset = self._interpolate_calibration_parameters(
                freq, lo_frequency / 1e9, amp)
            amp_a = amp * (1 + g)
            amp_b = amp * (1 - g)
            phase_b = float((self._q_base_phase_deg() + np.degrees(phi)) % 360.0)
            comps.append((amp_a, amp_b, phase_b))
            dc_offsets_i.append(i_offset)
            dc_offsets_q.append(q_offset)

        dc_a = float(np.mean(dc_offsets_i)) if dc_offsets_i else 0.0
        dc_b = float(np.mean(dc_offsets_q)) if dc_offsets_q else 0.0

        self.fgen3.load_cal_slot(slot, comps, dc_a, dc_b)
        self.logger.info(
            f'Loaded SSB cal into slot {slot} for LO={lo_frequency / 1e9:.6f} GHz '
            f'({len(comps)} comps, dc_a={dc_a:.5f}, dc_b={dc_b:.5f})')

    def set_fm_modulation_frequency(self, frequency: float, phase_offset: float = None) -> None:
        """
        Set the FM modulation frequency via iq0 module.

        Parameters:
        -----------
        frequency : float
            FM modulation frequency in Hz
        phase_offset : float, optional
            Phase offset in degrees for IQ demodulation. Adjust to maximize
            1f signal on the ODMR resonance slope. If None, phase is not changed.
        """
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        try:
            # Set the frequency on iq0 module which controls FM modulation
            self.pyrpl.rp.iq0.frequency = frequency
            self.pyrpl.rp.iq0.input = "in1"
            self.pyrpl.rp.iq0.bandwidth = [500, 500]
            self.pyrpl.rp.iq0.output_signal = "quadrature"
            self.pyrpl.rp.iq0.quadrature_factor = 1.0

            # Set phase offset if specified
            if phase_offset is not None:
                self.pyrpl.rp.iq0.phase = phase_offset
                self.logger.info(f"IQ0 phase offset set to {phase_offset:.2f} degrees")

            self.pyrpl.rp.lockin.ref_select1 = "sin_shifted"
            self.pyrpl.rp.lockin.ref_select2 = "cos_shifted"
            self.logger.warning(f"IQ Reference Info: {self.pyrpl.rp.lockin.get_iq_reference_info(channel=1)}")
            #self.pyrpl.rp.iq0.gain = 1.0
            #self.pyrpl.rp.iq0.output_direct = "off"

            self.pyrpl.rp.hk.configure_pin("P1", direction="output", source="module") # reference pin for external lock-in amplifier
            self.logger.info(f"FM modulation frequency set to {frequency / 1e3:.3f} kHz")

            # Update current config if it exists
            if self._current_config:
                self._current_config.fm_modulation_frequency = frequency

        except Exception as e:
            self.logger.error(f"Failed to set FM modulation frequency: {e}")
            raise

    def set_iq_phase_offset(self, phase_offset: float) -> None:
        """
        Set the IQ demodulation phase offset (iq0.phase).

        Parameters:
        -----------
        phase_offset : float
            Phase offset in degrees. Adjust to maximize 1f signal on ODMR resonance slope.
        """
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        try:
            self.pyrpl.rp.iq0.phase = phase_offset
            self.logger.info(f"IQ0 phase offset set to {phase_offset:.2f} degrees")

            # Update current config if it exists
            if self._current_config:
                self._current_config.iq_phase_offset = phase_offset

        except Exception as e:
            self.logger.error(f"Failed to set IQ phase offset: {e}")
            raise

    def configure_signal(self, config: IQDeviceConfig) -> None:
        """Configure the IQ signal parameters for all components."""
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        if config.num_components > self._max_components:
            raise ValueError(f"Too many components ({config.num_components}). Maximum: {self._max_components}")

        self.logger.debug(f"Configuring signal with {config.num_components} components")

        # First, ensure the generator is properly initialized
        self.fgen3.gen_enable = True
        self.fgen3.output_zero = False

        # Disable all components first
        for i in range(self._max_components):
            setattr(self.fgen3, f'enable{i}', False)
            setattr(self.fgen3, f'fm_enable{i}', False)

        # Configure active components
        for i, component in enumerate(config.components):
            if component.enabled:
                self.configure_component(i, component)

        # Set DC offsets (averaged across all active components)
        self.set_dc_offsets(config.dc_offset_i, config.dc_offset_q)

        # Set FM modulation frequency if specified (also sets IQ phase offset if provided)
        if config.fm_modulation_frequency is not None:
            self.set_fm_modulation_frequency(config.fm_modulation_frequency, config.iq_phase_offset)
        elif config.iq_phase_offset is not None:
            # Set phase offset even without FM modulation frequency
            self.set_iq_phase_offset(config.iq_phase_offset)

        self._current_config = config
        self.logger.debug("Signal configured")

    def configure_component(self, component_index: int, config: IQComponentConfig) -> None:
        """Configure a single frequency component including FM settings."""
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        if component_index >= self._max_components:
            raise ValueError(f"Component index {component_index} exceeds maximum ({self._max_components - 1})")

        # Set frequency
        setattr(self.fgen3, f'frequency{component_index}', config.frequency)

        # Set amplitudes
        setattr(self.fgen3, f'amplitude_a{component_index}', config.amplitude_i)
        setattr(self.fgen3, f'amplitude_b{component_index}', config.amplitude_q)

        # Set phase. I-phase (DAC A) is hardwired to 0 in the FPGA (cal-slot bank);
        # only the DAC-B (Q) phase offset is programmable.
        setattr(self.fgen3, f'phase_offset_b{component_index}', config.phase_q)

        # Set FM parameters
        setattr(self.fgen3, f'fm_enable{component_index}', config.fm_enabled)
        if config.fm_enabled:
            setattr(self.fgen3, f'fm_deviation_khz{component_index}', int(config.fm_deviation_khz))

        # Enable the component
        setattr(self.fgen3, f'enable{component_index}', config.enabled)

        self.logger.debug(f"Component {component_index} configured: "
                          f"freq={config.frequency / 1e6:.3f} MHz, "
                          f"amp_i={config.amplitude_i:.3f}, amp_q={config.amplitude_q:.3f}, "
                          f"FM={'on' if config.fm_enabled else 'off'}, "
                          f"FM_dev={config.fm_deviation_khz:.1f} kHz")

    def enable_output(self, enable: bool = True) -> None:
        """Enable or disable the output."""
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        self.fgen3.gen_enable = enable
        self.fgen3.output_zero = not enable
        self.logger.debug(f"Output {'enabled' if enable else 'disabled'}")

    def set_dc_offsets(self, i_offset: float, q_offset: float) -> None:
        """Set DC offsets for I and Q channels."""
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        self.fgen3.overall_dc_offset_a = i_offset
        self.fgen3.overall_dc_offset_b = q_offset

        if self._current_config:
            self._current_config.dc_offset_i = i_offset
            self._current_config.dc_offset_q = q_offset

        self.logger.debug(f"DC offsets set: I={i_offset:.5f}, Q={q_offset:.5f}")

    def set_iq_correction(self, component_index: int, if_amplitude: float,
                          gain_imbalance: float, phase_imbalance: float) -> Tuple[float, float, float, float]:
        """Apply IQ imbalance correction for a specific component."""
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        if component_index >= self._max_components:
            raise ValueError(f"Component index {component_index} exceeds maximum ({self._max_components - 1})")

        # Calculate corrected amplitudes
        amp_i = if_amplitude * (1 + gain_imbalance)
        amp_q = if_amplitude * (1 - gain_imbalance)

        # Calculate corrected phases
        phase_i = 0.0
        phase_q = float((self._q_base_phase_deg() + np.degrees(phase_imbalance)) % 360.0)

        # Apply corrections. I-phase (DAC A) is hardwired to 0 in the FPGA cal-slot
        # bank, so only amplitudes and the DAC-B phase offset are written.
        setattr(self.fgen3, f'amplitude_a{component_index}', amp_i)
        setattr(self.fgen3, f'amplitude_b{component_index}', amp_q)
        setattr(self.fgen3, f'phase_offset_b{component_index}', phase_q)
        setattr(self.fgen3, f'enable{component_index}', True)

        if self._current_config and component_index < len(self._current_config.components):
            comp = self._current_config.components[component_index]
            comp.amplitude_i = amp_i
            comp.amplitude_q = amp_q
            comp.phase_i = phase_i
            comp.phase_q = phase_q

        self.logger.debug(f"IQ correction applied to component {component_index}: "
                          f"g={gain_imbalance:.5f}, phi={phase_imbalance:.5f}")

        return amp_i, amp_q, phase_i, phase_q

    # Backward compatibility methods
    def set_amplitude(self, amplitude: float) -> None:
        """Set the IF signal amplitude (0 to 1) without correction. For backward compatibility."""
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        # Set amplitude on component 0 for backward compatibility
        # (I-phase / DAC A offset is hardwired to 0 in the FPGA cal-slot bank).
        self.fgen3.amplitude_a0 = amplitude
        self.fgen3.amplitude_b0 = amplitude
        self.fgen3.phase_offset_b0 = self._q_base_phase_deg()
        self.fgen3.enable0 = True

        self.logger.debug(f"Amplitude set to {amplitude:.3f} on component 0")

    def set_corrected_amplitude(self, amplitude: float, f_lo: float = 21.158e6) -> None:
        """Set the IF signal amplitude using calibration. For backward compatibility."""
        # Use the convenience method with a single frequency
        # Assume the frequency is already set on component 0
        if hasattr(self.fgen3, 'frequency0') and self.fgen3.frequency0:
            frequency = self.fgen3.frequency0
        else:
            # Default to a common IF frequency if not set
            frequency = 21.580e6

        self.set_multi_frequency_signal([frequency], [amplitude], f_lo)

    def get_current_config(self) -> IQDeviceConfig:
        """Get the current device configuration including FM settings."""
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        if not self._current_config:
            # Read current configuration from device
            components = []
            for i in range(self._max_components):
                if getattr(self.fgen3, f'enable{i}', False):
                    components.append(IQComponentConfig(
                        frequency=getattr(self.fgen3, f'frequency{i}'),
                        amplitude=(getattr(self.fgen3, f'amplitude_a{i}') +
                                   getattr(self.fgen3, f'amplitude_b{i}')) / 2,
                        amplitude_i=getattr(self.fgen3, f'amplitude_a{i}'),
                        amplitude_q=getattr(self.fgen3, f'amplitude_b{i}'),
                        phase_i=0.0,  # I-phase hardwired to 0 in the FPGA cal-slot bank
                        phase_q=getattr(self.fgen3, f'phase_offset_b{i}'),
                        enabled=True,
                        fm_enabled=getattr(self.fgen3, f'fm_enable{i}', False),
                        fm_deviation_khz=getattr(self.fgen3, f'fm_deviation_khz{i}', 0)
                    ))

            # Try to read FM modulation frequency from iq0
            fm_mod_freq = None
            try:
                fm_mod_freq = self.pyrpl.rp.iq0.frequency
            except:
                pass

            self._current_config = IQDeviceConfig(
                components=components,
                dc_offset_i=self.fgen3.overall_dc_offset_a,
                dc_offset_q=self.fgen3.overall_dc_offset_b,
                fm_modulation_frequency=fm_mod_freq
            )

        return self._current_config

    def verify_signal_generation(self) -> dict:
        """Verify that the signal is being generated properly including FM status."""
        if not self._is_connected:
            raise RuntimeError("Red Pitaya not connected")

        # Get FM modulation frequency
        fm_mod_freq = None
        try:
            fm_mod_freq = self.pyrpl.rp.iq0.frequency
        except:
            pass

        status = {
            'gen_enable': self.fgen3.gen_enable,
            'output_zero': self.fgen3.output_zero,
            'dc_offset_a': self.fgen3.overall_dc_offset_a,
            'dc_offset_b': self.fgen3.overall_dc_offset_b,
            'fm_modulation_frequency': fm_mod_freq,
            'components': []
        }

        for i in range(self._max_components):
            comp_status = {
                'index': i,
                'enabled': getattr(self.fgen3, f'enable{i}'),
                'frequency': getattr(self.fgen3, f'frequency{i}'),
                'amplitude_a': getattr(self.fgen3, f'amplitude_a{i}'),
                'amplitude_b': getattr(self.fgen3, f'amplitude_b{i}'),
                'phase_offset_a': 0.0,  # I-phase hardwired to 0 in the FPGA cal-slot bank
                'phase_offset_b': getattr(self.fgen3, f'phase_offset_b{i}'),
                'fm_enabled': getattr(self.fgen3, f'fm_enable{i}'),
                'fm_deviation_khz': getattr(self.fgen3, f'fm_deviation_khz{i}')
            }
            status['components'].append(comp_status)

        self.logger.info("Red Pitaya Signal Status:")
        self.logger.info(f"  Master enable: {status['gen_enable']}")
        self.logger.info(f"  Output zero: {status['output_zero']}")
        self.logger.info(f"  DC offsets: I={status['dc_offset_a']:.5f}, Q={status['dc_offset_b']:.5f}")
        if fm_mod_freq:
            self.logger.info(f"  FM modulation frequency: {fm_mod_freq / 1e3:.3f} kHz")

        for comp in status['components']:
            if comp['enabled']:
                fm_status = f"FM={'on' if comp['fm_enabled'] else 'off'}"
                if comp['fm_enabled']:
                    fm_status += f" (dev={comp['fm_deviation_khz']} kHz)"
                self.logger.info(f"  Component {comp['index']}: "
                                 f"freq={comp['frequency'] / 1e6:.3f} MHz, "
                                 f"amp_i={comp['amplitude_a']:.3f}, amp_q={comp['amplitude_b']:.3f}, "
                                 f"{fm_status}")

        return status


if __name__ == "__main__":
    # Example with FM modulation
    rp_source = RedPitayaIFSource("10.203.129.28")
    rp_source.connect()

    # Define frequencies and amplitudes
    frequencies = [19.422e6, 21.580e6, 23.738e6]  # Hz
    amplitudes = [0.1, 0.1, 0.1]  # 0-1 normalized
    lo_frequency = 2.6e9  # Hz

    # FM parameters
    fm_enables = [True, True, True]  # Enable FM on first two components
    fm_deviations_khz = [300.0, 300.0, 300.0]  # FM deviation in kHz
    fm_modulation_frequency = 14e3

    # Specify calibration files explicitly
    calibration_files = {
        19.422e6: r"C:\Users\aj92uwef\PycharmProjects\qudi-core\qudi-iqo-modules\src\qudi\hardware\microwave\redpitaya\calibration_results\2025-06-06-23-01-25\IF_19.422MHz\calibration_redpitaya_all_results_IF_19.422MHz.csv",
        21.580e6: r"C:\Users\aj92uwef\PycharmProjects\qudi-core\qudi-iqo-modules\src\qudi\hardware\microwave\redpitaya\calibration_results\2025-06-06-23-01-25\IF_21.580MHz\calibration_redpitaya_all_results_IF_21.580MHz.csv",
        23.738e6: r"C:\Users\aj92uwef\PycharmProjects\qudi-core\qudi-iqo-modules\src\qudi\hardware\microwave\redpitaya\calibration_results\2025-06-06-23-01-25\IF_23.738MHz\calibration_redpitaya_all_results_IF_23.738MHz.csv"
    }

    # Configure the multi-frequency signal with calibration and FM
    rp_source.set_multi_frequency_signal(
        frequencies, amplitudes, lo_frequency, calibration_files,
        fm_enables=fm_enables,
        fm_deviations_khz=fm_deviations_khz,
        fm_modulation_frequency=fm_modulation_frequency
    )

    # Enable output
    rp_source.enable_output(True)

    # Verify configuration
    status = rp_source.verify_signal_generation()

    # Keep the connection open for a moment to ensure settings are applied
    import time
    time.sleep(10)

    # Disconnect when done - also ends signal generation
    rp_source.disconnect()
