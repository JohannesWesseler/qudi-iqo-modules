"""
This file contains the Qudi hardware module for a combined microwave source consisting of:
- Red Pitaya as IF source (with IQ calibration and optional FM)
- IQ Mixer for upconversion
- Windfreak SynthNV Pro as LO source

The actual output frequency depends on sideband selection:
- Upper sideband (USB, default): RF = LO + IF
- Lower sideband (LSB):         RF = LO - IF
For multi-frequency excitation: RF_i = LO ± IF_i for each IF frequency

Device quirks:
    - ``Z0``: disable temperature compensation. A firmware bug otherwise forces
    the *sweep* power back to the *CW* power every ~10 s, causing ~10 dB power
    jumps between hops.
    - ``Y0``: trigger polarity active-low. The Windfreak must trigger on the low edge, problems otherwise.
    - ``f?`` (I think) f? reports the *next/pre-loaded* table point, NOT the frequency
    currently being output. Do not use ``f?`` to verify the live output in
    tabular trigger mode -- sometimes it lags the true output by one step - check on scope.
"""

import time
import pyvisa
import numpy as np
from typing import Optional, List

from qudi.util.mutex import Mutex
from qudi.core.configoption import ConfigOption
from qudi.interface.microwave_interface import MicrowaveInterface, MicrowaveConstraints
from qudi.util.enums import SamplingOutputMode

from .redpitaya.redpitaya_if_source import RedPitayaIFSource


class MicrowaveRedPitayaWindfreak(MicrowaveInterface):
    """ Hardware class to control a combined microwave source:
        - Red Pitaya as IF source (with IQ calibration and optional FM)
        - IQ Mixer for upconversion
        - Windfreak SynthNV Pro as LO source

    The actual RF output frequency depends on sideband selection:
    - Upper sideband (USB, default): RF = LO + IF
    - Lower sideband (LSB):         RF = LO - IF
    For multi-frequency excitation: RF_i = LO ± IF_i for each IF frequency

    Example config for copy-paste:

    mw_source_rp_windfreak:
        module.Class: 'microwave.mw_source_windfreak_synthnvpro_redpitaya.MicrowaveRedPitayaWindfreak'
        options:
            windfreak_serial_port: 'COM3'
            windfreak_comm_timeout: 10  # in seconds
            redpitaya_hostname: '10.203.129.28'
            redpitaya_config_name: 'rpy_shared_config'
             if_frequencies: [19.422e6, 21.580e6, 23.738e6]
             if_frequency_index: 1  # Use middle frequency (21.580 MHz) for calculations
             lo_power: 13  # dBm - fixed power for IQ mixer LO input
            sideband: 'upper'  # 'upper' (USB, default) or 'lower' (LSB)
             calibration_files:
                 19.422e6: 'path/to/calibration_IF_19.422MHz.csv'
                 21.580e6: 'path/to/calibration_IF_21.580MHz.csv'
                 23.738e6: 'path/to/calibration_IF_23.738MHz.csv'
            enable_fm: False  # Enable FM modulation capability
            fm_deviation_khz: 100.0  # Default FM deviation in kHz
            fm_modulation_frequency: 5000.0  # Default FM modulation frequency in Hz
            iq0_phase_offset: 0.0  # IQ demodulation phase offset in degrees (adjust for max 1f signal)
            multi_frequency_mode: 'triple'  # Options: 'single', 'dual', 'triple'
            multi_frequency_amplitudes: [0.1, 0.1, 0.1]  # RELATIVE amplitudes for multi-freq mode
    """

    # Windfreak config options
    _windfreak_serial_port = ConfigOption('windfreak_serial_port', missing='error')
    _windfreak_comm_timeout = ConfigOption('windfreak_comm_timeout', default=10, missing='warn')

    # Red Pitaya config options
    _redpitaya_hostname = ConfigOption('redpitaya_hostname', missing='error')
    _redpitaya_port = ConfigOption('redpitaya_port', default=2222, missing='info')
    # This is not the qudi config, but the pyrpl/Red Pitaya config
    _redpitaya_config_name = ConfigOption('redpitaya_config_name', default='rpy_shared_config', missing='info')
    _redpitaya_fpga_filename = ConfigOption(
        'redpitaya_fpga_filename', default=None, missing='info')

    # IF configuration
    _if_frequencies = ConfigOption('if_frequencies', missing='error')
    _if_frequency_index = ConfigOption('if_frequency_index', default=1, missing='info')
    _lo_power = ConfigOption('lo_power', default=13, missing='info')
    _calibration_files = ConfigOption('calibration_files', missing='error')

    # FM configuration
    _enable_fm = ConfigOption('enable_fm', default=False, missing='info')
    _fm_deviation_khz = ConfigOption('fm_deviation_khz', default=100.0, missing='info')
    _fm_modulation_frequency = ConfigOption('fm_modulation_frequency', default=5000.0, missing='info')

    # IQ demodulation phase offset (degrees) - adjust to maximize 1f signal on ODMR slope
    _iq0_phase_offset = ConfigOption('iq0_phase_offset', default=0.0, missing='info')

    # Power calibration
    _power_calibration_table = ConfigOption('power_calibration_table', default=None, missing='info')

    # Multi-frequency configuration
    _multi_frequency_mode = ConfigOption('multi_frequency_mode', default='single', missing='info')
    _multi_frequency_amplitudes = ConfigOption('multi_frequency_amplitudes', default=[0.5, 0.5, 0.5], missing='info')

    # Sideband selection for IQ mixing
    # - 'upper' / 'usb': RF = LO + IF (default)
    # - 'lower' / 'lsb': RF = LO - IF
    _sideband = ConfigOption('sideband', default='upper', missing='info')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._thread_lock = Mutex()

        # Windfreak resources
        self._rm = None
        self._windfreak_device = None
        self._windfreak_model = ''

        # Red Pitaya instance
        self._redpitaya = None

        # Constraints
        self._constraints = None

        # State variables
        self._scan_power = -20
        self._scan_mode = SamplingOutputMode.EQUIDISTANT_SWEEP
        self._scan_frequencies = None
        self._scan_sample_rate = 0.
        self._scan_step_time = 0.
        self._in_cw_mode = True

        # Jump-list (tabular hop) state
        self._list_num_points = 0
        self._list_continuous = False  # c0 (clamp, ODMR) by default; c1 (wrap) for tracking

        # IF frequency for calculations (from config)
        self._if_frequency = None
        self._active_if_frequencies = []
        self._active_if_amplitudes = []

        # Current settings
        self._current_rf_frequency = 2.87e9  # Hz
        self._current_rf_power = -20  # dBm

        # Power calibration data
        self._power_cal_data = None

        # Initialize per-component FM settings
        self._fm_enables_per_component = []
        self._fm_deviations_per_component = []

    @staticmethod
    def _normalize_sideband(sideband: str) -> str:
        sideband_normalized = str(sideband).strip().lower()
        if sideband_normalized in {"upper", "usb"}:
            return "upper"
        if sideband_normalized in {"lower", "lsb"}:
            return "lower"
        raise ValueError(f'Invalid sideband "{sideband}". Use "upper"/"usb" or "lower"/"lsb".')

    def _rf_to_lo_frequency(self, rf_frequency: float) -> float:
        avg_if_freq = self._get_average_if_frequency()
        if self._sideband == "upper":
            return rf_frequency - avg_if_freq
        return rf_frequency + avg_if_freq

    def _lo_to_rf_frequency(self, lo_frequency: float) -> float:
        avg_if_freq = self._get_average_if_frequency()
        if self._sideband == "upper":
            return lo_frequency + avg_if_freq
        return lo_frequency - avg_if_freq

    @property
    def sideband(self) -> str:
        return self._sideband

    def set_sideband(self, sideband: str) -> None:
        with self._thread_lock:
            if self.module_state() != "idle":
                raise RuntimeError("Unable to change sideband. Microwave output active.")

            self._sideband = self._normalize_sideband(sideband)
            if self._redpitaya and self._redpitaya.is_connected:
                self._redpitaya.set_sideband(self._sideband)

            self.log.info(f"Sideband set to {self._sideband.upper()}")

    def on_activate(self):
        """ Initialisation performed during activation of the module. """
        try:
            # Set up IF frequency from config
            self._if_frequency = self._if_frequencies[self._if_frequency_index]
            self.log.info(f'Using IF frequency: {self._if_frequency / 1e6:.3f} MHz for calculations')

            # Initialize multi-frequency settings
            self._update_active_frequencies()

            # Initialize per-component FM settings
            self._fm_enables_per_component = [self._enable_fm] * len(self._if_frequencies)
            self._fm_deviations_per_component = [self._fm_deviation_khz] * len(self._if_frequencies)

            # Load power calibration if available
            if self._power_calibration_table:
                try:
                    import pandas as pd
                    self._power_cal_data = pd.read_csv(self._power_calibration_table)
                    self.log.info(f'Loaded power calibration from {self._power_calibration_table}')
                except Exception as e:
                    self.log.warning(f'Could not load power calibration: {e}')

            # Connect to Windfreak
            self._rm = pyvisa.ResourceManager()
            self._windfreak_device = self._rm.open_resource(
                self._windfreak_serial_port,
                baud_rate=9600,
                read_termination='\n',
                write_termination='\n',
                timeout=int(self._windfreak_comm_timeout * 1000)
            )
            self._windfreak_model = self._windfreak_device.query('+')
            self.log.info(f'Connected to Windfreak: {self._windfreak_model}')

            # Connect to Red Pitaya
            self._redpitaya = RedPitayaIFSource(self._redpitaya_hostname, self._redpitaya_port)
            # Pass the config name to the connect method
            self._redpitaya.connect(
                config_name=self._redpitaya_config_name,
                fpga_filename=self._redpitaya_fpga_filename)
            self.log.info('Connected to Red Pitaya')

            # Sideband selection (must be set before loading calibration files)
            self._sideband = self._normalize_sideband(self._sideband)
            self._redpitaya.set_sideband(self._sideband)

            # Load calibration data
            for freq, cal_file in self._calibration_files.items():
                try:
                    self._redpitaya.load_calibration_data(freq, cal_file)
                    self.log.info(f'Loaded calibration for {freq / 1e6:.3f} MHz')
                except Exception as e:
                    self.log.error(f'Failed to load calibration for {freq / 1e6:.3f} MHz: {e}')

            # Set IQ phase offset from config (if specified and non-zero)
            if self._iq0_phase_offset != 0.0:
                try:
                    self._redpitaya.set_iq_phase_offset(self._iq0_phase_offset)
                    self.log.info(f'IQ0 phase offset set to {self._iq0_phase_offset:.2f} degrees')
                except Exception as e:
                    self.log.warning(f'Could not set IQ phase offset: {e}')

            # Generate constraints based on average IF frequency
            avg_if_freq = self._get_average_if_frequency()
            min_rf = 100e6 + avg_if_freq if self._sideband == 'upper' else 100e6  # conservative
            max_rf = 6.4e9 + avg_if_freq if self._sideband == 'upper' else 6.4e9 - avg_if_freq

            self._constraints = MicrowaveConstraints(
                power_limits=(-50, 13),
                frequency_limits=(min_rf, max_rf),
                scan_size_limits=(2, 2**12),
                sample_rate_limits=(0.1, 2500),
                scan_modes=(SamplingOutputMode.EQUIDISTANT_SWEEP, SamplingOutputMode.JUMP_LIST)
            )

            self._scan_power = -20
            self._scan_frequencies = None
            self._scan_sample_rate = self._constraints.max_sample_rate
            self._in_cw_mode = True

            # Ensure outputs are off on startup
            self.log.info('Ensuring all microwave outputs are off upon activation.')
            # self._windfreak_off() # fixme  : turning off disabled for testing temperature fluctuations when microwave goes off/on
            # self._redpitaya.enable_output(False) # fixme  : turning off disabled for testing temperature fluctuations when microwave goes off/on

        except Exception as e:
            self.log.error(f'Failed to activate module: {e}')
            raise

    def on_deactivate(self):
        """ Cleanup performed during deactivation of the module. """
        try:
            self.off()

            # Disconnect Red Pitaya
            if self._redpitaya:
                self._redpitaya.disconnect()
                self._redpitaya = None

            # Disconnect Windfreak
            if self._windfreak_device:
                self._windfreak_device.close()
                self._windfreak_device = None
            if self._rm:
                self._rm.close()
                self._rm = None

        except Exception as e:
            self.log.error(f'Error during deactivation: {e}')

    def _update_active_frequencies(self):
        """Update the active IF frequencies based on multi-frequency mode."""
        if self._multi_frequency_mode == 'single':
            self._active_if_frequencies = [self._if_frequency]
            self._active_if_amplitudes = [1.0]
        elif self._multi_frequency_mode == 'dual':
            # Use two frequencies symmetrically around the center frequency
            if len(self._if_frequencies) >= 2:
                self._active_if_frequencies = [self._if_frequencies[0], self._if_frequencies[2]] if len(self._if_frequencies) > 2 else self._if_frequencies[:2]
                self._active_if_amplitudes = self._multi_frequency_amplitudes[:2]
            else:
                self.log.warning('Not enough IF frequencies for dual mode, falling back to single')
                self._active_if_frequencies = [self._if_frequency]
                self._active_if_amplitudes = [1.0]
        elif self._multi_frequency_mode == 'triple':
            # Use all three frequencies
            if len(self._if_frequencies) >= 3:
                self._active_if_frequencies = self._if_frequencies[:3]
                self._active_if_amplitudes = self._multi_frequency_amplitudes[:3]
            else:
                self.log.warning('Not enough IF frequencies for triple mode, falling back to single')
                self._active_if_frequencies = [self._if_frequency]
                self._active_if_amplitudes = [1.0]
        else:
            self.log.error(f'Unknown multi-frequency mode: {self._multi_frequency_mode}')
            self._active_if_frequencies = [self._if_frequency]
            self._active_if_amplitudes = [1.0]

        # Normalize amplitudes
        total_amp = sum(self._active_if_amplitudes)
        if total_amp > 0:
            self._active_if_amplitudes = [a / total_amp for a in self._active_if_amplitudes]

    def _get_average_if_frequency(self):
        """Calculate the average of active IF frequencies."""
        return np.mean(self._active_if_frequencies)

    def set_multi_frequency_mode(self, mode: str, amplitudes: Optional[List[float]] = None):
        """Set the multi-frequency mode.

        @param str mode: 'single', 'dual', or 'triple'
        @param List[float] amplitudes: Optional relative amplitudes for each frequency
        """
        with self._thread_lock:
            if self.module_state() != 'idle':
                raise RuntimeError('Unable to change multi-frequency mode. Microwave output active.')

            if mode not in ['single', 'dual', 'triple']:
                raise ValueError(f'Invalid mode: {mode}. Must be "single", "dual", or "triple"')

            self._multi_frequency_mode = mode
            if amplitudes is not None:
                self._multi_frequency_amplitudes = amplitudes

            self._update_active_frequencies()

            # Update constraints based on new average IF frequency
            avg_if_freq = self._get_average_if_frequency()
            min_rf = 100e6 + avg_if_freq if self._sideband == 'upper' else 100e6  # conservative
            max_rf = 6.4e9 + avg_if_freq if self._sideband == 'upper' else 6.4e9 - avg_if_freq

            self._constraints = MicrowaveConstraints(
                power_limits=(-50, 10),
                frequency_limits=(min_rf, max_rf),
                scan_size_limits=(2, 2**12),
                sample_rate_limits=(0.1, 2500),
                scan_modes=(SamplingOutputMode.EQUIDISTANT_SWEEP, SamplingOutputMode.JUMP_LIST)
            )

            self.log.info(
                f'Set multi-frequency mode to {mode} with {len(self._active_if_frequencies)} active frequencies')

    @property
    def constraints(self):
        return self._constraints

    @property
    def is_scanning(self):
        """Read-Only boolean flag indicating if a scan is running at the moment. Can be used together with
        module_state() to determine if the currently running microwave output is a scan or CW.
        Should return False if module_state() is 'idle'.

        @return bool: Flag indicating if a scan is running (True) or not (False)
        """
        with self._thread_lock:
            return (self.module_state() != 'idle') and not self._in_cw_mode

    @property
    def cw_power(self):
        """The CW microwave power in dBm. Must implement setter as well.

        @return float: The currently set CW microwave power in dBm.
        """
        with self._thread_lock:
            return self._current_rf_power

    @property
    def cw_frequency(self):
        """The CW microwave frequency in Hz. Must implement setter as well.

        @return float: The currently set CW microwave frequency in Hz.
        """
        with self._thread_lock:
            lo_frequency = float(self._windfreak_device.query('f?')) * 1e6
            return self._lo_to_rf_frequency(lo_frequency)

    @property
    def scan_power(self):
        """The microwave power in dBm used for scanning. Must implement setter as well.

        @return float: The currently set scanning microwave power in dBm
        """
        with self._thread_lock:
            return self._scan_power

    @property
    def scan_frequencies(self):
        """The microwave frequencies used for scanning. Must implement setter as well.

        In case of scan_mode == SamplingOutputMode.JUMP_LIST, this will be a 1D numpy array.
        In case of scan_mode == SamplingOutputMode.EQUIDISTANT_SWEEP, this will be a tuple
        containing 3 values (freq_begin, freq_end, number_of_samples).
        If no frequency scan has been specified, return None.

        @return float[]: The currently set scanning frequencies. None if not set.
        """
        with self._thread_lock:
            return self._scan_frequencies

    @property
    def scan_mode(self):
        """Scan mode Enum. Must implement setter as well.

        @return SamplingOutputMode: The currently set scan mode Enum
        """
        with self._thread_lock:
            return self._scan_mode

    @property
    def scan_sample_rate(self):
        """Read-only property returning the currently configured scan sample rate in Hz.

        @return float: The currently set scan sample rate in Hz
        """
        with self._thread_lock:
            return self._scan_sample_rate

    def set_cw(self, frequency, power):
        """Configure the CW microwave output. Does not start physical signal output, see also
        "cw_on".

        @param float frequency: RF frequency to set in Hz (center frequency for multi-freq mode)
        @param float power: Requested power in dBm (does not change Windfreak LO power; that is set via `lo_power`).
                            This value is mapped to an IF amplitude via `_power_to_if_amplitude()`.
                            In multi-frequency modes, the resulting IF amplitude is distributed across the
                            active components according to `multi_frequency_amplitudes`.
        """
        with self._thread_lock:
            if self.module_state() != 'idle':
                raise RuntimeError('Unable to set CW parameters. Microwave output active.')
            self._assert_cw_parameters_args(frequency, power)

            self._current_rf_frequency = frequency
            self._current_rf_power = power

            # Calculate required LO frequency based on average IF and sideband selection
            lo_frequency = self._rf_to_lo_frequency(frequency)

            # Configure Windfreak for CW at calculated LO frequency
            self._windfreak_device.write('X0')  # sweep mode off
            self._windfreak_device.write('c1')  # continuous mode
            self._windfreak_device.write('y0')  # trigger mode: software
            self._windfreak_device.write(f'f{lo_frequency / 1e6:5.7f}')
            self._windfreak_device.write(f'l{lo_frequency / 1e6:5.7f}')
            self._windfreak_device.write(f'u{lo_frequency / 1e6:5.7f}')

            # Calculate IF amplitudes for each frequency component
            if_amplitudes = []
            for amp_ratio in self._active_if_amplitudes:
                if_amplitude = self._power_to_if_amplitude(power) * amp_ratio
                if_amplitudes.append(if_amplitude)

            # Use per-component FM settings if available
            fm_enables = getattr(self, '_fm_enables_per_component',
                                 [self._enable_fm] * len(self._active_if_frequencies))
            fm_deviations = getattr(self, '_fm_deviations_per_component',
                                    [self._fm_deviation_khz] * len(self._active_if_frequencies))

            # Only use FM settings for active components
            fm_enables = fm_enables[:len(self._active_if_frequencies)]
            fm_deviations = fm_deviations[:len(self._active_if_frequencies)]

            fm_mod_freq = self._fm_modulation_frequency if any(fm_enables) else None

            # Build calibration files dict for active frequencies
            active_cal_files = {freq: self._calibration_files[freq]
                                for freq in self._active_if_frequencies
                                if freq in self._calibration_files}

            # Use the calibrated multi-frequency setup
            self._redpitaya.set_multi_frequency_signal(
                frequencies=self._active_if_frequencies,
                amplitudes=if_amplitudes,
                lo_frequency=lo_frequency,
                calibration_files=active_cal_files,
                fm_enables=fm_enables,
                fm_deviations_khz=fm_deviations,
                fm_modulation_frequency=fm_mod_freq
            )

            self.log.debug(f'CW configured: RF center={frequency / 1e9:.4f} GHz, LO={lo_frequency / 1e9:.4f} GHz, '
                           f'Active IFs={[f / 1e6 for f in self._active_if_frequencies]} MHz, Power={power} dBm, '
                           f'Mode={self._multi_frequency_mode}, FM enables={fm_enables}')

    def configure_scan(self, power, frequencies, mode, sample_rate):
        """Configure frequency scan."""
        with self._thread_lock:
            # Sanity checks
            if self.module_state() != 'idle':
                raise RuntimeError('Unable to configure frequency scan. Microwave output active.')
            self._assert_scan_configuration_args(power, frequencies, mode, sample_rate)

            self._scan_power = power
            self._scan_mode = mode
            self._scan_sample_rate = sample_rate

            # Calculate IF amplitudes for each frequency component
            if_amplitudes = []
            for amp_ratio in self._active_if_amplitudes:
                if_amplitude = self._power_to_if_amplitude(power) * amp_ratio
                if_amplitudes.append(if_amplitude)

            # Use per-component FM settings
            fm_enables = getattr(self, '_fm_enables_per_component',
                                 [self._enable_fm] * len(self._active_if_frequencies))
            fm_deviations_khz = getattr(self, '_fm_deviations_per_component',
                                        [self._fm_deviation_khz] * len(self._active_if_frequencies))

            # Only use FM settings for active components
            fm_enables = fm_enables[:len(self._active_if_frequencies)]
            fm_deviations_khz = fm_deviations_khz[:len(self._active_if_frequencies)]

            fm_mod_freq = self._fm_modulation_frequency if any(fm_enables) else None

            # Calculate average IF frequency for LO calculations
            avg_if_freq = self._get_average_if_frequency()

            if mode == SamplingOutputMode.EQUIDISTANT_SWEEP:
                # For sweep mode, calculate LO frequencies
                rf_start, rf_stop, num_points = frequencies
                if self._sideband == 'upper':
                    lo_start = rf_start - avg_if_freq
                    lo_stop = rf_stop - avg_if_freq
                else:
                    lo_start = rf_start + avg_if_freq
                    lo_stop = rf_stop + avg_if_freq

                # Use mid-point LO frequency for calibration
                lo_mid = (lo_start + lo_stop) / 2

                # Build calibration files dict for active frequencies
                active_cal_files = {freq: self._calibration_files[freq]
                                    for freq in self._active_if_frequencies
                                    if freq in self._calibration_files}

                # Configure Red Pitaya with multi-frequency signal
                self._redpitaya.set_multi_frequency_signal(
                    frequencies=self._active_if_frequencies,
                    amplitudes=if_amplitudes,
                    lo_frequency=lo_mid,
                    calibration_files=active_cal_files,
                    fm_enables=fm_enables,
                    fm_deviations_khz=fm_deviations_khz,
                    fm_modulation_frequency=fm_mod_freq
                )

                # Configure Windfreak for sweep
                self._configure_windfreak_sweep(lo_start, lo_stop, num_points, sample_rate)
                self._scan_frequencies = tuple(frequencies)

            elif mode == SamplingOutputMode.JUMP_LIST:
                # For jump list, calculate all LO frequencies
                rf_frequencies = np.asarray(frequencies, dtype=np.float64)
                lo_frequencies = rf_frequencies - avg_if_freq if self._sideband == 'upper' else rf_frequencies + avg_if_freq

                # Use mid-point LO frequency for calibration
                lo_mid = np.mean([np.min(lo_frequencies), np.max(lo_frequencies)])

                # Build calibration files dict for active frequencies
                active_cal_files = {freq: self._calibration_files[freq]
                                    for freq in self._active_if_frequencies
                                    if freq in self._calibration_files}

                # Configure Red Pitaya with multi-frequency signal
                self._redpitaya.set_multi_frequency_signal(
                    frequencies=self._active_if_frequencies,
                    amplitudes=if_amplitudes,
                    lo_frequency=lo_mid,
                    calibration_files=active_cal_files,
                    fm_enables=fm_enables,
                    fm_deviations_khz=fm_deviations_khz,
                    fm_modulation_frequency=fm_mod_freq
                )

                # Configure Windfreak for jump list
                self._configure_windfreak_list(lo_frequencies, sample_rate)
                self._scan_frequencies = rf_frequencies

            # Wait for configuration to settle
            time.sleep(0.2)

            self.log.debug(f'Configured scan: mode={mode}, power={power} dBm, '
                           f'sample_rate={sample_rate} Hz, multi_freq_mode={self._multi_frequency_mode}, '
                           f'FM enables={fm_enables}')

    def off(self):
        """Switches off any microwave output (both scan and CW)."""
        with self._thread_lock:
            if self.module_state() != 'idle':
                # Turn off Windfreak
                #self._windfreak_off() # fixme  : turning off disabled for testing temperature fluctuations when microwave goes off/on
                # Turn off Red Pitaya
                # self._redpitaya.enable_output(False) #fixme  : turning off disabled for testing temperature fluctuations when microwave goes off/on
                self.module_state.unlock()
                self.log.debug('All outputs turned off')

    def cw_on(self):
        """Switches on cw microwave output."""
        with self._thread_lock:
            if self.module_state() != 'idle':
                if self._in_cw_mode:
                    return
                raise RuntimeError('Unable to start CW microwave output. Microwave output is currently active.')

            self._in_cw_mode = True

            # Enable Red Pitaya output first
            self._redpitaya.enable_output(True)
            time.sleep(0.1)  # Small delay to ensure IF is stable

            # Enable Windfreak
            self._windfreak_on()

            # For Windfreak, ensure we're in the right mode
            self._windfreak_device.write('g1')  # Enable generator

            self.module_state.lock()
            self.log.debug('CW output enabled')

    def start_scan(self):
        """Switches on the microwave scanning.

        Must return AFTER the output is actually active (and can receive triggers for example).
        """
        with self._thread_lock:
            if self.module_state() != 'idle':
                if not self._in_cw_mode:
                    return
                raise RuntimeError('Unable to start frequency scan. CW microwave output is active.')

            assert self._scan_frequencies is not None, 'No scan_frequencies set. Unable to start scan.'

            self._in_cw_mode = False

            # Enable Red Pitaya output
            self._redpitaya.enable_output(True)
            time.sleep(0.1)  # Small delay to ensure IF is stable

            # Enable Windfreak output
            self._windfreak_on()

            # Start Windfreak scan
            if self._scan_mode == SamplingOutputMode.EQUIDISTANT_SWEEP:
                self._windfreak_device.write('g1g0')  # Enable and reset sweep
            else:
                # Jump list: arm the table at point 0. The output then advances one
                # table point per external trigger, wrapping after the
                # last point (no redundant trigger -- scope-verified).
                self.arm_list()

            self.module_state.lock()
            self.log.debug('Scan started')

    def reset_scan(self):
        """Reset currently running scan and return to start frequency.
        Does not need to stop and restart the microwave output if the device allows soft scan reset.
        """
        with self._thread_lock:
            if self.module_state() == 'idle':
                return
            if self._in_cw_mode:
                raise RuntimeError('Can not reset frequency scan. CW microwave output active.')

            if self._scan_mode == SamplingOutputMode.EQUIDISTANT_SWEEP:
                # Reset Windfreak sweep
                self._windfreak_device.write('g1g0')
            else:
                # Jump list: re-arm at point 0 (each subsequent trigger advances one point).
                self.arm_list()

            self.log.debug('Scan reset')

    def _configure_windfreak_sweep(self, lo_start, lo_stop, points, sample_rate):
        """Configure Windfreak for sweep mode."""
        step = (lo_stop - lo_start) / (points - 1)

        # Set step time
        step_time_ms = 1000 * 0.75 / sample_rate
        self._windfreak_device.write(f't{step_time_ms:f}')
        self._scan_step_time = 0.75 / sample_rate

        # Disable temperature compensation (Windfreak bug workaround)
        self._windfreak_device.write('Z0')

        # Configure sweep parameters
        self._windfreak_device.write('X0')  # sweep mode: linear sweep
        self._windfreak_device.write('c0')  # non-continuous
        self._windfreak_device.write('y2')  # trigger mode: single step

        # Set sweep direction
        if lo_stop >= lo_start:
            self._windfreak_device.write('^1')
        else:
            self._windfreak_device.write('^0')

        # Set frequencies
        self._windfreak_device.write(f'l{lo_start / 1e6:5.7f}')
        self._windfreak_device.write(f'u{lo_stop / 1e6:5.7f}')
        self._windfreak_device.write(f's{step / 1e6:5.7f}')

        # Set power (constant LO power)
        self._windfreak_device.write(f'W{self._lo_power:2.3f}')
        self._windfreak_device.write(f'[{self._lo_power:2.3f}')  # sweep lower power
        self._windfreak_device.write(f']{self._lo_power:2.3f}')  # sweep upper power

    def _configure_windfreak_list(self, lo_frequencies, sample_rate, continuous=False):
        """Configure Windfreak for jump-list (tabular hop) mode.

        First frequency convention (consistent across both modes): after ``g1g0`` the output
        sits on table point 0; the ODMR logic issues one ``generate_pulse`` sync
        trigger (which holds frequency 0, absorbing the device's first-jump quirk) and then
        ``acquire_frame`` steps through points 1..N. The tracking mode instead reads
        ``current_step`` and never relies on a fixed trigger phase.

        """
        wf = self._windfreak_device

        # Step time
        step_time_ms = 1000 * 0.75 / sample_rate
        self._scan_step_time = 0.75 / sample_rate

        wf.write('g0')                       # stop any running sweep before reprogramming
        wf.write('Z0')                       # disable temp-comp (power-stability bug workaround)
        wf.write(f'W{self._lo_power:2.3f}')  # base/CW power
        wf.write(f't{step_time_ms:f}')

        # --- program the tabular hop table ---
        wf.write('Ld')                       # clear any previous table
        time.sleep(0.05)
        for i, freq in enumerate(lo_frequencies):
            wf.write(f'L{i}f{freq / 1e6:.6f}')
            wf.write(f'L{i}a{self._lo_power:2.3f}')
            time.sleep(0.02)

        # --- sweep type / trigger configuration ---
        wf.write('X1')                       # tabular hop table
        wf.write('y2')                       # advance one point per external trigger
        wf.write('Y0')                       # trigger polarity active-low
        wf.write('^1')                       # normal step order
        wf.write('c1' if continuous else 'c0')  # c0=clamp (ODMR, matches sweep) / c1=wrap (cycling the list)
        time.sleep(0.1)
        wf.write('g1g0')                     # arm: output sits at point 0, then waits for triggers
        time.sleep(0.1)

        self._list_num_points = len(lo_frequencies)
        self._list_continuous = continuous

    def arm_list(self):
        """(Re)arm the jump list so the output sits on table point 0.

        After this call the LO outputs the first table frequency (point 0); each
        subsequent external (Red Pitaya) trigger advances the output by exactly one
        table point, wrapping after the last (verified on the scope -- there is no
        redundant priming trigger). Call before starting a trigger sequence to
        guarantee a known starting point.
        """
        self._windfreak_device.write('g1g0')

    # -------------------------------------------------------------------------
    # Multi-resonance tracking: LO jump-list + per-slot SSB calibration
    # -------------------------------------------------------------------------
    def rf_to_lo(self, rf_frequency: float) -> float:
        """Public sideband-aware RF center frequency -> required LO frequency (Hz)."""
        return self._rf_to_lo_frequency(rf_frequency)

    def configure_jump_list(self, rf_frequencies, sample_rate, continuous=True,
                            enable_output=True):
        """Program the Windfreak JUMP_LIST from per-resonance RF center frequencies.

        Converts each RF center to its LO (sideband-aware), programs the tabular hop
        table, and sets one-step-per-trigger (``y2``), wrap (``c1`` when
        ``continuous``) so each Red Pitaya hop trigger advances the LO by one point
        and the table cycles indefinitely (the multi-resonance tracking LO source).

        Args:
            rf_frequencies: list of per-resonance RF center frequencies (Hz), in the
                order they map to scan ``current_step`` 0..N-1.
            sample_rate: hop rate (Hz); sets the Windfreak step time (0.75/rate).
            continuous: c1 (wrap) for indefinite tracking; c0 (clamp) otherwise.
            enable_output: also enable the RF output (E1 h1).

        Returns:
            list of LO frequencies (Hz) actually programmed (same order).
        """
        lo_frequencies = [self._rf_to_lo_frequency(float(f)) for f in rf_frequencies]
        with self._thread_lock:
            self._configure_windfreak_list(lo_frequencies, sample_rate, continuous=continuous)
            if enable_output:
                self._windfreak_device.write('E1h1')
        self.log.info(
            f'Jump-list configured: RF={[round(f/1e9, 6) for f in rf_frequencies]} GHz -> '
            f'LO={[round(f/1e9, 6) for f in lo_frequencies]} GHz, '
            f'rate={sample_rate:.1f} Hz, continuous={continuous}'
        )
        return lo_frequencies

    def calibration_load_slot(self, slot, rf_frequency, power=None):
        """Interpolate and load the per-LO SSB calibration for the resonance at RF
        center ``rf_frequency`` into FPGA cal-slot ``slot``.

        Uses the currently active IF frequencies/amplitudes (the shared triplet). The
        IF tones / FM are global (configured once); this only writes the per-LO SSB
        correction into the slot the FPGA selects in lockstep with the hop.
        """
        lo_frequency = self._rf_to_lo_frequency(float(rf_frequency))
        power = self._current_rf_power if power is None else power
        if_amplitudes = [self._power_to_if_amplitude(power) * amp_ratio
                         for amp_ratio in self._active_if_amplitudes]
        active_cal_files = {freq: self._calibration_files[freq]
                            for freq in self._active_if_frequencies
                            if freq in self._calibration_files}
        self._redpitaya.load_cal_into_slot(
            slot, lo_frequency, self._active_if_frequencies, if_amplitudes,
            calibration_files=active_cal_files)
        self.log.info(
            f'Cal slot {slot} loaded for RF={rf_frequency/1e9:.6f} GHz '
            f'(LO={lo_frequency/1e9:.6f} GHz)')

    def set_cal_slot_source(self, hardware: bool = True):
        """Make the fgen3 cal-slot follow the live hop index (``current_step``) when
        True (the tracking mode), or use the software-selected slot when False."""
        self._redpitaya.fgen3.active_slot_src = bool(hardware)
        self.log.info(f'fgen3 cal-slot source = '
                      f'{"current_step (hardware)" if hardware else "software"}')

    def _windfreak_off(self):
        """Turn off Windfreak output."""
        # disable sweep mode
        self._windfreak_device.write('g0')
        # set trigger source to software
        self._windfreak_device.write('y0')
        self._windfreak_device.write('E0h0')
        return self._windfreak_stat()

    def _windfreak_on(self):
        """Turn on Windfreak output."""
        self._windfreak_device.write(f'W{self._lo_power}')
        self._windfreak_device.write('E1h1')
        return self._windfreak_stat()

    def _windfreak_stat(self):
        """Return Windfreak status."""
        try:
            E = int(self._windfreak_device.query('E?'))
            h = int(self._windfreak_device.query('h?'))
            return E, h
        except Exception as e:
            self.log.error(f'Failed to get Windfreak status: {e}')
            return 0, 0

    def _power_to_if_amplitude(self, power_dbm):
        """Convert requested power (dBm) to Red Pitaya IF amplitude.

        This module maps the Qudi microwave `power` setting to the Red Pitaya IF sine-wave peak
        amplitude (`if_amplitude`, in volts, 0..1).

        If a `power_calibration_table` is provided, it is used for the mapping.
        Without a table, this is only an electrical 50 Ω equivalence at the IF output (not a calibrated RF output power).

        Fallback (no table):
        - Assumes a 50 Ω load
        - Interprets `if_amplitude` as V_peak (not V_rms)
        - Uses P = V_rms^2 / R = V_peak^2 / (2R), hence for R=50 Ω:
          P(dBm) = 20*log10(V_peak) + 10
        """
        if self._power_cal_data is not None:
            try:
                # Interpolate from calibration data
                # Assuming calibration data has columns: 'power_dbm', 'if_amplitude'
                return np.interp(power_dbm,
                                 self._power_cal_data['power_dbm'],
                                 self._power_cal_data['if_amplitude'])
            except Exception as e:
                self.log.warning(f'Error using power calibration data: {e}')

        if_amplitude = 10 ** ((power_dbm - 10) / 20)

        return if_amplitude

    def _if_amplitude_to_power(self, if_amplitude):
        """Convert Red Pitaya IF amplitude to equivalent power (dBm).

        If a `power_calibration_table` is provided, it is used for the mapping.
        Otherwise assumes a 50 Ω load and interprets `if_amplitude` as sine-wave peak voltage V_peak (0..1),
        i.e. P(dBm) = 20*log10(V_peak) + 10.
        """
        if self._power_cal_data is not None:
            try:
                # Interpolate from calibration data
                return np.interp(if_amplitude,
                                 self._power_cal_data['if_amplitude'],
                                 self._power_cal_data['power_dbm'])
            except Exception as e:
                self.log.warning(f'Error using power calibration data: {e}')

        power_dbm = 20 * np.log10(if_amplitude) + 10

        return power_dbm

    def set_component_enabled(self, component_index: int, enabled: bool):
        """Enable or disable a specific frequency component.

        @param int component_index: Index of the component (0-2)
        @param bool enabled: True to enable, False to disable
        """
        with self._thread_lock:
            if self.module_state() != 'idle':
                raise RuntimeError('Unable to change component state. Microwave output active.')

            if component_index >= len(self._if_frequencies):
                raise ValueError(f'Component index {component_index} out of range')

            # This would need to be implemented with a new data structure
            # to track which components are enabled
            # For now, we'll implement this through the Red Pitaya
            if self._redpitaya and self._redpitaya.is_connected:
                config = self._redpitaya.get_current_config()
                if config and component_index < len(config.components):
                    config.components[component_index].enabled = enabled
                    self._redpitaya.configure_signal(config)

    def set_fm_parameters(self, enable=None, deviation_khz=None, modulation_frequency=None):
        """Set FM modulation parameters.

        @param bool enable: Enable/disable FM modulation
        @param float deviation_khz: FM deviation in kHz
        @param float modulation_frequency: FM modulation frequency in Hz
        """
        with self._thread_lock:
            if enable is not None:
                self._enable_fm = bool(enable)
                # Also update the per-component enables list, which is what
                # configure_scan() actually reads when setting up fgen3 FM.
                if hasattr(self, '_fm_enables_per_component') and self._fm_enables_per_component:
                    self._fm_enables_per_component = [self._enable_fm] * len(self._fm_enables_per_component)
                elif hasattr(self, '_if_frequencies'):
                    self._fm_enables_per_component = [self._enable_fm] * len(self._if_frequencies)

            if deviation_khz is not None:
                self._fm_deviation_khz = float(deviation_khz)
                # This is the crucial part: we must also update the per-component list
                # that is actually used by configure_scan.
                if hasattr(self, '_fm_deviations_per_component'):
                    num_components = len(self._fm_deviations_per_component)
                    self._fm_deviations_per_component = [self._fm_deviation_khz] * num_components
                else:
                    # Fallback in case the list doesn't exist yet for some reason
                    num_if_freqs = len(getattr(self, '_if_frequencies', [1]))
                    self._fm_deviations_per_component = [self._fm_deviation_khz] * num_if_freqs

            if modulation_frequency is not None:
                self._fm_modulation_frequency = float(modulation_frequency)

            self.log.info(f'FM parameters updated: enable={self._enable_fm}, '
                          f'deviation={self._fm_deviation_khz} kHz, '
                          f'mod_freq={self._fm_modulation_frequency} Hz')

            # Toggle lock-in demodulation bypass based on FM enable state.
            # When FM is disabled, we use DC ODMR mode (bypass demodulation).
            # When FM is enabled, we need lock-in demodulation (disable bypass).
            if enable is not None and self._redpitaya and self._redpitaya.pyrpl:
                try:
                    lockin = self._redpitaya.pyrpl.rp.lockin
                    bypass = not self._enable_fm
                    lockin.demod_bypass_ch1 = bypass
                    lockin.demod_bypass_ch2 = bypass
                    self.log.info(f'Lock-in demod bypass set to {bypass} '
                                  f'(DC ODMR mode {"enabled" if bypass else "disabled"})')
                except Exception as e:
                    self.log.warning(f'Could not set lock-in demod bypass: {e}')

            # If CW is currently on, update the configuration
            if self.module_state() != 'idle' and self._in_cw_mode:
                self.set_cw(self._current_rf_frequency, self._current_rf_power)

    def set_fm_per_component(self, component_index: int, fm_enabled: bool, fm_deviation_khz: float = None):
        """Set FM parameters for a specific component.

        @param int component_index: Index of the component (0-2)
        @param bool fm_enabled: Enable/disable FM for this component
        @param float fm_deviation_khz: FM deviation in kHz (optional)
        """
        with self._thread_lock:
            if self.module_state() != 'idle':
                raise RuntimeError('Unable to change FM settings. Microwave output active.')

            if component_index >= len(self._if_frequencies):
                raise ValueError(f'Component index {component_index} out of range')

            # Store per-component FM settings
            if not hasattr(self, '_fm_enables_per_component'):
                self._fm_enables_per_component = [self._enable_fm] * len(self._if_frequencies)
            if not hasattr(self, '_fm_deviations_per_component'):
                self._fm_deviations_per_component = [self._fm_deviation_khz] * len(self._if_frequencies)

            self._fm_enables_per_component[component_index] = fm_enabled
            if fm_deviation_khz is not None:
                self._fm_deviations_per_component[component_index] = fm_deviation_khz

            self.log.info(f'Component {component_index} FM settings: enabled={fm_enabled}, '
                          f'deviation={self._fm_deviations_per_component[component_index]} kHz')

    def set_dc_odmr_mode(self, enable=True):
        """Enable or disable direct DC ODMR mode (bypasses lock-in demodulation).

        When enabled:
        - FM modulation is disabled on fgen3
        - Lock-in demodulation bypass is activated (ADC goes through CIC/FIR only)
        - The ODMR signal is measured as a direct DC fluorescence level change

        When disabled:
        - FM modulation and lock-in demodulation are restored

        @param bool enable: True to enable DC ODMR mode, False to restore lock-in mode
        """
        self.set_fm_parameters(enable=not enable)

    def get_multi_frequency_info(self):
        """Get information about the current multi-frequency configuration.

        @return dict: Dictionary containing multi-frequency information
        """
        with self._thread_lock:
            avg_if = self._get_average_if_frequency()

            # Calculate actual RF frequencies for each IF component
            if self.module_state() != 'idle':
                try:
                    lo_freq = float(self._windfreak_device.query('f?')) * 1e6
                    if self._sideband == 'upper':
                        rf_frequencies = [lo_freq + if_freq for if_freq in self._active_if_frequencies]
                    else:
                        rf_frequencies = [lo_freq - if_freq for if_freq in self._active_if_frequencies]
                except:
                    rf_frequencies = None
            else:
                rf_frequencies = None

            return {
                'sideband': self._sideband,
                'mode': self._multi_frequency_mode,
                'active_if_frequencies': self._active_if_frequencies.copy(),
                'active_if_amplitudes': self._active_if_amplitudes.copy(),
                'average_if_frequency': avg_if,
                'rf_frequencies': rf_frequencies,
                'fm_enabled': self._enable_fm,
                'fm_deviation_khz': self._fm_deviation_khz,
                'fm_modulation_frequency': self._fm_modulation_frequency,
                'fm_enables_per_component': getattr(self, '_fm_enables_per_component',
                                                    [self._enable_fm] * len(self._if_frequencies)),
                'fm_deviations_per_component': getattr(self, '_fm_deviations_per_component',
                                                       [self._fm_deviation_khz] * len(self._if_frequencies)),
                'iq0_phase_offset': self._iq0_phase_offset
            }

    def set_iq_phase_offset(self, phase_offset: float):
        """Set the IQ demodulation phase offset (iq0.phase).

        This adjusts the phase between modulation and demodulation in the IQ module.
        Fine-tune this value to maximize the 1f signal amplitude on the ODMR resonance slope.

        @param float phase_offset: Phase offset in degrees
        """
        with self._thread_lock:
            self._iq0_phase_offset = float(phase_offset)

            # Apply to Red Pitaya if connected
            if self._redpitaya and self._redpitaya.is_connected:
                try:
                    self._redpitaya.set_iq_phase_offset(phase_offset)
                    self.log.info(f'IQ phase offset set to {phase_offset:.2f} degrees')
                except Exception as e:
                    self.log.error(f'Failed to set IQ phase offset: {e}')

    # ─────────────────────────────────────────────────────────────────────────
    # IQ Calibration Methods
    # ─────────────────────────────────────────────────────────────────────────
    # These methods expose the RedPitayaIFSource calibration functions for use
    # during IQ mixer calibration workflows.

    def calibration_set_sideband(self, sideband: str) -> None:
        """Set sideband selection for IQ mixing during calibration.

        @param str sideband: 'upper'/'usb' or 'lower'/'lsb'
        """
        self.set_sideband(sideband)

    def calibration_set_dc_offsets(self, i_offset: float, q_offset: float) -> None:
        """Set DC offsets on the RedPitaya fgen3 for IQ calibration.

        This method directly sets the DC offsets used for LO leakage cancellation.
        DC offsets are in normalized units (-1.0 to 1.0).

        @param float i_offset: I channel DC offset (-1.0 to 1.0)
        @param float q_offset: Q channel DC offset (-1.0 to 1.0)
        """
        with self._thread_lock:
            if self._redpitaya and self._redpitaya.is_connected:
                self._redpitaya.set_dc_offsets(i_offset, q_offset)
                self.log.debug(f'Calibration DC offsets set: I={i_offset:.5f}, Q={q_offset:.5f}')
            else:
                raise RuntimeError('Red Pitaya not connected')

    def calibration_set_iq_correction(self, g: float, phi: float, amplitude: float,
                                       component_index: int = 0) -> tuple:
        """Apply IQ imbalance correction to fgen3 for calibration.

        This applies gain imbalance (g) and phase imbalance (phi) corrections
        to minimize image power during IQ mixer calibration.

         Correction formulas:
         - I amplitude = amplitude * (1 + g)
         - Q amplitude = amplitude * (1 - g)
         - I phase = 0°
        - Q phase = base_q + degrees(phi), where base_q depends on sideband:
          - USB (default): base_q = 270°
          - LSB: base_q = 90°

        @param float g: Gain imbalance parameter (typically -0.1 to 0.1)
        @param float phi: Phase imbalance in radians (typically 0 to 0.5)
        @param float amplitude: Base IF amplitude (0.0 to 1.0)
        @param int component_index: Frequency component index (default 0)

        @return tuple: (amp_i, amp_q, phase_i, phase_q) - the corrected values applied
        """
        with self._thread_lock:
            if self._redpitaya and self._redpitaya.is_connected:
                result = self._redpitaya.set_iq_correction(
                    component_index, amplitude, g, phi
                )
                self.log.debug(f'IQ correction applied: g={g:.5f}, phi={phi:.5f}, '
                              f'amp={amplitude:.3f} -> amp_i={result[0]:.4f}, '
                              f'amp_q={result[1]:.4f}, phase_q={result[3]:.2f}°')
                return result
            else:
                raise RuntimeError('Red Pitaya not connected')

    def calibration_set_if_amplitude(self, amplitude: float, component_index: int = 0) -> None:
        """Set the IF signal amplitude for calibration (uncorrected).

        This sets equal I and Q amplitudes without IQ imbalance correction.
        Used to set the initial amplitude before applying corrections.

        @param float amplitude: IF amplitude (0.0 to 1.0)
        @param int component_index: Frequency component index (default 0)
        """
        with self._thread_lock:
            if self._redpitaya and self._redpitaya.is_connected:
                max_components = int(getattr(self._redpitaya, 'max_components', 3))
                if not (0 <= component_index < max_components):
                    raise ValueError(f'component_index {component_index} out of range (0..{max_components - 1})')

                # Ensure only the selected component contributes to the output during calibration.
                for i in range(max_components):
                    setattr(self._redpitaya.fgen3, f'enable{i}', False)
                    setattr(self._redpitaya.fgen3, f'fm_enable{i}', False)

                setattr(self._redpitaya.fgen3, f'amplitude_a{component_index}', amplitude)
                setattr(self._redpitaya.fgen3, f'amplitude_b{component_index}', amplitude)
                setattr(self._redpitaya.fgen3, f'phase_offset_a{component_index}', 0.0)
                setattr(self._redpitaya.fgen3, f'phase_offset_b{component_index}',
                        270.0 if self._sideband == 'upper' else 90.0)
                setattr(self._redpitaya.fgen3, f'enable{component_index}', True)
                self.log.debug(f'IF amplitude set to {amplitude:.3f} (uncorrected)')
            else:
                raise RuntimeError('Red Pitaya not connected')

    def calibration_set_if_frequency(self, frequency_hz: float, component_index: int = 0) -> None:
        """Set the IF frequency for calibration.

        @param float frequency_hz: IF frequency in Hz
        @param int component_index: Frequency component index (default 0)
        """
        with self._thread_lock:
            if self._redpitaya and self._redpitaya.is_connected:
                max_components = int(getattr(self._redpitaya, 'max_components', 3))
                if not (0 <= component_index < max_components):
                    raise ValueError(f'component_index {component_index} out of range (0..{max_components - 1})')

                # Ensure only the selected component contributes to the output during calibration.
                for i in range(max_components):
                    setattr(self._redpitaya.fgen3, f'enable{i}', False)
                    setattr(self._redpitaya.fgen3, f'fm_enable{i}', False)

                setattr(self._redpitaya.fgen3, f'frequency{component_index}', frequency_hz)
                setattr(self._redpitaya.fgen3, f'fm_enable{component_index}', False)
                setattr(self._redpitaya.fgen3, f'enable{component_index}', True)
                self.log.debug(f'IF frequency set to {frequency_hz / 1e6:.3f} MHz')
            else:
                raise RuntimeError('Red Pitaya not connected')

    def calibration_enable_output(self, enable: bool = True) -> None:
        """Enable or disable the RedPitaya output for calibration.

        @param bool enable: True to enable output, False to disable
        """
        with self._thread_lock:
            if self._redpitaya and self._redpitaya.is_connected:
                self._redpitaya.enable_output(enable)
                self.log.debug(f'RedPitaya output {"enabled" if enable else "disabled"}')
            else:
                raise RuntimeError('Red Pitaya not connected')

    def calibration_get_if_source(self):
        """Get direct access to the RedPitayaIFSource for advanced calibration.

        This provides direct access to the IF source for advanced operations
        like loading calibration files or verifying signal generation.

        @return RedPitayaIFSource: The internal IF source instance
        """
        if self._redpitaya and self._redpitaya.is_connected:
            return self._redpitaya
        else:
            raise RuntimeError('Red Pitaya not connected')
