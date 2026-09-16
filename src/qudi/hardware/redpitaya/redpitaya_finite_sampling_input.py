# -*- coding: utf-8 -*-
"""
This file contains the Qudi hardware module to use a Red Pitaya with pyrpl scan module
for finite sampling input (replacing NIDAQ functionality).
"""

import numpy as np
import time
import pyrpl

from qudi.util.mutex import RecursiveMutex
from qudi.core.configoption import ConfigOption
from qudi.interface.finite_sampling_input_interface import FiniteSamplingInputInterface, FiniteSamplingInputConstraints
from .resource_manager import get_pyrpl_instance, release_pyrpl_instance


def _calculate_scan_timing(requested_rate_hz, trigger_time_s, settling_time_s,
                           fpga_clock_hz, valid_period_cycles=None,
                           minimum_valid_samples=1):
    """Quantize a point and guarantee its finite acquisition window.

    Demodulated data reaches the scan accumulator only on the lock-in valid
    strobe. A positive dwell shorter than one strobe period can therefore
    contain no samples, depending on phase. Work in FPGA cycles so the
    guarantee matches the hardware counters exactly.
    """
    fpga_clock_hz = int(fpga_clock_hz)
    requested_rate_hz = float(requested_rate_hz)
    if fpga_clock_hz <= 0 or requested_rate_hz <= 0:
        raise ValueError('FPGA clock and requested scan rate must be positive')

    trigger_cycles = max(0, int(round(float(trigger_time_s) * fpga_clock_hz)))
    settling_cycles = max(0, int(round(float(settling_time_s) * fpga_clock_hz)))
    requested_total_cycles = max(1, int(round(fpga_clock_hz / requested_rate_hz)))
    requested_dwell_cycles = requested_total_cycles - trigger_cycles - settling_cycles

    minimum_dwell_cycles = 1
    if valid_period_cycles is not None:
        valid_period_cycles = int(valid_period_cycles)
        minimum_valid_samples = max(1, int(minimum_valid_samples))
        if valid_period_cycles <= 0:
            raise ValueError('valid period must be positive')
        minimum_dwell_cycles = minimum_valid_samples * valid_period_cycles

    dwell_cycles = max(minimum_dwell_cycles, requested_dwell_cycles)
    actual_total_cycles = trigger_cycles + settling_cycles + dwell_cycles
    return {
        'trigger_cycles': trigger_cycles,
        'settling_cycles': settling_cycles,
        'dwell_cycles': dwell_cycles,
        'dwell_time_s': dwell_cycles / float(fpga_clock_hz),
        'actual_rate_hz': fpga_clock_hz / float(actual_total_cycles),
        'rate_limited': dwell_cycles > requested_dwell_cycles,
        'guaranteed_valid_samples': (
            dwell_cycles // valid_period_cycles
            if valid_period_cycles is not None else dwell_cycles),
    }


class RedPitayaFiniteSamplingInput(FiniteSamplingInputInterface):
    """
    A Red Pitaya device using pyrpl scan module for finite sampling input.

    This module replaces NIDAQ functionality for ODMR measurements by using
    the Red Pitaya's FPGA-based scan module for triggering and data acquisition.

    Example config:

    redpitaya_finite_sampling:
        module.Class: 'redpitaya.redpitaya_finite_sampling_input.RedPitayaFiniteSamplingInput'
        options:
            redpitaya_config_name: 'rpy_shared_config'  # pyrpl config name
            redpitaya_hostname: '192.168.1.100'  # IP address of Red Pitaya
            calibration_factor: 1.0  # optional, scaling factor for data
            trigger_output_duration: 50e-6  # trigger pulse duration in seconds
            settling_time: 100e-6  # settling time after trigger in seconds
            input_channel: 'in1'  # 'in1' or 'in2' (hardwired to in1 in FPGA)
            signal_scale: 1.0  # Scale factor to convert ADC units to physical units
            input_select: 'demod'  # 'adc', 'iq0', or 'demod'

            # Lock-in filter options (only used when input_select='demod'):
            lock_in_fir_bypass_ch1: False  # True: CIC only (~15 kHz BW, ~160 µs latency)
                                           # False: CIC+FIR (bandwidth per filter_select)
            lock_in_fir_bypass_ch2: False
            lock_in_filter_ch1: '2kHz_minphase'  # '2kHz_minphase' or '2kHz_linear'
            lock_in_filter_ch2: '2kHz_minphase'
    """

    # Config options
    _redpitaya_config_name = ConfigOption('redpitaya_config_name', default='rpy_shared_config', missing='info')
    _redpitaya_hostname = ConfigOption('redpitaya_hostname', missing='error')
    _redpitaya_fpga_filename = ConfigOption(
        'redpitaya_fpga_filename', default=None, missing='info')
    _calibration_factor = ConfigOption('calibration_factor', default=1.0, missing='info')
    _trigger_output_duration = ConfigOption('trigger_output_duration', default=50e-6, missing='info')
    _settling_time = ConfigOption('settling_time', default=100e-6, missing='info')
    _minimum_demod_samples_per_point = ConfigOption(
        'minimum_demod_samples_per_point', default=16, missing='info')
    _input_channel = ConfigOption('input_channel', default='in1', missing='info')
    _signal_scale = ConfigOption('signal_scale', default=1.0, missing='info')
    _input_select = ConfigOption('input_select', default='adc', missing='info')  # 'adc', 'iq0', or 'demod'

    # Lock-in filter configuration (applies when input_select='demod')
    # FIR bypass: True = CIC only (~15 kHz BW, ~160 µs latency), False = CIC+FIR
    _lock_in_fir_bypass_ch1 = ConfigOption('lock_in_fir_bypass_ch1', default=False, missing='info')
    _lock_in_fir_bypass_ch2 = ConfigOption('lock_in_fir_bypass_ch2', default=False, missing='info')
    # Filter selection (active when fir_bypass is False):
    # Both filters have the same CIC-compensated 2 kHz magnitude response.
    _lock_in_filter_ch1 = ConfigOption('lock_in_filter_ch1', default='2kHz_minphase', missing='info')
    _lock_in_filter_ch2 = ConfigOption('lock_in_filter_ch2', default='2kHz_minphase', missing='info')
    # Demodulation bypass: True = DC passthrough (no ref mixing), False = normal lock-in demod
    _lock_in_demod_bypass_ch1 = ConfigOption('lock_in_demod_bypass_ch1', default=False, missing='nothing')
    _lock_in_demod_bypass_ch2 = ConfigOption('lock_in_demod_bypass_ch2', default=False, missing='nothing')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._pyrpl = None
        self._scan_module = None

        self._thread_lock = RecursiveMutex()
        self._sample_rate = 1000.0  # Hz
        self._requested_sample_rate = self._sample_rate
        self._actual_sample_rate = self._sample_rate
        self._fpga_clock_hz = 125_000_000
        self._demod_sample_period_cycles = 4096
        self._frame_size = 100
        self._active_channel = 'ch1'
        self._constraints = None

        # Buffer for storing acquired data
        self._data_buffer = None
        self._buffer_position = 0

    def on_activate(self):
        """Initialize connection to Red Pitaya via pyrpl."""
        try:
            # Use the shared factory to get a pyrpl instance
            self._pyrpl, _ = get_pyrpl_instance(
                hostname=self._redpitaya_hostname,
                config_name=self._redpitaya_config_name,
                fpga_filename=self._redpitaya_fpga_filename
            )
            self.log.info(f'Acquired shared pyrpl instance for {self._redpitaya_hostname}')

            # Get the scan module
            self._scan_module = self._pyrpl.rp.scan
            self._hk_module   = self._pyrpl.rp.hk
            lock_caps = self._pyrpl.rp.lockin.require_compatible_firmware()
            self._fpga_clock_hz = int(lock_caps['fpga_clock_hz'])
            self._demod_sample_period_cycles = int(
                lock_caps['sample_period_cycles'])
            self._minimum_demod_samples_per_point = max(
                1, int(self._minimum_demod_samples_per_point))

            self._hk_module.configure_pin('P7', direction='output', source='module', invert=True)
            self.log.info("Configured trigger pin P7 (DIO7_P) as an inverted module output.")

            # Configure scan module defaults
            self._scan_module.trigger_length = self._trigger_output_duration
            self._scan_module.settling_time = self._settling_time

            # Set input select from config
            valid_inputs = {'adc', 'iq0', 'demod'}
            if self._input_select not in valid_inputs:
                self.log.warning(f'Invalid input_select "{self._input_select}" specified. Falling back to "adc".')
                self._input_select = 'adc'
            self._scan_module.input_select = self._input_select
            self.log.info(f'Scan input selected: {self._input_select}')

            # Configure lock-in filter settings (relevant when input_select='demod')
            if self._input_select == 'demod':
                self._configure_lock_in_filters()

            # Note: The physical input channel is set to ADC1 (out of ADC1 and ADC2)
            if self._input_channel != 'in1':
                self.log.warning('Scan module is hardwired to read from in1 (adc_a). '
                                 'Input channel setting will be ignored.')

            # Create constraints
            self._constraints = FiniteSamplingInputConstraints(
                channel_units={'ch1': 'V'},  # Single channel
                frame_size_limits=(1, 4096),  # Limited by FPGA BRAM
                sample_rate_limits=(0.1, 100000)  # Practical limits
            )

            self._sample_rate = 1000.0
            self._requested_sample_rate = self._sample_rate
            self._actual_sample_rate = self._sample_rate
            self._frame_size = min(100, self._constraints.max_frame_size)

            self.log.info('Connected to Red Pitaya scan module')
            self.log.info('Trigger output: DIO7_P (exp_p_io[7])')
            self.log.info('Input channel: In1 (adc_a) - hardwired in FPGA')

        except Exception as e:
            self.log.error(f'Failed to connect to Red Pitaya: {e}')
            raise

    def on_deactivate(self):
        """Clean up pyrpl connection."""
        try:
            if self._scan_module is not None:
                if self.module_state() == 'locked':
                    self._scan_module.stop()
                    self._scan_module.reset()

            if self._pyrpl is not None:
                # Release the instance, letting the shared manager handle cleanup
                release_pyrpl_instance(
                    hostname=self._redpitaya_hostname,
                    config_name=self._redpitaya_config_name
                )
                self._pyrpl = None

        except Exception as e:
            self.log.error(f'Error during deactivation: {e}')

    @property
    def constraints(self):
        return self._constraints

    @property
    def active_channels(self):
        return frozenset([self._active_channel])

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def frame_size(self):
        return self._frame_size

    @property
    def samples_in_buffer(self):
        """Return number of samples available in buffer."""
        with self._thread_lock:
            if self._data_buffer is None:
                return 0
            return len(self._data_buffer) - self._buffer_position

    def set_sample_rate(self, rate):
        """Set the requested point rate and expose the achievable FPGA rate."""
        rate = float(rate)
        assert self._constraints.sample_rate_in_range(rate)[0], \
            f'Sample rate {rate} Hz out of bounds'

        with self._thread_lock:
            assert self.module_state() == 'idle', \
                'Cannot change sample rate during acquisition'
            timing = _calculate_scan_timing(
                requested_rate_hz=rate,
                trigger_time_s=self._trigger_output_duration,
                settling_time_s=self._settling_time,
                fpga_clock_hz=self._fpga_clock_hz,
                valid_period_cycles=(self._demod_sample_period_cycles
                                     if self._input_select == 'demod' else None),
                minimum_valid_samples=self._minimum_demod_samples_per_point)
            self._requested_sample_rate = rate
            self._sample_rate = timing['actual_rate_hz']
            self._actual_sample_rate = self._sample_rate
            if timing['rate_limited']:
                self.log.warning(
                    'Requested scan rate %.1f Hz cannot fit %.1f us trigger + '
                    '%.1f us settling + %d valid %s samples. Clamping the '
                    'hardware point rate to %.1f Hz (%.1f us acquisition dwell).',
                    rate, self._trigger_output_duration * 1e6,
                    self._settling_time * 1e6,
                    timing['guaranteed_valid_samples'], self._input_select,
                    self._sample_rate, timing['dwell_time_s'] * 1e6)
            else:
                self.log.debug('Sample rate set to %.3f Hz', self._sample_rate)

    def set_active_channels(self, channels):
        """Set active channels (only one channel supported currently)."""
        with self._thread_lock:
            assert self.module_state() == 'idle', \
                'Cannot change channels during acquisition'
            # For now, we only support one channel
            if channels:
                self._active_channel = list(channels)[0]

    def set_frame_size(self, size):
        """Set number of samples per frame."""
        size = int(round(size))
        assert self._constraints.frame_size_in_range(size)[0], \
            f'Frame size {size} out of bounds'

        with self._thread_lock:
            assert self.module_state() == 'idle', \
                'Cannot change frame size during acquisition'
            self._frame_size = size
            self.log.debug(f'Frame size set to {size}')

    def start_buffered_acquisition(self):
        """Start data acquisition using scan module."""
        assert self.module_state() == 'idle', \
            'Acquisition already in progress'

        self.module_state.lock()

        try:
            # Configure scan module
            self._scan_module.num_steps = self._frame_size

            trigger_time = self._trigger_output_duration
            settling_time = self._settling_time
            timing = _calculate_scan_timing(
                requested_rate_hz=self._sample_rate,
                trigger_time_s=trigger_time,
                settling_time_s=settling_time,
                fpga_clock_hz=self._fpga_clock_hz,
                valid_period_cycles=(self._demod_sample_period_cycles
                                     if self._input_select == 'demod' else None),
                minimum_valid_samples=self._minimum_demod_samples_per_point)
            dwell_time = timing['dwell_time_s']
            self._actual_sample_rate = timing['actual_rate_hz']

            if timing['rate_limited']:
                # Defensive path for timing options changed after set_sample_rate.
                self.log.warning(
                    'Requested scan rate %.1f Hz leaves too little acquisition '
                    'time after %.1f us trigger + %.1f us settling. Using %.1f us '
                    'dwell and %.1f Hz actual rate to guarantee at least %d %s '
                    'samples per point.',
                    self._sample_rate, trigger_time * 1e6, settling_time * 1e6,
                    dwell_time * 1e6, self._actual_sample_rate,
                    timing['guaranteed_valid_samples'], self._input_select)

            self._scan_module.dwell_time = dwell_time
            self._scan_module.trigger_length = trigger_time
            self._scan_module.settling_time = settling_time

            # Clear buffer
            self._data_buffer = None
            self._buffer_position = 0

            # Start the scan
            self._scan_module.start()

            self.log.debug(f'Started scan: {self._frame_size} steps, '
                           f'dwell={dwell_time * 1e6:.1f}us, '
                           f'trigger={trigger_time * 1e6:.1f}us, '
                           f'settling={settling_time * 1e6:.1f}us, '
                           f'actual_rate={self._actual_sample_rate:.1f}Hz')

        except Exception as e:
            self.module_state.unlock()
            raise RuntimeError(f'Failed to start acquisition: {e}')

    def stop_buffered_acquisition(self):
        """Stop the acquisition."""
        if self.module_state() == 'locked':
            try:
                self._scan_module.stop()

                # Try to get any partial data
                if self._scan_module.done or self._scan_module.current_step > 0:
                    try:
                        raw_data = self._scan_module.get_data(average=True) #fixme: at some later point set to true
                        # Apply calibration and scaling
                        self._data_buffer = raw_data * self._calibration_factor * self._signal_scale
                        self._buffer_position = 0
                        self.log.debug(f'Retrieved {len(self._data_buffer)} samples')
                    except Exception as e:
                        self.log.warning(f'Could not retrieve partial data: {e}')

            finally:
                self.module_state.unlock()

    def get_buffered_samples(self, number_of_samples=None):
        """Get samples from buffer."""
        data = dict()

        # If no acquisition running and no buffered data, return empty
        if self.module_state() == 'idle' and self.samples_in_buffer < 1:
            return data

        # Wait for scan to complete if still running
        if self.module_state() == 'locked' and self._data_buffer is None:
            # Calculate timeout based on configured timing
            time_per_sample = 1.0 / self._actual_sample_rate
            timeout = (self._frame_size * time_per_sample) + 2.0

            self.log.debug(f'Waiting for scan completion (timeout={timeout:.1f}s)')

            if self._scan_module.wait_done(timeout=timeout):
                try:
                    raw_data = self._scan_module.get_data(average=True) #fixme: at some later point set to true
                    # Apply calibration and scaling
                    self._data_buffer = raw_data * self._calibration_factor * self._signal_scale
                    self._buffer_position = 0
                    self.log.debug(f'Scan completed, retrieved {len(self._data_buffer)} samples')
                except Exception as e:
                    self.log.error(f'Failed to retrieve data: {e}')
                    self.stop_buffered_acquisition()
                    return data
            else:
                self.log.error('Timeout waiting for scan completion')
                self.stop_buffered_acquisition()
                return data

        if self._data_buffer is not None:
            available = len(self._data_buffer) - self._buffer_position
            if number_of_samples is None:
                number_of_samples = available
            else:
                number_of_samples = min(number_of_samples, available)

            if number_of_samples > 0:
                end_pos = self._buffer_position + number_of_samples
                data[self._active_channel] = self._data_buffer[self._buffer_position:end_pos].copy()
                self._buffer_position = end_pos

        return data

    def acquire_frame(self, frame_size=None):
        """Acquire a complete frame of data."""
        with self._thread_lock:
            if frame_size is not None:
                original_frame_size = self._frame_size
                self.set_frame_size(frame_size)

            try:
                self.start_buffered_acquisition()
                data = self.get_buffered_samples(self._frame_size)
                self.stop_buffered_acquisition()

                if frame_size is not None:
                    self._frame_size = original_frame_size

                return data

            except Exception as e:
                self.log.error(f'Frame acquisition failed: {e}')
                if frame_size is not None:
                    self._frame_size = original_frame_size
                raise

    def generate_pulse(self, duration):
        """Generate a single trigger pulse.

        This method is used by ODMR logic for initial synchronization.
        During actual scans, the scan module generates triggers automatically.

        Args:
            duration (float): Pulse duration in seconds
        """
        with self._thread_lock:
            if self.module_state() != 'idle':
                self.log.warning('Cannot generate pulse during acquisition')
                return

            try:
                # Configure scan for single pulse
                self._scan_module.num_steps = 1
                self._scan_module.trigger_length = duration
                self._scan_module.dwell_time = 8e-9  # Minimum time
                self._scan_module.settling_time = 0

                # Start and wait for completion
                self._scan_module.start()
                time.sleep(duration + 0.01)  # Wait for pulse + margin

                # Clean up
                self._scan_module.stop()
                self._scan_module.reset()

                self.log.debug(f'Generated single trigger pulse of {duration * 1e6:.1f} us')

            except Exception as e:
                self.log.error(f'Failed to generate pulse: {e}')

    def _configure_lock_in_filters(self):
        """Configure lock-in module filter settings from config options.

        Applies FIR bypass and filter selection settings to the lock-in module.
        Called automatically during activation when input_select='demod'.
        """
        try:
            # Module is named 'lockin' (not 'lock_in') per PyRPL naming convention
            lock_in = self._pyrpl.rp.lockin

            # Configure FIR bypass (True = CIC only, ~15 kHz BW, ~160 µs latency)
            lock_in.fir_bypass_ch1 = self._lock_in_fir_bypass_ch1
            lock_in.fir_bypass_ch2 = self._lock_in_fir_bypass_ch2

            # Derive selections from the loaded FPGA profile so an invalid
            # setting cannot fall back to a FIR absent from the fast image.
            valid_filters = {
                name for name, available in lock_in.filter_capabilities.items()
                if available and name != 'fir_bypass'}
            if '2kHz_minphase' in valid_filters:
                valid_filters.add('2kHz')  # legacy alias
            fallback_filter = ('2kHz_minphase'
                               if '2kHz_minphase' in valid_filters
                               else next(iter(sorted(valid_filters))))

            if self._lock_in_filter_ch1 not in valid_filters:
                self.log.warning(f'Invalid lock_in_filter_ch1 "{self._lock_in_filter_ch1}". '
                                 f'Using "{fallback_filter}". Valid options: {valid_filters}')
                self._lock_in_filter_ch1 = fallback_filter
            lock_in.filter_select_ch1 = self._lock_in_filter_ch1

            if self._lock_in_filter_ch2 not in valid_filters:
                self.log.warning(f'Invalid lock_in_filter_ch2 "{self._lock_in_filter_ch2}". '
                                 f'Using "{fallback_filter}". Valid options: {valid_filters}')
                self._lock_in_filter_ch2 = fallback_filter
            lock_in.filter_select_ch2 = self._lock_in_filter_ch2

            # Configure demodulation bypass (DC ODMR mode)
            lock_in.demod_bypass_ch1 = self._lock_in_demod_bypass_ch1
            lock_in.demod_bypass_ch2 = self._lock_in_demod_bypass_ch2

            # Log configuration summary
            ch1_demod = 'DC bypass' if self._lock_in_demod_bypass_ch1 else 'lock-in demod'
            ch2_demod = 'DC bypass' if self._lock_in_demod_bypass_ch2 else 'lock-in demod'
            ch1_mode = 'CIC only (~15 kHz)' if self._lock_in_fir_bypass_ch1 else f'CIC+FIR ({self._lock_in_filter_ch1})'
            ch2_mode = 'CIC only (~15 kHz)' if self._lock_in_fir_bypass_ch2 else f'CIC+FIR ({self._lock_in_filter_ch2})'
            self.log.info(f'Lock-in configured - Ch1: {ch1_demod}, {ch1_mode} | Ch2: {ch2_demod}, {ch2_mode}')

        except AttributeError as e:
            self.log.warning(f'Could not configure lock-in filters (module not available): {e}')
        except Exception as e:
            self.log.error(f'Failed to configure lock-in filters: {e}')
