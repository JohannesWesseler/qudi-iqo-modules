# -*- coding: utf-8 -*-
"""
Averaged DC readings of the Red Pitaya fast inputs IN1/IN2 (e.g. the Nirvana signal monitor),
as qudi process values.

The board scope (region 1) runs continuously in rolling mode with averaging decimation; a
reading is the mean of the newest samples that cover `averaging_time`. The call first waits one
averaging time, so every sample of a reading was taken after the call (i.e. after a preceding
motor move). One block read per value: ~10 ms overhead plus the averaging time.

A fast modulation of the signal (e.g. the ~10-50 kHz ODMR FM on the fluorescence) averages out:
with T_avg = 20 ms the residual of a sine of period P is below P / (pi * T_avg), i.e. < 0.2 % of
its amplitude at 10 kHz.

The module shares the pyrpl instance of the other Red Pitaya modules (same hostname and
redpitaya_config_name, same FPGA image). It takes over the scope; nothing else in qudi uses it.
Every image of this pyrpl fork has the board scope, so the module works with any of them.

Example config:

    redpitaya_input_monitor:
        module.Class: 'redpitaya.redpitaya_input_monitor.RedPitayaInputMonitor'
        options:
            redpitaya_hostname: '10.203.129.28'
            redpitaya_config_name: 'rpy_shared_config'   # same as the other Red Pitaya modules
            redpitaya_fpga_filename: null                # same as the other Red Pitaya modules
            averaging_time: 0.02      # s per reading
            decimation: 1024          # scope decimation (8.192 us per averaged sample)
            channels:                 # process-value name -> input, gain, offset (value = gain * (V - offset))
                nirvana_monitor:
                    input: 'in2'
                    gain: -1000.0     # Nirvana signal monitor is negative; report mV of light signal
                    offset: 0.0       # V, e.g. the dark level
                    unit: 'mV'

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

import time

import numpy as np

from qudi.core.configoption import ConfigOption
from qudi.util.mutex import RecursiveMutex
from qudi.interface.process_control_interface import ProcessValueInterface, ProcessControlConstraints
from .resource_manager import get_pyrpl_instance, release_pyrpl_instance

_DEFAULT_CHANNELS = {'in1': {'input': 'in1', 'gain': 1.0, 'offset': 0.0, 'unit': 'V'},
                     'in2': {'input': 'in2', 'gain': 1.0, 'offset': 0.0, 'unit': 'V'}}
_SCOPE_BASE = {'in1': 0x10000, 'in2': 0x20000}


class RedPitayaInputMonitor(ProcessValueInterface):
    """Averaged IN1/IN2 levels from the Red Pitaya scope in rolling mode."""

    _hostname = ConfigOption('redpitaya_hostname', missing='error')
    _config_name = ConfigOption('redpitaya_config_name', default='rpy_shared_config', missing='info')
    _fpga_filename = ConfigOption('redpitaya_fpga_filename', default=None, missing='info')
    _averaging_time = ConfigOption('averaging_time', default=0.02, missing='nothing',
                                   checker=lambda x: 0 < x <= 0.13)   # scope buffer: 16384 samples
    _decimation = ConfigOption('decimation', default=1024, missing='nothing',
                               checker=lambda x: x in [2 ** n for n in range(17)])
    _channels = ConfigOption('channels', default=_DEFAULT_CHANNELS, missing='nothing')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = RecursiveMutex()
        self._pyrpl = None
        self._scope = None
        self._active = dict()

    def on_activate(self):
        for name, ch in self._channels.items():
            if ch.get('input') not in _SCOPE_BASE:
                raise ValueError(f'channel {name}: input must be "in1" or "in2"')
        self._pyrpl, _ = get_pyrpl_instance(self._hostname, self._config_name,
                                            fpga_filename=self._fpga_filename)
        self._scope = self._pyrpl.rp.scope
        self._active = {name: True for name in self._channels}
        self._setup_scope()

    def on_deactivate(self):
        self._scope = None
        self._pyrpl = None
        release_pyrpl_instance(self._hostname, self._config_name)

    # --- scope -----------------------------------------------------------------
    def _setup_scope(self):
        with self._lock:
            s = self._scope
            # decimation is not a scope setup attribute (setup() would ignore it): set it directly
            s.setup(input1='in1', input2='in2', average=True, trigger_source='immediately',
                    rolling_mode=True)
            s.decimation = int(self._decimation)
            s._start_acquisition_rolling_mode()
            if int(s.decimation) != int(self._decimation):
                raise RuntimeError(f'scope decimation is {s.decimation}, wanted {self._decimation}')
            # an averaging time is only meaningful once the buffer holds that much new data
            time.sleep(min(1.0, 8e-9 * self._decimation * s.data_length))

    def _scope_is_ours(self):
        s = self._scope
        return (int(s.decimation) == int(self._decimation) and bool(s.average)
                and s.input1 == 'in1' and s.input2 == 'in2')

    @property
    def averaging_time(self):
        return float(self._averaging_time)

    @averaging_time.setter
    def averaging_time(self, value):
        value = float(value)
        if not 0 < value <= 0.13:
            raise ValueError('averaging time must be in (0, 0.13] s')
        self._averaging_time = value

    def read_raw(self, inp, settle=True):
        """Newest samples [V] of input 'in1'/'in2' covering the averaging time."""
        with self._lock:
            if not self._scope_is_ours():
                self.log.warning('Scope settings were changed elsewhere; restoring rolling mode')
                self._setup_scope()
            s = self._scope
            dt = 8e-9 * self._decimation
            n = int(max(1, min(s.data_length - 16, round(self._averaging_time / dt))))
            if settle:
                time.sleep(n * dt + 2 * dt)
            length = s.data_length
            wp = int(s._write_pointer_current)
            start = (wp - n + 1) % length
            base = _SCOPE_BASE[inp]
            if start + n <= length:
                x = np.array(s._reads(base + 4 * start, n), dtype=np.int32)
            else:
                k = length - start
                x = np.concatenate([np.array(s._reads(base + 4 * start, k), dtype=np.int32),
                                    np.array(s._reads(base, n - k), dtype=np.int32)])
        x &= 0x3FFF
        x[x >= 2 ** 13] -= 2 ** 14
        return x / 2 ** 13

    # --- ProcessValueInterface ---------------------------------------------------
    @property
    def constraints(self) -> ProcessControlConstraints:
        return ProcessControlConstraints(
            process_channels=list(self._channels),
            units={n: c.get('unit', 'V') for n, c in self._channels.items()},
            dtypes={n: float for n in self._channels})

    def set_activity_state(self, channel: str, active: bool) -> None:
        self._active[channel] = bool(active)

    def get_activity_state(self, channel: str) -> bool:
        return self._active[channel]

    def get_process_value(self, channel: str) -> float:
        ch = self._channels[channel]
        v = float(self.read_raw(ch['input']).mean())
        return float(ch.get('gain', 1.0)) * (v - float(ch.get('offset', 0.0)))
