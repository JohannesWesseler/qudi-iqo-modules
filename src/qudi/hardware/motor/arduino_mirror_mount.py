# -*- coding: utf-8 -*-
"""
Arduino Mega 2560 driving four geared 28BYJ-type steppers (two motorized mirror mounts) through
ULN2003 boards, firmware 'Motorized_mount_new_architecturev1-3' (binary protocol, 115200 baud).

Protocol (see Laser_alignment/PROJECT_DOCUMENTATION.md):
    host -> board: 0xFC, type, b2, b3, b4[, speed], xor-checksum of the fields
    board -> host: 6-byte packets 0xFC, type, b2, b3, b4, checksum(b1..b4)
      0xAA heartbeat / per-step, 0xCC move complete (b2 = motor),
      0xFF 0xFF 0xFF emergency halt (pin-21 interrupt)

Opening the port resets the board (DTR). Coils stay off and nothing moves, but the firmware's
step phase restarts, so the first move of each motor can jump by up to +-4 steps.

Example config:

    mirror_mount:
        module.Class: 'motor.arduino_mirror_mount.ArduinoMirrorMount'
        options:
            port: null                 # null: auto-detect the Arduino Mega (USB 2341:0042)
            motor_names: ['M1', 'M2', 'M3', 'M4']
            firmware_slots: [0, 1, 2, 3]   # firmware motor index per motor; slot 0: pins 8-11,
                                           # 1: 4-7, 2: 22-25, 3: 26-29, 4: 30-33, 5: 34-37, ...
            release_after_move: True   # de-energize coils after each move (avoids USB drop-outs)
            speed: 1                   # firmware speed 1 (5 ms/step) .. 10 (14 ms/step)
            step_limit: 4000

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
from typing import Tuple

from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.util.mutex import RecursiveMutex
from qudi.interface.stepper_mirror_mount_interface import StepperMirrorMountInterface

PANIC = bytes([0xFF] * 3)


class MountError(RuntimeError):
    pass


def _step_time(speed):
    """Firmware step interval in seconds (speed 1 -> 5 ms, 10 -> 14 ms, +3 ms ramp on 15 steps)."""
    return (5000 + (speed - 1) * 1000) * 1e-6


class ArduinoMirrorMount(StepperMirrorMountInterface):
    """Arduino stepper controller for two motorized mirror mounts."""

    _port = ConfigOption('port', default=None, missing='nothing')
    _baud = ConfigOption('baud_rate', default=115200, missing='nothing')
    _motor_names = ConfigOption('motor_names', default=['M1', 'M2', 'M3', 'M4'], missing='nothing')
    _release_after_move = ConfigOption('release_after_move', default=True, missing='nothing')
    _speed = ConfigOption('speed', default=1, missing='nothing', checker=lambda x: 1 <= x <= 10)
    _step_limit = ConfigOption('step_limit', default=4000, missing='nothing')
    _slots = ConfigOption('firmware_slots', default=None, missing='nothing')

    # logical positions survive restarts (no encoders: this is the only memory of the state)
    _positions = StatusVar('positions', default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = RecursiveMutex()
        self._ser = None
        self._buf = bytearray()

    # --- qudi --------------------------------------------------------------
    def on_activate(self):
        n = len(self._motor_names)
        if not self._positions or len(self._positions) != n:
            self._positions = [0] * n
        self._positions = [int(p) for p in self._positions]
        self._open()

    def on_deactivate(self):
        try:
            self.release()
        except Exception:
            pass
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    # --- connection ----------------------------------------------------------
    @staticmethod
    def _find_port():
        from serial.tools import list_ports
        ports = list(list_ports.comports())
        for p in ports:
            if (p.vid, p.pid) == (0x2341, 0x0042):
                return p.device
        for p in ports:
            if p.vid in (0x2341, 0x2A03):
                return p.device
        raise MountError('Arduino Mega not found; set the "port" option. Ports: '
                         + ', '.join(f'{p.device} ({p.description})' for p in ports))

    def _open(self):
        import serial
        port = self._port or self._find_port()
        for attempt in range(15):   # the port can be briefly held, e.g. by the Arduino IDE
            try:
                self._ser = serial.Serial(port, self._baud, timeout=0.05)
                break
            except serial.SerialException:
                if attempt == 14:
                    raise
                time.sleep(1)
        self._buf.clear()
        self._wait_for(0xAA, timeout=4.0)   # board resets on open; wait for its first heartbeat
        self.log.info(f'Mirror mount Arduino on {port}, positions {self._positions}')

    def reconnect(self):
        """Reopen the port after a USB drop-out (resets the board, coils off)."""
        with self._lock:
            try:
                if self._ser is not None:
                    self._ser.close()
            except Exception:
                pass
            self._ser = None
            time.sleep(2.0)
            self._open()

    # --- framing -------------------------------------------------------------
    def _send(self, ctype, b2, b3, b4, speed=None):
        fields = [ctype, b2 & 0xFF, b3 & 0xFF, b4 & 0xFF] + ([speed] if speed is not None else [])
        chk = 0
        for f in fields:
            chk ^= f
        self._ser.write(bytes([0xFC, *fields, chk]))

    def _packets(self):
        chunk = self._ser.read(max(1, self._ser.in_waiting))
        self._buf += chunk
        if PANIC in self._buf:
            self._buf.clear()
            raise MountError('firmware emergency halt (pin 21 interrupt); reconnect to reset')
        while True:
            i = self._buf.find(0xFC)
            if i < 0:
                self._buf.clear()
                return
            del self._buf[:i]
            if len(self._buf) < 6:
                return
            p = bytes(self._buf[:6])
            if p[1] ^ p[2] ^ p[3] ^ p[4] != p[5]:
                del self._buf[:1]
                continue
            del self._buf[:6]
            yield p

    def _wait_for(self, ptype, timeout, match=None):
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            for p in self._packets():
                if p[1] == ptype and (match is None or match(p)):
                    return p
        raise MountError(f'timeout waiting for packet 0x{ptype:02X}')

    # --- StepperMirrorMountInterface ----------------------------------------
    @property
    def motor_names(self) -> Tuple[str, ...]:
        return tuple(self._motor_names)

    @property
    def step_limit(self) -> int:
        return int(self._step_limit)

    def move_steps(self, motor: int, steps: int) -> None:
        steps = int(steps)
        if steps == 0:
            return
        target = self._positions[motor] + steps
        if abs(target) > self._step_limit:
            raise ValueError(f'{self._motor_names[motor]} target {target} outside '
                             f'+-{self._step_limit} steps')
        if not -32768 <= steps <= 32767:
            raise ValueError('steps out of int16 range')
        with self._lock:
            slot = self._slot(motor)
            self._send(0x01, slot, steps >> 8, steps, self._speed)
            t = abs(steps) * _step_time(self._speed) + 15 * 3e-3 + 0.5
            try:
                self._wait_for(0xCC, 1.5 * t, match=lambda p: p[2] == slot)
            except Exception:
                # the move may or may not have happened completely: the counter is now uncertain
                self.log.error(f'No completion from {self._motor_names[motor]} for {steps:+d} steps; '
                               f'logical position may be off by up to that amount')
                raise
            self._positions[motor] = target
            if self._release_after_move:
                self._send(0x03, slot, 0, 0)

    def _slot(self, motor):
        return int(self._slots[motor]) if self._slots else int(motor)

    def get_positions(self) -> Tuple[int, ...]:
        return tuple(self._positions)

    def set_positions(self, positions) -> None:
        if len(positions) != len(self._motor_names):
            raise ValueError('need one position per motor')
        self._positions = [int(p) for p in positions]

    def release(self) -> None:
        with self._lock:
            for m in range(len(self._motor_names)):
                self._send(0x03, self._slot(m), 0, 0)
