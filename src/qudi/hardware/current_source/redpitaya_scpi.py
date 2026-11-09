# -*- coding: utf-8 -*-
"""
Minimal SCPI client for a Red Pitaya on its stock OS (checked on OS 3.00-57, STEMlab 125-14 LN):
generator channel 1, acquisition of both inputs, triggers. No qudi dependencies; used by
redpitaya_lm1875_source.py.

Threading: every method holds an internal lock for its whole exchange with the board, so two
threads never mix their commands and replies. A sequence of calls that must not be interleaved
(set up, arm, wait, read) needs a lock in the caller as well; the qudi module has one.

Never point pyrpl at this board: pyrpl loads its own FPGA image and the SCPI server stops
working until the board is rebooted. RedPitayaScpi.board_info() checks that the stock OS runs.

"Quirky" behaviour that is worth knowing (measured Nov 2026 on OS 3.00-57):
- Unknown commands get no reply at all (a query then times out); the socket is reopened.
- After SOUR1:FREQ:FIX or SOUR1:FUNC the generator keeps the old waveform until
  SOUR1:TRig:INT. Amplitude and offset changes apply at once.
- ACQ:DEC accepts only powers of two and silently keeps the old value otherwise;
  ACQ:DEC:Factor accepts 1..65536. Always read back.
- The trigger output on DIO0_N exists with the 2.00-15..2.00-30 names (DAISY:TRig:Out:ENable,
  DAISY:TRig:Out:SOUR DAC|ADC); the DAISY:TRIG_O:* names of newer OS versions do not answer.
- Acquisition: ACQ:TRig:DLY 8192 puts the trigger at the start of the 16384-sample buffer;
  ACQ:SOURn:DATA? returns the buffer from the oldest sample, so the trigger is at index
  8192 - DLY.
- The SCPI server and the web applications (oscilloscope, ...) must not run together: both
  drive the FPGA. Start/stop the server over HTTP (start_server / stop_server).
"""

import functools
import json
import logging
import socket
import threading
import time
import urllib.request

import numpy as np

FS_ADC = 125e6
N_BUF = 16384
ARB_LENGTH = 16384
FUNCS = {'dc': 'DC', 'sine': 'SINE', 'square': 'SQUARE', 'triangle': 'TRIANGLE',
         'arbitrary': 'ARBITRARY'}


def _locked(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._io_lock:
            return method(self, *args, **kwargs)
    return wrapper


class RedPitayaScpi:
    """SCPI connection to a Red Pitaya with the stock OS. Each call is atomic (see the module
    docstring)."""

    def __init__(self, host, port=5000, timeout=10.0, logger=None, start_server=True):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.log = logger or logging.getLogger(__name__)
        self._io_lock = threading.RLock()
        self.sock = None
        self._rf = None
        self._dec = 1
        if start_server and not self._port_open():
            self.start_server()
        self._connect()
        self.idn = self.query('*IDN?')
        if not self.idn.startswith('REDPITAYA'):
            raise IOError(f'{host}: unexpected *IDN? answer {self.idn!r}')
        stale = self.errors()
        if stale:
            self.log.info(f'cleared stale Red Pitaya errors: {stale}')

    # --- server and connection -----------------------------------------------------------------
    def _http(self, path, timeout=10.0):
        with urllib.request.urlopen(f'http://{self.host}/{path}', timeout=timeout) as r:
            return r.read().decode(errors='replace')

    def board_info(self):
        """/get_info of the stock OS (fails on a board running pyrpl's image and server)."""
        return json.loads(self._http('get_info'))

    @_locked
    def start_server(self, wait=20.0):
        self.log.info(f'starting the SCPI server on {self.host}')
        try:
            # the request sometimes returns only after the server is up, sometimes not at all
            self._http('start_scpi_manager_tcp', timeout=5.0)
        except OSError as e:
            self.log.debug(f'start_scpi_manager_tcp: {e}; waiting for the port')
        t0 = time.time()
        while time.time() - t0 < wait:
            if self._port_open():
                time.sleep(0.5)
                return
            time.sleep(0.3)
        raise IOError(f'SCPI server on {self.host} did not start: close the Red Pitaya web '
                      f'applications (oscilloscope etc.) of this board, or reboot it')

    @_locked
    def stop_server(self):
        self.close()
        try:
            self._http('stop_scpi_manager')
        except Exception as e:
            self.log.warning(f'stopping the SCPI server failed: {e}')

    def _port_open(self):
        try:
            with socket.create_connection((self.host, self.port), timeout=1.5):
                return True
        except OSError:
            return False

    @_locked
    def _connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._rf = self.sock.makefile('rb')

    @_locked
    def reconnect(self):
        self.close()
        time.sleep(0.3)
        self._connect()

    @_locked
    def close(self):
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass
        self.sock = None
        self._rf = None

    # --- low level -------------------------------------------------------------------------------
    @_locked
    def write(self, cmd: str):
        self.log.debug(f'RP <- {cmd[:120]}')
        self.sock.sendall((cmd + '\r\n').encode())

    @_locked
    def query(self, cmd: str, timeout=None) -> str:
        self.write(cmd)
        if timeout is not None:
            self.sock.settimeout(timeout)
        try:
            return self._rf.readline().decode().strip()
        except socket.timeout:
            # OS 3.00: no reply to unknown commands; the stream is out of step afterwards
            self.reconnect()
            raise IOError(f'no reply to {cmd!r}')
        finally:
            if timeout is not None and self.sock is not None:
                self.sock.settimeout(self.timeout)

    @_locked
    def errors(self):
        out = []
        for _ in range(20):
            e = self.query('SYST:ERR?')
            if e.split(',')[0].strip() in ('0', '+0'):
                break
            out.append(e)
        return out

    @_locked
    def _read_block(self) -> bytes:
        head = self._rf.read(2)
        if head[:1] != b'#':
            rest = self._rf.readline()
            raise IOError(f'unexpected SCPI block header {head + rest!r}')
        nd = int(head[1:2])
        n = int(self._rf.read(nd))
        data = self._rf.read(n)
        self._rf.readline()  # trailing \r\n
        return data

    # --- generator, channel 1 --------------------------------------------------------------------
    @_locked
    def gen_off(self):
        """Amplitude and offset 0, output off. Never raises while the socket works."""
        self.write('SOUR1:VOLT 0')
        self.write('SOUR1:VOLT:OFFS 0')
        self.write('OUTPUT1:STATE OFF')

    @_locked
    def gen_state(self):
        return dict(on=self.query('OUTPUT1:STATE?'), func=self.query('SOUR1:FUNC?'),
                    freq=float(self.query('SOUR1:FREQ:FIX?')), amp=float(self.query('SOUR1:VOLT?')),
                    offs=float(self.query('SOUR1:VOLT:OFFS?')), burst=self.query('SOUR1:BURS:STAT?'),
                    trig_source=self.query('SOUR1:TRig:SOUR?'))

    @_locked
    def set_func(self, func: str):
        self.write(f'SOUR1:FUNC {FUNCS[func]}')

    @_locked
    def set_freq(self, f: float):
        self.write(f'SOUR1:FREQ:FIX {f:.9g}')

    @_locked
    def set_amp_offs(self, amp: float, offs: float):
        self.write(f'SOUR1:VOLT {amp:.5f}')
        self.write(f'SOUR1:VOLT:OFFS {offs:.5f}')

    @_locked
    def output(self, on: bool):
        self.write(f'OUTPUT1:STATE {"ON" if on else "OFF"}')

    @_locked
    def trigger_now(self):
        self.write('SOUR1:TRig:INT')

    @_locked
    def set_trigger_source(self, source: str):
        """'internal' (SOUR1:TRig:INT starts it), 'external' (rising edge on DIO0_P)."""
        self.write(f'SOUR1:TRig:SOUR {"INT" if source == "internal" else "EXT_PE"}')

    @_locked
    def set_burst(self, n_cycles=None, n_repetitions=1, period=None):
        """n_cycles None: continuous. period in s (between burst starts), resolution 1 us."""
        if n_cycles is None:
            self.write('SOUR1:BURS:STAT CONTINUOUS')
            return
        self.write('SOUR1:BURS:STAT BURST')
        self.write(f'SOUR1:BURS:NCYC {int(n_cycles)}')
        self.write(f'SOUR1:BURS:NOR {int(n_repetitions)}')
        if period is not None:
            self.write(f'SOUR1:BURS:INT:PER {int(round(period * 1e6))}')

    @_locked
    def set_arbitrary(self, table):
        """One period, ARB_LENGTH points in [-1, 1]."""
        table = np.asarray(table, float)
        if len(table) != ARB_LENGTH or np.abs(table).max() > 1.0 + 1e-9:
            raise ValueError(f'arbitrary table must have {ARB_LENGTH} points within [-1, 1]')
        self.write('SOUR1:TRAC:DATA:DATA ' + ','.join(f'{v:.5f}' for v in table))

    @_locked
    def set_trigger_output(self, enabled: bool, source: str = 'DAC'):
        """Marker on DIO0_N at every generator ('DAC') or acquisition ('ADC') trigger."""
        self.write(f'DAISY:TRig:Out:SOUR {source}')
        self.write(f'DAISY:TRig:Out:ENable {"ON" if enabled else "OFF"}')

    @_locked
    def trigger_output_state(self):
        return self.query('DAISY:TRig:Out:ENable?') == 'ON', self.query('DAISY:TRig:Out:SOUR?')

    # --- acquisition ---------------------------------------------------------------------------
    @_locked
    def acq_setup(self, dec: int, pre_trigger: int = 0):
        """Both inputs at LV (+-1 V), volts, binary transfer; the trigger at sample pre_trigger."""
        if not 0 <= pre_trigger < N_BUF:
            raise ValueError(f'pre_trigger must be in 0..{N_BUF - 1}')
        dec = valid_decimation(dec)
        self.write('ACQ:RST')
        self.write(f'ACQ:DEC:Factor {int(dec)}')
        self.write('ACQ:SOUR1:GAIN LV')
        self.write('ACQ:SOUR2:GAIN LV')
        self.write('ACQ:DATA:FORMAT BIN')
        self.write('ACQ:DATA:Units VOLTS')
        self.write(f'ACQ:TRig:DLY {N_BUF // 2 - int(pre_trigger)}')
        got = int(self.query('ACQ:DEC:Factor?'))
        gains = (self.query('ACQ:SOUR1:GAIN?'), self.query('ACQ:SOUR2:GAIN?'))
        errs = self.errors()
        if got != int(dec) or gains != ('LV', 'LV') or errs:
            raise IOError(f'acquisition setup failed: dec {got} (wanted {dec}), gains {gains}, '
                          f'errors {errs}')
        self._dec = got

    @property
    def decimation(self):
        return self._dec

    @property
    def fs(self):
        return FS_ADC / self._dec

    @_locked
    def acquire_now(self, timeout=10.0):
        """Immediate trigger. Returns (in1, in2) in V (physical inputs)."""
        self.write('ACQ:START')
        time.sleep(max(1e-3, N_BUF * self._dec / FS_ADC * 1.1))   # let the pre-trigger part fill
        self.write('ACQ:TRig NOW')
        return self.wait_and_read(timeout)

    @_locked
    def arm(self, source: str = 'generator'):
        """Start the acquisition and wait for a trigger: 'generator' (AWG_PE, the generator's
        own start) or 'external' (EXT_PE, rising edge on DIO0_P)."""
        self.write('ACQ:START')
        time.sleep(max(1e-3, N_BUF * self._dec / FS_ADC * 1.1))
        self.write(f'ACQ:TRig {"AWG_PE" if source == "generator" else "EXT_PE"}')

    @_locked
    def triggered(self) -> bool:
        return self.query('ACQ:TRig:STAT?') == 'TD'

    @_locked
    def wait_and_read(self, timeout=10.0):
        t0 = time.time()
        while not self.triggered():
            if time.time() - t0 > timeout:
                self.write('ACQ:STOP')
                raise TimeoutError('Red Pitaya acquisition: no trigger')
            time.sleep(1e-3)
        while self.query('ACQ:TRig:FILL?') != '1':
            if time.time() - t0 > timeout:
                self.write('ACQ:STOP')
                raise TimeoutError('Red Pitaya acquisition: buffer not filled')
            time.sleep(1e-3)
        self.write('ACQ:SOUR1:DATA?')
        d1 = np.frombuffer(self._read_block(), dtype='>f4').astype(float)
        self.write('ACQ:SOUR2:DATA?')
        d2 = np.frombuffer(self._read_block(), dtype='>f4').astype(float)
        self.write('ACQ:STOP')
        return d1, d2


def valid_decimation(dec):
    """The next factor OS 3.00 accepts at or above dec: 1, 2, 4, 8, 16, then any integer up to
    65536 ('ACQ:DEC:Factor 11' -> error 9506 'Value out of range', 7 Nov 2026)."""
    dec = int(np.ceil(dec))
    if dec >= 16:
        return min(dec, 65536)
    return next(d for d in (1, 2, 4, 8, 16) if d >= max(dec, 1))


def choose_decimation(f, periods=20, min_samples_per_period=40):
    """Finest decimation that still fits >= `periods` periods of f in one buffer."""
    if f <= 0:
        return 64
    dec_min = int(np.ceil(periods * FS_ADC / (N_BUF * f)))
    return valid_decimation(max(dec_min, 1))
