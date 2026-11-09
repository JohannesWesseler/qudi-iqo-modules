# -*- coding: utf-8 -*-
"""
Rohde & Schwarz NGP800 series power supply (NGP802/804/814/822/824), with a per-channel write
guard: only `writable_channels` can be configured or switched; every other channel is read only.
All channels can be read (ProcessValueInterface: '<n>_voltage', '<n>_current').

Behaviour checked on the NGP824 of the current source (manual NGP800 v14, 5601.5610.02):
- The electronic fuse (OCP) trips when the channel reaches its current limit (manual 5.5.1);
  there is no separate fuse threshold. "Fuse at 2.1 A" = current limit 2.1 A with FUSE on.
- OUTPut:GENeral (master) switches *all* selected channels. Channels are therefore switched with
  OUTPut:SELect plus a channel list, and the master is only switched on (never off) when it is
  off (e.g. after a restart of the supply), all writable channels are off and deselected, and
  every other selected channel is listed in `master_allowed_channels` at its set voltage.
- Per-channel commands use the channel-list suffix "(@n)". FUSE:LINK has no list form: it acts
  on the channel chosen with INSTrument:NSELect, which is restored afterwards.
- The output-state readback lags an off command by up to ~1 s.

Example config (the current source's supply; Ch3 is used by another setup and may stay on at 1.0 V):

    ngp824:
        module.Class: 'power_supply.rohde_schwarz_ngp800.RohdeSchwarzNGP800'
        options:
            resource: 'USB0::0x0AAD::0x0197::5601.4007k03-101169::INSTR'
            fallback_resource: 'TCPIP0::169.254.8.183::inst0::INSTR'   # optional
            serial: '101169'            # refuse another instrument
            writable_channels: [1, 4]
            master_allowed_channels: {3: 1.0}   # may come on with the master at this voltage
            voltage_max: 16.5
            current_max: 2.5

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
from typing import Dict, Iterable, Sequence

from qudi.core.configoption import ConfigOption
from qudi.util.mutex import RecursiveMutex
from qudi.interface.process_control_interface import ProcessValueInterface, ProcessControlConstraints


class RohdeSchwarzNGP800(ProcessValueInterface):
    """NGP800 power supply with a channel write guard."""

    _resource = ConfigOption('resource', missing='error')
    _fallback_resource = ConfigOption('fallback_resource', default=None, missing='nothing')
    _serial = ConfigOption('serial', default=None, missing='warn')
    _writable = ConfigOption('writable_channels', default=[], missing='warn')
    _master_allowed = ConfigOption('master_allowed_channels', default=dict(), missing='nothing')
    _voltage_max = ConfigOption('voltage_max', default=16.5, missing='nothing')
    _current_max = ConfigOption('current_max', default=2.5, missing='nothing')
    _timeout_ms = ConfigOption('timeout_ms', default=3000, missing='nothing')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = RecursiveMutex()
        self._rm = None
        self._inst = None
        self.idn = ''
        self._n_channels = 4

    # --- qudi --------------------------------------------------------------------------------
    def on_activate(self):
        import pyvisa
        self._rm = pyvisa.ResourceManager()
        last = None
        for res in [r for r in (self._resource, self._fallback_resource) if r]:
            try:
                self._inst = self._rm.open_resource(res)
                self._inst.timeout = int(self._timeout_ms)
                self.idn = self.query('*IDN?')
                break
            except Exception as e:
                last = e
                self.log.warning(f'NGP800 at {res} failed: {e}')
                self._inst = None
        if self._inst is None:
            raise IOError(f'no NGP800 reachable: {last}')
        if 'NGP8' not in self.idn or (self._serial and str(self._serial) not in self.idn):
            raise IOError(f'unexpected instrument: {self.idn}')
        model = self.idn.split(',')[1].strip()          # e.g. NGP824: 4 channels, NGP822: 2
        self._n_channels = 2 if model.endswith('2') else 4
        self._master_allowed = {int(k): float(v) for k, v in dict(self._master_allowed).items()}
        self._writable = [int(c) for c in self._writable]
        self.log.info(f'{self.idn}; writable channels {self._writable}')

    def on_deactivate(self):
        try:
            if self._inst is not None:
                self._inst.close()
        finally:
            self._inst = None

    # --- low level ---------------------------------------------------------------------------
    def query(self, cmd: str) -> str:
        with self._lock:
            return self._inst.query(cmd).strip()

    def _write(self, cmd: str, channels: Sequence[int]):
        bad = [c for c in channels if c not in self._writable]
        if bad or not channels:
            raise PermissionError(f'NGP800: write to channel(s) {bad or channels} refused: {cmd}')
        with self._lock:
            self.log.debug(f'NGP <- {cmd}')
            self._inst.write(cmd)

    def errors(self):
        out = []
        for _ in range(20):
            e = self.query('SYST:ERR?')
            if e.startswith('0,'):
                break
            out.append(e)
        return out

    @property
    def channels(self):
        return tuple(range(1, self._n_channels + 1))

    @property
    def writable_channels(self):
        return tuple(self._writable)

    # --- read back (any channel) -------------------------------------------------------------
    def channel_state(self, ch: int) -> Dict[str, float]:
        a = f'(@{int(ch)})'
        f = lambda c: float(self.query(f'{c}? {a}'))
        with self._lock:
            return dict(ch=int(ch), v_set=f('VOLT'), i_set=f('CURR'), sel=int(f('OUTP:SEL')),
                        out=int(f('OUTP')), fuse=int(f('FUSE')), fuse_delay=f('FUSE:DEL'),
                        fuse_delay_init=f('FUSE:DEL:INIT'), fuse_trip=int(f('FUSE:TRIP')),
                        v_meas=f('MEAS:VOLT'), i_meas=f('MEAS:CURR'))

    def state_all(self) -> Dict:
        with self._lock:
            return dict(master=int(self.query('OUTP:GEN?')), selected=int(self.query('INST:NSEL?')),
                        channels=[self.channel_state(c) for c in self.channels])

    def measure(self, channels: Iterable[int]) -> Dict[str, float]:
        """Measured V, I plus fuse-trip and output flags: keys v<n>, i<n>, trip<n>, on<n>."""
        out = {}
        with self._lock:
            for ch in channels:
                out[f'v{ch}'] = float(self.query(f'MEAS:VOLT? (@{ch})'))
                out[f'i{ch}'] = float(self.query(f'MEAS:CURR? (@{ch})'))
                out[f'trip{ch}'] = int(self.query(f'FUSE:TRIP? (@{ch})'))
                out[f'on{ch}'] = int(self.query(f'OUTP? (@{ch})'))
        return out

    def output_states(self, channels: Iterable[int]) -> Dict[int, int]:
        with self._lock:
            return {int(ch): int(self.query(f'OUTP? (@{ch})')) for ch in channels}

    # --- configuration (writable channels only) -----------------------------------------------
    def configure_channels(self, channels: Sequence[int], voltage: float, current_limit: float,
                           fuse: bool = True, fuse_delay: float = 0.05,
                           fuse_delay_init: float = 0.1, link_fuses: bool = True) -> Dict:
        """Voltage, current limit (= fuse level), fuse with delays and (optionally) FuseLink
        among the channels. Only while all of them are off. Returns the verified state."""
        channels = [int(c) for c in channels]
        if not 0 <= voltage <= self._voltage_max:
            raise ValueError(f'voltage {voltage} outside 0..{self._voltage_max} V')
        if not 0 <= current_limit <= self._current_max:
            raise ValueError(f'current limit {current_limit} outside 0..{self._current_max} A')
        with self._lock:
            if any(self.output_states(channels).values()):
                raise RuntimeError(f'configure channels {channels} only while they are off')
            for ch in channels:
                self._write(f'VOLT {voltage:.3f},(@{ch})', [ch])
                self._write(f'CURR {current_limit:.3f},(@{ch})', [ch])
                self._write(f'FUSE:DEL {fuse_delay},(@{ch})', [ch])
                self._write(f'FUSE:DEL:INIT {fuse_delay_init},(@{ch})', [ch])
                self._write(f'FUSE {int(bool(fuse))},(@{ch})', [ch])
            if link_fuses and len(channels) > 1:
                prev = int(self.query('INST:NSEL?'))
                try:
                    for ch in channels:
                        self._write(f'INST:NSEL {ch}', [ch])
                        for other in channels:
                            if other != ch:
                                self._write(f'FUSE:LINK {other}', channels)
                finally:
                    self._inst.write(f'INST:NSEL {prev}')   # selection only, no output change
            errs = self.errors()
            if errs:
                raise IOError(f'NGP800 errors: {errs}')
            return self.verify_channels(channels, voltage, current_limit, fuse, fuse_delay,
                                        fuse_delay_init, link_fuses)

    def verify_channels(self, channels, voltage, current_limit, fuse=True, fuse_delay=0.05,
                        fuse_delay_init=0.1, link_fuses=True) -> Dict:
        rep = {}
        ok = True
        with self._lock:
            for ch in channels:
                s = self.channel_state(ch)
                rep[ch] = s
                ok &= abs(s['v_set'] - voltage) < 1e-3 and abs(s['i_set'] - current_limit) < 1e-3
                ok &= s['fuse'] == int(bool(fuse)) and abs(s['fuse_delay'] - fuse_delay) < 1e-3
                ok &= abs(s['fuse_delay_init'] - fuse_delay_init) < 1e-3
            if link_fuses and len(channels) > 1:
                prev = int(self.query('INST:NSEL?'))
                try:
                    for ch in channels:
                        self._inst.write(f'INST:NSEL {ch}')
                        links = self.query('FUSE:LINK?').replace(' ', '').split(',')
                        rep[f'link{ch}'] = links
                        ok &= all(str(o) in links for o in channels if o != ch)
                finally:
                    self._inst.write(f'INST:NSEL {prev}')
        rep['ok'] = bool(ok)
        return rep

    # --- switching (writable channels only) ---------------------------------------------------
    def switch_on(self, channels: Sequence[int]) -> None:
        """Switch the channels on together (one command). Switches the master on first if it is
        off, under the conditions of master_on()."""
        channels = [int(c) for c in channels]
        with self._lock:
            if int(self.query('OUTP:GEN?')) != 1:
                self.master_on()
            self._write(f'OUTP:SEL 1,(@{",".join(map(str, channels))})', channels)

    def switch_off(self, channels: Sequence[int]) -> None:
        channels = [int(c) for c in channels]
        self._write(f'OUTP:SEL 0,(@{",".join(map(str, channels))})', channels)

    def master_on(self) -> None:
        """Master output on, only if all writable channels are off (they are deselected here)
        and every other selected channel is allowed (master_allowed_channels) at its set
        voltage; non-selected channels stay off anyway."""
        with self._lock:
            if any(self.output_states(self._writable).values()):
                raise RuntimeError('switch the master only with the writable channels off')
            self.switch_off(self._writable)
            for ch in self.channels:
                if ch in self._writable:
                    continue
                if not int(self.query(f'OUTP:SEL? (@{ch})')):
                    continue
                v = float(self.query(f'VOLT? (@{ch})'))
                allowed = self._master_allowed.get(ch)
                if allowed is None or abs(v - allowed) > 1e-3:
                    raise PermissionError(f'Ch{ch} is selected at {v} V and would come on with the '
                                          f'master (allowed: {allowed}); master switch refused')
            self.log.info('NGP <- OUTP:GEN 1')
            self._inst.write('OUTP:GEN 1')
        time.sleep(0.5)

    # --- ProcessValueInterface ----------------------------------------------------------------
    @property
    def constraints(self) -> ProcessControlConstraints:
        names = [f'{c}_{q}' for c in self.channels for q in ('voltage', 'current')]
        return ProcessControlConstraints(
            process_channels=names,
            units={n: ('V' if n.endswith('voltage') else 'A') for n in names},
            dtypes={n: float for n in names})

    def set_activity_state(self, channel: str, active: bool) -> None:
        if not active:
            raise ValueError('NGP800 read-back channels are always active')

    def get_activity_state(self, channel: str) -> bool:
        return True

    def get_process_value(self, channel: str) -> float:
        ch, q = channel.split('_')
        return float(self.query(f'MEAS:{"VOLT" if q == "voltage" else "CURR"}? (@{int(ch)})'))
