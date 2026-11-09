# -*- coding: utf-8 -*-
"""
Model of the load connected to the amplifier, from a measured load file (calibration/load_*.json).

The amplifier is a voltage amplifier, so the current it drives is set by the load:
I(f) = gm(f) * OUT1, with the transconductance gm(f) = I / OUT1 measured for this amplifier with
this load. LoadModel holds that table, the load impedance Z(f) = output voltage / I, and the DC
values, and continues them outside the measured range with an R-L model.

Load file (JSON, see calibration/load_helmholtz_r17_5mm.json):
    dc_resistance, dc_transconductance     Ohm, A/V (DC measurement)
    rated_current                          A peak, default current limit with this load
    heating                                limits above the continuous currents (see the driver)
    table: f, z_re, z_im, i_out_abs, i_out_ph_deg (+ mon_out_*)
    fit: R0 (Ohm), L_lin_uH, amp_fc_kHz    R-L fit and amplifier bandwidth, for extrapolation
    description, measured, notes           text

A new load needs its own file: measure gm(f) and Z(f) with this amplifier (README, "Using
another load").
"""

import json

import numpy as np

AMP_GAIN = 16.0          # nominal voltage gain (1 + R8/R7); cancels in the extrapolation


class LoadModel:
    """Transconductance gm(f) = I/OUT1 (complex, A/V) and load Z(f) = Vout/I (complex, Ohm).

    Inside the measured range: interpolation in log f (magnitude in log, phase linear).
    Outside: R-L model with the fitted R0, L and the amplifier pole fc, scaled to the nearest
    measured point. DC: gm_dc (measured DC transconductance) and r_dc."""

    def __init__(self, f, z, gm, r0, inductance, gain=AMP_GAIN, amp_fc=735e3, gm_dc=None,
                 r_dc=None, info=None):
        o = np.argsort(f)
        self.f = np.asarray(f, float)[o]
        self.z = np.asarray(z, complex)[o]
        self.gm = np.asarray(gm, complex)[o]
        self.r0 = float(r0)
        self.inductance = float(inductance)
        self.gain = float(gain)
        self.amp_fc = float(amp_fc)
        self.r_dc = float(r_dc) if r_dc else self.r0
        self.gm_dc = float(gm_dc) if gm_dc else abs(self.gm[0])
        self.info = dict(info or {})

    @classmethod
    def from_file(cls, path):
        """Read a load file. info keeps everything except the table (rated_current, heating,
        description, ...)."""
        with open(path) as fh:
            d = json.load(fh)
        tab, fit = d['table'], d['fit']
        z = np.asarray(tab['z_re']) + 1j * np.asarray(tab['z_im'])
        gm = np.asarray(tab['i_out_abs']) * np.exp(1j * np.radians(tab['i_out_ph_deg']))
        info = {k: v for k, v in d.items() if k != 'table'}
        info['file'] = str(path)
        return cls(tab['f'], z, gm, fit['R0'], fit['L_lin_uH'] * 1e-6,
                   amp_fc=fit.get('amp_fc_kHz', 735.0) * 1e3, gm_dc=d['dc_transconductance'],
                   r_dc=d['dc_resistance'], info=info)

    # model parts
    def _z_model(self, f):
        return self.r0 + 2j * np.pi * f * self.inductance

    def _gm_model(self, f):
        return self.gain / (1 + 1j * f / self.amp_fc) / self._z_model(f)

    def _interp(self, f, table, model):
        f = np.atleast_1d(np.asarray(f, float))
        out = np.empty(f.shape, complex)
        lo, hi = f < self.f[0], f > self.f[-1]
        mid = ~(lo | hi)
        if mid.any():
            lf = np.log(f[mid])
            mag = np.exp(np.interp(lf, np.log(self.f), np.log(np.abs(table))))
            ph = np.interp(lf, np.log(self.f), np.unwrap(np.angle(table)))
            out[mid] = mag * np.exp(1j * ph)
        if lo.any():
            out[lo] = model(f[lo]) * table[0] / model(self.f[0])
        if hi.any():
            out[hi] = model(f[hi]) * table[-1] / model(self.f[-1])
        return out

    def transconductance(self, f):
        """I/OUT1 (A/V, complex); f = 0: gm_dc."""
        f = np.atleast_1d(np.asarray(f, float))
        out = self._interp(np.maximum(f, 1e-3), self.gm, self._gm_model)
        out[f <= 0] = self.gm_dc
        return out

    def impedance(self, f):
        """Vout/I (Ohm, complex); f = 0: r_dc."""
        f = np.atleast_1d(np.asarray(f, float))
        out = self._interp(np.maximum(f, 1e-3), self.z, self._z_model)
        out[f <= 0] = self.r_dc
        return out

    def predicted_output(self, current, fs):
        """Output voltage expected for a current trace (A) sampled at fs: Z(f) applied in the
        frequency domain (periodic continuation; window the trace for non-periodic records)."""
        current = np.asarray(current, float)
        spec = np.fft.rfft(current)
        freqs = np.fft.rfftfreq(len(current), 1 / fs)
        return np.fft.irfft(spec * self.impedance(freqs), n=len(current))

    def output_ratio(self, current, output, fs, bandwidth=200e3):
        """rms(measured output) / rms(output expected from the measured current), both
        Hann-windowed and limited to `bandwidth`. ~1 for a healthy load; >> 1 open load,
        << 1 short. Works for any waveform including DC. The band limit matters: |Z| grows
        with f (~500 Ohm at 4 MHz), so ADC noise at high sample rates would otherwise look like
        a large expected output (false 'short', seen 2026-11-06 at decimation 16). The window
        keeps the FFT's periodic continuation from turning the jump at the record ends into a
        spike (L dI/dt). Returns (ratio, rms of the windowed expected output)."""
        n = len(current)
        w = np.hanning(n)
        freqs = np.fft.rfftfreq(n, 1 / fs)
        band = freqs <= bandwidth
        si = np.fft.rfft(np.asarray(current, float) * w) * band
        pred = np.fft.irfft(si * self.impedance(freqs), n=n)
        meas = np.fft.irfft(np.fft.rfft(np.asarray(output, float) * w) * band, n=n)
        rp = float(np.sqrt(np.mean(pred ** 2)))
        rm = float(np.sqrt(np.mean(meas ** 2)))
        return (rm / rp if rp > 0 else np.inf), rp
