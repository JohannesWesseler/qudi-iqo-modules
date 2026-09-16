"""Pure helpers for reproducible ODMR tracking-sensitivity comparisons.

The helpers deliberately have no Qudi or hardware dependencies.  This keeps the
definition of a comparison matrix and the discriminator/correction sign
convention unit-testable without a Red Pitaya.
"""

from itertools import product
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


NV_GYROMAGNETIC_RATIO_HZ_PER_NT = 28.024


def _finite_positive(values: Iterable[float], name: str) -> List[float]:
    result = [float(value) for value in values]
    if not result or any(not np.isfinite(value) or value <= 0 for value in result):
        raise ValueError(f'{name} must contain finite positive values')
    return result


def build_tracking_conditions(modes: Iterable[str],
                              bandwidths_hz: Iterable[float],
                              smith_gain_multipliers: Iterable[float],
                              smith_delay_samples: Iterable[Optional[int]]) -> List[Dict[str, Any]]:
    """Expand GUI settings into a deterministic measurement-condition list.

    Open loop is emitted once.  Conventional tracking is emitted once per
    bandwidth.  Smith tracking spans bandwidth, gain multiplier and delay.
    ``None`` for the delay means to use the delay advertised by the FPGA image.
    """
    normalized_modes = []
    for mode in modes:
        mode = str(mode).strip().lower()
        if mode not in ('open_loop', 'closed_loop_conventional', 'closed_loop_smith'):
            raise ValueError(f'unknown tracking sensitivity mode {mode!r}')
        if mode not in normalized_modes:
            normalized_modes.append(mode)

    if not normalized_modes:
        raise ValueError('at least one tracking sensitivity mode must be selected')

    needs_closed = any(mode != 'open_loop' for mode in normalized_modes)
    bandwidths = (_finite_positive(bandwidths_hz, 'controller bandwidths')
                  if needs_closed else [])
    gains = (_finite_positive(smith_gain_multipliers, 'Smith gain multipliers')
             if 'closed_loop_smith' in normalized_modes else [])

    delays: List[Optional[int]] = []
    if 'closed_loop_smith' in normalized_modes:
        for delay in smith_delay_samples:
            if delay is None:
                value = None
            else:
                value = int(delay)
                if not 1 <= value < 128:
                    raise ValueError('Smith delays must be in [1, 127] samples or None')
            if value not in delays:
                delays.append(value)
        if not delays:
            delays = [None]

    conditions: List[Dict[str, Any]] = []
    if 'open_loop' in normalized_modes:
        conditions.append({
            'measurement_mode': 'open_loop',
            'controller_algorithm': 'disabled',
            'controller_bandwidth_hz': None,
            'smith_gain_multiplier': None,
            'smith_delay_samples': None,
        })
    if 'closed_loop_conventional' in normalized_modes:
        for bandwidth in bandwidths:
            conditions.append({
                'measurement_mode': 'closed_loop',
                'controller_algorithm': 'conventional',
                'controller_bandwidth_hz': bandwidth,
                'smith_gain_multiplier': None,
                'smith_delay_samples': None,
            })
    if 'closed_loop_smith' in normalized_modes:
        for bandwidth, gain, delay in product(bandwidths, gains, delays):
            conditions.append({
                'measurement_mode': 'closed_loop',
                'controller_algorithm': 'smith_linear',
                'controller_bandwidth_hz': bandwidth,
                'smith_gain_multiplier': gain,
                'smith_delay_samples': delay,
            })
    return conditions


def reconstruct_field_traces(error_signal: np.ndarray,
                             correction_hz: Optional[np.ndarray],
                             signed_slope_per_hz: float,
                             correction_inverted: bool = False,
                             gyromagnetic_ratio_hz_per_nt: float =
                             NV_GYROMAGNETIC_RATIO_HZ_PER_NT) -> Dict[str, np.ndarray]:
    """Convert synchronized discriminator and correction traces to field.

    The fitted discriminator follows ``error = slope * (drive - resonance)``.
    Consequently the residual resonance displacement is ``-error/slope``.
    The FPGA FTW correction has the RF sign for upper-sideband operation and the
    opposite sign for lower-sideband (``correction_inverted=True``) operation.

    Returned arrays retain NaNs.  Means are not removed here so the original
    operating point remains available in the saved raw dataset; spectral
    analysis may remove a mean independently.
    """
    error = np.asarray(error_signal, dtype=np.float64)
    slope = float(signed_slope_per_hz)
    gamma = float(gyromagnetic_ratio_hz_per_nt)
    if not np.isfinite(slope) or slope == 0:
        raise ValueError('signed discriminator slope must be finite and non-zero')
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('gyromagnetic ratio must be finite and positive')

    residual_hz = -error / slope
    if correction_hz is None:
        correction_rf_hz = np.zeros_like(residual_hz)
    else:
        correction_rf_hz = np.asarray(correction_hz, dtype=np.float64)
        if correction_rf_hz.shape != residual_hz.shape:
            raise ValueError('error and correction traces must have identical shapes')
        if correction_inverted:
            correction_rf_hz = -correction_rf_hz

    estimate_hz = correction_rf_hz + residual_hz
    return {
        'residual_frequency_hz': residual_hz,
        'correction_frequency_hz': correction_rf_hz,
        'estimated_frequency_hz': estimate_hz,
        'residual_field_nt': residual_hz / gamma,
        'correction_field_nt': correction_rf_hz / gamma,
        'estimated_field_nt': estimate_hz / gamma,
    }


def json_safe(value: Any) -> Any:
    """Recursively convert NumPy-heavy metadata into strict JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
