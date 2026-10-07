# IQ Mixer Calibration

This document describes the automated IQ mixer calibration system for Qudi. The calibration logic optimizes DC offsets and IQ imbalance parameters to minimize spurious signals (LO leakage and image) in the RF output.

## Why IQ Calibration is Needed

### The IQ Mixing Problem

An IQ (In-phase/Quadrature) mixer generates an RF output by combining two baseband signals—I and Q—with a local oscillator (LO). In an ideal mixer:

```
RF_out = I(t)·cos(ω_LO·t) ± Q(t)·sin(ω_LO·t)
```

With ideal quadrature, you can generate either sideband depending on the sign convention:
- **Upper sideband (USB)**: `f_RF = f_LO + f_IF` (default)
- **Lower sideband (LSB)**: `f_RF = f_LO - f_IF`

### Real-World Imperfections

#### 1. DC Offsets → LO Leakage

Any DC offset on the I or Q inputs creates a carrier at the LO frequency:

```
I(t) = I_offset + A·cos(ω_IF·t)
```

The DC term mixes with the LO to produce power at f_LO itself, called **LO leakage** or carrier feedthrough.

#### 2. IQ Imbalance → Image Rejection

If the I and Q paths have gain imbalance (g) or phase imbalance (φ ≠ 90°), the mixer produces power at both sidebands: the desired tone and the unwanted **image** at the opposite sideband. Even 1% gain error and 1° phase error yields only ~40 dB image rejection.

### Why This Matters for ODMR

In NV center ODMR experiments, poor mixer calibration causes:
- **LO leakage**: Spurious excitation at the wrong frequency
- **Image**: Excitation at a mirror frequency that may hit other resonances
- **Reduced dynamic range**: Spurs limit sensitivity

### Higher-Order Mixing Products

Higher-order mixing terms appear at multiples of the IF frequency: f_LO ± n·f_IF. Keeping IF power several dB below the mixer's 1 dB compression point helps minimize these harmonics.

## Calibration Approach

### Correction Model

We correct imperfections by adjusting the IF signal:

**DC Offset Correction:**
```
I_corrected = I_signal + I_offset
Q_corrected = Q_signal + Q_offset
```

**IQ Imbalance Correction:**
```
I_amplitude = A · (1 + g)
Q_amplitude = A · (1 - g)
I_phase = 0°
Q_phase = base_q + φ    (φ in radians, converted to degrees)
```

where `g` is the gain imbalance parameter and `φ` is the phase imbalance. `base_q` depends on the selected sideband:
- USB (default): `base_q = 270°`
- LSB: `base_q = 90°`

### Optimization Strategy

The calibration uses **Nelder-Mead optimization** (derivative-free simplex method) in two sequential stages:

1. **DC Offset Optimization**: Minimize power at f_LO (LO leakage)
   - Parameters: `[I_offset, Q_offset]`
   - Objective: Power measured at LO frequency (marker 1)

2. **IQ Imbalance Optimization**: Minimize power at the image sideband
   - Parameters: `[g, φ]`
   - Objective: Power measured at image frequency (marker 2)

The process iterates (default: 2 iterations) since DC and IQ corrections interact slightly through mixer nonlinearities. The best parameters from each DC/IQ step are applied immediately. After all iterations, the logic keeps the result with the lowest combined spur power (LO leakage + image, combined in linear units).

**Implementation details (robustness):**
- Each Nelder–Mead run starts with an explicit initial simplex around the current best parameters (configurable via `dc_simplex_step`, `iq_g_simplex_step`, `iq_phi_simplex_step`).
- During a full sweep, each point is warm-started from the previous successful point (`warm_start: true` by default).
- Transient instrument errors are handled by retrying failed points (`point_retry_count`, `point_retry_delay_s`). Failed points are logged in a separate `calibration_failed_points.tsv`.

### Calibration Grid

Calibration parameters depend on:
- **LO frequency**: Mixer behavior varies across the band
- **IF amplitude**: Nonlinearities cause amplitude-dependent distortion

We calibrate over a 2D grid (LO frequencies × IF amplitudes). During operation we use **2D interpolation**
(`scipy.interpolate.griddata`), for frequencies or amplitudes that don't correspond to any grid point exactly.


## Implementation

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    IQCalibrationLogic                           │
│  (qudi-iqo-modules/src/qudi/logic/iq_calibration_logic.py)      │
├─────────────────────────────────────────────────────────────────┤
│  • Nelder-Mead optimization loops                               │
│  • Calibration grid management                                  │
│  • SFDR calculation and spectrum plotting                       │
│  • Incremental CSV export with resume capability                │
│  • Qt signals for progress/status                               │
└──────────────┬─────────────────────────┬────────────────────────┘
               │                         │
    Connector  │                         │  Connector
               ▼                         ▼
┌──────────────────────────┐  ┌──────────────────────────────────┐
│ SpectrumMeasurement      │  │ MicrowaveInterface               │
│ Interface                │  │ (MicrowaveRedPitayaWindfreak)    │
├──────────────────────────┤  ├──────────────────────────────────┤
│ RhodeSchwarzRTO6         │  │ • calibration_set_dc_offsets()   │
│ (oscilloscope/)          │  │ • calibration_set_iq_correction()│
│                          │  │ • calibration_set_if_amplitude() │
│ • FFT via CALC:MATH1     │  │ • calibration_set_if_frequency() │
│ • Marker-based power     │  │ • calibration_enable_output()    │
│   measurement            │  │                                  │
└──────────────────────────┘  │         ▼                        │
                              │   RedPitayaIFSource              │
                              │   (ifsynth FPGA module)            │
                              └──────────────────────────────────┘
```

### Files

| File | Purpose |
|------|---------|
| `interface/spectrum_measurement_interface.py` | Abstract interface for spectrum analyzers |
| `hardware/oscilloscope/rohde_schwarz_rto6.py` | R&S RTO6 driver using RsInstrument |
| `logic/iq_calibration_logic.py` | Calibration orchestration and optimization |
| `hardware/microwave/mw_source_windfreak_synthnvpro_redpitaya.py` | `calibration_*` methods for IQ calibration |

### Key Classes

#### `IQCalibrationLogic`
Main orchestrator. Key methods:
- `calibrate_single_point(lo_freq_ghz, if_amplitude)` → `CalibrationPoint`
- `run_full_calibration(lo_frequencies, if_amplitudes)` → CSV path (single IF)
- `run_multi_if_calibration(if_frequencies, lo_frequencies, if_amplitudes)` → output directory (multi-IF)
- `get_calibration_parameters(filepath, lo_ghz, amplitude)` → interpolated (g, φ, I_off, Q_off)

#### `RhodeSchwarzRTO6`
Implements `SpectrumMeasurementInterface`:
- Uses FFT math function (`CALC:MATH1 "FFTmag(Ch1)"`)
- Cursor queries for amplitude measurement (`CURSn:Y1P?`)
- Properties: `center_frequency`, `span`, `resolution_bandwidth`


#### `MicrowaveRedPitayaWindfreak` (extended)
Added calibration methods that delegate to `RedPitayaIFSource`:
- `calibration_set_dc_offsets(i, q)` → sets `ifsynth.overall_dc_offset_a/b`
- `calibration_set_iq_correction(g, phi, amp)` → applies amplitude/phase correction
- `calibration_set_if_amplitude(amp)` → sets uncorrected amplitude
- `calibration_set_if_frequency(freq_hz)` → sets IF frequency
- `calibration_enable_output(enable)` → enables/disables IF output

## Configuration

### Example Configuration File

Config file: `C:\Users\aj92uwef\qudi\config\iq_calibration.cfg`

### Configuration Options Reference

#### Optimization Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `optimization_iterations` | 2 | Number of DC→IQ optimization cycles |
| `xatol` | 1e-4 | Nelder-Mead parameter tolerance |
| `fatol` | 3.0 | Nelder-Mead function tolerance (dB) |
| `maxiter` | 50 | Maximum iterations per optimization |

#### Robustness / Performance

| Option | Default | Description |
|--------|---------|-------------|
| `warm_start` | true | Use previous point's result as initial guess |
| `dc_simplex_step` | 0.05 | Initial simplex size for DC optimization |
| `iq_g_simplex_step` | 0.02 | Initial simplex size for g parameter |
| `iq_phi_simplex_step` | 0.05 | Initial simplex size for φ parameter |
| `point_retry_count` | 3 | Number of retries for failed points |
| `point_retry_delay_s` | 5.0 | Delay between retries (seconds) |

#### Spectrum Measurement Setup

| Option | Default | Description |
|--------|---------|-------------|
| `measurement_rbw_hz` | 1e6 | Resolution bandwidth for marker measurements (Hz) |
| `measurement_span_factor` | 4.1 | Span = factor × IF frequency |

#### Signal Presence Verification

| Option | Default | Description |
|--------|---------|-------------|
| `signal_presence_threshold_dbm` | -70.0 | Minimum signal power to proceed (dBm) |
| `abort_on_weak_signal` | false | Abort calibration if signal is below threshold |

#### SFDR Measurement

| Option | Default | Description |
|--------|---------|-------------|
| `enable_sfdr_measurement` | true | Capture wideband spectrum and calculate SFDR |
| `sfdr_span_multiplier` | 3.0 | SFDR span = base_span × multiplier |
| `sfdr_bandwidth_factor` | 8.0 | SFDR search bandwidth = factor × IF frequency |
| `sfdr_sweep_rbw_hz` | 1e6 | Resolution bandwidth for SFDR sweeps (Hz) |
| `save_spectrum_plots` | true | Generate before/after PDF comparison plots |
| `negative_sfdr_retry_count` | 2 | Retries with perturbed initial conditions if USB < LSB |

## Usage

### Starting Qudi

```bash
qudi -c C:\Users\aj92uwef\qudi\config\iq_calibration.cfg
```

### Multi-IF Calibration (Recommended)

For overnight calibration runs with multiple IF frequencies:

```python
# In IPython console after Qudi starts
cal = qudi.module_manager.get('iq_calibration_logic').instance

# Define calibration grid
lo_freqs = cal.get_default_lo_frequencies(center_ghz=2.87, span_ghz=0.3, num_points=40)
if_amps  = cal.get_default_if_amplitudes(min_amp=0.0005, max_amp=0.3, num_points=25)

# Define IF frequencies to calibrate
if_freqs = [19.422e6, 21.580e6, 23.738e6]

# Run calibration for all IFs (takes several hours). The command can time-out in the ipython console,
# but calibration still continues to run
out_dir = cal.run_multi_if_calibration(if_freqs, lo_freqs, if_amps)

```

### Resuming an Interrupted Calibration

If a calibration is interrupted (crash, power loss, etc.), you can resume from the checkpoint:

```python
# Resume from existing partial results
out_dir = cal.run_multi_if_calibration(
    if_freqs, lo_freqs, if_amps,
    resume_from_checkpoint=True,
    output_dir='C:/calibration_results/2026-01-20-15-30-00'  # Path to interrupted run
)
```

The resume feature:
- Loads completed points from existing CSV files
- Skips already-completed (LO, amplitude) combinations
- Initializes warm-start from the last successful point
- Continues saving incrementally to the same files

### Single-IF Calibration

For calibrating a single IF frequency:

```python
cal = qudi.module_manager.get('iq_calibration_logic').instance

lo_freqs = cal.get_default_lo_frequencies(center_ghz=2.87, span_ghz=0.3, num_points=61)
if_amps  = cal.get_default_if_amplitudes(min_amp=0.01, max_amp=0.3, num_points=10)

# Run calibration (takes ~1-2 hours)
output_file = cal.run_full_calibration(lo_freqs, if_amps, if_frequency_hz=21.58e6)
```

### Single Point Calibration (Testing)

For testing or debugging:

```python
result = cal.calibrate_single_point(
    lo_freq_ghz=2.87,
    if_amplitude=0.1,
    if_frequency_hz=21.58e6
)
print(f'LO leakage: {result.lo_leakage_dbm:.1f} dBm')
print(f'Image: {result.image_power_dbm:.1f} dBm')
print(f'SFDR: {result.sfdr_db:.1f} dB')
```

## Output Format

### Directory Structure

```
calibration_results/
└── 2026-01-20-21-30-00/              # Timestamped main directory
    ├── IF_19.422MHz/
    │   ├── calibration_redpitaya_all_results_IF_19.422MHz.csv
    │   ├── spectrum_if_0.100_lo_2.870_GHz.pdf   # Before/after plots
    │   └── calibration_failed_points.tsv        # (if any failures)
    ├── IF_21.580MHz/
    │   └── ...
    ├── IF_23.738MHz/
    │   └── ...
    ├── calibration_redpitaya_GRAND_SUMMARY_all_IFs.csv  # Combined results
    ├── per_if_calibration_files.txt                     # Index of per-IF files
    └── failed_if_frequencies.tsv                        # (if any IFs failed)
```

### CSV Columns

The calibration CSV files (tab-separated) contain:

| Column | Description |
|--------|-------------|
| `sideband` | `'upper'` (USB) or `'lower'` (LSB) |
| `if_frequency_hz` | IF frequency used for this point (multi-IF mode) |
| `base_if_frequency_mhz` | IF frequency in MHz (for reference) |
| `lo_frequency_ghz` | LO frequency |
| `if_amplitude` | IF amplitude (0–1) |
| `g` | Gain imbalance parameter |
| `phi` | Phase imbalance (radians) |
| `I_offset` | I channel DC offset |
| `Q_offset` | Q channel DC offset |
| `lo_leakage_dbm` | LO leakage power (instrument units) |
| `image_power_dbm` | Image power (instrument units) |
| `sfdr_db` | Spurious-free dynamic range (dB) |

Note: The `*_dbm` columns contain values in the instrument's native units. For the RTO6 FFT, these are typically relative dB (not calibrated dBm).

### Applying Calibration

After calibration, update `calibration_files` in your microwave hardware config:

```yaml
mw_source_rp_windfreak:
    options:
        calibration_files:
            19.422e6: 'C:/calibration_results/2026-01-20-21-30-00/IF_19.422MHz/calibration_redpitaya_all_results_IF_19.422MHz.csv'
            21.580e6: 'C:/calibration_results/2026-01-20-21-30-00/IF_21.580MHz/calibration_redpitaya_all_results_IF_21.580MHz.csv'
            23.738e6: 'C:/calibration_results/2026-01-20-21-30-00/IF_23.738MHz/calibration_redpitaya_all_results_IF_23.738MHz.csv'
```

The `RedPitayaIFSource` automatically interpolates and applies corrections during normal operation via `set_multi_frequency_signal()`.

## Behavior Notes

### Incremental Saving

Each calibration point is saved to disk immediately after completion. This prevents data loss if the calibration is interrupted (crash, timeout, power loss). The feature works automatically—no configuration needed.

### Warm-Start Between Points

When `warm_start: true` (default), each point uses the previous successful point's parameters as its initial guess. This typically improves convergence and reduces calibration time.

### Single-Tone Enforcement

The calibration logic enforces single-tone mode during calibration. It automatically:
1. Switches the microwave module to `multi_frequency_mode='single'`
2. Verifies only one IF component is active
3. Restores the previous mode after calibration completes

This ensures calibration measures only the target IF component without interference from other tones.

### SFDR Measurement

When `enable_sfdr_measurement: true` (default):
1. Captures wideband spectrum before calibration
2. Runs the DC/IQ optimization
3. Captures wideband spectrum after calibration
4. Calculates SFDR = target_power − max_spur_power
5. Identifies the harmonic origin of the maximum spur (LO, LO±2IF, LO±3IF, etc.)
6. Generates annotated PDF comparison plots (if `save_spectrum_plots: true`)

## Expected Results

| Metric | Uncalibrated | Automated Calibration |
|--------|--------------|----------------------|
| LO leakage | −20 to −30 dBm | < −50 dBm |
| Image rejection | 20–30 dB | ~40 dB |
| SFDR | — | Computed automatically (see `sfdr_db` column) |

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| Optimization doesn't converge | Tolerances too tight | Increase `fatol` to 5 dB |
| Poor LO suppression | DC offset range exceeded | Check ifsynth limits (±1.0) |
| Poor image rejection | Phase out of range | Verify φ stays within ±30° |
| Interpolation warnings | Point outside calibration grid | Expand LO/amplitude ranges |
| Calibration degrades over time | Temperature drift | Recalibrate after thermal equilibrium |
| Visible harmonics at n×f_IF | IF power too high | Reduce IF amplitude |
| "Weak signal detected" warning | Signal below threshold | Check RF connections, enable outputs |
| Calibration aborts on weak signal | `abort_on_weak_signal: true` | Set to false or fix signal chain |
| Resume doesn't find previous data | Wrong output_dir path | Verify path to interrupted run |
