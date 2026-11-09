# Automatic fiber coupling (motorized mirror mounts)

Two motorized mirror mounts (four geared 28BYJ-48 steppers on an Arduino Mega, no encoders) steer
the laser into a single-mode fiber. The coupling metric is the NV fluorescence seen by the
Newport Nirvana detector: its signal-monitor output goes to Red Pitaya IN2. The fluorescence is
proportional to the power through the fiber, so it peaks at the best mirror positions.
Hardware characterisation and the history of the algorithms: `Laser_alignment/docs/`
(`REPORT.md`, `CHARACTERIZATION_2026-10-19.md`).

## Use

| Button | What it does | Time | When |
|---|---|---|---|
| **Quick align** | peak scan (steep axis) of both motor pairs, repeated while a pair moves by more than its peak width | ~30 s | after drifts, e.g. 2-3 times a day |
| **Fine align** | quick align, then a beam-walk scan (flat axis) of each pair with both pairs re-peaked at every point, then quick align. It moves on the flat axis only if the parabola fit predicts a significant gain | ~8 min | when quick align does not reach the usual value |
| **Calibrate** | every motor scanned up, down and after +-300-step excursions; then the peak width of each pair | ~5 min | weekly (the GUI shows when it is due), and after mechanical work |

Keep the microwave off (or at least not sweeping) while aligning: ODMR contrast changes the
fluorescence and with it the metric.

After every run the GUI shows the result with checks (green: fine, orange: look at it):

- the signal before and after, and the fraction of the best value so far (orange below 90 %);
- per pair: the peak width relative to the calibration (a slipping motor or loose driver cable
  makes it ~2x wider) and the signal after parking relative to the scan maximum;
- the estimated power at the NVs and coupling efficiency, if a power-meter calibration exists;
- whether the motor calibration is due.

Each run is saved to the data directory (`fiber_coupling_<routine>`: time, motor positions and
signal of every reading; metadata with the result and checks; PNG with the scans).

**Power meter.** Measure the power in front of the 99:1 fiber splitter, reconnect, then measure at
the 1 % port (Nirvana reference fiber disconnected; the signal monitor only sees the signal
photodiode, so the metric is unaffected), enter both and press Store. The logic reads the
coupling signal at that moment and stores: efficiency = P_tap (1 + r) / P_in and the power at the
main port = r P_tap, with r = `splitter_ratio` (main/tap; measure it once for the splitter in use).
Afterwards the GUI scales both with the live signal. That estimate assumes an unchanged laser
power and a fluorescence linear in the power (checked to r = 0.998 at lower power); repeat the
calibration after a change of the laser power, and press "Reset best value" (manual control).

**Scripts / other modules** (e.g. a qudi task or notebook between measurements):

```python
fc = fiber_coupling_logic          # module instance
fc.start_routine('quick')
result = fc.wait_until_idle(timeout=300)   # dict with 'verdict', 'final', 'checks', ...
```

## Modules and configuration

| Module | Class | Role |
|---|---|---|
| `interface/stepper_mirror_mount_interface.py` | `StepperMirrorMountInterface` | relative steps, logical positions, coil release |
| `hardware/motor/arduino_mirror_mount.py` | `ArduinoMirrorMount` | Arduino firmware v1-3 (binary protocol) |
| `hardware/redpitaya/redpitaya_input_monitor.py` | `RedPitayaInputMonitor` | averaged IN1/IN2 levels as process values (scope in rolling mode, shared pyrpl instance) |
| `logic/fiber_coupling_logic.py` | `FiberCouplingLogic` | routines, checks, calibrations, history, saving |
| `gui/fiber_coupling/fiber_coupling_gui.py` | `FiberCouplingGui` | two buttons, result, power meter, calibration, history, manual jogs |

Example: `C:\Users\aj92uwef\qudi\config\fiber_coupling.cfg` (stand-alone). To add it to an
experiment config, copy the four entries **with the same module names**, and give
`redpitaya_input_monitor` the experiment's `redpitaya_config_name` and `redpitaya_fpga_filename`
so it shares the pyrpl instance and the FPGA image. Every image of the ODMR fork has the board
scope; nothing else in qudi uses it.

Where the state lives:

| What | Where | Shared between configs |
|---|---|---|
| pair definitions (motors, start values of steep/flat directions and width), timing options, splitter ratio, metric gain/offset | config file | no, but copy it |
| calibrated ridge frame, calibration history, power calibration, alignment history, best value | status variables of `fiber_coupling_logic` (`%APPDATA%\qudi\status-FiberCouplingLogic_logic_fiber_coupling_logic.cfg`) | yes, by module name |
| logical motor positions | status variables of `mirror_mount` | yes, by module name |
| every run (raw readings, checks, plots) | data directory | - |

The calibrated frame replaces the config values of a pair only while its `motors` match.
Logical positions are saved when the module is deactivated; they matter only as differences.

## Metric: Nirvana signal monitor on IN2

- The monitor output is negative; `gain` and `offset` of the channel give mV at the monitor
  output above the dark level. With the BNC "6 dB" pad on the monitor cable (0.3795 into the
  high-impedance input, measured): `gain: -2635`, `offset: -0.00512` V. Change both if the pad
  changes. Without a pad IN2 clips at ~1 V (the peak was ~2.6 V at the monitor after the power
  increase of 2026-10-19).
- A reading averages the newest scope samples covering `averaging_time` (20 ms) and waits one
  averaging time first, so all samples postdate the preceding motor move. A 10-50 kHz modulation
  averages out (< 0.2 % residual at 10 kHz).
- The scope decimation must be set as an attribute: `scope.setup(decimation=...)` silently
  ignores it. The module checks the value.
- Noise at the optimum (MW off): 0.15 % reading to reading, 1.35 % rms slow fluctuation
  (seconds; laser or fluorescence). `metric_noise` (1.5 %) and `compare_tolerance` (2 %) are set
  from that.

## Algorithm

Per motor pair (one motor on each mirror tilting in the same plane; on the table (M1, M4) and
(M2, M3)), in the ridge frame (steep = mirrors against each other, narrow peak, FWHM ~50-65
steps; flat = both together, beam walk, flat within a few % over +-360 steps):

- **Backlash** (40-60 steps measured): a motor reverses only through an overshoot of
  `backlash_approach` (120) steps.
- **Steep peak (replay rule):** pre-load `preload_steps` (130) in scan direction, scan
  +-`steep_range_sigma` sigma in steps of sigma/`steep_points_per_sigma`, then repeat the same
  pre-load and sub-steps up to the best grid point. Recentres if the peak is at the scan edge.
  Parking on the fitted Gaussian centre instead was tested (6 quick runs each, alternating): it
  ended ~3.5 % lower. The replay of M1_M4 ends systematically at ~96 % of its scan maximum,
  M2_M3 at ~98 %; `park_tolerance` (6 %) warns only above that.
- **Beam walk (fine):** the flat axis is too flat for hill climbing (gain per step ~ fluctuation),
  so it is scanned over +-`flat_span` in steps of `flat_step`, with both pairs steep-peaked at
  every point (each mount carries one motor of each pair, so flat moves detune the other pair).
  A parabola is fitted; the logic moves to its vertex only if the curvature is significant
  against `metric_noise` and the predicted gain exceeds `compare_tolerance`, and goes back to the
  start point if the vertex does not reproduce.
- **Calibration:** per motor, up/down scans (peak offset = backlash) and up scans after +-300
  steps (return error = lost steps); per pair, the steep width from a +-4 sigma scan and the
  steep direction from the single-motor widths (|e_i| = sigma / sigma_i, signs kept; checked
  against the characterisation: within 2-5 deg and 5 %). Implausible results (width outside
  0.5-2x, direction turned by > 20 deg, failed fits) keep the previous frame.

## Troubleshooting

- **Peak width ~2x the calibration, parking far below the scan maximum, peak running away at the
  scan edge:** one motor of the pair does not move properly. On 2026-10-19 M2 lost 60-85 steps per
  excursion because of a loose jumper contact between the sensor shield (S4-S7, V4, G4 = firmware
  slot 1) and its ULN2003 board; after reseating it returns within ~2-8 steps. Calibrate shows
  this per motor ("returns within").
- **No light in the steep scan:** the beam is far off; jog by hand (manual control) or use
  `Laser_alignment/tools/cold_search.py`.
- **Timeouts / USB drop-outs:** the logic reconnects the Arduino once; positions may be off by the
  last move. A separate 5 V supply for the ULN2003 boards (common ground) should remove the cause.
- Pins 0/1 are the USB serial line; never connect a driver there. `firmware_slots` in the
  hardware config maps motors to other firmware slots if a header is damaged.
