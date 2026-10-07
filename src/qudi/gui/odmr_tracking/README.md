# ODMR Tracking

ODMR scanning plus hardware frequency tracking of one or two resonances on the
Red Pitaya (pyrpl ODMR image).

| Module | File |
|---|---|
| GUI `OdmrTrackingGui` (extends `OdmrGui`) | `gui/odmr_tracking/odmr_tracking_gui.py` |
| Logic `OdmrTrackingLogic` (extends `OdmrLogic`) | `logic/odmr_tracking_logic.py` |
| Interface `OdmrTrackingInterface` | `interface/odmr_tracking_interface.py` |
| Hardware `RedPitayaOdmrLockHardware` | `hardware/redpitaya/redpitaya_odmr_lock.py` |

N = 1 and N = 2 use the same modules and the same FPGA path:

- **N = 1**: the IF sits on resonance 0's zero crossing; resonance 0 stays live
  and the controller corrects it continuously.
- **N = 2**: the LO hops between the resonances (constant IF, two-point jump
  list); each resonance has its own SSB calibration slot, demodulation chain
  and integrator, frozen while the other one is live.

## Workflow

1. Select N in *Tracking Mode*.
2. For each resonance: drag the pale region onto the feature, *Use Region i as
   scan range*, *Start Scan*; drag the darker fit window onto the linear part
   and click *Fit Res i* (zero crossing and discriminator slope).
3. Set f_m, demodulation phase (*← HW* reads the phase set in hardware),
   controller (*Integral* or *PI* with zero ratio), bandwidth and correction
   limit; for N = 2 also dwell and settle time. Click *Configure Tracking*.
4. *Start Stream* / *Start Tracking*, see below.
5. *Save Measurement* saves the ODMR scan and the tracking data.

The error-signal polarity is set from the IF sideband on *Configure Tracking*.
A new ODMR scan stops tracking and the stream first.

## Lock and stream

*Start/Stop Tracking* (controller) and *Start/Stop Stream* (high-rate traces)
are independent:

| Tracking | Stream | Result |
|---|---|---|
| off | off | idle |
| on | off | closed loop (N = 2: hopping runs); low-rate register polling only |
| off | on | open loop: error traces; the cleared correction stays zero |
| on | on | closed loop with high-rate error, correction and pre-FIR traces |

Low-rate history (register polling, whole session) is collected whenever either
is on. In N = 2 high-rate data only one resonance is live at a time; the parked
one is held on the common time axis.

## Saved data

*Save Measurement* writes, under the same name tag as the ODMR scan:

- `ODMR_tracking_history.dat`: low-rate per-resonance error and correction of the
  session (host monotonic clock from the engine start).
- `ODMR_tracking_high_rate.npy` plus `_metadata.txt`: the current rolling
  high-rate window (FPGA time base from the stream start), error, correction
  and pre-FIR (CIC) per resonance, with the filter and slope calibration in
  the metadata.

Both can be saved while tracking runs or after it stopped.

## Other users of the tracker

- `motor_scan_logic`:
  - `CONTINUOUS_FREQ_TRACK` (N = 1): streams the correction while the lock runs,
    via the `tracking_logic` connector.
  - `KDC_HW_SYNC_MULTIRES`: 2D maps of every resonance's error and correction,
    via the `tracking_hw` connector. Needs the engine (*Start Stream* for
    open-loop maps or *Start Tracking*); the scan reads its own recording, so
    *Stop Stream* stays available.
- `sensitivity_sweep_logic`: open/closed-loop comparisons on resonance 0 via
  `tracking_hw`.
- Field logging (`OdmrTrackingLogic.start_field_logging(path)`): one row per
  resonance visit (time, resonance, mean correction, mean error, samples) for
  long single-position runs; reads its own recording while the engine runs
  (*Start Stream* or *Start Tracking*) and closes when the engine stops.

## Configuration

```yaml
gui:
    odmr_tracking_gui:
        module.Class: 'odmr_tracking.odmr_tracking_gui.OdmrTrackingGui'
        connect:
            odmr_logic: 'odmr_tracking_logic'

logic:
    odmr_tracking_logic:
        module.Class: 'odmr_tracking_logic.OdmrTrackingLogic'
        options:
            default_scan_mode: 'EQUIDISTANT_SWEEP'
            num_resonances: 1                 # start value; selectable in the GUI
            modulation_frequency: 15258.789   # f_m [Hz]
            settle_time: 300e-6               # freeze window after each hop [s]
            dwell_time: 2.0e-3                # live time per resonance [s] (N = 2)
            scan_settling_time: 100e-6        # LO settle before the dwell [s]
            trigger_length: 50e-6             # LO-hop trigger pulse [s]
            status_poll_interval: 0.2         # s
        connect:
            microwave: 'mw_source_synthnv'
            data_scanner: 'redpitaya_finite_sampling'
            tracking_hw: 'redpitaya_odmr_lock'

hardware:
    redpitaya_odmr_lock:
        module.Class: 'redpitaya.redpitaya_odmr_lock.RedPitayaOdmrLockHardware'
        options:
            redpitaya_config_name: 'rpy_shared_config'   # shared pyrpl instance
            redpitaya_hostname: '10.203.129.28'
            lock_in_filter_resonance_1: '2kHz_minphase'
            lock_in_filter_resonance_2: '2kHz_minphase'
            tracking_algorithm: 'conventional'           # or 'smith_linear' (N = 1)
```

Lock bandwidth, controller, correction limit, demodulation phase and the fit
regions are kept as status variables between sessions.
