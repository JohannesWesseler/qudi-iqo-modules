# Board tests

Scripts that need the Red Pitaya (and, for tracking, the microwave source and an
NV sample). Their names do not start with `test_`, so pytest does not collect
them. Run them with the qudi environment after changes to the Red Pitaya
hardware modules, the tracking logic or the FPGA image.

| Script | Checks | Moves hardware |
|---|---|---|
| `tracking_board_test.py` | firmware build info; ODMR scan and fits; N=2/N=1 open and closed loop; PI; sequencer hand-off; recordings (next to the display stream, after a display stop, in parallel with field logging); the motor scan's multi-resonance data path (no stage); the sensitivity sweep's acquisition | microwave only |
| `gui_activation_test.py` | tracking GUI, motor scan and sensitivity sweep activate with all connectors | no |

```
python tests/board/tracking_board_test.py --config <cfg> --scan 2.73e9 2.76e9 301 \
    --windows 2.7442e9:2.7475e9 2.7420e9:2.7445e9
python tests/board/gui_activation_test.py --config <cfg>
```

`--windows` brackets each resonance's zero crossing in the lock-in ODMR signal;
adjust them to the sample. The tracking test writes `log.txt` and `report.json`
to `~/qudi/Data/board_tests/` (or `--out`).

Not covered: a 2D motor scan with real KDC position pulses (it moves the stage).
The marker arithmetic is covered offline in `tests/test_odmr_tracking_recordings.py`.
