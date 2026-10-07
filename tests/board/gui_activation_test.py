# -*- coding: utf-8 -*-
"""
Activation check of the tracking GUI, motor scan and sensitivity sweep with
their connectors (offscreen; connects to the hardware of the config, so it is
a board test and not collected by pytest).

    python tests/board/gui_activation_test.py --config <qudi cfg> [--dummy-stage]

``--dummy-stage`` runs a copy of the config whose motor-scan logic uses the
dummy XY stage, for when the stages are not connected. Exits with 0 when every
module activates and every connector is connected.
"""
import argparse
import os
import re
import sys
import tempfile

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

MODULES = ('odmr_tracking_gui', 'motor_scan_logic', 'sensitivity_sweep_logic',
           'motor_scan_gui', 'sensitivity_sweep_gui')
CONNECTORS = {
    'motor_scan_logic': ('_tracking_logic', '_tracking_hw', '_streamer'),
    'sensitivity_sweep_logic': ('_tracking_hw', '_streamer'),
}
DUMMY_STAGE = ("    board_test_dummy_stage:\n"
               "        module.Class: 'dummy.motor_scan_dummy.XYStageDummy'\n")


def with_dummy_stage(path):
    """Copy of the config whose motor_hardware connector names the dummy stage."""
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    text, n = re.subn(r'(\n +motor_hardware:) *\S+',
                      lambda m: m.group(1) + ' board_test_dummy_stage', text)
    if n != 1:
        sys.exit('motor_hardware connector not found in the config')
    text, n = re.subn(r'\nhardware:\n', lambda m: m.group(0) + DUMMY_STAGE, text, count=1)
    if n != 1:
        sys.exit('hardware section not found in the config')
    fd, copy = tempfile.mkstemp(suffix='.cfg')
    with os.fdopen(fd, 'w', encoding='utf-8') as fh:
        fh.write(text)
    return copy


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--config', required=True, help='qudi config file')
    p.add_argument('--dummy-stage', action='store_true',
                   help='replace the motor-scan stage by the dummy XY stage')
    args = p.parse_args()
    config = with_dummy_stage(args.config) if args.dummy_stage else args.config

    from PySide2 import QtCore, QtWidgets
    from qudi.core.application import Qudi
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    qudi = Qudi(no_gui=False, config_file=config)
    failures = []

    def drive():
        mm = qudi.module_manager
        for name in MODULES:
            try:
                mm.activate_module(name)
                state = mm[name].state
            except Exception as e:
                state = f'ERROR {e!r}'
            print('RESULT', name, ':', state, flush=True)
            if state not in ('idle', 'locked'):
                failures.append(name)
        for name, attrs in CONNECTORS.items():
            inst = mm[name].instance
            for attr in attrs:
                target = getattr(inst, attr)() if inst is not None else None
                print('RESULT', name, attr, ':', type(target).__name__, flush=True)
                if target is None:
                    failures.append(f'{name}.{attr}')
        gui = mm['odmr_tracking_gui'].instance
        print('RESULT controller :', gui._controller_combo.currentData(),
              gui._zero_ratio_spin.value(), flush=True)
        print('FAIL' if failures else 'PASS', failures, flush=True)
        QtCore.QTimer.singleShot(500, qudi.quit)

    QtCore.QTimer.singleShot(1500, drive)
    qudi.run()
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
