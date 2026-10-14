# -*- coding: utf-8 -*-
"""
Board test of ODMR tracking on the Red Pitaya (needs the board, the microwave
source and an NV sample; not collected by pytest).

Runs headless qudi with the given config and checks, in order: firmware build
info, an ODMR scan and two zero-crossing fits, the N=2 open loop (both lock-in
chains on the demodulation-phase reference), refusal of a finite sweep while
the LO hops, N=2 and N=1 closed loop, PI control, the independent recordings
(next to the display stream, across a display-stream stop, in parallel with
field logging) and the sensitivity sweep's open-loop acquisition.

    python tests/board/tracking_board_test.py --config <qudi cfg> \\
        --scan 2.73e9 2.76e9 301 --windows 2.7442e9:2.7475e9 2.7420e9:2.7445e9

``--windows`` are the frequency ranges (Hz) around each resonance's zero
crossing in the lock-in ODMR signal; within each, the crossing is the midpoint
of the signal's maximum and minimum. A JSON report and log go to --out.
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--config', required=True, help='qudi config file')
    p.add_argument('--scan', nargs=3, type=float, metavar=('START', 'STOP', 'POINTS'),
                   required=True, help='ODMR scan range (Hz) and number of points')
    p.add_argument('--windows', nargs=2, required=True, metavar='LO:HI',
                   help='zero-crossing search window (Hz) of resonance 0 and 1')
    p.add_argument('--runtime', type=float, default=15.0, help='ODMR scan time (s)')
    p.add_argument('--image-abi-major', type=int, default=None,
                   help="default: the major ABI pyrpl requires (OdmrInstrument)")
    p.add_argument('--out', default=os.path.join(
        os.path.expanduser('~'), 'qudi', 'Data', 'board_tests',
        'tracking_' + time.strftime('%Y%m%d_%H%M%S')))
    args = p.parse_args()
    args.windows = [tuple(float(v) for v in w.split(':')) for w in args.windows]
    return args


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(os.path.join(args.out, 'log.txt'), 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')

    from PySide2 import QtCore
    from qudi.core.application import Qudi
    from qudi.logic.sensitivity_sweep_logic import SensitivitySweepLogic
    from qudi.logic.motor_scan.hw_sync_scan import HwSyncScanMixin
    qapp = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication(sys.argv)
    qudi = Qudi(no_gui=True, config_file=args.config)
    res = {'checks': {}}
    checks = res['checks']

    def wait(seconds, sample=None, out=None):
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            qapp.processEvents()
            if sample is not None:
                value = sample()
                if out is not None:
                    out.append(value)
            time.sleep(0.02)

    def slot_summary(lock):
        return [{k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v)
                 for k, v in st.items() if k in ('slot', 'locked', 'saturated', 'error_lsb',
                                                  'correction_hz')}
                for st in lock.get_all_status()]

    def trace_stats(lock, n):
        tr = lock.read_traces()
        if not tr:
            return None
        out = []
        for i in range(n):
            err = np.asarray(tr['err'][i], dtype=float)
            corr = np.asarray(tr['corr_hz'][i], dtype=float)
            err = err[np.isfinite(err)][-20000:]
            corr = corr[np.isfinite(corr)][-20000:]
            out.append(dict(n=int(err.size), err_mean=float(np.mean(err)) if err.size else None,
                            err_std=float(np.std(err)) if err.size else None,
                            corr_mean_hz=float(np.mean(corr)) if corr.size else None,
                            corr_std_hz=float(np.std(corr)) if corr.size else None))
        return out

    def offset_hz(trace, slope):
        # mean error converted to a frequency offset with the fitted slope (LSB/Hz)
        return abs(trace['err_mean'] / slope)

    def locked(trace, slope):
        # A locked loop holds the resonance within a few hundred Hz while it
        # drifts by tens of kHz (the drift leaves an integral lag, so the mean
        # error is not compared with the noise).
        return offset_hz(trace, slope) < 1e3

    def lossless(stats):
        t, f = stats.get('transport', {}), stats.get('fpga', {})
        return (all(int(t.get(k, 0)) == 0 for k in
                    ('n_gap', 'n_seq_skips', 'n_missing', 'n_dropped', 'n_backlog_dropped'))
                and int(f.get('fpga_overflows', 0)) == 0)

    def record(lock, handle, seconds, during=None):
        """Read a recording for ``seconds``; returns all events."""
        chunks = []
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            qapp.processEvents()
            chunks.append(lock.read_recording(handle)['events'])
            if during is not None:
                during()
            time.sleep(0.05)
        chunks.append(lock.read_recording(handle)['events'])
        return np.concatenate(chunks)

    def drive():
        try:
            mm = qudi.module_manager
            mm.activate_module('odmr_tracking_logic')
            logic = mm['odmr_tracking_logic'].instance
            lock = mm['redpitaya_odmr_lock'].instance
            sampler = mm['redpitaya_finite_sampling'].instance
            odmr = lock._odmr                     # instrument (test-only inspection)
            fw = odmr.check_firmware()
            res['build'] = fw['build']
            log(f"build info {fw['build']}")
            abi_major = (args.image_abi_major if args.image_abi_major is not None
                         else odmr.IMAGE_ABI_REQUIRED >> 16)
            checks['build_info'] = (fw['build'] is not None and
                                    fw['build']['image_abi'] >> 16 == abi_major
                                    and not fw['build']['git_dirty'])
            phase = logic.get_demod_phase()
            logic.set_demod_phase(phase)
            log(f'demod phase {phase} deg; scan power {logic.scan_power} dBm')

            # 1. ODMR scan and two zero-crossing fits; the finite sweep keeps
            #    resonance 0 live in every block.
            logic.set_num_resonances(2)
            logic.set_scan_region(args.scan[0], args.scan[1], int(args.scan[2]))
            logic.set_runtime(args.runtime)
            logic.start_odmr_scan()
            chans = []
            wait(args.runtime + 4, lambda: (odmr.channels.selected_channel, odmr.lock.live_slot,
                                            odmr.synth.live_slot, odmr.sequencer.current_step),
                 chans)
            logic.stop_odmr_scan()
            wait(1)
            steps_seen = {c[3] for c in chans}
            checks['sweep_channel_stays_0'] = (len(steps_seen) > 20 and
                                               all(c[:3] == (0, 0, 0) for c in chans))
            log(f'sweep: {len(chans)} samples, {len(steps_seen)} distinct steps, '
                f'channels/slots seen {sorted({c[:3] for c in chans})}')
            f = np.asarray(logic.frequency_data[0], float)
            y = np.asarray(list(logic.signal_data.values())[0][0], float)

            def window(lo, hi):
                m = (f > lo) & (f < hi)
                c = 0.5 * (f[m][np.argmax(y[m])] + f[m][np.argmin(y[m])])
                return (c - 0.25e6, c + 0.25e6)
            wins = [window(*w) for w in args.windows]
            log(f'fit windows {wins}')
            fits = [logic.fit_resonance(i, *wins[i]) for i in range(2)]
            res['fits'] = [dict(zc_ghz=f['zero_crossing_freq'] / 1e9, r2=f['r_squared'],
                                slope=f['slope']) for f in fits]
            log(f'fits {res["fits"]}')
            checks['fits_good'] = all(f['r_squared'] > 0.9 for f in fits)

            # 2. N=2 open loop: both chains on the demodulation-phase reference
            logic.configure_tracking()
            checks['both_chains_shifted_ref'] = odmr.reference_selection() == [
                ('sin_shifted', 'cos_shifted')] * 2
            logic.start_streaming()
            checks['sequencer_claimed_n2'] = odmr.sequencer_owner == lock.module_name
            wait(4)
            lock.read_traces()
            wait(3)
            res['open_loop'] = trace_stats(lock, 2)
            log(f'open loop: {res["open_loop"]}')

            # 3. a finite sweep is refused while N=2 hops
            try:
                sampler.start_buffered_acquisition()
                checks['sweep_refused_while_hopping'] = False
                sampler.stop_buffered_acquisition()
            except Exception as e:
                checks['sweep_refused_while_hopping'] = 'in use by' in str(e)
                log(f'sweep during hopping refused: {e}')
            checks['hopping_survived'] = bool(odmr.sequencer.busy)

            # 4. N=2 closed loop
            logic.start_tracking()
            slots = []
            wait(8, lambda: (odmr.lock.live_slot, odmr.synth.live_slot, odmr.lock.hopping), slots)
            res['n2_slots_seen'] = sorted({tuple(x) for x in slots})
            log(f'N=2 (lock slot, synth slot, hopping) seen: {res["n2_slots_seen"]}')
            checks['n2_both_slots_hopping'] = ({x[0] for x in slots} == {0, 1} and
                                               {x[1] for x in slots} == {0, 1} and
                                               all(x[2] for x in slots))
            st2 = slot_summary(lock)
            tr2 = trace_stats(lock, 2)
            res['closed_loop_n2'] = dict(status=st2, traces=tr2)
            log(f'N=2 closed loop: {st2}')
            log(f'N=2 traces: {tr2}')
            slopes = [float(fit['slope']) for fit in fits]
            res['n2_offset_hz'] = [offset_hz(t, s) for t, s in zip(tr2, slopes)]
            checks['n2_locked'] = all(locked(t, s) for t, s in zip(tr2, slopes))

            # 5. N=2 recording next to the display stream
            handle = lock.open_recording()
            display_t = []
            calls = [0]

            def display():                    # the GUI polls at ~5 Hz
                calls[0] += 1
                if calls[0] % 4:
                    return
                tr = lock.read_traces()
                if tr is not None and tr['times'].size:
                    display_t.append(float(tr['times'][-1]))
            events = record(lock, handle, 4.0, during=display)
            rec = lock.reconstruct_recording(events)
            n_t = rec['err'].shape[1]
            live = [int(np.count_nonzero(np.isfinite(rec['err'][k]))) for k in range(2)]
            res['recording_n2'] = dict(samples=n_t, live=live,
                                       counts=rec['event_counts'],
                                       span_s=float(rec['times'][-1]) if n_t else 0.0,
                                       display_span_s=(display_t[-1] - display_t[0])
                                       if display_t else 0.0)
            log(f'N=2 recording: {res["recording_n2"]}')
            rate = lock.stream_sample_rate
            # While the LO hops, no FIR events are produced during the sequencer
            # settle, so the FIR count is below the grid length.
            checks['recording_span'] = n_t > 0.95 * 4.0 * rate
            checks['recording_both_slots'] = min(live) > 0.2 * n_t
            checks['display_runs_during_recording'] = res['recording_n2']['display_span_s'] > 3.0

            # 5b. the motor scan's multi-resonance data path, without a stage:
            #     its own recording, drained like during a scan, reconstructed
            #     whole and as a line window.
            scan = SimpleNamespace(
                _get_tracking_hw=lambda: lock, _motor_hardware=lambda: None,
                _scan_data=None, _hw_sync_active=True, _hw_multires_nslots=2,
                _hw_word_chunks=[], _hw_xmarks_flat=np.empty(0, dtype=np.int64),
                _hw_ymarks_flat=np.empty(0, dtype=np.int64),
                _hw_recording=lock.open_recording(), log=SimpleNamespace(
                    warning=lambda *a: log('scan warning: ' + a[0] % a[1:]),
                    debug=lambda *a: None))
            for name in ('_hw_sync_multires_drain', '_hw_sync_multires_traces',
                         '_hw_sync_multires_teardown'):
                setattr(scan, name, getattr(HwSyncScanMixin, name).__get__(scan))
            wait(2.0, scan._hw_sync_multires_drain)
            whole = scan._hw_sync_multires_traces()
            part = scan._hw_sync_multires_traces(w_lo=1000, w_hi=3000)
            scan_recording = scan._hw_recording
            scan._hw_sync_multires_teardown()
            n_whole = whole['err'].shape[1] if whole else 0
            res['motor_scan_path'] = dict(
                chunks=len(scan._hw_word_chunks), samples=n_whole,
                window=part['err'].shape if part else None)
            log(f"motor-scan data path: {res['motor_scan_path']}")
            checks['motor_scan_path'] = bool(
                whole and n_whole > 0.9 * 2.0 * lock.stream_sample_rate and
                part['err'].shape == (2, 2000) and
                np.array_equal(part['err'], whole['err'][:, 1000:3000], equal_nan=True) and
                scan_recording not in lock._recordings and scan._hw_recording is None)

            # 6. stopping the display stream leaves the recording running
            logic.stop_streaming()
            wait(0.5)
            checks['engine_kept_for_lock'] = lock.tracking_engine_active and not lock.trace_stream_active
            after = record(lock, handle, 2.0)
            n_fir = int(np.count_nonzero(after['source'] == odmr.streamer.SOURCE_IDS['fir']))
            log(f'recording after display stop: {n_fir} FIR events in 2 s')
            checks['recording_survives_display_stop'] = n_fir > 0.9 * 2.0 * rate
            stats = lock.close_recording(handle)
            log(f'recording stats: {stats}')
            checks['recording_lossless'] = lossless(stats)
            checks['recording_closed'] = handle not in lock._recordings
            logic.stop_tracking()
            wait(1)
            checks['sequencer_released'] = odmr.sequencer_owner is None

            # 7. finite sweep works again
            data = sampler.acquire_frame(10)
            checks['sweep_after_tracking'] = bool(data) and len(next(iter(data.values()))) == 10

            # 8. N=1 closed loop
            logic.set_num_resonances(1)
            logic.configure_tracking()
            logic.start_streaming()
            logic.start_tracking()
            checks['n1_no_sequencer_claim'] = odmr.sequencer_owner is None
            slots1 = []
            wait(8, lambda: (odmr.lock.live_slot, odmr.synth.live_slot, odmr.lock.hopping), slots1)
            checks['n1_slot0_not_hopping'] = all(tuple(x) == (0, 0, False) for x in slots1)
            st1 = slot_summary(lock)
            tr1 = trace_stats(lock, 1)
            res['closed_loop_n1'] = dict(status=st1, traces=tr1)
            log(f'N=1 closed loop: {st1} {tr1}')
            res['n1_offset_hz'] = offset_hz(tr1[0], slopes[0])
            checks['n1_locked'] = locked(tr1[0], slopes[0])
            # The resonance drifts by up to ~200 kHz within a minute of the
            # fit (warming); the following configurations measure the slope at
            # the configured frequency, so start them where the loop is now.
            res['recentered_hz'] = logic.recenter_on_tracked()
            logic.stop_tracking()
            logic.stop_streaming()
            wait(1)

            # 9. N=1 PI controller at zero ratio 0.2
            logic.set_controller('pi', 0.2)
            logic.configure_tracking()
            logic.start_streaming()
            logic.start_tracking()
            wait(8)
            st_pi = slot_summary(lock)
            tr_pi = trace_stats(lock, 1)
            k = abs(logic._resonance_slopes[0])
            pi_settings = odmr.lock.controller.settings()
            res['closed_loop_n1_pi'] = dict(status=st_pi, traces=tr_pi,
                                            K_Kp=float(pi_settings['kp']) * k)
            log(f'N=1 PI: K*Kp={pi_settings["kp"] * k:.2f}, {st_pi[0]} {tr_pi}')
            # An oscillating loop shows a few hundred times the integral-only
            # error noise, a ringing one (zero ratio 1) about 1.3 times; at 0.2
            # PI is as quiet as integral control.
            checks['n1_pi_stable'] = (bool(pi_settings['proportional']) and
                                      locked(tr_pi[0], slopes[0]) and
                                      tr_pi[0]['err_std'] < 1.2 * tr1[0]['err_std'])
            logic.recenter_on_tracked()
            logic.stop_tracking()
            logic.stop_streaming()
            wait(1)

            # 9b. N=1 PII (second integrator), when the image has it
            if 'pii' in logic.available_controllers:
                logic.set_controller('pii', 0.2)
                logic.configure_tracking()
                logic.start_streaming()
                logic.start_tracking()
                wait(8)
                st_pii = slot_summary(lock)
                tr_pii = trace_stats(lock, 1)
                pii_settings = odmr.lock.controller.settings()
                res['closed_loop_n1_pii'] = dict(status=st_pii, traces=tr_pii,
                                                 settings=pii_settings,
                                                 controller=odmr.lock.controller.name)
                log(f'N=1 PII: {pii_settings}, {st_pii[0]} {tr_pii}')
                checks['n1_pii_stable'] = (odmr.lock.controller.name == 'pii' and
                                           not odmr.lock.overrun and
                                           locked(tr_pii[0], slopes[0]) and
                                           tr_pii[0]['err_std'] < 1.2 * tr1[0]['err_std'])
                logic.stop_tracking()
                logic.stop_streaming()
                wait(1)
            logic.set_controller('integral')

            # 10. N=2 field logging, with another recording open in parallel
            logic.set_num_resonances(2)
            logic.configure_tracking()
            logic.start_streaming()
            logic.start_tracking()
            wait(2)
            other = lock.open_recording()
            other_events = []
            prefix = os.path.join(args.out, 'field_log')

            # Headless, the logic's status timer does not run; drive its poll.
            def poll():
                logic._poll_status()
                other_events.append(lock.read_recording(other)['events'])
            logic.start_field_logging(prefix)
            wait(10, poll)
            logic.stop_field_logging()
            wait(0.5, poll)
            other_stats = lock.close_recording(other)
            logic.stop_tracking()
            logic.stop_streaming()
            checks['field_log_closed'] = not logic.field_logging_active
            rows = np.fromfile(prefix + '.bin', dtype=logic._FIELD_DTYPE)
            slot_rows = [int(np.sum(rows['slot'] == k)) for k in (0, 1)]
            dt = np.diff(rows['t_s'])
            n_other = int(np.count_nonzero(np.concatenate(other_events)['source'] ==
                                           odmr.streamer.SOURCE_IDS['fir']))
            res['field_log'] = dict(rows=int(rows.size), per_slot=slot_rows,
                                    t_span_s=float(rows['t_s'][-1] - rows['t_s'][0])
                                    if rows.size else 0,
                                    n_median=float(np.median(rows['n'])) if rows.size else 0,
                                    dt_min=float(dt.min()) if dt.size else None,
                                    parallel_recording_fir=n_other)
            log(f"field log: {res['field_log']}")
            expected = 10.0 / (2 * float(logic._dwell_time))     # visits per resonance
            checks['field_log_rows'] = bool(min(slot_rows) > 0.5 * expected and
                                            rows.size and dt.min() >= 0)
            checks['parallel_recording'] = n_other > 0.8 * 10.0 * rate and lossless(other_stats)

            # 11. sensitivity-sweep acquisition (open loop, N=1, no engine)
            lock.enable_lock(False)
            lock.enable_channel_selection(False)
            sweep = SimpleNamespace(
                _tracking_hw=lambda: lock,
                _time_series_logic=lambda: SimpleNamespace(module_state=lambda: 'idle'),
                _stream_parameters={'n_time_traces': 1, 'trace_duration': 6.0},
                _sleep_interruptible=lambda s, check_interval=0.5: (wait(s) or True),
                log=SimpleNamespace(warning=log, info=log))
            time_s, err, corr, ev, diag = \
                SensitivitySweepLogic._acquire_timestamped_tracking_events(
                    sweep, closed_loop=False, lock_hw=lock)
            finite = int(np.count_nonzero(np.isfinite(err)))
            res['sweep_acquisition'] = dict(samples=int(err.size), finite=finite,
                                            fir_events=diag['fir_event_count'],
                                            transport=diag['transport'])
            log(f"sweep acquisition: {res['sweep_acquisition']}")
            # From an idle streamer the first events can arrive up to ~3 s late
            # (stream start-up and epoch resolution); the samples are complete
            # from the start, the tail after the deadline is not read.
            checks['sweep_acquisition'] = (finite == err.size and
                                           err.size > 2.5 * rate and
                                           lossless({'transport': diag['transport'],
                                                     'fpga': diag['fpga']}))

            res['pass'] = all(bool(v) for v in checks.values())
            log(json.dumps({k: bool(v) for k, v in checks.items()}, indent=1))
            log('PASS' if res['pass'] else 'FAIL')
        except Exception:
            import traceback
            log(traceback.format_exc())
            res['pass'] = False
        with open(os.path.join(args.out, 'report.json'), 'w') as fh:
            json.dump(res, fh, indent=2, default=str)
        QtCore.QTimer.singleShot(500, qudi.quit)

    QtCore.QTimer.singleShot(1500, drive)
    qudi.run()


if __name__ == '__main__':
    main()
