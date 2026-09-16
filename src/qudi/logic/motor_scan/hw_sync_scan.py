# -*- coding: utf-8 -*-
"""
Hardware-synchronized motor scanning (ScanMode.KDC_HW_SYNC).

The motor moves continuously along each fast-axis line while the KDC101
controllers emit encoder-referenced position-step trigger pulses (one per fast-axis
*bin boundary* on the fast axis; one per line on the slow axis). These pulses are
captured by the Red Pitaya FPGA *in the same clock domain as the demod data
stream*, so each spatial bin is defined by a hardware edge rather than a software
timestamp. The PC slices the continuous demod trace at the recorded marker sample
indices, giving exact data<->position allocation with no line shifts.

Threading model
---------------
This is **not** a blocking loop. ``HwSyncScanMixin`` plugs into the existing
non-blocking line state machine in :class:`ContinuousLineScanMixin`: the proven
movement / retry / timeout logic drives the stage line-by-line, and at the two
hook points (line-start and line-end) we swap the *data path* from software
position sampling to the FPGA hardware-marker stream. Each completed line is
reconstructed immediately and ``sigScanDataUpdated`` is emitted, so the GUI paints
the map line-by-line and stop/pause stay responsive.

This module contains:
  * ``reconstruct_hw_sync_line`` - pure: turn one line's demod slice + x-markers
    into that line's grid-row values (used by the live path, unit-tested).
  * ``reconstruct_hw_sync_scan`` - pure: turn a whole-scan demod trace + x/y marker
    arrays into the full map (used for offline re-processing of saved raw data,
    unit-tested).
  * ``HwSyncScanMixin`` - non-blocking orchestration helpers: acquire the pyrpl
    scan module, configure the KDC triggers, drain the marker stream, and
    reconstruct each line as the stage finishes it.

See pyrpl ``docs/developer_guide/motor_position_sync_scan.md`` for the full design
and the FPGA/register details.
"""

from typing import Optional, Dict, Any, Callable, List

import numpy as np

from .data_structures import ScanMode


# ---------------------------------------------------------------------------
# Multi-resonance (KDC_HW_SYNC_MULTIRES) channel convention
# ---------------------------------------------------------------------------
# Per resonance we bin BOTH the demod error and the frequency correction, so the
# 2D maps are named res{k}_err / res{k}_corr. The GUI derives difference/sum of
# the corrections from these at display time.
MULTIRES_QUANTITIES = ('err', 'corr')


def multires_channel_names(nslots: int) -> List[str]:
    """Canonical channel names for an N-resonance mapped scan (res0_err, res0_corr,
    res1_err, ...). Order is (resonance-major, quantity-minor)."""
    names = []
    for k in range(int(nslots)):
        for q in MULTIRES_QUANTITIES:
            names.append(f'res{k}_{q}')
    return names


def _multires_trace_for_channel(traces: Dict[str, Any], channel: str) -> Optional[np.ndarray]:
    """Return the 1-D fresh-only trace for a res{k}_{err|corr} channel, or None."""
    try:
        res_str, quantity = channel.split('_', 1)
        k = int(res_str[3:])  # strip 'res'
    except (ValueError, IndexError):
        return None
    key = 'err' if quantity == 'err' else 'corr_hz'
    arr = traces.get(key)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim != 2 or not (0 <= k < arr.shape[0]):
        return None
    return arr[k]


# ---------------------------------------------------------------------------
# Pure reconstruction (hardware-free, unit-tested)
# ---------------------------------------------------------------------------
def reconstruct_hw_sync_line(scan_data,
                             demod: np.ndarray,
                             line_x_markers: np.ndarray,
                             line_index: int,
                             channel: str = 'demod',
                             reducer: Callable[[np.ndarray], float] = np.nanmean,
                             store_raw: bool = True,
                             logger=None) -> Dict[str, Any]:
    """Allocate one line's demod samples to its grid row using that line's x-markers.

    ``line_x_markers`` are the absolute demod-sample indices of the bin boundaries
    captured *during this line's sweep only* (the caller tracks a per-line
    baseline, so markers from other lines / inter-line moves are excluded). For
    ``ppl`` points per line there should be ``ppl + 1`` boundary markers giving
    ``ppl`` bins; bin ``k`` is ``demod[xm[k] : xm[k+1]]`` and maps to the grid cell
    of the ``k``-th point traversed on this line (snake reversal handled by
    ``scan_data.point_index_to_grid_index``).

    Fills ``scan_data.stream_data_mean[channel]`` for this line's grid cells and,
    if ``store_raw``, the corresponding ``scan_data.stream_data_raw[channel]``
    entries. Returns a small diagnostics dict.
    """
    demod = np.asarray(demod)
    xm = np.asarray(line_x_markers, dtype=np.int64)
    n_total = demod.shape[0]
    ppl = scan_data.get_points_per_line()

    if scan_data.stream_data_mean is None or channel not in scan_data.stream_data_mean:
        scan_data.initialize_data_arrays([channel])
    mean_map = scan_data.stream_data_mean[channel]
    raw_store = scan_data.stream_data_raw.get(channel) if (store_raw and scan_data.stream_data_raw) else None

    n_bins = max(0, xm.size - 1)
    diag = {'line': line_index, 'n_x_markers': int(xm.size),
            'n_bins': int(n_bins), 'expected_bins': int(ppl), 'warning': False}
    if n_bins != ppl:
        diag['warning'] = True
        if logger is not None:
            logger.warning(
                "HW-sync line %d: got %d bins from %d x-markers, expected %d. "
                "Mapping the overlap; check KDC trigger Count / pulse capture / "
                "fast-axis over-travel margin.",
                line_index, n_bins, int(xm.size), ppl)

    for k in range(min(ppl, n_bins)):
        a, b = int(xm[k]), int(xm[k + 1])
        a = max(0, a)
        b = min(n_total, b)
        if b <= a:
            continue
        seg = demod[a:b]
        flat = line_index * ppl + k
        grid_idx = scan_data.point_index_to_grid_index(flat)
        if seg.size == 0 or np.all(np.isnan(seg)):
            value = np.nan
        else:
            value = reducer(seg)
        mean_map[grid_idx] = value
        if raw_store is not None and flat < len(raw_store):
            raw_store[flat] = seg.tolist()

    return diag


def reconstruct_hw_sync_scan(scan_data,
                             demod: np.ndarray,
                             x_markers: np.ndarray,
                             y_markers: Optional[np.ndarray] = None,
                             channel: str = 'demod',
                             reducer: Callable[[np.ndarray], float] = np.nanmean,
                             store_raw: bool = True,
                             logger=None) -> Dict[str, Any]:
    """Allocate a whole continuous demod trace to grid points using hardware markers.

    Used for **offline re-processing** of saved raw data (the live scan path uses
    ``reconstruct_hw_sync_line`` per line). ``y_markers`` split the stream into
    lines; within a line the ``x_markers`` split it into bins.

    Robust to whether a leading (line-0) y-marker is present: the slow-axis KDC
    pulse fires when the stage *crosses* a row, so line 0 (which starts *at* its
    row) usually has no pulse and there are ``n_lines - 1`` y-markers. A synthetic
    edge at index 0 always begins line 0, and a redundant near-zero marker (e.g.
    from synthesized test data) is dropped so both conventions reconstruct
    identically.
    """
    demod = np.asarray(demod)
    x_markers = np.asarray(x_markers, dtype=np.int64)
    n_total = demod.shape[0]

    ppl = scan_data.get_points_per_line()
    n_lines = scan_data.get_num_lines()

    if scan_data.stream_data_mean is None or channel not in scan_data.stream_data_mean:
        scan_data.initialize_data_arrays([channel])

    mean_map = scan_data.stream_data_mean[channel]
    raw_store = scan_data.stream_data_raw.get(channel) if (store_raw and scan_data.stream_data_raw) else None

    # --- line sample ranges from the y (slow-axis) markers ---------------------
    ym = np.asarray(y_markers, dtype=np.int64) if y_markers is not None else np.array([], dtype=np.int64)
    if n_lines <= 1 or ym.size == 0:
        line_bounds = [(0, n_total)]
    else:
        seps = sorted(int(v) for v in ym)
        # De-duplicate spurious near-coincident y-pulses: real line separators are
        # ~one line-length apart, so drop any that are closer than a fraction of the
        # mean line length to the previous (otherwise a spurious extra y-marker
        # creates a near-zero-length 'line' and steals a real one).
        if n_total > 0:
            min_gap = 0.3 * (n_total / max(1, n_lines))
            dd = []
            for s in seps:
                if not dd or (s - dd[-1]) >= min_gap:
                    dd.append(s)
            seps = dd
        if n_lines > 1 and len(seps) >= n_lines:
            # A line-0 marker (or extras) is present: the last n_lines-1 markers are
            # the internal line separators.
            internal = seps[-(n_lines - 1):]
        else:
            # n_lines-1 (no line-0 pulse) or fewer (missed pulses -> bin-count warns)
            internal = seps
        edges = [0] + internal + [n_total]
        line_bounds = []
        for r in range(n_lines):
            lo = edges[r] if r < len(edges) else n_total
            hi = edges[r + 1] if (r + 1) < len(edges) else n_total
            line_bounds.append((lo, hi))

    diag = {'n_lines': n_lines, 'points_per_line': ppl, 'lines': [], 'n_warnings': 0}

    for r, (lo, hi) in enumerate(line_bounds):
        sel = x_markers[(x_markers >= lo) & (x_markers < hi)]
        # Robust to leading spurious x-markers (e.g. first-line extras): the real
        # bin boundaries are the trailing ppl+1, identical to the live per-line path.
        if sel.size > ppl + 1:
            sel = sel[-(ppl + 1):]
        n_bins = max(0, sel.size - 1)
        diag['lines'].append({'line': r, 'range': (lo, hi),
                              'n_x_markers': int(sel.size), 'n_bins': int(n_bins)})
        if n_bins != ppl:
            diag['n_warnings'] += 1
            if logger is not None:
                logger.warning(
                    "HW-sync line %d: got %d bins from %d x-markers, expected %d. "
                    "Mapping the overlap; check KDC trigger Count / pulse capture.",
                    r, n_bins, int(sel.size), ppl)

        for k in range(min(ppl, n_bins)):
            a, b = int(sel[k]), int(sel[k + 1])
            a = max(0, a)
            b = min(n_total, b)
            if b <= a:
                continue
            seg = demod[a:b]
            flat = r * ppl + k
            grid_idx = scan_data.point_index_to_grid_index(flat)
            if seg.size == 0 or np.all(np.isnan(seg)):
                value = np.nan
            else:
                value = reducer(seg)
            mean_map[grid_idx] = value
            if raw_store is not None and flat < len(raw_store):
                raw_store[flat] = seg.tolist()

    return diag


def reconstruct_hw_sync_multires_line(scan_data,
                                      traces: Dict[str, Any],
                                      line_x_markers: np.ndarray,
                                      line_index: int,
                                      nslots: int,
                                      reducer: Callable[[np.ndarray], float] = np.nanmean,
                                      store_raw: bool = True,
                                      logger=None) -> Dict[str, Any]:
    """Bin one line for ALL res{k}_{err,corr} channels from reconstructed traces.

    ``traces`` is the fresh-only per-resonance reconstruction
    (``{'err': (N,T), 'corr_hz': (N,T)}``) whose sample axis (0..T-1) is the same
    triplet axis the (triplet-index) x-markers address. Reuses the single-channel
    binner per channel so the marker/bin/snake logic stays in one place; the
    warning is logged once (channel 0) to avoid 2N-fold spam.
    """
    channels = multires_channel_names(nslots)
    diag = {'line': line_index, 'channels': channels}
    for i, ch in enumerate(channels):
        trace = _multires_trace_for_channel(traces, ch)
        if trace is None:
            continue
        d = reconstruct_hw_sync_line(
            scan_data, trace, line_x_markers, line_index, channel=ch,
            reducer=reducer, store_raw=store_raw,
            logger=logger if i == 0 else None)
        if i == 0:
            diag.update({k: d[k] for k in ('n_x_markers', 'n_bins',
                                           'expected_bins', 'warning')})
    return diag


def reconstruct_hw_sync_multires_scan(scan_data,
                                      traces: Dict[str, Any],
                                      x_markers: np.ndarray,
                                      y_markers: Optional[np.ndarray],
                                      nslots: int,
                                      reducer: Callable[[np.ndarray], float] = np.nanmean,
                                      store_raw: bool = True,
                                      logger=None) -> Dict[str, Any]:
    """Whole-scan re-bin of ALL res{k}_{err,corr} channels (offline / finalize)."""
    channels = multires_channel_names(nslots)
    diag = {'channels': channels}
    for i, ch in enumerate(channels):
        trace = _multires_trace_for_channel(traces, ch)
        if trace is None:
            continue
        d = reconstruct_hw_sync_scan(
            scan_data, trace, x_markers, y_markers=y_markers, channel=ch,
            reducer=reducer, store_raw=store_raw,
            logger=logger if i == 0 else None)
        if i == 0:
            diag.update({k: d.get(k) for k in ('n_lines', 'points_per_line',
                                               'lines', 'n_warnings')})
    return diag


# ---------------------------------------------------------------------------
# Non-blocking orchestration mixin
# ---------------------------------------------------------------------------
class HwSyncScanMixin:
    """Non-blocking helpers for KDC marker scanning, driven by the existing line
    state machine in :class:`ContinuousLineScanMixin`.

    Lifecycle (called from the line state machine / scan_logic):
      * ``_hw_sync_scan_setup()``   - once at scan start: acquire scan module,
        configure the slow-axis (line) trigger, start the FPGA marker stream,
        reset the drain buffers.
      * ``_hw_sync_line_endpoints(line)`` - fast-axis move targets for a line,
        over-travelling half a bin past each grid endpoint so the stage crosses
        every bin boundary (first/last boundary pulses included).
      * ``_hw_sync_line_start(line)`` - at the start of each line's sweep:
        (re)configure the fast-axis trigger for this line's direction and reset
        the per-line x-marker baseline.
      * ``_hw_sync_drain()`` - pull new demod samples + markers from the FPGA
        (called frequently while sweeping so the rings never overflow).
      * ``_hw_sync_line_finish(line)`` - at line end: final drain + reconstruct
        this line's grid row from its x-markers.
      * ``_hw_sync_scan_teardown()`` - at scan end / abort: stop the stream,
        disable the KDC triggers.
      * ``_release_scan_module()`` - on module deactivate.

    Requires the parent (``MotorScanLogic``) to provide ``_motor_hardware()``,
    ``_scan_data``, ``log``, and the ConfigOptions ``redpitaya_hostname`` /
    ``redpitaya_config_name`` (matching the other Red Pitaya modules so the pyrpl
    instance is shared) plus optional ``hw_sync_*`` tuning options.
    """

    def _init_hw_sync_state(self):
        """Initialize HW-sync buffers/handles. Call from ``__init__``."""
        self._scan_module = None
        self._pyrpl = None
        self._scan_via_streamer = False     # True if scan module came from redpitaya_stream
        # Multi-resonance (KDC_HW_SYNC_MULTIRES) state: the tracker owns the stream,
        # we drain raw triplet WORDS (reconstructed on the PC into 2N fresh-only
        # traces) instead of a single demod array.
        self._hw_word_chunks = []           # list[np.ndarray] - raw triplet words
        self._hw_multires_nslots = 0        # resonance count for the active mapped scan
        self._hw_multires_words_per_sample = 3  # legacy default; queried at setup
        self._hw_demod_chunks = []          # list[np.ndarray] - continuous demod
        self._hw_xmarks_flat = np.empty(0, dtype=np.int64)  # absolute x-marker indices
        self._hw_ymarks_flat = np.empty(0, dtype=np.int64)  # absolute y-marker indices (diagnostic)
        self._hw_line_xmark_start = 0       # index into _hw_xmarks_flat at current line start
        self._hw_sync_active = False
        self._hw_line_restart_on_resume = False
        # USB cross-check of the slow-axis (y) position: the measured encoder position
        # of each line and the commanded row, so the actual y of every line's data is
        # recorded (the FPGA marker separates lines but does NOT measure position).
        self._hw_line_y_actual = []         # measured slow-axis position per line [m]
        self._hw_line_y_target = []         # commanded slow-axis row per line [m]
        self._hw_empty_line_streak = 0      # consecutive 0-marker (empty) lines
        self._hw_empty_line_total = 0       # total empty lines this scan

    # ---- marker-ring role mapping (fast-axis = bins, slow-axis = lines) --------
    # The FPGA's two marker rings are SYMMETRIC (scan_new.v): the x ring (ram_lsb,
    # fed by DIO5/x_pos_trig_i) and the y ring (ram_msb, DIO6/y_pos_trig_i) each
    # record the demod sample index on their stage's trigger pulse, with identical
    # edge-detect/holdoff/depth and a shared sample counter. Which ring holds the
    # BIN boundaries vs the LINE boundaries depends ONLY on which stage is the fast
    # axis: x for *_X patterns, y for *_Y. So bins come from the fast stage's ring
    # and lines from the slow stage's ring -- supporting LINE_BY_LINE_Y needs no
    # FPGA change, just this role mapping. (_hw_xmarks_flat / _hw_ymarks_flat always
    # hold the physical x-ring / y-ring contents drained from the board.)
    def _hw_bin_marks(self) -> np.ndarray:
        """Absolute marker indices for the FAST axis (= bin boundaries)."""
        return (self._hw_xmarks_flat if self._scan_data.get_fast_axis() == 'x'
                else self._hw_ymarks_flat)

    def _hw_line_marks(self) -> np.ndarray:
        """Absolute marker indices for the SLOW axis (= line boundaries)."""
        return (self._hw_ymarks_flat if self._scan_data.get_fast_axis() == 'x'
                else self._hw_xmarks_flat)

    # ---- config accessors (ConfigOptions on the parent, with safe defaults) ----
    def _hw_sync_channel_name(self) -> str:
        ch = getattr(self, '_hw_sync_channel', 'demod')
        return ch if ch in ('demod', 'ftw_corr') else 'demod'

    def _hw_sync_pulse_width_s(self) -> float:
        return float(getattr(self, '_hw_sync_pulse_width', 1e-4))

    def _hw_sync_trig_port_num(self) -> int:
        return int(getattr(self, '_hw_sync_trig_port', 1))

    def _hw_sync_margin_fraction(self) -> float:
        """Fraction of a bin to over-travel past each grid endpoint (default 0.5)."""
        return float(getattr(self, '_hw_sync_margin_frac', 0.5))

    def _hw_sync_runup_m(self) -> float:
        """Absolute fast-axis run-up distance (metres) before the first (throwaway)
        trigger pulse, so the stage is already at constant velocity when it crosses
        it. Default 0.5 mm — much larger than the ~0.1 mm acceleration distance at
        typical scan speeds, and (unlike a bin-relative margin) still adequate for
        sub-micron bins. Verified on hardware: the run-up amount does NOT prevent
        the first-pulse drop (that is what the throwaway is for); it only needs to
        be enough to be moving steadily across the throwaway + first boundary."""
        return float(getattr(self, '_hw_sync_runup', 0.5e-3))

    # ---- pyrpl scan-module acquisition (shared-instance, refcounted) -----------
    def _get_streamer(self):
        """Return the connected Red Pitaya stream owner (``redpitaya_stream``) or
        None. When present it owns the board's single push stream and arbitrates
        the KDC_HW_SYNC takeover; when absent we drive the pyrpl scan module
        directly (headless tests)."""
        conn = getattr(self, '_streamer', None)
        if conn is None:
            return None
        try:
            return conn()
        except Exception:
            return None

    def _get_scan_module(self):
        """Return the pyrpl ``Scan`` module. Prefers the shared ``redpitaya_stream``
        owner's scan module (so the board's single stream is coordinated, not
        duplicated); falls back to acquiring the shared pyrpl instance directly.
        Caches on ``self._scan_module``."""
        if getattr(self, '_scan_module', None) is not None:
            return self._scan_module
        streamer = self._get_streamer()
        if streamer is not None:
            try:
                sm = streamer.get_scan_module()
            except Exception as e:
                sm = None
                self.log.warning("redpitaya_stream.get_scan_module() failed: %s", e)
            if sm is not None:
                self._scan_module = sm
                self._scan_via_streamer = True
                return sm
        host = getattr(self, '_redpitaya_hostname', None)
        cfg = getattr(self, '_redpitaya_config_name', None)
        if not host:
            self.log.error("KDC_HW_SYNC needs a 'streamer' connector OR the "
                           "'redpitaya_hostname'/'redpitaya_config_name' ConfigOptions.")
            return None
        try:
            from qudi.hardware.redpitaya.resource_manager import get_pyrpl_instance
            self._pyrpl, _ = get_pyrpl_instance(hostname=host, config_name=cfg)
            self._scan_module = self._pyrpl.rp.scan
            self._scan_via_streamer = False
            return self._scan_module
        except Exception as e:
            self.log.error("Failed to acquire pyrpl scan module: %s", e)
            return None

    def _release_scan_module(self):
        """Release our reference to the shared pyrpl instance (call on deactivate).
        No-op for the streamer path -- the ``redpitaya_stream`` module owns that
        instance and releases it itself."""
        if getattr(self, '_scan_module', None) is None:
            return
        if getattr(self, '_scan_via_streamer', False):
            self._scan_module = None
            return
        host = getattr(self, '_redpitaya_hostname', None)
        cfg = getattr(self, '_redpitaya_config_name', None)
        try:
            from qudi.hardware.redpitaya.resource_manager import release_pyrpl_instance
            release_pyrpl_instance(hostname=host, config_name=cfg)
        except Exception as e:
            self.log.warning("Failed to release pyrpl instance: %s", e)
        self._scan_module = None
        self._pyrpl = None

    # ---- multi-resonance (KDC_HW_SYNC_MULTIRES) data path -----------------------
    # The multi-resonance tracker (redpitaya_odmr_lock, MultiResonanceTrackingInterface)
    # owns the continuous LO-hop loop + self-describing MARKED stream. A mapped 2D scan
    # keeps ALL the shared geometry/motion/trigger logic below and only swaps the data
    # source: instead of the scan module's single demod stream we (a) ask the tracker
    # to add x/y markers (which restarts + resets its stream for clean alignment), and
    # (b) drain raw triplet WORDS + markers through the tracker, reconstructing 2N
    # fresh-only per-resonance traces on the PC and binning each into its own map.
    def _hw_sync_is_multires(self) -> bool:
        sd = getattr(self, '_scan_data', None)
        return sd is not None and sd.scan_mode == ScanMode.KDC_HW_SYNC_MULTIRES

    def _get_multi_track_hw(self):
        """The connected MultiResonanceTrackingInterface hardware, or None."""
        conn = getattr(self, '_multi_track_hw', None)
        if conn is None:
            return None
        try:
            return conn()
        except Exception:
            return None

    def _hw_sync_multires_setup(self) -> bool:
        """Setup for KDC_HW_SYNC_MULTIRES: verify the trace stream is live, add x/y
        markers, configure the slow-axis trigger, and warm up.

        The FPGA lock is intentionally not required: with it enabled the correction
        maps are closed-loop data; with it disabled the same path provides open-loop
        demod-error maps (and normally zero correction maps).
        """
        sd = self._scan_data
        motor = self._motor_hardware()
        hw = self._get_multi_track_hw()
        if sd is None or motor is None or hw is None:
            self.log.error("KDC_HW_SYNC_MULTIRES setup failed: missing scan_data / "
                           "motor / multi_track_hw connector. Connect the multi-"
                           "resonance tracking hardware and start its stream first.")
            return False
        if not getattr(hw, 'nslots', 0):
            self.log.error("KDC_HW_SYNC_MULTIRES: multi_track_hw reports 0 slots.")
            return False
        if not bool(getattr(hw, 'trace_stream_active', False)):
            self.log.error(
                "KDC_HW_SYNC_MULTIRES needs the multi-resonance trace stream. In the "
                "Multi-Resonance ODMR Tracking GUI, configure tracking and click "
                "Start Stream. Start Tracking is optional (lock-on and open-loop "
                "motor scans are both supported).")
            return False
        self._hw_multires_nslots = int(hw.nslots)
        self._hw_multires_words_per_sample = int(
            getattr(hw, 'stream_words_per_sample', 3))

        # reset drain buffers
        self._hw_word_chunks = []
        self._hw_xmarks_flat = np.empty(0, dtype=np.int64)
        self._hw_ymarks_flat = np.empty(0, dtype=np.int64)
        self._hw_line_xmark_start = 0
        self._hw_line_restart_on_resume = False
        self._hw_line_y_actual = []
        self._hw_line_y_target = []
        self._hw_sync_clamp_warned = False
        self._hw_empty_line_streak = 0
        self._hw_empty_line_total = 0

        # Add x/y position markers to the tracker's running MARKED stream. This
        # restarts + resets the stream (word 0 aligns with marker 0) but leaves the
        # hop loop + per-slot lock running. Do this before configuring motor outputs,
        # so a stream race/failure cannot leave a KDC trigger armed after setup aborts.
        try:
            hw.enable_position_markers(True)
        except Exception as e:
            self.log.error("Failed to enable multi-resonance position markers: %s", e)
            return False
        if not bool(getattr(hw, 'position_markers_active', True)):
            self.log.error("Multi-resonance hardware did not enter position-marker mode.")
            return False

        # slow-axis 'In Motion' line-boundary trigger (identical to single-res)
        n_lines = sd.get_num_lines()
        if n_lines > 1:
            slow = sd.get_slow_axis()
            try:
                motor.setup_motion_trigger(
                    slow, trig_port=self._hw_sync_trig_port_num(), polarity='high')
                self.log.info("KDC_HW_SYNC_MULTIRES slow-axis (%s) trigger: 'In Motion'.",
                              slow)
            except Exception as e:
                self.log.warning("Could not configure slow-axis 'In Motion' trigger: %s", e)

        self._hw_sync_active = True
        self.log.info("KDC_HW_SYNC_MULTIRES marker stream started (N=%d, channels=%s).",
                      self._hw_multires_nslots,
                      multires_channel_names(self._hw_multires_nslots))
        self._hw_sync_warmup_sweep()
        return True

    def _hw_sync_multires_drain(self):
        """Drain new triplet words + x/y markers from the tracker."""
        hw = self._get_multi_track_hw()
        if hw is None:
            return
        try:
            w = hw.read_stream_words()
            if w is not None and len(w):
                # Region-11 uses structured timestamped events; the legacy scan
                # stream uses float words. Preserve whichever representation the
                # hardware adapter returned.
                self._hw_word_chunks.append(np.asarray(w))
            xm, ym = hw.read_position_markers()
            if xm is not None and len(xm):
                self._hw_xmarks_flat = np.concatenate(
                    [self._hw_xmarks_flat, np.asarray(xm, dtype=np.int64)])
            if ym is not None and len(ym):
                self._hw_ymarks_flat = np.concatenate(
                    [self._hw_ymarks_flat, np.asarray(ym, dtype=np.int64)])
        except Exception as e:
            self.log.warning("KDC_HW_SYNC_MULTIRES drain error: %s", e)

    def _hw_sync_multires_traces(self, w_lo: Optional[int] = None,
                                 w_hi: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Reconstruct accumulated words into fresh-only per-resonance traces.

        If ``w_lo``/``w_hi`` (sample indices) are given, only that record window
        is decoded (cheap, for the live per-line map); the
        returned traces are then indexed from 0 at ``w_lo``. Marked-series is
        fresh-only and per-sample self-labelled, so slicing the words is exact.
        """
        hw = self._get_multi_track_hw()
        if hw is None or not self._hw_word_chunks:
            return None
        words = np.concatenate(self._hw_word_chunks)
        if words.dtype.names is not None:
            try:
                traces = hw.reconstruct_mapped_traces(words)
            except Exception as e:
                self.log.warning("KDC_HW_SYNC_MULTIRES reconstruct failed: %s", e)
                return None
            if w_lo is None:
                return traces
            lo = max(0, int(w_lo))
            hi = None if w_hi is None else max(lo, int(w_hi))
            sliced = dict(traces)
            for key in ('err', 'corr_hz', 'cic'):
                if key in sliced:
                    sliced[key] = np.asarray(sliced[key])[:, lo:hi]
            for key in ('times', 'dead'):
                if key in sliced:
                    sliced[key] = np.asarray(sliced[key])[lo:hi]
            return sliced
        if w_lo is not None:
            width = int(self._hw_multires_words_per_sample)
            a = max(0, width * int(w_lo))
            b = words.size if w_hi is None else min(words.size, width * int(w_hi))
            if b <= a:
                return None
            words = words[a:b]
        try:
            return hw.reconstruct_mapped_traces(words)
        except Exception as e:
            self.log.warning("KDC_HW_SYNC_MULTIRES reconstruct failed: %s", e)
            return None

    def _hw_sync_multires_line_finish(self, line_index: int) -> Dict[str, Any]:
        """Final drain + reconstruct + bin this line for all 2N channels."""
        self._hw_sync_multires_drain()
        line_xm = self._hw_bin_marks()[self._hw_line_xmark_start:]

        # hardware y-anchor clamp (identical policy to single-res)
        line_marks = self._hw_line_marks()
        y_anchor = int(line_marks[-1]) if line_marks.size else None
        if y_anchor is not None and line_xm.size and int(line_xm[0]) < y_anchor:
            n_leak = int(np.sum(line_xm < y_anchor))
            self.log.warning(
                "KDC_HW_SYNC_MULTIRES line %d: %d x-marker(s) precede the hardware "
                "y-row crossing (sample %d); clamping.", line_index, n_leak, y_anchor)
            line_xm = line_xm[line_xm >= y_anchor]

        ppl = self._scan_data.get_points_per_line()
        if line_xm.size > ppl + 1:
            line_xm = line_xm[-(ppl + 1):]

        # Reconstruct ONLY this line's word window (cheap), then bin with markers
        # rebased to the window origin. Full-scan re-bin happens once at finalize.
        traces = None
        if line_xm.size >= 2:
            w_lo = int(line_xm[0])
            traces = self._hw_sync_multires_traces(w_lo=w_lo, w_hi=int(line_xm[-1]) + 1)
            line_xm = line_xm - w_lo
        if traces is None:
            self.log.warning("KDC_HW_SYNC_MULTIRES line %d: no traces yet; skipping.",
                             line_index)
            self._hw_empty_line_streak = getattr(self, '_hw_empty_line_streak', 0) + 1
            self._hw_empty_line_total = getattr(self, '_hw_empty_line_total', 0) + 1
            return {'line': line_index, 'n_bins': 0}
        diag = reconstruct_hw_sync_multires_line(
            self._scan_data, traces, line_xm, line_index, self._hw_multires_nslots,
            store_raw=bool(getattr(self, '_save_full_traces', False)), logger=self.log)
        diag['y_anchor'] = y_anchor
        self.log.info("KDC_HW_SYNC_MULTIRES line %d: %d x-markers -> %d/%d bins "
                      "(y-anchor=%s)%s.", line_index, diag.get('n_x_markers', 0),
                      diag.get('n_bins', 0), diag.get('expected_bins', ppl), y_anchor,
                      ' (WARN)' if diag.get('warning') else '')
        fast = self._scan_data.get_fast_axis()
        dio = 'DIO5_P' if fast == 'x' else 'DIO6_P'
        if diag.get('n_bins', 0) == 0:
            self._hw_empty_line_streak = getattr(self, '_hw_empty_line_streak', 0) + 1
            self._hw_empty_line_total = getattr(self, '_hw_empty_line_total', 0) + 1
            self.log.error("KDC_HW_SYNC_MULTIRES line %d EMPTY: 0 fast-axis (%s) pulses "
                           "-> no data (check %s TRIG path). %d in a row.",
                           line_index, fast, dio, self._hw_empty_line_streak)
        else:
            self._hw_empty_line_streak = 0
        return diag

    def _hw_sync_multires_finalize(self) -> Dict[str, Any]:
        """Whole-scan hardware re-bin of all 2N channels (both axes hw-anchored)."""
        sd = self._scan_data
        if sd is None:
            return {}
        traces = self._hw_sync_multires_traces()
        xm = np.asarray(self._hw_bin_marks(), dtype=np.int64)
        ym = np.asarray(self._hw_line_marks(), dtype=np.int64)
        n_lines = sd.get_num_lines()
        # stash faithful hardware dataset for offline re-binning / saving
        try:
            words = (np.concatenate(self._hw_word_chunks)
                     if self._hw_word_chunks else np.empty(0, dtype=np.float64))
            sd.hw_stream_words = words
            sd.hw_x_markers = xm
            sd.hw_y_markers = ym
            sd.hw_multires_nslots = self._hw_multires_nslots
            sd.hw_y_positions_actual = np.asarray(self._hw_line_y_actual, dtype=float)
            sd.hw_y_positions_target = np.asarray(self._hw_line_y_target, dtype=float)
        except Exception as e:
            self.log.warning("Could not stash multires hw-sync raw trace: %s", e)

        if traces is None:
            self.log.warning("KDC_HW_SYNC_MULTIRES finalize: no reconstructed traces; "
                             "keeping the live per-line map.")
            return {'applied': False}

        # Snapshot the N-specific reconstruction at scan finalization. Saving may
        # happen later, after the tracking GUI has been reconfigured to a different
        # slot count; the saved time/error/correction traces must still describe this
        # scan rather than the hardware's later configuration.
        if bool(getattr(self, '_save_full_traces', False)):
            try:
                sd.hw_multires_times = np.asarray(traces['times'], dtype=np.float64)
                sd.hw_multires_err = np.asarray(traces['err'], dtype=np.float64)
                sd.hw_multires_corr_hz = np.asarray(traces['corr_hz'], dtype=np.float64)
                sd.hw_multires_sample_rate = float(traces['sample_rate'])
            except Exception as e:
                self.log.warning("Could not snapshot multi-resonance trace/time data: %s", e)

        # y-marker spread sanity (identical policy to single-res)
        need_y = max(0, n_lines - 1)
        n_total = int(np.asarray(traces.get('err')).shape[1]) if traces.get('err') is not None else 0
        ym_sorted = np.sort(ym)
        y_spread_ok = (n_lines <= 1) or (
            ym_sorted.size >= need_y and n_total > 0 and
            int(ym_sorted[-1]) >= 0.5 * n_total)
        if not y_spread_ok:
            self.log.warning("KDC_HW_SYNC_MULTIRES: y-markers unusable as line "
                             "delimiters (%d for %d lines); keeping the live map.",
                             int(ym.size), n_lines)
            return {'y_markers': int(ym.size), 'y_hardware_complete': False,
                    'applied': False}

        snap = None
        if sd.stream_data_mean is not None:
            snap = {k: np.array(v, copy=True) for k, v in sd.stream_data_mean.items()}
        diag = reconstruct_hw_sync_multires_scan(
            sd, traces, xm, y_markers=ym, nslots=self._hw_multires_nslots,
            store_raw=bool(getattr(self, '_save_full_traces', False)), logger=self.log)
        empty_lines = sum(1 for l in diag.get('lines', []) if l['n_bins'] == 0)
        if empty_lines > 0 and snap is not None:
            for k, v in snap.items():
                sd.stream_data_mean[k] = v
            self.log.warning("KDC_HW_SYNC_MULTIRES: hardware re-bin produced %d empty "
                             "line(s); reverted to the live map.", empty_lines)
            return {'applied': False, 'empty_lines': empty_lines}
        diag['applied'] = True
        diag['y_hardware_complete'] = True
        self.log.info("KDC_HW_SYNC_MULTIRES finalize: re-binned %d channels over "
                      "%d samples / %d x-markers / %d y-markers.",
                      len(diag.get('channels', [])), n_total, int(xm.size), int(ym.size))
        return diag

    def _hw_sync_multires_teardown(self):
        """Stop x/y markers and disable KDC triggers, leaving stream/lock as found."""
        if not getattr(self, '_hw_sync_active', False):
            return
        self._hw_sync_active = False
        hw = self._get_multi_track_hw()
        if hw is not None:
            try:
                hw.enable_position_markers(False)
            except Exception as e:
                self.log.warning("Error disabling multires position markers: %s", e)
        motor = self._motor_hardware()
        if motor is not None:
            sd = self._scan_data
            axes = set()
            if sd is not None:
                axes.add(sd.get_fast_axis())
                axes.add(sd.get_slow_axis())
            for ax in axes:
                try:
                    motor.disable_position_trigger(ax, trig_port=self._hw_sync_trig_port_num())
                except Exception:
                    pass
        self.log.debug("KDC_HW_SYNC_MULTIRES teardown complete (stream/lock left running).")

    # ---- geometry --------------------------------------------------------------
    def _hw_sync_fast_axis_limits(self):
        """(pos_min, pos_max) travel limits [m] for the fast axis, or None."""
        try:
            motor = self._motor_hardware()
            fast = self._scan_data.get_fast_axis()
            c = motor.get_constraints()[fast]
            return float(c['pos_min']), float(c['pos_max'])
        except Exception:
            return None

    def _hw_sync_line_endpoints(self, line_index: int):
        """Return (start_pos_dict, end_pos_dict) for a line's fast-axis sweep.

        The fast-axis endpoints are pushed half a bin (``margin_fraction``) *past*
        the outermost bin boundaries so the stage is already moving when it crosses
        the first boundary and still moving past the last one -> all ``ppl + 1``
        boundary pulses fire. The slow-axis coordinate is the line's grid row.
        """
        sd = self._scan_data
        fast = sd.get_fast_axis()
        boundaries = sd.get_bin_boundaries(line_index)  # ppl+1, in travel order
        start_grid, end_grid = sd.get_line_start_end_positions(line_index)
        start = dict(start_grid)
        end = dict(end_grid)
        if boundaries.size >= 2:
            step = abs(float(boundaries[1] - boundaries[0]))
            forward = boundaries[-1] >= boundaries[0]
            sgn = 1.0 if forward else -1.0
            runup = self._hw_sync_runup_m()
            # The fast-axis trigger uses a leading THROWAWAY pulse one interval
            # before the first real boundary (see _hw_sync_line_start), so the
            # stage must already be moving one full interval *plus* the run-up
            # before boundaries[0]. Over-travel the run-up past the last boundary
            # too, so it is crossed cleanly while still moving.
            start[fast] = float(boundaries[0]) - sgn * (step + runup)
            end[fast] = float(boundaries[-1]) + sgn * runup
        else:
            start[fast] = float(boundaries[0]) if boundaries.size else start.get(fast, 0.0)
            end[fast] = float(boundaries[-1]) if boundaries.size else end.get(fast, 0.0)
        # Clamp the over-travel endpoints to the fast-axis travel range. A move
        # outside [pos_min, pos_max] is REJECTED by the KDC firmware (not clamped),
        # so an un-clamped over-travel target (e.g. -2.45 mm for a scan starting at
        # x=0) silently fails and the stage never reaches the line start -> the scan
        # stalls. Clamping is correctness-safe: markers are position-based, so the
        # boundaries the stage does cross while moving still bin correctly; only bins
        # right at the travel limit may capture fewer samples (no room to run up past
        # the hard stop).
        lim = self._hw_sync_fast_axis_limits()
        if lim is not None:
            pos_min, pos_max = lim
            raw_start, raw_end = start[fast], end[fast]
            start[fast] = min(max(start[fast], pos_min), pos_max)
            end[fast] = min(max(end[fast], pos_min), pos_max)
            if ((abs(start[fast] - raw_start) > 1e-9 or abs(end[fast] - raw_end) > 1e-9)
                    and not getattr(self, '_hw_sync_clamp_warned', False)):
                self._hw_sync_clamp_warned = True
                margin = self._hw_sync_runup_m()
                if boundaries.size >= 2:
                    margin += abs(float(boundaries[1] - boundaries[0]))
                self.log.warning(
                    "KDC_HW_SYNC: fast-axis sweep over-travels outside the stage "
                    "limits [%.3f, %.3f] mm; clamped (start %.3f->%.3f, end "
                    "%.3f->%.3f mm). Bins near the limit may capture fewer samples; "
                    "for full run-up start the fast axis >= ~%.1f mm inside the limit.",
                    pos_min * 1e3, pos_max * 1e3, raw_start * 1e3, start[fast] * 1e3,
                    raw_end * 1e3, end[fast] * 1e3, margin * 1e3)
        return start, end

    # ---- lifecycle hooks -------------------------------------------------------
    def _hw_sync_scan_setup(self) -> bool:
        """Acquire the scan module, configure the slow-axis trigger, and start the
        FPGA marker stream. Returns True on success."""
        if self._hw_sync_is_multires():
            return self._hw_sync_multires_setup()
        sd = self._scan_data
        motor = self._motor_hardware()
        scan = self._get_scan_module()
        if sd is None or motor is None or scan is None:
            self.log.error("KDC_HW_SYNC setup failed: missing scan_data / motor / scan module.")
            return False

        # reset drain buffers
        self._hw_demod_chunks = []
        self._hw_xmarks_flat = np.empty(0, dtype=np.int64)
        self._hw_ymarks_flat = np.empty(0, dtype=np.int64)
        self._hw_line_xmark_start = 0
        self._hw_line_restart_on_resume = False
        self._hw_line_y_actual = []
        self._hw_line_y_target = []
        self._hw_sync_clamp_warned = False
        self._hw_empty_line_streak = 0
        self._hw_empty_line_total = 0

        # Configure the slow-axis (line) trigger ONCE as an 'In Motion' output: the
        # KDC drives its TRIG port active for the whole duration of each move, so the
        # FPGA records exactly ONE y-marker (rising edge) at the instant the stage
        # STARTS each inter-line move = one hardware line boundary in the demod stream.
        #
        # Why not 'At Position Steps' (the old scheme): that trigger fires only when
        # the encoder *matches* a one-at-a-time scheduled position while in motion.
        # The slow axis decelerates to a STOP at each row, so the match lands right at
        # the servo settle point and the KDC firmware intermittently SKIPS it under
        # concurrent USB/poll load -- the scheduled position stays armed and fires ~1
        # row late on the next departure, producing the doubled-pair + missing-row
        # y-markers seen on hardware (varying run-to-run with load). 'In Motion' has
        # nothing to match, so it cannot be skipped: validated clean even while
        # get_position() was hammered (the exact load that broke position-step). The
        # fast axis keeps position-step (it sweeps THROUGH all its bin boundaries
        # mid-motion and over-provides markers, so it is reliable).
        #
        # One marker fires per y move: the initial move to row 0 plus each inter-line
        # transition (n_lines markers total). reconstruct_hw_sync_scan robustly keeps
        # the trailing n_lines-1 as the internal line separators (the leading
        # before-line-0 marker is dropped, exactly like the old throwaway handling).
        n_lines = sd.get_num_lines()
        if n_lines > 1:
            slow = sd.get_slow_axis()
            try:
                motor.setup_motion_trigger(
                    slow, trig_port=self._hw_sync_trig_port_num(), polarity='high')
                self.log.info("KDC_HW_SYNC slow-axis (%s) trigger: 'In Motion' "
                              "(one robust line-boundary marker per inter-line move).",
                              slow)
            except Exception as e:
                self.log.warning("Could not configure slow-axis 'In Motion' trigger "
                                 "(line cross-check disabled): %s", e)

        # Start a dedicated Region-11 subscription. The scan module no longer owns
        # continuous transport, so a configured RedPitayaDataInStream is required.
        channel = self._hw_sync_channel_name()
        streamer = self._get_streamer() if getattr(self, '_scan_via_streamer', False) else None
        if streamer is None:
            self.log.error(
                'KDC_HW_SYNC requires the configured Red Pitaya data-stream module; '
                'scan-local FPGA streaming has been removed.')
            return False
        try:
            self._scan_module = streamer.begin_scan_stream(input_source=channel)
        except Exception as e:
            self.log.error("Failed to start FPGA marker stream: %s", e)
            return False

        self._hw_sync_active = True
        self.log.info("KDC_HW_SYNC Region-11 subscription started (channel=%s).", channel)

        self._hw_sync_warmup_sweep()
        return True

    def _hw_sync_warmup_sweep(self):
        """Cold-start fix for the FAST axis. On a fresh scan the very first line's
        sweep loses most of its x-markers (e.g. 4/40 bins) while every later line is
        perfect: the KDC position-step trigger comparator cannot keep up with the
        dense crossings until the system has completed one full sweep (verified on
        hardware -- line 0 short, lines 1+ full; worse at higher bin density / longer
        sweeps / higher speed). Two things are cold on line 0: the stream/USB drain
        cadence is still settling, and the fast axis approaches its start from the
        HOME side instead of the reverse+backlash approach every later line makes.

        We therefore run one DISCARDED warm-up sweep here -- move the fast axis to
        line 0's (over-travelled) start, arm its trigger, and sweep to the end --
        replicating exactly the conditions of a normal line so the trigger engine,
        stream pipeline and drain cadence are all warm before line 0 is recorded.
        The warm-up's demod + x-markers land in the pre-line-0 region of the stream
        and are excluded from line 0 by the per-line baseline (live path) and the
        trailing-trim (finalize); no y-marker is produced (the slow axis stays put).
        Blocking is fine here -- this runs in setup, before the non-blocking line
        state machine starts."""
        sd = self._scan_data
        motor = self._motor_hardware()
        if motor is None:
            return
        try:
            fast = sd.get_fast_axis()
            start0, end0 = self._hw_sync_line_endpoints(0)
            if fast not in start0 or fast not in end0:
                return
            # move to the (over-travelled) line-0 start
            motor.move_abs({fast: float(start0[fast])})
            if hasattr(motor, 'wait_for_idle'):
                motor.wait_for_idle(timeout=60.0)
            # arm the same trigger line 0 will use, then sweep to the end (discarded)
            self._hw_sync_config_fast_trigger(0)
            motor.move_abs({fast: float(end0[fast])})
            if hasattr(motor, 'wait_for_idle'):
                motor.wait_for_idle(timeout=120.0)
            self._hw_sync_drain()   # pull the warm-up samples/markers into the buffers
            self.log.info("KDC_HW_SYNC fast-axis (%s) warm-up sweep done "
                          "(%.3f -> %.3f mm); line 0 starts warm.",
                          fast, float(start0[fast]) * 1e3, float(end0[fast]) * 1e3)
        except Exception as e:
            self.log.warning("KDC_HW_SYNC fast-axis warm-up sweep failed "
                             "(line 0 may be short): %s", e)

    def _hw_sync_config_fast_trigger(self, line_index: int):
        """Configure the fast-axis position-step trigger for one line (the bin
        boundaries), with the leading THROWAWAY that absorbs the KDC first-pulse
        drop. Shared by the live line start and the cold-start warm-up sweep."""
        sd = self._scan_data
        motor = self._motor_hardware()
        fast = sd.get_fast_axis()
        boundaries = sd.get_bin_boundaries(line_index)  # ppl+1, in travel order
        if motor is None or boundaries.size < 2:
            return
        forward = boundaries[-1] >= boundaries[0]
        direction = 'fwd' if forward else 'rev'
        x_step = abs(float(boundaries[1] - boundaries[0]))
        sgn = 1.0 if forward else -1.0
        # The KDC reliably drops the FIRST scheduled position-step pulse (verified on
        # hardware: with StartPos = boundaries[0] that boundary is never emitted;
        # every later absolute position IS). Add a leading THROWAWAY pulse one
        # interval before the first real boundary so the dropped pulse is the
        # throwaway and all ppl+1 real boundaries are captured. The throwaway is
        # removed in _hw_sync_line_finish (keep the trailing ppl+1 markers).
        x_start = float(boundaries[0]) - sgn * x_step  # throwaway position
        x_num = int(boundaries.size) + 1               # ppl+1 real + 1 throwaway
        try:
            motor.setup_position_trigger(
                fast, start=x_start, step=x_step, num=x_num,
                pulse_width=self._hw_sync_pulse_width_s(),
                direction=direction, trig_port=self._hw_sync_trig_port_num())
        except Exception as e:
            self.log.error("Failed to configure fast-axis trigger for line %d: %s",
                           line_index, e)

    def _hw_sync_line_start(self, line_index: int):
        """(Re)configure the fast-axis trigger for this line and reset the per-line
        x-marker baseline. Called at the line-start -> sweep transition."""
        self._hw_sync_config_fast_trigger(line_index)
        self._hw_sync_record_line_y(line_index)

        # Discard markers accumulated before the sweep (inter-line move etc.) so
        # this line only owns the fast-axis (bin) markers captured from here on.
        self._hw_sync_drain()
        self._hw_line_xmark_start = int(self._hw_bin_marks().size)

    def _hw_sync_record_line_y(self, line_index: int):
        """USB cross-check of the slow-axis position for this line. Called at line
        start -- y is settled at its row and x has NOT begun sweeping yet, so a single
        encoder read is safe (it cannot starve the fast-axis trigger, which only
        happens when a *moving* trigger-armed axis is polled continuously). Records
        the measured y and the commanded row so the actual position of every line's
        data is stored; warns if the stage settled off-target (surfaces any
        accel/decel/settle offset instead of hiding it)."""
        sd = self._scan_data
        motor = self._motor_hardware()
        if sd is None or motor is None:
            return
        slow = sd.get_slow_axis()
        try:
            start_pos, _ = sd.get_line_start_end_positions(line_index)
            target = float(start_pos.get(slow)) if start_pos and slow in start_pos else float('nan')
        except Exception:
            target = float('nan')
        try:
            pos = motor.get_pos([slow])
            actual = float(pos.get(slow, float('nan')))
        except Exception as e:
            self.log.warning("KDC_HW_SYNC: could not read slow-axis position for "
                             "line %d: %s", line_index, e)
            actual = float('nan')
        self._hw_line_y_target.append(target)
        self._hw_line_y_actual.append(actual)
        if np.isfinite(actual) and np.isfinite(target):
            dev = actual - target
            lvl = self.log.warning if abs(dev) > 50e-6 else self.log.debug
            lvl("KDC_HW_SYNC line %d slow-axis (%s): target=%.4f mm actual=%.4f mm "
                "(dev=%+.1f um)%s", line_index, slow, target * 1e3, actual * 1e3,
                dev * 1e6, "  <-- OFF TARGET" if abs(dev) > 50e-6 else "")

    def _hw_sync_drain(self):
        """Pull new demod samples + markers from the FPGA into the buffers."""
        if self._hw_sync_is_multires():
            return self._hw_sync_multires_drain()
        scan = self._scan_module
        if scan is None:
            return
        try:
            d = scan.mapped_stream_read()
            if d is not None and len(d):
                d = np.asarray(d, dtype=np.float64)
                if self._hw_sync_channel_name() == 'ftw_corr':
                    # Convert raw FTW (LSB) -> Hz so the binned map is in physical
                    # frequency units, matching the lock-controls 'Correction [Hz]'
                    # display (ftw_to_hz is a linear scale, so NaN gap-fills pass
                    # through). Without this the map shows raw FTW counts (~34.36
                    # per Hz), not Hz.
                    d = np.asarray(scan.ftw_to_hz(d), dtype=np.float64)
                self._hw_demod_chunks.append(d)
            xm = scan.read_x_markers()
            if xm is not None and len(xm):
                self._hw_xmarks_flat = np.concatenate(
                    [self._hw_xmarks_flat, np.asarray(xm, dtype=np.int64)])
            ym = scan.read_y_markers()
            if ym is not None and len(ym):
                self._hw_ymarks_flat = np.concatenate(
                    [self._hw_ymarks_flat, np.asarray(ym, dtype=np.int64)])
                if getattr(self, '_hw_sync_ydiag', False):
                    self._hw_sync_ymarker_diag(ym)
        except Exception as e:
            self.log.warning("KDC_HW_SYNC drain error: %s", e)

    def _hw_sync_ymarker_diag(self, new_ym):
        """DIAGNOSTIC (enabled by the _hw_sync_ydiag flag): the instant a new
        y-marker is captured, log its sample index together with the scan phase and
        the live motor position, so a doubled/missing y-pulse can be pinned to what
        the stages were doing (y settling vs x reset) when it fired. y-markers are
        rare (~one per line) so the extra get_pos() over USB is negligible."""
        try:
            import time as _t
            phase = ('move_to_start' if getattr(self, '_waiting_for_line_start', False)
                     else ('sweep' if getattr(self, '_waiting_for_line_end', False)
                           else 'other'))
            motor = self._motor_hardware()
            pos = motor.get_pos() if motor is not None else {}
            moving = motor.is_moving() if (motor is not None and hasattr(motor, 'is_moving')) else None
            line = getattr(self, '_current_line_index', '?')
            self.log.warning(
                "YDIAG: new y-marker(s)=%s  line=%s phase=%s moving=%s  pos=%s  t=%.3f",
                [int(v) for v in new_ym], line, phase, moving,
                {k: round(v * 1e3, 4) for k, v in pos.items()}, _t.time())
        except Exception as e:
            self.log.warning("YDIAG failed: %s", e)

    def _hw_sync_line_finish(self, line_index: int) -> Dict[str, Any]:
        """Final drain + reconstruct this line's grid row from its x-markers.

        Hardware enforcement of the slow-axis (line) boundary: the FPGA records, at
        the encoder crossing into this row, the demod sample index into the y-marker
        ring. Any x-marker preceding that index cannot belong to this line, so it is
        clamped out here. The line boundary is therefore anchored to the hardware
        y-pulse, not to the software move sequencing -- the inter-line/return move
        cannot leak a bin into the wrong row even if the USB idle detection is late.
        """
        if self._hw_sync_is_multires():
            return self._hw_sync_multires_line_finish(line_index)
        self._hw_sync_drain()
        demod = np.concatenate(self._hw_demod_chunks) if self._hw_demod_chunks else np.empty(0)
        line_xm = self._hw_bin_marks()[self._hw_line_xmark_start:]

        line_marks = self._hw_line_marks()
        y_anchor = int(line_marks[-1]) if line_marks.size else None
        if y_anchor is not None and line_xm.size and int(line_xm[0]) < y_anchor:
            n_leak = int(np.sum(line_xm < y_anchor))
            self.log.warning(
                "KDC_HW_SYNC line %d: %d x-marker(s) precede the hardware y-row "
                "crossing (demod sample %d); clamping to the hardware line boundary.",
                line_index, n_leak, y_anchor)
            line_xm = line_xm[line_xm >= y_anchor]

        # Keep exactly the ppl+1 real boundary markers. The fast-axis trigger emits
        # a leading throwaway pulse (see _hw_sync_line_start) to work around the KDC
        # first-pulse drop; if the KDC happened to emit it (rather than drop it),
        # the throwaway is the first captured marker. Taking the trailing ppl+1
        # removes it (and any stray pre-sweep marker) robustly, whether or not it
        # was dropped. If fewer than ppl+1 were captured, leave them for the
        # reconstruction's bin-count warning.
        ppl = self._scan_data.get_points_per_line()
        if line_xm.size > ppl + 1:
            line_xm = line_xm[-(ppl + 1):]

        diag = reconstruct_hw_sync_line(
            self._scan_data, demod, line_xm, line_index,
            channel=self._hw_sync_channel_name(),
            store_raw=bool(getattr(self, '_save_full_traces', False)), logger=self.log)
        diag['y_anchor'] = y_anchor
        self.log.info("KDC_HW_SYNC line %d: %d x-markers -> %d/%d bins (y-anchor=%s)%s.",
                      line_index, diag['n_x_markers'], diag['n_bins'],
                      diag['expected_bins'], y_anchor,
                      ' (WARN)' if diag['warning'] else '')

        # LOUD alarm on an EMPTY line: 0 fast-axis trigger pulses captured -> the
        # whole line has no data. This is the hardware fast-axis trigger DROPOUT
        # symptom (the stage sweeps normally but no pulses reach the FPGA), distinct
        # from a mildly short line. Track the streak so a dead trigger path is
        # obvious immediately instead of only at scan end.
        fast = self._scan_data.get_fast_axis()
        dio = 'DIO5_P' if fast == 'x' else 'DIO6_P'
        if diag['n_bins'] == 0:
            self._hw_empty_line_streak = getattr(self, '_hw_empty_line_streak', 0) + 1
            self._hw_empty_line_total = getattr(self, '_hw_empty_line_total', 0) + 1
            self.log.error(
                "KDC_HW_SYNC line %d EMPTY: 0 fast-axis (%s) trigger pulses captured "
                "-> 0/%d bins, the ENTIRE line has NO data. The %s-stage position "
                "trigger (-> %s) delivered nothing this line (the stage IS sweeping). "
                "Empty lines: %d in a row, %d total this scan.",
                line_index, fast, diag['expected_bins'], fast, dio,
                self._hw_empty_line_streak, self._hw_empty_line_total)
            if self._hw_empty_line_streak >= 2:
                self.log.error(
                    "KDC_HW_SYNC: %d CONSECUTIVE EMPTY LINES -- the fast-axis (%s) "
                    "trigger path looks DEAD. STOP the scan and check the %s-stage "
                    "TRIG cable / %s connection / level-shifter; no trigger pulses "
                    "are reaching the FPGA.",
                    self._hw_empty_line_streak, fast, fast, dio)
        else:
            self._hw_empty_line_streak = 0
        return diag

    def _hw_sync_finalize(self) -> Dict[str, Any]:
        """At scan completion, re-bin the WHOLE demod trace from the hardware
        markers — y-markers split it into lines, x-markers split each line into
        bins — so BOTH axes are hardware-anchored, replacing the live
        software-sequenced map. Also stashes the raw demod trace + marker index
        arrays on ``scan_data`` (``hw_demod_trace`` / ``hw_x_markers`` /
        ``hw_y_markers``) as the faithful, position-exact dataset for saving and
        offline re-binning.

        Logs explicitly whether every line boundary came from a hardware y-pulse
        (full both-axis guarantee) or whether any had to be inferred.
        """
        if self._hw_sync_is_multires():
            return self._hw_sync_multires_finalize()
        sd = self._scan_data
        if sd is None:
            return {}
        demod = (np.concatenate(self._hw_demod_chunks)
                 if self._hw_demod_chunks else np.empty(0, dtype=np.float64))
        # Role-mapped markers: bins from the fast-axis ring, lines from the slow-axis
        # ring (identical to x/y for *_X patterns; swapped for *_Y). Stored as
        # hw_x_markers/hw_y_markers (= bin/line) so offline reconstruct_hw_sync_scan,
        # which takes x_markers=bins + y_markers=lines, reproduces the binning directly.
        xm = np.asarray(self._hw_bin_marks(), dtype=np.int64)
        ym = np.asarray(self._hw_line_marks(), dtype=np.int64)
        n_lines = sd.get_num_lines()
        # Stash the faithful hardware-defined dataset (stream + marker indices) so
        # the exact binning can always be reproduced offline (reconstruct_hw_sync_scan).
        try:
            sd.hw_demod_trace = demod
            sd.hw_x_markers = xm
            sd.hw_y_markers = ym
            # USB cross-check: measured vs commanded slow-axis position per line
            ya = np.asarray(self._hw_line_y_actual, dtype=float)
            yt = np.asarray(self._hw_line_y_target, dtype=float)
            sd.hw_y_positions_actual = ya
            sd.hw_y_positions_target = yt
            m = np.isfinite(ya) & np.isfinite(yt)
            if m.any():
                dev = (ya[m] - yt[m]) * 1e6  # um
                self.log.info(
                    "KDC_HW_SYNC slow-axis (%s) USB cross-check over %d lines: "
                    "deviation from target mean=%+.1f um, max|dev|=%.1f um, "
                    "std=%.1f um (reproducibility). Actual slow-axis positions stored.",
                    sd.get_slow_axis(), int(m.sum()), float(dev.mean()),
                    float(np.abs(dev).max()), float(dev.std()))
        except Exception as e:
            self.log.warning("Could not stash hw-sync raw trace on scan_data: %s", e)

        # Sanity-check the y-markers as line delimiters BEFORE trusting them: they
        # must number >= n_lines-1 and be spread across the stream (the last
        # internal separator should sit in the latter part of the trace). If they
        # are missing or clustered (the observed failure: y-pulses bunched at low
        # sample indices, so all data falls into the last line), the hardware line
        # split is invalid -> KEEP the proven live software-sequenced map instead
        # of overwriting it with garbage.
        need_y = max(0, n_lines - 1)
        n_total = int(demod.size)
        ym_sorted = np.sort(ym)
        # DIAGNOSTIC: see whether marker indices align with the accumulated demod.
        self.log.warning(
            "KDC_HW_SYNC finalize DIAG: demod_len=%d, x-markers n=%d range=[%s..%s], "
            "y-markers n=%d idx=%s",
            n_total, int(xm.size),
            (int(xm.min()) if xm.size else None), (int(xm.max()) if xm.size else None),
            int(ym.size), [int(v) for v in ym_sorted[:8]])
        y_spread_ok = (n_lines <= 1) or (
            ym_sorted.size >= need_y and n_total > 0 and
            int(ym_sorted[-1]) >= 0.5 * n_total)   # last separator in 2nd half
        if not y_spread_ok:
            self.log.warning(
                "KDC_HW_SYNC: y-markers unusable as line delimiters (%d markers for "
                "%d lines; last at %s of %d samples). KEEPING the live "
                "software-sequenced map; the slow-axis (y) boundary is NOT yet "
                "hardware-guaranteed. Investigate y-axis trigger/marker capture.",
                int(ym.size), n_lines,
                (int(ym_sorted[-1]) if ym_sorted.size else None), n_total)
            return {'y_markers': int(ym.size), 'y_hardware_complete': False,
                    'applied': False}

        # Snapshot the live map so we can revert if the hardware re-bin turns out
        # to leave whole lines empty (another sign the y-split was wrong).
        snap = None
        if sd.stream_data_mean is not None:
            snap = {k: np.array(v, copy=True) for k, v in sd.stream_data_mean.items()}
        diag = reconstruct_hw_sync_scan(
            sd, demod, xm, y_markers=ym,
            channel=self._hw_sync_channel_name(),
            store_raw=bool(getattr(self, '_save_full_traces', False)), logger=self.log)
        empty_lines = sum(1 for l in diag.get('lines', []) if l['n_bins'] == 0)
        if empty_lines > 0 and snap is not None:
            for k, v in snap.items():
                sd.stream_data_mean[k] = v
            self.log.warning(
                "KDC_HW_SYNC: hardware re-bin produced %d empty line(s); reverted to "
                "the live map. Slow-axis (y) boundary NOT hardware-guaranteed this "
                "scan -- investigate y-markers.", empty_lines)
            return {'y_markers': int(ym.size), 'y_hardware_complete': False,
                    'applied': False, 'empty_lines': empty_lines}

        diag['y_markers'] = int(ym.size)
        diag['y_hardware_complete'] = True
        diag['applied'] = True
        n_bins = sum(l['n_bins'] for l in diag.get('lines', []))
        full = diag.get('n_warnings', 0) == 0
        self.log.info(
            "KDC_HW_SYNC finalize: hardware re-bin of %d demod samples / %d x-markers "
            "/ %d y-markers -> %d lines, %d bins. BOTH-AXIS HARDWARE GUARANTEE: %s.",
            int(demod.size), int(xm.size), int(ym.size), n_lines, n_bins,
            "YES" if full else "PARTIAL (see per-line warnings)")
        return diag

    def _hw_sync_scan_teardown(self):
        """Stop the FPGA stream and disable the KDC triggers (idempotent)."""
        if self._hw_sync_is_multires():
            return self._hw_sync_multires_teardown()
        if not getattr(self, '_hw_sync_active', False):
            return
        self._hw_sync_active = False
        scan = getattr(self, '_scan_module', None)
        streamer = self._get_streamer() if getattr(self, '_scan_via_streamer', False) else None
        try:
            if streamer is not None:
                streamer.end_scan_stream()
        except Exception as e:
            self.log.warning("Error stopping FPGA marker stream: %s", e)
        motor = self._motor_hardware()
        if motor is not None:
            sd = self._scan_data
            axes = set()
            if sd is not None:
                axes.add(sd.get_fast_axis())
                axes.add(sd.get_slow_axis())
            for ax in axes:
                try:
                    motor.disable_position_trigger(ax, trig_port=self._hw_sync_trig_port_num())
                except Exception:
                    pass

        # Sanity: the slow-axis KDC should emit ~one hardware pulse per line
        # transition. Far fewer means the slow-axis trigger isn't wired/configured
        # and the hardware line-boundary guard was inoperative this scan.
        sd = self._scan_data
        if sd is not None and sd.get_num_lines() > 1:
            expected = sd.get_num_lines() - 1
            slow = sd.get_slow_axis()
            dio = 'DIO6_P' if slow == 'y' else 'DIO5_P'
            got = int(self._hw_line_marks().size)
            if got < expected:
                self.log.warning(
                    "KDC_HW_SYNC: captured %d slow-axis (%s) hardware pulses, expected "
                    ">= %d (one per line transition). The hardware line-boundary guard "
                    "was not fully active -- check the %s-axis KDC TRIG wiring (%s) "
                    "and the slow-axis motion trigger.", got, slow, expected, slow, dio)
        self.log.debug("KDC_HW_SYNC teardown complete.")
