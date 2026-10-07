# -*- coding: utf-8 -*-
"""
Data saving mixin for MotorScanLogic.

Handles saving scan data to files and generating figures.
"""

import os
import datetime
import numpy as np
from typing import Dict

import matplotlib.pyplot as plt
from PySide2 import QtCore

from qudi.util.datastorage import TextDataStorage
from qudi.util.units import ScaledFloat

from .data_structures import ScanMode


class DataSavingMixin:
    """
    Mixin class providing data saving functionality for MotorScanLogic.
    
    This mixin provides:
    - Saving scan data to files (multiple formats based on scan mode)
    - Drawing figures/thumbnails for visual preview
    - Saving raw ODMR data per pixel
    - Saving position data with statistics
    
    The parent class must provide:
    - _scan_data: MotorScanData instance
    - _scan_state: Current ScanState
    - _thread_lock: RecursiveMutex for thread safety
    - _save_thumbnails: ConfigOption bool
    - _current_scan_folder: str path
    - _tracking_zero_crossing: float
    - _tracking_zero_crossing_history: list
    - module_default_data_dir: str property
    - module_state: Module state accessor
    - sigSaveStateChanged: Signal(bool)
    - log: Logger instance
    """

    @QtCore.Slot(str)
    def save_scan_data(self, tag: str = None):
        """
        Save current scan data to file.
        
        Creates a dedicated folder for each XY scan containing all data files
        and subfolders (e.g., raw ODMR data per pixel).
        
        Folder structure:
            <module_data_dir>/YYYYMMDD-HHMM-SS_tag_motor_scan_MODE/
                center_frequency.dat
                center_frequency.pdf
                linewidth.dat
                ...
                odmr_raw_per_pixel/
                    pixel_x000_y000_odmr.dat
                    ...
        
        Args:
            tag: Optional tag to include in folder name.
        """
        from .data_structures import ScanState
        
        with self._thread_lock:
            if self._scan_data is None:
                self.log.warning("No scan data to save.")
                return

            # Block save only if scan is actively running or stopping (not paused)
            # Allow saving when:
            # - Scan state is IDLE (no scan in progress)
            # - Scan state is PAUSED (data is stable, safe to save)
            if self._scan_state == ScanState.RUNNING:
                self.log.error('Unable to save scan data. Scan actively running. '
                               'Pause the scan first to save.')
                return
            if self._scan_state == ScanState.STOPPING:
                self.log.warning('Unable to save scan data. Scan is stopping, please wait.')
                return

            # Track if we're already locked (e.g., from a paused scan)
            # to avoid double-lock/unlock issues
            already_locked = (self.module_state() == 'locked')

            self.sigSaveStateChanged.emit(True)
            if not already_locked:
                self.module_state.lock()

            try:
                timestamp = datetime.datetime.now()
                timestamp_str = timestamp.strftime('%Y%m%d-%H%M-%S')
                
                # Use existing scan folder if available (created during scan for fit plots)
                # Otherwise create a new one
                if self._current_scan_folder is not None and os.path.isdir(self._current_scan_folder):
                    scan_folder = self._current_scan_folder
                    # Optionally rename to include tag
                    if tag:
                        old_folder = scan_folder
                        parent_dir = os.path.dirname(scan_folder)
                        old_name = os.path.basename(scan_folder)
                        # Insert tag after timestamp
                        parts = old_name.split('_', 1)
                        if len(parts) == 2:
                            new_name = f'{parts[0]}_{tag}_{parts[1]}'
                        else:
                            new_name = f'{old_name}_{tag}'
                        scan_folder = os.path.join(parent_dir, new_name)
                        if old_folder != scan_folder:
                            try:
                                os.rename(old_folder, scan_folder)
                                self._current_scan_folder = scan_folder
                            except OSError:
                                # If rename fails, use original folder
                                scan_folder = old_folder
                else:
                    # Build folder name following qudi convention: YYYYMMDD-HHMM-SS_nametag
                    nametag = f'{tag}_' if tag else ''
                    nametag += f'motor_scan_{self._scan_data.scan_mode.name}'
                    scan_folder_name = f'{timestamp_str}_{nametag}'
                    
                    # Create scan folder inside module_default_data_dir
                    scan_folder = os.path.join(self.module_default_data_dir, scan_folder_name)
                    os.makedirs(scan_folder, exist_ok=True)
                
                self.log.info(f"Saving scan data to {scan_folder}")
                
                # Create storage pointing to the scan folder
                data_storage = TextDataStorage(root_dir=scan_folder)
                
                # Prepare metadata
                metadata = {
                    'Scan Mode': self._scan_data.scan_mode.name,
                    'Scan Pattern': self._scan_data.scan_pattern.name,
                    'Scan Axes': str(self._scan_data.scan_axes),
                    'Scan Range': str(self._scan_data.scan_range),
                    'Scan Resolution': str(self._scan_data.scan_resolution),
                    'Total Points': self._scan_data.total_points,
                    'Completed Points': self._scan_data.current_point_index,
                    'Scan Duration (s)': self._scan_data.scan_duration,
                    'Completed': self._scan_data.completed,
                }
                
                # Add axis-specific metadata
                for i, axis in enumerate(self._scan_data.scan_axes):
                    metadata[f'{axis} axis min'] = self._scan_data.scan_range[i][0]
                    metadata[f'{axis} axis max'] = self._scan_data.scan_range[i][1]
                    metadata[f'{axis} axis resolution'] = self._scan_data.scan_resolution[i]
                
                file_path = None
                
                if self._scan_data.scan_mode in (ScanMode.CONTINUOUS_STREAM,
                                                 ScanMode.KDC_HW_SYNC,
                                                 ScanMode.KDC_HW_SYNC_MULTIRES):
                    # Save streaming data - one file per channel. KDC_HW_SYNC fills
                    # stream_data_mean per grid point from the hardware-marker
                    # reconstruction (KDC_HW_SYNC_MULTIRES: 2N channels =
                    # res{k}_err/res{k}_corr), same shape as CONTINUOUS_STREAM.
                    if self._scan_data.stream_data_mean:
                        for channel, data in self._scan_data.stream_data_mean.items():
                            if (channel.endswith('_corr') or
                                    (self._scan_data.scan_mode == ScanMode.KDC_HW_SYNC and
                                     channel == 'ftw_corr')):
                                unit = 'Hz'
                            elif (self._scan_data.scan_mode == ScanMode.KDC_HW_SYNC_MULTIRES and
                                  channel.endswith('_err')):
                                unit = 'LSB'
                            else:
                                unit = 'V'
                            file_path, _, _ = data_storage.save_data(
                                data,
                                metadata=metadata,
                                nametag=channel,
                                timestamp=timestamp,
                                column_headers=(
                                    f'{channel} data ({unit}; columns is X, rows is Y)'),
                                use_timestamp=False
                            )

                            # Save thumbnail if configured
                            if self._save_thumbnails and file_path:
                                fig = self._draw_figure(data, channel, unit=unit)
                                fig_path = file_path.rsplit('.', 1)[0]
                                data_storage.save_thumbnail(fig, file_path=fig_path)
                                plt.close(fig)
                    else:
                        self.log.warning("No stream data to save.")

                    # KDC_HW_SYNC: also persist the per-bin CUT TIME-TRACES and the
                    # compact faithful dataset (continuous demod trace + hardware
                    # x/y marker indices) so every position bin keeps both its raw
                    # acquired trace and its average, and any binning is exactly
                    # reproducible offline.
                    if self._scan_data.scan_mode in (ScanMode.KDC_HW_SYNC,
                                                     ScanMode.KDC_HW_SYNC_MULTIRES):
                        try:
                            self._save_hw_sync_raw(scan_folder)
                        except Exception as e:
                            self.log.warning("Failed to save KDC_HW_SYNC raw "
                                             "traces/markers: %s", e)

                elif self._scan_data.scan_mode == ScanMode.CONTINUOUS_FREQ_TRACK:
                    # Save absolute frequency data
                    # Add frequency tracking metadata
                    metadata['Zero-crossing History'] = str(self._tracking_zero_crossing_history)
                    if self._tracking_zero_crossing is not None:
                        metadata['Final Zero-crossing (Hz)'] = self._tracking_zero_crossing

                    if self._scan_data.stream_data_mean:
                        for channel, data in self._scan_data.stream_data_mean.items():
                            # Determine unit based on channel name
                            if channel == 'absolute_frequency':
                                unit = 'Hz'
                                header = 'Absolute Frequency (Hz) (columns is X, rows is Y)'
                            else:
                                unit = ''
                                header = f'{channel} data (columns is X, rows is Y)'

                            file_path, _, _ = data_storage.save_data(
                                data,
                                metadata=metadata,
                                nametag=channel,
                                timestamp=timestamp,
                                column_headers=header,
                                use_timestamp=False
                            )

                            # Save thumbnail - display in GHz for readability
                            if self._save_thumbnails and file_path:
                                if channel == 'absolute_frequency':
                                    # Convert Hz to GHz for display
                                    data_ghz = data / 1e9
                                    fig = self._draw_figure(data_ghz, 'Absolute Frequency', unit='GHz')
                                else:
                                    fig = self._draw_figure(data, channel, unit=unit)
                                fig_path = file_path.rsplit('.', 1)[0]
                                data_storage.save_thumbnail(fig, file_path=fig_path)
                                plt.close(fig)
                    else:
                        self.log.warning("No frequency tracking data to save.")

                elif self._scan_data.scan_mode == ScanMode.STEP_ODMR:
                    # Save ODMR fit result arrays
                    if self._scan_data.center_frequency is not None:
                        file_path, _, _ = data_storage.save_data(
                            self._scan_data.center_frequency,
                            metadata=metadata,
                            nametag='center_frequency',
                            timestamp=timestamp,
                            column_headers='Center Frequency (Hz) (columns is X, rows is Y)',
                            use_timestamp=False
                        )
                        
                        if self._save_thumbnails and file_path:
                            fig = self._draw_figure(
                                self._scan_data.center_frequency, 
                                'Center Frequency', 
                                unit='Hz'
                            )
                            fig_path = file_path.rsplit('.', 1)[0]
                            data_storage.save_thumbnail(fig, file_path=fig_path)
                            plt.close(fig)
                        
                    if self._scan_data.linewidth is not None:
                        file_path, _, _ = data_storage.save_data(
                            self._scan_data.linewidth,
                            metadata=metadata,
                            nametag='linewidth',
                            timestamp=timestamp,
                            column_headers='Linewidth (Hz) (columns is X, rows is Y)',
                            use_timestamp=False
                        )
                        
                        if self._save_thumbnails and file_path:
                            fig = self._draw_figure(
                                self._scan_data.linewidth,
                                'Linewidth',
                                unit='Hz'
                            )
                            fig_path = file_path.rsplit('.', 1)[0]
                            data_storage.save_thumbnail(fig, file_path=fig_path)
                            plt.close(fig)
                        
                    if self._scan_data.splitting is not None:
                        file_path, _, _ = data_storage.save_data(
                            self._scan_data.splitting,
                            metadata=metadata,
                            nametag='splitting',
                            timestamp=timestamp,
                            column_headers='Splitting (Hz) (columns is X, rows is Y)',
                            use_timestamp=False
                        )
                        
                        if self._save_thumbnails and file_path:
                            fig = self._draw_figure(
                                self._scan_data.splitting,
                                'Splitting',
                                unit='Hz'
                            )
                            fig_path = file_path.rsplit('.', 1)[0]
                            data_storage.save_thumbnail(fig, file_path=fig_path)
                            plt.close(fig)
                        
                    if self._scan_data.fit_quality is not None:
                        file_path, _, _ = data_storage.save_data(
                            self._scan_data.fit_quality,
                            metadata=metadata,
                            nametag='fit_quality',
                            timestamp=timestamp,
                            column_headers='Fit Quality (columns is X, rows is Y)',
                            use_timestamp=False
                        )
                        
                        if self._save_thumbnails and file_path:
                            fig = self._draw_figure(
                                self._scan_data.fit_quality,
                                'Fit Quality',
                                unit=''
                            )
                            fig_path = file_path.rsplit('.', 1)[0]
                            data_storage.save_thumbnail(fig, file_path=fig_path)
                            plt.close(fig)
                    
                    # Save raw ODMR scans per pixel in a subfolder
                    if self._scan_data.odmr_raw_per_pixel is not None:
                        self._save_odmr_raw_per_pixel(
                            scan_folder, 
                            timestamp, 
                            metadata
                        )
                
                # Save positions data (for all modes)
                self._save_positions_data(scan_folder, timestamp, metadata)
                    
                self.log.info(f"Scan data saved to: {scan_folder}")

            finally:
                # Only unlock if we locked it ourselves
                if not already_locked:
                    self.module_state.unlock()
                self.sigSaveStateChanged.emit(False)
    
    def _draw_figure(self, data: np.ndarray, data_label: str, unit: str = '') -> plt.Figure:
        """
        Draw a 2D color map figure of the scan data.
        
        Args:
            data: 2D numpy array with scan data (shape: nx, ny)
            data_label: Label for the data (e.g., 'Center Frequency')
            unit: Unit string for the data (e.g., 'Hz')
            
        Returns:
            matplotlib.figure.Figure: Figure object ready for saving.
        """
        if self._scan_data is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return fig
        
        # Get scan ranges
        x_range = self._scan_data.scan_range[0]
        y_range = self._scan_data.scan_range[1] if len(self._scan_data.scan_range) > 1 else (0, 1)

        # Detect degenerate ranges (line scan where one axis has start == stop)
        x_degenerate = (x_range[0] == x_range[1])
        y_degenerate = (y_range[0] == y_range[1])

        # For degenerate axes, create artificial extent so imshow renders a visible strip
        if y_degenerate and not x_degenerate:
            x_span = abs(x_range[1] - x_range[0])
            artificial_half = x_span / max(data.shape[0], 1) / 2
            x_range_display = x_range
            y_range_display = (y_range[0] - artificial_half, y_range[1] + artificial_half)
        elif x_degenerate and not y_degenerate:
            y_span = abs(y_range[1] - y_range[0])
            artificial_half = y_span / max(data.shape[1] if data.ndim > 1 else 1, 1) / 2
            x_range_display = (x_range[0] - artificial_half, x_range[1] + artificial_half)
            y_range_display = y_range
        else:
            x_range_display = x_range
            y_range_display = y_range

        # Handle colorbar range - ignore NaN values
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            cbar_range = (0, 1)
        else:
            cbar_range = (np.nanmin(data), np.nanmax(data))

        # Calculate SI scaling for axes
        # For degenerate axes, use absolute position value for SI prefix instead of zero span
        x_span_display = abs(x_range_display[1] - x_range_display[0])
        y_span_display = abs(y_range_display[1] - y_range_display[0])
        si_prefix_x = ScaledFloat(x_span_display).scale if x_span_display > 0 else ScaledFloat(abs(x_range[0])).scale
        si_factor_x = ScaledFloat(x_span_display).scale_val if x_span_display > 0 else ScaledFloat(abs(x_range[0])).scale_val
        si_prefix_y = ScaledFloat(y_span_display).scale if y_span_display > 0 else ScaledFloat(abs(y_range[0])).scale
        si_factor_y = ScaledFloat(y_span_display).scale_val if y_span_display > 0 else ScaledFloat(abs(y_range[0])).scale_val
        # Avoid division by zero for SI factors
        if si_factor_x == 0:
            si_factor_x = 1
        if si_factor_y == 0:
            si_factor_y = 1

        # Calculate SI scaling for colorbar
        if cbar_range[1] != cbar_range[0]:
            si_prefix_cb = ScaledFloat(cbar_range[1] - cbar_range[0]).scale
            si_factor_cb = ScaledFloat(cbar_range[1] - cbar_range[0]).scale_val
        else:
            si_prefix_cb = ScaledFloat(cbar_range[1]).scale if cbar_range[1] != 0 else ''
            si_factor_cb = ScaledFloat(cbar_range[1]).scale_val if cbar_range[1] != 0 else 1

        # Create figure
        fig, ax = plt.subplots()

        # Create image plot
        # Data shape is (nx, ny), but imshow expects (rows, cols) = (ny, nx)
        # So we transpose the data for correct display
        cfimage = ax.imshow(
            data.T / si_factor_cb,
            cmap='inferno',
            origin='lower',
            vmin=cbar_range[0] / si_factor_cb,
            vmax=cbar_range[1] / si_factor_cb,
            interpolation='none',
            extent=(
                x_range_display[0] / si_factor_x,
                x_range_display[1] / si_factor_x,
                y_range_display[0] / si_factor_y,
                y_range_display[1] / si_factor_y
            )
        )

        # Set axis labels
        x_axis_name = self._scan_data.scan_axes[0] if self._scan_data.scan_axes else 'x'
        y_axis_name = self._scan_data.scan_axes[1] if len(self._scan_data.scan_axes) > 1 else 'y'
        ax.set_xlabel(f'{x_axis_name} position ({si_prefix_x}m)')
        ax.set_ylabel(f'{y_axis_name} position ({si_prefix_y}m)')

        # Use 'auto' aspect for line scans (degenerate axis), equal aspect for 2D
        ax.set_aspect('auto' if (x_degenerate or y_degenerate) else 1)
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        
        # Draw the colorbar
        cbar = plt.colorbar(cfimage, shrink=0.8)
        if unit:
            cbar.set_label(f'{data_label} ({si_prefix_cb}{unit})')
        else:
            cbar.set_label(f'{data_label}')
        
        # Remove ticks from colorbar for cleaner image
        cbar.ax.tick_params(which='both', length=0)
        
        # Add scan metadata annotation
        metainfo_str = self._get_figure_metadata_string()
        if metainfo_str:
            ax.annotate(
                metainfo_str,
                xy=(1.10, -0.17),
                xycoords='axes fraction',
                horizontalalignment='left',
                verticalalignment='bottom',
                fontsize=7,
                color='grey'
            )
        
        return fig
    
    def _get_figure_metadata_string(self) -> str:
        """
        Generate a metadata string for the figure annotation.
        
        Returns:
            Formatted string with scan metadata.
        """
        if self._scan_data is None:
            return ''
        
        lines = []
        
        # Scan mode and pattern
        lines.append(f"Mode: {self._scan_data.scan_mode.name}")
        lines.append(f"Pattern: {self._scan_data.scan_pattern.name}")
        
        # Resolution
        if self._scan_data.scan_resolution:
            res_str = ' x '.join(str(r) for r in self._scan_data.scan_resolution)
            lines.append(f"Resolution: {res_str} points")
        
        # Duration
        if self._scan_data.scan_duration > 0:
            lines.append(f"Duration: {self._scan_data.scan_duration:.1f}s")
        
        return '\n'.join(lines)
    
    def _save_hw_sync_raw(self, scan_folder: str):
        """Persist KDC hardware-sync raw deliverables into the scan folder.

          * ``hw_sync_bin_traces.npy`` -- object array indexed by flat point index;
            each entry is that position bin's CUT demod time-trace (float32). The
            flat index maps to grid (ix, iy) via the scan pattern, identical to the
            average map; ``mean(trace) == average`` for that bin.
          * ``hw_sync_demod_trace.npy`` (float32) + ``hw_sync_x_markers.npy`` +
            ``hw_sync_y_markers.npy`` -- the compact FAITHFUL dataset: the whole
            continuous demod stream and the hardware marker sample-indices (x = bin
            boundaries, y = line boundaries). These three reproduce ANY binning
            exactly offline, independent of velocity/accel.

        Binary ``.npy`` (not ``.dat``) because the trace is multi-million samples.
        """
        sd = self._scan_data
        full = bool(getattr(self, '_save_full_traces', False))

        # Always save the tiny hardware-sync record + USB position cross-check
        # (negligible size, documents the scan's fidelity regardless of the toggle):
        # the x/y marker indices and the measured-vs-target slow-axis position per line.
        np.save(os.path.join(scan_folder, 'hw_sync_x_markers.npy'),
                np.asarray(getattr(sd, 'hw_x_markers', []), dtype=np.int64))
        np.save(os.path.join(scan_folder, 'hw_sync_y_markers.npy'),
                np.asarray(getattr(sd, 'hw_y_markers', []), dtype=np.int64))
        ya = getattr(sd, 'hw_y_positions_actual', None)
        if ya is not None and len(ya):
            np.save(os.path.join(scan_folder, 'hw_sync_y_actual_per_line.npy'),
                    np.asarray(ya, dtype=np.float64))
            np.save(os.path.join(scan_folder, 'hw_sync_y_target_per_line.npy'),
                    np.asarray(getattr(sd, 'hw_y_positions_target', []), dtype=np.float64))

        if not full:
            self.log.info("KDC_HW_SYNC: 'save full traces' is OFF -> saved mean map + "
                          "markers + per-line y only (no per-bin traces / demod trace).")
            return

        # Multi-resonance mode has one cut trace per resonance/quantity plus the
        # raw stream events. Keep all of them: selecting just
        # the first channel would silently discard either correction or demod error.
        if sd.scan_mode == ScanMode.KDC_HW_SYNC_MULTIRES:
            saved_channels = 0
            for ch, segments in (sd.stream_data_raw or {}).items():
                if not segments:
                    continue
                traces = np.array(
                    [np.asarray(seg, dtype=np.float32) for seg in segments],
                    dtype=object)
                np.save(os.path.join(scan_folder, f'hw_sync_{ch}_bin_traces.npy'),
                        traces, allow_pickle=True)
                saved_channels += 1

            # Raw timestamped stream events (structured array; decode with
            # pyrpl DataStreamer.reconstruct_tracking).
            events = getattr(sd, 'hw_stream_events', None)
            if events is not None and len(events):
                np.save(os.path.join(scan_folder, 'hw_sync_stream_events.npy'), events)
            try:
                times = getattr(sd, 'hw_multires_times', None)
                err = getattr(sd, 'hw_multires_err', None)
                corr = getattr(sd, 'hw_multires_corr_hz', None)
                sample_rate = getattr(sd, 'hw_multires_sample_rate', None)
                if times is None or err is None or corr is None:
                    # Backward-compatible fallback for scan-data objects created
                    # before finalization began snapshotting the reconstruction.
                    hw = self._get_tracking_hw()
                    reconstructed = (
                        hw.reconstruct_recording(
                            events, nslots=getattr(sd, 'hw_multires_nslots', None))
                        if hw is not None and events is not None and len(events)
                        else None)
                    if reconstructed is not None:
                        times = reconstructed['times']
                        err = reconstructed['err']
                        corr = reconstructed['corr_hz']
                        sample_rate = reconstructed.get('sample_rate')
                if times is not None and err is not None and corr is not None:
                    times = np.asarray(times, dtype=np.float64)
                    err = np.asarray(err, dtype=np.float32)
                    corr = np.asarray(corr, dtype=np.float64)
                    np.save(os.path.join(scan_folder, 'hw_sync_time_s.npy'), times)
                    if sample_rate is not None:
                        np.save(os.path.join(scan_folder, 'hw_sync_sample_rate_hz.npy'),
                                np.asarray(float(sample_rate), dtype=np.float64))
                    for k in range(min(err.shape[0], corr.shape[0])):
                        np.save(os.path.join(scan_folder, f'hw_sync_res{k}_err_trace.npy'),
                                err[k])
                        np.save(os.path.join(scan_folder, f'hw_sync_res{k}_corr_hz_trace.npy'),
                                corr[k])
            except Exception as e:
                self.log.warning("Could not save reconstructed multi-resonance "
                                 "faithful traces: %s", e)
            self.log.info("KDC_HW_SYNC_MULTIRES raw saved: %d per-channel cut-trace "
                          "arrays + stream events/markers in %s",
                          saved_channels, scan_folder)
            return

        # Single-resonance full per-bin CUT time-traces + faithful demod trace.
        n_saved = 0
        ch = None
        if sd.stream_data_raw:
            ch = 'demod' if 'demod' in sd.stream_data_raw else next(iter(sd.stream_data_raw))
        if ch is not None and sd.stream_data_raw.get(ch):
            traces = np.array(
                [np.asarray(seg, dtype=np.float32) for seg in sd.stream_data_raw[ch]],
                dtype=object)
            np.save(os.path.join(scan_folder, 'hw_sync_bin_traces.npy'),
                    traces, allow_pickle=True)
            n_saved = len(traces)
        demod = getattr(sd, 'hw_demod_trace', None)
        if demod is not None and len(demod):
            np.save(os.path.join(scan_folder, 'hw_sync_demod_trace.npy'),
                    np.asarray(demod, dtype=np.float32))
        self.log.info("KDC_HW_SYNC raw saved: %d per-bin traces + faithful "
                      "demod/markers + per-line y(USB) in %s", n_saved, scan_folder)

    def _save_odmr_raw_per_pixel(
        self,
        scan_folder: str,
        timestamp: datetime.datetime,
        metadata: Dict
    ):
        """
        Save raw ODMR scans for each pixel in a subfolder.
        
        Creates a subfolder 'odmr_raw_per_pixel' inside the scan folder containing
        individual ODMR data files for each pixel, enabling detailed post-analysis.
        
        Args:
            scan_folder: Path to the main scan folder
            timestamp: Timestamp of the save operation
            metadata: Base metadata dict
        """
        if self._scan_data is None or self._scan_data.odmr_raw_per_pixel is None:
            return
        
        # Count how many pixels have data
        valid_pixels = [p for p in self._scan_data.odmr_raw_per_pixel if p is not None]
        if not valid_pixels:
            return
        
        try:
            # Create subfolder inside the scan folder
            odmr_subfolder = os.path.join(scan_folder, 'odmr_raw_per_pixel')
            os.makedirs(odmr_subfolder, exist_ok=True)
            
            for point_idx, pixel_data in enumerate(self._scan_data.odmr_raw_per_pixel):
                if pixel_data is None:
                    continue
                
                grid_idx = pixel_data.get('grid_index', (point_idx,))
                target_pos = pixel_data.get('target_position', {})
                actual_pos = pixel_data.get('actual_position', {})
                freq_data = pixel_data.get('frequency_data')
                signal_data = pixel_data.get('signal_data', {})
                
                if freq_data is None or len(signal_data) == 0:
                    continue
                
                # Create pixel-specific metadata
                pixel_metadata = metadata.copy()
                pixel_metadata['Pixel Index'] = point_idx
                pixel_metadata['Grid Index'] = str(grid_idx)
                for axis, val in target_pos.items():
                    pixel_metadata[f'Target {axis} (m)'] = val
                for axis, val in actual_pos.items():
                    pixel_metadata[f'Actual {axis} (m)'] = val
                
                # Create filename with grid indices for easy sorting
                if len(grid_idx) == 2:
                    pixel_tag = f'pixel_x{grid_idx[0]:03d}_y{grid_idx[1]:03d}'
                else:
                    pixel_tag = f'pixel_{point_idx:04d}'
                
                # Build data array: frequency column + signal columns
                n_points = len(freq_data)
                columns = [freq_data]
                col_headers = ['Frequency (Hz)']
                
                for ch_name, ch_data in signal_data.items():
                    if len(ch_data) == n_points:
                        columns.append(ch_data)
                        col_headers.append(f'{ch_name} (V)')
                
                # Stack columns into 2D array
                data_array = np.column_stack(columns)
                
                # Save to file
                file_path = os.path.join(odmr_subfolder, f'{pixel_tag}_odmr.dat')
                
                # Write file manually with header
                with open(file_path, 'w') as f:
                    # Write metadata header
                    f.write('# ODMR Raw Data for Motor Scan Pixel\n')
                    f.write(f'# Saved: {timestamp.isoformat()}\n')
                    f.write('#\n')
                    for key, val in pixel_metadata.items():
                        f.write(f'# {key}: {val}\n')
                    f.write('#\n')
                    f.write('# ' + '\t'.join(col_headers) + '\n')
                    
                    # Write data
                    for row in data_array:
                        f.write('\t'.join(f'{v:.15e}' for v in row) + '\n')
            
        except Exception as e:
            self.log.error(f"Failed to save raw ODMR data per pixel: {e}")

    def _save_positions_data(
        self,
        scan_folder: str,
        timestamp: datetime.datetime,
        metadata: Dict
    ):
        """
        Save target and actual positions for all scan points.
        
        Creates a 'positions.dat' file in the scan folder containing a table with:
        - Point index and grid indices
        - Target positions for each axis
        - Actual (measured) positions for each axis
        - Position errors (actual - target) for each axis
        
        This enables post-scan analysis of positioning accuracy.
        
        Args:
            scan_folder: Path to the main scan folder
            timestamp: Timestamp of the save operation
            metadata: Base metadata dict
        """
        if self._scan_data is None:
            return
        
        if self._scan_data.target_positions is None:
            return
        
        try:
            file_path = os.path.join(scan_folder, 'positions.dat')
            axes = self._scan_data.scan_axes
            n_axes = len(axes)
            n_points = self._scan_data.total_points

            # KDC_HW_SYNC: the per-point actual position is not polled (the fast axis
            # sweeps continuously), but the SLOW axis is measured once per line via USB
            # (the cross-check stored on scan_data). Surface that so the slow-axis
            # Actual/Error columns are populated; the fast (per-bin) axis stays nan,
            # since it genuinely is not measured per point.
            hw_slow_actual = getattr(self._scan_data, 'hw_y_positions_actual', None)
            hw_slow_axis = None
            hw_slow_col = -1
            if (hw_slow_actual is not None and len(hw_slow_actual)
                    and self._scan_data.is_2d):
                try:
                    hw_slow_axis = self._scan_data.get_slow_axis()
                    hw_slow_col = list(axes).index(hw_slow_axis)
                except Exception:
                    hw_slow_actual = None

            def _hw_slow_line(grid_idx):
                """Slow-axis line index for a point's (ix, iy) grid index."""
                return int(grid_idx[0]) if hw_slow_axis == 'x' else int(grid_idx[1])
            
            # Build column headers
            col_headers = ['Point_Index']
            if self._scan_data.is_2d:
                col_headers.extend(['Grid_X', 'Grid_Y'])
            else:
                col_headers.append('Grid_Index')
            
            for axis in axes:
                col_headers.append(f'Target_{axis} (m)')
            for axis in axes:
                col_headers.append(f'Actual_{axis} (m)')
            for axis in axes:
                col_headers.append(f'Error_{axis} (m)')
            
            with open(file_path, 'w') as f:
                # Write header
                f.write('# Motor Scan Position Data\n')
                f.write(f'# Saved: {timestamp.isoformat()}\n')
                f.write('#\n')
                for key, val in metadata.items():
                    f.write(f'# {key}: {val}\n')
                f.write('#\n')
                
                # Compute position statistics
                if self._scan_data.actual_positions is not None:
                    valid_mask = ~np.isnan(self._scan_data.actual_positions).any(axis=1)
                    if valid_mask.any():
                        errors = self._scan_data.actual_positions[valid_mask] - self._scan_data.target_positions[valid_mask]
                        mean_error = np.mean(np.abs(errors), axis=0)
                        max_error = np.max(np.abs(errors), axis=0)
                        rms_error = np.sqrt(np.mean(errors**2, axis=0))
                        
                        f.write('# Position Statistics:\n')
                        for i, axis in enumerate(axes):
                            f.write(f'#   {axis}-axis: mean_abs_error={mean_error[i]*1e6:.2f}um, '
                                    f'max_abs_error={max_error[i]*1e6:.2f}um, '
                                    f'rms_error={rms_error[i]*1e6:.2f}um\n')
                        f.write('#\n')

                # KDC_HW_SYNC slow-axis (per-line USB cross-check) position accuracy.
                # In this mode the fast/per-bin axis Actual is not measured (nan); the
                # slow axis is measured once per settled line -- the authoritative
                # position record. Per-bin position is hardware-anchored via markers.
                if hw_slow_actual is not None:
                    hw_slow_target = getattr(self._scan_data, 'hw_y_positions_target', None)
                    ya = np.asarray(hw_slow_actual, dtype=float)
                    yt = (np.asarray(hw_slow_target, dtype=float)
                          if hw_slow_target is not None else None)
                    f.write('# Note: KDC_HW_SYNC -- Actual/Error filled for the SLOW '
                            f'axis ({hw_slow_axis}) only (per-line USB cross-check); the '
                            'fast/per-bin axis is hardware-anchored via markers, not '
                            'polled per point (nan).\n')
                    if yt is not None and yt.shape == ya.shape:
                        dev = ya - yt
                        m = np.isfinite(dev)
                        if m.any():
                            f.write(f'#   {hw_slow_axis}-axis (slow, per line, n={int(m.sum())}): '
                                    f'mean_abs_error={np.mean(np.abs(dev[m]))*1e6:.3f}um, '
                                    f'max_abs_error={np.max(np.abs(dev[m]))*1e6:.3f}um, '
                                    f'rms_error={np.sqrt(np.mean(dev[m]**2))*1e6:.3f}um\n')
                    f.write('#\n')

                # Write column headers
                f.write('# ' + '\t'.join(col_headers) + '\n')
                
                # Write data rows
                for point_idx in range(n_points):
                    row = [str(point_idx)]
                    
                    # Grid indices
                    grid_idx = self._scan_data.point_index_to_grid_index(point_idx)
                    if self._scan_data.is_2d:
                        row.extend([str(grid_idx[0]), str(grid_idx[1])])
                    else:
                        row.append(str(grid_idx[0]))
                    
                    # Target positions
                    target = self._scan_data.target_positions[point_idx]
                    for i in range(n_axes):
                        row.append(f'{target[i]:.9e}')
                    
                    # Actual positions: start from the per-point poll (if any), then
                    # overlay the per-line slow-axis USB cross-check for KDC_HW_SYNC.
                    actual_vals = [np.nan] * n_axes
                    if self._scan_data.actual_positions is not None:
                        ap = self._scan_data.actual_positions[point_idx]
                        for i in range(n_axes):
                            actual_vals[i] = ap[i]
                    if hw_slow_actual is not None:
                        line = _hw_slow_line(grid_idx)
                        if 0 <= line < len(hw_slow_actual):
                            v = hw_slow_actual[line]
                            if np.isfinite(v):
                                actual_vals[hw_slow_col] = float(v)

                    for i in range(n_axes):
                        row.append('nan' if not np.isfinite(actual_vals[i])
                                   else f'{actual_vals[i]:.9e}')
                    for i in range(n_axes):
                        row.append('nan' if not np.isfinite(actual_vals[i])
                                   else f'{actual_vals[i] - target[i]:.9e}')

                    f.write('\t'.join(row) + '\n')
            
            self.log.debug(f"Saved positions data to {file_path}")
            
        except Exception as e:
            self.log.error(f"Failed to save positions data: {e}")
