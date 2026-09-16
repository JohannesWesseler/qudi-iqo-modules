# -*- coding: utf-8 -*-
"""
Sensitivity Sweep Visualization Module

Generates summary plots from sensitivity parameter sweep results, showing how
sensitivity and ODMR parameters vary with the swept parameters (power_dbm,
f_mod_hz, f_dev_khz).

This module is designed to be called from sensitivity_sweep_logic.py but is
kept separate for maintainability and to avoid bloating the main logic module.

Copyright (c) 2021, the qudi developers. See the AUTHORS.md file at the top-level
directory of this distribution and on <https://github.com/Ulm-IQO/qudi-core/>
"""

__all__ = ['SensitivitySweepVisualizer']

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from typing import Dict, Any, Optional, List, Tuple
import logging

# Use non-interactive backend for file saving (avoids Qt conflicts in threaded context)
matplotlib.use('Agg')


class SensitivitySweepVisualizer:
    """
    Generates summary visualization plots for sensitivity parameter sweeps.

    This class takes the results DataFrame and metadata from a sensitivity sweep
    and produces a set of plots showing how sensitivity and ODMR parameters
    vary with the swept parameters.

    Usage:
        visualizer = SensitivitySweepVisualizer(results_df, metadata, output_folder)
        visualizer.generate_all_plots()
    """

    def __init__(self, results_df: pd.DataFrame, metadata: Dict[str, Any],
                 output_folder: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the visualizer.

        Args:
            results_df: DataFrame with sweep results (from parameter_sweep_summary.csv)
            metadata: Dictionary with sweep metadata (from sweep_metadata.json)
            output_folder: Folder where plots will be saved
            logger: Optional logger instance (uses module logger if not provided)
        """
        self.df = results_df
        self.metadata = metadata
        self.output_folder = output_folder
        self.log = logger or logging.getLogger(__name__)

        # Create summary_plots subfolder
        self.plots_folder = os.path.join(output_folder, 'summary_plots')
        if not os.path.exists(self.plots_folder):
            os.makedirs(self.plots_folder)

        # Extract sweep loop order
        self.loop_order = metadata.get('sweep_loop_order', ['power', 'f_mod', 'f_dev'])

        # Determine which parameters were actually varied (more than 1 unique value)
        self.varied_params = self._identify_varied_parameters()

    def _identify_varied_parameters(self) -> Dict[str, np.ndarray]:
        """Identify which parameters were varied in the sweep."""
        varied = {}
        param_columns = {
            'power': 'power_dbm',
            'f_mod': 'f_mod_hz',
            'f_dev': 'f_dev_khz'
        }

        for param_key, col_name in param_columns.items():
            if col_name in self.df.columns:
                unique_vals = self.df[col_name].unique()
                if len(unique_vals) > 1:
                    varied[param_key] = np.sort(unique_vals)

        return varied

    @staticmethod
    def _dbm_to_if_amplitude(dbm):
        """Convert dBm to IF amplitude (V_peak, 0-1 range).

        This matches the formula in mw_source_windfreak_synthnvpro_redpitaya.py:
        if_amplitude = 10^((dBm - 10) / 20)

        Assumes 50 ohm load without calibration data.
        Note: If calibration data is used in the actual measurement,
        the real IF amplitude may differ from this calculation.

        Args:
            dbm: Power in dBm (scalar or array)

        Returns:
            IF amplitude in volts (0-1 range)
        """
        return 10 ** ((np.asarray(dbm) - 10) / 20)

    def generate_all_plots(self) -> List[str]:
        """
        Generate all summary plots.

        Returns:
            List of paths to generated plot files
        """
        generated_files = []

        # Avoid averaging different filter responses or controller conditions
        # together in an automated filter comparison.
        if 'fir_filter_bandwidth' in self.df and self.df['fir_filter_bandwidth'].nunique() > 1:
            columns = [name for name in (
                'fir_filter_bandwidth', 'measurement_mode', 'controller_bandwidth_hz')
                if name in self.df]
            for index, (key, frame) in enumerate(self.df.groupby(columns, dropna=False)):
                label = '_'.join(str(value) for value in (key if isinstance(key, tuple) else (key,)))
                folder = os.path.join(self.plots_folder, f'{index:03d}_{label}')
                metadata = self.metadata.copy()
                valid = frame.dropna(subset=['sensitivity_nT_rtHz'])
                if not valid.empty:
                    best = valid.loc[valid['sensitivity_nT_rtHz'].idxmin()]
                    metadata['best_sensitivity_nT_rtHz'] = float(best['sensitivity_nT_rtHz'])
                    metadata['best_parameters'] = {
                        'power': best.get('power_dbm'), 'f_mod': best.get('f_mod_hz'),
                        'f_dev': best.get('f_dev_khz')}
                metadata['total_measurements'] = len(frame)
                child = SensitivitySweepVisualizer(frame, metadata, folder, self.log)
                generated_files.extend(child.generate_all_plots())
            return generated_files

        if len(self.df) == 0:
            self.log.warning('No data to visualize - results DataFrame is empty')
            return generated_files

        # Only generate plots if we have at least 2 data points
        if len(self.df) < 2:
            self.log.warning('Insufficient data for visualization (need at least 2 points)')
            return generated_files

        try:
            # 1. Sensitivity heatmap (if 2 parameters varied)
            if len(self.varied_params) >= 2:
                path = self._plot_sensitivity_heatmap()
                if path:
                    generated_files.append(path)

            # 2. Sensitivity vs each varied parameter (line plots)
            for param_key in self.varied_params:
                path = self._plot_sensitivity_vs_parameter(param_key)
                if path:
                    generated_files.append(path)

            # 2b. Averaged sensitivity plots (marginal effects)
            if len(self.varied_params) >= 2:
                paths = self._plot_averaged_sensitivity()
                generated_files.extend(paths)

            # 3. ODMR center vs parameters (if varied)
            if len(self.varied_params) >= 1:
                path = self._plot_odmr_center_vs_parameters()
                if path:
                    generated_files.append(path)

            # 4. Linewidth vs parameters
            if 'linewidth_hz' in self.df.columns and len(self.varied_params) >= 1:
                path = self._plot_linewidth_vs_parameters()
                if path:
                    generated_files.append(path)

            # 5. Summary statistics plot
            path = self._plot_summary_statistics()
            if path:
                generated_files.append(path)

            self.log.info(f'Generated {len(generated_files)} summary plots in {self.plots_folder}')

        except Exception as e:
            self.log.error(f'Error generating plots: {e}', exc_info=True)

        return generated_files

    def _plot_sensitivity_heatmap(self) -> Optional[str]:
        """
        Create 2D heatmap of sensitivity vs two varied parameters.

        Uses logarithmic colorbar for better visualization of sensitivity values
        (typically spanning 0.05-1 nT/√Hz). Power axis shows IF amplitude instead
        of dBm for clearer interpretation.

        Returns:
            Path to saved figure, or None if plot couldn't be generated
        """
        # Determine which two parameters to use for the heatmap
        # Prefer power and f_dev as they are most commonly varied together
        param_priority = ['power', 'f_dev', 'f_mod']
        x_param, y_param = None, None

        for param in param_priority:
            if param in self.varied_params:
                if x_param is None:
                    x_param = param
                elif y_param is None:
                    y_param = param
                    break

        if x_param is None or y_param is None:
            self.log.debug('Not enough varied parameters for heatmap')
            return None

        # Map parameter keys to column names
        col_map = {'power': 'power_dbm', 'f_mod': 'f_mod_hz', 'f_dev': 'f_dev_khz'}
        x_col, y_col = col_map[x_param], col_map[y_param]

        # Get unique values (in original units for data lookup)
        x_vals_raw = np.sort(self.df[x_col].unique())
        y_vals_raw = np.sort(self.df[y_col].unique())

        # Convert to display units (IF amplitude for power, kHz for f_mod)
        if x_param == 'power':
            x_vals_display = self._dbm_to_if_amplitude(x_vals_raw)
            x_label = 'IF Amplitude [V]'
        elif x_param == 'f_mod':
            x_vals_display = x_vals_raw / 1e3  # Convert Hz to kHz
            x_label = 'Modulation Frequency [kHz]'
        else:
            x_vals_display = x_vals_raw
            x_label = 'FM Deviation [kHz]'

        if y_param == 'power':
            y_vals_display = self._dbm_to_if_amplitude(y_vals_raw)
            y_label = 'IF Amplitude [V]'
        elif y_param == 'f_mod':
            y_vals_display = y_vals_raw / 1e3
            y_label = 'Modulation Frequency [kHz]'
        else:
            y_vals_display = y_vals_raw
            y_label = 'FM Deviation [kHz]'

        # Create sensitivity matrix
        sens_matrix = np.full((len(y_vals_raw), len(x_vals_raw)), np.nan)

        for i, y_val in enumerate(y_vals_raw):
            for j, x_val in enumerate(x_vals_raw):
                mask = (self.df[x_col] == x_val) & (self.df[y_col] == y_val)
                if mask.any():
                    sens_matrix[i, j] = self.df.loc[mask, 'sensitivity_nT_rtHz'].values[0]

        # Find best point (in display coordinates)
        best_idx = self.df['sensitivity_nT_rtHz'].idxmin()
        best_x_raw = self.df.loc[best_idx, x_col]
        best_y_raw = self.df.loc[best_idx, y_col]
        best_sens = self.df.loc[best_idx, 'sensitivity_nT_rtHz']

        # Convert best point to display coordinates
        if x_param == 'power':
            best_x = self._dbm_to_if_amplitude(best_x_raw)
        elif x_param == 'f_mod':
            best_x = best_x_raw / 1e3
        else:
            best_x = best_x_raw

        if y_param == 'power':
            best_y = self._dbm_to_if_amplitude(best_y_raw)
        elif y_param == 'f_mod':
            best_y = best_y_raw / 1e3
        else:
            best_y = best_y_raw

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Use LogNorm for colorbar (sensitivity values typically span 0.05-1 nT/√Hz)
        valid_sens = sens_matrix[~np.isnan(sens_matrix)]
        if len(valid_sens) > 0 and valid_sens.min() > 0:
            vmin = max(valid_sens.min() * 0.9, 0.01)  # Floor at 0.01 to avoid log issues
            vmax = valid_sens.max() * 1.1
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None

        # Plot heatmap with display coordinates
        im = ax.imshow(sens_matrix, aspect='auto', origin='lower',
                       extent=[x_vals_display.min(), x_vals_display.max(),
                               y_vals_display.min(), y_vals_display.max()],
                       cmap='viridis_r', norm=norm)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, label=r'Sensitivity [nT/$\sqrt{\mathrm{Hz}}$]')

        # Mark best point
        ax.scatter([best_x], [best_y], marker='*', s=300, c='red', edgecolors='white',
                   linewidths=2, zorder=5, label=f'Best: {best_sens:.3f} nT/√Hz')

        # Labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title('Magnetic Field Sensitivity vs Parameters')
        ax.legend(loc='upper right')

        # Add grid lines at parameter values (in display units)
        ax.set_xticks(x_vals_display)
        ax.set_yticks(y_vals_display)

        # Format tick labels based on axis type
        if x_param == 'power':
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
        if y_param == 'power':
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.2f}'))

        # Rotate x-axis labels if there are many values
        if len(x_vals_display) > 8:
            ax.tick_params(axis='x', rotation=45)

        ax.grid(True, alpha=0.3, linestyle='--')

        fig.tight_layout()

        # Save
        filepath = os.path.join(self.plots_folder, 'sensitivity_heatmap.pdf')
        fig.savefig(filepath, dpi=150)
        plt.close(fig)

        self.log.debug(f'Saved sensitivity heatmap: {filepath}')
        return filepath

    def _plot_sensitivity_vs_parameter(self, param_key: str) -> Optional[str]:
        """
        Create line plot of sensitivity vs one parameter, with lines for each
        value of the other varied parameter.

        Power axis displays IF amplitude instead of dBm for clearer interpretation.

        Args:
            param_key: Parameter key ('power', 'f_mod', or 'f_dev')

        Returns:
            Path to saved figure, or None if plot couldn't be generated
        """
        col_map = {'power': 'power_dbm', 'f_mod': 'f_mod_hz', 'f_dev': 'f_dev_khz'}
        x_col = col_map[param_key]

        if x_col not in self.df.columns:
            return None

        # Determine grouping parameter (the other varied parameter)
        other_params = [p for p in self.varied_params if p != param_key]
        group_param = other_params[0] if other_params else None
        group_col = col_map.get(group_param) if group_param else None

        fig, ax = plt.subplots(figsize=(10, 6))

        # Conversion function for x values
        def convert_x(x_raw):
            if param_key == 'power':
                return self._dbm_to_if_amplitude(x_raw)
            elif param_key == 'f_mod':
                return x_raw / 1e3  # Hz to kHz
            else:
                return x_raw

        if group_col and group_col in self.df.columns:
            # Plot lines for each group value
            group_vals = np.sort(self.df[group_col].unique())
            colors = plt.cm.viridis(np.linspace(0, 0.9, len(group_vals)))

            for color, group_val in zip(colors, group_vals):
                mask = self.df[group_col] == group_val
                subset = self.df[mask].sort_values(x_col)

                # Convert x values for display
                x_display = convert_x(subset[x_col].values)

                # Format label based on parameter type
                if group_param == 'power':
                    if_amp = self._dbm_to_if_amplitude(group_val)
                    label = f'{if_amp:.2f} V'
                elif group_param == 'f_mod':
                    label = f'{group_val/1e3:.1f} kHz'
                elif group_param == 'f_dev':
                    label = f'{group_val:.1f} kHz'
                else:
                    label = str(group_val)

                ax.plot(x_display, subset['sensitivity_nT_rtHz'],
                        'o-', color=color, label=label, markersize=8, linewidth=2)
        else:
            # Single line if no grouping
            sorted_df = self.df.sort_values(x_col)
            x_display = convert_x(sorted_df[x_col].values)
            ax.plot(x_display, sorted_df['sensitivity_nT_rtHz'],
                    'o-', markersize=8, linewidth=2)

        # Mark best point
        best_idx = self.df['sensitivity_nT_rtHz'].idxmin()
        best_x_raw = self.df.loc[best_idx, x_col]
        best_x = convert_x(best_x_raw)
        best_sens = self.df.loc[best_idx, 'sensitivity_nT_rtHz']
        ax.scatter([best_x], [best_sens], marker='*', s=300, c='red', edgecolors='white',
                   linewidths=2, zorder=10, label=f'Best: {best_sens:.3f}')

        # Labels and formatting
        if param_key == 'power':
            x_label = 'IF Amplitude [V]'
            title_suffix = 'IF Amplitude'
        elif param_key == 'f_mod':
            x_label = 'Modulation Frequency [kHz]'
            title_suffix = 'Modulation Frequency'
        else:
            x_label = 'FM Deviation [kHz]'
            title_suffix = 'FM Deviation'

        ax.set_xlabel(x_label)
        ax.set_ylabel(r'Sensitivity [nT/$\sqrt{\mathrm{Hz}}$]')
        ax.set_title(f'Sensitivity vs {title_suffix}')

        # Legend title based on grouping parameter
        if group_param == 'power':
            legend_title = 'IF Amplitude'
        elif group_param == 'f_mod':
            legend_title = 'Mod. Freq.'
        elif group_param == 'f_dev':
            legend_title = 'f_dev [kHz]'
        else:
            legend_title = None

        ax.grid(True, alpha=0.3)
        ax.legend(title=legend_title, loc='best', fontsize=9)

        fig.tight_layout()

        # Save
        filepath = os.path.join(self.plots_folder, f'sensitivity_vs_{param_key}.pdf')
        fig.savefig(filepath, dpi=150)
        plt.close(fig)

        self.log.debug(f'Saved sensitivity vs {param_key}: {filepath}')
        return filepath

    def _plot_averaged_sensitivity(self) -> List[str]:
        """
        Create averaged sensitivity plots showing marginal effects.

        Generates:
        1. Sensitivity vs IF amplitude, averaged across all f_dev values
        2. Sensitivity vs f_dev, averaged across all IF amplitude values

        Each plot shows mean sensitivity with standard deviation error bars.

        Returns:
            List of paths to generated figures
        """
        generated_files = []
        col_map = {'power': 'power_dbm', 'f_dev': 'f_dev_khz'}

        for param_key in ['power', 'f_dev']:
            if param_key not in self.varied_params:
                continue

            x_col = col_map[param_key]

            fig, ax = plt.subplots(figsize=(10, 6))

            # Group by parameter and calculate mean/std across other parameters
            grouped = self.df.groupby(x_col)['sensitivity_nT_rtHz'].agg(['mean', 'std', 'count'])
            x_vals_raw = grouped.index.values
            y_mean = grouped['mean'].values
            y_std = grouped['std'].values

            # Convert x values to display units
            if param_key == 'power':
                x_vals_display = self._dbm_to_if_amplitude(x_vals_raw)
                x_label = 'IF Amplitude [V]'
                title = 'Sensitivity vs IF Amplitude (averaged over f_dev)'
            else:
                x_vals_display = x_vals_raw
                x_label = 'FM Deviation [kHz]'
                title = 'Sensitivity vs FM Deviation (averaged over IF amplitude)'

            # Plot with error bars
            ax.errorbar(x_vals_display, y_mean, yerr=y_std, fmt='o-', capsize=5,
                        markersize=8, linewidth=2, color='steelblue',
                        label='Mean ± Std', ecolor='gray', elinewidth=1.5)

            # Mark the minimum point
            min_idx = np.argmin(y_mean)
            ax.scatter([x_vals_display[min_idx]], [y_mean[min_idx]], marker='*',
                       s=300, c='red', edgecolors='white', linewidths=2, zorder=10,
                       label=f'Best: {y_mean[min_idx]:.3f} nT/√Hz')

            # Formatting
            ax.set_xlabel(x_label)
            ax.set_ylabel(r'Sensitivity [nT/$\sqrt{\mathrm{Hz}}$]')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')

            # Format x-axis for power (IF amplitude)
            if param_key == 'power':
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

            fig.tight_layout()

            filepath = os.path.join(self.plots_folder, f'sensitivity_vs_{param_key}_averaged.pdf')
            fig.savefig(filepath, dpi=150)
            plt.close(fig)

            generated_files.append(filepath)
            self.log.debug(f'Saved averaged sensitivity vs {param_key}: {filepath}')

        return generated_files

    def _plot_odmr_center_vs_parameters(self) -> Optional[str]:
        """
        Plot ODMR center frequency vs swept parameters.

        Returns:
            Path to saved figure, or None if plot couldn't be generated
        """
        if 'odmr_center_hz' not in self.df.columns:
            return None

        col_map = {'power': 'power_dbm', 'f_mod': 'f_mod_hz', 'f_dev': 'f_dev_khz'}

        # Create subplots for each varied parameter
        n_params = len(self.varied_params)
        if n_params == 0:
            return None

        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5))
        if n_params == 1:
            axes = [axes]

        for ax, param_key in zip(axes, self.varied_params):
            x_col = col_map[param_key]

            # Group by this parameter and calculate mean ODMR center
            grouped = self.df.groupby(x_col)['odmr_center_hz'].agg(['mean', 'std'])
            x_vals = grouped.index.values
            y_vals = grouped['mean'].values / 1e6  # Convert to MHz
            y_err = grouped['std'].values / 1e3  # Convert to kHz for error bars

            ax.errorbar(x_vals, y_vals, yerr=y_err / 1e3, fmt='o-',
                        capsize=4, markersize=8, linewidth=2)

            # Labels
            x_label_map = {'power_dbm': 'Power [dBm]', 'f_mod_hz': 'Mod. Freq. [Hz]',
                           'f_dev_khz': 'FM Deviation [kHz]'}
            ax.set_xlabel(x_label_map.get(x_col, x_col))
            ax.set_ylabel('ODMR Center [MHz]')
            ax.grid(True, alpha=0.3)

        fig.suptitle('ODMR Center Frequency vs Parameters', fontsize=12)
        fig.tight_layout()

        # Save
        filepath = os.path.join(self.plots_folder, 'odmr_center_vs_parameters.pdf')
        fig.savefig(filepath, dpi=150)
        plt.close(fig)

        self.log.debug(f'Saved ODMR center vs parameters: {filepath}')
        return filepath

    def _plot_linewidth_vs_parameters(self) -> Optional[str]:
        """
        Plot ODMR linewidth vs swept parameters.

        Returns:
            Path to saved figure, or None if plot couldn't be generated
        """
        if 'linewidth_hz' not in self.df.columns:
            return None

        col_map = {'power': 'power_dbm', 'f_mod': 'f_mod_hz', 'f_dev': 'f_dev_khz'}

        n_params = len(self.varied_params)
        if n_params == 0:
            return None

        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5))
        if n_params == 1:
            axes = [axes]

        for ax, param_key in zip(axes, self.varied_params):
            x_col = col_map[param_key]

            # Group by this parameter
            grouped = self.df.groupby(x_col)['linewidth_hz'].agg(['mean', 'std'])
            x_vals = grouped.index.values
            y_vals = grouped['mean'].values / 1e3  # Convert to kHz
            y_err = grouped['std'].values / 1e3

            ax.errorbar(x_vals, y_vals, yerr=y_err, fmt='o-',
                        capsize=4, markersize=8, linewidth=2, color='orange')

            # Labels
            x_label_map = {'power_dbm': 'Power [dBm]', 'f_mod_hz': 'Mod. Freq. [Hz]',
                           'f_dev_khz': 'FM Deviation [kHz]'}
            ax.set_xlabel(x_label_map.get(x_col, x_col))
            ax.set_ylabel('Linewidth [kHz]')
            ax.grid(True, alpha=0.3)

        fig.suptitle('ODMR Linewidth vs Parameters', fontsize=12)
        fig.tight_layout()

        # Save
        filepath = os.path.join(self.plots_folder, 'linewidth_vs_parameters.pdf')
        fig.savefig(filepath, dpi=150)
        plt.close(fig)

        self.log.debug(f'Saved linewidth vs parameters: {filepath}')
        return filepath

    def _plot_summary_statistics(self) -> Optional[str]:
        """
        Create a summary dashboard with key metrics and visualizations.

        Layout:
        Row 1: Sens vs f_dev | Sens vs IF amp | Mini heatmap
        Row 2: Summary table | Best ASD plot

        Returns:
            Path to saved figure, or None if plot couldn't be generated
        """
        # Get valid sensitivity data
        sens = self.df['sensitivity_nT_rtHz'].values
        valid_sens = sens[~np.isnan(sens)]

        if len(valid_sens) == 0:
            return None

        # Create figure with GridSpec for flexible layout
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

        # Row 1, Col 0: Sensitivity vs f_dev with error bars
        ax_fdev = fig.add_subplot(gs[0, 0])
        self._plot_averaged_subplot(ax_fdev, 'f_dev')

        # Row 1, Col 1: Sensitivity vs IF amplitude with error bars
        ax_power = fig.add_subplot(gs[0, 1])
        self._plot_averaged_subplot(ax_power, 'power')

        # Row 1, Col 2: Mini heatmap
        ax_heatmap = fig.add_subplot(gs[0, 2])
        self._plot_mini_heatmap(ax_heatmap)

        # Row 2, Col 0: Summary table
        ax_table = fig.add_subplot(gs[1, 0])
        self._plot_summary_table(ax_table)

        # Row 2, Col 1-2: Best ASD plot (spans 2 columns)
        ax_asd = fig.add_subplot(gs[1, 1:])
        self._plot_best_asd(ax_asd)

        fig.suptitle('Sensitivity Sweep Summary', fontsize=14, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # Save
        filepath = os.path.join(self.plots_folder, 'sweep_summary.pdf')
        fig.savefig(filepath, dpi=150)
        plt.close(fig)

        self.log.debug(f'Saved summary statistics: {filepath}')
        return filepath

    def _plot_averaged_subplot(self, ax, param_key: str):
        """
        Plot averaged sensitivity vs parameter on given axes.

        Args:
            ax: Matplotlib axes to plot on
            param_key: 'power' or 'f_dev'
        """
        col_map = {'power': 'power_dbm', 'f_dev': 'f_dev_khz'}

        if param_key not in col_map or param_key not in self.varied_params:
            ax.text(0.5, 0.5, f'No {param_key} variation', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(f'Sensitivity vs {param_key}')
            return

        x_col = col_map[param_key]
        grouped = self.df.groupby(x_col)['sensitivity_nT_rtHz'].agg(['mean', 'std'])

        # Convert x values for display
        if param_key == 'power':
            x_display = self._dbm_to_if_amplitude(grouped.index.values)
            x_label = 'IF Amplitude [V]'
            title = 'Sensitivity vs IF Amplitude'
        else:
            x_display = grouped.index.values
            x_label = 'f_dev [kHz]'
            title = 'Sensitivity vs FM Deviation'

        ax.errorbar(x_display, grouped['mean'], yerr=grouped['std'],
                    fmt='o-', capsize=3, markersize=6, linewidth=1.5,
                    color='steelblue', ecolor='gray')

        # Mark minimum
        min_idx = np.argmin(grouped['mean'].values)
        ax.scatter([x_display[min_idx]], [grouped['mean'].values[min_idx]],
                   marker='*', s=150, c='red', edgecolors='white', linewidths=1, zorder=10)

        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(r'Sens [nT/$\sqrt{Hz}$]', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        if param_key == 'power':
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

    def _plot_mini_heatmap(self, ax):
        """
        Plot compact sensitivity heatmap on given axes.

        Args:
            ax: Matplotlib axes to plot on
        """
        # Need at least 2 varied parameters for heatmap
        if len(self.varied_params) < 2:
            ax.text(0.5, 0.5, 'Need 2 parameters\nfor heatmap', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10)
            ax.set_title('Sensitivity Heatmap')
            return

        # Determine parameters (prefer power and f_dev)
        param_priority = ['power', 'f_dev', 'f_mod']
        x_param, y_param = None, None
        for param in param_priority:
            if param in self.varied_params:
                if x_param is None:
                    x_param = param
                elif y_param is None:
                    y_param = param
                    break

        col_map = {'power': 'power_dbm', 'f_mod': 'f_mod_hz', 'f_dev': 'f_dev_khz'}
        x_col, y_col = col_map[x_param], col_map[y_param]

        # Get unique values
        x_vals_raw = np.sort(self.df[x_col].unique())
        y_vals_raw = np.sort(self.df[y_col].unique())

        # Convert to display units
        if x_param == 'power':
            x_vals_display = self._dbm_to_if_amplitude(x_vals_raw)
            x_label = 'IF Amp [V]'
        else:
            x_vals_display = x_vals_raw
            x_label = 'f_dev [kHz]' if x_param == 'f_dev' else 'f_mod [kHz]'

        if y_param == 'power':
            y_vals_display = self._dbm_to_if_amplitude(y_vals_raw)
            y_label = 'IF Amp [V]'
        else:
            y_vals_display = y_vals_raw
            y_label = 'f_dev [kHz]' if y_param == 'f_dev' else 'f_mod [kHz]'

        # Create sensitivity matrix
        sens_matrix = np.full((len(y_vals_raw), len(x_vals_raw)), np.nan)
        for i, y_val in enumerate(y_vals_raw):
            for j, x_val in enumerate(x_vals_raw):
                mask = (self.df[x_col] == x_val) & (self.df[y_col] == y_val)
                if mask.any():
                    sens_matrix[i, j] = self.df.loc[mask, 'sensitivity_nT_rtHz'].values[0]

        # Use LogNorm
        valid_sens = sens_matrix[~np.isnan(sens_matrix)]
        if len(valid_sens) > 0 and valid_sens.min() > 0:
            norm = LogNorm(vmin=max(valid_sens.min() * 0.9, 0.01),
                          vmax=valid_sens.max() * 1.1)
        else:
            norm = None

        im = ax.imshow(sens_matrix, aspect='auto', origin='lower',
                       extent=[x_vals_display.min(), x_vals_display.max(),
                               y_vals_display.min(), y_vals_display.max()],
                       cmap='viridis_r', norm=norm)

        # Mark best point
        best_idx = self.df['sensitivity_nT_rtHz'].idxmin()
        best_x_raw = self.df.loc[best_idx, x_col]
        best_y_raw = self.df.loc[best_idx, y_col]
        best_x = self._dbm_to_if_amplitude(best_x_raw) if x_param == 'power' else best_x_raw
        best_y = self._dbm_to_if_amplitude(best_y_raw) if y_param == 'power' else best_y_raw

        ax.scatter([best_x], [best_y], marker='*', s=150, c='red',
                   edgecolors='white', linewidths=1, zorder=5)

        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title('Sensitivity Heatmap', fontsize=10)
        ax.tick_params(labelsize=8)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(r'nT/$\sqrt{Hz}$', fontsize=8)

    def _plot_summary_table(self, ax):
        """
        Plot summary table with best parameters and statistics.

        Args:
            ax: Matplotlib axes to plot on
        """
        ax.axis('off')

        # Get best sensitivity parameters
        best_params = self.metadata.get('best_parameters', {})
        best_sens = self.metadata.get('best_sensitivity_nT_rtHz', np.nan)

        # Get best sensitivity row from dataframe for more details
        best_idx = self.df['sensitivity_nT_rtHz'].idxmin()
        best_row = self.df.loc[best_idx]

        # Convert power to IF amplitude
        best_power_dbm = best_params.get('power', best_row.get('power_dbm', np.nan))
        best_if_amp = self._dbm_to_if_amplitude(best_power_dbm) if not np.isnan(best_power_dbm) else np.nan

        # Get minimum linewidth and its parameters
        min_lw_if_amp = np.nan
        min_linewidth = np.nan
        min_lw_f_dev = np.nan
        if 'linewidth_hz' in self.df.columns:
            lw_valid = self.df['linewidth_hz'].dropna()
            if len(lw_valid) > 0:
                min_lw_idx = self.df['linewidth_hz'].idxmin()
                min_linewidth = self.df.loc[min_lw_idx, 'linewidth_hz'] / 1e3  # kHz
                min_lw_power_dbm = self.df.loc[min_lw_idx, 'power_dbm']
                min_lw_if_amp = self._dbm_to_if_amplitude(min_lw_power_dbm)
                min_lw_f_dev = self.df.loc[min_lw_idx, 'f_dev_khz']

        # Build table text
        lines = [
            "═" * 35,
            "     MINIMUM SENSITIVITY",
            "═" * 35,
            f"  Sensitivity:  {best_sens:.4f} nT/√Hz",
            f"  IF Amplitude: {best_if_amp:.3f} V",
            f"  f_mod:        {best_params.get('f_mod', np.nan)/1e3:.1f} kHz",
            f"  f_dev:        {best_params.get('f_dev', np.nan):.1f} kHz",
            "",
            "═" * 35,
            "     MINIMUM LINEWIDTH",
            "═" * 35,
            f"  Linewidth:    {min_linewidth:.2f} kHz",
            f"  IF Amplitude: {min_lw_if_amp:.3f} V",
            f"  f_dev:        {min_lw_f_dev:.1f} kHz",
            "",
            "─" * 35,
            f"  Total measurements: {len(self.df)}",
        ]

        ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def _plot_best_asd(self, ax):
        """
        Plot ASD for the best sensitivity measurement.

        Searches for the ASD CSV file matching the best measurement parameters
        and plots the amplitude spectral density data.

        Args:
            ax: Matplotlib axes to plot on
        """
        # Get best measurement parameters
        best_idx = self.df['sensitivity_nT_rtHz'].idxmin()
        best_row = self.df.loc[best_idx]

        power_dbm = best_row['power_dbm']
        f_mod_hz = best_row.get('f_mod_hz', 0)
        f_dev_khz = best_row['f_dev_khz']
        best_sens = best_row['sensitivity_nT_rtHz']

        # Construct filename pattern (without timestamp - need to search)
        power_str = f'n{abs(power_dbm):.2f}' if power_dbm < 0 else f'{power_dbm:.2f}'
        pattern = f'P_{power_str}dBm_fmod_{f_mod_hz/1e3:.1f}k_fdev_{f_dev_khz:.1f}k_*_ON-resonant_ASD.csv'

        # Search for matching file
        matches = glob.glob(os.path.join(self.output_folder, pattern))

        if not matches:
            ax.text(0.5, 0.5, f'ASD data not found\n\nSearched for:\n{pattern}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title('ASD for Best Measurement')
            self.log.warning(f'Could not find ASD file matching pattern: {pattern}')
            return

        # Use first match
        asd_file = matches[0]

        try:
            asd_df = pd.read_csv(asd_file, sep='\t')

            frequencies = asd_df['frequencies'].values
            asd_hanning = asd_df['asd_hanning'].values

            # Plot ASD (skip DC component at index 0)
            ax.loglog(frequencies[1:], asd_hanning[1:], 'k-', linewidth=1,
                      label='ASD (Hanning)')

            # Add sensitivity line
            ax.axhline(best_sens, color='r', linestyle='--', linewidth=2,
                       label=f'Sensitivity: {best_sens:.3f} nT/√Hz')

            # Formatting
            ax.set_xlabel('Frequency [Hz]', fontsize=10)
            ax.set_ylabel(r'ASD [nT/$\sqrt{Hz}$]', fontsize=10)

            if_amp = self._dbm_to_if_amplitude(power_dbm)
            ax.set_title(f'ASD for Best Measurement (IF={if_amp:.2f}V, f_dev={f_dev_khz:.0f}kHz)',
                         fontsize=10)

            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.3, which='both')
            ax.set_xlim([1, frequencies.max()])
            ax.set_ylim([best_sens * 0.1, asd_hanning[1:].max() * 2])

        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading ASD:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title('ASD for Best Measurement')
            self.log.error(f'Error loading ASD file {asd_file}: {e}')


def generate_sweep_visualizations(results_csv_path: str, metadata_json_path: str,
                                   output_folder: Optional[str] = None,
                                   logger: Optional[logging.Logger] = None) -> List[str]:
    """
    Convenience function to generate all visualizations from file paths.

    Args:
        results_csv_path: Path to parameter_sweep_summary.csv
        metadata_json_path: Path to sweep_metadata.json
        output_folder: Optional output folder (defaults to same as CSV file)
        logger: Optional logger instance

    Returns:
        List of paths to generated plot files
    """
    import json

    # Load data
    df = pd.read_csv(results_csv_path, sep='\t')

    with open(metadata_json_path, 'r') as f:
        metadata = json.load(f)

    # Determine output folder
    if output_folder is None:
        output_folder = os.path.dirname(results_csv_path)

    # Create visualizer and generate plots
    visualizer = SensitivitySweepVisualizer(df, metadata, output_folder, logger)
    return visualizer.generate_all_plots()
