# -*- coding: utf-8 -*-
"""Render battery-monitor data with a non-interactive Matplotlib backend."""

import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib

matplotlib.use('Agg')

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def render_monitor_plot(data_file, title='Battery tester') -> str:
    """Create a voltage/current PNG beside a Qudi battery-monitor data file."""
    data_path = Path(data_file).expanduser().resolve()
    data = np.genfromtxt(
        str(data_path),
        delimiter='\t',
        comments='#',
        usecols=(1, 2, 13, 16),
        dtype=float,
    )
    if data.size == 0:
        raise RuntimeError('The battery monitoring file contains no channel samples.')
    data = np.atleast_2d(data)
    valid = (
        np.isfinite(data[:, 0])
        & np.isfinite(data[:, 1])
        & (data[:, 1] >= 0)
        & (np.isfinite(data[:, 2]) | np.isfinite(data[:, 3]))
    )
    data = data[valid]
    if not len(data):
        raise RuntimeError('The battery monitoring file contains no plottable samples.')

    figure = Figure(figsize=(9, 6))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1, sharex=True)
    channel_numbers = data[:, 1].astype(int)
    for channel_id in sorted(set(channel_numbers)):
        rows = data[channel_numbers == channel_id]
        label = f'Channel {channel_id}'
        axes[0].plot(rows[:, 0], rows[:, 2], linewidth=1.2, marker='o', label=label)
        axes[1].plot(rows[:, 0], rows[:, 3] * 1000, linewidth=1.2, marker='o', label=label)

    axes[0].set_ylabel('Voltage (V)')
    axes[1].set_ylabel('Current (mA)')
    axes[1].set_xlabel('Elapsed time (s)')
    axes[0].set_title(f'{title} voltage and current')
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc='best')
    figure.tight_layout()

    plot_path = data_path.with_suffix('.png')
    figure.savefig(str(plot_path), bbox_inches='tight', pad_inches=0.05)
    return str(plot_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_file')
    parser.add_argument('--title', default='Battery tester')
    arguments = parser.parse_args()
    print(render_monitor_plot(arguments.data_file, title=arguments.title))


if __name__ == '__main__':
    main()
