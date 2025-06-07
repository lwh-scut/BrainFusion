# -*- coding: utf-8 -*-
"""
Module for plotting EEG raw data and power spectral density topography.

Provides functions to visualize EEG time series and frequency distributions.
"""
from tkinter import Tk, filedialog
import matplotlib
import mne
import numpy as np
import matplotlib.pyplot as plt

from BrainFusion.io.File_IO import read_neuracle_bdf, read_edf, read_csv, read_txt, read_npy, read_mat, read_json, \
    read_file, read_file_by_qt
from BrainFusion.viewer.plot_with_dialog import min_max_scaling_to_range, min_max_scaling_by_arrays

matplotlib.use('QtAgg')


def plot_raw_by_file(widget, path=None):
    """
    Plot raw EEG data from a file using a QT file dialog.

    :param widget: Parent widget for file dialog
    :type widget: QWidget
    :param path: Optional file path to bypass dialog
    :type path: str, optional
    """
    data, _ = read_file_by_qt(widget, path)
    if data:
        plot_raw(data=data['data'], channel=data['ch_names'])


def plot_raw(data, channel=None, sharey=False, line_color='black', linewidth=0.5,
             is_save=False, save_path=None):
    """
    Visualize EEG time series data for single or multiple channels.

    :param data: EEG time series data (1D vector or 2D array)
    :type data: numpy.ndarray
    :param channel: List of channel names
    :type channel: list[str], optional
    :param sharey: Share Y-axis scales across channels
    :type sharey: bool
    :param line_color: Color for signal traces
    :type line_color: str
    :param linewidth: Width of signal traces
    :type linewidth: float
    :param is_save: Flag to save the plot
    :type is_save: bool
    :param save_path: Output path for saving the plot
    :type save_path: str, optional
    """
    # Validate data dimensions and plot accordingly
    if isinstance(data, np.ndarray):
        dimensions = data.ndim
    elif isinstance(data, list):
        dimensions = len(np.array(data).shape)
    else:
        print("Unsupported data type.")
        return

    # Single channel visualization
    if dimensions == 1:
        length = np.array(data).shape[0]
        num_channels = 1
        if channel is None:
            channel = ['channel']

        fig, axes = plt.subplots(num_channels, 1, figsize=(10, 6), sharex=True, sharey=sharey)
        axes.plot(data, color=line_color, linewidth=linewidth)
        axes.set_ylabel(f' {channel[0]}', rotation=0, ha='right')
        axes.tick_params(axis='both', which='both', bottom=False, top=False,
                         labelbottom=False, left=False, right=False, labelleft=False)

        # Format axis spines
        for spine in axes.spines.values():
            spine.set_color('lightgrey')

        axes.set_xlim(left=-10, right=length)
        fig.subplots_adjust(hspace=0, wspace=0, bottom=0.02, left=0.1, top=0.98, right=0.98)

    # Multi-channel visualization
    elif dimensions == 2:
        data = np.array(data)
        length = data.shape[1]
        num_channels = data.shape[0]
        if channel is None:
            channel = [str(i) for i in range(1, num_channels + 1)]

        fig, axes = plt.subplots(num_channels, 1, figsize=(10, 6), sharex=True, sharey=sharey)

        # Plot each channel
        for i, ax in enumerate(axes):
            ax.plot(data[i, :30000], color=line_color, linewidth=linewidth)
            ax.set_ylabel(f' {channel[i]}', rotation=0, ha='right')
            ax.tick_params(axis='both', which='both', bottom=False, top=False,
                           labelbottom=False, left=False, right=False, labelleft=False)

            # Format axis spines
            for spine in ax.spines.values():
                spine.set_color('lightgrey')

            ax.set_xlim(left=-10)

        # Format bottom axis for time scale
        axes[-1].tick_params(axis='both', which='both', bottom=True, top=False,
                             left=False, right=False, labelbottom=True)
        for spine in axes[-1].spines.values():
            spine.set_color('lightgrey')

        fig.subplots_adjust(hspace=0, wspace=0, bottom=0.05, left=0.1, top=0.98, right=0.98)

    else:
        print(f"Unsupported data dimension: {dimensions}")
        return

    # Save plot if requested
    if is_save and save_path:
        plt.savefig(save_path, dpi=300)


def plot_eeg_psd(data, is_relative=False, is_norm=True, title=''):
    """
    Visualize EEG power spectral density (PSD) using topographic maps.

    :param data: EEG data dictionary with signals and metadata
    :type data: dict
    :param is_relative: Flag for relative power normalization
    :type is_relative: bool
    :param is_norm: Flag for min-max normalization
    :type is_norm: bool
    :param title: Plot title
    :type title: str
    """
    num_bands = np.array(data['data']).shape[1]
    title_list = ['Δ wave band', 'θ wave band', 'α wave band', 'β wave band', 'γ wave band']

    # Setup figure
    fig, axes = plt.subplots(1, num_bands, figsize=(8, 5), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0.05, bottom=0.08, left=0.05, top=0.88, right=0.98)
    fig.suptitle(f"{title} EEG Power Spectral Density")

    # Prepare EEG data structure
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(
        ch_names=data['ch_names'],
        sfreq=data['srate'],
        ch_types=['eeg'] * len(data['ch_names'])
    )
    evoked = mne.EvokedArray(data=np.array(data['data']), info=info)
    evoked.set_montage(montage)

    # Process data for visualization
    if is_norm:
        data_to_plot = min_max_scaling_to_range(np.array(data['data']).T) if is_relative \
            else min_max_scaling_by_arrays(np.array(data['data']).T)
        vlim = (-1, 1)
    else:
        data_to_plot = np.array(data['data']).T
        vlim = (np.min(data_to_plot), np.max(data_to_plot))

    # Create topographic maps for each frequency band
    for i, (ax, band_name) in enumerate(zip(axes, title_list)):
        ax.set_title(band_name)
        mne.viz.plot_topomap(
            data_to_plot[i],
            evoked.info,
            axes=ax,
            show=False,
            sensors=True,
            vlim=vlim
        )