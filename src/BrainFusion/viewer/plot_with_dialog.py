# -*- coding: utf-8 -*-
# @Time    : 2024/3/4 17:49
# @Author  : Li WenHao
# @Site    : South China University of Technology
# @File    : plot_with_dialog.py
# @Software: PyCharm 
# @Comment :
import os

import matplotlib
import mne
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QFileDialog, QDialog, QSizePolicy, QVBoxLayout, QPushButton, QHBoxLayout, QCheckBox, \
    QLineEdit, QLabel, QWidget, QMainWindow, QFormLayout, QApplication, QMessageBox, QListWidget, QListWidgetItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from BrainFusion.io.File_IO import read_neuracle_bdf, read_edf, read_csv, read_txt, read_npy, read_mat, read_json, \
    read_file_by_qt

matplotlib.use('QtAgg')

import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QSizePolicy)

from pyqtgraph import PlotWidget, ScatterPlotItem, mkPen, InfiniteLine
from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy


class RawCurvePlotDialog(QMainWindow):
    """
    Dialog for visualizing and annotating raw EEG data signals.

    :param data: EEG data dictionary containing raw signals and metadata
    :type data: dict
    :param filePath: Path to the EEG data file
    :type filePath: str
    :param parent: Parent widget container
    :type parent: QWidget, optional
    """

    def __init__(self, data, filePath, parent=None):
        super(RawCurvePlotDialog, self).__init__(parent)
        self.setWindowTitle("Raw Curve Plot Dialog")
        self.setGeometry(100, 100, 800, 600)
        self.data = np.array(data['data'])
        self.nchan = data['nchan']
        self.srate = data['srate']
        self.ch_names = data['ch_names']
        self.filePath = self.modify_file_path(filePath)  # File path for annotations

        # Data navigation parameters
        self.total_samples = self.data.shape[1]  # Total data points in EEG recording
        self.current_page = 0  # Current page index
        self.num_time_per_page = 5  # Page duration in seconds
        self.num_samples_per_page = int(self.num_time_per_page * self.srate)

        # Time axis parameters
        self.sample_interval = 1 / self.srate  # Time per sample
        self.time_values = np.arange(0, self.total_samples) * self.sample_interval  # Time vector

        self.central_widget = PlotWidget()
        self.setCentralWidget(self.central_widget)

        # List for per-channel plot widgets
        self.plot_widgets = []
        for i in range(self.nchan):
            plot_widget = PlotWidget()
            plot_widget.setMinimumHeight(20)
            plot_widget.setMouseEnabled(x=False, y=False)  # Disable mouse interaction
            plot_widget.wheelEvent = self.wheelEvent
            plot_widget.mousePressEvent = lambda event, idx=i: self.customMousePressEvent(event, idx)
            self.plot_widgets.append(plot_widget)

        # UI controls
        self.button_prev = QPushButton('Previous')
        self.button_prev.setFixedWidth(200)
        self.button_next = QPushButton('Next')
        self.button_next.setFixedWidth(200)
        self.button_prev.clicked.connect(self.on_prev_click)
        self.button_next.clicked.connect(self.on_next_click)

        self.num_samples_per_page_lineedit = QLineEdit(str(self.num_time_per_page))
        self.num_samples_per_page_lineedit.setFixedWidth(100)
        self.num_samples_per_page_lineedit.setValidator(QIntValidator())  # Integer input only
        self.num_samples_per_page_lineedit.editingFinished.connect(self.on_samples_per_page_change)
        self.tip_label = QLabel('Sec/Page: ')
        self.tip_label.setFixedWidth(100)

        # Layout configuration
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.button_prev)
        h_layout.addWidget(self.button_next)
        h_layout.addSpacing(200)
        h_layout.addWidget(self.tip_label)
        h_layout.addWidget(self.num_samples_per_page_lineedit)

        # Main layout with channel plots
        main_layout = QFormLayout()
        for i, plot_widget in enumerate(self.plot_widgets):
            main_layout.addRow(self.ch_names[i], plot_widget)
            main_layout.setSpacing(0)
        main_layout.addItem(h_layout)

        # Event annotation list
        self.clicked_x_list = QListWidget()  # Event marker storage
        self.clicked_x_list.itemDoubleClicked.connect(self.remove_clicked_x)
        main_layout.addWidget(self.clicked_x_list)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Initialize annotation file
        self.create_file()
        self.plot_data(self.current_page)

    def modify_file_path(self, filePath):
        """
        Generate annotation file path from EEG data path.

        :param filePath: Original EEG data file path
        :type filePath: str
        :return: Modified annotation file path
        :rtype: str
        """
        filename, ext = os.path.splitext(filePath)
        return filename + "_event.tsv"

    def create_file(self):
        """Create annotation file or load existing data."""
        if os.path.exists(self.filePath):
            self.load_data_from_file()
        else:
            with open(self.filePath, 'w') as f:
                f.write('')

    def save_data_to_file(self):
        """Save event markers to annotation file."""
        with open(self.filePath, 'w') as f:
            for i in range(self.clicked_x_list.count()):
                item = self.clicked_x_list.item(i)
                f.write(item.text() + '\n')

    def load_data_from_file(self):
        """Load existing event markers from annotation file."""
        with open(self.filePath, 'r') as f:
            for line in f:
                self.clicked_x_list.addItem(line.strip())

    def plot_data(self, page):
        """
        Render EEG data for the specified page.

        :param page: Page index to display
        :type page: int
        """
        start_index = page * self.num_samples_per_page
        end_index = min(start_index + self.num_samples_per_page, self.total_samples)

        for i in range(self.nchan):
            plot_widget = self.plot_widgets[i]
            plot_widget.clear()
            plot_widget.setBackground('w')  # White background

            # Plot EEG signal
            plot_widget.plot(self.time_values[start_index:end_index],
                             self.data[i, start_index:end_index], pen='k')
            plot_widget.getAxis('left').setStyle(showValues=False)  # Hide y-axis labels

            # Configure axes visibility
            if i < self.nchan - 1:
                plot_widget.getAxis('bottom').setStyle(showValues=False)
            else:
                plot_widget.getAxis('bottom').setStyle(showValues=True)  # Show x-axis on last channel

        # Draw event markers
        self.draw_green_line()
        self.draw_yellow_line()

    def on_prev_click(self):
        """Navigate to the previous data page."""
        self.current_page = max(self.current_page - 1, 0)
        self.plot_data(self.current_page)

    def on_next_click(self):
        """Navigate to the next data page."""
        max_pages = self.total_samples // self.num_samples_per_page
        self.current_page = min(self.current_page + 1, max_pages)
        self.plot_data(self.current_page)

    def on_samples_per_page_change(self):
        """Handle page duration configuration change."""
        try:
            new_value = int(self.num_samples_per_page_lineedit.text())
            if new_value > 0:
                self.num_time_per_page = new_value
                self.num_samples_per_page = int(new_value * self.srate)
                self.plot_data(self.current_page)
                return
        except ValueError:
            pass
        # Reset to previous valid value on invalid input
        self.num_samples_per_page_lineedit.setText(str(self.num_time_per_page))

    def wheelEvent(self, event):
        """Handle mouse wheel scrolling for vertical zoom."""
        delta = event.angleDelta().y() / 120
        zoom_factor = 0.8 if delta > 0 else 1.2  # Zoom in/out factor

        for plot_widget in self.plot_widgets:
            ymin, ymax = plot_widget.getViewBox().viewRange()[1]
            plot_widget.setYRange(ymin * zoom_factor, ymax * zoom_factor, padding=0)

    def customMousePressEvent(self, event, idx):
        """
        Handle mouse click events on EEG plots.

        :param event: Mouse event object
        :type event: QMouseEvent
        :param idx: Index of the channel plot
        :type idx: int
        """
        if event.button() == Qt.LeftButton:
            # Left-click adds start marker
            x = self.plot_widgets[idx].plotItem.vb.mapSceneToView(event.pos()).x()
            self.add_start_x(x)
            self.draw_green_line()
        elif event.button() == Qt.RightButton:
            # Right-click adds end marker
            x = self.plot_widgets[idx].plotItem.vb.mapSceneToView(event.pos()).x()
            self.add_end_x(x)
            self.draw_yellow_line()

    def add_start_x(self, x):
        """
        Add start marker at specified time position.

        :param x: Time position for marker
        :type x: float
        """
        item = QListWidgetItem(f"Start:{x}")
        self.clicked_x_list.addItem(item)
        self.save_data_to_file()

    def add_end_x(self, x):
        """
        Add end marker at specified time position.

        :param x: Time position for marker
        :type x: float
        """
        item = QListWidgetItem(f"End:{x}")
        self.clicked_x_list.addItem(item)
        self.save_data_to_file()

    def draw_green_line(self):
        """Draw green vertical lines for start markers."""
        for x_item in self.clicked_x_list.findItems("Start:", Qt.MatchStartsWith):
            x = float(x_item.text().split(":")[1])
            for plot_widget in self.plot_widgets:
                current_range = (self.current_page * self.num_time_per_page,
                                 (self.current_page + 1) * self.num_time_per_page)
                if current_range[0] <= x <= current_range[1]:
                    plot_widget.addItem(InfiniteLine(pos=(x, 0), angle=90, pen=mkPen('g', width=2)))

    def draw_yellow_line(self):
        """Draw yellow vertical lines for end markers."""
        for x_item in self.clicked_x_list.findItems("End:", Qt.MatchStartsWith):
            x = float(x_item.text().split(":")[1])
            for plot_widget in self.plot_widgets:
                current_range = (self.current_page * self.num_time_per_page,
                                 (self.current_page + 1) * self.num_time_per_page)
                if current_range[0] <= x <= current_range[1]:
                    plot_widget.addItem(InfiniteLine(pos=(x, 0), angle=90, pen=mkPen('y', width=2)))

    def remove_clicked_x(self, item):
        """
        Remove event marker when double-clicked.

        :param item: Event marker item to remove
        :type item: QListWidgetItem
        """
        self.clicked_x_list.takeItem(self.clicked_x_list.row(item))
        self.plot_data(self.current_page)


class EEGPSDPlotDialog(QDialog):
    """
    Dialog for visualizing EEG power spectral density (PSD) topography.

    :param data: EEG power data dictionary
    :type data: dict
    :param parent: Parent widget container
    :type parent: QWidget, optional
    """

    def __init__(self, data, parent=None):
        super(EEGPSDPlotDialog, self).__init__(parent)
        self.setWindowTitle("EEG Topomap Plot Dialog")
        self.setGeometry(100, 100, 1000, 400)
        self.data = data
        self.is_relative = True

        # Create Matplotlib figure
        num_fig = np.array(self.data['data']).shape[1]
        self.fig, self.axes = plt.subplots(1, num_fig, figsize=(8, 6), sharex=True, sharey=True)

        # Adjust subplot spacing
        self.fig.subplots_adjust(hspace=0, wspace=0.05, bottom=0.08, left=0.05, top=0.98, right=0.98)

        # Create Matplotlib canvas
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Setup UI layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        # Add relative/absolute toggle
        self.relative_checkbox = QCheckBox('Relative')
        self.relative_checkbox.setChecked(True)
        self.relative_checkbox.stateChanged.connect(self.choose_show_type)
        layout.addWidget(self.relative_checkbox)

        # Render initial visualization
        self.plot(type=self.data['type'])

    def choose_show_type(self):
        """
        Toggle between relative and absolute power visualization.
        """
        self.is_relative = self.relative_checkbox.isChecked()
        self.plot(type=self.data['type'])

    def plot(self, type='eeg_psd'):
        """
        Render EEG power topographic maps.

        :param type: Visualization type ('eeg_psd' for power, 'eeg_microstate' for microstates)
        :type type: str
        """
        title_list = None
        figure_title = ""

        # Configure titles based on visualization type
        if type == 'eeg_psd':
            title_list = ['Δ wave band', 'θ wave band', 'α wave band', 'β wave band', 'γ wave band']
            figure_title = "EEG Power Spectral Density"
        elif type == 'eeg_microstate':
            title_list = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
            figure_title = "EEG Microstate"

        self.fig.suptitle(figure_title)

        # Only plot if we have data
        if self.data:
            # Create EEG sensor layout
            montage = mne.channels.make_standard_montage('standard_1020')
            info = mne.create_info(
                ch_names=self.data['ch_names'],
                sfreq=self.data['srate'],
                ch_types=['eeg'] * len(self.data['ch_names']))

            # Normalize data based on display type
            if self.is_relative:
                norm_data = min_max_scaling_to_range(np.array(self.data['data']).T)
                data_range = (-1, 1)
            else:
                norm_data = min_max_scaling_by_arrays(np.array(self.data['data']).T)
                data_range = (-1, 1)

            # Generate topographic maps
            for i, psd in enumerate(norm_data):
                evoked = mne.EvokedArray(
                    data=np.array(self.data['data']),
                    info=info)
            evoked.set_montage(montage)

            # Configure axes
            self.axes[i].clear()
            if title_list:
                self.axes[i].set_title(title_list[i])

            # Plot topographic map
            mne.viz.plot_topomap(
                psd,
                evoked.info,
                axes=self.axes[i],
                show=False,
                sensors=True,
                vlim=data_range
            )
            self.axes[i].figure.canvas.draw()


def min_max_scaling_to_range(array, new_min=-1, new_max=1):
    """
    Normalize array values to specified range using min-max scaling.

    :param array: Input data array
    :type array: numpy.ndarray
    :param new_min: Minimum value of output range
    :type new_min: float
    :param new_max: Maximum value of output range
    :type new_max: float
    :return: Normalized array
    :rtype: numpy.ndarray
    """
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = new_min + (new_max - new_min) * (array - min_val) / (max_val - min_val)
    return normalized_array


def min_max_scaling_by_arrays(arrays, new_min=-1, new_max=1):
    """
    Normalize multiple arrays to specified range using min-max scaling.

    :param arrays: Sequence of input arrays
    :type arrays: list of numpy.ndarray
    :param new_min: Minimum value of output range
    :type new_min: float
    :param new_max: Maximum value of output range
    :type new_max: float
    :return: List of normalized arrays
    :rtype: list of numpy.ndarray
    """
    normalized_arrays = []
    for array in arrays:
        normalized_array = min_max_scaling_to_range(array, new_min, new_max)
        normalized_arrays.append(normalized_array)
    return np.array(normalized_arrays)


def plot_raw_by_file(widget, path=None):
    """
    Open raw EEG visualization dialog from file selector.

    :param widget: Parent widget for file dialog
    :type widget: QWidget
    :param path: Optional file path to bypass dialog
    :type path: str, optional
    """

    def transform_data(data):
        """Reformat fNIRS preprocessing data for visualization."""
        if data['type'] == 'fnirs_preprocessed':
            result = data['data'][0]
            result.extend(data['data'][1])
            channel = data['ch_names'][0]
            channel.extend(data['ch_names'][1])
            data['data'] = result
            data['ch_names'] = channel
        return data

    data, path = read_file_by_qt(widget, path)

    if data:
        data = transform_data(data)
        dialog = RawCurvePlotDialog(data=data, filePath=path[0], parent=widget)
        dialog.show()


def plot_eeg_psd_by_file(widget, path=None):
    """
    Open PSD visualization dialog from file selector.

    :param widget: Parent widget for file dialog
    :type widget: QWidget
    :param path: Optional file path to bypass dialog
    :type path: str, optional
    """
    data = None
    if path is None:
        # Open file dialog with supported formats
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        try:
            path, _ = QFileDialog.getOpenFileNames(
                widget,
                'Open EEG Files',
                '',
                ('All Files (*);;'
                 'BDF Files (*.bdf);;'
                 'EDF Files (*.edf);;'
                 'Text Files (*.txt);;'
                 'JSON Files (*.json);;'
                 'MAT Files (*.mat)'),
                options=options
            )
        except Exception as e:
            print(e)

        # Process selected files
        if path:
            if len(path) == 1:
                file_type = path[0].split('.')[-1].lower()
                if file_type == 'edf':
                    data = read_edf(path[0])
                elif file_type == 'csv':
                    data = read_csv(path[0])
                elif file_type == 'txt':
                    data = read_txt(path[0])
                elif file_type == 'json':
                    data = read_json(path[0])
                elif file_type == 'mat':
                    data = read_mat(path[0])
            elif len(path) == 2:
                # Handle Neuracle dual-BDF format
                if path[0].split('.')[-1].lower() == 'bdf':
                    data = read_neuracle_bdf(path, is_data_transform=True)

        # Launch dialog if we have data
        if data:
            dialog = EEGPSDPlotDialog(data=data, parent=widget)
            dialog.show()


def plot_raw(data, channel=None, sharey=False, line_color='black', linewidth=0.5):
    """
    Visualize raw EEG time series using matplotlib.

    :param data: EEG time series data
    :type data: numpy.ndarray or list
    :param channel: List of channel names
    :type channel: list[str], optional
    :param sharey: Share Y-axis across channels
    :type sharey: bool
    :param line_color: Color of EEG traces
    :type line_color: str
    :param linewidth: Width of EEG traces
    :type linewidth: float
    """
    # Determine data dimensions
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
        axes.tick_params(
            axis='both', which='both',
            bottom=False, top=False,
            labelbottom=False, left=False,
            right=False, labelleft=False
        )

        # Format axis appearance
        for spine in axes.spines.values():
            spine.set_color('lightgrey')

        # Set x-axis limits with padding
        axes.set_xlim(left=-10, right=length)
        fig.subplots_adjust(hspace=0, wspace=0, bottom=0.02, left=0.1, top=0.98, right=0.98)
        plt.show()

    # Multi-channel visualization
    elif dimensions == 2:
        data = np.array(data)
        length = data.shape[1]
        num_channels = data.shape[0]
        if channel is None:
            channel = [f'Channel {i + 1}' for i in range(num_channels)]

        fig, axes = plt.subplots(num_channels, 1, figsize=(10, 6), sharex=True, sharey=sharey)

        # Plot each channel
        for i, ax in enumerate(axes):
            ax.plot(data[i, :30000], color=line_color, linewidth=linewidth)
            ax.set_ylabel(f' {channel[i]}', rotation=0, ha='right')
            ax.tick_params(
                axis='both', which='both',
                bottom=False, top=False,
                labelbottom=False, left=False,
                right=False, labelleft=False
            )

            # Format axis appearance
            for spine in ax.spines.values():
                spine.set_color('lightgrey')

            # Set x-axis limit with left padding
            ax.set_xlim(left=-10)

        # Configure bottom axis
        axes[-1].tick_params(
            axis='both', which='both',
            bottom=True, top=False,
            left=False, right=False,
            labelbottom=True
        )
        for spine in axes[-1].spines.values():
            spine.set_color('lightgrey')

        # Adjust overall figure layout
        fig.subplots_adjust(hspace=0, wspace=0, bottom=0.05, left=0.05, top=0.98, right=0.98)
        plt.show()

    # Unsupported data format
    else:
        print(f"Unsupported data dimension: {dimensions}")
