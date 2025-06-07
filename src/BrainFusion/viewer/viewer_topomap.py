# -*- coding: utf-8 -*-
import os

import matplotlib
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtCore import Qt, QEvent, pyqtSignal, QAbstractTableModel
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QFileDialog, QDialog, QSizePolicy, QVBoxLayout, QPushButton, QHBoxLayout, QCheckBox, \
    QLineEdit, QLabel, QWidget, QMainWindow, QFormLayout, QApplication, QMessageBox, QListWidget, QListWidgetItem, \
    QDialogButtonBox, QScrollArea, QTableView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from BrainFusion.io.File_IO import read_neuracle_bdf, read_edf, read_csv, read_txt, read_npy, read_mat, read_json, \
    read_file_by_qt
from scipy import signal
from scipy.signal import iirnotch

from BrainFusion.utils.channels import drop_channels
from BrainFusion.utils.normalize import min_max_scaling_to_range, min_max_scaling_by_arrays
from UI.ui_component import BFPushButton

matplotlib.use('QtAgg')

import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QSizePolicy)

from pyqtgraph import PlotWidget, ScatterPlotItem, mkPen, InfiniteLine, TextItem
from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy


class ChannelsDialog(QDialog):
    """Dialog interface for selecting channels to exclude from visualization."""

    def __init__(self, channel_list, parent=None):
        """
        Initialize channel exclusion dialog.

        :param channel_list: List of channel names
        :type channel_list: list[str]
        :param parent: Parent widget
        :type parent: QWidget
        """
        super().__init__(parent)
        self.setWindowTitle('Exclude Channels')
        self.setGeometry(400, 400, 300, 800)
        self.init_ui(channel_list)

    def init_ui(self, channel_list):
        """Initialize user interface components."""
        # Create scrollable area for channel selection
        central_widget = QWidget()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(central_widget)
        self.scroll_area.setWidgetResizable(True)

        # Main layout configuration
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.scroll_area)
        layout = QVBoxLayout(central_widget)

        # Create checkboxes for each channel
        self.checkbox_dict = {}
        for channel in channel_list:
            checkbox = QCheckBox(channel)
            self.checkbox_dict[channel] = checkbox
            layout.addWidget(checkbox)

        # Add dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        # Connect button signals
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

    def get_selected_channels(self):
        """
        Retrieve names of selected channels for exclusion.

        :return: List of channel names to exclude
        :rtype: list[str]
        """
        return [channel for channel, checkbox in self.checkbox_dict.items() if checkbox.isChecked()]


class TopomapViewer(QDialog):
    """Application for visualizing topographical maps of EEG data."""

    def __init__(self, data, parent=None):
        """
        Initialize topographical map visualization application.

        :param data: EEG data to visualize
        :type data: dict
        :param parent: Parent widget
        :type parent: QWidget
        """
        super(TopomapViewer, self).__init__(parent)
        self.is_show_sensor = False
        self.setWindowTitle("Topomap Viewer")
        self.setGeometry(100, 100, 1000, 400)
        self.data = data
        self.is_relative = True
        self.init_ui()
        self.plot(type=self.data['type'])

    def init_ui(self):
        """Initialize user interface components."""
        # Prepare data for visualization
        self.show_data = np.array([self.data['feature'][key] for key in self.data['feature'].keys()]).T
        self.show_channel_names = self.data['ch_names']
        num_fig = self.show_data.shape[1]

        # Create Matplotlib figure
        self.fig, self.axes = plt.subplots(1, num_fig, figsize=(8, 6))
        self.fig.subplots_adjust(hspace=0, wspace=0.05, bottom=0.08, left=0.05, top=0.98, right=0.98)

        # Create Matplotlib canvas
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create visualization titles
        self.title_list = None
        if self.data['type'] == 'eeg_psd':
            self.title_list = ['Delta band', 'Theta band', 'Alpha band', 'Beta band', 'Gamma band']
        elif self.data['type'] == 'eeg_microstate':
            self.title_list = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

        # Create table widget for numerical data
        self.table_widget = TableWidget(self.show_data, self.show_channel_names, self.title_list)

        # Create control buttons
        self.bnt_save = BFPushButton('Save')
        self.bnt_save.setFixedWidth(150)
        self.bnt_setting = BFPushButton('Settings')
        self.bnt_setting.setFixedWidth(150)
        self.bnt_layout = QHBoxLayout()
        self.bnt_layout.addStretch(1)
        self.bnt_layout.addWidget(self.bnt_save)
        self.bnt_layout.addWidget(self.bnt_setting)

        # Create bottom control panel
        bottom_layout = QHBoxLayout()
        self.relative_checkbox = QCheckBox('Relative')
        self.relative_checkbox.setChecked(True)
        self.relative_checkbox.stateChanged.connect(self.set_relative)
        self.sensor_checkbox = QCheckBox('Sensors')
        self.sensor_checkbox.setChecked(False)
        self.sensor_checkbox.stateChanged.connect(self.set_sensor)
        self.bnt_select_channel = QPushButton('Excluded Channels')
        self.bnt_select_channel.clicked.connect(self.show_remove_bad_channels_dialog)
        self.lineedit_excluded_channels = QLineEdit()
        bottom_layout.addWidget(self.relative_checkbox)
        bottom_layout.addSpacing(20)
        bottom_layout.addWidget(self.sensor_checkbox)
        bottom_layout.addSpacing(20)
        bottom_layout.addWidget(self.bnt_select_channel)
        bottom_layout.addSpacing(10)
        bottom_layout.addWidget(self.lineedit_excluded_channels)
        bottom_layout.addSpacing(20)
        bottom_layout.addStretch(1)

        # Configure main layout
        layout = QVBoxLayout(self)
        layout.addLayout(self.bnt_layout)
        layout.addWidget(self.canvas)
        layout.addWidget(self.table_widget)
        layout.addLayout(bottom_layout)

    def show_remove_bad_channels_dialog(self):
        """Show channel exclusion dialog and update visualization."""
        if self.data is not None:
            dialog = ChannelsDialog(self.data['ch_names'], self)
            result = dialog.exec_()
            if result == QDialog.Accepted:
                selected_channels = dialog.get_selected_channels()
                self.lineedit_excluded_channels.setText(', '.join(selected_channels))
                bad_channels = [chan.strip() for chan in self.lineedit_excluded_channels.text().split(',') if
                                chan.strip()]
                self.show_data, self.show_channel_names = drop_channels(
                    raw_data=self.show_data,
                    channels=self.data['ch_names'],
                    bad_channels=bad_channels
                )
                self.plot(type=self.data['type'])
        else:
            QMessageBox.warning(None, 'Warning', 'Please load data first', QMessageBox.Ok)

    def set_relative(self):
        """Toggle relative scaling mode for visualization."""
        self.is_relative = self.relative_checkbox.isChecked()
        self.plot(type=self.data['type'])

    def set_sensor(self):
        """Toggle sensor visibility in visualization."""
        self.is_show_sensor = self.sensor_checkbox.isChecked()
        self.plot(type=self.data['type'])

    def plot(self, type='eeg_psd'):
        """
        Render topographical map visualization.

        :param type: Type of visualization to create
        :type type: str
        """
        # Set figure title based on visualization type
        if type == 'eeg_psd':
            self.fig.suptitle("EEG Power Spectral Density")
        elif type == 'eeg_microstate':
            self.fig.suptitle("EEG Microstate")

        if self.data:
            # Prepare EEG montage
            montage = mne.channels.make_standard_montage('standard_1005')
            info = mne.create_info(ch_names=self.show_channel_names[:30], sfreq=1000, ch_types='eeg')

            # Prepare normalized data
            if self.is_relative:
                norm_data = min_max_scaling_to_range(self.show_data.T[:, :30])
                data_range = (-1, 1)
            else:
                norm_data = min_max_scaling_by_arrays(self.show_data.T[:, :30])
                data_range = (-1, 1)

            # Create individual plots
            for i, feature_data in enumerate(norm_data):
                evoked = mne.EvokedArray(data=self.show_data[:30, :], info=info)
                evoked.set_montage(montage)
                self.axes[i].clear()

                # Set subplot title if available
                if self.title_list:
                    self.axes[i].set_title(self.title_list[i])

                # Create topographical visualization
                mne.viz.plot_topomap(
                    feature_data,
                    evoked.info,
                    axes=self.axes[i],
                    show=False,
                    sensors=self.is_show_sensor,
                    vlim=data_range,
                    names=self.show_channel_names if self.is_show_sensor else None
                )

                # Update canvas
                self.axes[i].figure.canvas.draw()


class PandasModel(QAbstractTableModel):
    """Table model for displaying pandas DataFrame in QtTableView."""

    def __init__(self, df=pd.DataFrame(), parent=None):
        """
        Initialize pandas table model.

        :param df: Data to display in table
        :type df: pd.DataFrame
        :param parent: Parent widget
        :type parent: QWidget
        """
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent=None):
        """
        Return number of rows in table.

        :return: Number of rows
        :rtype: int
        """
        return self._df.shape[0]

    def columnCount(self, parent=None):
        """
        Return number of columns in table.

        :return: Number of columns
        :rtype: int
        """
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        """
        Retrieve data for specified index and role.

        :param index: Index of data to retrieve
        :type index: QModelIndex
        :param role: Data role to retrieve
        :type role: int
        :return: Requested data value
        :rtype: str
        """
        if index.isValid() and role == Qt.DisplayRole:
            return str(self._df.iloc[index.row(), index.column()])
        elif index.isValid() and role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """
        Retrieve header data for table.

        :param section: Header section index
        :type section: int
        :param orientation: Header orientation (horizontal/vertical)
        :type orientation: Qt.Orientation
        :param role: Data role to retrieve
        :type role: int
        :return: Header data
        :rtype: str
        """
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._df.columns[section]
            elif orientation == Qt.Vertical:
                return self._df.index[section]
        return None


class TableWidget(QMainWindow):
    """Widget for displaying EEG data in table format."""

    def __init__(self, data, ch_names, columns):
        """
        Initialize data table widget.

        :param data: Data to display
        :type data: np.ndarray
        :param ch_names: Channel names for row labels
        :type ch_names: list[str]
        :param columns: Column names for data
        :type columns: list[str]
        """
        super().__init__()
        self.setGeometry(100, 100, 800, 600)
        self.init_ui(data, ch_names, columns)

    def init_ui(self, data, ch_names, columns):
        """Initialize user interface components."""
        # Prepare DataFrame from input data
        df = pd.DataFrame(data, index=ch_names, columns=columns)
        self.model = PandasModel(df)

        # Create table view
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.horizontalHeader().setDefaultSectionSize(100)
        self.table.resizeColumnsToContents()

        # Create UI controls
        self.checkbox = QCheckBox("Show Table")
        self.checkbox.setChecked(True)
        self.checkbox.stateChanged.connect(self.toggle_table)
        self.lineedit = QLineEdit()
        self.lineedit.setText('100')
        self.lineedit.returnPressed.connect(self.adjust_column_width)
        self.label = QLabel("Column Width:")

        # Configure layouts
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.label)
        h_layout.addWidget(self.lineedit)

        layout = QVBoxLayout()
        layout.addWidget(self.checkbox)
        layout.addLayout(h_layout)
        layout.addWidget(self.table)

        # Set central widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def toggle_table(self, state):
        """
        Toggle table visibility.

        :param state: Current checkbox state
        :type state: int
        """
        self.table.setVisible(state == Qt.Checked)

    def adjust_column_width(self):
        """Adjust column widths based on user input."""
        try:
            width = int(self.lineedit.text())
            if width > 0:
                self.table.horizontalHeader().setDefaultSectionSize(width)
        except ValueError:
            pass  # Ignore invalid input