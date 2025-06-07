# -*- coding: utf-8 -*-
import sys

import matplotlib
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSizePolicy, QFormLayout, QComboBox
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from BrainFusion.pipeLine.pipeLine import short_time_Fourier_transform

matplotlib.use('QtAgg')


class TimeFrequencyViewer(QMainWindow):
    """Application for visualizing time-frequency analysis of electrophysiological data."""

    def __init__(self, data, parent=None):
        """
        Initialize time-frequency visualization application.

        :param data: Time-frequency analysis results
        :type data: dict
        :param parent: Parent widget
        :type parent: QWidget
        """
        super().__init__(parent)
        self.data = data
        self.setWindowTitle("Time Frequency Viewer")
        self.current_colorbar = None
        self.init_ui()

    def init_ui(self):
        """Initialize user interface components."""
        # Create channel selection dropdown
        self.combox_channel = QComboBox()
        self.combox_channel.setFixedWidth(120)
        self.combox_channel.addItems(self.data['ch_names'])
        self.combox_channel.currentIndexChanged.connect(self.on_state_changed)

        # Layout for controls
        top_hlayout = QFormLayout()
        top_hlayout.addRow("Channel: ", self.combox_channel)

        # Create visualization canvas
        self.fig, self.axes = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(top_hlayout)
        layout.addStretch(1)
        layout.addWidget(self.canvas)
        layout.addStretch(1)

        # Central widget
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Show initial visualization
        self.plot(0)

    def plot(self, channel_index):
        """
        Render time-frequency visualization for selected channel.

        :param channel_index: Index of channel to visualize
        :type channel_index: int
        """
        # Clear previous visualization
        if self.current_colorbar:
            self.current_colorbar.remove()
            self.current_colorbar = None
        self.axes.clear()

        # Extract data for channel
        frequencies, times, power = self.data['data'][channel_index]
        frequencies = frequencies[0]
        times = times[0]

        # Create spectrogram visualization
        image = self.axes.pcolormesh(times, frequencies, 10 * np.log10(power), shading='gouraud')
        self.axes.set_ylabel('Frequency [Hz]')
        self.axes.set_xlabel('Time [sec]')
        self.axes.set_title(f'STFT of Channel {channel_index}')

        # Add colorbar
        self.current_colorbar = self.fig.colorbar(image, ax=self.axes, label='Power/Frequency (dB/Hz)')
        self.canvas.draw()

    def on_state_changed(self):
        """
        Handle channel selection change. Updates visualization when user selects a different channel.
        """
        index = self.combox_channel.currentIndex()
        self.plot(index)


if __name__ == '__main__':
    # Create and configure application
    app = QApplication(sys.argv)

    # Generate sample data for demonstration
    n_channels = 32
    sample_rate = 1000  # Samples per second
    duration = 10  # Seconds
    n_samples = sample_rate * duration

    # Simulate electrophysiological data
    np.random.seed(42)
    sample_data = np.random.randn(n_channels, n_samples)

    # Create data structure
    simulated_data = {
        'data': sample_data,
        'srate': sample_rate,
        'nchan': n_channels,
        'ch_names': [f'Channel_{i}' for i in range(n_channels)],
        'events': [],
        'montage': 'standard_1020'
    }

    # Perform time-frequency analysis
    segment_length = 256
    segment_overlap = 128
    stft_result = short_time_Fourier_transform(
        simulated_data, segment_length, segment_overlap, window_method='hamming'
    )

    # Launch application
    viewer = TimeFrequencyViewer(stft_result)
    viewer.show()
    sys.exit(app.exec_())