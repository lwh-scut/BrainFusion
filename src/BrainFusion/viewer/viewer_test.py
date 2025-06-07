import json
import os
import sys

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
from matplotlib.figure import Figure
from scipy.stats import ttest_ind
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import GridSearchCV

from BrainFusion.io.File_IO import read_neuracle_bdf, read_edf, read_csv, read_txt, read_npy, read_mat, read_json, \
    read_file_by_qt
from scipy import signal
from scipy.signal import iirnotch

from BrainFusion.utils.channels import drop_channels
from BrainFusion.utils.normalize import min_max_scaling_to_range, min_max_scaling_by_arrays
from BrainFusion.viewer.viewer_curve import RawCurvePlotDialog
from UI.ui_component import BFPushButton
import seaborn as sns

matplotlib.use('QtAgg')

import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QSizePolicy)

from pyqtgraph import PlotWidget, ScatterPlotItem, mkPen, InfiniteLine, TextItem
from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy


class MatplotlibWidget(QWidget):
    """
    Custom widget for embedding Matplotlib plots in PyQt applications.

    Provides functionality for creating, updating, and displaying boxplots.
    """

    def __init__(self, parent=None):
        """
        Initialize Matplotlib widget.

        :param parent: Parent widget
        :type parent: QWidget
        """
        super(MatplotlibWidget, self).__init__(parent)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.plot_data()

    def plot_default(self):
        """
        Create a default boxplot with random data.
        """
        data = [np.random.randn(100) for _ in range(5)]
        self.plot(data, title="Default Boxplot", y_label="Value", x_label="Group",
                  x_ticks=["A", "B", "C", "D", "E"])

    def plot(self, data, title="", y_label="", x_label="", legend=None, x_ticks=None, y_range=None, color="blue"):
        """
        Plot boxplot visualization.

        :param data: Input data to visualize
        :type data: list[np.ndarray]
        :param title: Plot title
        :type title: str
        :param y_label: Y-axis label
        :type y_label: str
        :param x_label: X-axis label
        :type x_label: str
        :param legend: Legend labels
        :type legend: list[str]
        :param x_ticks: X-axis tick labels
        :type x_ticks: list[str]
        :param y_range: Y-axis range
        :type y_range: tuple[float]
        :param color: Boxplot color
        :type color: str
        """
        self.ax.clear()
        self.ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor=color))
        self.ax.set_title(title)
        self.ax.set_ylabel(y_label)
        self.ax.set_xlabel(x_label)

        if x_ticks:
            self.ax.set_xticklabels(x_ticks)
        if y_range:
            self.ax.set_ylim(y_range)
        if legend:
            self.ax.legend(legend)

        self.canvas.draw()

    def plot_data(self):
        """
        Generate neurovascular coupling (NVC) boxplot visualization.

        Loads experimental data, processes NVC values, creates boxplots for left/right hand motor imagery,
        and performs statistical significance testing.
        """
        # Define EEG and fNIRS channel selections
        eeg_channel_select = [['FCC5h'], ['FCC3h'], ['FCC4h'], ['FCC6h'],
                              ['CCP5h'], ['CCP3h'], ['CCP4h'], ['CCP6h']]
        fnirs_channel_select = [['S8_D9', 'S8_D10', 'S7_D10', 'S7_D9'],
                                ['S8_D11', 'S10_D11', 'S10_D10', 'S8_D10'],
                                ['S12_D13', 'S12_D15', 'S11_D15', 'S11_D13'],
                                ['S12_D16', 'S14_D16', 'S14_D15', 'S12_D15'],
                                ['S7_D10', 'S9_D10', 'S9_D5', 'S7_D5'],
                                ['S10_D10', 'S10_D12', 'S9_D12', 'S9_D10'],
                                ['S11_D15', 'S13_D15', 'S13_D14', 'S11_D14'],
                                ['S14_D15', 'S14_D8', 'S13_D8', 'S13_D15']]

        # NVC results file path
        nvc_output_folder = 'E:\\DATA\\公开数据集\\EEG-fNIRS\\TUBerlinBCI\\Analysis Folder\\NVC\\02'
        json_output_file = os.path.join(nvc_output_folder, 'nvc_results.json')

        # Load NVC results
        with open(json_output_file, 'r') as json_file:
            nvc_results = json.load(json_file)

        # Focus on specific subject
        subject = 'subject 15'
        subject_data = nvc_results['data'].get(subject, [])

        # Initialize storage
        nvc_left = {eeg_ch[0]: [] for eeg_ch in eeg_channel_select}
        nvc_right = {eeg_ch[0]: [] for eeg_ch in eeg_channel_select}

        # Process data
        for epoch in subject_data:
            # Get motor imagery label
            label = nvc_results['Labels'][subject][subject_data.index(epoch)]

            # Process each EEG channel
            for eeg_idx, eeg_ch_list in enumerate(eeg_channel_select):
                eeg_ch = eeg_ch_list[0]  # Current EEG channel

                # Calculate NVC mean for associated fNIRS channels
                fnirs_channels = [ch + ' hbo' for ch in fnirs_channel_select[eeg_idx]]
                nvc_values = []

                for result in epoch:
                    if result['EEG_Channel'] == eeg_ch and result['fNIRS_Channel'] in fnirs_channels:
                        # Store absolute NVC value
                        nvc_values.append(abs(result['NVC_Value']))

                # Save processed values
                if nvc_values:
                    nvc_mean = np.mean(nvc_values)
                    if label == 'left':
                        nvc_left[eeg_ch].append(nvc_mean)
                    elif label == 'right':
                        nvc_right[eeg_ch].append(nvc_mean)

        eeg_channels = list(nvc_left.keys())
        self.ax.clear()

        # Prepare boxplot data
        boxplot_data_left = [nvc_left[ch] for ch in eeg_channels]
        boxplot_data_right = [nvc_right[ch] for ch in eeg_channels]

        # Plot left hand data
        positions_left = np.arange(len(eeg_channels)) * 2.0
        self.ax.boxplot(boxplot_data_left, positions=positions_left, widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor="skyblue"), labels=eeg_channels)

        # Plot right hand data
        positions_right = positions_left + 0.8
        self.ax.boxplot(boxplot_data_right, positions=positions_right, widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor="salmon"))

        # Statistical significance markers
        significance_level = 0.05
        bracket_height = 0.05
        asterisk_offset = 0.01
        line_width = 1.5

        for i, ch in enumerate(eeg_channels):
            # Perform t-test
            t_stat, p_value = ttest_ind(nvc_left[ch], nvc_right[ch], nan_policy='omit')

            # Add significance markers if significant
            if p_value < significance_level:
                max_value = max(max(nvc_left[ch]), max(nvc_right[ch]))
                y_bracket = max_value + bracket_height
                y_asterisk = y_bracket + asterisk_offset

                # Add asterisk
                self.ax.text(positions_left[i] + 0.4, y_asterisk, '*', ha='center', va='bottom', fontsize=14,
                             color='red')

                # Add significance bracket
                self.ax.plot([positions_left[i], positions_right[i]], [y_bracket, y_bracket], color='black',
                             lw=line_width)
                self.ax.plot([positions_left[i], positions_left[i]], [y_bracket, y_bracket - 0.02], color='black',
                             lw=line_width)
                self.ax.plot([positions_right[i], positions_right[i]], [y_bracket, y_bracket - 0.02], color='black',
                             lw=line_width)

        # Set labels and titles
        self.ax.set_xlabel('EEG Channels', fontsize=12)
        self.ax.set_ylabel('Absolute NVC Value', fontsize=12)
        self.ax.set_title('Neurovascular Coupling (NVC) Boxplot for Left and Right Hand Motor Imagery', fontsize=12)

        # Position ticks and legend
        self.ax.set_xticks(positions_left + 0.4)
        self.ax.set_xticklabels(eeg_channels)

        legend_elements = [plt.Line2D([0], [0], color="skyblue", lw=4, label='Left Hand'),
                           plt.Line2D([0], [0], color="salmon", lw=4, label='Right Hand'),
                           plt.Line2D([0], [0], marker='*', color='w', label='p<0.05',
                                      markerfacecolor='red', markersize=10)]

        self.ax.legend(handles=legend_elements, loc='upper right')

        self.canvas.draw()


class TestBoxPlot(QDialog):
    """
    Dialog for displaying neurovascular coupling boxplots.

    Provides save and settings functionality for visualization.
    """

    def __init__(self, parent=None):
        """
        Initialize boxplot visualization dialog.

        :param parent: Parent widget
        :type parent: QWidget
        """
        super(TestBoxPlot, self).__init__(parent)
        self.matplotlib_widget = MatplotlibWidget()
        self.vlayout = QVBoxLayout(self)

        # Create control buttons
        self.bnt_save = BFPushButton('Save')
        self.bnt_save.setFixedWidth(150)
        self.bnt_setting = BFPushButton('Settings')
        self.bnt_setting.setFixedWidth(150)

        # Add buttons to layout
        self.bnt_layout = QHBoxLayout()
        self.bnt_layout.addStretch(1)
        self.bnt_layout.addWidget(self.bnt_save)
        self.bnt_layout.addWidget(self.bnt_setting)

        # Add widgets to dialog
        self.vlayout.addLayout(self.bnt_layout)
        self.vlayout.addWidget(self.matplotlib_widget)


class MplCanvas(FigureCanvas):
    """
    Custom Matplotlib canvas for embedding plots in PyQt applications.

    Creates a standardized plotting area with configurable size.
    """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Initialize Matplotlib canvas.

        :param parent: Parent widget
        :type parent: QWidget
        :param width: Canvas width
        :type width: int
        :param height: Canvas height
        :type height: int
        :param dpi: Canvas resolution (dots per inch)
        :type dpi: int
        """
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class TestMLPlot(QMainWindow):
    """
    Application for visualizing machine learning model performance.

    Displays ROC curves and confusion matrices for SVM models.
    """

    def __init__(self):
        """Initialize machine learning visualization application."""
        super().__init__()
        self.setGeometry(100, 100, 1200, 600)
        self.init_ui()
        self.run_svm_and_plot()

    def init_ui(self):
        """Initialize user interface components."""
        # Create central widget
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)

        # Add control buttons
        self.bnt_save = BFPushButton('Save')
        self.bnt_save.setFixedWidth(150)
        self.bnt_setting = BFPushButton('Settings')
        self.bnt_setting.setFixedWidth(150)
        self.bnt_layout = QHBoxLayout()
        self.bnt_layout.addStretch(1)
        self.bnt_layout.addWidget(self.bnt_save)
        self.bnt_layout.addWidget(self.bnt_setting)
        layout.addLayout(self.bnt_layout)

        # Create visualization area
        figure_layout = QHBoxLayout()
        self.roc_canvas = MplCanvas(self, width=5, height=4)
        self.cm_canvas = MplCanvas(self, width=5, height=4)
        figure_layout.addWidget(self.roc_canvas)
        figure_layout.addWidget(self.cm_canvas)
        layout.addLayout(figure_layout)

    def run_svm_and_plot(self):
        """
        Train SVM model and visualize performance.

        Generates synthetic data, trains SVM with grid search, and creates ROC curve
        and confusion matrix visualizations.
        """
        # Generate synthetic classification dataset
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

        # Configure SVM with grid search
        svm_model = svm.SVC(probability=True)
        param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
        grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='roc_auc')
        grid_search.fit(X, y)

        # Get best model
        best_model = grid_search.best_estimator_
        y_pre = best_model.predict(X)
        y_scores = best_model.predict_proba(X)[:, 1]

        # Create ROC curve
        fpr, tpr, _ = roc_curve(y, y_scores)
        roc_auc = auc(fpr, tpr)
        self.roc_canvas.axes.clear()
        self.roc_canvas.axes.plot(fpr, tpr, color='darkorange', lw=2,
                                  label='ROC curve (area = %0.2f)' % roc_auc)
        self.roc_canvas.axes.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        self.roc_canvas.axes.set_xlim([0.0, 1.0])
        self.roc_canvas.axes.set_ylim([0.0, 1.05])
        self.roc_canvas.axes.set_xlabel('False Positive Rate')
        self.roc_canvas.axes.set_ylabel('True Positive Rate')
        self.roc_canvas.axes.set_title('Receiver Operating Characteristic (ROC)')
        self.roc_canvas.axes.legend(loc="lower right")
        self.roc_canvas.draw()

        # Create confusion matrix
        cm = confusion_matrix(y, y_pre)
        self.cm_canvas.axes.clear()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=self.cm_canvas.axes)
        self.cm_canvas.axes.set_title('Confusion Matrix')
        self.cm_canvas.axes.set_xlabel('Predicted Label')
        self.cm_canvas.axes.set_ylabel('True Label')
        self.cm_canvas.draw()


import os
import sys
from PyQt5 import QtWidgets
import mne
from mne.datasets import sample
import sys
import pyvista as pv
from PyQt5 import QtWidgets
from mne.viz import create_3d_figure, set_3d_backend
from pyvistaqt import QtInteractor, BackgroundPlotter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Set 3D visualization backend
set_3d_backend('pyvistaqt')


class Sensors3D(QtWidgets.QMainWindow):
    """
    Application for 3D visualization of EEG/fNIRS sensor arrangements. Uses PyVista and MNE for creating 3D brain models with sensor placements.
    """

    def __init__(self):
        """Initialize 3D sensor visualization application."""
        super().__init__()

        # Window setup
        self.setGeometry(100, 100, 800, 600)
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)

        # Create layout and 3D visualization
        self.layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(self.layout)
        self.plotter = BackgroundPlotter(show=False)
        self.figure = create_3d_figure((800, 600))
        self.figure._plotter = self.plotter

        # Add plotter to interface
        self.layout.addWidget(self.plotter.interactor)


class TestEEGPlot(QWidget):
    """
    Application for visualizing raw EEG/fNIRS data. Combines traditional waveform visualization with 3D sensor visualization.
    """

    def __init__(self):
        """Initialize EEG/fNIRS visualization application."""
        super().__init__()
        self.setGeometry(100, 100, 1200, 600)
        self.layout = QVBoxLayout(self)
        self.init_ui()

    def init_ui(self):
        """Initialize user interface components."""
        # Create control buttons
        self.bnt_save = BFPushButton('Save')
        self.bnt_save.setFixedWidth(150)
        self.bnt_setting = BFPushButton('Settings')
        self.bnt_setting.setFixedWidth(150)
        self.bnt_sensor = BFPushButton('Plot Sensors')
        self.bnt_sensor.setFixedWidth(150)

        # Add buttons to layout
        self.bnt_layout = QHBoxLayout()
        self.bnt_layout.addStretch(1)
        self.bnt_layout.addWidget(self.bnt_sensor)
        self.bnt_layout.addWidget(self.bnt_save)
        self.bnt_layout.addWidget(self.bnt_setting)
        self.layout.addLayout(self.bnt_layout)

        # Load and plot raw data
        data, file_path = read_file_by_qt(self)
        self.drawing_widget = RawCurvePlotDialog(data=data, filePath=file_path[0])
        self.eeg_sensor = TestEEGSensorPlot()
        self.layout.addWidget(self.drawing_widget)

        # Connect button to sensor visualization
        self.bnt_sensor.clicked.connect(lambda: self.eeg_sensor.show())
        self.drawing_widget.plot_data(self.drawing_widget.current_page)


class TestEEGSensorPlot(QWidget):
    """
    Dialog for 3D visualization of EEG/fNIRS sensor positions. Displays sensor positions on a 3D brain model using MNE standard dataset.
    """

    def __init__(self):
        """Initialize 3D sensor visualization dialog."""
        super().__init__()
        self.setGeometry(100, 100, 1200, 600)
        self.layout = QVBoxLayout(self)
        self.init_ui()

    def init_ui(self):
        """Initialize user interface components."""
        # Create control buttons
        self.bnt_save = BFPushButton('Save')
        self.bnt_save.setFixedWidth(150)

        # Add buttons and 3D viewer
        self.bnt_layout = QHBoxLayout()
        self.bnt_layout.addStretch(1)
        self.bnt_layout.addWidget(self.bnt_save)
        self.layout.addLayout(self.bnt_layout)

        sensors = Sensors3D()
        self.layout.addWidget(sensors)

        # Paths to sample data
        eeg_path = 'C:\\Users\\28164\\Desktop\\test\\open_dataset\\eeg\\struct_1.bdf'
        fnirs_path = 'C:\\Users\\28164\\Desktop\\test\\open_dataset\\struct_1.snirf'

        # Load and prepare EEG/fNIRS data
        raw_nirs = mne.io.read_raw_snirf(fnirs_path, preload=True)
        raw_eeg = mne.io.read_raw_bdf(eeg_path, preload=True)
        events, _ = mne.events_from_annotations(raw_eeg)
        raw_eeg.set_channel_types({"VEOG": "eog", "HEOG": "eog"})
        raw_eeg.set_montage('standard_1005')

        # Create and configure 3D brain visualization
        subjects_dir = os.path.join(mne.datasets.sample.data_path(), "subjects")
        brain_eeg_fnirs = mne.viz.Brain(
            "fsaverage", subjects_dir=subjects_dir, background="w", cortex="0.5", alpha=0.3,
            title='EEG and fNIRS Sensors', show=False, figure=sensors.figure
        )
        # Add sensor positions to brain model
        brain_eeg_fnirs.add_sensors(raw_eeg.info, trans="fsaverage")
        brain_eeg_fnirs.show_view(azimuth=0, elevation=0, distance=500)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestBoxPlot()
    window.show()
    sys.exit(app.exec_())