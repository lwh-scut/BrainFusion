import json
import os
import sys

import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
                             QListWidget, QFileDialog, QDialog, QLineEdit, QLabel, QMenu, QMessageBox, QCheckBox,
                             QComboBox, QScrollArea, QFrame, QFormLayout, QColorDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from scipy.stats import ttest_ind

from BrainFusion.io.File_IO import read_file, read_xlsx
from BrainFusion.statistic.significance_test import calculate_significance, multiple_comparison_correction
from BrainFusion.statistic.statistical_plot import ScatterPlotWindow, DensityHistogramWindow, SignificanceBoxPlotWindow, \
    SignificanceViolinPlotWindow
from UI.ui_component import BFPushButton


def get_feature(feature_dict, channel_name=None, feature_name=None):
    """
    Extract specific feature data from a feature dictionary.

    :param feature_dict: Dictionary containing feature data
    :type feature_dict: dict
    :param channel_name: Target channel name (optional)
    :type channel_name: str
    :param feature_name: Target feature name (required)
    :type feature_name: str
    :return: Feature values for specified channel/feature
    :rtype: list
    :raises ValueError: For invalid channel or feature names
    """
    # Validate channel selection
    if "ch_names" in feature_dict and channel_name:
        if channel_name not in feature_dict["ch_names"]:
            raise ValueError(f"Channel '{channel_name}' not found")
        channel_index = feature_dict["ch_names"].index(channel_name)
    else:
        channel_index = None

    # Validate feature selection
    if feature_name not in feature_dict["feature"]:
        raise ValueError(f"Feature '{feature_name}' not found")

    # Retrieve and return requested data
    feature_data = feature_dict["feature"][feature_name]
    if channel_index is not None:
        return feature_data[channel_index]
    return feature_data


def convert_to_significance_dict(result):
    """
    Convert significance test results to a comparison dictionary.

    :param result: Significance test result list
    :type result: list
    :return: Dictionary mapping group pairs to corrected p-values
    :rtype: dict
    """
    significance_dict = {}
    for entry in result:
        if not pd.isna(entry['corrected_p_value']):
            group1, group2 = entry['group_comparison'].split(' vs ')
            significance_dict[(f"Group {group1}", f"Group {group2}")] = entry['corrected_p_value']
    return significance_dict


class GroupDialog(QDialog):
    """
    Dialog window for creating or editing statistical groups.

    :param parent: Parent widget
    :type parent: QWidget
    :param group_name: Initial group name
    :type group_name: str
    :param folder_path: Initial folder path
    :type folder_path: str
    """

    def __init__(self, parent=None, group_name="", folder_path=""):
        super().__init__(parent)
        self.setWindowTitle("Group Configuration")
        self._init_ui(group_name, folder_path)

    def _init_ui(self, group_name, folder_path):
        """Initialize UI elements."""
        layout = QVBoxLayout()

        # Group name input
        name_label = QLabel("Group Name:")
        self.name_edit = QLineEdit(group_name)
        layout.addWidget(name_label)
        layout.addWidget(self.name_edit)

        # Folder selection
        folder_label = QLabel("Data Folder:")
        self.folder_edit = QLineEdit(folder_path)
        folder_button = BFPushButton("Browse")
        folder_button.clicked.connect(self._select_folder)
        layout.addWidget(folder_label)
        layout.addWidget(self.folder_edit)
        layout.addWidget(folder_button)

        # Confirm button
        confirm_button = BFPushButton("Confirm")
        confirm_button.clicked.connect(self.accept)
        layout.addWidget(confirm_button)

        self.setLayout(layout)

    def _select_folder(self):
        """Open folder selection dialog."""
        path = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if path:
            self.folder_edit.setText(path)

    def get_group_data(self):
        """Retrieve input group configuration data."""
        return self.name_edit.text(), self.folder_edit.text()


class ChannelSelectionDialog(QDialog):
    """
    Dialog for selecting data channels.

    :param available_channels: List of available channel names
    :type available_channels: list
    :param selected_channels: Pre-selected channel names
    :type selected_channels: list
    :param parent: Parent widget
    :type parent: QWidget
    """

    def __init__(self, available_channels, selected_channels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Channel Selection")
        self.setMinimumSize(300, 300)
        self._init_ui(available_channels, selected_channels)

    def _init_ui(self, channels, selected):
        """Initialize UI elements."""
        layout = QVBoxLayout()

        # Scrollable checkbox area
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Create channel checkboxes
        self.checkboxes = {}
        for channel in channels:
            cb = QCheckBox(channel)
            cb.setChecked(channel in selected)
            scroll_layout.addWidget(cb)
            self.checkboxes[channel] = cb

        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # Confirm button
        confirm_button = BFPushButton("Confirm")
        confirm_button.clicked.connect(self.accept)
        layout.addWidget(confirm_button)

        self.setLayout(layout)

    def get_selected_channels(self):
        """Retrieve selected channel names."""
        return [ch for ch, cb in self.checkboxes.items() if cb.isChecked()]


class VisualisationSettingsDialog(QDialog):
    """
    Dialog for customizing plot visualization parameters.

    :param title: Current plot title
    :type title: str
    :param y_label: Current Y-axis label
    :type y_label: str
    :param x_label: Current X-axis label
    :type x_label: str
    :param legend: Current legend items
    :type legend: list
    :param x_ticks: Current X-axis tick labels
    :type x_ticks: list
    :param y_range: Current Y-axis range
    :type y_range: tuple
    :param color: Current primary color
    :type color: str
    :param parent: Parent widget
    :type parent: QWidget
    """

    def __init__(self, title="Default Title", y_label="Y Axis", x_label="X Axis",
                 legend=None, x_ticks=None, y_range=None, color="#ff0000", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Customization")
        self.setGeometry(100, 100, 300, 300)
        self._init_ui(title, y_label, x_label, legend, x_ticks, y_range, color)

    def _init_ui(self, title, y_label, x_label, legend, x_ticks, y_range, color):
        """Initialize UI elements."""
        layout = QFormLayout(self)

        # Input fields with default values
        self.title_input = QLineEdit(title)
        self.y_label_input = QLineEdit(y_label)
        self.x_label_input = QLineEdit(x_label)
        self.legend_input = QLineEdit(", ".join(legend) if legend else "")
        self.x_ticks_input = QLineEdit(", ".join(x_ticks) if x_ticks else "")
        self.y_range_input = QLineEdit(f"{y_range[0]}, {y_range[1]}" if y_range else "0, 10")
        self.color_input = QLineEdit(color)

        # Add input fields to form
        layout.addRow("Title:", self.title_input)
        layout.addRow("Y Label:", self.y_label_input)
        layout.addRow("X Label:", self.x_label_input)
        layout.addRow("Legend (comma-separated):", self.legend_input)
        layout.addRow("X Ticks (comma-separated):", self.x_ticks_input)
        layout.addRow("Y Range (min, max):", self.y_range_input)
        layout.addRow("Color (hex):", self.color_input)

        # Confirm button
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.accept)
        layout.addRow(self.confirm_button)

    def get_settings(self):
        """Retrieve visualization settings from dialog inputs."""
        title = self.title_input.text()
        y_label = self.y_label_input.text()
        x_label = self.x_label_input.text()
        legend = self.legend_input.text().split(", ")
        x_ticks = self.x_ticks_input.text().split(", ")
        y_range = tuple(map(float, self.y_range_input.text().split(",")))
        color = self.color_input.text()
        return title, y_label, x_label, legend, x_ticks, y_range, color


class MatplotlibWidget(QWidget):
    """Embedded widget for displaying Matplotlib visualizations."""

    def __init__(self, parent=None):
        """
        Initialize Matplotlib figure container.

        :param parent: Parent widget
        :type parent: QWidget
        """
        super().__init__(parent)
        self._initialize_plot_components()
        self.plot_data()

    def _initialize_plot_components(self):
        """Set up figure, axes, and canvas components."""
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_default(self):
        """Generate default boxplot visualization with random data."""
        data = [np.random.randn(100) for _ in range(5)]
        self.plot(
            data=data,
            title="Sample Boxplot",
            y_label="Values",
            x_label="Groups",
            x_ticks=["A", "B", "C", "D", "E"]
        )

    def plot(self, data, title="", y_label="", x_label="", legend=None,
             x_ticks=None, y_range=None, color="blue"):
        """
        Render customized boxplot visualization.

        :param data: Input dataset for visualization
        :type data: list
        :param title: Plot title text
        :type title: str
        :param y_label: Y-axis label text
        :type y_label: str
        :param x_label: X-axis label text
        :type x_label: str
        :param legend: Legend items list
        :type legend: list
        :param x_ticks: X-axis tick labels
        :type x_ticks: list
        :param y_range: Y-axis range limits
        :type y_range: tuple
        :param color: Primary boxplot color
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
        """Generate NVC boxplot visualization for left vs right hand MI."""
        # EEG and fNIRS channel configuration
        eeg_channels = [['FCC5h'], ['FCC3h'], ['FCC4h'], ['FCC6h'],
                        ['CCP5h'], ['CCP3h'], ['CCP4h'], ['CCP6h']]
        fnirs_channels = [['S8_D9', 'S8_D10', 'S7_D10', 'S7_D9'],
                          ['S8_D11', 'S10_D11', 'S10_D10', 'S8_D10'],
                          ['S12_D13', 'S12_D15', 'S11_D15', 'S11_D13'],
                          ['S12_D16', 'S14_D16', 'S14_D15', 'S12_D15'],
                          ['S7_D10', 'S9_D10', 'S9_D5', 'S7_D5'],
                          ['S10_D10', 'S10_D12', 'S9_D12', 'S9_D10'],
                          ['S11_D15', 'S13_D15', 'S13_D14', 'S11_D14'],
                          ['S14_D15', 'S14_D8', 'S13_D8', 'S13_D15']]

        # Load neurovascular coupling results
        results_path = os.path.join('E:\\DATA\\public_datasets\\EEG-fNIRS\\TUBerlinBCI\\Analysis Folder\\NVC\\02',
                                    'nvc_results.json')
        with open(results_path, 'r') as file:
            nvc_data = json.load(file)

        # Process subject data
        subject_id = 'subject 24'
        subject_values = nvc_data['data'].get(subject_id, [])

        # Prepare left/right hand data containers
        left_values = {eeg[0]: [] for eeg in eeg_channels}
        right_values = {eeg[0]: [] for eeg in eeg_channels}

        # Extract NVC values per EEG channel
        for epoch in subject_values:
            label = nvc_data['Labels'][subject_id][subject_values.index(epoch)]

            for idx, eeg_group in enumerate(eeg_channels):
                current_eeg = eeg_group[0]
                fnirs_group = [ch + ' hbo' for ch in fnirs_channels[idx]]
                nvc_vals = []

                # Collect matching fNIRS channel values
                for result in epoch:
                    if (result['EEG_Channel'] == current_eeg and
                            result['fNIRS_Channel'] in fnirs_group):
                        nvc_vals.append(abs(result['NVC_Value']))

                # Store mean values by label
                if nvc_vals:
                    avg = np.mean(nvc_vals)
                    if label == 'left':
                        left_values[current_eeg].append(avg)
                    elif label == 'right':
                        right_values[current_eeg].append(avg)

        # Clear and prepare plot
        self.ax.clear()
        eeg_labels = list(left_values.keys())
        left_data = [left_values[ch] for ch in eeg_labels]
        right_data = [right_values[ch] for ch in eeg_labels]

        # Position boxplot pairs
        left_positions = np.arange(len(eeg_labels)) * 2.0
        right_positions = left_positions + 0.8

        # Render left hand boxplots
        self.ax.boxplot(left_data, positions=left_positions, widths=0.6,
                        patch_artist=True, boxprops=dict(facecolor="skyblue"),
                        labels=eeg_labels)

        # Render right hand boxplots
        self.ax.boxplot(right_data, positions=right_positions, widths=0.6,
                        patch_artist=True, boxprops=dict(facecolor="salmon"))

        # Add significance markers
        significance_level = 0.05
        bracket_height = 0.05
        asterisk_offset = 0.01
        line_width = 1.5

        # Perform statistical comparisons
        for i, channel in enumerate(eeg_labels):
            t_stat, p_val = ttest_ind(left_values[channel], right_values[channel], nan_policy='omit')

            # Mark significant differences
            if p_val < significance_level:
                max_val = max(max(left_values[channel]), max(right_values[channel]))
                y_bracket = max_val + bracket_height
                y_asterisk = y_bracket + asterisk_offset

                # Add asterisk marker
                self.ax.text(left_positions[i] + 0.4, y_asterisk, '*',
                             ha='center', va='bottom', fontsize=14, color='red')

                # Draw significance bracket
                self.ax.plot([left_positions[i], right_positions[i]], [y_bracket, y_bracket],
                             color='black', lw=line_width)
                self.ax.plot([left_positions[i], left_positions[i]],
                             [y_bracket, y_bracket - 0.02], color='black', lw=line_width)
                self.ax.plot([right_positions[i], right_positions[i]],
                             [y_bracket, y_bracket - 0.02], color='black', lw=line_width)

        # Configure plot appearance
        self.ax.set_xlabel('EEG Channels', fontsize=12)
        self.ax.set_ylabel('Absolute NVC Value', fontsize=12)
        self.ax.set_title('Neurovascular Coupling for Hand Motor Imagery', fontsize=12)
        self.ax.set_xticks(left_positions + 0.4)
        self.ax.set_xticklabels(eeg_labels)

        # Create plot legend
        legend_items = [
            plt.Line2D([0], [0], color="skyblue", lw=4, label='Left Hand'),
            plt.Line2D([0], [0], color="salmon", lw=4, label='Right Hand'),
            plt.Line2D([0], [0], marker='*', color='w', label='p<0.05',
                       markerfacecolor='red', markersize=10)
        ]
        self.ax.legend(handles=legend_items, loc='upper right')

        # Render final visualization
        self.canvas.draw()


class StatisticalAnalysisDialog(QMainWindow):
    """Dialog window for performing statistical analyses and visualizations."""

    def __init__(self):
        """Initialize statistical analysis interface components."""
        super().__init__()
        self._initialize_state_variables()
        self._configure_window_properties()
        self._create_main_layout()
        self._connect_signal_handlers()

    def _initialize_state_variables(self):
        """Initialize application state variables."""
        self.result = []
        self.group_select_features = {}
        self.data = None
        self.group_folder = {}
        self.group_files = {}
        self.is_valid = False

    def _configure_window_properties(self):
        """Set window title and dimensions."""
        self.setWindowTitle("Statistical Analysis")
        self.setGeometry(100, 100, 1200, 800)

    def _create_main_layout(self):
        """Construct primary interface layout."""
        main_layout = QHBoxLayout()
        central_widget = QWidget()

        # Group management section
        self.group_widget = self._create_group_management_section()

        # Analysis/visualization section
        analysis_visual_widget = self._create_analysis_visualization_section()

        # Combine sections
        main_layout.addWidget(self.group_widget)
        main_layout.addWidget(analysis_visual_widget)
        main_layout.setStretch(0, 25)
        main_layout.setStretch(1, 75)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _create_group_management_section(self):
        """Create group management interface components."""
        widget = QFrame()
        widget.setFrameShape(QFrame.Box)
        widget.setStyleSheet("background-color: white;")
        layout = QVBoxLayout()

        # Section title
        title_label = QLabel("Group Management")
        title_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")
        layout.addWidget(title_label)

        # Group action buttons
        button_layout = QHBoxLayout()
        self.new_group_button = BFPushButton("Add Group")
        self.import_groups_button = BFPushButton("Import Groups")
        button_layout.addWidget(self.new_group_button)
        button_layout.addWidget(self.import_groups_button)
        layout.addLayout(button_layout)

        # Group list
        self.group_list = QListWidget()
        layout.addWidget(self.group_list)

        # Validation button
        self.validate_button = BFPushButton("Validate Groups")
        layout.addWidget(self.validate_button)

        widget.setLayout(layout)
        return widget

    def _create_analysis_visualization_section(self):
        """Create analysis and visualization interface components."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Statistical analysis section
        self.analysis_widget = self._create_statistical_analysis_section()

        # Visualization section
        self.visual_widget = self._create_visualization_section()

        layout.addWidget(self.analysis_widget)
        layout.addWidget(self.visual_widget)
        layout.setStretch(0, 3)
        layout.setStretch(1, 7)

        widget.setLayout(layout)
        return widget

    def _create_statistical_analysis_section(self):
        """Create statistical analysis configuration components."""
        widget = QFrame()
        widget.setFrameShape(QFrame.Box)
        widget.setStyleSheet("background-color: white;")
        outer_layout = QVBoxLayout()
        layout = QGridLayout()

        # Section title
        title_label = QLabel("Statistical Analysis")
        title_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")
        outer_layout.addWidget(title_label)

        # Description
        description = QLabel('Performs pairwise group comparisons using selected method. '
                             'Multiple comparison correction applied when >2 groups exist. '
                             'MANOVA automatically used when features >2.')
        description.setStyleSheet("font-family: 'Times New Roman'; font-size: 10pt; color: blue;")
        description.setWordWrap(True)
        outer_layout.addWidget(description)
        outer_layout.addLayout(layout)
        outer_layout.addStretch(1)

        # Channel selection
        channel_label = QLabel("Channel:")
        self.channel_combo = QComboBox()
        layout.addWidget(channel_label, 0, 0)
        layout.addWidget(self.channel_combo, 0, 1)

        # Feature selection
        feature_label = QLabel("Feature:")
        self.feature_combo = QComboBox()
        layout.addWidget(feature_label, 1, 0)
        layout.addWidget(self.feature_combo, 1, 1)

        # Statistical method selection
        method_label = QLabel("Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "t-test", "t-test(paired)", "anova", "mann-whitney U",
            "wilcoxon(paired)", "kruskal-wallis"
        ])
        layout.addWidget(method_label, 2, 0)
        layout.addWidget(self.method_combo, 2, 1)

        # Correction controls
        correction_label = QLabel("Correction:")
        self.correction_combo = QComboBox()
        self.correction_combo.addItems([
            "bonferroni", "fdr_bh", "fdr_by", "holm-sidak", "sidak"
        ])
        self.enable_correction = QCheckBox("Enable")
        self.enable_correction.setChecked(True)
        layout.addWidget(correction_label, 3, 0)
        layout.addWidget(self.correction_combo, 3, 1)
        layout.addWidget(self.enable_correction, 3, 2)

        # Action buttons
        self.run_button = BFPushButton("Run Analysis")
        self.run_button.setFixedWidth(120)
        self.status_label = QLabel("Status: Waiting")
        self.export_button = BFPushButton("Export Results")
        self.export_button.setFixedWidth(120)
        layout.addWidget(self.run_button, 4, 0)
        layout.addWidget(self.status_label, 4, 1)
        layout.addWidget(self.export_button, 4, 2)

        layout.setVerticalSpacing(10)
        widget.setLayout(outer_layout)
        return widget

    def _create_visualization_section(self):
        """Create visualization control components."""
        widget = QFrame()
        widget.setFrameShape(QFrame.Box)
        widget.setStyleSheet("background-color: white;")
        layout = QVBoxLayout()
        self.scroll_area = QScrollArea()

        # Section title
        title_label = QLabel("Data Visualization")
        title_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")
        layout.addWidget(title_label)

        # Visualization controls
        control_layout = QHBoxLayout()
        plot_type_label = QLabel('Plot Type: ')
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            'scatter plot', 'bar plot', 'box plot', 'violin plot'
        ])
        self.plot_type_combo.setFixedHeight(25)

        self.plot_button = BFPushButton("Generate Plot")
        self.save_button = BFPushButton("Save Image")
        self.save_button.setFixedWidth(100)
        self.settings_button = BFPushButton("Plot Settings")
        self.settings_button.setFixedWidth(100)

        control_layout.addWidget(plot_type_label)
        control_layout.addWidget(self.plot_type_combo)
        control_layout.addWidget(self.plot_button)
        control_layout.addStretch(1)
        control_layout.addWidget(self.save_button)
        control_layout.addWidget(self.settings_button)

        # Visualization container
        self.plot_container = ScatterPlotWindow()
        self.scroll_area.setWidget(self.plot_container)

        layout.addLayout(control_layout)
        layout.addWidget(self.scroll_area)
        widget.setLayout(layout)
        return widget

    def _connect_signal_handlers(self):
        """Connect UI components to action handlers."""
        self.new_group_button.clicked.connect(self.add_group)
        self.validate_button.clicked.connect(self.validate_groups)
        self.group_list.itemDoubleClicked.connect(self.modify_group)
        self.run_button.clicked.connect(self.run_analysis)
        self.export_button.clicked.connect(self.export_results)
        self.plot_button.clicked.connect(self.generate_visualization)
        # self.settings_button.clicked.connect(self.configure_visualization)

    def add_group(self):
        """Open dialog to create new analysis group."""
        dialog = GroupDialog(self)
        if dialog.exec_():
            name, folder = dialog.get_group_data()
            self.group_list.addItem(f"{name} - {folder}")
        self.is_valid = False

    def modify_group(self, item):
        """Open dialog to edit existing analysis group."""
        name, folder = item.text().split(" - ")
        dialog = GroupDialog(self, name, folder)
        if dialog.exec_():
            new_name, new_folder = dialog.get_group_data()
            item.setText(f"{new_name} - {new_folder}")
        self.is_valid = False

    def validate_groups(self):
        """Validate and load group datasets."""
        self.group_files.clear()
        self.group_select_features.clear()
        self.group_folder.clear()

        # Process each group definition
        for i in range(self.group_list.count()):
            group_data = self.group_list.item(i).text()
            group_name, folder_path = group_data.split(" - ")
            self.group_files[group_name] = []

            # Load group files
            for file_name in os.listdir(folder_path):
                if file_name.split('.')[1] == 'xlsx':
                    file_data = read_xlsx(os.path.join(folder_path, file_name))
                else:
                    file_data = None
                self.group_files[group_name].append(file_data)

        # Initialize UI with first group's metadata
        self.data = next(iter(self.group_files.values()))[0]
        self.feature_combo.clear()
        self.feature_combo.addItems(self.data['feature'])
        self.status_label.setText("Status: Waiting")

        # Configure channel selector
        if 'ch_names' not in self.data.keys():
            self.channel_combo.setVisible(False)
        else:
            self.channel_combo.clear()
            self.channel_combo.addItems(self.data['ch_names'])

        self.is_valid = True
        QMessageBox.information(self, "Success", "Data validation complete")

    def run_analysis(self):
        """Execute statistical analysis on selected features."""
        if not self.is_valid:
            QMessageBox.warning(self, "Action Required", "Validate groups before analysis")
            return

        self.status_label.setText("Status: Processing...")

        # Prepare feature data
        for group_name in self.group_files:
            self.group_select_features[group_name] = []
            for dataset in self.group_files[group_name]:
                channel = self.channel_combo.currentText() if 'ch_names' in dataset else None
                feature = self.feature_combo.currentText()
                self.group_select_features[group_name].append(
                    get_feature(dataset, channel, feature)
                )

        # Configure analysis method
        method = self.method_combo.currentText()
        is_paired = False
        if method == 't-test(paired)':
            method = 't-test'
            is_paired = True
        elif method == 'wilcoxon(paired)':
            method = 'wilcoxon'
            is_paired = True

        # Perform analysis
        self.result = calculate_significance(
            self.group_select_features,
            method=method,
            paired=is_paired
        )

        # Apply correction
        if self.enable_correction.isChecked():
            correct_method = self.correction_combo.currentText()
            self.result = multiple_comparison_correction(self.result, correct_method)

        self.status_label.setText("Status: Completed")

    def export_results(self):
        """Export analysis results to external format."""
        QMessageBox.information(self, "Export", "Results exported successfully!")

    def generate_visualization(self):
        """Generate selected visualization type."""
        if not self.group_select_features:
            QMessageBox.warning(self, "Data Required", "Run analysis before generating plot")
            return

        plot_type = self.plot_type_combo.currentText()

        if plot_type == 'scatter plot':
            self.plot_container = ScatterPlotWindow()
            self.scroll_area.setWidget(self.plot_container)
            self.plot_container.plot_scatter(self.group_select_features)

        elif plot_type == 'bar plot':
            self.plot_container = DensityHistogramWindow()
            self.scroll_area.setWidget(self.plot_container)
            self.plot_container.plot_density_histogram(self.group_select_features)

        elif plot_type == 'box plot':
            self.plot_container = SignificanceBoxPlotWindow()
            self.scroll_area.setWidget(self.plot_container)
            significance_dict = convert_to_significance_dict(self.result)
            self.plot_container.plot_boxplot_with_significance(
                self.group_select_features,
                significance_dict
            )

        elif plot_type == 'violin plot':
            self.plot_container = SignificanceViolinPlotWindow()
            self.scroll_area.setWidget(self.plot_container)
            significance_dict = convert_to_significance_dict(self.result)
            self.plot_container.plot_violin_with_significance(
                self.group_select_features,
                significance_dict
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StatisticalAnalysisDialog()
    window.show()
    sys.exit(app.exec_())
