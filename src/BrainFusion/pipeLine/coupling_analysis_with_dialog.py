# -*- coding: utf-8 -*-
"""
Neurovascular Coupling Analysis Interface

Provides graphical interfaces for single file, batch processing, and BIDS-structured
neurovascular coupling analysis between EEG and fNIRS signals.
"""

import os
import sys
import threading
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QComboBox, QLabel, QSpinBox, QDoubleSpinBox,
                             QLineEdit, QFormLayout, QMessageBox, QFileDialog)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from BrainFusion.utils.normalize import normalize

# Configure matplotlib backend
matplotlib.use('QtAgg')

from BrainFusion.pipeLine.coupling_analysis import compute_neurovascular_coupling_by_dict, \
    compute_neurovascular_coupling
from BrainFusion.utils.files import are_filenames_equal, getFileNameWithoutSuffix
from BrainFusion.io.File_IO import read_file, read_file_by_qt
from UI.ui_component import (BFPanel, BFPushButton, BFFileListPanel,
                             BFLogWidget, BFTabNavigation)


class SingleEEGfNIRSAnalysis(QWidget):
    """Interface for single EEG-fNIRS file coupling analysis"""

    def __init__(self):
        """
        Initialize the single file analysis interface

        Creates widgets and layout for:
        - EEG and fNIRS file selection
        - Signal visualization
        - Analysis parameter controls
        - Result display
        """
        super().__init__()
        self.fnirs_data = None
        self.eeg_data = None
        self._initialize_interface()

    def _initialize_interface(self):
        """Configure UI components and layout"""
        self.setWindowTitle('EEG-fNIRS Single Analysis')
        self.resize(1200, 800)

        main_layout = QVBoxLayout()

        # Top panel with input file selection
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout)

        # EEG file section
        eeg_groupbox = BFPanel('Import EEG File')
        top_layout.addWidget(eeg_groupbox)
        self.eeg_button = BFPushButton('Select EEG File')
        self.eeg_button.clicked.connect(self._load_eeg)
        eeg_groupbox.vlayout.addWidget(self.eeg_button)

        # EEG channel selection and visualization
        self.eeg_combo = QComboBox()
        self.eeg_combo.currentIndexChanged.connect(self._plot_eeg)
        eeg_groupbox.vlayout.addWidget(QLabel('EEG Channel:'))
        eeg_groupbox.vlayout.addWidget(self.eeg_combo)

        self.eeg_fig, self.eeg_axis = plt.subplots()
        self.eeg_canvas = FigureCanvas(self.eeg_fig)
        eeg_groupbox.vlayout.addWidget(self.eeg_canvas)

        # fNIRS file section
        fnirs_groupbox = BFPanel('Import fNIRS File')
        top_layout.addWidget(fnirs_groupbox)
        self.fnirs_button = BFPushButton('Select fNIRS File')
        self.fnirs_button.clicked.connect(self._load_fnirs)
        fnirs_groupbox.vlayout.addWidget(self.fnirs_button)

        # fNIRS channel selection and visualization
        self.fnirs_combo = QComboBox()
        self.fnirs_combo.currentIndexChanged.connect(self._plot_fnirs)
        fnirs_groupbox.vlayout.addWidget(QLabel('fNIRS Channel:'))
        fnirs_groupbox.vlayout.addWidget(self.fnirs_combo)

        self.fnirs_fig, self.fnirs_axis = plt.subplots()
        self.fnirs_canvas = FigureCanvas(self.fnirs_fig)
        fnirs_groupbox.vlayout.addWidget(self.fnirs_canvas)

        # Analysis parameters panel
        analysis_panel = BFPanel('Neurovascular Coupling Analysis')
        main_layout.addWidget(analysis_panel)

        parameter_layout = QFormLayout()
        analysis_panel.vlayout.addLayout(parameter_layout)

        # Analysis controls
        self.start_time_input = QLineEdit('0')
        self.end_time_input = QLineEdit('1000')
        parameter_layout.addRow('Analysis Start (s):', self.start_time_input)
        parameter_layout.addRow('Analysis End (s):', self.end_time_input)

        # HRF parameters
        self.tr_spinner = QDoubleSpinBox()
        self.tr_spinner.setValue(1.0)
        self.tr_spinner.setSingleStep(0.1)
        parameter_layout.addRow('TR (Repetition Time):', self.tr_spinner)

        self.oversampling_spinner = QSpinBox()
        self.oversampling_spinner.setValue(1)
        parameter_layout.addRow('Oversampling Factor:', self.oversampling_spinner)

        self.time_length_spinner = QDoubleSpinBox()
        self.time_length_spinner.setValue(32.0)
        self.time_length_spinner.setSingleStep(1.0)
        parameter_layout.addRow('HRF Duration (s):', self.time_length_spinner)

        # EEG processing method
        self.eeg_method_combo = QComboBox()
        self.eeg_method_combo.addItems([
            'Raw Signal Average',
            'Resampled Raw',
            'Average PSD',
            'Alpha Band PSD',
            'Beta Band PSD',
            'Gamma Band PSD',
            'Delta Band PSD',
            'Theta Band PSD'
        ])
        parameter_layout.addRow('EEG Processing Method:', self.eeg_method_combo)

        # Window size
        self.window_size_spinner = QSpinBox()
        self.window_size_spinner.setRange(1, 10000)
        self.window_size_spinner.setValue(100)
        parameter_layout.addRow('Processing Window Size:', self.window_size_spinner)

        # Compute button
        self.compute_button = BFPushButton('Compute Neurovascular Coupling')
        self.compute_button.clicked.connect(self._compute_coupling)
        parameter_layout.addRow(self.compute_button)

        # Results display
        self.result_label = QLabel('Results will appear here')
        self.result_fig, self.result_axis = plt.subplots()
        self.result_canvas = FigureCanvas(self.result_fig)

        analysis_panel.vlayout.addWidget(self.result_label)
        analysis_panel.vlayout.addWidget(self.result_canvas)

        self.setLayout(main_layout)

    def _load_eeg(self):
        """Load EEG data file and populate channel options"""
        self.eeg_data = read_file()
        if self.eeg_data and self.eeg_data['type'] in ('eeg', 'eeg_preprocessed'):
            self.eeg_combo.clear()
            self.eeg_combo.addItems(self.eeg_data['ch_names'])
            self._plot_eeg()

    def _load_fnirs(self):
        """Load fNIRS data file and populate channel options"""
        self.fnirs_data = read_file()
        if self.fnirs_data and self.fnirs_data['type'] in ('fnirs', 'fnirs_preprocessed'):
            self.fnirs_combo.clear()
            self.fnirs_combo.addItems(self.fnirs_data['ch_names'])
            self._plot_fnirs()

    def _plot_eeg(self):
        """Visualize selected EEG channel"""
        if not self.eeg_data:
            return

        channel_index = self.eeg_combo.currentIndex()
        self.eeg_axis.clear()

        # Plot full signal
        signal = self.eeg_data['data'][channel_index]
        self.eeg_axis.plot(signal, linewidth=0.5)
        self.eeg_axis.set_title(f'EEG Signal: {self.eeg_combo.currentText()}')
        self.eeg_axis.set_xlabel('Samples')
        self.eeg_axis.set_ylabel('Amplitude')
        self.eeg_canvas.draw()

    def _plot_fnirs(self):
        """Visualize selected fNIRS channel"""
        if not self.fnirs_data:
            return

        channel_index = self.fnirs_combo.currentIndex()
        self.fnirs_axis.clear()

        # Plot full signal
        signal = self.fnirs_data['data'][channel_index]
        self.fnirs_axis.plot(signal, linewidth=0.5)
        self.fnirs_axis.set_title(f'fNIRS Signal: {self.fnirs_combo.currentText()}')
        self.fnirs_axis.set_xlabel('Samples')
        self.fnirs_axis.set_ylabel('Concentration')
        self.fnirs_canvas.draw()

    def _compute_coupling(self):
        """Perform neurovascular coupling calculation"""
        if not self.eeg_data or not self.fnirs_data:
            QMessageBox.warning(self, "Missing Data", "Please load both EEG and fNIRS files first")
            return

        # Convert method selection to compatible internal code
        method_map = {
            'Raw Signal Average': 'avg_raw',
            'Resampled Raw': 'resample_raw',
            'Average PSD': 'avg_psd',
            'Alpha Band PSD': 'alpha_psd',
            'Beta Band PSD': 'beta_psd',
            'Gamma Band PSD': 'gamma_psd',
            'Delta Band PSD': 'delta_psd',
            'Theta Band PSD': 'theta_psd'
        }
        processing_method = method_map.get(self.eeg_method_combo.currentText(), 'avg_psd')

        # Extract analysis range
        try:
            start_time = float(self.start_time_input.text())
            end_time = float(self.end_time_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Time", "Please enter valid start and end times")
            return

        # Get selected channels
        eeg_channel_idx = self.eeg_combo.currentIndex()
        fnirs_channel_idx = self.fnirs_combo.currentIndex()

        # Extract EEG segment
        eeg_start_idx = int(start_time * self.eeg_data['srate'])
        eeg_end_idx = int(end_time * self.eeg_data['srate'])
        eeg_signal = self.eeg_data['data'][eeg_channel_idx][eeg_start_idx:eeg_end_idx]
        eeg_srate = self.eeg_data['srate']

        # Extract fNIRS segment
        fnirs_start_idx = int(start_time * self.fnirs_data['srate'])
        fnirs_end_idx = int(end_time * self.fnirs_data['srate'])
        fnirs_signal = self.fnirs_data['data'][fnirs_channel_idx][fnirs_start_idx:fnirs_end_idx]
        fnirs_srate = self.fnirs_data['srate']

        # Compute coupling
        try:
            _, correlation = compute_neurovascular_coupling(
                eeg_signal=eeg_signal,
                eeg_srate=eeg_srate,
                fnirs_signal=fnirs_signal,
                fnirs_srate=fnirs_srate,
                window_size=self.window_size_spinner.value(),
                eeg_processing_method=processing_method,
                hrf_tr=self.tr_spinner.value(),
                hrf_oversampling=self.oversampling_spinner.value(),
                hrf_time_length=self.time_length_spinner.value(),
                display_plots=False
            )
        except Exception as e:
            QMessageBox.critical(self, "Computation Error", f"Analysis failed: {str(e)}")
            return

        # Visualize results
        self.result_axis.clear()
        fnirs_normalized = normalize(fnirs_signal)
        self.result_axis.plot(fnirs_normalized, label='Normalized fNIRS')

        # Display correlation value
        self.result_label.setText(
            f'Neurovascular Coupling: Pearson r = {correlation:.3f}'
        )

        self.result_axis.set_title('Normalized Signals Comparison')
        self.result_axis.set_xlabel('Samples')
        self.result_axis.set_ylabel('Normalized Amplitude')
        self.result_axis.legend()
        self.result_canvas.draw()

        QMessageBox.information(self, "Analysis Complete",
                                f"Neurovascular coupling coefficient: {correlation:.3f}")


class BatchEEGfNIRSAnalysis(QWidget):
    """Interface for batch EEG-fNIRS file coupling analysis"""

    def __init__(self):
        """
        Initialize batch processing interface

        Creates widgets for:
        - EEG and fNIRS file selection
        - File matching
        - Analysis parameter controls
        - Log display
        """
        super().__init__()
        self.eeg_files = []
        self.fnirs_files = []
        self.matched_eeg_files = []
        self.matched_fnirs_files = []
        self._initialize_interface()

    def _initialize_interface(self):
        """Configure UI components and layout"""
        self.setWindowTitle('EEG-fNIRS Batch Analysis')
        main_layout = QVBoxLayout(self)

        # File import panels
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout)

        # EEG files panel
        eeg_panel = BFPanel('Import EEG Files')
        eeg_panel.vlayout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(eeg_panel)
        self.eeg_file_list = BFFileListPanel()
        eeg_panel.vlayout.addWidget(self.eeg_file_list)

        # fNIRS files panel
        fnirs_panel = BFPanel('Import fNIRS Files')
        fnirs_panel.vlayout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(fnirs_panel)
        self.fnirs_file_list = BFFileListPanel()
        fnirs_panel.vlayout.addWidget(self.fnirs_file_list)

        # File matching button
        self.match_button = BFPushButton('Match EEG and fNIRS Files')
        self.match_button.clicked.connect(self._match_files)
        self.match_button.setFixedWidth(300)
        main_layout.addWidget(self.match_button)

        # Analysis panel
        analysis_panel = BFPanel('Analysis Parameters')
        main_layout.addWidget(analysis_panel)

        param_layout = QFormLayout()
        analysis_panel.vlayout.addLayout(param_layout)

        # Time range selection
        self.start_input = QLineEdit('0')
        self.end_input = QLineEdit('1000')
        param_layout.addRow('Analysis Start (s):', self.start_input)
        param_layout.addRow('Analysis End (s):', self.end_input)

        # HRF parameters
        self.tr_spinner = QDoubleSpinBox()
        self.tr_spinner.setValue(1.0)
        param_layout.addRow('TR (Repetition Time):', self.tr_spinner)

        self.oversampling_spinner = QSpinBox()
        self.oversampling_spinner.setValue(1)
        param_layout.addRow('Oversampling Factor:', self.oversampling_spinner)

        self.time_length_spinner = QDoubleSpinBox()
        self.time_length_spinner.setValue(32.0)
        param_layout.addRow('HRF Duration (s):', self.time_length_spinner)

        # EEG processing method
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            'Raw Signal Average',
            'Resampled Raw',
            'Average PSD',
            'Alpha Band PSD',
            'Beta Band PSD',
            'Gamma Band PSD',
            'Delta Band PSD',
            'Theta Band PSD'
        ])
        param_layout.addRow('EEG Processing Method:', self.method_combo)

        # Window size
        self.window_size_spinner = QSpinBox()
        self.window_size_spinner.setRange(1, 10000)
        self.window_size_spinner.setValue(100)
        param_layout.addRow('Processing Window Size:', self.window_size_spinner)

        # Output directory
        self.save_button = BFPushButton('Select Output Directory')
        self.save_button.clicked.connect(self._choose_output_directory)
        self.save_path_display = QLineEdit()
        self.save_path_display.setReadOnly(True)
        param_layout.addRow(self.save_button, self.save_path_display)

        # Run analysis button
        self.run_button = BFPushButton('Run Batch Analysis')
        self.run_button.clicked.connect(self._start_analysis)
        param_layout.addRow(self.run_button)

        # Log display
        self.log_panel = BFLogWidget()
        analysis_panel.vlayout.addWidget(self.log_panel)
        analysis_panel.vlayout.setStretch(0, 1)
        analysis_panel.vlayout.setStretch(1, 3)

    def _has_valid_files(self):
        """Check if valid files are available"""
        has_eeg = bool(self.eeg_file_list.get_files_by_extension(['mat', 'bdf', 'edf', 'nirs']))
        has_fnirs = bool(self.fnirs_file_list.get_files_by_extension(['mat', 'bdf', 'edf', 'nirs']))

        if not has_eeg or not has_fnirs:
            QMessageBox.warning(self, "Missing Files",
                                "Please import both EEG and fNIRS files")
            return False
        return True

    def _match_files(self):
        """Match EEG and fNIRS files by filename"""
        if not self._has_valid_files():
            return

        self.matched_eeg_files.clear()
        self.matched_fnirs_files.clear()

        eeg_files = self.eeg_file_list.file_list
        fnirs_files = self.fnirs_file_list.file_list

        # Match files by name
        for eeg_file in eeg_files:
            eeg_base = getFileNameWithoutSuffix(eeg_file)
            for fnirs_file in fnirs_files:
                fnirs_base = getFileNameWithoutSuffix(fnirs_file)
                if are_filenames_equal(eeg_base, fnirs_base):
                    if eeg_file not in self.matched_eeg_files:
                        self.matched_eeg_files.append(eeg_file)
                    if fnirs_file not in self.matched_fnirs_files:
                        self.matched_fnirs_files.append(fnirs_file)

        # Display matched files
        self.eeg_file_list.show_files(self.matched_eeg_files)
        self.fnirs_file_list.show_files(self.matched_fnirs_files)

        # Check if all files matched
        if len(self.matched_eeg_files) < len(eeg_files):
            self.log_panel.add_log("Some EEG files could not be matched", is_warning=True)
        if len(self.matched_fnirs_files) < len(fnirs_files):
            self.log_panel.add_log("Some fNIRS files could not be matched", is_warning=True)

        if not self.matched_eeg_files or not self.matched_fnirs_files:
            QMessageBox.warning(self, "No Matches",
                                "Could not find matching EEG-fNIRS file pairs")
            return False

        return True

    def _start_analysis(self):
        """Start batch analysis in a separate thread"""
        if not self._match_files():
            return

        if not self.save_path_display.text():
            QMessageBox.warning(self, "Output Required",
                                "Please select an output directory")
            return

        self.log_panel.add_log("Starting batch processing...")

        # Start processing in a separate thread
        thread = threading.Thread(target=self._process_files)
        thread.start()

    def _choose_output_directory(self):
        """Select output directory for results"""
        directory = QFileDialog.getExistingDirectory(self, 'Select Output Directory')
        if directory:
            self.save_path_display.setText(directory)

    def _process_files(self):
        """Process matched file pairs in batch mode"""
        self.log_panel.add_log("Beginning batch processing", is_start=True)
        success_count = 0

        # Process each matched pair
        for eeg_path, fnirs_path in zip(self.matched_eeg_files, self.matched_fnirs_files):
            try:
                # Get base name for output
                base_name = getFileNameWithoutSuffix(eeg_path)
                self.log_panel.add_log(f"Processing: {base_name}")

                # Load EEG data
                eeg_data, _ = read_file_by_qt(self, [eeg_path])
                if not eeg_data:
                    self.log_panel.add_log(f"Error loading EEG: {eeg_path}", is_error=True)
                    continue

                # Load fNIRS data
                fnirs_data, _ = read_file_by_qt(self, [fnirs_path])
                if not fnirs_data:
                    self.log_panel.add_log(f"Error loading fNIRS: {fnirs_path}", is_error=True)
                    continue

                # Apply time range
                start_time = float(self.start_input.text())
                end_time = float(self.end_input.text())

                eeg_start = int(start_time * eeg_data['srate'])
                eeg_end = int(end_time * eeg_data['srate'])
                eeg_data['data'] = eeg_data['data'][:, eeg_start:eeg_end]

                fnirs_start = int(start_time * fnirs_data['srate'])
                fnirs_end = int(end_time * fnirs_data['srate'])
                fnirs_data['data'] = fnirs_data['data'][:, fnirs_start:fnirs_end]

                # Map method selection to internal code
                method_map = {
                    'Raw Signal Average': 'avg_raw',
                    'Resampled Raw': 'resample_raw',
                    'Average PSD': 'avg_psd',
                    'Alpha Band PSD': 'alpha_psd',
                    'Beta Band PSD': 'beta_psd',
                    'Gamma Band PSD': 'gamma_psd',
                    'Delta Band PSD': 'delta_psd',
                    'Theta Band PSD': 'theta_psd'
                }
                processing_method = method_map.get(self.method_combo.currentText(), 'avg_psd')

                # Prepare output path
                output_dir = self.save_path_display.text()
                results_dir = os.path.join(output_dir, 'coupling_results')
                os.makedirs(results_dir, exist_ok=True)
                output_path = os.path.join(results_dir, f"{base_name}_coupling.zip")

                # Run analysis
                compute_neurovascular_coupling_by_dict(
                    eeg_dict=eeg_data,
                    fnirs_dict=fnirs_data,
                    window_size=self.window_size_spinner.value(),
                    eeg_processing_method=processing_method,
                    hrf_tr=self.tr_spinner.value(),
                    hrf_oversampling=self.oversampling_spinner.value(),
                    hrf_time_length=self.time_length_spinner.value(),
                    save_results=True,
                    output_path=output_path
                )

                success_count += 1
                self.log_panel.add_log(f"Completed: {base_name}", is_success=True)

            except Exception as e:
                self.log_panel.add_log(f"Error processing {base_name}: {str(e)}", is_error=True)

        # Final report
        total_count = len(self.matched_eeg_files)
        if success_count == total_count:
            self.log_panel.add_log(f"Batch completed ({success_count}/{total_count} files)", is_end=True)
        else:
            self.log_panel.add_log(
                f"Batch completed with errors ({success_count}/{total_count} files)",
                is_warning=True
            )


class NeurovascularCouplingDialog(BFTabNavigation):
    """Main application interface for neurovascular coupling analysis"""

    def __init__(self):
        """Initialize the application interface"""
        super().__init__()
        self.setWindowTitle("Neurovascular Coupling Analysis Suite")

        # Create analysis interfaces
        self.single_analysis = SingleEEGfNIRSAnalysis()
        self.batch_analysis = BatchEEGfNIRSAnalysis()
        self.bids_interface = QWidget()  # Placeholder for future BIDS integration

        # Add tabs to interface
        self.add_tab('Single Analysis', self.single_analysis)
        self.add_tab('Batch Analysis', self.batch_analysis)
        self.add_tab('BIDS Integration', self.bids_interface)

        # Configure window
        self.center_on_screen()
        self.resize(1200, 800)

    def center_on_screen(self):
        """Center the window on the primary display"""
        screen_geometry = QApplication.primaryScreen().geometry()
        center_x = (screen_geometry.width() - self.width()) // 2
        center_y = (screen_geometry.height() - self.height()) // 2
        self.move(center_x, center_y)


if __name__ == '__main__':
    # Launch application
    app = QApplication(sys.argv)
    analysis_suite = NeurovascularCouplingDialog()
    analysis_suite.show()
    sys.exit(app.exec_())