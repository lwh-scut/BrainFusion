# -*- coding: utf-8 -*-
# @Time    : 2024/3/5 9:08
# @Author  : Li WenHao
# @Site    : South China University of Technology
# @File    : pipeLine_with_dialog.py
# @Software: PyCharm 
# @Comment :
import ast
import os
import sys
import threading

import matplotlib
import mne.io
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QSpinBox, QDoubleSpinBox, \
    QComboBox, QCheckBox, QPushButton, QFormLayout, QVBoxLayout, QLineEdit, QFileDialog, QGroupBox, QDialogButtonBox, \
    QDialog, QMessageBox, QRadioButton, QFrame, QListWidget, QGridLayout, QHBoxLayout, QListWidgetItem, QScrollArea

from BrainFusion.io.File_IO import read_file_by_qt, read_file, read_file_by_path, create_data_dict, read_file_by_mne, \
    save_file
from BrainFusion.pipeLine.pipeLine import eeg_power_spectral_density, \
    eeg_multiscale_entropy, \
    sample_entropy, eeg_microstate, root_mean_square, variance, mean_absolute_value, zero_crossing, \
    wavelet_transform, wavelet_packet_energy, short_time_Fourier_transform, continuous_wavelet_transform, \
    hjorth_parameters, aperiodic_parameters, local_network_features, global_network_features
from BrainFusion.pipeLine.preprocessing import eeg_preprocessing, create_epoch, fnirs_preprocessing, emg_preprocessing, \
    ecg_preprocessing, eeg_preprocessing_by_dict, fnirs_preprocessing_by_raw
from BrainFusion.utils.transform import raw_to_dict, dict_to_info
from UI.ui_component import BFGroupBox, BFPushButton, ProcessDialog

"""
EEG Preprocessing Module.

Provides configuration dialogs for EEG preprocessing operations.
"""

import ast
import os
import sys
import threading

import matplotlib
import mne.io
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QSpinBox, QDoubleSpinBox, \
    QComboBox, QCheckBox, QPushButton, QFormLayout, QVBoxLayout, QLineEdit, QFileDialog, QGroupBox, QDialogButtonBox, \
    QDialog, QMessageBox, QRadioButton, QFrame, QListWidget, QGridLayout, QHBoxLayout, QListWidgetItem, QScrollArea

from BrainFusion.io.File_IO import read_file_by_qt, read_file, read_file_by_path, create_data_dict, read_file_by_mne, \
    save_file
from BrainFusion.pipeLine.pipeLine import eeg_power_spectral_density, \
    eeg_multiscale_entropy, \
    sample_entropy, eeg_microstate, root_mean_square, variance, mean_absolute_value, zero_crossing, \
    wavelet_transform, wavelet_packet_energy, short_time_Fourier_transform, continuous_wavelet_transform, \
    hjorth_parameters, aperiodic_parameters, local_network_features, global_network_features
from BrainFusion.pipeLine.preprocessing import eeg_preprocessing, create_epoch, fnirs_preprocessing, emg_preprocessing, \
    ecg_preprocessing, eeg_preprocessing_by_dict, fnirs_preprocessing_by_raw
from BrainFusion.utils.transform import raw_to_dict, dict_to_info
from UI.ui_component import BFGroupBox, BFPushButton, ProcessDialog

matplotlib.use('QtAgg')


class FeatureDialog(QMainWindow):
    """Base dialog for feature configuration and processing."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize feature dialog.

        :param str feature_name: Name of the feature
        :param QWidget parent: Parent widget
        """
        super().__init__(parent)
        self.feature_name = feature_name
        self.data = None
        self.filePath = None
        self.savefileName = None
        self.is_folder = False
        self.folderPath = None
        self.savefolderName = None
        self.import_file_list = None
        self.save_file_list = []
        self.init_ui()

    def init_ui(self):
        """Initialize UI components."""
        self.central_widget = QWidget(self)
        self.resize(600, 600)
        scroll_area = QScrollArea(self)
        scroll_area.setWidget(self.central_widget)
        scroll_area.setWidgetResizable(True)
        self.setCentralWidget(scroll_area)
        self.layout = QVBoxLayout(self.central_widget)
        self.left_layout = QFormLayout()
        self.layout.addLayout(self.left_layout)

        # Import data section
        self.import_data_groupbox = BFGroupBox('Data Import Parameters')
        self.import_data_layout = QFormLayout(self.import_data_groupbox)
        self.import_file_button = BFPushButton('Import File')
        self.import_file_lineedit = QLineEdit()
        self.import_file_lineedit.setReadOnly(True)
        self.import_data_layout.addRow(self.import_file_button, self.import_file_lineedit)
        self.import_folder_button = BFPushButton('Import Folder')
        self.import_folder_lineedit = QLineEdit()
        self.import_folder_lineedit.setReadOnly(True)
        self.import_data_layout.addRow(self.import_folder_button, self.import_folder_lineedit)
        self.left_layout.addRow(self.import_data_groupbox)

        # Export data section
        self.save_data_groupbox = BFGroupBox('Data Export Parameters')
        self.save_data_layout = QFormLayout(self.save_data_groupbox)

        self.output_folder_button = BFPushButton('Output Folder')
        self.output_folder_lineedit = QLineEdit()
        self.output_folder_lineedit.setReadOnly(True)

        self.export_info_checkbox = QCheckBox('Export metadata as separate file')

        self.file_type_combobox = QComboBox()
        self.file_type_combobox.addItems(['xlsx', 'tsv', 'mat'])
        self.save_data_layout.addRow('File format:', self.file_type_combobox)
        self.save_data_layout.addRow(self.output_folder_button, self.output_folder_lineedit)
        self.save_data_layout.addRow(self.export_info_checkbox, QLabel(''))

        self.left_layout.addRow(self.save_data_groupbox)

        # Execute processing
        self.bnt_run = BFPushButton('Run')
        self.bnt_run.clicked.connect(self.start_task)
        self.layout.addWidget(self.bnt_run)

        # Connect signals
        self.output_folder_button.clicked.connect(self.choose_output_folder)
        self.import_file_button.clicked.connect(self.choose_import_file)
        self.import_folder_button.clicked.connect(self.choose_import_folder)

    def prepare(self):
        """Prepare output directory structure and file paths."""
        if self.is_folder:
            if self.output_folder_lineedit.text():
                output_path = os.path.join(self.output_folder_lineedit.text(), self.savefolderName)
            else:
                output_path = os.path.join(os.path.dirname(self.folderPath), self.savefolderName)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            if os.path.isdir(self.folderPath):
                for root, dirs, files in os.walk(self.folderPath):
                    relative_path = os.path.relpath(root, self.folderPath)
                    target_path = os.path.join(output_path, relative_path)
                    if not os.path.exists(target_path):
                        os.makedirs(target_path)
            file_type = self.file_type_combobox.currentText()
            for file_name in self.import_file_list:
                if isinstance(file_name, tuple):
                    self.save_file_list.append(
                        file_name[0].replace(self.folderPath, output_path).rsplit('.', 1)[0] + '.' + file_type)
                else:
                    self.save_file_list.append(
                        file_name.replace(self.folderPath, output_path).rsplit('.', 1)[0] + '.' + file_type)
            return output_path
        else:
            if self.output_folder_lineedit.text():
                output_path = os.path.join(self.output_folder_lineedit.text(),
                                           self.savefileName + self.file_type_combobox.currentText())
            else:
                output_path = os.path.join(os.path.dirname(self.filePath[0]),
                                           self.savefileName + self.file_type_combobox.currentText())
            return output_path

    def show_message(self, title, message):
        """Display informational message dialog.

        :param str title: Dialog title
        :param str message: Message content
        """
        self.message = QMessageBox(self)
        self.message.setWindowTitle(title)
        self.message.setText(message)
        self.message.show()

    def run(self):
        """Execute processing task (to be implemented in subclasses)."""
        pass

    def start_task(self):
        """Initiate processing task in separate thread."""
        if self.data:
            if self.is_folder:
                self.process_dialog = ProcessDialog(self.import_file_list)
            else:
                self.process_dialog = ProcessDialog([self.filePath[0]])
            self.process_dialog.setTitle(f'{self.feature_name} Processing')
            self.process_dialog.show()
            self.close()
            thread = threading.Thread(target=self.run)
            thread.start()
        else:
            self.show_message('Warning', 'Please import data first.')

    def choose_output_folder(self):
        """Select output directory for processed data."""
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Output Folder', '.')
        if folder_path:
            self.output_folder_lineedit.setText(folder_path)

            def choose_import_file(self):
                """Import single data file."""
                self.data, self.filePath = read_file_by_qt(self)
                if self.filePath and self.data:
                    if len(self.filePath) == 1:
                        self.import_file_lineedit.setText(self.filePath[0])
                    elif len(self.filePath) == 2:
                        self.import_file_lineedit.setText(self.filePath[0] + " " + self.filePath[1])
                    self.import_folder_lineedit.setText('')
                    if self.filePath[0]:
                        base_name = os.path.basename(self.filePath[0])
                        name, ext = os.path.splitext(base_name)
                        self.savefileName = f'{name}_' + self.feature_name + '.'
                    self.is_folder = False

            def choose_import_folder(self):
                """Import folder with data files."""

                def get_folder_contents(folder_path):
                    file_list = []
                    for root, dirs, files in os.walk(folder_path):
                        if 'data.bdf' in files and 'evt.bdf' in files:
                            data_path = os.path.join(root, 'data.bdf')
                            evt_path = os.path.join(root, 'evt.bdf')
                            file_list.append((data_path, evt_path))
                        else:
                            for file in files:
                                file_list.append(os.path.join(root, file))
                    return file_list

                folder_dialog = QFileDialog()
                folder_dialog.setFileMode(QFileDialog.Directory)
                folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
                if folder_dialog.exec_():
                    self.folderPath = folder_dialog.selectedFiles()[0]
                    self.import_folder_lineedit.setText(self.folderPath)
                    self.import_file_lineedit.setText('')
                    base_name = os.path.basename(self.folderPath)
                    self.savefolderName = f'{base_name}_' + self.feature_name
                    self.import_file_list = get_folder_contents(self.folderPath)
                    if self.import_file_list:
                        self.data = read_file_by_path(self.import_file_list[0])
                    else:
                        QMessageBox.warning(None, 'Warning', 'Folder is empty', QMessageBox.Ok)
                self.is_folder = True

class ChannelsDialog(QDialog):
    """Dialog for channel selection."""

    def __init__(self, channel_list, parent=None):
        """
    Initialize channel selector.

    :param list channel_list: Available channel names
    :param QWidget parent: Parent widget
    """
        super().__init__(parent)
        self.setWindowTitle('Select Channels')
        self.setGeometry(400, 400, 300, 800)
        self.center_widget = QWidget()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.center_widget)
        self.scroll_area.setWidgetResizable(True)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.scroll_area)
        layout = QVBoxLayout(self.center_widget)

        self.checkbox_dict = {}
        for channel in channel_list:
            checkbox = QCheckBox(channel)
            self.checkbox_dict[channel] = checkbox
            layout.addWidget(checkbox)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

    def get_selected_channels(self):
        """Get selected channel names.

    :return: List of selected channels
    :rtype: list
    """
        return [channel for channel, checkbox in self.checkbox_dict.items() if checkbox.isChecked()]

class EEGPreprocessingConfigWindow(FeatureDialog):
    """Configuration UI for EEG preprocessing."""

    def __init__(self, feature_name, parent=None):
        """
    Initialize EEG preprocessing dialog.

    :param str feature_name: Preprocessing operation name
    :param QWidget parent: Parent widget
    """
        super().__init__(feature_name, parent)

        # Sampling rate configuration
        self.fs_spinbox = QDoubleSpinBox()
        self.fs_spinbox.setRange(1, 10000)
        self.fs_spinbox.setValue(1000)
        self.left_layout.addRow('Sampling Rate (Hz):', self.fs_spinbox)

        # Channel rejection
        btn_remove_bad_channels = BFPushButton('Exclude Bad Channels')
        btn_remove_bad_channels.clicked.connect(self.show_remove_bad_channels_dialog)
        self.bad_channels_lineedit = QLineEdit(self)
        self.bad_channels_lineedit.setReadOnly(True)
        self.left_layout.addRow(btn_remove_bad_channels, self.bad_channels_lineedit)

        # Bandpass filter configuration
        filter_groupbox = BFGroupBox('Bandpass Filter')
        filter_layout = QFormLayout(filter_groupbox)

        self.lowcut_spinbox = QDoubleSpinBox()
        self.lowcut_spinbox.setRange(0, 10000)
        self.lowcut_spinbox.setValue(2)
        filter_layout.addRow('Low Cutoff (Hz):', self.lowcut_spinbox)

        self.highcut_spinbox = QDoubleSpinBox()
        self.highcut_spinbox.setRange(0, 10000)
        self.highcut_spinbox.setValue(80)
        filter_layout.addRow('High Cutoff (Hz):', self.highcut_spinbox)

        self.filter_combobox = QComboBox()
        self.filter_combobox.addItems(['Butterworth', 'Bessel', 'Chebyshev'])
        filter_layout.addRow('Filter Type:', self.filter_combobox)

        self.filterorder_spinbox = QSpinBox()
        self.filterorder_spinbox.setRange(0, 20)
        self.filterorder_spinbox.setValue(4)
        filter_layout.addRow('Filter Order:', self.filterorder_spinbox)

        self.left_layout.addRow(filter_groupbox)

        # Notch filter configuration
        notch_groupbox = BFGroupBox('Notch Filter')
        notch_layout = QFormLayout(notch_groupbox)

        self.notch_spinbox = QDoubleSpinBox()
        self.notch_spinbox.setRange(0, 10000)
        self.notch_spbox.setValue(50.0)
        notch_layout.addRow('Notch Frequency (Hz):', self.notch_spinbox)

        self.Q_spinbox = QSpinBox()
        self.Q_spinbox.setRange(0, 1000)
        self.Q_spinbox.setValue(200)
        notch_layout.addRow('Quality Factor:', self.Q_spinbox)

        self.left_layout.addRow(notch_groupbox)

        # EOG regression section
        eog_groupbox = BFGroupBox('EOG Artifact Removal')
        eog_layout = QFormLayout(eog_groupbox)

        self.eog_checkbox = QCheckBox('Apply EOG Regression')
        self.eog_checkbox.setChecked(False)
        eog_layout.addRow(self.eog_checkbox)

        btn_eog_chan = QPushButton('Select EOG Channels')
        btn_eog_chan.clicked.connect(self.show_eog_channels_dialog)
        self.eog_chan_lineedit = QLineEdit()
        eog_layout.addRow(btn_eog_chan, self.eog_chan_lineedit)

        self.left_layout.addRow(eog_groupbox)

        # ICA configuration
        ica_groupbox = BFGroupBox('ICA Configuration')
        ica_layout = QFormLayout(ica_groupbox)

        self.is_ica_checkbox = QCheckBox('Enable ICA')
        self.is_ica_checkbox.setChecked(True)
        ica_layout.addRow(self.is_ica_checkbox)

        self.ica_method_label = QLabel('ICA Method')
        ica_layout.addRow(self.ica_method_label)

        self.ica_method_combobox = QComboBox()
        self.ica_method_combobox.addItems(['fastica', 'infomax', 'picard'])
        ica_layout.addRow(self.ica_method_label, self.ica_method_combobox)

        self.ica_components_spinbox = QSpinBox()
        self.ica_components_spinbox.setRange(0, 1000)
        ica_layout.addRow('ICA Components', self.ica_components_spinbox)

        self.left_layout.addRow(ica_groupbox)

        # Reference configuration
        ref_groupbox = BFGroupBox('Re-reference Settings')
        ref_layout = QFormLayout(ref_groupbox)
        btn_ref_chan = QPushButton('Select Reference Channels')
        btn_ref_chan.clicked.connect(self.show_ref_channels_dialog)
        self.ref_chan_lineedit = QLineEdit()
        ref_layout.addRow(btn_ref_chan, self.ref_chan_lineedit)

        self.ref_method_combobox = QComboBox()
        self.ref_method_combobox.addItems(['average reference', 'other'])
        ref_layout.addRow('Re-reference Method:', self.ref_method_combobox)

        self.left_layout.addRow(ref_groupbox)
        self.left_layout.addRow(self.save_data_groupbox)

        # Connect signals
        self.is_ica_checkbox.stateChanged.connect(self.update_ica_controls_visibility)

        # UI setup
        self.setGeometry(300, 300, 600, 800)
        self.setWindowTitle('EEG Preprocessing Settings')
        self.file_type_combobox.clear()
        self.file_type_combobox.addItems(['mat'])

    def update_ica_controls_visibility(self):
        """Toggle visibility of ICA settings based on checkbox state."""
        is_ica_enabled = self.is_ica_checkbox.isChecked()
        self.ica_method_label.setVisible(is_ica_enabled)
        self.ica_method_combobox.setVisible(is_ica_enabled)

    def show_remove_bad_channels_dialog(self):
        """Show dialog to select channels to exclude."""
        if self.data is not None:
            dialog = ChannelsDialog(self.data['ch_names'], self)
            result = dialog.exec_()

            if result == QDialog.Accepted:
                selected_channels = dialog.get_selected_channels()
                self.bad_channels_lineedit.setText(', '.join(selected_channels))
                self.ica_components_spinbox.setValue(
                    self.ica_components_spinbox.value() - len(selected_channels)
                )
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)

    def show_ref_channels_dialog(self):
        """Show dialog to select reference channels."""
        if self.data is not None:
            dialog = ChannelsDialog(self.data['ch_names'], self)
            result = dialog.exec_()

            if result == QDialog.Accepted:
                selected_channels = dialog.get_selected_channels()
                self.ref_chan_lineedit.setText(', '.join(selected_channels))
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)

    def show_eog_channels_dialog(self):
        """Show dialog to select EOG channels."""
        if self.data is not None:
            dialog = ChannelsDialog(self.data['ch_names'], self)
            result = dialog.exec_()

            if result == QDialog.Accepted:
                selected_channels = dialog.get_selected_channels()
                self.eog_chan_lineedit.setText(', '.join(selected_channels))
                self.ica_components_spinbox.setValue(
                    self.data['nchan'] - len(selected_channels)
                )
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)

    def run(self):
        """Execute EEG preprocessing pipeline."""
        if self.data or self.import_file_list:
            # Get parameters
            bad_channels = self.bad_channels_lineedit.text().split(',')
            bad_channels = [chan.strip() for chan in bad_channels]
            fs = self.fs_spinbox.value()
            lowcut = self.lowcut_spinbox.value()
            highcut = self.highcut_spinbox.value()
            filter_type = self.filter_combobox.currentText()
            filter_order = self.filterorder_spinbox.value()
            notch_freq = self.notch_spinbox.value()
            Q_factor = self.Q_spinbox.value()
            is_ica_enabled = self.is_ica_checkbox.isChecked()
            ica_method = self.ica_method_combobox.currentText() if is_ica_enabled else None
            ica_components = self.ica_components_spinbox.value()
            eog_regression = self.eog_checkbox.isChecked()
            eog_channels = self.eog_chan_lineedit.text().split(',')
            eog_channels = [chan.strip() for chan in eog_channels]
            ref_chan_list = self.ref_chan_lineedit.text().split(',')
            ref_chan_list = [chan.strip() for chan in ref_chan_list]
            ref_method = self.ref_method_combobox.currentText()
            output_file = self.prepare()
            file_type = self.file_type_combobox.currentText()
            is_export_info = self.export_info_checkbox.isChecked()

            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    eeg_preprocessing(
                        data=data['data'],
                        chan_list=data['ch_names'],
                        fs=fs,
                        events=data['events'],
                        bad_channels=bad_channels,
                        lowcut=lowcut,
                        highcut=highcut,
                        montage=data['montage'],
                        filter_order=filter_order,
                        filter=filter_type,
                        north_f=notch_freq,
                        Q=Q_factor,
                        eog_regression=eog_regression,
                        eog_channels=eog_channels,
                        is_ICA=is_ica_enabled,
                        ICA_component=ica_components,
                        ICA_method=ica_method,
                        is_ref=True,
                        ref_chan=ref_chan_list,
                        is_save=True,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if is_export_info:
                        info_file_name = self.save_file_list[i].rsplit('.', 1)[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Completed")
            else:
                eeg_preprocessing(
                    data=self.data['data'],
                    chan_list=self.data['ch_names'],
                    fs=fs,
                    events=self.data['events'],
                    bad_channels=bad_channels,
                    lowcut=lowcut,
                    highcut=highcut,
                    montage=self.data['montage'],
                    filter_order=filter_order,
                    filter=filter_type,
                    north_f=notch_freq,
                    Q=Q_factor,
                    eog_regression=eog_regression,
                    eog_channels=eog_channels,
                    is_ICA=is_ica_enabled,
                    ICA_component=ica_components,
                    ICA_method=ica_method,
                    is_ref=True,
                    ref_chan=ref_chan_list,
                    is_save=True,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if is_export_info:
                    info_file_name = output_file.rsplit('.', 1)[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Completed")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)

    def choose_import_file(self):
        """Handle file import with parameter initialization."""
        self.data, self.filePath = read_file_by_qt(self)
        if self.filePath and self.data:
            if len(self.filePath) == 1:
                self.import_file_lineedit.setText(self.filePath[0])
            elif len(self.filePath) == 2:
                self.import_file_lineedit.setText(self.filePath[0] + " " + self.filePath[1])
            self.import_folder_lineedit.setText('')
            self.fs_spinbox.setValue(self.data['srate'])
            self.ica_components_spinbox.setValue(self.data['nchan'])
            self.ica_components_spinbox.setRange(0, self.data['nchan'])
            self.highcut_spinbox.setRange(0, self.data['srate'])
            self.lowcut_spinbox.setRange(0, self.data['srate'])
            self.notch_spinbox.setRange(0, self.data['srate'])
            if self.filePath[0]:
                base_name = os.path.basename(self.filePath[0])
                name, ext = os.path.splitext(base_name)
                self.savefileName = self.feature_name + f'_{name}.'
            self.is_folder = False

class EEGPreprocessingByRawConfigWindow(FeatureDialog):
    """Configuration UI for EEG preprocessing using raw data format."""

    def __init__(self, feature_name, parent=None):
        """
    Initialize raw data preprocessing dialog.

    :param str feature_name: Preprocessing operation name
    :param QWidget parent: Parent widget
    """
        super().__init__(feature_name, parent)

        # Sampling rate and montage
        self.fs_spinbox = QDoubleSpinBox()
        self.fs_spinbox.setRange(1, 10000)
        self.fs_spinbox.setValue(1000)
        self.left_layout.addRow('Sampling Rate (Hz):', self.fs_spinbox)

        self.montage_combox = QComboBox()
        self.montage_combox.addItems(['standard_1020', 'standard_1005', 'none'])
        self.left_layout.addRow('Channel Montage:', self.montage_combox)

        # Channel rejection
        btn_remove_bad_channels = BFPushButton('Exclude Bad Channels')
        btn_remove_bad_channels.clicked.connect(self.show_remove_bad_channels_dialog)
        self.bad_channels_lineedit = QLineEdit(self)
        self.bad_channels_lineedit.setReadOnly(True)
        self.left_layout.addRow(btn_remove_bad_channels, self.bad_channels_lineedit)

        # Filter configuration
        filter_groupbox = BFGroupBox('Bandpass Filter')
        filter_layout = QFormLayout(filter_groupbox)

        self.lowcut_spinbox = QDoubleSpinBox()
        self.lowcut_spinbox.setRange(0, 10000)
        self.lowcut_spinbox.setValue(2)
        filter_layout.addRow('Low Cutoff (Hz):', self.lowcut_spinbox)

        self.highcut_spinbox = QDoubleSpinBox()
        self.highcut_spinbox.setRange(0, 10000)
        self.highcut_spinbox.setValue(80)
        filter_layout.addRow('High Cutoff (Hz):', self.highcut_spinbox)

        self.left_layout.addRow(filter_groupbox)

        # Notch filter
        notch_groupbox = BFGroupBox('Notch Filter')
        notch_layout = QFormLayout(notch_groupbox)

        self.notch_spinbox = QDoubleSpinBox()
        self.notch_spinbox.setRange(0, 10000)
        self.notch_spinbox.setValue(50.0)
        notch_layout.addRow('Notch Frequency (Hz):', self.notch_spinbox)

        self.left_layout.addRow(notch_groupbox)

        # Denoising
        denoise_groupbox = BFGroupBox('Artifact Removal')
        denoise_layout = QFormLayout(denoise_groupbox)

        self.rm_distortion_checkbox = QCheckBox('Remove distortion segments')
        self.rm_distortion_checkbox.setChecked(True)
        denoise_layout.addRow(self.rm_distortion_checkbox)
        self.left_layout.addRow(denoise_groupbox)

        # EOG regression
        eog_groupbox = BFGroupBox('EOG Artifact Removal')
        eog_layout = QFormLayout(eog_groupbox)

        self.eog_checkbox = QCheckBox('Apply EOG Regression')
        self.eog_checkbox.setChecked(False)
        eog_layout.addRow(self.eog_checkbox)

        btn_eog_chan = QPushButton('Select EOG Channels')
        btn_eog_chan.clicked.connect(self.show_eog_channels_dialog)
        self.eog_chan_lineedit = QLineEdit()
        eog_layout.addRow(btn_eog_chan, self.eog_chan_lineedit)

        self.left_layout.addRow(eog_groupbox)

        # ICA configuration
        ica_groupbox = BFGroupBox('ICA Configuration')
        ica_layout = QFormLayout(ica_groupbox)

        self.is_ica_checkbox = QCheckBox('Enable ICA')
        self.is_ica_checkbox.setChecked(True)
        ica_layout.addRow(self.is_ica_checkbox)

        self.ica_method_label = QLabel('ICA Method')
        ica_layout.addRow(self.ica_method_label)

        self.ica_method_combobox = QComboBox()
        self.ica_method_combobox.addItems(['fastica', 'infomax', 'picard'])
        ica_layout.addRow(self.ica_method_label, self.ica_method_combobox)

        self.ica_components_spinbox = QSpinBox()
        self.ica_components_spinbox.setRange(0, 1000)
        ica_layout.addRow('ICA Components', self.ica_components_spinbox)

        self.left_layout.addRow(ica_groupbox)

        # Reference configuration
        ref_groupbox = BFGroupBox('Re-reference Settings')
        ref_layout = QFormLayout(ref_groupbox)
        btn_ref_chan = QPushButton('Select Reference Channels')
        btn_ref_chan.clicked.connect(self.show_ref_channels_dialog)
        self.ref_chan_lineedit = QLineEdit()
        ref_layout.add(btn_ref_chan, self.ref_chan_lineedit)

        self.ref_method_combobox = QComboBox()
        self.ref_method_combobox.addItems(['average reference', 'other'])
        ref_layout.addRow('Re-reference Method:', self.ref_method_combobox)

        self.left_layout.addRow(ref_groupbox)
        self.left_layout.addRow(self.save_data_groupbox)

        # Connect signals
        self.is_ica_checkbox.stateChanged.connect(self.update_ica_controls_visibility)

        # UI setup
        self.setGeometry(300, 300, 600, 800)
        self.setWindowTitle('EEG Raw Preprocessing Settings')
        self.file_type_combobox.clear()
        self.file_type_combobox.addItems(['mat'])

    def update_ica_controls_visibility(self):
        """Toggle visibility of ICA settings based on checkbox state."""
        is_ica_enabled = self.is_ica_checkbox.isChecked()
        self.ica_method_label.setVisible(is_ica_enabled)
        self.ica_method_combobox.setVisible(is_ica_enabled)

    def show_remove_bad_channels_dialog(self):
        """Show dialog to select channels to exclude."""
        if self.data is not None:
            dialog = ChannelsDialog(self.data['ch_names'], self)
            result = dialog.exec_()

            if result == QDialog.Accepted:
                selected_channels = dialog.get_selected_channels()
                self.bad_channels_lineedit.setText(', '.join(selected_channels))
                self.ica_components_spinbox.setValue(
                    self.ica_components_spinbox.value() - len(selected_channels)
                )
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)

    def show_ref_channels_dialog(self):
        """Show dialog to select reference channels."""
        if self.data is not None:
            dialog = ChannelsDialog(self.data['ch_names'], self)
            result = dialog.exec_()

            if result == QDialog.Accepted:
                selected_channels = dialog.get_selected_channels()
                self.ref_chan_lineedit.setText(', '.join(selected_channels))
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)

    def show_eog_channels_dialog(self):
        """Show dialog to select EOG channels."""
        if self.data is not None:
            dialog = ChannelsDialog(self.data['ch_names'], self)
            result = dialog.exec_()

            if result == QDialog.Accepted:
                selected_channels = dialog.get_selected_channels()
                self.eog_chan_lineedit.setText(', '.join(selected_channels))
                self.ica_components_spinbox.setValue(
                    self.data['nchan'] - len(selected_channels)
                )
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)

    def run(self):
        """Execute EEG preprocessing pipeline using raw data."""
        if self.data or self.import_file_list:
            # Get parameters
            montage = self.montage_combox.currentText()
            if montage == 'none':
                montage = None
            bad_channels = self.bad_channels_lineedit.text().split(',')
            bad_channels = [chan.strip() for chan in bad_channels]
            fs = self.fs_spinbox.value()
            lowcut = self.lowcut_spinbox.value()
            highcut = self.highcut_spinbox.value()
            notch_freq = self.notch_spinbox.value()
            is_ica_enabled = self.is_ica_checkbox.isChecked()
            ica_method = self.ica_method_combobox.currentText() if is_ica_enabled else None
            ica_components = self.ica_components_spinbox.value()
            eog_regression = self.eog_checkbox.isChecked()
            eog_channels = self.eog_chan_lineedit.text().split(',')
            eog_channels = [chan.strip() for chan in eog_channels]
            rm_distortion = self.rm_distortion_checkbox.isChecked()
            ref_chan_list = self.ref_chan_lineedit.text().split(',')
            ref_chan_list = [chan.strip() for chan in ref_chan_list]
            ref_method = self.ref_method_combobox.currentText()
            output_file = self.prepare()
            file_type = self.file_type_combobox.currentText()

            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    data['montage'] = self.montage_combox.currentText()
                    eeg_preprocessing_by_dict(
                        data_dict=data,
                        bad_channels=bad_channels,
                        lowcut=lowcut,
                        highcut=highcut,
                        montage=montage,
                        rm_outlier=rm_distortion,
                        notch_f=notch_freq,
                        eog_regression=eog_regression,
                        eog_channels=eog_channels,
                        is_ICA=is_ica_enabled,
                        ICA_component=ica_components,
                        ICA_method=ica_method,
                        is_ref=True,
                        ref_chan=ref_chan_list,
                        is_save=True,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].rsplit('.', 1)[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Completed")
            else:
                self.data['montage'] = self.montage_combox.currentText()
                eeg_preprocessing_by_dict(
                    data_dict=self.data,
                    bad_channels=bad_channels,
                    lowcut=lowcut,
                    highcut=highcut,
                    montage=montage,
                    rm_outlier=rm_distortion,
                    notch_f=notch_freq,
                    eog_regression=eog_regression,
                    eog_channels=eog_channels,
                    is_ICA=is_ica_enabled,
                    ICA_component=ica_components,
                    ICA_method=ica_method,
                    is_ref=True,
                    ref_chan=ref_chan_list,
                    is_save=True,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.rsplit('.', 1)[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Completed")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)

    def choose_import_file(self):
        """Handle file import with parameter initialization."""
        self.data, self.filePath = read_file_by_qt(self)
        if self.filePath and self.data:
            if len(self.filePath) == 1:
                self.import_file_lineedit.setText(self.filePath[0])
            elif len(self.filePath) == 2:
                self.import_file_lineedit.setText(self.filePath[0] + " " + self.filePath[1])
            self.import_folder_lineedit.setText('')
            self.fs_spinbox.setValue(self.data['srate'])
            self.ica_components_spinbox.setValue(self.data['nchan'])
            self.ica_components_spinbox.setRange(0, self.data['nchan'])
            self.highcut_spinbox.setRange(0, self.data['srate'])
            self.lowcut_spinbox.setRange(0, self.data['srate'])
            self.notch_spinbox.setRange(0, self.data['srate'])
            if self.filePath[0]:
                base_name = os.path.basename(self.filePath[0])
                name, ext = os.path.splitext(base_name)
                self.savefileName = self.feature_name + f'_{name}.'
            self.is_folder = False


class fNIRSPreprocessingConfigWindow(FeatureDialog):
    """Configuration UI for fNIRS preprocessing operations."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize fNIRS preprocessing configuration window.

        :param feature_name: Preprocessing operation name
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.fs_spinbox = QDoubleSpinBox()
        self.fs_spinbox.setRange(1, 10000)
        self.fs_spinbox.setValue(10)
        self.left_layout.addRow('Sampling Rate (Hz):', self.fs_spinbox)

        # Add channel exclusion button
        btn_remove_bad_channels = BFPushButton('Exclude Channels')
        btn_remove_bad_channels.clicked.connect(self.show_remove_bad_channels_dialog)
        self.bad_channels_lineedit = QLineEdit(self)
        self.bad_channels_lineedit.setReadOnly(True)
        self.left_layout.addRow(btn_remove_bad_channels, self.bad_channels_lineedit)

        self.src_freqs_spinbox = QSpinBox()
        self.src_freqs_spinbox.setRange(1, 10000)
        self.src_freqs_spinbox.setValue(730)

        self.det_freqs_spinbox = QSpinBox()
        self.det_freqs_spinbox.setRange(1, 10000)
        self.det_freqs_spinbox.setValue(850)
        self.left_layout.addRow('Emitted Light Frequency (nm):', self.src_freqs_spinbox)
        self.left_layout.addRow('Received Light Frequency (nm):', self.det_freqs_spinbox)

        self.dwt_groupbox = BFGroupBox('Wavelet Filtering')
        dwt_layout = QFormLayout(self.dwt_groupbox)
        self.level_spinbox = QSpinBox()
        self.level_spinbox.setRange(1, 1000)
        self.level_spinbox.setValue(4)
        dwt_layout.addRow('Decomposition Level:', self.level_spinbox)
        self.basis_function_combox = QComboBox()
        self.basis_function_combox.addItems(['db4', 'db1', 'db2', 'db3'])
        dwt_layout.addRow('Wavelet Basis Function:', self.basis_function_combox)
        self.alpha_spinbox = QDoubleSpinBox()
        self.alpha_spinbox.setRange(0, 1000)
        self.alpha_spinbox.setValue(0.5)
        dwt_layout.addRow('Threshold Coefficient:', self.alpha_spinbox)

        self.left_layout.addRow(self.dwt_groupbox)

        # Band-pass filter settings
        filter_groupbox = BFGroupBox('Band-pass Filtering')
        filter_layout = QFormLayout(filter_groupbox)

        self.lowcut_spinbox = QDoubleSpinBox()
        self.lowcut_spinbox.setRange(0, 10000)
        self.lowcut_spinbox.setValue(0.01)
        filter_layout.addRow('Low Cutoff (Hz):', self.lowcut_spinbox)

        self.highcut_spinbox = QDoubleSpinBox()
        self.highcut_spinbox.setRange(0, 10000)
        self.highcut_spinbox.setValue(0.7)
        filter_layout.addRow('High Cutoff (Hz):', self.highcut_spinbox)

        self.left_layout.addRow(filter_groupbox)

        # Artifact detection settings
        denoise_groupbox = BFGroupBox('Artifact Detection')
        denoise_layout = QFormLayout(denoise_groupbox)

        self.rm_distortion_checkbox = QCheckBox('Enable')
        self.rm_distortion_checkbox.setChecked(True)
        denoise_layout.addRow(self.rm_distortion_checkbox)

        self.left_layout.addRow(denoise_groupbox)
        self.left_layout.addRow(self.save_data_groupbox)

        self.rm_distortion_checkbox.stateChanged.connect(self.update_rm_distortion_controls_visibility)

        self.setGeometry(300, 300, 600, 800)
        self.setWindowTitle('fNIRS Preprocessing Configuration')

        self.file_type_combobox.clear()
        self.file_type_combobox.addItems(['mat'])

    def update_rm_distortion_controls_visibility(self):
        """Toggle artifact detection parameter controls visibility."""
        is_rm_distortion_enabled = self.rm_distortion_checkbox.isChecked()
        self.rm_std_spinbox.setVisible(is_rm_distortion_enabled)
        self.rm_std_label.setVisible(is_rm_distortion_enabled)
        self.rm_window_spinbox.setVisible(is_rm_distortion_enabled)
        self.rm_window_label.setVisible(is_rm_distortion_enabled)

    def show_remove_bad_channels_dialog(self):
        """Show dialog for selecting channels to exclude."""
        if self.data is not None:
            dialog = ChannelsDialog(self.data['ch_names'], self)
            result = dialog.exec_()

            if result == QDialog.Accepted:
                selected_channels = dialog.get_selected_channels()
                self.bad_channels_lineedit.setText(', '.join(selected_channels))
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)

    def run(self):
        """Execute fNIRS preprocessing pipeline."""
        if self.data or self.import_file_list:
            # Get channel exclusion list
            bad_channels = [chan.strip() for chan in self.bad_channels_lineedit.text().split(',')]
            # Band-pass filter parameters
            lowcut = self.lowcut_spinbox.value()
            highcut = self.highcut_spinbox.value()
            wavelet = self.basis_function_combox.currentText()
            level = self.level_spinbox.value()
            alpha = self.alpha_spinbox.value()
            # Artifact detection parameter
            rm_distortion = self.rm_distortion_checkbox.isChecked()

            output_file = self.prepare()
            file_type = self.file_type_combobox.currentText()
            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    raw = mne.io.read_raw_snirf(filePath)
                    raw_preprocess = fnirs_preprocessing_by_raw(
                        raw=raw,
                        bad_channels=bad_channels,
                        lowcut=lowcut,
                        highcut=highcut,
                        wavelet=wavelet,
                        level=level,
                        alpha=alpha,
                        enable_interpolate=rm_distortion
                    )
                    raw_dict = raw_to_dict(raw_preprocess)
                    raw_dict['type'] = 'fnirs_preprocess'
                    save_file(data=raw_dict, save_path=self.save_file_list[i], save_filestyle=file_type)
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                        dict_to_info(raw_dict, info_file_name)
                    self.process_dialog.update_status(i, "Yes")
            else:
                raw_preprocess = fnirs_preprocessing_by_raw(
                    raw=self.data,
                    bad_channels=bad_channels,
                    lowcut=lowcut,
                    highcut=highcut,
                    wavelet=wavelet,
                    level=level,
                    alpha=alpha,
                    enable_interpolate=rm_distortion
                )
                raw_dict = raw_to_dict(raw_preprocess)
                raw_dict['type'] = 'fnirs_preprocess'
                save_file(data=raw_dict, save_path=output_file, save_filestyle=file_type)
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.split('.')[0] + '_info.json'
                    dict_to_info(raw_dict, info_file_name)
                self.process_dialog.update_status(0, "Yes")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)

    def choose_import_file(self):
        """Select and import fNIRS data files."""
        self.data, self.filePath = read_file_by_mne(self)
        if self.filePath and self.data:
            if len(self.filePath) == 1:
                self.import_file_lineedit.setText(self.filePath[0])
            elif len(self.filePath) == 2:
                self.import_file_lineedit.setText(self.filePath[0] + " " + self.filePath[1])
            self.import_folder_lineedit.setText('')
            self.fs_spinbox.setValue(self.data.info['sfreq'])
            self.highcut_spinbox.setRange(0, self.data.info['sfreq'])
            self.lowcut_spinbox.setRange(0, self.data.info['sfreq'])
            data_dict = raw_to_dict(self.data)
            if data_dict.get('wavelengths', None) and len(data_dict['wavelengths']) >= 2:
                self.src_freqs_spinbox.setValue(data_dict['wavelengths'][0])
                self.det_freqs_spinbox.setValue(data_dict['wavelengths'][1])
            if self.filePath[0]:
                base_name = os.path.basename(self.filePath[0])
                name, _ = os.path.splitext(base_name)
                self.savefileName = f'{self.feature_name}_{name}.'
                self.is_folder = False


class EMGPreprocessingConfigWindow(FeatureDialog):
    """Configuration UI for EMG preprocessing operations."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize EMG preprocessing configuration window.

        :param feature_name: Preprocessing operation name
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.fs_spinbox = QDoubleSpinBox()
        self.fs_spinbox.setRange(1, 10000)
        self.fs_spinbox.setValue(1000)
        self.left_layout.addRow('Sampling Rate (Hz):', self.fs_spinbox)

        # Band-pass filter settings
        bandpass_groupbox = BFGroupBox('Band-pass Filter')
        bandpass_layout = QFormLayout(bandpass_groupbox)

        self.bf_lowcut_spinbox = QDoubleSpinBox()
        self.bf_lowcut_spinbox.setRange(0, 10000)
        self.bf_lowcut_spinbox.setValue(20)
        bandpass_layout.addRow('Low Cutoff (Hz):', self.bf_lowcut_spinbox)

        self.bf_highcut_spinbox = QDoubleSpinBox()
        self.bf_highcut_spinbox.setRange(0, 10000)
        self.bf_highcut_spinbox.setValue(450)
        bandpass_layout.addRow('High Cutoff (Hz):', self.bf_highcut_spinbox)

        self.bf_order_spinbox = QSpinBox()
        self.bf_order_spinbox.setRange(1, 20)
        self.bf_order_spinbox.setValue(6)
        bandpass_layout.addRow('Filter Order:', self.bf_order_spinbox)

        self.filter_combobox = QComboBox()
        self.filter_combobox.addItems(['Butterworth', 'Bessel', 'Chebyshev'])
        bandpass_layout.addRow('Filter Type:', self.filter_combobox)

        self.left_layout.addRow(bandpass_groupbox)

        # Notch filter settings
        notch_groupbox = BFGroupBox('Notch Filter')
        notch_layout = QFormLayout(notch_groupbox)

        self.north_f_spinbox = QDoubleSpinBox()
        self.north_f_spinbox.setRange(0, 10000)
        self.north_f_spinbox.setValue(50.0)
        notch_layout.addRow('Notch Frequency (Hz):', self.north_f_spinbox)

        self.Q_spinbox = QSpinBox()
        self.Q_spinbox.setRange(0, 50)
        self.Q_spinbox.setValue(30)
        notch_layout.addRow('Quality Factor (Q):', self.Q_spinbox)

        self.left_layout.addRow(notch_groupbox)

        # Low-pass filter settings
        lowpass_groupbox = BFGroupBox('Low-pass Filter')
        lowpass_layout = QFormLayout(lowpass_groupbox)

        self.lf_cutoff_spinbox = QDoubleSpinBox()
        self.lf_cutoff_spinbox.setRange(0, 10000)
        self.lf_cutoff_spinbox.setValue(20)
        lowpass_layout.addRow('Cutoff Frequency (Hz):', self.lf_cutoff_spinbox)

        self.lf_order_spinbox = QSpinBox()
        self.lf_order_spinbox.setRange(1, 20)
        self.lf_order_spinbox.setValue(4)
        lowpass_layout.addRow('Filter Order:', self.lf_order_spinbox)

        self.left_layout.addRow(lowpass_groupbox)

        # Export settings
        self.left_layout.addRow(self.save_data_groupbox)

        self.setGeometry(300, 300, 600, 800)
        self.setWindowTitle('EMG Preprocessing Configuration')

        self.file_type_combobox.clear()
        self.file_type_combobox.addItems(['mat'])

    def run(self):
        """Execute EMG preprocessing pipeline."""
        if self.data or self.import_file_list:
            # Band-pass filter parameters
            fs = self.fs_spinbox.value()
            bf_lowcut = self.bf_lowcut_spinbox.value()
            bf_highcut = self.bf_highcut_spinbox.value()
            bf_order = self.bf_order_spinbox.value()
            filter_type = self.filter_combobox.currentText()
            # Notch filter parameters
            north_f = self.north_f_spinbox.value()
            Q = self.Q_spinbox.value()
            # Low-pass filter parameters
            lf_cutoff = self.lf_cutoff_spinbox.value()
            lf_order = self.lf_order_spinbox.value()
            output_file = self.prepare()
            file_type = self.file_type_combobox.currentText()
            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    emg_preprocessing(
                        emg_signal=data['data'],
                        fs=fs,
                        bf_lowcut=bf_lowcut,
                        bf_highcut=bf_highcut,
                        lf_cutoff=lf_cutoff,
                        events=data['events'],
                        chan_list=data['ch_names'],
                        bf_order=bf_order,
                        filter=filter_type,
                        lf_order=lf_order,
                        north_f=north_f,
                        Q=Q,
                        is_save=True,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Yes")
            else:
                emg_preprocessing(
                    emg_signal=self.data['data'],
                    fs=fs,
                    bf_lowcut=bf_lowcut,
                    bf_highcut=bf_highcut,
                    lf_cutoff=lf_cutoff,
                    events=self.data['events'],
                    chan_list=self.data['ch_names'],
                    bf_order=bf_order,
                    filter=filter_type,
                    lf_order=lf_order,
                    north_f=north_f,
                    Q=Q,
                    is_save=True,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.split('.')[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Yes")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)

    def choose_import_file(self):
        """Select and import EMG data files."""
        self.data, self.filePath = read_file_by_qt(self)
        if self.filePath and self.data:
            if len(self.filePath) == 1:
                self.import_file_lineedit.setText(self.filePath[0])
            elif len(self.filePath) == 2:
                self.import_file_lineedit.setText(self.filePath[0] + " " + self.filePath[1])
            self.import_folder_lineedit.setText('')
            self.fs_spinbox.setValue(self.data['srate'])
            self.bf_highcut_spinbox.setRange(0, self.data['srate'])
            self.bf_lowcut_spinbox.setRange(0, self.data['srate'])
            self.north_f_spinbox.setRange(0, self.data['srate'])
            self.lf_cutoff_spinbox.setRange(0, self.data['srate'])
            if self.filePath[0]:
                base_name = os.path.basename(self.filePath[0])
                name, _ = os.path.splitext(base_name)
                self.savefileName = f'{self.feature_name}_{name}.'
                self.is_folder = False


class ECGPreprocessingConfigWindow(FeatureDialog):
    """Configuration UI for ECG preprocessing operations."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize ECG preprocessing configuration window.

        :param feature_name: Preprocessing operation name
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.fs_spinbox = QDoubleSpinBox()
        self.fs_spinbox.setRange(1, 10000)
        self.fs_spinbox.setValue(1000)
        self.left_layout.addRow('Sampling Rate (Hz):', self.fs_spinbox)

        # Band-pass filter settings
        bandpass_groupbox = BFGroupBox('Band-pass Filter')
        bandpass_layout = QFormLayout(bandpass_groupbox)

        self.lowcut_spinbox = QDoubleSpinBox()
        self.lowcut_spinbox.setRange(0, 10000)
        self.lowcut_spinbox.setValue(0.5)
        bandpass_layout.addRow('Low Cutoff (Hz):', self.lowcut_spinbox)

        self.highcut_spinbox = QDoubleSpinBox()
        self.highcut_spinbox.setRange(0, 10000)
        self.highcut_spinbox.setValue(45)
        bandpass_layout.addRow('High Cutoff (Hz):', self.highcut_spinbox)

        self.filter_order_spinbox = QSpinBox()
        self.filter_order_spinbox.setRange(1, 20)
        self.filter_order_spinbox.setValue(4)
        bandpass_layout.addRow('Filter Order:', self.filter_order_spinbox)

        self.filter_combobox = QComboBox()
        self.filter_combobox.addItems(['Butterworth', 'Bessel', 'Chebyshev'])
        bandpass_layout.addRow('Filter Type:', self.filter_combobox)

        self.left_layout.addRow(bandpass_groupbox)

        # Notch filter settings
        notch_groupbox = BFGroupBox('Notch Filter')
        notch_layout = QFormLayout(notch_groupbox)

        self.north_f_spinbox = QDoubleSpinBox()
        self.north_f_spinbox.setRange(0, 10000)
        self.north_f_spinbox.setValue(50.0)
        notch_layout.addRow('Notch Frequency (Hz):', self.north_f_spinbox)

        self.Q_spinbox = QSpinBox()
        self.Q_spinbox.setRange(0, 50)
        self.Q_spinbox.setValue(30)
        notch_layout.addRow('Quality Factor (Q):', self.Q_spinbox)

        self.left_layout.addRow(notch_groupbox)

        # Signal conditioning settings
        lowpass_groupbox = BFGroupBox('Signal Conditioning')
        lowpass_layout = QFormLayout(lowpass_groupbox)

        self.downsample_fs_spinbox = QDoubleSpinBox()
        self.downsample_fs_spinbox.setRange(0, 10000)
        self.downsample_fs_spinbox.setValue(250)
        lowpass_layout.addRow('Downsample Frequency (Hz):', self.downsample_fs_spinbox)

        self.median_filter_window_spinbox = QDoubleSpinBox()
        self.median_filter_window_spinbox.setRange(0, 1)
        self.median_filter_window_spinbox.setValue(0.2)
        lowpass_layout.addRow('Median Filter Window (s):', self.median_filter_window_spinbox)

        self.wavelet_checkbox = QCheckBox('Wavelet Denoising')
        self.wavelet_combobox = QComboBox()
        self.wavelet_combobox.addItems(['db4', 'sym5', 'coif5'])
        self.wavelet_level_spinbox = QSpinBox()
        self.wavelet_level_spinbox.setRange(1, 10)
        self.wavelet_level_spinbox.setValue(1)

        lowpass_layout.addRow(self.wavelet_checkbox)
        lowpass_layout.addRow('Wavelet Type:', self.wavelet_combobox)
        lowpass_layout.addRow('Decomposition Level:', self.wavelet_level_spinbox)

        self.left_layout.addRow(lowpass_groupbox)

        # Export settings
        self.left_layout.addRow(self.save_data_groupbox)

        self.setGeometry(300, 300, 600, 800)
        self.setWindowTitle('ECG Preprocessing Configuration')

        self.file_type_combobox.clear()
        self.file_type_combobox.addItems(['mat'])

    def run(self):
        """Execute ECG preprocessing pipeline."""
        if self.data or self.import_file_list:
            # Band-pass filter parameters
            fs = self.fs_spinbox.value()
            lowcut = self.lowcut_spinbox.value()
            highcut = self.highcut_spinbox.value()
            filter_order = self.filter_order_spinbox.value()
            filter_type = self.filter_combobox.currentText()
            # Notch filter parameters
            north_f = self.north_f_spinbox.value()
            Q = self.Q_spinbox.value()
            # Signal conditioning parameters
            downsample_fs = self.downsample_fs_spinbox.value()
            median_filter_window = self.median_filter_window_spinbox.value()
            is_wavelet_denoise = self.wavelet_checkbox.isChecked()
            wavelet = self.wavelet_combobox.currentText()
            level = self.wavelet_level_spinbox.value()

            output_file = self.prepare()
            file_type = self.file_type_combobox.currentText()
            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    ecg_preprocessing(
                        ecg_signal=data['data'],
                        fs=fs,
                        lowcut=lowcut,
                        highcut=highcut,
                        events=data['events'],
                        chan_list=data['ch_names'],
                        downsample_fs=downsample_fs,
                        filter_order=filter_order,
                        filter=filter_type,
                        north_f=north_f,
                        Q=Q,
                        median_filter_window=median_filter_window,
                        is_wavelet_denoise=is_wavelet_denoise,
                        wavelet=wavelet,
                        level=level,
                        is_save=True,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Yes")
            else:
                ecg_preprocessing(
                    ecg_signal=self.data['data'],
                    fs=fs,
                    lowcut=lowcut,
                    highcut=highcut,
                    events=self.data['events'],
                    chan_list=self.data['ch_names'],
                    downsample_fs=downsample_fs,
                    filter_order=filter_order,
                    filter=filter_type,
                    north_f=north_f,
                    Q=Q,
                    median_filter_window=median_filter_window,
                    is_wavelet_denoise=is_wavelet_denoise,
                    wavelet=wavelet,
                    level=level,
                    is_save=True,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.split('.')[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Yes")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)

    def choose_import_file(self):
        """Select and import ECG data files."""
        self.data, self.filePath = read_file_by_qt(self)
        if self.filePath and self.data:
            if len(self.filePath) == 1:
                self.import_file_lineedit.setText(self.filePath[0])
            elif len(self.filePath) == 2:
                self.import_file_lineedit.setText(self.filePath[0] + " " + self.filePath[1])
            self.import_folder_lineedit.setText('')
            self.fs_spinbox.setValue(self.data['srate'])
            self.lowcut_spinbox.setRange(0, self.data['srate'])
            self.highcut_spinbox.setRange(0, self.data['srate'])
            self.north_f_spinbox.setRange(0, self.data['srate'])
            self.downsample_fs_spinbox.setRange(0, self.data['srate'])
            self.median_filter_window_spinbox.setRange(0, self.data['srate'])
            if self.filePath[0]:
                base_name = os.path.basename(self.filePath[0])
                name, _ = os.path.splitext(base_name)
                self.savefileName = f'{self.feature_name}_{name}.'
                self.is_folder = False


class CreateFilesDialog(FeatureDialog):
    """Configuration UI for creating standardized neurophysiology data files."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize file creation configuration dialog.

        :param feature_name: Name of the file creation operation
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.setWindowTitle('Create Files Configuration')

        # File information section
        create_file_groupbox = QGroupBox('File Information', self)
        create_file_layout = QFormLayout(create_file_groupbox)

        # Sampling rate input
        self.srate_lineedit = QLineEdit()
        create_file_layout.addRow('Sampling Rate (Hz)', self.srate_lineedit)

        # Channels count input
        self.nchan_lineedit = QLineEdit()
        create_file_layout.addRow('Number of Channels', self.nchan_lineedit)

        # Channel names input
        self.ch_names_lineedit = QLineEdit()
        create_file_layout.addRow('Channel Names', self.ch_names_lineedit)

        # Data type selection
        self.type_combox = QComboBox()
        self.type_combox.addItems(['eeg', 'emg', 'fnirs', 'ecg', 'other'])
        create_file_layout.addRow('Data Type', self.type_combox)

        # Montage selection
        self.montage_combox = QComboBox()
        self.montage_combox.addItems([
            'standard_1020',
            'standard_1005',
            'standard_alphabetic',
            'standard_postfixed',
            'standard_prefixed',
            'standard_primed',
            'biosemi16',
            'biosemi32',
            'biosemi64',
            'biosemi128',
            'biosemi160',
            'biosemi256',
            'easycap-M1',
            'easycap-M10',
            'EGI_256',
            'GSN-HydroCel-32',
            'GSN-HydroCel-64_1.0',
            'GSN-HydroCel-65_1.0',
            'GSN-HydroCel-128',
            'GSN-HydroCel-129',
            'GSN-HydroCel-256',
            'GSN-HydroCel-257',
            'mgh60',
            'mgh70',
            'artinis-octamon',
            'artinis-brite23',
            'brainproducts-RNP-BA-128'
        ])
        create_file_layout.addRow('Montage Protocol', self.montage_combox)

        # Connect type change to montage update
        self.type_combox.currentIndexChanged.connect(self.updateMontage)

        self.left_layout.addRow(create_file_groupbox)
        self.left_layout.addRow(self.save_data_groupbox)

    def updateMontage(self, index):
        """
        Update montage options based on selected data type.

        :param index: Index of selected data type
        :type index: int
        """
        self.montage_combox.clear()
        if index == 0:  # EEG selected
            self.montage_combox.addItems(['standard_1020', 'standard_1010', 'standard_1005'])

    def run(self):
        """Execute file creation and data processing."""
        if self.data or self.import_file_list:
            output_file = self.prepare()
            file_type = self.file_type_combobox.currentText()
            srate = float(self.srate_lineedit.text())
            nchan = int(self.nchan_lineedit.text())
            ch_names = ast.literal_eval(self.ch_names_lineedit.text())
            data_type = self.type_combox.currentText()
            montage = self.montage_combox.currentText()

            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    create_data_dict(
                        data['data'], srate, nchan=nchan,
                        ch_names=ch_names, events=data['events'],
                        type=data_type, montage=montage,
                        is_save=True, save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                QMessageBox.information(None, 'Successful', 'Data processing completed!', QMessageBox.Ok)
            else:
                create_data_dict(
                    self.data['data'], srate, nchan=nchan,
                    ch_names=ch_names, events=self.data['events'],
                    type=data_type, montage=montage,
                    is_save=True, save_path=output_file,
                    save_filestyle=file_type
                )
                QMessageBox.information(None, 'Successful', 'Data processing completed!', QMessageBox.Ok)
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)

    def choose_import_file(self):
        """Select and import data files for processing."""
        self.data, self.filePath = read_file_by_qt(self)
        if self.filePath and self.data:
            if len(self.filePath) == 1:
                self.import_file_lineedit.setText(self.filePath[0])
            elif len(self.filePath) == 2:
                self.import_file_lineedit.setText(self.filePath[0] + " " + self.filePath[1])
            self.import_folder_lineedit.setText('')

            # Extract filename components
            if self.filePath[0]:
                base_name = os.path.basename(self.filePath[0])
                name, _ = os.path.splitext(base_name)
                self.savefileName = f'{name}_{self.feature_name}.'
            self.is_folder = False
            self.srate_lineedit.setText(str(self.data['srate']))
            self.nchan_lineedit.setText(str(self.data['nchan']))
            self.ch_names_lineedit.setText(str(self.data['ch_names']))

    def choose_import_folder(self):
        """Select and import folder contents for batch processing."""

        def get_folder_contents(folder_path):
            file_list = []
            for root, _, files in os.walk(folder_path):
                if 'data.bdf' in files and 'evt.bdf' in files:
                    data_path = os.path.join(root, 'data.bdf')
                    evt_path = os.path.join(root, 'evt.bdf')
                    file_list.append((data_path, evt_path))
                else:
                    for file in files:
                        file_list.append(os.path.join(root, file))
            return file_list

        folder_dialog = QFileDialog()
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        if folder_dialog.exec_():
            self.folderPath = folder_dialog.selectedFiles()[0]
            self.import_folder_lineedit.setText(self.folderPath)
            self.import_file_lineedit.setText('')
            base_name = os.path.basename(self.folderPath)
            self.savefolderName = f'{base_name}_{self.feature_name}'
            self.import_file_list = get_folder_contents(self.folderPath)
            if self.import_file_list:
                self.data = read_file_by_path(self.import_file_list[0])
            else:
                QMessageBox.warning(None, 'Warning', 'No files found in folder', QMessageBox.Ok)
            self.is_folder = True
            self.srate_lineedit.setText(str(self.data['srate']))
            self.nchan_lineedit.setText(str(self.data['nchan']))
            self.ch_names_lineedit.setText(str(self.data['ch_names']))


class CreateSegments(FeatureDialog):
    """UI for creating neurophysiological data segments/epochs."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize segmentation configuration dialog.

        :param feature_name: Name of segmentation operation
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.create_method = None
        self.setWindowTitle('Create Epochs Configuration')

        self.file_type_combobox.clear()
        self.file_type_combobox.addItems(['mat'])

        # Segmentation options section
        vbox = QVBoxLayout()
        segments_group_box = QGroupBox('Segmentation Methods', self)

        # Segmentation method options
        self.radio_btn1 = QRadioButton('Segment by Events')
        self.radio_btn2 = QRadioButton('Window Around Events')
        self.radio_btn3 = QRadioButton('Fixed Duration Segmentation')
        self.radio_btn4 = QRadioButton('Custom Segmentation')

        # Option-specific frames
        self.frame1 = QFrame(self)
        self.frame2 = QFrame(self)
        self.frame3 = QFrame(self)
        self.frame4 = QFrame(self)

        # Hide all frames initially
        self.frame1.hide()
        self.frame2.hide()
        self.frame3.hide()
        self.frame4.hide()

        # Event-based segmentation frame
        self.choose_events_btn1 = QPushButton('Select Events')
        self.select_mark_list1 = QLineEdit(self)
        form_layout1 = QFormLayout(self.frame1)
        form_layout1.addRow(self.choose_events_btn1, self.select_mark_list1)

        # Windowed event segmentation frame
        self.choose_events_btn2 = QPushButton('Select Events')
        self.select_mark_list2 = QLineEdit(self)
        form_layout2 = QFormLayout(self.frame2)
        form_layout2.addRow(self.choose_events_btn2, self.select_mark_list2)
        self.before_spinbox = QSpinBox(self)
        self.before_spinbox.setRange(0, 2147483647)
        self.after_spinbox = QSpinBox(self)
        self.after_spinbox.setRange(0, 2147483647)
        form_layout2.addRow('Pre-event Duration (samples)', self.before_spinbox)
        form_layout2.addRow('Post-event Duration (samples)', self.after_spinbox)

        # Fixed duration segmentation frame
        form_layout3 = QFormLayout(self.frame3)
        self.duration_spinbox = QSpinBox(self)
        self.duration_spinbox.setRange(0, 2147483647)
        form_layout3.addRow('Segment Duration (samples)', self.duration_spinbox)

        # Custom segmentation frame
        self.list_widget = QListWidget(self.frame4)
        self.add_row_button = QPushButton('Add Row', self.frame4)
        self.add_row_button.clicked.connect(self.add_row_to_list)
        grid_layout = QGridLayout(self.frame4)
        grid_layout.addWidget(self.list_widget, 0, 0)
        grid_layout.addWidget(self.add_row_button, 1, 0)

        # Connect radio buttons to toggle handler
        self.radio_btn1.toggled.connect(self.on_radio_toggled)
        self.radio_btn2.toggled.connect(self.on_radio_toggled)
        self.radio_btn3.toggled.connect(self.on_radio_toggled)
        self.radio_btn4.toggled.connect(self.on_radio_toggled)

        # Add widgets to layout
        self.point_checkbox = QCheckBox('Split by Data Points (not time)')
        self.point_checkbox.setStyleSheet("QCheckBox { color: red; }")
        vbox.addWidget(self.point_checkbox)
        vbox.addWidget(self.radio_btn1)
        vbox.addWidget(self.frame1)
        vbox.addWidget(self.radio_btn2)
        vbox.addWidget(self.frame2)
        vbox.addWidget(self.radio_btn3)
        vbox.addWidget(self.frame3)
        vbox.addWidget(self.radio_btn4)
        vbox.addWidget(self.frame4)

        segments_group_box.setLayout(vbox)
        self.left_layout.addRow(segments_group_box)
        self.left_layout.addRow(self.save_data_groupbox)

        # Connect event buttons to dialog
        self.choose_events_btn1.clicked.connect(self.show_events_dialog)
        self.choose_events_btn2.clicked.connect(self.show_events_dialog)

    def prepare(self):
        """
        Prepare output directory for saving processed data.

        :return: Output path for saved files
        :rtype: str
        """
        if self.is_folder:
            output_path = os.path.join(
                self.output_folder_lineedit.text() or os.path.dirname(self.folderPath),
                self.savefolderName
            )
            os.makedirs(output_path, exist_ok=True)
            for root, _, _ in os.walk(self.folderPath):
                relative_path = os.path.relpath(root, self.folderPath)
                os.makedirs(os.path.join(output_path, relative_path), exist_ok=True)
            self.save_file_list = [
                os.path.join(output_path, os.path.relpath(file, self.folderPath)).rsplit('.', 1)[0]
                for file in self.import_file_list
            ]
            return output_path
        else:
            return os.path.join(
                self.output_folder_lineedit.text() or os.path.dirname(self.filePath[0]),
                self.savefileName
            )

    def run(self):
        """Execute data segmentation and saving operations."""
        if self.data or self.import_file_list:
            # Initialize segmentation parameters
            events_list = None
            events_range = None
            fixed_duration = None

            # Set parameters based on selected method
            if self.create_method == 'split_by_events':
                events_list = [sublist[0] for sublist in ast.literal_eval(self.select_mark_list1.text())]
            elif self.create_method == 'split_by_front_and_back_of_events':
                events_list = [sublist[0] for sublist in ast.literal_eval(self.select_mark_list2.text())]
                events_range = (self.before_spinbox.value(), self.after_spinbox.value())
            elif self.create_method == 'split_by_fixed_length':
                fixed_duration = self.duration_spinbox.value()

            # Prepare output and process data
            output_dir = self.prepare()
            file_type = self.file_type_combobox.currentText()
            use_points = self.point_checkbox.isChecked()

            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    create_epoch(
                        data=data, method=self.create_method,
                        is_save=True, save_path=self.save_file_list[i],
                        save_filestyle=file_type, events_list=events_list,
                        events_range=events_range, fix_length=fixed_duration,
                        custom_events=None, is_point=use_points
                    )
                    self.process_dialog.update_status(i, "Yes")
            else:
                create_epoch(
                    data=self.data, method=self.create_method,
                    is_save=True, save_path=output_dir,
                    save_filestyle=file_type, events_list=events_list,
                    events_range=events_range, fix_length=fixed_duration,
                    custom_events=None, is_point=use_points
                )
                self.process_dialog.update_status(0, "Yes")

    def on_radio_toggled(self):
        """Toggle visibility of segmentation option frames."""
        self.frame1.setVisible(self.radio_btn1.isChecked())
        self.frame2.setVisible(self.radio_btn2.isChecked())
        self.frame3.setVisible(self.radio_btn3.isChecked())
        self.frame4.setVisible(self.radio_btn4.isChecked())

        # Set creation method based on selection
        if self.radio_btn1.isChecked():
            self.create_method = 'split_by_events'
        elif self.radio_btn2.isChecked():
            self.create_method = 'split_by_front_and_back_of_events'
        elif self.radio_btn3.isChecked():
            self.create_method = 'split_by_fixed_length'
        elif self.radio_btn4.isChecked():
            self.create_method = 'custom_split'

    def show_events_dialog(self):
        """Show dialog for event selection."""
        if self.data:
            dialog = EventsSelectionDialog(self.data['events'], self)
            result = dialog.exec_()
            if result == QDialog.Accepted:
                result_text = ', '.join(dialog.selected_items)
                # Determine which button triggered the signal
                sender_button = self.sender()
                if sender_button == self.choose_events_btn1:
                    self.select_mark_list1.setText(result_text)
                elif sender_button == self.choose_events_btn2:
                    self.select_mark_list2.setText(result_text)
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)

    def add_row_to_list(self):
        """Add row to custom segmentation list."""
        item = QListWidgetItem()
        widget = QWidget()
        left_edit = QLineEdit()
        right_edit = QLineEdit()
        layout = QHBoxLayout(widget)
        layout.addWidget(left_edit)
        layout.addWidget(right_edit)
        layout.setContentsMargins(10, 10, 10, 0)
        item.setSizeHint(widget.sizeHint())
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, widget)


class EventsSelectionDialog(QDialog):
    """Dialog for selecting events from a list."""

    def __init__(self, items, parent=None):
        """
        Initialize events selection dialog.

        :param items: List of events to display
        :type items: list
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(parent)
        self.selected_items = []
        self.initUI(items)

    def initUI(self, items):
        """Setup dialog UI with event list and controls."""
        self.setWindowTitle('Select Events')
        self.setGeometry(100, 100, 300, 500)
        vbox = QVBoxLayout()

        self.list_widget = QListWidget(self)
        for item in items:
            list_item = QListWidgetItem(str(item))
            list_item.setCheckState(2)  # Checked by default
            self.list_widget.addItem(list_item)

        vbox.addWidget(self.list_widget)
        self.setLayout(vbox)

        button_box = QHBoxLayout()
        self.select_all_btn = QCheckBox('Select All', self)
        self.select_all_btn.stateChanged.connect(self.toggle_select_all)
        self.select_all_btn.setChecked(True)
        ok_btn = QPushButton('OK', self)
        ok_btn.clicked.connect(self.accept)
        button_box.addWidget(self.select_all_btn)
        button_box.addStretch(1)
        button_box.addWidget(ok_btn)
        vbox.addLayout(button_box)

    def toggle_select_all(self):
        """Toggle selection state of all items."""
        if self.select_all_btn.isChecked():
            for i in range(self.list_widget.count()):
                self.list_widget.item(i).setCheckState(2)
        else:
            for i in range(self.list_widget.count()):
                self.list_widget.item(i).setCheckState(0)

    def exec_(self):
        """Execute dialog and collect selected items."""
        result = super().exec_()
        if result == QDialog.Accepted:
            self.selected_items = [
                self.list_widget.item(i).text()
                for i in range(self.list_widget.count())
                if self.list_widget.item(i).checkState() == 2
            ]
        return result


class EEGPowerSpectralDensityDialog(FeatureDialog):
    """UI for EEG power spectral density analysis."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize PSD analysis configuration.

        :param feature_name: Analysis operation name
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.setWindowTitle('EEG Power Spectral Density Analysis')
        self.left_layout.addRow(self.save_data_groupbox)

    def run(self):
        """Execute PSD analysis and save results."""
        if self.data or self.import_file_list:
            output_file = self.prepare()
            file_type = self.file_type_combobox.currentText()
            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    eeg_power_spectral_density(
                        data=data,
                        is_save=True,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Yes")
            else:
                eeg_power_spectral_density(
                    data=self.data,
                    is_save=True,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.split('.')[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Yes")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)


class MultiscaleEntropyDialog(FeatureDialog):
    """UI for multiscale entropy calculation."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize multiscale entropy configuration.

        :param feature_name: Analysis operation name
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.setWindowTitle('Multiscale Entropy Analysis')
        # MSE parameters section
        mse_groupbox = BFGroupBox('MSE Parameters')
        mse_layout = QFormLayout(mse_groupbox)
        self.scale_factor_spinbox = QSpinBox()
        self.scale_factor_spinbox.setRange(1, 100)
        self.scale_factor_spinbox.setValue(5)
        mse_layout.addRow('Scale Factor:', self.scale_factor_spinbox)
        self.left_layout.addRow(mse_groupbox)
        self.left_layout.addRow(self.save_data_groupbox)

    def run(self):
        """Execute MSE calculation and save results."""
        if self.data or self.import_file_list:
            output_file = self.prepare()
            file_type = self.file_type_combobox.currentText()
            scale_factor = self.scale_factor_spinbox.value()
            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    eeg_multiscale_entropy(
                        data=data,
                        scale_factor=scale_factor,
                        is_save=True,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Yes")
            else:
                eeg_multiscale_entropy(
                    data=self.data,
                    scale_factor=scale_factor,
                    is_save=True,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.split('.')[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Yes")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)


class SampleEntropyDialog(FeatureDialog):
    """UI for sample entropy calculation."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize sample entropy configuration.

        :param feature_name: Analysis operation name
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.setWindowTitle('Sample Entropy Analysis')
        self.left_layout.addRow(self.save_data_groupbox)

    def run(self):
        """Execute sample entropy calculation and save results."""
        if self.data or self.import_file_list:
            output_file = self.prepare()
            file_type = self.file_type_combobox.currentText()
            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    try:
                        sample_entropy(
                            data=data,
                            is_save=True,
                            save_path=self.save_file_list[i],
                            save_filestyle=file_type
                        )
                        if self.export_info_checkbox.isChecked():
                            info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                            dict_to_info(data, info_file_name)
                        self.process_dialog.update_status(i, "Yes")
                    except Exception as e:
                        QMessageBox.critical(
                            self, "Error",
                            f"Data length exceeds limit (32768): {str(e)}"
                        )
            else:
                try:
                    sample_entropy(
                        data=self.data,
                        is_save=True,
                        save_path=output_file,
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = output_file.split('.')[0] + '_info.json'
                        dict_to_info(self.data, info_file_name)
                    self.process_dialog.update_status(0, "Yes")
                except Exception as e:
                    QMessageBox.critical(
                        self, "Error",
                        f"Data length exceeds limit (32768): {str(e)}"
                    )
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)


class EEGMicrostateDialog(FeatureDialog):
    """UI for EEG microstate analysis."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize microstate analysis configuration.

        :param feature_name: Analysis operation name
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.setWindowTitle('EEG Microstate Analysis')
        # Microstate parameters section
        microstate_groupbox = BFGroupBox('Microstate Parameters')
        microstate_layout = QFormLayout(microstate_groupbox)
        self.n_clusters_spinbox = QSpinBox()
        self.n_clusters_spinbox.setRange(1, 10)
        self.n_clusters_spinbox.setValue(4)
        microstate_layout.addRow('Number of States:', self.n_clusters_spinbox)
        self.peak_threshold_spinbox = QDoubleSpinBox()
        self.peak_threshold_spinbox.setRange(0, 1)
        self.peak_threshold_spinbox.setValue(0)
        microstate_layout.addRow('GFP Peak Threshold:', self.peak_threshold_spinbox)
        self.left_layout.addRow(microstate_groupbox)
        self.left_layout.addRow(self.save_data_groupbox)

    def run(self):
        """Execute microstate analysis and save results."""
        if self.data or self.import_file_list:
            output_file = self.prepare()
            file_type = self.file_type_combobox.currentText()
            n_clusters = self.n_clusters_spinbox.value()
            peak_threshold = self.peak_threshold_spinbox.value()
            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    eeg_microstate(
                        data=data,
                        n_clusters=n_clusters,
                        peak_threshold=peak_threshold,
                        is_show=False,
                        is_save=True,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Yes")
            else:
                eeg_microstate(
                    data=self.data,
                    n_clusters=n_clusters,
                    peak_threshold=peak_threshold,
                    is_show=False,
                    is_save=True,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.split('.')[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Yes")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)


class RootMeanSquareDialog(FeatureDialog):
    """UI for root mean square (RMS) calculation."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize RMS configuration.

        :param feature_name: Analysis operation name
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.setWindowTitle('Root Mean Square Analysis')
        # RMS parameters section
        rms_groupbox = BFGroupBox('RMS Parameters')
        rms_layout = QFormLayout(rms_groupbox)

        self.sliding_checkbox = QCheckBox('Enable Sliding Window')
        self.sliding_checkbox.setChecked(True)
        rms_layout.addRow(self.sliding_checkbox)

        # Sliding window sub-section
        sliding_groupbox = BFGroupBox('Sliding Window Parameters')
        sliding_layout = QFormLayout(sliding_groupbox)
        self.window_size_spinbox = QSpinBox()
        self.window_size_spinbox.setRange(0, 1000)
        sliding_layout.addRow('Window Size (samples)', self.window_size_spinbox)
        self.overlap_spinbox = QDoubleSpinBox()
        self.overlap_spinbox.setRange(0, 1)
        sliding_layout.addRow('Overlap Ratio', self.overlap_spinbox)
        rms_layout.addRow(sliding_groupbox)

        self.left_layout.addRow(rms_groupbox)
        self.left_layout.addRow(self.save_data_groupbox)

        # Connect checkbox to visibility toggle
        self.sliding_checkbox.stateChanged.connect(self.toggle_sliding_visibility)

    def toggle_sliding_visibility(self):
        """Toggle visibility of sliding window controls."""
        sliding_enabled = self.sliding_checkbox.isChecked()
        self.sliding_groupbox.setVisible(sliding_enabled)

    def run(self):
        """Execute RMS calculation and save results."""
        if self.data or self.import_file_list:
            output_file = self.prepare()
            use_sliding = self.sliding_checkbox.isChecked()
            window_size = self.window_size_spinbox.value()
            overlap = self.overlap_spinbox.value()
            file_type = self.file_type_combobox.currentText()
            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    root_mean_square(
                        data=data,
                        is_save=True,
                        is_sliding=use_sliding,
                        window_size=window_size,
                        overlap_rate=overlap,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Yes")
            else:
                root_mean_square(
                    data=self.data,
                    is_save=True,
                    is_sliding=use_sliding,
                    window_size=window_size,
                    overlap_rate=overlap,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.split('.')[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Yes")
        else:
            self.show_message('Warning', 'Import data first')


class VarianceDialog(FeatureDialog):
    """UI for variance calculation."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize variance analysis configuration.

        :param feature_name: Analysis operation name
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.setWindowTitle('Variance Analysis')
        # Variance parameters section
        var_groupbox = BFGroupBox('Variance Parameters')
        var_layout = QFormLayout(var_groupbox)

        self.sliding_checkbox = QCheckBox('Enable Sliding Window')
        self.sliding_checkbox.setChecked(True)
        var_layout.addRow(self.sliding_checkbox)

        # Sliding window sub-section
        sliding_groupbox = BFGroupBox('Sliding Window Parameters')
        sliding_layout = QFormLayout(sliding_groupbox)
        self.window_size_spinbox = QSpinBox()
        self.window_size_spinbox.setRange(0, 1000)
        sliding_layout.addRow('Window Size (samples)', self.window_size_spinbox)
        self.overlap_spinbox = QDoubleSpinBox()
        self.overlap_spinbox.setRange(0, 1)
        sliding_layout.addRow('Overlap Ratio', self.overlap_spinbox)
        var_layout.addRow(sliding_groupbox)

        self.left_layout.addRow(var_groupbox)
        self.left_layout.addRow(self.save_data_groupbox)

        # Connect checkbox to visibility toggle
        self.sliding_checkbox.stateChanged.connect(self.toggle_sliding_visibility)

    def toggle_sliding_visibility(self):
        """Toggle visibility of sliding window controls."""
        sliding_enabled = self.sliding_checkbox.isChecked()
        self.sliding_groupbox.setVisible(sliding_enabled)

    def run(self):
        """Execute variance calculation and save results."""
        if self.data or self.import_file_list:
            output_file = self.prepare()
            use_sliding = self.sliding_checkbox.isChecked()
            window_size = self.window_size_spinbox.value()
            overlap = self.overlap_spinbox.value()
            file_type = self.file_type_combobox.currentText()
            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    variance(
                        data=data,
                        is_save=True,
                        is_sliding=use_sliding,
                        window_size=window_size,
                        overlap_rate=overlap,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Yes")
            else:
                variance(
                    data=self.data,
                    is_save=True,
                    is_sliding=use_sliding,
                    window_size=window_size,
                    overlap_rate=overlap,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.split('.')[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Yes")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)


class MeanAbsoluteValueDialog(FeatureDialog):
    """UI for mean absolute value (MAV) calculation."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize MAV analysis configuration.

        :param feature_name: Analysis operation name
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.setWindowTitle('Mean Absolute Value Analysis')
        # MAV parameters section
        mav_groupbox = BFGroupBox('MAV Parameters')
        mav_layout = QFormLayout(mav_groupbox)

        self.sliding_checkbox = QCheckBox('Enable Sliding Window')
        self.sliding_checkbox.setChecked(True)
        mav_layout.addRow(self.sliding_checkbox)

        # Sliding window sub-section
        sliding_groupbox = BFGroupBox('Sliding Window Parameters')
        sliding_layout = QFormLayout(sliding_groupbox)
        self.window_size_spinbox = QSpinBox()
        self.window_size_spinbox.setRange(0, 1000)
        sliding_layout.addRow('Window Size (samples)', self.window_size_spinbox)
        self.overlap_spinbox = QDoubleSpinBox()
        self.overlap_spinbox.setRange(0, 1)
        sliding_layout.addRow('Overlap Ratio', self.overlap_spinbox)
        mav_layout.addRow(sliding_groupbox)

        self.left_layout.addRow(mav_groupbox)
        self.left_layout.addRow(self.save_data_groupbox)

        # Connect checkbox to visibility toggle
        self.sliding_checkbox.stateChanged.connect(self.toggle_sliding_visibility)

    def toggle_sliding_visibility(self):
        """Toggle visibility of sliding window controls."""
        sliding_enabled = self.sliding_checkbox.isChecked()
        self.sliding_groupbox.setVisible(sliding_enabled)

    def run(self):
        """Execute MAV calculation and save results."""
        if self.data or self.import_file_list:
            output_file = self.prepare()
            use_sliding = self.sliding_checkbox.isChecked()
            window_size = self.window_size_spinbox.value()
            overlap = self.overlap_spinbox.value()
            file_type = self.file_type_combobox.currentText()
            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    mean_absolute_value(
                        data=data,
                        is_save=True,
                        is_sliding=use_sliding,
                        window_size=window_size,
                        overlap_rate=overlap,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Yes")
            else:
                mean_absolute_value(
                    data=self.data,
                    is_save=True,
                    is_sliding=use_sliding,
                    window_size=window_size,
                    overlap_rate=overlap,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.split('.')[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Yes")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)


class ZeroCrossingDialog(FeatureDialog):
    """UI for zero-crossing rate calculation."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize zero-crossing analysis configuration.

        :param feature_name: Analysis operation name
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.setWindowTitle('Zero-Crossing Rate Analysis')
        # ZC parameters section
        zc_groupbox = BFGroupBox('Zero-Crossing Parameters')
        zc_layout = QFormLayout(zc_groupbox)

        self.sliding_checkbox = QCheckBox('Enable Sliding Window')
        self.sliding_checkbox.setChecked(True)
        zc_layout.addRow(self.sliding_checkbox)

        # Sliding window sub-section
        sliding_groupbox = BFGroupBox('Sliding Window Parameters')
        sliding_layout = QFormLayout(sliding_groupbox)
        self.window_size_spinbox = QSpinBox()
        self.window_size_spinbox.setRange(0, 1000)
        sliding_layout.addRow('Window Size (samples)', self.window_size_spinbox)
        self.overlap_spinbox = QDoubleSpinBox()
        self.overlap_spinbox.setRange(0, 1)
        sliding_layout.addRow('Overlap Ratio', self.overlap_spinbox)
        zc_layout.addRow(sliding_groupbox)

        self.left_layout.addRow(zc_groupbox)
        self.left_layout.addRow(self.save_data_groupbox)

        # Connect checkbox to visibility toggle
        self.sliding_checkbox.stateChanged.connect(self.toggle_sliding_visibility)

    def toggle_sliding_visibility(self):
        """Toggle visibility of sliding window controls."""
        sliding_enabled = self.sliding_checkbox.isChecked()
        self.sliding_groupbox.setVisible(sliding_enabled)

    def run(self):
        """Execute zero-crossing calculation and save results."""
        if self.data or self.import_file_list:
            output_file = self.prepare()
            use_sliding = self.sliding_checkbox.isChecked()
            window_size = self.window_size_spinbox.value()
            overlap = self.overlap_spinbox.value()
            file_type = self.file_type_combobox.currentText()
            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    zero_crossing(
                        data=data,
                        is_save=True,
                        is_sliding=use_sliding,
                        window_size=window_size,
                        overlap_rate=overlap,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Yes")
            else:
                zero_crossing(
                    data=self.data,
                    is_save=True,
                    is_sliding=use_sliding,
                    window_size=window_size,
                    overlap_rate=overlap,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.split('.')[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Yes")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)


class WaveletTransformDialog(FeatureDialog):
    """Dialog window for configuring wavelet transform parameters."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize wavelet transform configuration UI.

        :param feature_name: Name of the transform operation
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.setWindowTitle('Wavelet Transform Configuration')
        self.dwt_groupbox = BFGroupBox('Wavelet Transform Parameters')
        dwt_layout = QFormLayout(self.dwt_groupbox)

        self.level_spinbox = QSpinBox()
        self.level_spinbox.setRange(0, 1000)
        dwt_layout.addRow('Decomposition Level', self.level_spinbox)

        self.basis_function_combox = QComboBox()
        self.basis_function_combox.addItems(['db1', 'db2', 'db3', 'db4'])
        dwt_layout.addRow('Wavelet Basis Function', self.basis_function_combox)

        self.left_layout.addRow(self.dwt_groupbox)
        self.left_layout.addRow(self.save_data_groupbox)

    def run(self):
        """Execute wavelet transform processing and save results."""
        if self.data or self.import_file_list:
            output_file = self.prepare()
            level = self.level_spinbox.value()
            basis_function = self.basis_function_combox.currentText()
            file_type = self.file_type_combobox.currentText()
            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    wavelet_transform(
                        data=data,
                        level=level,
                        basis_function=basis_function,
                        is_save=True,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Yes")
            else:
                wavelet_transform(
                    data=self.data,
                    level=level,
                    basis_function=basis_function,
                    is_save=True,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.split('.')[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Yes")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)


class WaveletPacketEnergyDialog(FeatureDialog):
    """Dialog window for configuring wavelet packet energy analysis."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize wavelet packet energy configuration UI.

        :param feature_name: Name of the analysis operation
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.setWindowTitle('Wavelet Packet Energy Analysis')
        self.dwt_groupbox = BFGroupBox('Wavelet Packet Parameters')
        dwt_layout = QFormLayout(self.dwt_groupbox)

        self.level_spinbox = QSpinBox()
        self.level_spinbox.setRange(0, 1000)
        dwt_layout.addRow('Decomposition Level', self.level_spinbox)

        self.basis_function_combox = QComboBox()
        self.basis_function_combox.addItems(['db1', 'db2', 'db3', 'db4'])
        dwt_layout.addRow('Wavelet Basis Function', self.basis_function_combox)

        self.left_layout.addRow(self.dwt_groupbox)
        self.left_layout.addRow(self.save_data_groupbox)

    def run(self):
        """Execute wavelet packet energy calculation and save results."""
        if self.data or self.import_file_list:
            output_file = self.prepare()
            level = self.level_spinbox.value()
            basis_function = self.basis_function_combox.currentText()
            file_type = self.file_type_combobox.currentText()
            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    wavelet_packet_energy(
                        data=data,
                        level=level,
                        basis_function=basis_function,
                        is_save=True,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Yes")
            else:
                wavelet_packet_energy(
                    data=self.data,
                    level=level,
                    basis_function=basis_function,
                    is_save=True,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.split('.')[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Yes")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)


class ShortTimeFourierTransformDialog(FeatureDialog):
    """Dialog window for configuring Short-Time Fourier Transform."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize STFT configuration UI.

        :param feature_name: Name of the transform operation
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.setWindowTitle('Short-Time Fourier Transform Configuration')
        self.stft_groupbox = BFGroupBox('STFT Parameters')
        stft_layout = QFormLayout(self.stft_groupbox)

        self.window_size_spinbox = QSpinBox()
        self.window_size_spinbox.setRange(0, 100000)
        stft_layout.addRow('Window Size (samples)', self.window_size_spinbox)

        self.noverlap_spinbox = QSpinBox()
        self.noverlap_spinbox.setRange(0, 100000)
        stft_layout.addRow('Overlap Size (samples)', self.noverlap_spinbox)

        self.left_layout.addRow(self.stft_groupbox)
        self.left_layout.addRow(self.save_data_groupbox)

    def run(self):
        """Execute STFT processing and save results."""
        if self.data or self.import_file_list:
            output_file = self.prepare()
            window_size = self.window_size_spinbox.value()
            noverlap = self.noverlap_spinbox.value()
            file_type = self.file_type_combobox.currentText()
            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    short_time_Fourier_transform(
                        data,
                        window_size,
                        noverlap,
                        window_method='hamming',
                        is_save=True,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Yes")
            else:
                short_time_Fourier_transform(
                    self.data,
                    window_size,
                    noverlap,
                    window_method='hamming',
                    is_save=True,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.split('.')[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Yes")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)


class ContinuousWaveletTransformDialog(FeatureDialog):
    """Dialog window for configuring Continuous Wavelet Transform."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize CWT configuration UI.

        :param feature_name: Name of the transform operation
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.setWindowTitle('Continuous Wavelet Transform Configuration')
        self.cwt_groupbox = BFGroupBox('CWT Parameters')
        cwt_layout = QFormLayout(self.cwt_groupbox)

        self.low_scale_spinbox = QSpinBox()
        self.low_scale_spinbox.setRange(0, 100000)
        cwt_layout.addRow('Minimum Scale', self.low_scale_spinbox)

        self.high_scale_spinbox = QSpinBox()
        self.high_scale_spinbox.setRange(0, 100000)
        cwt_layout.addRow('Maximum Scale', self.high_scale_spinbox)

        self.basis_function_combox = QComboBox()
        self.basis_function_combox.addItems(['cmor'])
        cwt_layout.addRow('Wavelet Basis Function', self.basis_function_combox)

        self.left_layout.addRow(self.cwt_groupbox)
        self.left_layout.addRow(self.save_data_groupbox)

    def run(self):
        """Execute CWT processing and save results."""
        if self.data or self.import_file_list:
            output_file = self.prepare()
            widths = np.arange(self.low_scale_spinbox.value(), self.high_scale_spinbox.value())
            basis_function = self.basis_function_combox.currentText()
            file_type = self.file_type_combobox.currentText()
            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    continuous_wavelet_transform(
                        data,
                        widths=widths,
                        basis_function=basis_function,
                        is_save=True,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Yes")
            else:
                continuous_wavelet_transform(
                    self.data,
                    widths=widths,
                    basis_function=basis_function,
                    is_save=True,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.split('.')[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Yes")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)


class HjorthParametersDialog(FeatureDialog):
    """Dialog window for configuring Hjorth parameter calculation."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize Hjorth parameters configuration UI.

        :param feature_name: Name of the analysis operation
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.setWindowTitle('Hjorth Parameters Analysis')
        self.hjorth_groupbox = BFGroupBox('Hjorth Parameters')
        hjorth_layout = QFormLayout(self.hjorth_groupbox)

        self.sliding_checkbox = QCheckBox('Enable Sliding Window')
        self.sliding_checkbox.setChecked(True)
        hjorth_layout.addRow(self.sliding_checkbox)

        self.sliding_groupbox = BFGroupBox('Sliding Window Parameters')
        sliding_layout = QFormLayout(self.sliding_groupbox)
        self.window_size_spinbox = QSpinBox()
        self.window_size_spinbox.setRange(0, 1000)
        sliding_layout.addRow('Window Size (samples)', self.window_size_spinbox)
        self.overlap_rate_spinbox = QDoubleSpinBox()
        self.overlap_rate_spinbox.setRange(0, 1)
        sliding_layout.addRow('Overlap Ratio', self.overlap_rate_spinbox)
        hjorth_layout.addRow(self.sliding_groupbox)

        self.left_layout.addRow(self.hjorth_groupbox)
        self.left_layout.addRow(self.save_data_groupbox)

        # Connect checkbox to visibility toggle
        self.sliding_checkbox.stateChanged.connect(self.toggle_sliding_visibility)
        self.sliding_groupbox.setVisible(True)

    def toggle_sliding_visibility(self):
        """Toggle visibility of sliding window controls."""
        sliding_enabled = self.sliding_checkbox.isChecked()
        self.sliding_groupbox.setVisible(sliding_enabled)

    def run(self):
        """Execute Hjorth parameters calculation and save results."""
        if self.data or self.import_file_list:
            output_file = self.prepare()
            use_sliding = self.sliding_checkbox.isChecked()
            window_size = self.window_size_spinbox.value()
            overlap_rate = self.overlap_rate_spinbox.value()
            file_type = self.file_type_combobox.currentText()
            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    hjorth_parameters(
                        data=data,
                        is_save=True,
                        is_sliding=use_sliding,
                        window_size=window_size,
                        overlap_rate=overlap_rate,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Yes")
            else:
                hjorth_parameters(
                    data=self.data,
                    is_save=True,
                    is_sliding=use_sliding,
                    window_size=window_size,
                    overlap_rate=overlap_rate,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.split('.')[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Yes")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)


class AperiodicParametersDialog(FeatureDialog):
    """Dialog window for configuring aperiodic parameter calculation."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize aperiodic parameters configuration UI.

        :param feature_name: Name of the analysis operation
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.setWindowTitle('Aperiodic Parameters Analysis')
        self.left_layout.addRow(self.save_data_groupbox)

    def run(self):
        """Execute aperiodic parameters calculation and save results."""
        if self.data or self.import_file_list:
            output_file = self.prepare()
            file_type = self.file_type_combobox.currentText()
            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    aperiodic_parameters(
                        data,
                        is_save=True,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Yes")
            else:
                aperiodic_parameters(
                    self.data,
                    is_save=True,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.split('.')[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Yes")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)


class LocalNetworkDialog(FeatureDialog):
    """Dialog window for configuring local network feature extraction."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize local network analysis configuration UI.

        :param feature_name: Name of the analysis operation
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.setWindowTitle('Local Network Feature Extraction')
        self.local_network_groupbox = BFGroupBox('Local Network Parameters')
        local_network_layout = QFormLayout(self.local_network_groupbox)

        self.method_combox = QComboBox()
        self.method_combox.addItems(['cov'])
        local_network_layout.addRow('Correlation Method', self.method_combox)

        self.edge_retention_spinbox = QDoubleSpinBox()
        self.edge_retention_spinbox.setRange(0, 1)
        local_network_layout.addRow('Edge Retention Ratio', self.edge_retention_spinbox)

        self.absolute_threshold_checkbox = QCheckBox('Use Absolute Threshold')
        self.absolute_threshold_checkbox.setChecked(False)
        local_network_layout.addRow(self.absolute_threshold_checkbox)

        self.absolute_groupbox = BFGroupBox('Absolute Threshold Parameters')
        absolute_layout = QFormLayout(self.absolute_groupbox)
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0, 1)
        absolute_layout.addRow('Threshold Value', self.threshold_spinbox)
        local_network_layout.addRow(self.absolute_groupbox)

        self.left_layout.addRow(self.local_network_groupbox)
        self.left_layout.addRow(self.save_data_groupbox)

        # Connect checkbox to visibility toggle
        self.absolute_threshold_checkbox.stateChanged.connect(self.toggle_threshold_visibility)
        self.absolute_groupbox.setVisible(False)

    def toggle_threshold_visibility(self):
        """Toggle visibility of absolute threshold controls."""
        threshold_enabled = self.absolute_threshold_checkbox.isChecked()
        self.absolute_groupbox.setVisible(threshold_enabled)

    def run(self):
        """Execute local network feature extraction and save results."""
        if self.data or self.import_file_list:
            output_file = self.prepare()
            file_type = self.file_type_combobox.currentText()
            edge_retention = self.edge_retention_spinbox.value()
            use_absolute = self.absolute_threshold_checkbox.isChecked()
            threshold = self.threshold_spinbox.value()
            method = self.method_combox.currentText()

            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    local_network_features(
                        data,
                        edge_retention_rate=edge_retention,
                        is_relative_thresholds=not use_absolute,
                        threshold=threshold,
                        method=method,
                        is_save=True,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Yes")
            else:
                local_network_features(
                    self.data,
                    edge_retention_rate=edge_retention,
                    is_relative_thresholds=not use_absolute,
                    threshold=threshold,
                    method=method,
                    is_save=True,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.split('.')[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Yes")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)


class GlobalNetworkDialog(FeatureDialog):
    """Dialog window for configuring global network feature extraction."""

    def __init__(self, feature_name, parent=None):
        """
        Initialize global network analysis configuration UI.

        :param feature_name: Name of the analysis operation
        :type feature_name: str
        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(feature_name, parent)
        self.setWindowTitle('Global Network Feature Extraction')
        self.global_network_groupbox = BFGroupBox('Global Network Parameters')
        global_network_layout = QFormLayout(self.global_network_groupbox)

        self.method_combox = QComboBox()
        self.method_combox.addItems(['cov'])
        global_network_layout.addRow('Correlation Method', self.method_combox)

        self.edge_retention_spinbox = QDoubleSpinBox()
        self.edge_retention_spinbox.setRange(0, 1)
        global_network_layout.addRow('Edge Retention Ratio', self.edge_retention_spinbox)

        self.absolute_threshold_checkbox = QCheckBox('Use Absolute Threshold')
        self.absolute_threshold_checkbox.setChecked(False)
        global_network_layout.addRow(self.absolute_threshold_checkbox)

        self.absolute_groupbox = BFGroupBox('Absolute Threshold Parameters')
        absolute_layout = QFormLayout(self.absolute_groupbox)
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0, 1)
        absolute_layout.addRow('Threshold Value', self.threshold_spinbox)
        global_network_layout.addRow(self.absolute_groupbox)

        self.left_layout.addRow(self.global_network_groupbox)
        self.left_layout.addRow(self.save_data_groupbox)

        # Connect checkbox to visibility toggle
        self.absolute_threshold_checkbox.stateChanged.connect(self.toggle_threshold_visibility)
        self.absolute_groupbox.setVisible(False)

    def toggle_threshold_visibility(self):
        """Toggle visibility of absolute threshold controls."""
        threshold_enabled = self.absolute_threshold_checkbox.isChecked()
        self.absolute_groupbox.setVisible(threshold_enabled)

    def run(self):
        """Execute global network feature extraction and save results."""
        if self.data or self.import_file_list:
            output_file = self.prepare()
            file_type = self.file_type_combobox.currentText()
            edge_retention = self.edge_retention_spinbox.value()
            use_absolute = self.absolute_threshold_checkbox.isChecked()
            threshold = self.threshold_spinbox.value()
            method = self.method_combox.currentText()

            if self.is_folder:
                for i, filePath in enumerate(self.import_file_list):
                    data = read_file_by_path(filePath)
                    global_network_features(
                        data,
                        edge_retention_rate=edge_retention,
                        is_relative_thresholds=not use_absolute,
                        threshold=threshold,
                        method=method,
                        is_save=True,
                        save_path=self.save_file_list[i],
                        save_filestyle=file_type
                    )
                    if self.export_info_checkbox.isChecked():
                        info_file_name = self.save_file_list[i].split('.')[0] + '_info.json'
                        dict_to_info(data, info_file_name)
                    self.process_dialog.update_status(i, "Yes")
            else:
                global_network_features(
                    self.data,
                    edge_retention_rate=edge_retention,
                    is_relative_thresholds=not use_absolute,
                    threshold=threshold,
                    method=method,
                    is_save=True,
                    save_path=output_file,
                    save_filestyle=file_type
                )
                if self.export_info_checkbox.isChecked():
                    info_file_name = output_file.split('.')[0] + '_info.json'
                    dict_to_info(self.data, info_file_name)
                self.process_dialog.update_status(0, "Yes")
        else:
            QMessageBox.warning(None, 'Warning', 'Import data first', QMessageBox.Ok)


def main():
    app = QApplication(sys.argv)
    window = CreateFilesDialog(feature_name='LocalNetwork')
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
