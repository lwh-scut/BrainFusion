import shutil
import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem, QPushButton,
                             QLineEdit, QFileDialog, QHBoxLayout, QGroupBox, QFormLayout, QComboBox, QScrollArea,
                             QMessageBox)
from PyQt5.QtCore import Qt

from BrainFusion.io.File_IO import save_metadata, read_file_by_qt
from BrainFusion.utils.files import compareFileSizes
from UI.ui_component import BFSelectWidget, BFPushButton, BFGroupBox, BFScrollArea


class BIDSConverter(QWidget):
    """BIDS format conversion tool for EEG/fNIRS data."""

    def __init__(self):
        """
        Initialize BIDS Converter application.
        """
        super().__init__()

        self.data_dict = {
            'data': None,
            'srate': None,
            'events': None,
            'nchan': None,
            'ch_names': None,
            'type': 'eeg',
            'montage': None,
            'channel_type': None,
            'units': None,
            'channel_status': None,
            'channel_status_description': None,
            'EEGReference': None,
            'PowerLineFrequency': None
        }
        self.initUI()

    def initUI(self):
        """Initialize user interface components."""
        layout = QVBoxLayout()
        # File selection section
        file_group = BFGroupBox("File Selection")
        file_layout = QHBoxLayout()
        self.browse_button = BFPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_button)
        self.file_input = QLineEdit(self)
        self.file_input.setPlaceholderText("Select a BDF/EDF file")
        file_layout.addWidget(self.file_input)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Experiment information section
        experiment_group = BFGroupBox("Experiment Information")
        experiment_layout = QFormLayout()

        self.subject_id_input = QLineEdit(self)
        self.subject_id_input.setPlaceholderText("Enter Subject ID")
        experiment_layout.addRow("Subject ID:", self.subject_id_input)

        self.task_name_input = QLineEdit(self)
        self.task_name_input.setPlaceholderText("Enter Task Name")
        experiment_layout.addRow("Task Name:", self.task_name_input)

        self.task_number_input = QLineEdit(self)
        self.task_number_input.setPlaceholderText("Enter Task Number")
        experiment_layout.addRow("Task Number:", self.task_number_input)

        experiment_group.setLayout(experiment_layout)
        layout.addWidget(experiment_group)

        # Data information section
        data_info_group = BFGroupBox("Data Information")
        data_info_layout = QFormLayout()

        self.type_label = QLabel(f"{self.data_dict['type']}")
        data_info_layout.addRow("Data Type:", self.type_label)

        self.srate_label = QLabel("N/A")
        self.srate_input = QLineEdit(self)
        self.srate_input.setPlaceholderText("Enter Sample Rate")
        self.srate_input.returnPressed.connect(self.on_srate_changed)
        data_info_layout.addRow("Sample Rate:", self.srate_label)
        data_info_layout.addRow("Edit Sample Rate:", self.srate_input)

        self.data_length_label = QLabel("N/A seconds")
        data_info_layout.addRow("Data Duration:", self.data_length_label)

        self.nchan_label = QLabel("N/A")
        data_info_layout.addRow("Channel Count:", self.nchan_label)

        self.powerline = QComboBox()
        self.powerline.addItems(['50 Hz', '60 Hz'])
        data_info_layout.addRow('PowerLine Frequency:', self.powerline)

        self.montage_label = QLabel("N/A")
        self.montage_input = QLineEdit(self)
        self.montage_input.setPlaceholderText("Enter Montage")
        self.montage_input.returnPressed.connect(self.on_montage_changed)
        data_info_layout.addRow("Montage:", self.montage_label)
        data_info_layout.addRow("Edit Montage:", self.montage_input)

        self.reference = BFSelectWidget('Reference')
        data_info_layout.addRow(self.reference)

        data_info_group.setLayout(data_info_layout)
        layout.addWidget(data_info_group)

        # Channel information section
        ch_info_group = BFGroupBox("Channel Information")
        ch_info_layout = QVBoxLayout()

        self.ch_names_list = QTreeWidget()
        self.ch_names_list.setColumnCount(5)
        self.ch_names_list.setHeaderLabels(
            ['Name', 'Type', 'Unit', 'Status', 'Description'])
        self.ch_names_list.itemChanged.connect(self.handle_item_changed)
        ch_info_layout.addWidget(self.ch_names_list)

        ch_info_group.setLayout(ch_info_layout)
        layout.addWidget(ch_info_group)

        # Events information section
        events_group = BFGroupBox("Events")
        events_layout = QVBoxLayout()

        self.add_event_button = BFPushButton("Add Event")
        self.add_event_button.setFixedWidth(120)
        self.add_event_button.clicked.connect(self.add_event)
        events_layout.addWidget(self.add_event_button)
        self.events_list = QTreeWidget()
        self.events_list.setColumnCount(3)
        self.events_list.setHeaderLabels(['Onset', 'Duration', 'Value'])
        self.events_list.itemChanged.connect(self.handle_event_item_changed)
        events_layout.addWidget(self.events_list)

        events_group.setLayout(events_layout)
        layout.addWidget(events_group)

        # BIDS conversion button
        self.convert_button = BFPushButton("Convert to BIDS")
        self.convert_button.clicked.connect(self.convert_to_bids)
        layout.addWidget(self.convert_button)

        # Create scrollable content area
        scroll_area = BFScrollArea()
        scroll_area.set_layout(layout)

        # Set main window layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)
        self.resize(600, 800)
        self.setWindowTitle('BIDS Converter')
        self.show()

    def browse_file(self):
        """
        Browse for EEG/fNIRS data files.

        :returns: None
        """
        data, file_path = read_file_by_qt(self)

        if data:
            if len(file_path) > 1:
                oneset_list, duration_list, value_list = zip(*data['events'])
                self.data_dict['events'] = [oneset_list, duration_list, value_list]
                file_path = compareFileSizes(file_path[0], file_path[1])
            else:
                file_path = file_path[0]
                self.data_dict['events'] = data['events']
            self.file_input.setText(file_path)
            self.data_dict['data'] = data['data']
            self.data_dict['srate'] = data['srate']

            print(self.data_dict['events'])
            self.data_dict['nchan'] = data['nchan']
            self.data_dict['ch_names'] = data['ch_names']
            self.data_dict['type'] = 'eeg'
            self.data_dict['montage'] = None
            self.data_dict['channel_type'] = ['Unknown' for label in data['ch_names']]
            if 'units' in data.keys():
                self.data_dict['units'] = data['units']
            else:
                self.data_dict['units'] = ['uV' for label in data['ch_names']]
            self.data_dict['channel_status'] = ['good' for label in data['ch_names']]
            self.data_dict['channel_status_description'] = ['N/A' for label in data['ch_names']]
            self.update_ui()

    def update_ui(self):
        """Update UI with current data information."""
        self.srate_label.setText(f"{self.data_dict['srate']} Hz")
        self.type_label.setText(f"{self.data_dict['type']}")
        self.montage_label.setText(f"{self.data_dict['montage']}")
        self.nchan_label.setText(f"{self.data_dict['nchan']}")

        if self.data_dict['data'] is not None and len(self.data_dict['data']) > 0:
            data_length = len(self.data_dict['data'][0]) / self.data_dict['srate']
        else:
            data_length = 0
        self.data_length_label.setText(f"{data_length:.2f} seconds")

        self.reference.addItems(self.data_dict['ch_names'])

        self.ch_names_list.clear()
        if self.data_dict['ch_names']:
            for i, (ch, ch_type, ch_unit, ch_status, ch_description) in enumerate(
                    zip(self.data_dict['ch_names'], self.data_dict['channel_type'],
                        self.data_dict['units'], self.data_dict['channel_status'],
                        self.data_dict['channel_status_description'])):
                item = QTreeWidgetItem([
                    ch,
                    ch_type,
                    ch_unit,
                    ch_status,
                    ch_description
                ])
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                self.ch_names_list.addTopLevelItem(item)

        self.events_list.clear()
        if self.data_dict['events'] and len(self.data_dict['events']) == 3:
            onset, duration, value = self.data_dict['events']
            events_df = pd.DataFrame({'onset': onset, 'duration': duration, 'value': value})
            for _, row in events_df.iterrows():
                item = QTreeWidgetItem([
                    str(row['onset']),
                    str(row['duration']),
                    str(row['value'])
                ])
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                self.events_list.addTopLevelItem(item)

    def add_event(self):
        """Add new event to events list."""
        # Ensure events structure exists
        if self.data_dict['events'] is None:
            self.data_dict['events'] = [[], [], []]
        if isinstance(self.data_dict['events'], np.ndarray):
            self.data_dict['events'] = [self.data_dict['events'][0].tolist(), self.data_dict['events'][1].tolist(),
                                        self.data_dict['events'][2].tolist()]

        if isinstance(self.data_dict['events'][0], np.ndarray):
            self.data_dict['events'] = [self.data_dict['events'][0].tolist(), self.data_dict['events'][1].tolist(),
                                        self.data_dict['events'][2].tolist()]

        self.data_dict['events'][0].append(0.0)
        self.data_dict['events'][1].append(0.0)
        self.data_dict['events'][2].append(0)
        self.update_ui()

    def handle_item_changed(self, item, column):
        """
        Handle channel property changes.

        :param item: Modified item
        :type item: QTreeWidgetItem
        :param column: Modified column index
        :type column: int
        """
        index = self.ch_names_list.indexOfTopLevelItem(item)
        if column == 1:  # Channel type
            new_channel_type = item.text(1)
            self.data_dict['channel_type'][index] = new_channel_type
        elif column == 2:  # Unit
            unit = item.text(2)
            self.data_dict['units'][index] = unit
        elif column == 3:  # Status
            status = item.text(3)
            self.data_dict['channel_status'][index] = status
        elif column == 4:  # Description
            description = item.text(4)
            self.data_dict['channel_status_description'][index] = description

    def handle_event_item_changed(self, item, column):
        """
        Handle event property changes.

        :param item: Modified event item
        :type item: QTreeWidgetItem
        :param column: Modified column index
        :type column: int
        """
        if self.data_dict['events']:
            index = self.events_list.indexOfTopLevelItem(item)
            if column == 0:  # Onset time
                new_onset = float(item.text(0))
                self.data_dict['events'][0][index] = new_onset
            elif column == 1:  # Duration
                new_duration = float(item.text(1))
                self.data_dict['events'][1][index] = new_duration
            elif column == 2:  # Value
                new_value = int(item.text(2))
                self.data_dict['events'][2][index] = new_value

    def convert_to_bids(self):
        """Convert data to BIDS format structure."""
        # Select output folder
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder_path:
            return

        subject_id = self.subject_id_input.text()
        task_name = self.task_name_input.text()
        task_number = self.task_number_input.text()

        if not subject_id or not task_name or not task_number:
            return

        # Create BIDS directory structure
        sub_folder = os.path.join(folder_path, f"sub-{subject_id}")
        eeg_folder = os.path.join(sub_folder, "eeg")
        os.makedirs(eeg_folder, exist_ok=True)

        # Copy original data file
        original_file_path = self.file_input.text()
        _, ext = os.path.splitext(original_file_path)
        new_file_name = f"sub-{subject_id}_task-{task_name}_{task_number}_eeg{ext}"
        new_file_path = os.path.join(eeg_folder, new_file_name)
        shutil.copy(original_file_path, new_file_path)

        # Save metadata files
        channels_filename = f"sub-{subject_id}_task-{task_name}_{task_number}_channels.tsv"
        channels_file_path = os.path.join(eeg_folder, channels_filename)
        self.save_channels_tsv(channels_file_path)

        events_filename = f"sub-{subject_id}_task-{task_name}_{task_number}_events.tsv"
        events_file_path = os.path.join(eeg_folder, events_filename)
        self.save_events_tsv(events_file_path)

        datainfo_filename = f"sub-{subject_id}_task-{task_name}_{task_number}_eeg.json"
        datainfo_file_path = os.path.join(eeg_folder, datainfo_filename)
        self.save_datainfo_json(datainfo_file_path)

        QMessageBox.information(self, 'Success', 'BIDS conversion successful')

    def save_channels_tsv(self, file_path):
        """
        Save channel metadata to TSV file.

        :param file_path: Output file path
        :type file_path: str
        """
        data = {
            'name': self.data_dict['ch_names'],
            'type': self.data_dict['channel_type'],
            'unit': self.data_dict['units'],
            'status': self.data_dict['channel_status'],
            'status_description': self.data_dict['channel_status_description']
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, sep='\t', index=False)

    def save_events_tsv(self, file_path):
        """
        Save events metadata to TSV file.

        :param file_path: Output file path
        :type file_path: str
        """
        data = {
            'onset': self.data_dict['events'][0],
            'duration': self.data_dict['events'][1],
            'value': self.data_dict['events'][2]
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, sep='\t', index=False)

    def save_datainfo_json(self, file_path):
        """
        Save dataset metadata to JSON file.

        :param file_path: Output file path
        :type file_path: str
        """
        data = {
            'TASKNAME': self.task_name_input.text(),
            'SamplingFrequency': self.data_dict['srate'],
            'EEGReference': self.reference.getSelectItem(),
            'PowerlineFrequency': self.powerline.currentText(),
            'Montage': self.data_dict['montage']
        }
        save_metadata(data, path=file_path)

    def on_srate_changed(self):
        """Update sample rate value."""
        new_srate = self.srate_input.text()
        reply = QMessageBox.question(self, 'Confirmation',
                                     f"Change sample rate to {new_srate}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.srate_label.setText(new_srate)
            self.data_dict['srate'] = float(new_srate)

    def on_montage_changed(self):
        """Update electrode montage."""
        new_montage = self.montage_input.text()
        reply = QMessageBox.question(self, 'Confirmation',
                                     f"Change montage to {new_montage}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.montage_label.setText(new_montage)
            self.data_dict['montage'] = new_montage


def main():
    """Launch BIDS converter application."""
    app = QApplication(sys.argv)
    ex = BIDSConverter()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()