# -*- coding: utf-8 -*-
# @Time    : 2024/6/15 22:14
# @Author  : XXX
# @Site    : 
# @File    : bids.py
# @Software: PyCharm 
# @Comment :
import shutil
import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem, QPushButton,
                             QLineEdit, QFileDialog, QHBoxLayout, QGroupBox, QFormLayout, QComboBox, QScrollArea,
                             QMessageBox)
from PyQt5.QtCore import Qt
import pyedflib

from BrainFusion.io.File_IO import read_bdf, read_edf, save_metadata, read_file, read_file_by_qt
from BrainFusion.utils.files import compareFileSizes
from UI.ui_component import BFSelectWidget, BFPushButton, BFGroupBox, BFScrollArea


class BIDSConverter(QWidget):
    def __init__(self):
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
        layout = QVBoxLayout()
        # 文件选择部分
        file_group = BFGroupBox("File Selection")
        file_layout = QHBoxLayout()
        self.browse_button = BFPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_button)
        self.file_input = QLineEdit(self)
        self.file_input.setPlaceholderText("Select a BDF file")
        file_layout.addWidget(self.file_input)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # 实验信息部分
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

        # 数据信息部分
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
        data_info_layout.addRow("Data Length:", self.data_length_label)

        self.nchan_label = QLabel("N/A")
        data_info_layout.addRow("Number of Channels:", self.nchan_label)

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

        # 可编辑的通道信息部分
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

        # 可编辑的事件信息部分
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

        # BIDS转换按钮
        self.convert_button = BFPushButton("Convert to BIDS")
        self.convert_button.clicked.connect(self.convert_to_bids)
        layout.addWidget(self.convert_button)

        # 创建一个 ScrollArea 并设置它的内容
        scroll_area = BFScrollArea()
        scroll_area.set_layout(layout)

        # 设置主窗口的布局
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)
        self.resize(600, 800)
        self.setWindowTitle('BIDS Converter')
        self.show()

    def browse_file(self):
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
        # 确保 self.data_dict['events'] 存在且正确格式化
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
        index = self.ch_names_list.indexOfTopLevelItem(item)
        if column == 1:  # 如果编辑的是通道类型
            new_channel_type = item.text(1)
            self.data_dict['channel_type'][index] = new_channel_type
        elif column == 2:  # 如果编辑的是物理量最小值
            unit = item.text(2)
            self.data_dict['units'][index] = unit
        elif column == 3:  # 如果编辑的是物理量最大值
            status = item.text(3)
            self.data_dict['channel_status'][index] = status
        elif column == 4:  # 如果编辑的是数字量最小值
            description = item.text(4)
            self.data_dict['channel_status_description'][index] = description

    def handle_event_item_changed(self, item, column):
        if self.data_dict['events']:
            index = self.events_list.indexOfTopLevelItem(item)
            if column == 0:  # 编辑 Onset
                new_onset = float(item.text(0))
                self.data_dict['events'][0][index] = new_onset
            elif column == 1:  # 编辑 Duration
                new_duration = float(item.text(1))
                self.data_dict['events'][1][index] = new_duration
            elif column == 2:  # 编辑 Value
                new_value = int(item.text(2))
                self.data_dict['events'][2][index] = new_value

    def convert_to_bids(self):
        # 打开文件夹选择对话框
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder_path:
            return

        subject_id = self.subject_id_input.text()
        task_name = self.task_name_input.text()
        task_number = self.task_number_input.text()

        if not subject_id or not task_name or not task_number:
            # 如果subject_id, task_name或task_number为空，提示用户填写这些信息
            return

        sub_folder = os.path.join(folder_path, f"sub-{subject_id}")
        eeg_folder = os.path.join(sub_folder, "eeg")
        os.makedirs(eeg_folder, exist_ok=True)

        # 保存原始数据文件
        original_file_path = self.file_input.text()
        _, ext = os.path.splitext(original_file_path)
        new_file_name = f"sub-{subject_id}_task-{task_name}_{task_number}_eeg{ext}"
        new_file_path = os.path.join(eeg_folder, new_file_name)
        shutil.copy(original_file_path, new_file_path)

        # 保存 channels.tsv 和 events.tsv
        channels_filename = f"sub-{subject_id}_task-{task_name}_{task_number}_channels.tsv"
        channels_file_path = os.path.join(eeg_folder, channels_filename)
        self.save_channels_tsv(channels_file_path)

        events_filename = f"sub-{subject_id}_task-{task_name}_{task_number}_events.tsv"
        events_file_path = os.path.join(eeg_folder, events_filename)
        self.save_events_tsv(events_file_path)

        datainfo_filename = f"sub-{subject_id}_task-{task_name}_{task_number}_eeg.json"
        datainfo_file_path = os.path.join(eeg_folder, datainfo_filename)
        self.save_datainfo_json(datainfo_file_path)

        QMessageBox.information(self, 'Success', 'BIDS converted Successfully')

    def save_channels_tsv(self, file_path):
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
        data = {
            'onset': self.data_dict['events'][0],
            'duration': self.data_dict['events'][1],
            'value': self.data_dict['events'][2]
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, sep='\t', index=False)

    def save_datainfo_json(self, file_path):
        data = {
            'TASKNAME': self.task_name_input.text(),
            'SamplingFrequency': self.data_dict['srate'],
            'EEGReference': self.reference.getSelectItem(),
            'PowerlineFrequency': self.powerline.currentText(),
            'Montage': self.data_dict['montage']
        }
        save_metadata(data, path=file_path)

    def on_srate_changed(self):
        # Get the new value from the QLineEdit
        new_srate = self.srate_input.text()

        # Show a confirmation dialog
        reply = QMessageBox.question(self, 'Confirmation',
                                     f"Are you sure you want to change the sample rate to {new_srate}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        # If the user confirms, update the label and the dictionary
        if reply == QMessageBox.Yes:
            self.srate_label.setText(new_srate)
            self.data_dict['srate'] = float(new_srate)

    def on_montage_changed(self):
        new_montage = self.montage_input.text()

        reply = QMessageBox.question(self, 'Confirmation',
                                     f"Are you sure you want to change the montage to {new_montage}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.montage_label.setText(new_montage)
            self.data_dict['montage'] = new_montage


def main():
    app = QApplication(sys.argv)
    ex = BIDSConverter()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
