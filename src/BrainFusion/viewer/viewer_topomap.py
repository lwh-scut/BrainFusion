# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 15:31
# @Author  : XXX
# @Site    : 
# @File    : viewer_topomap.py
# @Software: PyCharm 
# @Comment :
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
    def __init__(self, channel_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle('剔除坏导')
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
        selected_channels = [channel for channel, checkbox in self.checkbox_dict.items() if checkbox.isChecked()]
        return selected_channels


class TopomapViewer(QDialog):
    def __init__(self, data, parent=None):
        super(TopomapViewer, self).__init__(parent)
        self.is_show_sensor = False
        self.setWindowTitle("Topomap Viewer")
        self.setGeometry(100, 100, 1000, 400)
        self.data = data
        self.show_data = np.array([self.data['feature'][key] for key in self.data['feature'].keys()]).T
        self.show_channel_names = self.data['ch_names']
        self.is_relative = True
        # 创建Matplotlib图形
        num_fig = self.show_data.shape[1]
        self.fig, self.axes = plt.subplots(1, num_fig, figsize=(8, 6), sharex=True, sharey=True)
        self.fig.subplots_adjust(hspace=0, wspace=0.05, bottom=0.08, left=0.05, top=0.98, right=0.98)  # 去除子图间的垂直间隙
        # 创建Matplotlib画布
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.title_list = None
        if self.data['type'] == 'eeg_psd':
            self.title_list = ['delta band', 'theta wave band', 'alpha wave band', 'beta wave band', 'gamma wave band']
        elif self.data['type'] == 'eeg_microstate':
            self.title_list = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

        self.table_widget = TabelWidget(self.show_data, self.show_channel_names, self.title_list)

        bottom_layout = QHBoxLayout()

        self.bnt_save = BFPushButton('Save')
        self.bnt_save.setFixedWidth(150)
        # self.bnt_save.setFixedHeight(60)
        self.bnt_setting = BFPushButton('Settings')
        self.bnt_setting.setFixedWidth(150)
        self.bnt_layout = QHBoxLayout()
        self.bnt_layout.addStretch(1)
        self.bnt_layout.addWidget(self.bnt_save)
        self.bnt_layout.addWidget(self.bnt_setting)

        # 创建布局并添加组件
        layout = QVBoxLayout(self)
        layout.addLayout(self.bnt_layout)
        layout.addWidget(self.canvas)
        layout.addWidget(self.table_widget)


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
        self.plot(type=self.data['type'])
        layout.addLayout(bottom_layout)

    def show_remove_bad_channels_dialog(self):
        if self.data is not None:
            dialog = ChannelsDialog(self.data['ch_names'], self)
            result = dialog.exec_()
            if result == QDialog.Accepted:
                selected_channels = dialog.get_selected_channels()
                result_text = ', '.join(selected_channels)
                self.lineedit_excluded_channels.setText(f'{result_text}')
                self.bad_channels = self.lineedit_excluded_channels.text().split(',')
                self.bad_channels = [chan.replace(' ', '') for chan in self.bad_channels]
                self.show_data, self.show_channel_names = drop_channels(raw_data=self.show_data, channels=self.data['ch_names'], bad_channels=self.bad_channels)
                self.plot(type=self.data['type'])
        else:
            QMessageBox.warning(None, 'Warning', '请先导入数据', QMessageBox.Ok)

    def set_relative(self):
        self.is_relative = self.relative_checkbox.isChecked()
        self.plot(type=self.data['type'])

    def set_sensor(self):
        self.is_show_sensor = self.sensor_checkbox.isChecked()
        self.plot(type=self.data['type'])

    def plot(self, type='eeg_psd'):
        if type == 'eeg_psd':
            self.fig.suptitle("EEG Power Spectral Density")
        elif type == 'eeg_microstate':
            self.fig.suptitle("EEG Microstate")
        if self.data:
            # self.fig.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            montage = mne.channels.make_standard_montage('standard_1005')
            info = mne.create_info(ch_names=self.show_channel_names[:30], sfreq=1000, ch_types='eeg')
            if self.is_relative:
                norm_data = min_max_scaling_to_range(self.show_data.T[:, :30])
                data_range = (-1, 1)
            else:
                norm_data = min_max_scaling_by_arrays(self.show_data.T[:, :30])
                data_range = (-1, 1)
            for i, psd in enumerate(norm_data):
                evoked = mne.EvokedArray(data=self.show_data[:30, :], info=info)
                evoked.set_montage(montage)
                self.axes[i].clear()
                if self.title_list:
                    self.axes[i].set_title(self.title_list[i])
                if self.is_show_sensor:
                    mne.viz.plot_topomap(psd, evoked.info,
                                         axes=self.axes[i], show=False
                                         , sensors=True, vlim=data_range, names=self.show_channel_names)
                else:
                    mne.viz.plot_topomap(psd, evoked.info,
                                         axes=self.axes[i], show=False
                                         , sensors=True, vlim=data_range, names=None)
                self.axes[i].figure.canvas.draw()


class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            return str(self._df.iloc[index.row(), index.column()])
        elif index.isValid() and role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._df.columns[section]
            elif orientation == Qt.Vertical:
                return self._df.index[section]
        return None


class TabelWidget(QMainWindow):
    def __init__(self, data, ch_names, columns):
        super().__init__()
        self.setGeometry(100, 100, 800, 600)
        df = pd.DataFrame(data, index=ch_names, columns=columns)
        print(columns)

        self.model = PandasModel(df)
        self.table = QTableView()
        self.table.setModel(self.model)

        # 设置初始列宽度为100
        self.table.horizontalHeader().setDefaultSectionSize(100)
        self.table.resizeColumnsToContents()

        self.checkbox = QCheckBox("Show Table")
        self.checkbox.setChecked(True)
        self.checkbox.stateChanged.connect(self.toggle_table)

        self.lineedit = QLineEdit()
        self.lineedit.setText('100')
        self.lineedit.returnPressed.connect(self.adjust_column_width)

        self.label = QLabel("Column Width:")

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.label)
        h_layout.addWidget(self.lineedit)

        layout = QVBoxLayout()
        layout.addWidget(self.checkbox)
        layout.addLayout(h_layout)
        layout.addWidget(self.table)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def toggle_table(self, state):
        if state == Qt.Checked:
            self.table.show()
        else:
            self.table.hide()

    def adjust_column_width(self):
        try:
            width = int(self.lineedit.text())
            if width > 0:
                self.table.horizontalHeader().setDefaultSectionSize(width)
        except ValueError:
            pass  # 忽略无效输入
