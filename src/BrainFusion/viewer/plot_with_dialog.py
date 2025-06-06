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
    def __init__(self, data, filePath, parent=None):
        super(RawCurvePlotDialog, self).__init__(parent)
        self.setWindowTitle("Raw Curve Plot Dialog")
        self.setGeometry(100, 100, 800, 600)
        self.data = np.array(data['data'])
        self.nchan = data['nchan']
        self.srate = data['srate']
        self.ch_names = data['ch_names']
        self.filePath = self.modify_file_path(filePath)  # 文件路径
        # 总数据长度
        self.total_samples = self.data.shape[1]
        # 当前页数
        self.current_page = 0
        # 每页显示数据
        self.num_time_per_page = 5
        self.num_samples_per_page = int(self.num_time_per_page * self.srate)

        self.sample_interval = 1 / self.srate  # 计算每个数据点之间的时间间隔
        self.time_values = np.arange(0, self.total_samples) * self.sample_interval  # 生成时间值数组

        self.central_widget = PlotWidget()
        self.setCentralWidget(self.central_widget)

        # 创建每个通道的 PlotWidget
        self.plot_widgets = []
        for i in range(self.nchan):
            plot_widget = PlotWidget()
            plot_widget.setMinimumHeight(20)
            plot_widget.setMouseEnabled(x=False, y=False)  # 禁用鼠标操作
            plot_widget.wheelEvent = self.wheelEvent
            # plot_widget.getViewBox().installEventFilter(self)  # 安装事件过滤器
            plot_widget.mousePressEvent = lambda event, idx=i: self.customMousePressEvent(event, idx)  # 绑定鼠标点击事件
            self.plot_widgets.append(plot_widget)

        # 添加向前和向后翻页的按钮
        self.button_prev = QPushButton('Previous')
        self.button_prev.setFixedWidth(200)
        self.button_next = QPushButton('Next')
        self.button_next.setFixedWidth(200)
        self.button_prev.clicked.connect(self.on_prev_click)
        self.button_next.clicked.connect(self.on_next_click)

        self.num_samples_per_page_lineedit = QLineEdit(str(self.num_time_per_page))
        self.num_samples_per_page_lineedit.setFixedWidth(100)
        self.num_samples_per_page_lineedit.setValidator(QIntValidator())  # 仅允许输入整数
        self.num_samples_per_page_lineedit.editingFinished.connect(self.on_samples_per_page_change)
        self.tip_label = QLabel('Sec/Page: ')
        self.tip_label.setFixedWidth(100)

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.button_prev)
        h_layout.addWidget(self.button_next)
        h_layout.addSpacing(200)
        h_layout.addWidget(self.tip_label)
        h_layout.addWidget(self.num_samples_per_page_lineedit)  # 添加 LineEdit 控件

        main_layout = QFormLayout()
        for i, plot_widget in enumerate(self.plot_widgets):
            main_layout.addRow(self.ch_names[i], plot_widget)
            main_layout.setSpacing(0)
        main_layout.addItem(h_layout)
        self.clicked_x_list = QListWidget()  # 创建 QListWidget 来显示点击过的 x 值
        self.clicked_x_list.itemDoubleClicked.connect(self.remove_clicked_x)  # 绑定双击删除点击过的 x 值

        # 在布局中添加 QListWidget
        main_layout.addWidget(self.clicked_x_list)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.create_file()

        self.plot_data(self.current_page)

    def modify_file_path(self, filePath):
        # 去除后缀名，加上 event.tsv
        filename, ext = os.path.splitext(filePath)
        return filename + "_event.tsv"

    def create_file(self):
        # 创建一个.tsv文件
        if os.path.exists(self.filePath):
            self.load_data_from_file()
            print(111)
        else:
            with open(self.filePath, 'w') as f:
                f.write('')

    def save_data_to_file(self):
        # 将 QListWidget 中的数据保存到文件中
        with open(self.filePath, 'w') as f:
            for i in range(self.clicked_x_list.count()):
                item = self.clicked_x_list.item(i)
                f.write(item.text() + '\n')

    def load_data_from_file(self):
        # 从文件中加载数据到 QListWidget 中
        with open(self.filePath, 'r') as f:
            for line in f:
                self.clicked_x_list.addItem(line.strip())

    def plot_data(self, page):
        start_index = page * self.num_samples_per_page
        end_index = start_index + self.num_samples_per_page

        for i in range(self.nchan):
            plot_widget = self.plot_widgets[i]
            plot_widget.clear()
            plot_widget.setBackground('w')  # 设置背景为白色

            # 绘制数据
            plot_widget.plot(self.time_values[start_index:end_index],
                             self.data[i, start_index:end_index], pen='k')
            # 设置 y 轴标签
            # plot_widget.setLabel('left', self.data['ch_names'][i], units='V')
            # 隐藏 y 轴刻度数值
            plot_widget.getAxis('left').setStyle(showValues=False)

            # 隐藏 x 轴
            if i < self.nchan - 1:
                plot_widget.getAxis('bottom').setStyle(showValues=False)
                plot_widget.getAxis('bottom').setPen(None)
            else:
                plot_widget.getAxis('bottom').setStyle(showValues=True)
            plot_widget.getAxis('left').setPen(None)
        self.draw_green_line()
        self.draw_yellow_line()

    def on_prev_click(self):
        self.current_page = max(self.current_page - 1, 0)
        self.plot_data(self.current_page)

    def on_next_click(self):
        max_pages = self.total_samples // self.num_samples_per_page
        self.current_page = min(self.current_page + 1, max_pages)
        self.plot_data(self.current_page)

    def on_samples_per_page_change(self):
        self.num_time_per_page = int(self.num_samples_per_page_lineedit.text())
        if self.num_time_per_page > 0:
            self.num_samples_per_page = int(self.num_time_per_page * self.srate)
            self.plot_data(self.current_page)
        else:
            # 如果输入的值不合法，恢复原始值
            self.num_samples_per_page_lineedit.setText(str(self.num_time_per_page))

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        if delta > 0:  # 向上滚动
            for plot_widget in self.plot_widgets:
                ylim = plot_widget.getViewBox().viewRange()[1]
                new_ylim = [ylim[0] * 0.8, ylim[1] * 0.8]
                plot_widget.setYRange(*new_ylim, padding=0)
        elif delta < 0:  # 向下滚动
            for plot_widget in self.plot_widgets:
                ylim = plot_widget.getViewBox().viewRange()[1]
                new_ylim = [ylim[0] * 1.2, ylim[1] * 1.2]
                plot_widget.setYRange(*new_ylim, padding=0)

    def customMousePressEvent(self, event, idx):
        if event.button() == Qt.LeftButton:
            # 左键点击事件
            pos = event.pos()
            x = self.plot_widgets[idx].plotItem.vb.mapSceneToView(pos).x()
            self.add_start_x(x)  # 添加 Start:x 到 QListWidget 中
            self.draw_green_line()  # 绘制绿色直线
        elif event.button() == Qt.RightButton:
            # 右键点击事件
            pos = event.pos()
            x = self.plot_widgets[idx].plotItem.vb.mapSceneToView(pos).x()
            self.add_end_x(x)  # 添加 End:x 到 QListWidget 中
            self.draw_yellow_line()  # 绘制黄色直线

    def add_start_x(self, x):
        item = QListWidgetItem("Start:{}".format(x))  # 创建一个 QListWidgetItem 来显示 Start:x
        self.clicked_x_list.addItem(item)  # 将 QListWidgetItem 添加到 QListWidget 中
        self.save_data_to_file()  # 保存数据到文件

    def add_end_x(self, x):
        item = QListWidgetItem("End:{}".format(x))  # 创建一个 QListWidgetItem 来显示 End:x
        self.clicked_x_list.addItem(item)  # 将 QListWidgetItem 添加到 QListWidget 中
        self.save_data_to_file()  # 保存数据到文件

    def draw_green_line(self):
        # 绘制绿色直线
        for x_item in self.clicked_x_list.findItems("Start:", Qt.MatchStartsWith):
            x = float(x_item.text().split(":")[1])
            for plot_widget in self.plot_widgets:
                page_start_time = self.current_page * self.num_time_per_page
                page_end_time = (self.current_page + 1) * self.num_time_per_page
                if page_start_time <= x <= page_end_time:
                    plot_widget.addItem(InfiniteLine(pos=(x, 0), angle=90, pen=mkPen('g', width=2)))

    def draw_yellow_line(self):
        # 绘制黄色直线
        for x_item in self.clicked_x_list.findItems("End:", Qt.MatchStartsWith):
            x = float(x_item.text().split(":")[1])
            for plot_widget in self.plot_widgets:
                page_start_time = self.current_page * self.num_time_per_page
                page_end_time = (self.current_page + 1) * self.num_time_per_page
                if page_start_time <= x <= page_end_time:
                    plot_widget.addItem(InfiniteLine(pos=(x, 0), angle=90, pen=mkPen('y', width=2)))

    def add_clicked_x(self, x):
        item = QListWidgetItem(str(x))  # 创建一个 QListWidgetItem 来显示 x 值
        self.clicked_x_list.addItem(item)  # 将 QListWidgetItem 添加到 QListWidget 中

    def remove_clicked_x(self, item):
        self.clicked_x_list.takeItem(self.clicked_x_list.row(item))  # 删除选定的 x 值
        self.plot_data(self.current_page)


# class RawCurvePlotDialog(QDialog):
#     def __init__(self, data, parent=None):
#         super(RawCurvePlotDialog, self).__init__(parent)
#         self.setWindowTitle("Raw Curve Plot Dialog")
#         self.setGeometry(100, 100, 800, 600)
#         self.data = data
#         # 总数据长度
#         self.total_samples = np.array(self.data['data']).shape[1]
#         # 当前页数
#         self.current_page = 0
#         # 每页显示数据
#         self.num_time_per_page = 5
#         self.num_samples_per_page = int(self.num_time_per_page * self.data['srate'])
#         # 创建Matplotlib图形
#         self.fig, self.axes = plt.subplots(self.data['nchan'], 1, figsize=(8, 6), sharex=True, sharey=False)
#
#         self.sample_interval = 1 / self.data['srate']  # 计算每个数据点之间的时间间隔
#         self.time_values = np.arange(0, self.total_samples) * self.sample_interval  # 生成时间值数组
#         self.plot_data(self.current_page)
#         # 添加时间轴
#         self.fig.subplots_adjust(hspace=0, wspace=0, bottom=0.08, left=0.05, top=0.98, right=0.98)  # 去除子图间的垂直间隙
#
#         self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
#
#         # 创建Matplotlib画布
#         self.canvas = FigureCanvas(self.fig)
#         self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#
#         self.num_samples_per_page_lineedit = QLineEdit(str(self.num_time_per_page))
#         self.num_samples_per_page_lineedit.setFixedWidth(100)
#         self.num_samples_per_page_lineedit.setValidator(QIntValidator())  # 仅允许输入整数
#         self.num_samples_per_page_lineedit.editingFinished.connect(self.on_samples_per_page_change)
#         self.tip_label = QLabel('Sec/Page: ')
#         self.tip_label.setFixedWidth(100)
#
#         # 创建布局并添加组件
#         layout = QVBoxLayout()
#         layout.addWidget(self.canvas)
#         # 添加向前和向后翻页的按钮
#         self.button_prev = QPushButton('Previous')
#         self.button_prev.setFixedWidth(200)
#         self.button_next = QPushButton('Next')
#         self.button_next.setFixedWidth(200)
#         self.button_prev.clicked.connect(self.on_prev_click)
#         self.button_next.clicked.connect(self.on_next_click)
#
#         h_layout = QHBoxLayout()
#         h_layout.addWidget(self.button_prev)
#         h_layout.addWidget(self.button_next)
#         h_layout.addSpacing(200)
#         h_layout.addWidget(self.tip_label)
#         h_layout.addWidget(self.num_samples_per_page_lineedit)  # 添加 LineEdit 控件
#         layout.addLayout(h_layout)
#
#         self.setLayout(layout)
#
#     def plot_data(self, page):
#         start_index = page * self.num_samples_per_page
#         end_index = start_index + self.num_samples_per_page
#         for i in range(self.data['nchan']):
#             self.axes[i].clear()
#             self.axes[i].plot(self.time_values[start_index:end_index],
#                               np.array(self.data['data'])[i, start_index:end_index], color='black', linewidth=0.5)
#             self.axes[i].set_ylabel(self.data['ch_names'][i], rotation=0, ha='right')
#             self.axes[i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False,
#                                      left=False,
#                                      right=False, labelleft=False)
#             self.axes[i].spines['top'].set_color('lightgrey')
#             self.axes[i].spines['bottom'].set_color('lightgrey')
#             self.axes[i].spines['right'].set_color('lightgrey')
#             self.axes[i].spines['left'].set_color('lightgrey')
#         # 设置 x 轴刻度
#         self.axes[-1].tick_params(axis='both', which='both', bottom=True, top=False, left=False, right=False,
#                                   labelbottom=True)
#         self.axes[-1].set_xlabel('Time (s)')  # 设置 x 轴标签
#         self.axes[-1].set_xlim(left=self.time_values[start_index],
#                                right=self.time_values[end_index - 1])  # 设置 x 轴范围
#
#     def on_scroll(self, event):
#         if event.button == 'up':
#             for ax in self.axes:
#                 ax.set_ylim(ax.get_ylim()[0] * 0.8, ax.get_ylim()[1] * 0.8)
#         elif event.button == 'down':
#             for ax in self.axes:
#                 ax.set_ylim(ax.get_ylim()[0] * 1.2, ax.get_ylim()[1] * 1.2)
#         self.canvas.draw_idle()
#
#     def on_prev_click(self):
#         self.current_page = max(self.current_page - 1, 0)
#         self.plot_data(self.current_page)
#         self.canvas.draw()
#
#     def on_next_click(self):
#         max_pages = self.total_samples // self.num_samples_per_page
#         self.current_page = min(self.current_page + 1, max_pages)
#         self.plot_data(self.current_page)
#         self.canvas.draw()
#
#     def trans_fnirs_data(self):
#         if self.data['type'] == 'fnirs_preprocessed':
#             result = self.data['data'][0]
#             result.extend(self.data['data'][1])
#             channel = self.data['ch_names'][0]
#             channel.extend(self.data['ch_names'][1])
#             self.data['data'] = result
#             self.data['ch_names'] = channel
#
#     def on_samples_per_page_change(self):
#         new_samples_per_page = int(self.num_samples_per_page_lineedit.text())
#         if new_samples_per_page > 0:
#             self.num_samples_per_page = int(new_samples_per_page * self.data['srate'])
#             self.plot_data(self.current_page)
#             self.canvas.draw()
#         else:
#             # 如果输入的值不合法，恢复原始值
#             self.num_samples_per_page_lineedit.setText(str(self.num_samples_per_page))


class EEGPSDPlotDialog(QDialog):
    def __init__(self, data, parent=None):
        super(EEGPSDPlotDialog, self).__init__(parent)
        self.setWindowTitle("EEG Topomap Plot Dialog")
        self.setGeometry(100, 100, 1000, 400)
        self.data = data
        self.is_relative = True
        # 创建Matplotlib图形
        num_fig = np.array(self.data['data']).shape[1]
        self.fig, self.axes = plt.subplots(1, num_fig, figsize=(8, 6), sharex=True, sharey=True)
        self.fig.subplots_adjust(hspace=0, wspace=0.05, bottom=0.08, left=0.05, top=0.98, right=0.98)  # 去除子图间的垂直间隙
        # 创建Matplotlib画布
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 创建布局并添加组件
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.relative_checkbox = QCheckBox('Relative')
        self.relative_checkbox.setChecked(True)
        self.relative_checkbox.stateChanged.connect(self.choose_show_type)
        layout.addWidget(self.relative_checkbox)
        self.plot(type=self.data['type'])

    def choose_show_type(self):
        self.is_relative = self.relative_checkbox.isChecked()
        self.plot(type=self.data['type'])

    def plot(self, type='eeg_psd'):
        title_list = None
        if type == 'eeg_psd':
            title_list = ['Δ wave band', 'θ wave band', 'α wave band', 'β wave band', 'γ wave band']
            self.fig.suptitle("EEG Power Spectral Density")
        elif type == 'eeg_microstate':
            title_list = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
            self.fig.suptitle("EEG Microstate")
        if self.data:
            # self.fig.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            montage = mne.channels.make_standard_montage('standard_1020')
            info = mne.create_info(ch_names=self.data['ch_names'], sfreq=self.data['srate'], ch_types='eeg')
            if self.is_relative:
                norm_data = min_max_scaling_to_range(np.array(self.data['data']).T)
                data_range = (-1, 1)
            else:
                norm_data = min_max_scaling_by_arrays(np.array(self.data['data']).T)
                data_range = (-1, 1)
            for i, psd in enumerate(norm_data):
                evoked = mne.EvokedArray(data=np.array(self.data['data']), info=info)
                evoked.set_montage(montage)
                self.axes[i].clear()
                if title_list:
                    self.axes[i].set_title(title_list[i])
                mne.viz.plot_topomap(psd, evoked.info,
                                     axes=self.axes[i], show=False
                                     , sensors=True, vlim=data_range)
                self.axes[i].figure.canvas.draw()


def min_max_scaling_to_range(array, new_min=-1, new_max=1):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = new_min + (new_max - new_min) * (array - min_val) / (max_val - min_val)
    return normalized_array


def min_max_scaling_by_arrays(arrays, new_min=-1, new_max=1):
    normalized_arrays = []
    for array in arrays:
        normalized_array = min_max_scaling_to_range(array, new_min, new_max)
        normalized_arrays.append(normalized_array)
    return np.array(normalized_arrays)


def plot_raw_by_file(widget, path=None):
    def trans_data(data):
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
        data = trans_data(data)
        # plot_raw(data=data['data'], channel=data['ch_names'])
        raw_curve_dialog = RawCurvePlotDialog(data=data, filePath=path[0], parent=widget)
        raw_curve_dialog.show()


def plot_eeg_psd_by_file(widget, path=None):
    data = None
    if path is None:
        # 打开文件对话框，支持选择多个文件
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        try:
            # select bdf or edf file
            path, _ = QFileDialog.getOpenFileNames(widget, 'Open Files', '',
                                                   'All Files (*);;Two Bdf Files (*.bdf);;Edf '
                                                   'File (*.edf);;Text File (*.txt);;Json '
                                                   'File(*.json);;Mat File(*.mat)',
                                                   options=options)
        except Exception as e:
            print(e)
        if len(path) == 1:
            file_type = path[0].split('.')[-1]
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
            if path[0].split('.')[-1] == 'bdf':
                data = read_neuracle_bdf(path, is_data_transform=True)
        if data:
            eeg_psd_dialog = EEGPSDPlotDialog(data=data, parent=widget)
            eeg_psd_dialog.show()


def plot_raw(data, channel=None, sharey=False, line_color='black', linewidth=0.5):
    # 判断数据类型
    if isinstance(data, np.ndarray):
        # 获取数据维度
        dimensions = data.ndim
    elif isinstance(data, list):
        # 获取数据维度
        dimensions = len(np.array(data).shape)
    else:
        print("Unsupported data type.")
        return None
    if dimensions == 1:
        length = np.array(data).shape[0]
        num_channels = 1
        if channel is None:
            channel = ['channel']
        fig, axes = plt.subplots(num_channels, 1, figsize=(10, 6), sharex=True, sharey=sharey)
        axes.plot(data, color=line_color, linewidth=linewidth)
        axes.set_ylabel(f' {channel[0]}', rotation=0, ha='right')
        axes.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False,
                         right=False,
                         labelleft=False)
        axes.spines['top'].set_color('lightgrey')  # 设置坐标轴边框颜色
        axes.spines['bottom'].set_color('lightgrey')
        axes.spines['right'].set_color('lightgrey')
        axes.spines['left'].set_color('lightgrey')
        # 调整曲线的起点离y轴的距离
        axes.set_xlim(left=-10, right=length)
        fig.subplots_adjust(hspace=0, wspace=0, bottom=0.02, left=0.1, top=0.98, right=0.98)  # 去除子图间的垂直间隙
        plt.show()

    elif dimensions == 2:
        data = np.array(data)
        length = data.shape[1]
        num_channels = data.shape[0]
        if channel is None:
            channel = [str(i) for i in range(1, num_channels + 1)]
        fig, axes = plt.subplots(num_channels, 1, figsize=(10, 6), sharex=True, sharey=sharey)
        for i in range(num_channels):
            axes[i].plot(data[i, :30000], color=line_color, linewidth=linewidth)
            axes[i].set_ylabel(f' {channel[i]}', rotation=0, ha='right')
            axes[i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False,
                                right=False,
                                labelleft=False)
            axes[i].spines['top'].set_color('lightgrey')  # 设置坐标轴边框颜色
            axes[i].spines['bottom'].set_color('lightgrey')
            axes[i].spines['right'].set_color('lightgrey')
            axes[i].spines['left'].set_color('lightgrey')
            # 调整曲线的起点离y轴的距离
            axes[i].set_xlim(left=-10)
        # 添加时间轴
        # 生成时间轴数据
        axes[-1].tick_params(axis='both', which='both', bottom=True, top=False, left=False, right=False,
                             labelbottom=True)
        axes[-1].spines['top'].set_color('lightgrey')
        axes[-1].spines['bottom'].set_color('lightgrey')
        axes[-1].spines['right'].set_color('lightgrey')
        axes[-1].spines['left'].set_color('lightgrey')
        fig.subplots_adjust(hspace=0, wspace=0, bottom=0.05, left=0.05, top=0.98, right=0.98)  # 去除子图间的垂直间隙
        plt.show()
    else:
        print(f"Data has {dimensions} dimensions.Not support.")
        return None
