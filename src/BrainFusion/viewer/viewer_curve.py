# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 11:05
# @Author  : XXX
# @Site    : 
# @File    : viewer_curve.py
# @Software: PyCharm 
# @Comment :
import os

import matplotlib
import mne
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, QEvent, pyqtSignal
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QFileDialog, QDialog, QSizePolicy, QVBoxLayout, QPushButton, QHBoxLayout, QCheckBox, \
    QLineEdit, QLabel, QWidget, QMainWindow, QFormLayout, QApplication, QMessageBox, QListWidget, QListWidgetItem, \
    QDialogButtonBox, QScrollArea
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from BrainFusion.io.File_IO import read_neuracle_bdf, read_edf, read_csv, read_txt, read_npy, read_mat, read_json, \
    read_file_by_qt
from scipy import signal
from scipy.signal import iirnotch
from UI.ui_component import BFPushButton

matplotlib.use('QtAgg')

import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QSizePolicy)

from pyqtgraph import PlotWidget, ScatterPlotItem, mkPen, InfiniteLine, TextItem
from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy


def signal_filter(data, fs, lowcut, highcut, filter_order=4, method='Butterworth'):
    """
    Apply bandpass filtering to multi-channel signals.

    :param data: Input signal data
    :type data: np.ndarray
    :param fs: Sampling frequency
    :type fs: float
    :param lowcut: Low frequency cutoff
    :type lowcut: float
    :param highcut: High frequency cutoff
    :type highcut: float
    :param filter_order: Filter order
    :type filter_order: int
    :param method: Filter design method
    :type method: str
    :return: Filtered signal data
    :rtype: np.ndarray
    """
    if method == 'Butterworth':
        b, a = butter_bandpass(lowcut, highcut, fs, filter_order)
    elif method == 'Bessel':
        b, a = bessel_bandpass(lowcut, highcut, fs, filter_order)
    elif method == 'chebyshev':
        b, a = chebyshev_bandpass(lowcut, highcut, fs, filter_order, rp=1)
    filter_data = signal.filtfilt(b, a, data)
    return filter_data


def notch_filter(data, fs, f0=50, Q=30):
    """
    Apply notch filtering to remove powerline interference.

    :param data: Input signal data
    :type data: np.ndarray
    :param fs: Sampling frequency
    :type fs: float
    :param f0: Center frequency
    :type f0: float
    :param Q: Quality factor
    :type Q: float
    :return: Filtered signal data
    :rtype: np.ndarray
    """
    b, a = iirnotch(f0, Q, fs)
    filter_data = signal.filtfilt(b, a, data)
    return filter_data


def butter_bandpass(lowcut, highcut, fs, filter_order=4):
    """
    Design Butterworth bandpass filter coefficients.

    :param lowcut: Low frequency cutoff
    :type lowcut: float
    :param highcut: High frequency cutoff
    :type highcut: float
    :param fs: Sampling frequency
    :type fs: float
    :param filter_order: Filter order
    :type filter_order: int
    :return: Filter coefficients
    :rtype: tuple
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(filter_order, [low, high], btype='band')
    return b, a


def chebyshev_bandpass(lowcut, highcut, fs, filter_order=4, rp=1):
    """
    Design Chebyshev Type I bandpass filter coefficients.

    :param lowcut: Low frequency cutoff
    :type lowcut: float
    :param highcut: High frequency cutoff
    :type highcut: float
    :param fs: Sampling frequency
    :type fs: float
    :param filter_order: Filter order
    :type filter_order: int
    :param rp: Passband ripple (dB)
    :type rp: float
    :return: Filter coefficients
    :rtype: tuple
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.cheby1(filter_order, rp, [low, high], btype='band')
    return b, a


def bessel_bandpass(lowcut, highcut, fs, filter_order=4):
    """
    Design Bessel bandpass filter coefficients.

    :param lowcut: Low frequency cutoff
    :type lowcut: float
    :param highcut: High frequency cutoff
    :type highcut: float
    :param fs: Sampling frequency
    :type fs: float
    :param filter_order: Filter order
    :type filter_order: int
    :return: Filter coefficients
    :rtype: tuple
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.bessel(filter_order, [low, high], btype='band')
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply Butterworth lowpass filtering to signals.

    :param data: Input signal data
    :type data: np.ndarray
    :param cutoff: Cutoff frequency
    :type cutoff: float
    :param fs: Sampling frequency
    :type fs: float
    :param order: Filter order
    :type order: int
    :return: Filtered signal data
    :rtype: np.ndarray
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


class RawCurvePlotDialog(QMainWindow):
    """
    Dialog window for visualizing and annotating physiological signals.

    :param data: Dictionary containing signal data and metadata
    :type data: dict
    :param filePath: Path to original data file
    :type filePath: str
    :param parent: Parent widget
    :type parent: QWidget
    """

    add_start = pyqtSignal(str)
    """Signal emitted when a start marker is added"""

    add_end = pyqtSignal(str)
    """Signal emitted when an end marker is added"""

    remove = pyqtSignal(str)
    """Signal emitted when a marker is removed"""

    import_event = pyqtSignal(str)
    """Signal emitted when events are imported"""

    def __init__(self, data, filePath, parent=None):
        """Initialize visualization interface with signal data."""
        super(RawCurvePlotDialog, self).__init__(parent)
        self.setWindowTitle("Raw Curve Plot Dialog")
        self.setGeometry(100, 100, 800, 600)

        self.srate = data['srate']
        self.data_type = data['type']
        self.ch_names = data['ch_names']
        self.nchan = len(data['ch_names'])
        self.data = np.array(data['data']).copy()
        self.show_data = self.data.copy()

        self.filePath = self.modify_file_path(filePath)  # 文件路径
        self.adjust_file_path()
        self.unit = 0.000001
        self.low_amplitude = -100
        self.high_amplitude = 100
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

        top_h_layout = QHBoxLayout()
        self.checkbox_norch = QCheckBox("Notch Filter")
        self.checkbox_norch.setChecked(True)
        self.lineedit_low_filter = QLineEdit('0.3')
        self.lineedit_low_filter.setFixedWidth(100)
        self.lineedit_high_filter = QLineEdit('35')
        self.lineedit_high_filter.setFixedWidth(100)
        self.lineedit_order_filter = QLineEdit('4')
        self.lineedit_order_filter.setFixedWidth(100)
        self.lineedit_low_amplitude = QLineEdit('-100')
        self.lineedit_low_amplitude.setFixedWidth(50)
        self.lineedit_high_amplitude = QLineEdit('100')
        self.lineedit_high_amplitude.setFixedWidth(50)

        top_h_layout.addWidget(self.checkbox_norch)
        top_h_layout.addSpacing(10)
        top_h_layout.addWidget(QLabel('low-pass frequency:'))
        top_h_layout.addWidget(self.lineedit_low_filter)
        top_h_layout.addSpacing(10)
        top_h_layout.addWidget(QLabel('high-pass frequency:'))
        top_h_layout.addWidget(self.lineedit_high_filter)
        top_h_layout.addSpacing(10)
        top_h_layout.addWidget(QLabel('filter order:'))
        top_h_layout.addWidget(self.lineedit_order_filter)
        top_h_layout.addSpacing(20)
        top_h_layout.addWidget(QLabel('Amplitudes:'))
        top_h_layout.addWidget(self.lineedit_low_amplitude)
        top_h_layout.addWidget(QLabel('~'))
        top_h_layout.addWidget(self.lineedit_high_amplitude)
        top_h_layout.addWidget(QLabel('µV'))
        top_h_layout.addSpacing(10)

        top_h_layout.addStretch(1)
        self.checkbox_norch.stateChanged.connect(self.on_filter_change)
        self.lineedit_low_filter.returnPressed.connect(self.on_filter_change)
        self.lineedit_high_filter.returnPressed.connect(self.on_filter_change)
        self.lineedit_order_filter.returnPressed.connect(self.on_filter_change)

        self.lineedit_low_amplitude.returnPressed.connect(self.on_amplitude_change)
        self.lineedit_high_amplitude.returnPressed.connect(self.on_amplitude_change)

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
        self.tip_label = QLabel('Seconds/per page')
        self.tip_label.setFixedWidth(100)

        self.checkbox_useMouse = QCheckBox("Enable marking")
        self.checkbox_useMouse.setChecked(False)  # 默认不启用鼠标交互

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.button_prev)
        h_layout.addWidget(self.button_next)
        h_layout.addSpacing(200)
        h_layout.addWidget(self.num_samples_per_page_lineedit)  # 添加 LineEdit 控件
        h_layout.addWidget(self.tip_label)
        h_layout.addSpacing(200)
        h_layout.addWidget(self.checkbox_useMouse)

        main_layout = QVBoxLayout()

        main_layout.addLayout(top_h_layout)

        plot_layout = QFormLayout()
        for i, plot_widget in enumerate(self.plot_widgets):
            plot_layout.addRow(self.ch_names[i], plot_widget)
            plot_layout.setSpacing(0)

        main_layout.addLayout(plot_layout)
        main_layout.addLayout(h_layout)
        main_layout.addStretch(1)
        self.clicked_x_list = QListWidget()  # 创建 QListWidget 来显示点击过的 x 值
        self.clicked_x_list.itemDoubleClicked.connect(self.remove_clicked_x)  # 绑定双击删除点击过的 x 值
        self.clicked_x_list.setVisible(False)

        # 在布局中添加 QListWidget
        main_layout.addWidget(self.clicked_x_list)

        self.checkbox_useMouse.stateChanged.connect(lambda: self.clicked_x_list.setVisible(self.checkbox_useMouse.isChecked()))

        # 执行滤波
        if 'preprocess' not in self.data_type:
            self.on_filter_change()

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.plot_data(self.current_page)

    def on_filter_change(self):
        """Apply selected filters to displayed data."""
        try:
            lowcut = float(self.lineedit_low_filter.text())
            highcut = float(self.lineedit_high_filter.text())
            filter_order = int(self.lineedit_order_filter.text())
            msg = self.show_message('waitting', 'data processing...')
            msg.show()
            self.show_data = signal_filter(self.data, self.srate, lowcut=lowcut, highcut=highcut,
                                           filter_order=filter_order)
            if self.checkbox_norch.isChecked():
                self.show_data = np.array(notch_filter(self.show_data, self.srate, f0=50, Q=50))
            msg.close()
            self.plot_data(self.current_page)
        except Exception as e:
            print(e)

    def on_amplitude_change(self):
        try:
            self.low_amplitude = float(self.lineedit_low_amplitude.text())
            self.high_amplitude = float(self.lineedit_high_amplitude.text())
            self.plot_data(self.current_page)
        except Exception as e:
            print(e)

    def show_message(self, title, text):
        """
        Display informational message dialog.

        :param title: Dialog title
        :type title: str
        :param text: Message content
        :type text: str
        :return: Created message box
        :rtype: QMessageBox
        """
        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowTitle(title)
        self.msg_box.setText(text)
        return self.msg_box


    def modify_file_path(self, filePath):
        """
        Generate event marker file path from data file path.

        :param filePath: Original data file path
        :type filePath: str
        :return: Modified event file path
        :rtype: str
        """
        # 去除后缀名，加上 event.tsv
        if filePath:
            filename, ext = os.path.splitext(filePath)
            return filename + "_event.tsv"

    def adjust_file_path(self):
        """Redirect file path to event subdirectory. """
        if self.filePath:
            # 获取原始文件路径的目录和文件名
            original_dir, filename = os.path.split(self.filePath)
            # 更改上级目录为 'event'
            event_dir = os.path.join(os.path.dirname(original_dir), 'event')

            # 更新文件路径
            self.filePath = os.path.join(event_dir, filename)

    def create_file(self):
        """Initialize event marker storage file."""
        # 检测文件夹是否存在，不存在则创建
        directory = os.path.dirname(self.filePath)
        if not os.path.exists(directory):
            os.makedirs(directory)  # 创建文件夹，包括所有必需的中间文件夹

        # 创建文件（如果不存在）
        if not os.path.exists(self.filePath):
            with open(self.filePath, 'w') as f:
                f.write('')
        else:
            self.load_data_from_file()

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
                item = QListWidgetItem(line.strip())
                self.clicked_x_list.addItem(item)

    def plot_data(self, page):
        start_index = page * self.num_samples_per_page
        end_index = start_index + self.num_samples_per_page

        for i in range(self.nchan):
            plot_widget = self.plot_widgets[i]
            plot_widget.clear()
            plot_widget.setBackground('w')  # 设置背景为白色

            # 绘制数据
            plot_widget.plot(self.time_values[start_index:end_index],
                             self.show_data[i, start_index:end_index], pen='k')
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
            plot_widget.setYRange(self.low_amplitude * self.unit, self.high_amplitude * self.unit)

        # 绘制间隔相等的竖直直线
        interval = (self.time_values[end_index - 1] - self.time_values[start_index]) / 6
        for j in range(1, 7):
            x_value = self.time_values[start_index] + interval * j
            for i in range(self.nchan):
                plot_widget = self.plot_widgets[i]
                plot_widget.plot([x_value, x_value], [plot_widget.viewRect().bottom(), plot_widget.viewRect().top()],
                                 pen=mkPen(color=(150, 150, 150), width=2))  # 竖直直线的颜色为灰色，线宽为2像素

        self.draw_green_line()
        self.draw_red_line()

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
                self.low_amplitude = int(new_ylim[0] / self.unit)
                self.high_amplitude = int(new_ylim[1] / self.unit)
                self.lineedit_low_amplitude.setText(str(int(new_ylim[0] / self.unit)))
                self.lineedit_high_amplitude.setText(str(int(new_ylim[1] / self.unit)))
                plot_widget.setYRange(*new_ylim, padding=0)
        elif delta < 0:  # 向下滚动
            for plot_widget in self.plot_widgets:
                ylim = plot_widget.getViewBox().viewRange()[1]
                new_ylim = [ylim[0] * 1.2, ylim[1] * 1.2]
                self.low_amplitude = int(new_ylim[0] / self.unit)
                self.high_amplitude = int(new_ylim[1] / self.unit)
                self.lineedit_low_amplitude.setText(str(int(new_ylim[0] / self.unit)))
                self.lineedit_high_amplitude.setText(str(int(new_ylim[1] / self.unit)))
                plot_widget.setYRange(*new_ylim, padding=0)

    def customMousePressEvent(self, event, idx):
        if self.checkbox_useMouse.isChecked():
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
                self.draw_red_line()  # 绘制黄色直线

    def add_start_x(self, x):
        item = QListWidgetItem("Start:{}".format(round(x, 2)))  # 创建一个 QListWidgetItem 来显示 Start:x
        self.clicked_x_list.addItem(item)  # 将 QListWidgetItem 添加到 QListWidget 中
        self.save_data_to_file()  # 保存数据到文件

    def add_sleep_label(self, x):
        if int(x) == 0:
            item = QListWidgetItem(
                "{}~{}秒-睡眠阶段:未知".format((self.current_page) * 30, (self.current_page + 1) * 30))
        else:
            item = QListWidgetItem("{}~{}秒-睡眠阶段:{}".format((self.current_page) * 30, (self.current_page + 1) * 30,
                                                           int(x)))  # 创建一个 QListWidgetItem 来显示 Start:x
        self.clicked_x_list.addItem(item)  # 将 QListWidgetItem 添加到 QListWidget 中
        self.save_data_to_file()  # 保存数据到文件

    def add_end_x(self, x):
        item = QListWidgetItem("End:{}".format(round(x, 2)))  # 创建一个 QListWidgetItem 来显示 End:x
        self.clicked_x_list.addItem(item)  # 将 QListWidgetItem 添加到 QListWidget 中
        self.save_data_to_file()  # 保存数据到文件

    def draw_green_line(self):
        for x_item in self.clicked_x_list.findItems("Start:", Qt.MatchStartsWith):
            x = float(x_item.text().split(":")[1])
            page_start_time = self.current_page * self.num_time_per_page
            page_end_time = (self.current_page + 1) * self.num_time_per_page
            if page_start_time <= x <= page_end_time:
                for plot_widget in self.plot_widgets:
                    line = InfiniteLine(pos=(x, 0), angle=90, pen=mkPen('g', width=2))
                    plot_widget.addItem(line)
                # 使用HTML格式化文本
                label = TextItem(html='<div style="background-color:green; color: white;">Start</div>',
                                 anchor=(0, 1))
                label.setPos(x, 0)
                self.plot_widgets[0].addItem(label)

    def draw_red_line(self):
        for x_item in self.clicked_x_list.findItems("End:", Qt.MatchStartsWith):
            x = float(x_item.text().split(":")[1])
            page_start_time = self.current_page * self.num_time_per_page
            page_end_time = (self.current_page + 1) * self.num_time_per_page
            if page_start_time <= x <= page_end_time:
                for plot_widget in self.plot_widgets:
                    line = InfiniteLine(pos=(x, 0), angle=90, pen=mkPen('r', width=2))
                    plot_widget.addItem(line)
                # 使用HTML格式化文本
                label = TextItem(html='<div style="background-color:red; color: white;">End</div>', anchor=(0, 1))
                label.setPos(x, 0)
                self.plot_widgets[0].addItem(label)

    def add_clicked_x(self, x):
        item = QListWidgetItem(str(round(x, 2)))  # 创建一个 QListWidgetItem 来显示 x 值
        self.clicked_x_list.addItem(item)  # 将 QListWidgetItem 添加到 QListWidget 中

    def remove_clicked_x(self, text):
        # 遍历列表，找到并删除指定值的项
        for i in range(self.clicked_x_list.count()):
            item = self.clicked_x_list.item(i)
            if item.text() == text:
                self.clicked_x_list.takeItem(i)
                break  # 跳出循环，因为已经找到并删除了项
        self.save_data_to_file()  # 保存数据到文件
        self.plot_data(self.current_page)


class ChannelsDialog(QDialog):
    def __init__(self, channel_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Remove Bad Channels')
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
            checkbox.setChecked(True)
            self.checkbox_dict[channel] = checkbox
            layout.addWidget(checkbox)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

    def get_selected_channels(self):
        selected_channels = [channel for channel, checkbox in self.checkbox_dict.items() if checkbox.isChecked()]
        return selected_channels


class fNIRSPlotDialog(QMainWindow):
    """Dialog for visualizing fNIRS signal data with interactive controls."""

    def __init__(self, data, filePath, parent=None):
        """
        Initialize fNIRS visualization dialog.

        :param data: fNIRS data dictionary containing channels and signals
        :type data: dict
        :param filePath: Path to source data file
        :type filePath: str
        :param parent: Parent widget
        :type parent: QWidget
        """
        super(fNIRSPlotDialog, self).__init__(parent)
        self.setWindowTitle("fNIRS Plot Dialog")
        self.setGeometry(100, 100, 800, 600)
        self.nchan = data['nchan']
        self.srate = data['srate']
        self.ch_names = data['ch_names']
        self.data = np.array(data['data']).copy()
        self.show_data = self.data.copy()
        self.selected_channels = self.ch_names.copy()

        self.filePath = self.modify_file_path(filePath)  # File path
        self.adjust_file_path()
        self.unit = 0.01

        # Total data length
        self.total_samples = self.data.shape[1]
        # Current page
        self.current_page = 0
        # Time per page
        self.num_time_per_page = 5
        self.num_samples_per_page = int(self.num_time_per_page * self.srate)

        self.sample_interval = 1 / self.srate  # Time interval between samples
        self.time_values = np.arange(0, self.total_samples) * self.sample_interval  # Time values array

        self.central_widget = PlotWidget()
        self.setCentralWidget(self.central_widget)

        top_h_layout = QHBoxLayout()
        # Add "Select Channel" button
        btn_remove_bad_channels = BFPushButton('Select Channel')
        btn_remove_bad_channels.clicked.connect(self.show_remove_bad_channels_dialog)
        self.bad_channels_lineedit = QLineEdit(self)
        self.bad_channels_lineedit.setReadOnly(True)
        top_h_layout.addWidget(btn_remove_bad_channels)
        top_h_layout.addWidget(self.bad_channels_lineedit)
        self.dialog = ChannelsDialog(self.ch_names, self)

        # Create PlotWidget for each channel
        self.plot_widgets = []
        for i in range(self.nchan):
            plot_widget = PlotWidget()
            plot_widget.setMinimumHeight(20)
            plot_widget.setMouseEnabled(x=False, y=False)  # Disable mouse interaction
            plot_widget.wheelEvent = self.wheelEvent
            plot_widget.mousePressEvent = lambda event, idx=i: self.customMousePressEvent(event,
                                                                                          idx)  # Bind mouse click
            self.plot_widgets.append(plot_widget)

        # Add navigation buttons
        self.button_prev = QPushButton('Previous')
        self.button_prev.setFixedWidth(200)
        self.button_next = QPushButton('Next')
        self.button_next.setFixedWidth(200)
        self.button_prev.clicked.connect(self.on_prev_click)
        self.button_next.clicked.connect(self.on_next_click)

        self.num_samples_per_page_lineedit = QLineEdit(str(self.num_time_per_page))
        self.num_samples_per_page_lineedit.setFixedWidth(100)
        self.num_samples_per_page_lineedit.setValidator(QIntValidator())  # Only allow integers
        self.num_samples_per_page_lineedit.editingFinished.connect(self.on_samples_per_page_change)
        self.tip_label = QLabel('Seconds/per page')
        self.tip_label.setFixedWidth(100)

        self.checkbox_useMouse = QCheckBox("Enable marking")
        self.checkbox_useMouse.setChecked(False)  # Mouse interaction disabled by default

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.button_prev)
        h_layout.addWidget(self.button_next)
        h_layout.addSpacing(200)
        h_layout.addWidget(self.num_samples_per_page_lineedit)  # Add line edit control
        h_layout.addWidget(self.tip_label)
        h_layout.addSpacing(200)
        h_layout.addWidget(self.checkbox_useMouse)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_h_layout)

        self.plot_layout = QFormLayout()
        for i, plot_widget in enumerate(self.plot_widgets):
            self.plot_layout.addRow(self.ch_names[i], plot_widget)
            self.plot_layout.setSpacing(0)

        main_layout.addLayout(self.plot_layout)
        main_layout.addLayout(h_layout)
        main_layout.addStretch(1)
        self.clicked_x_list = QListWidget()  # Create QListWidget to track marked positions
        self.clicked_x_list.itemDoubleClicked.connect(self.remove_clicked_x)  # Bind double-click to remove items
        self.clicked_x_list.setVisible(False)

        # Add QListWidget to layout
        main_layout.addWidget(self.clicked_x_list)

        self.checkbox_useMouse.stateChanged.connect(
            lambda: self.clicked_x_list.setVisible(self.checkbox_useMouse.isChecked()))

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.plot_data(self.current_page)

    def show_remove_bad_channels_dialog(self):
        """Display channel selection dialog for filtering channels."""
        if self.data is not None:
            result = self.dialog.exec_()
            if result == QDialog.Accepted:
                self.selected_channels = self.dialog.get_selected_channels()
                result_text = ', '.join(self.selected_channels)
                self.bad_channels_lineedit.setText(f'{result_text}')
                self.plot_data(self.current_page)
        else:
            QMessageBox.warning(None, 'Warning', 'Please import data first', QMessageBox.Ok)

    def on_filter_change(self):
        """Apply bandpass and notch filtering to displayed signals."""
        try:
            lowcut = float(self.lineedit_low_filter.text())
            highcut = float(self.lineedit_high_filter.text())
            filter_order = int(self.lineedit_order_filter.text())
            msg = self.show_message('waitting', 'data processing...')
            msg.show()
            self.show_data = signal_filter(self.data, self.srate, lowcut=lowcut, highcut=highcut,
                                           filter_order=filter_order)
            if self.checkbox_norch.isChecked():
                self.show_data = np.array(notch_filter(self.show_data, self.srate, f0=50, Q=50))
            msg.close()
            self.plot_data(self.current_page)
        except Exception as e:
            print(e)

    def on_amplitude_change(self):
        """Adjust vertical scale range of signal waveforms."""
        try:
            self.low_amplitude = float(self.lineedit_low_amplitude.text())
            self.high_amplitude = float(self.lineedit_high_amplitude.text())
            self.plot_data(self.current_page)
        except Exception as e:
            print(e)

    def show_message(self, title, text):
        """
        Create and return informational message dialog.

        :param title: Dialog title
        :type title: str
        :param text: Message content
        :type text: str
        :return: Configured message box
        :rtype: QMessageBox
        """
        self.msg_box = QMessageBox(self)
        self.msg_box.setWindowTitle(title)
        self.msg_box.setText(text)
        return self.msg_box

    def modify_file_path(self, filePath):
        """
        Generate event marker file path from data file path.

        :param filePath: Original data file path
        :type filePath: str
        :return: Modified event file path
        :rtype: str
        """
        # Replace extension with _event.tsv
        filename, ext = os.path.splitext(filePath)
        return filename + "_event.tsv"

    def adjust_file_path(self):
        """Adjust file path to point to event subdirectory."""
        # Get directory and filename
        original_dir, filename = os.path.split(self.filePath)
        # Navigate to 'event' subdirectory
        event_dir = os.path.join(os.path.dirname(original_dir), 'event')
        # Update file path
        self.filePath = os.path.join(event_dir, filename)

    def create_file(self):
        """Initialize event marker storage file."""
        # Check if directory exists
        directory = os.path.dirname(self.filePath)
        if not os.path.exists(directory):
            os.makedirs(directory)  # Create all intermediate directories

        # Create file if missing
        if not os.path.exists(self.filePath):
            with open(self.filePath, 'w') as f:
                f.write('')
        else:
            self.load_data_from_file()

    def save_data_to_file(self):
        """Save event markers to disk file."""
        # Save QListWidget content to file
        with open(self.filePath, 'w') as f:
            for i in range(self.clicked_x_list.count()):
                item = self.clicked_x_list.item(i)
                f.write(item.text() + '\n')

    def load_data_from_file(self):
        """Load event markers from disk file into UI."""
        # Load data from file to QListWidget
        with open(self.filePath, 'r') as f:
            for line in f:
                item = QListWidgetItem(line.strip())
                self.clicked_x_list.addItem(item)

    def set_plot_visiable(self, index, isVisiable=True):
        """
        Set visibility of channel plot widgets.

        :param index: Plot index
        :type index: int
        :param isVisiable: Visibility flag
        :type isVisiable: bool
        """
        widget = self.plot_layout.itemAt(index, QFormLayout.FieldRole).widget()
        label = self.plot_layout.itemAt(index, QFormLayout.LabelRole).widget()
        if widget is not None:
            if isVisiable:
                widget.show()
                label.show()
            else:
                widget.hide()
                label.hide()

    def plot_data(self, page):
        """
        Render fNIRS signals for specified time window.

        :param page: Page index to display
        :type page: int
        """
        start_index = page * self.num_samples_per_page
        end_index = start_index + self.num_samples_per_page
        for i in range(self.nchan):
            if self.ch_names[i] in self.selected_channels:
                self.set_plot_visiable(i, True)
                plot_widget = self.plot_widgets[i]
                plot_widget.clear()
                plot_widget.setBackground('w')  # White background

                # Plot data
                plot_widget.plot(self.time_values[start_index:end_index],
                                 self.show_data[i, start_index:end_index], pen='k')

                # Hide Y axis values
                plot_widget.getAxis('left').setStyle(showValues=False)

                # Hide X axis for all but bottom channel
                if i < self.nchan - 1:
                    plot_widget.getAxis('bottom').setStyle(showValues=False)
                    plot_widget.getAxis('bottom').setPen(None)
                else:
                    plot_widget.getAxis('bottom').setStyle(showValues=True)
                plot_widget.getAxis('left').setPen(None)
            else:
                self.set_plot_visiable(i, False)

        # Draw vertical grid lines
        interval = (self.time_values[end_index - 1] - self.time_values[start_index]) / 6
        for j in range(1, 7):
            x_value = self.time_values[start_index] + interval * j
            for i in range(self.nchan):
                if self.ch_names[i] in self.selected_channels:
                    self.set_plot_visiable(i, True)
                    plot_widget = self.plot_widgets[i]
                    plot_widget.plot([x_value, x_value],
                                     [plot_widget.viewRect().bottom(), plot_widget.viewRect().top()],
                                     pen=mkPen(color=(150, 150, 150), width=2))  # Gray lines
                else:
                    self.set_plot_visiable(i, False)

        self.draw_green_line()
        self.draw_red_line()

    def on_prev_click(self):
        """Navigate to previous time segment."""
        self.current_page = max(self.current_page - 1, 0)
        self.plot_data(self.current_page)

    def on_next_click(self):
        """Navigate to next time segment."""
        max_pages = self.total_samples // self.num_samples_per_page
        self.current_page = min(self.current_page + 1, max_pages)
        self.plot_data(self.current_page)

    def on_samples_per_page_change(self):
        """Adjust time window duration per page."""
        self.num_time_per_page = int(self.num_samples_per_page_lineedit.text())
        if self.num_time_per_page > 0:
            self.num_samples_per_page = int(self.num_time_per_page * self.srate)
            self.plot_data(self.current_page)
        else:
            # Restore original value on invalid input
            self.num_samples_per_page_lineedit.setText(str(self.num_time_per_page))

    def wheelEvent(self, event):
        """
        Handle mouse wheel for vertical scaling of waveforms.

        :param event: Wheel event
        :type event: QWheelEvent
        """
        delta = event.angleDelta().y() / 120
        if delta > 0:  # Scrolling up
            for plot_widget in self.plot_widgets:
                ylim = plot_widget.getViewBox().viewRange()[1]
                new_ylim = [ylim[0] * 0.95, ylim[1] * 0.95]
                plot_widget.setYRange(*new_ylim, padding=0)
        elif delta < 0:  # Scrolling down
            for plot_widget in self.plot_widgets:
                ylim = plot_widget.getViewBox().viewRange()[1]
                new_ylim = [ylim[0] * 1.05, ylim[1] * 1.05]
                plot_widget.setYRange(*new_ylim, padding=0)

    def customMousePressEvent(self, event, idx):
        """
        Handle plot clicks for event markers placement.

        :param event: Mouse event
        :type event: QMouseEvent
        :param idx: Channel index
        :type idx: int
        """
        if self.checkbox_useMouse.isChecked():
            if event.button() == Qt.LeftButton:
                # Left click - add start marker
                pos = event.pos()
                x = self.plot_widgets[idx].plotItem.vb.mapSceneToView(pos).x()
                self.add_start_x(x)
                self.draw_green_line()
            elif event.button() == Qt.RightButton:
                # Right click - add end marker
                pos = event.pos()
                x = self.plot_widgets[idx].plotItem.vb.mapSceneToView(pos).x()
                self.add_end_x(x)
                self.draw_red_line()

    def add_start_x(self, x):
        """
        Add start marker at specified time position.

        :param x: Time position in seconds
        :type x: float
        """
        item = QListWidgetItem("Start:{}".format(round(x, 2)))
        self.clicked_x_list.addItem(item)
        self.save_data_to_file()

    def add_sleep_label(self, x):
        """
        Add sleep stage annotation to event markers.

        :param x: Sleep stage identifier
        :type x: int
        """
        if int(x) == 0:
            item = QListWidgetItem(
                "{}~{}秒-睡眠阶段:未知".format((self.current_page) * 30, (self.current_page + 1) * 30))
        else:
            item = QListWidgetItem("{}~{}秒-睡眠阶段:{}".format((self.current_page) * 30, (self.current_page + 1) * 30,
                                                                int(x)))
        self.clicked_x_list.addItem(item)
        self.save_data_to_file()

    def add_end_x(self, x):
        """
        Add end marker at specified time position.

        :param x: Time position in seconds
        :type x: float
        """
        item = QListWidgetItem("End:{}".format(round(x, 2)))
        self.clicked_x_list.addItem(item)
        self.save_data_to_file()

    def draw_green_line(self):
        """Render green vertical lines for all start markers in current view."""
        for x_item in self.clicked_x_list.findItems("Start:", Qt.MatchStartsWith):
            x = float(x_item.text().split(":")[1])
            page_start_time = self.current_page * self.num_time_per_page
            page_end_time = (self.current_page + 1) * self.num_time_per_page
            if page_start_time <= x <= page_end_time:
                for plot_widget in self.plot_widgets:
                    line = InfiniteLine(pos=(x, 0), angle=90, pen=mkPen('g', width=2))
                    plot_widget.addItem(line)
                # Format text with HTML
                label = TextItem(html='<div style="background-color:green; color: white;">Start</div>',
                                 anchor=(0, 1))
                label.setPos(x, 0)
                self.plot_widgets[0].addItem(label)

    def draw_red_line(self):
        """Render red vertical lines for all end markers in current view."""
        for x_item in self.clicked_x_list.findItems("End:", Qt.MatchStartsWith):
            x = float(x_item.text().split(":")[1])
            page_start_time = self.current_page * self.num_time_per_page
            page_end_time = (self.current_page + 1) * self.num_time_per_page
            if page_start_time <= x <= page_end_time:
                for plot_widget in self.plot_widgets:
                    line = InfiniteLine(pos=(x, 0), angle=90, pen=mkPen('r', width=2))
                    plot_widget.addItem(line)
                # Format text with HTML
                label = TextItem(html='<div style="background-color:red; color: white;">End</div>', anchor=(0, 1))
                label.setPos(x, 0)
                self.plot_widgets[0].addItem(label)

    def add_clicked_x(self, x):
        """
        Add generic time marker at specified position.

        :param x: Time position in seconds
        :type x: float
        """
        item = QListWidgetItem(str(round(x, 2)))
        self.clicked_x_list.addItem(item)

    def remove_clicked_x(self, text):
        """
        Remove specified marker from UI and storage.

        :param text: Marker identifier to remove
        :type text: str
        """
        # Find and remove item with matching text
        for i in range(self.clicked_x_list.count()):
            item = self.clicked_x_list.item(i)
            if item.text() == text:
                self.clicked_x_list.takeItem(i)
                break
        self.save_data_to_file()
        self.plot_data(self.current_page)


def min_max_scaling_to_range(array, new_min=-1, new_max=1):
    """
    Scale array values to specified range using min-max normalization.

    :param array: Input array to be normalized
    :type array: np.ndarray
    :param new_min: Minimum value of target range
    :type new_min: float
    :param new_max: Maximum value of target range
    :type new_max: float
    :return: Scaled array
    :rtype: np.ndarray
    """
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = new_min + (new_max - new_min) * (array - min_val) / (max_val - min_val)
    return normalized_array


def min_max_scaling_by_arrays(arrays, new_min=-1, new_max=1):
    """
    Scale multiple arrays to specified range using min-max normalization.

    :param arrays: List of input arrays to be normalized
    :type arrays: list[np.ndarray]
    :param new_min: Minimum value of target range
    :type new_min: float
    :param new_max: Maximum value of target range
    :type new_max: float
    :return: Array of scaled arrays
    :rtype: np.ndarray
    """
    normalized_arrays = []
    for array in arrays:
        normalized_array = min_max_scaling_to_range(array, new_min, new_max)
        normalized_arrays.append(normalized_array)
    return np.array(normalized_arrays)


def plot_raw_by_file(widget, path=None):
    """
    Launch raw signal visualization dialog for file data.

    :param widget: Parent widget for the dialog
    :type widget: QWidget
    :param path: Optional file path to load
    :type path: str
    :return: Tuple containing file path and plot dialog reference
    :rtype: tuple(str, RawCurvePlotDialog)
    """
    data, file_path = read_file_by_qt(widget, path)
    raw_curve_dialog = None
    if data:
        raw_curve_dialog = RawCurvePlotDialog(data=data, filePath=file_path[0], parent=widget)

    return file_path[0], raw_curve_dialog
