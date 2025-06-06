# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 16:40
# @Author  : XXX
# @Site    : 
# @File    : viewer_time_frequency.py
# @Software: PyCharm 
# @Comment :
import sys

import matplotlib
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSizePolicy, QFormLayout, QComboBox
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from BrainFusion.pipeLine.pipeLine import short_time_Fourier_transform

matplotlib.use('QtAgg')


class TimeFrequencyViewer(QMainWindow):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.setWindowTitle("Time Frequency Viewer")

        self.combox_channel = QComboBox()
        self.combox_channel.setFixedWidth(120)
        self.combox_channel.addItems(self.data['ch_names'])
        self.combox_channel.currentIndexChanged.connect(self.onStateChanged)

        top_hlayout = QFormLayout()
        top_hlayout.addRow("Channel: ", self.combox_channel)

        self.fig, self.axes = plt.subplots(figsize=(8, 6))
        # self.fig.subplots_adjust(hspace=0, wspace=0.05, bottom=0.08, left=0.05, top=0.98, right=0.98)  # 去除子图间的垂直间隙
        # 创建Matplotlib画布
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout()
        layout.addLayout(top_hlayout)
        layout.addStretch(1)
        layout.addWidget(self.canvas)
        layout.addStretch(1)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.current_colorbar = None
        self.plot(0)

    def plot(self, chan):
        # 清除先前的绘图和colorbar
        if self.current_colorbar:
            self.current_colorbar.remove()
            self.current_colorbar = None

        self.axes.clear()
        # self.axes.figure.clear()
        frequencies, times, Sxx = self.data['data'][chan]  # 选择第一个通道的数据
        print(len(frequencies[0]))
        print(len(times[0]))
        frequencies = frequencies[0]
        times = times[0]

        # print(Sxx)
        self.axes.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
        self.axes.set_ylabel('Frequency [Hz]')
        self.axes.set_xlabel('Time [sec]')
        self.axes.set_title('STFT of Channel ' + str(chan))
        self.current_colorbar = self.axes.figure.colorbar(
            self.axes.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud'),
            ax=self.axes, label='Power/Frequency (dB/Hz)')
        self.axes.figure.canvas.draw()

    def onStateChanged(self):
        index = self.combox_channel.currentIndex()
        self.plot(index)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 生成模拟数据
    n_channels = 32
    srate = 1000  # 采样率
    duration = 10  # 持续时间，以秒为单位
    n_samples = srate * duration  # 每个通道的样本数

    # 模拟的EEG数据
    np.random.seed(42)  # 为了可重复性
    data = np.random.randn(n_channels, n_samples)

    # 创建一个包含模拟数据的字典
    simulated_data = {
        'data': data,
        'srate': srate,
        'nchan': n_channels,
        'ch_names': [f'ch_{i}' for i in range(n_channels)],
        'events': [],  # 假设没有事件
        'montage': 'standard_1020'
    }

    # 调用short_time_Fourier_transform函数
    nperseg = 256
    noverlap = 128
    stft_result = short_time_Fourier_transform(simulated_data, nperseg, noverlap, window_method='hamming')

    main = TimeFrequencyViewer(stft_result)
    main.show()
    sys.exit(app.exec_())
