import sys

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QComboBox, QListWidget, QListWidgetItem,
    QLabel, QDialog, QFormLayout, QLineEdit, QSpinBox, QMessageBox, QInputDialog, QScrollArea, QCheckBox,
    QTableWidgetItem, QTableWidget
)
from PyQt5.QtCore import Qt, pyqtSignal

from BrainFusion.io.File_IO import read_xlsx
from BrainFusion.utils.normalize import min_max_scaling_to_range, min_max_scaling_by_arrays
from BrainFusion.utils.transform import read_info
from UI.ui_component import BFPushButton


def feature_dict_to_dataframe(feature_dict):
    """
    将 feature_dict 转换为 pandas DataFrame，并确保 Type 列位于最后。

    :param feature_dict: read_xlsx 函数生成的字典
    :return: pandas DataFrame
    """
    # 提取通道名
    ch_names = feature_dict["ch_names"]
    type_value = feature_dict["type"]  # Type 是唯一值

    # 创建数据字典，先不加 Type 列
    data = {"Channel": ch_names}

    # 添加特征列
    for feature, values in feature_dict["feature"].items():
        data[feature] = values

    # 转换为 DataFrame
    df = pd.DataFrame(data)

    # 将 Type 列添加到最后
    df["Type"] = type_value

    return df


def get_selected_data(feature_dict, selected_channels, selected_features):
    """
    获取选定通道和特征的数据。

    :param feature_dict: 从 `read_xlsx` 返回的字典
    :param selected_channels: 选定的通道列表
    :param selected_features: 选定的特征列表
    :return: 一个字典，键是特征名，值是一个包含选定通道对应数值的列表
    """
    selected_data = {}

    # 获取通道索引
    channel_indices = [i for i, ch in enumerate(feature_dict['ch_names']) if ch in selected_channels]

    for feature in selected_features:
        if feature not in feature_dict['feature']:
            continue

        # 根据索引筛选对应的特征数据
        selected_data[feature] = []
        for idx in channel_indices:
            selected_data[feature].append(feature_dict['feature'][feature][idx])

    return selected_data


class SelectItemsDialog(QDialog):
    def __init__(self, items, title, selected_items=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)

        self.items = items
        self.selected_items = selected_items if selected_items else []

        layout = QVBoxLayout()

        # 创建复选框
        self.checkboxes = []
        for item in self.items:
            checkbox = QCheckBox(item)
            checkbox.setChecked(item in self.selected_items)  # 如果该项已在selected_items中，勾选
            self.checkboxes.append(checkbox)
            layout.addWidget(checkbox)

        # 全选和全不选
        hlayout = QHBoxLayout()
        self.select_all_button = QPushButton("全选")
        self.select_all_button.clicked.connect(self.select_all)
        hlayout.addWidget(self.select_all_button)

        # 确认按钮
        self.confirm_button = QPushButton("确认")
        self.confirm_button.clicked.connect(self.accept)
        hlayout.addWidget(self.confirm_button)

        layout.addLayout(hlayout)

        self.setLayout(layout)

    def select_all(self):
        if self.select_all_button.isChecked():
            for checkbox in self.checkboxes:
                checkbox.setChecked(False)
        else:
            for checkbox in self.checkboxes:
                checkbox.setChecked(True)

    def get_selected_items(self):
        self.selected_items = [checkbox.text() for checkbox in self.checkboxes if checkbox.isChecked()]
        return self.selected_items


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)

    def plot(self, df, selected_channels, selected_features, plot_type, settings):
        self.ax.clear()  # 清空之前的绘图

        if plot_type == "曲线图":
            x_labels = selected_features  # X轴使用特征名称
            for channel in selected_channels:
                # 获取该通道的数据，按顺序提取选中的特征值
                data = [
                    df.loc[df.iloc[:, 0] == channel, feature].values[0] if not df.loc[
                        df.iloc[:, 0] == channel, feature].empty else None
                    for feature in selected_features
                ]
                self.ax.plot(x_labels, data, marker='o', label=channel)

            self.ax.set_xticks(range(len(x_labels)))
            self.ax.set_xticklabels(x_labels, rotation=45, ha="right")

        elif plot_type == "柱状图":
            x_indices = range(len(selected_channels))
            bar_width = 0.8 / len(selected_features)  # 根据特征数量计算每组柱子的宽度
            for i, feature in enumerate(selected_features):
                values = [
                    df[df.iloc[:, 0] == channel][feature].mean()
                    for channel in selected_channels
                ]
                positions = [x + i * bar_width for x in x_indices]
                self.ax.bar(positions, values, bar_width, label=feature)

            self.ax.set_xticks([x + (len(selected_features) - 1) * bar_width / 2 for x in x_indices])
            self.ax.set_xticklabels(selected_channels)

        # 应用用户设置
        self.ax.set_title(settings.get("title", ""))
        self.ax.set_xlabel(settings.get("xlabel", ""))
        self.ax.set_ylabel(settings.get("ylabel", ""))
        self.ax.legend()
        self.draw()  # 刷新图像


class BarPlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)

    def plot(self, data, selected_channels, selected_features, settings):
        self.ax.clear()  # 清空之前的绘图
        x_indices = range(len(selected_channels))
        bar_width = 0.8 / len(selected_features)  # 根据特征数量计算每组柱子的宽度
        select_data = get_selected_data(data, selected_channels, selected_features)
        for i, feature in enumerate(selected_features):
            values = select_data[feature]
            positions = [x + i * bar_width for x in x_indices]
            self.ax.bar(positions, values, bar_width, label=feature)

        self.ax.set_xticks([x + (len(selected_features) - 1) * bar_width / 2 for x in x_indices])
        self.ax.set_xticklabels(selected_channels)

        self.ax.set_title(settings.get("title", ""))
        self.ax.set_xlabel(settings.get("xlabel", ""))
        self.ax.set_ylabel(settings.get("ylabel", ""))
        self.ax.legend()
        self.draw()


class TablePlotCanvas(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.horizontalHeader().setDefaultSectionSize(200)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    def plot(self, df):
        """
        在表格中显示数据。

        :param df: pandas DataFrame，包含要显示的数据
        """
        self.clear()

        # 设置行列数和标题
        self.setRowCount(len(df))
        self.setColumnCount(len(df.columns))
        self.setHorizontalHeaderLabels(df.columns)

        # 填充数据
        for i in range(len(df)):
            for j in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iloc[i, j]))
                self.setItem(i, j, item)


class CurvePlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)

    def plot(self, data, selected_channels, selected_features, settings):
        self.ax.clear()  # 清空之前的绘图
        x_labels = selected_features  # X轴使用特征名称

        select_data = get_selected_data(data, selected_channels, selected_features)
        select_channel_data = []
        for i, feature in enumerate(selected_features):
            select_channel_data.append(select_data[feature])
        select_channel_data = np.array(select_channel_data).T

        for i, channel in enumerate(selected_channels):
            chan_data = select_channel_data[i]
            self.ax.plot(x_labels, chan_data, marker='o', label=channel)

        self.ax.set_xticks(range(len(x_labels)))
        self.ax.set_xticklabels(x_labels, rotation=45, ha="right")
        # 应用用户设置
        self.ax.set_title(settings.get("title", ""))
        self.ax.set_xlabel(settings.get("xlabel", ""))
        self.ax.set_ylabel(settings.get("ylabel", ""))
        self.ax.legend()
        self.draw()  # 刷新图像


class TopomapCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot(self, data, info_dict, selected_channels, selected_features, is_relative=False, is_show_sensor=False):
        if data:
            num_fig = len(selected_features)
            self.axes = self.fig.subplots(1, num_fig, sharex=True, sharey=True)
            # self.fig.subplots_adjust(hspace=0, wspace=0.05, bottom=0.08, left=0.05, top=0.98, right=0.98)  # 去除子图间的垂直间隙

            select_data = get_selected_data(data, selected_channels, selected_features)
            select_channel_data = []
            for i, feature in enumerate(selected_features):
                select_channel_data.append(select_data[feature])
            select_channel_data = np.array(select_channel_data)

            if is_relative:
                norm_data = min_max_scaling_to_range(select_channel_data)
                data_range = (-1, 1)
            else:
                norm_data = min_max_scaling_by_arrays(select_channel_data)
                data_range = (-1, 1)

            if 'montage' in info_dict.keys():
                montage = mne.channels.make_standard_montage(info_dict['montage'])
                info = mne.create_info(ch_names=selected_channels, sfreq=info_dict['srate'], ch_types='eeg')

                for i, psd in enumerate(norm_data):
                    evoked = mne.EvokedArray(data=select_channel_data.T, info=info)
                    evoked.set_montage(montage)
                    self.axes[i].clear()
                    self.axes[i].set_title(selected_features[i])
                    if is_show_sensor:
                        mne.viz.plot_topomap(psd, evoked.info,
                                             axes=self.axes[i], show=False
                                             , sensors=True, vlim=data_range, names=selected_channels)
                    else:
                        mne.viz.plot_topomap(psd, evoked.info,
                                             axes=self.axes[i], show=False
                                             , sensors=False, vlim=data_range, names=None)
                    self.axes[i].figure.canvas.draw()

            elif 'loc' in info_dict.keys():
                pass
            else:
                return


class PlotSettingsDialog(QWidget):
    closed_signal = pyqtSignal()

    def __init__(self, parent=None, initial_settings=None):
        super().__init__(parent)
        self.setWindowTitle("绘图设置")
        self.resize(300, 200)

        self.layout = QVBoxLayout()
        self.form_layout = QFormLayout()

        # 添加表单字段
        self.title_input = QLineEdit()
        self.xlabel_input = QLineEdit()
        self.ylabel_input = QLineEdit()
        self.width_input = QSpinBox()
        self.width_input.setRange(1, 50)
        self.height_input = QSpinBox()
        self.height_input.setRange(1, 50)

        # 默认值加载
        if initial_settings:
            self.title_input.setText(initial_settings.get("title", ""))
            self.xlabel_input.setText(initial_settings.get("xlabel", ""))
            self.ylabel_input.setText(initial_settings.get("ylabel", ""))
            self.width_input.setValue(initial_settings.get("width", 8))
            self.height_input.setValue(initial_settings.get("height", 6))
        else:
            self.width_input.setValue(8)
            self.height_input.setValue(6)

        self.form_layout.addRow("主标题:", self.title_input)
        self.form_layout.addRow("xLabel:", self.xlabel_input)
        self.form_layout.addRow("yLabel:", self.ylabel_input)
        self.form_layout.addRow("宽度:", self.width_input)
        self.form_layout.addRow("高度:", self.height_input)

        self.layout.addLayout(self.form_layout)
        self.save_button = QPushButton("保存设置")
        self.save_button.clicked.connect(self.close)
        self.layout.addWidget(self.save_button)

        self.setLayout(self.layout)

    def get_settings(self):
        return {
            "title": self.title_input.text(),
            "xlabel": self.xlabel_input.text(),
            "ylabel": self.ylabel_input.text(),
            "width": self.width_input.value(),
            "height": self.height_input.value()
        }

    def closeEvent(self, event):
        # 在关闭窗口时发射信号
        print("QWidget 被关闭了")
        self.closed_signal.emit()  # 发射信号

        event.accept()  # 确保窗口关闭


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("特征绘图工具")
        self.resize(800, 600)

        self.df = None
        self.channels = []
        self.features = []

        self.channel_dialog = SelectItemsDialog(self.channels, "选择通道")
        self.feature_dialog = SelectItemsDialog(self.features, "选择特征")
        self.settings = {"width": 8, "height": 6, "title": "", "xlabel": "", "ylabel": ""}

        self.init_ui()

    def init_ui(self):
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 文件加载
        self.load_button = QPushButton("加载文件")
        self.load_button.clicked.connect(self.load_file)
        main_layout.addWidget(self.load_button)

        # 通道选择
        self.channel_lineedit = QLineEdit()
        self.channel_button = QPushButton("选择通道")
        self.channel_button.clicked.connect(self.select_channels)
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(self.channel_button)
        channel_layout.addWidget(self.channel_lineedit)
        main_layout.addLayout(channel_layout)

        # 特征选择
        # self.feature_label = QLabel("选择特征:")
        # main_layout.addWidget(self.feature_label)
        self.feature_lineedit = QLineEdit()
        self.feature_button = QPushButton("选择特征")
        self.feature_button.clicked.connect(self.select_features)
        feature_layout = QHBoxLayout()
        feature_layout.addWidget(self.feature_button)
        feature_layout.addWidget(self.feature_lineedit)
        main_layout.addLayout(feature_layout)

        # 绘图类型选择
        self.plot_type_label = QLabel("选择绘图类型:")
        main_layout.addWidget(self.plot_type_label)
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["曲线图", "柱状图", "数据表格"])
        self.plot_type_combo.currentIndexChanged.connect(self.update_plot_area)
        main_layout.addWidget(self.plot_type_combo)

        # 绘图设置
        self.settings_button = QPushButton("设置绘图参数")
        self.settings_dialog = PlotSettingsDialog()
        self.settings_dialog.save_button.clicked.connect(self.save_settings)
        self.settings_button.clicked.connect(self.settings_dialog.show)
        main_layout.addWidget(self.settings_button)

        # 添加滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        main_layout.addWidget(self.scroll_area)

        # 绘图区域嵌套到滚动区域
        canvas_container = QWidget()
        self.canvas_layout = QVBoxLayout(canvas_container)
        self.canvas = PlotCanvas(self, width=8, height=6)
        self.canvas_layout.addWidget(self.canvas)
        self.scroll_area.setWidget(canvas_container)

        self.table = QTableWidget()

        # 绘图和保存按钮
        button_layout = QHBoxLayout()
        self.plot_button = QPushButton("绘制图像")
        self.plot_button.clicked.connect(self.plot)
        button_layout.addWidget(self.plot_button)

        self.save_button = QPushButton("保存图像")
        self.save_button.clicked.connect(self.save_plot)
        button_layout.addWidget(self.save_button)

        main_layout.addLayout(button_layout)

    def update_plot_area(self):
        plot_type = self.plot_type_combo.currentText()
        if plot_type == "数据表格":
            self.scroll_area.setWidget(self.table)
            self.scroll_area.setWidgetResizable(True)
            if self.df is not None:
                self.load_data_to_table()
        else:
            self.canvas = PlotCanvas(self, width=8, height=6)
            self.scroll_area.setWidget(self.canvas)
            self.scroll_area.setWidgetResizable(False)

    def load_data_to_table(self):
        if self.df is None:
            QMessageBox.warning(self, "警告", "请先加载数据文件！")
            return

        self.table.clear()
        self.table.setRowCount(len(self.df))
        self.table.setColumnCount(len(self.df.columns))
        self.table.setHorizontalHeaderLabels(self.df.columns)

        for i in range(len(self.df)):
            for j in range(len(self.df.columns)):
                item = QTableWidgetItem(str(self.df.iloc[i, j]))
                self.table.setItem(i, j, item)

    def save_table_to_dataframe(self):
        if self.table.isVisible():
            row_count = self.table.rowCount()
            col_count = self.table.columnCount()
            data = []
            for i in range(row_count):
                row = []
                for j in range(col_count):
                    item = self.table.item(i, j)
                    row.append(item.text() if item else "")
                data.append(row)
            self.df = pd.DataFrame(data, columns=self.df.columns)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "加载特征数据文件", "", "Excel Files (*.xlsx)")
        if not file_path:
            return

        try:
            self.df = pd.read_excel(file_path)
            self.channels = self.df.iloc[:, 0].unique()
            self.features = [col for col in self.df.columns[1:] if col != "Type"]  # 排除 "Type"
            print(self.features)

            self.channel_lineedit.clear()
            self.feature_lineedit.clear()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法加载文件: {str(e)}")

    def select_channels(self):
        dialog = SelectItemsDialog(self.channels, "选择通道")
        if dialog.exec_() == QDialog.Accepted:
            selected_channels = dialog.get_selected_items()
            self.channel_lineedit.setText(", ".join(selected_channels))

    def select_features(self):
        dialog = SelectItemsDialog(self.features, "选择特征")
        if dialog.exec_() == QDialog.Accepted:
            selected_features = dialog.get_selected_items()
            self.feature_lineedit.setText(", ".join(selected_features))

    def save_settings(self):
        self.settings = self.settings_dialog.get_settings()

    def plot(self):
        if self.df is None:
            QMessageBox.warning(self, "警告", "请先加载数据文件！")
            return

        selected_channels = self.channel_lineedit.text().split(", ")
        selected_features = self.feature_lineedit.text().split(", ")
        plot_type = self.plot_type_combo.currentText()

        if not selected_channels or not selected_features:
            QMessageBox.warning(self, "警告", "请选择通道和特征！")
            return

        settings = getattr(self, 'settings', {"width": 8, "height": 6, "title": "", "xlabel": "", "ylabel": ""})

        # 获取宽度和高度
        width = settings.get("width", 8)
        height = settings.get("height", 6)
        print(width, height)

        # self.clear_layout(self.canvas_layout)
        self.canvas = PlotCanvas(self, width=width, height=height)
        self.scroll_area.setWidget(self.canvas)
        # self.canvas_layout.addWidget(self.canvas)

        # 更新画布大小
        self.canvas.fig.set_size_inches(width, height, forward=True)

        # 绘制图像
        self.canvas.plot(self.df, selected_channels, selected_features, plot_type, settings)

    def save_plot(self):
        if self.plot_type_combo.currentText() == "数据表格":
            file_path, _ = QFileDialog.getSaveFileName(self, "保存表格", "", "Excel Files (*.xlsx);;All Files (*)")
            if not file_path:
                return
            try:
                self.save_table_to_dataframe()
                self.df.to_excel(file_path, index=False)
                QMessageBox.information(self, "成功", f"表格已保存到 {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存表格失败: {str(e)}")
        else:
            if not self.canvas.fig.axes:
                QMessageBox.warning(self, "警告", "请先绘制图像！")
                return

            file_path, _ = QFileDialog.getSaveFileName(self, "保存图像", "",
                                                       "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)")
            if not file_path:
                return

            dpi, ok = QInputDialog.getInt(self, "设置 DPI", "请输入 DPI 值（默认 100）:", value=300, min=50, max=600)
            if ok:
                try:
                    self.canvas.fig.savefig(file_path, dpi=dpi)
                    QMessageBox.information(self, "成功", f"图像已保存到 {file_path}，DPI: {dpi}")
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"保存图像失败: {str(e)}")

    def clear_layout(self, layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()  # Remove the widget from layout and delete it


class FeaturePlotWidget(QWidget):
    def __init__(self, data, channels, features):
        super().__init__()
        self.resize(800, 600)
        self.data = data
        self.channels = channels
        self.features = features

        self.channel_dialog = SelectItemsDialog(self.channels, "select channel")
        self.feature_dialog = SelectItemsDialog(self.features, "select feature")
        self.settings = {"width": 8, "height": 6, "title": "", "xlabel": "", "ylabel": ""}
        self.init_ui()

    def init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.channel_lineedit = QLineEdit()
        self.channel_button = BFPushButton("select channels")
        self.channel_button.clicked.connect(self.select_channels)
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(self.channel_button)
        channel_layout.addWidget(self.channel_lineedit)
        self.main_layout.addLayout(channel_layout)

        # 特征选择
        self.feature_lineedit = QLineEdit()
        self.feature_button = BFPushButton("select features")
        self.feature_button.clicked.connect(self.select_features)
        feature_layout = QHBoxLayout()
        feature_layout.addWidget(self.feature_button)
        feature_layout.addWidget(self.feature_lineedit)

        self.mid_layout = QHBoxLayout()
        self.main_layout.addLayout(self.mid_layout)
        self.main_layout.addLayout(feature_layout)

        # 绘图和保存按钮
        button_layout = QHBoxLayout()

        self.plot_button = BFPushButton("Plot")
        self.plot_button.setFixedWidth(120)
        self.plot_button.clicked.connect(self.plot)
        button_layout.addWidget(self.plot_button)

        self.save_button = BFPushButton("Save")
        self.save_button.setFixedWidth(120)
        self.save_button.clicked.connect(self.save_plot)
        button_layout.addWidget(self.save_button)

        button_layout.addStretch(1)

        self.settings_dialog = PlotSettingsDialog()
        self.settings_dialog.save_button.clicked.connect(self.save_settings)

        self.settings_button = BFPushButton("Settings")
        self.settings_button.setFixedWidth(120)
        self.settings_button.clicked.connect(self.settings_dialog.show)
        button_layout.addWidget(self.settings_button)

        self.main_layout.addLayout(button_layout)

        # 添加滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.main_layout.addWidget(self.scroll_area)

    def select_channels(self):
        dialog = SelectItemsDialog(self.channels, "Select Channels")
        if dialog.exec_() == QDialog.Accepted:
            selected_channels = dialog.get_selected_items()
            self.channel_lineedit.setText(", ".join(selected_channels))

    def select_features(self):
        dialog = SelectItemsDialog(self.features, "Select Features")
        if dialog.exec_() == QDialog.Accepted:
            selected_features = dialog.get_selected_items()
            self.feature_lineedit.setText(", ".join(selected_features))

    def save_settings(self):
        self.settings = self.settings_dialog.get_settings()

    def plot(self):
        pass

    def save_plot(self):
        pass

    def clear_layout(self, layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()  # Remove the widget from layout and delete it


class CurvePlotWidget(FeaturePlotWidget):
    def __init__(self, data, channels, features):
        super().__init__(data, channels, features)
        self.canvas = CurvePlotCanvas()
        self.scroll_area.setWidget(self.canvas)

    def plot(self):
        if self.data is None:
            QMessageBox.warning(self, "warning", "Please load the data file first!")
            return

        selected_channels = self.channel_lineedit.text().split(", ")
        selected_features = self.feature_lineedit.text().split(", ")

        if not selected_channels or not selected_features:
            QMessageBox.warning(self, "warning", "Please select channels and features!")
            return

        settings = getattr(self, 'settings', {"width": 8, "height": 6, "title": "", "xlabel": "", "ylabel": ""})
        width = settings.get("width", 8)
        height = settings.get("height", 6)

        self.canvas = CurvePlotCanvas(self, width=width, height=height)
        self.scroll_area.setWidget(self.canvas)

        self.canvas.fig.set_size_inches(width, height, forward=True)
        self.canvas.plot(self.data, selected_channels, selected_features, settings)

    def save_plot(self):
        if not self.canvas.fig.axes:
            QMessageBox.warning(self, "warning", "Please plot the figure first!")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Figure", "",
                                                   "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)")
        if not file_path:
            return

        dpi, ok = QInputDialog.getInt(self, "Setting dpi", "Please enter a dpi value (default 300) :", value=300,
                                      min=50, max=600)
        if ok:
            try:
                self.canvas.fig.savefig(file_path, dpi=dpi)
                QMessageBox.information(self, "success", f"The figure has been saved to {file_path}，dpi: {dpi}")
            except Exception as e:
                QMessageBox.critical(self, "error", f"Failed to save: {str(e)}")


class BarPlotWidget(FeaturePlotWidget):
    def __init__(self, data, channels, features):
        super().__init__(data, channels, features)
        self.canvas = BarPlotCanvas()
        self.scroll_area.setWidget(self.canvas)

    def plot(self):
        if self.data is None:
            QMessageBox.warning(self, "warning", "Please load the data file first!")
            return

        selected_channels = self.channel_lineedit.text().split(", ")
        selected_features = self.feature_lineedit.text().split(", ")

        if not selected_channels or not selected_features:
            QMessageBox.warning(self, "warning", "Please select channels and features!")
            return

        settings = getattr(self, 'settings', {"width": 8, "height": 6, "title": "", "xlabel": "", "ylabel": ""})
        width = settings.get("width", 8)
        height = settings.get("height", 6)

        self.canvas = BarPlotCanvas(self, width=width, height=height)
        self.scroll_area.setWidget(self.canvas)

        self.canvas.fig.set_size_inches(width, height, forward=True)
        self.canvas.plot(self.data, selected_channels, selected_features, settings)

    def save_plot(self):
        if not self.canvas.fig.axes:
            QMessageBox.warning(self, "warning", "Please plot the figure first!")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Figure", "",
                                                   "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)")
        if not file_path:
            return

        dpi, ok = QInputDialog.getInt(self, "Setting dpi", "Please enter a dpi value (default 300) :", value=300,
                                      min=50, max=600)
        if ok:
            try:
                self.canvas.fig.savefig(file_path, dpi=dpi)
                QMessageBox.information(self, "success", f"The figure has been saved to {file_path}，dpi: {dpi}")
            except Exception as e:
                QMessageBox.critical(self, "error", f"Failed to save: {str(e)}")


class TablePlotWidget(FeaturePlotWidget):
    def __init__(self, data, channels, features):
        super().__init__(data, channels, features)
        self.data = data
        self.canvas = TablePlotCanvas()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.canvas)
        self.feature_button.setVisible(False)
        self.feature_lineedit.setVisible(False)
        self.channel_button.setVisible(False)
        self.channel_lineedit.setVisible(False)
        self.settings_button.setVisible(False)

    def plot(self):
        data = feature_dict_to_dataframe(self.data)
        self.canvas.plot(data)

    def save_plot(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save table", "", "Excel Files (*.xlsx);;All Files (*)")
        if not file_path:
            return
        try:
            self.save_table_to_dataframe()
            self.data.to_excel(file_path, index=False)
            QMessageBox.information(self, "success", f"The table has been saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "error", f"Failed to save: {str(e)}")

    def save_table_to_dataframe(self):
        if self.canvas.isVisible():
            row_count = self.canvas.rowCount()
            col_count = self.canvas.columnCount()
            data = []
            for i in range(row_count):
                row = []
                for j in range(col_count):
                    item = self.canvas.item(i, j)
                    row.append(item.text() if item else "")
                data.append(row)
            self.data = pd.DataFrame(data, columns=self.data.columns)


class TopoMapPlotWidget(FeaturePlotWidget):
    def __init__(self, data, channels, features):
        super().__init__(data, channels, features)
        self.canvas = TopomapCanvas()
        self.scroll_area.setWidget(self.canvas)
        self.info = None

        self.info_select_button = BFPushButton('select info.json')
        self.info_select_button.setFixedWidth(120)
        self.info_select_lineedit = QLineEdit()
        self.mid_layout.addWidget(self.info_select_button)
        self.mid_layout.addWidget(self.info_select_lineedit)
        # 设置按钮点击事件
        self.info_select_button.clicked.connect(self.select_info_json)

        self.bottom_layout = QHBoxLayout()
        self.relative_checkbox = QCheckBox('Set Relative')
        self.show_sensor_checkbox = QCheckBox('Show Sensors')
        self.bottom_layout.addWidget(self.relative_checkbox)
        self.bottom_layout.addWidget(self.show_sensor_checkbox)
        self.bottom_layout.addStretch(1)
        self.main_layout.addLayout(self.bottom_layout)


    def select_info_json(self):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Info JSON File", "", "JSON Files (*.json)")

        if file_path:
            self.info = read_info(file_path)
            self.info_select_lineedit.setText(file_path)

    def plot(self):
        if self.data is None:
            QMessageBox.warning(self, "warning", "Please load the data file first!")
            return

        selected_channels = self.channel_lineedit.text().split(", ")
        selected_features = self.feature_lineedit.text().split(", ")

        if not selected_channels or not selected_features:
            QMessageBox.warning(self, "warning", "Please select channels and features!")
            return

        settings = getattr(self, 'settings', {"width": 8, "height": 6, "title": "", "xlabel": "", "ylabel": ""})
        width = settings.get("width", 8)
        height = settings.get("height", 6)

        self.canvas = TopomapCanvas(self, width=width, height=height)
        self.scroll_area.setWidget(self.canvas)

        is_relative = self.relative_checkbox.isChecked()
        is_show_sensor = self.show_sensor_checkbox.isChecked()

        if self.info:
            self.canvas.plot(self.data, self.info, selected_channels, selected_features, is_relative, is_show_sensor)
        else:
            QMessageBox.warning(self, "warning", "Please select a info.json!")
            return

    def save_plot(self):
        if not self.canvas.fig.axes:
            QMessageBox.warning(self, "warning", "Please plot the figure first!")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Figure", "",
                                                   "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)")
        if not file_path:
            return

        dpi, ok = QInputDialog.getInt(self, "Setting dpi", "Please enter a dpi value (default 300) :", value=300,
                                      min=50, max=600)
        if ok:
            try:
                self.canvas.fig.savefig(file_path, dpi=dpi)
                QMessageBox.information(self, "success", f"The figure has been saved to {file_path}，dpi: {dpi}")
            except Exception as e:
                QMessageBox.critical(self, "error", f"Failed to save: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # window = FeaturePlotWidget()
    # window.show()

    data = read_xlsx('C:\\Users\\28164\\Desktop\\test\\open_dataset\\MI_1_eeg_psd.xlsx')
    select_data = get_selected_data(data, ['F7', 'AFF5h', 'F3'], ['Delta', 'Beta'])
    print(select_data)
    sys.exit(app.exec())
