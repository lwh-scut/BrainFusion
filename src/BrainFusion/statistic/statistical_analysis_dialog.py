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
    获取特定通道和特征的数据，若通道不存在，则仅返回特征的数据

    Args:
        feature_dict (dict): 包含所有通道和特征的字典
        channel_name (str, optional): 需要选择的通道名, 默认为 None
        feature_name (str, optional): 需要选择的特征名, 必须提供

    Returns:
        list: 对应特征的数据列表
    """
    # 如果 ch_names 存在并且指定了 channel_name
    if "ch_names" in feature_dict and channel_name:
        if channel_name not in feature_dict["ch_names"]:
            raise ValueError(f"Channel '{channel_name}' not found in the data.")
        channel_index = feature_dict["ch_names"].index(channel_name)
    else:
        # 如果没有指定 channel_name 或者 ch_names 不存在，则跳过通道筛选
        channel_index = None

    if feature_name not in feature_dict["feature"]:
        raise ValueError(f"Feature '{feature_name}' not found in the data.")

    feature_data = feature_dict["feature"][feature_name]

    # 如果指定了通道，则返回该通道在特定特征上的数据
    if channel_index is not None:
        return feature_data[channel_index]

    # 如果没有指定通道，则返回该特征的所有数据
    return feature_data


def convert_to_significance_dict(result):
    significance_dict = {}

    for entry in result:
        # 获取比较组的名称和 corrected_p_value
        group_comparison = entry['group_comparison']
        corrected_p_value = entry['corrected_p_value']

        # 仅当 corrected_p_value 有效时才添加到字典中
        if not pd.isna(corrected_p_value):
            # 提取组别名称
            group1, group2 = group_comparison.split(' vs ')
            significance_dict[(f"Group {group1}", f"Group {group2}")] = corrected_p_value

    return significance_dict


class GroupDialog(QDialog):
    """Dialog to create or edit a group (with name and folder)."""

    def __init__(self, parent=None, group_name="", folder_path=""):
        super(GroupDialog, self).__init__(parent)
        self.setWindowTitle("New/Edit Group")

        layout = QVBoxLayout()

        self.group_name_label = QLabel("Group Name:")
        self.group_name_edit = QLineEdit(group_name)
        layout.addWidget(self.group_name_label)
        layout.addWidget(self.group_name_edit)

        self.folder_label = QLabel("Data Folder:")
        self.folder_edit = QLineEdit(folder_path)
        self.folder_button = BFPushButton("Select Folder")
        self.folder_button.clicked.connect(self.select_folder)
        layout.addWidget(self.folder_label)
        layout.addWidget(self.folder_edit)
        layout.addWidget(self.folder_button)

        self.confirm_button = BFPushButton("Confirm")
        self.confirm_button.clicked.connect(self.accept)
        layout.addWidget(self.confirm_button)

        self.setLayout(layout)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.folder_edit.setText(folder_path)

    def get_group_data(self):
        return self.group_name_edit.text(), self.folder_edit.text()


class ChannelSelectionDialog(QDialog):
    """Dialog for selecting channels."""

    def __init__(self, available_channels, selected_channels, parent=None):
        super(ChannelSelectionDialog, self).__init__(parent)
        self.setWindowTitle("Select")
        self.setMinimumSize(300, 300)

        layout = QVBoxLayout()

        # Scrollable area to contain checkboxes for channels
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Create checkboxes for each channel
        self.channel_checkboxes = {}
        for channel in available_channels:
            checkbox = QCheckBox(channel)
            checkbox.setChecked(channel in selected_channels)
            scroll_layout.addWidget(checkbox)
            self.channel_checkboxes[channel] = checkbox

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        # Confirm button
        self.confirm_button = BFPushButton("Confirm")
        self.confirm_button.clicked.connect(self.accept)
        layout.addWidget(self.confirm_button)

        self.setLayout(layout)

    def get_selected_channels(self):
        """Return the list of selected channels."""
        return [channel for channel, checkbox in self.channel_checkboxes.items() if checkbox.isChecked()]


class VisualisationSettingsDialog(QDialog):
    def __init__(self, title="Default Title", y_label="Y Axis", x_label="X Axis",
                 legend=None, x_ticks=None, y_range=None, color="#ff0000", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Visualisation Settings")
        self.setGeometry(100, 100, 300, 300)
        layout = QFormLayout(self)

        # Set values from the current plot or defaults
        self.title_input = QLineEdit(title)
        self.y_label_input = QLineEdit(y_label)
        self.x_label_input = QLineEdit(x_label)
        self.legend_input = QLineEdit(", ".join(legend) if legend else "")
        self.x_ticks_input = QLineEdit(", ".join(x_ticks) if x_ticks else "")
        self.y_range_input = QLineEdit(f"{y_range[0]}, {y_range[1]}" if y_range else "0, 10")
        self.color_input = QLineEdit(color)  # Default color in hex or provided

        layout.addRow("Title:", self.title_input)
        layout.addRow("Y Label:", self.y_label_input)
        layout.addRow("X Label:", self.x_label_input)
        layout.addRow("Legend (comma-separated):", self.legend_input)
        layout.addRow("X Ticks (comma-separated):", self.x_ticks_input)
        layout.addRow("Y Range (min, max):", self.y_range_input)
        layout.addRow("Color (hex):", self.color_input)

        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.accept)
        layout.addRow(self.confirm_button)

        self.setLayout(layout)

    def get_settings(self):
        title = self.title_input.text()
        y_label = self.y_label_input.text()
        x_label = self.x_label_input.text()
        legend = self.legend_input.text().split(", ")
        x_ticks = list(map(str, self.x_ticks_input.text().split(", ")))
        y_range = list(map(float, self.y_range_input.text().split(",")))
        color = self.color_input.text()
        return title, y_label, x_label, legend, x_ticks, y_range, color


class MatplotlibWidget(QWidget):
    """Widget for displaying a Matplotlib plot."""

    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        # self.plot_default()
        self.plot_data()

    def plot_default(self):
        """Default boxplot."""
        data = [np.random.randn(100) for _ in range(5)]
        self.plot(data, title="Default Boxplot", y_label="Value", x_label="Group", x_ticks=["A", "B", "C", "D", "E"])

    def plot(self, data, title="", y_label="", x_label="", legend=None, x_ticks=None, y_range=None, color="blue"):
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
        # 定义EEG通道和fNIRS通道选择
        eeg_channel_select = [['FCC5h'], ['FCC3h'], ['FCC4h'], ['FCC6h'],
                              ['CCP5h'], ['CCP3h'], ['CCP4h'], ['CCP6h']]
        fnirs_channel_select = [['S8_D9', 'S8_D10', 'S7_D10', 'S7_D9'],
                                ['S8_D11', 'S10_D11', 'S10_D10', 'S8_D10'],
                                ['S12_D13', 'S12_D15', 'S11_D15', 'S11_D13'],
                                ['S12_D16', 'S14_D16', 'S14_D15', 'S12_D15'],
                                ['S7_D10', 'S9_D10', 'S9_D5', 'S7_D5'],
                                ['S10_D10', 'S10_D12', 'S9_D12', 'S9_D10'],
                                ['S11_D15', 'S13_D15', 'S13_D14', 'S11_D14'],
                                ['S14_D15', 'S14_D8', 'S13_D8', 'S13_D15']]

        # NVC结果文件路径
        nvc_output_folder = 'E:\\DATA\\公开数据集\\EEG-fNIRS\\TUBerlinBCI\\Analysis Folder\\NVC\\02'
        json_output_file = os.path.join(nvc_output_folder, 'nvc_results.json')

        # 加载NVC结果
        with open(json_output_file, 'r') as json_file:
            nvc_results = json.load(json_file)

        # 仅分析subject 12
        subject = 'subject 24'
        subject_data = nvc_results['data'].get(subject, [])

        # 初始化保存左右手运动想象的NVC值
        nvc_left = {eeg_ch[0]: [] for eeg_ch in eeg_channel_select}
        nvc_right = {eeg_ch[0]: [] for eeg_ch in eeg_channel_select}

        # 遍历被试的数据
        for epoch in subject_data:
            # 根据标签确定是左手还是右手
            label = nvc_results['Labels'][subject][subject_data.index(epoch)]

            # 遍历每个EEG通道
            for eeg_idx, eeg_ch_list in enumerate(eeg_channel_select):
                eeg_ch = eeg_ch_list[0]  # 当前EEG通道

                # 计算对应四个fNIRS通道的NVC均值
                fnirs_channels = [ch + ' hbo' for ch in fnirs_channel_select[eeg_idx]]
                nvc_values = []

                for result in epoch:
                    if result['EEG_Channel'] == eeg_ch and result['fNIRS_Channel'] in fnirs_channels:
                        # 取NVC的绝对值
                        nvc_values.append(abs(result['NVC_Value']))

                # 保存NVC值
                if nvc_values:
                    nvc_mean = np.mean(nvc_values)
                    if label == 'left':
                        nvc_left[eeg_ch].append(nvc_mean)
                    elif label == 'right':
                        nvc_right[eeg_ch].append(nvc_mean)

        eeg_channels = list(nvc_left.keys())
        self.ax.clear()

        # 绘制左右手的箱线图
        boxplot_data_left = [nvc_left[ch] for ch in eeg_channels]
        boxplot_data_right = [nvc_right[ch] for ch in eeg_channels]

        # 左手箱线图
        positions_left = np.arange(len(eeg_channels)) * 2.0
        self.ax.boxplot(boxplot_data_left, positions=positions_left, widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor="skyblue"), labels=eeg_channels)

        # 右手箱线图
        positions_right = positions_left + 0.8
        self.ax.boxplot(boxplot_data_right, positions=positions_right, widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor="salmon"))

        # 添加显著性标记
        significance_level = 0.05
        bracket_height = 0.05  # 括号的高度
        asterisk_offset = 0.01  # 星号相对于括号的偏移
        line_width = 1.5  # 横线宽度

        for i, ch in enumerate(eeg_channels):
            # 进行 t 检验，比较左手和右手的 NVC 值
            t_stat, p_value = ttest_ind(nvc_left[ch], nvc_right[ch], nan_policy='omit')

            # 如果 p 值小于显著性水平，则标记星号和括号
            if p_value < significance_level:
                max_value = max(max(nvc_left[ch]), max(nvc_right[ch]))  # 获取最大值用于放置星号和括号
                y_bracket = max_value + bracket_height  # 括号位置
                y_asterisk = y_bracket + asterisk_offset  # 星号位置

                # 在两个箱线图之间绘制星号
                self.ax.text(positions_left[i] + 0.4, y_asterisk, '*', ha='center', va='bottom', fontsize=14,
                             color='red')

                # 在星号下方绘制横线括号
                self.ax.plot([positions_left[i], positions_right[i]], [y_bracket, y_bracket], color='black',
                             lw=line_width)
                self.ax.plot([positions_left[i], positions_left[i]], [y_bracket, y_bracket - 0.02], color='black',
                             lw=line_width)
                self.ax.plot([positions_right[i], positions_right[i]], [y_bracket, y_bracket - 0.02], color='black',
                             lw=line_width)

        # 添加标签和图例
        self.ax.set_xlabel('EEG Channels', fontsize=12)
        self.ax.set_ylabel('Absolute NVC Value', fontsize=12)
        self.ax.set_title(f'Neurovascular Coupling (NVC) Boxplot for Left and Right Hand Motor Imagery',
                          fontsize=12)

        # 调整 X 轴标签位置
        self.ax.set_xticks(positions_left + 0.4)
        self.ax.set_xticklabels(eeg_channels)

        # 添加图例
        legend_elements = [plt.Line2D([0], [0], color="skyblue", lw=4, label='Left Hand'),
                           plt.Line2D([0], [0], color="salmon", lw=4, label='Right Hand'),
                           plt.Line2D([0], [0], marker='*', color='w', label='p<0.05',
                                      markerfacecolor='red', markersize=10)]

        self.ax.legend(handles=legend_elements, loc='upper right')

        self.canvas.draw()


class StatisticalAnalysisDialog(QMainWindow):
    def __init__(self):
        super().__init__()
        self.result = []
        self.group_select_features = {}
        self.data = None
        self.group_folder = {}
        self.group_files = {}
        self.is_valid = False
        # Window title and dimensions
        self.setWindowTitle("Statistical Analysis Dialog")
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
        main_layout = QHBoxLayout()

        # Create Groups widget
        self.create_groups_widget = QFrame()
        self.create_groups_widget.setFrameShape(QFrame.Box)
        self.create_groups_widget.setStyleSheet("background-color: white;")
        create_groups_layout = QVBoxLayout()

        # Top part: New Group and Import Groups buttons
        self.new_group_button = BFPushButton("New Group")
        self.import_groups_button = BFPushButton("Import Groups")
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.new_group_button)
        top_layout.addWidget(self.import_groups_button)

        # Middle part: Group ListWidget
        self.group_list_widget = QListWidget()

        # Bottom part: Groups Validation button
        self.groups_validation_button = BFPushButton("Groups Validation")

        self.create_groups_label = QLabel("Create Groups")
        self.create_groups_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")
        create_groups_layout.addWidget(self.create_groups_label)
        create_groups_layout.addLayout(top_layout)
        create_groups_layout.addWidget(self.group_list_widget)
        create_groups_layout.addWidget(self.groups_validation_button)
        self.create_groups_widget.setLayout(create_groups_layout)

        # Connect buttons to actions
        self.new_group_button.clicked.connect(self.add_new_group)
        self.groups_validation_button.clicked.connect(self.validate_groups)

        # Connect double click event for editing group
        self.group_list_widget.itemDoubleClicked.connect(self.edit_group)

        # Right side layout: contains two widgets (Statistical Analysis and Visualisation)
        right_widget = QWidget()
        right_layout = QVBoxLayout()

        stat_layout = QVBoxLayout()

        # Statistical Analysis Widget
        self.stat_analysis_widget = QFrame()
        self.stat_analysis_widget.setFrameShape(QFrame.Box)
        self.stat_analysis_widget.setStyleSheet("background-color: white;")
        stat_analysis_layout = QGridLayout()

        self.stat_analysis_label = QLabel("Statistical Analysis")
        self.stat_analysis_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")

        self.stat_analysis_text = QLabel('Statistical Analysis will test each group of data two-by-two according to the Statistical Method, '
                                         'while the Correction Method is only used if the number of groups is greater than 2. '
                                         'When the number of input features is greater than 2, a MANOVA is run by default. ')
        self.stat_analysis_text.setStyleSheet("font-family: 'Times New Roman'; font-size: 10pt; color: blue;")
        self.stat_analysis_text.setWordWrap(True)
        stat_layout.addWidget(self.stat_analysis_label)
        stat_layout.addWidget(self.stat_analysis_text)
        stat_layout.addLayout(stat_analysis_layout)
        stat_layout.addStretch(1)
        # 1st row: Channel selection
        self.channel_label = QLabel("Channel:")
        self.channel_combox = QComboBox()
        # self.channel_edit = QLineEdit()
        # self.channel_select_button = BFPushButton("Select")
        # self.channel_select_button.clicked.connect(self.select_channels)
        stat_analysis_layout.addWidget(self.channel_label, 0, 0)
        stat_analysis_layout.addWidget(self.channel_combox, 0, 1)
        # stat_analysis_layout.addWidget(self.channel_select_button, 0, 2)

        # 2nd row: Feature selection
        self.feature_label = QLabel("Feature:")
        self.feature_combox = QComboBox()
        # self.feature_lineedit = QLineEdit()
        # self.feature_select_button = BFPushButton("Select")
        # self.feature_select_button.clicked.connect(self.select_feature)
        stat_analysis_layout.addWidget(self.feature_label, 1, 0)
        stat_analysis_layout.addWidget(self.feature_combox, 1, 1)
        # stat_analysis_layout.addWidget(self.feature_select_button, 1, 2)

        # 3rd row: Statistical method selection
        self.stat_method_label = QLabel("Statistical Method:")
        self.stat_method_combobox = QComboBox()
        self.stat_method_combobox.addItems(["t-test", "t-test(paired)", "anova",
                      "mann-whitney U", "wilcoxon(paired)", "kruskal-wallis"])
        stat_analysis_layout.addWidget(self.stat_method_label, 2, 0)
        stat_analysis_layout.addWidget(self.stat_method_combobox, 2, 1)

        # 4th row: Correction method and enable checkbox
        self.correction_label = QLabel("Correction Method:")
        self.correction_combobox = QComboBox()
        self.correction_combobox.addItems(["bonferroni", "fdr_bh", "fdr_by", "holm-sidak", "sidak"])
        self.enable_checkbox = QCheckBox("Enable")
        self.enable_checkbox.setChecked(True)
        stat_analysis_layout.addWidget(self.correction_label, 3, 0)
        stat_analysis_layout.addWidget(self.correction_combobox, 3, 1)
        stat_analysis_layout.addWidget(self.enable_checkbox, 3, 2)

        # 5th row: Run button, Status label, and Export button
        self.run_button = BFPushButton("Run")
        self.run_button.setFixedWidth(120)
        self.status_label = QLabel("Status: Waiting")
        self.export_button = BFPushButton("Export")
        self.export_button.setFixedWidth(120)
        self.run_button.clicked.connect(self.run_analysis)
        self.export_button.clicked.connect(self.export_results)
        stat_analysis_layout.addWidget(self.run_button, 4, 0)
        stat_analysis_layout.addWidget(self.status_label, 4, 1)
        stat_analysis_layout.addWidget(self.export_button, 4, 2)

        stat_analysis_layout.setVerticalSpacing(10)
        self.stat_analysis_widget.setLayout(stat_layout)

        # Visualisation Widget
        self.visualisation_widget = QFrame()
        self.visualisation_widget.setFrameShape(QFrame.Box)
        self.visualisation_widget.setStyleSheet("background-color: white;")
        self.visualisation_layout = QVBoxLayout()
        self.scroll_widget = QScrollArea()

        self.visualisation_label = QLabel("Visualisation")
        self.visualisation_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")

        # Buttons: Save and Settings
        self.plot_type = QComboBox()
        self.plot_type.addItems(['scatter plot', 'bar plot', 'box plot', 'violin diagram'])
        self.plot_type.setFixedHeight(25)
        self.plot_button = BFPushButton("Plot")
        self.plot_button.clicked.connect(self.plot)
        self.save_button = BFPushButton("Save")
        self.save_button.setFixedWidth(100)
        self.settings_button = BFPushButton("Settings")
        self.settings_button.setFixedWidth(100)
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(QLabel('Select Plot Type: '))
        buttons_layout.addWidget(self.plot_type)
        buttons_layout.addWidget(self.plot_button)
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.settings_button)

        # Matplotlib plot
        self.plot_widget = ScatterPlotWindow()
        self.scroll_widget.setWidget(self.plot_widget)

        self.visualisation_layout.addWidget(self.visualisation_label)
        self.visualisation_layout.addLayout(buttons_layout)
        self.visualisation_layout.addWidget(self.scroll_widget)
        self.visualisation_widget.setLayout(self.visualisation_layout)

        # Add widgets to the right layout
        right_layout.addWidget(self.stat_analysis_widget)
        right_layout.addWidget(self.visualisation_widget)
        right_layout.setStretch(0, 3)
        right_layout.setStretch(1, 7)
        right_widget.setLayout(right_layout)

        # Add left and right widgets to main layout
        main_layout.addWidget(self.create_groups_widget)
        main_layout.addWidget(right_widget)
        # main_layout.setStretch()
        main_layout.setStretch(0, 25)
        main_layout.setStretch(1, 75)

        # Central widget for the main window
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Connect Settings button
        # self.settings_button.clicked.connect(self.open_visualisation_settings)

    def add_new_group(self):
        dialog = GroupDialog(self)
        if dialog.exec_():
            group_name, folder_path = dialog.get_group_data()
            self.group_list_widget.addItem(f"{group_name} - {folder_path}")
        self.is_valid = False

    def edit_group(self, item):
        group_data = item.text().split(" - ")
        dialog = GroupDialog(self, group_data[0], group_data[1])
        if dialog.exec_():
            new_group_name, new_folder_path = dialog.get_group_data()
            item.setText(f"{new_group_name} - {new_folder_path}")
        self.is_valid = False

    def validate_groups(self):
        groups = {}
        self.group_files.clear()
        self.group_select_features.clear()
        self.group_folder.clear()
        for i in range(self.group_list_widget.count()):
            group_item = self.group_list_widget.item(i).text()
            group_name, folder_path = group_item.split(" - ")
            self.group_files[group_name] = []
            for file_name in os.listdir(folder_path):
                if file_name.split('.')[1] == 'xlsx':
                    file_data = read_xlsx(os.path.join(folder_path, file_name))
                else:
                    file_data = None
                self.group_files[group_name].append(file_data)
        first_key, first_value = next(iter(self.group_files.items()))
        self.data = first_value[0]
        self.feature_combox.clear()
        self.feature_combox.addItems(self.data['feature'])
        self.status_label.setText("Status: Waiting")
        if 'ch_names' not in self.data.keys():
            self.channel_combox.setVisible(False)
            self.channel_label.setVisible(False)
        else:
            self.channel_combox.clear()
            self.channel_combox.addItems(self.data['ch_names'])
        self.is_valid = True
        QMessageBox.information(self, "Success", f"Data imported successfully")

    # def select_channels(self):
    #     if self.data:
    #         available_channels = self.data['ch_names']
    #         selected_channels = self.channel_edit.text().split(", ")
    #         dialog = ChannelSelectionDialog(available_channels, selected_channels, self)
    #         if dialog.exec_():
    #             selected_channels = dialog.get_selected_channels()
    #             self.channel_edit.setText(", ".join(selected_channels))
    #
    # def select_feature(self):
    #     if self.data:
    #         available_feature = self.data['feature'].keys()
    #         selected_feature = self.feature_lineedit.text().split(", ")
    #         dialog = ChannelSelectionDialog(available_feature, selected_feature, self)
    #         if dialog.exec_():
    #             selected_feature = dialog.get_selected_channels()
    #             self.feature_lineedit.setText(", ".join(selected_feature))

    def run_analysis(self):
        if self.is_valid:
            self.status_label.setText("Status: Running...")
            # Simulate analysis (replace with actual logic)
            groups = self.group_files.keys()
            for group_name in groups:
                self.group_select_features[group_name] = []
                for feature_dict in self.group_files[group_name]:
                    if 'ch_names' in feature_dict.keys():
                        channel = self.channel_combox.currentText()
                    else:
                        channel = None
                    feature = self.feature_combox.currentText()
                    select_feature = get_feature(feature_dict, channel, feature)
                    self.group_select_features[group_name].append(select_feature)
            print(self.group_select_features)
            self.status_label.setText("Status: Completed")

            method = self.stat_method_combobox.currentText()
            if method == 't-test(paired)':
                method = 't-test'
                is_paired = True
            elif method == 'wilcoxon(paired)':
                method = 'wilcoxon'
                is_paired = True
            else:
                is_paired = False
            self.result = calculate_significance(self.group_select_features, method=method, paired=is_paired)
            if self.enable_checkbox.isChecked():
                correct_method = self.correction_combobox.currentText()
                self.result = multiple_comparison_correction(self.result, correct_method)
            print(self.result)
        else:
            QMessageBox.information(self, "Data not verified", "Please use Groups Validation to check the data groups before Run.")

    def export_results(self):
        QMessageBox.information(self, "Export", "Results exported successfully!")

    def open_visualisation_settings(self):
        # 获取当前绘制的图的参数
        title = self.ax.get_title()
        y_label = self.ax.get_ylabel()
        x_label = self.ax.get_xlabel()
        legend = [text.get_text() for text in self.ax.get_legend().get_texts()] if self.ax.get_legend() else []
        x_ticks = [tick.get_text() for tick in self.ax.get_xticklabels()]
        y_range = self.ax.get_ylim()
        color = self.ax.patches[0].get_facecolor() if self.ax.patches else "#ff0000"  # 读取第一个图形的颜色

        # 打开设置窗口并将参数传递进去
        settings_dialog = VisualisationSettingsDialog(title, y_label, x_label, legend, x_ticks, y_range, color)

        if settings_dialog.exec_() == QDialog.Accepted:
            # 获取更新的设置
            new_title, new_y_label, new_x_label, new_legend, new_x_ticks, new_y_range, new_color = settings_dialog.get_settings()

            # 更新图形
            self.ax.set_title(new_title)
            self.ax.set_ylabel(new_y_label)
            self.ax.set_xlabel(new_x_label)
            self.ax.set_xticklabels(new_x_ticks)
            self.ax.set_ylim(new_y_range)

            for patch, color in zip(self.ax.patches, new_color):
                patch.set_facecolor(color)

            self.canvas.draw()

    def plot(self):
        if self.group_select_features:
            plot_type = self.plot_type.currentText()
            if plot_type == 'scatter plot':
                print(plot_type)
                self.plot_widget = ScatterPlotWindow()
                self.scroll_widget.setWidget(self.plot_widget)
                self.plot_widget.plot_scatter(self.group_select_features)
            elif plot_type == 'bar plot':
                self.plot_widget = DensityHistogramWindow()
                self.scroll_widget.setWidget(self.plot_widget)
                self.plot_widget.plot_density_histogram(self.group_select_features)
            elif plot_type == 'box plot':
                self.plot_widget = SignificanceBoxPlotWindow()
                self.scroll_widget.setWidget(self.plot_widget)
                significance_dict = convert_to_significance_dict(self.result)
                self.plot_widget.plot_boxplot_with_significance(self.group_select_features, significance_dict)
            elif plot_type == 'violin diagram':
                significance_dict = convert_to_significance_dict(self.result)
                self.plot_widget = SignificanceViolinPlotWindow()
                self.scroll_widget.setWidget(self.plot_widget)
                self.plot_widget.plot_violin_with_significance(self.group_select_features, significance_dict)
        else:
            QMessageBox.information(self, "Error", "Please import the data groups and Run before plotting.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StatisticalAnalysisDialog()
    window.show()
    sys.exit(app.exec_())
