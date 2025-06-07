# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 10:34
# @Author  : XXX
# @Site    : 
# @File    : viewer.py
# @Software: PyCharm 
# @Comment :
import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QLineEdit, QPushButton, QDialog, QVBoxLayout, QFrame, \
    QFileDialog, QListWidget, QListWidgetItem, QGroupBox, QTabWidget, QLabel, QAction, QMenu

from BrainFusion.io.File_IO import read_file_by_qt
from BrainFusion.viewer.viewer_curve import RawCurvePlotDialog, plot_raw_by_file, fNIRSPlotDialog
from BrainFusion.viewer.viewer_plot import BarPlotWidget, CurvePlotWidget, TablePlotWidget, TopoMapPlotWidget
from BrainFusion.viewer.viewer_test import TestBoxPlot, TestMLPlot, TestEEGSensorPlot, TestEEGPlot
from BrainFusion.viewer.viewer_time_frequency import TimeFrequencyViewer
from BrainFusion.viewer.viewer_topomap import TopomapViewer
from UI.ui_component import BFPushButton


class BrainFusionViewer(QWidget):
    """Main visualization application for BrainFusion data."""

    def __init__(self):
        """
        Initialize visualization application.
        """
        super(BrainFusionViewer, self).__init__()
        # Create import button and file display
        self.setWindowTitle("BrainFusion Viewer")
        self.setGeometry(800, 600, 1200, 600)

        # Define visualization type mappings
        self.curve_type = ['eeg', 'eeg_preprocess', 'fnirs', 'fnirs_preprocess', 'emg', 'emg_preprocess', 'ecg',
                           'ecg_preprocess']
        self.topomap_type = ['eeg_psd', 'eeg_microstate']
        self.time_frequency_type = ['stft']

        # Create import button
        self.import_button = BFPushButton("Select Folder")
        self.import_button.setFixedWidth(150)

        # Create file display field
        self.file_name_lineedit = QLineEdit()
        self.file_name_lineedit.setFixedWidth(150)
        self.file_name_lineedit.setReadOnly(True)  # Read-only to prevent manual editing

        # Create import group box
        self.groupbox_import = QGroupBox("Data Files")
        import_layout = QVBoxLayout(self.groupbox_import)
        self.import_button.clicked.connect(self.open_folder)

        # Add widgets to top layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.import_button)
        top_layout.addWidget(self.file_name_lineedit)

        # Create file list widget
        self.listWidget = QListWidget(self)
        self.listWidget.setFixedWidth(300)
        # Enable context menu
        self.listWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.listWidget.customContextMenuRequested.connect(self.show_context_menu)
        self.listWidget.doubleClicked.connect(self.list_item_double_clicked)

        # Build layout structure
        import_layout.addLayout(top_layout)
        import_layout.addWidget(self.listWidget)
        self.groupbox_import.setFixedWidth(325)
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.groupbox_import)

        # Create drawing workspace
        self.drawing_frame = QFrame()
        self.drawing_frame.setFrameShape(QFrame.StyledPanel)  # Set frame style
        self.drawing_frame.setStyleSheet("background-color: white;")
        self.drawing_layout = QVBoxLayout(self.drawing_frame)

        # Create visualization label
        self.bnt_label = QLabel("Visualisation")
        self.bnt_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")
        self.bnt_layout = QHBoxLayout()
        self.bnt_layout.addWidget(self.bnt_label)
        self.bnt_layout.addStretch(1)
        self.drawing_layout.addLayout(self.bnt_layout)

        # Create tabbed visualization area
        self.drawing_widget = QTabWidget()
        self.drawing_widget.setTabsClosable(True)
        self.drawing_widget.tabCloseRequested.connect(self.close_tab)
        self.drawing_layout.addWidget(self.drawing_widget)

        # Build main layout
        layout = QHBoxLayout(self)
        layout.addLayout(left_layout)
        layout.addWidget(self.drawing_frame)  # Add drawing workspace

    def show_context_menu(self, position):
        """
        Display context menu with visualization options for selected file.

        :param position: Click position in the list widget
        :type position: QPoint
        """
        # Get selected item
        item = self.listWidget.itemAt(position)
        if not item:
            return

        # Load file data
        file_path = item.data(32)  # Retrieve stored file path
        data, file_path = read_file_by_qt(self, [file_path])
        name = item.text()

        # Create visualization menu
        menu = QMenu(self)

        # Add visualization actions
        curve_action = QAction("Line Chart", self)
        bar_action = QAction("Bar Chart", self)
        topo_action = QAction("Topographic Map", self)
        table_action = QAction("Data Table", self)

        # Connect actions to handlers
        curve_action.triggered.connect(lambda: self.plot_feature_curve(data, name + '_curve'))
        bar_action.triggered.connect(lambda: self.plot_bar(data, name + '_bar'))
        topo_action.triggered.connect(lambda: self.plot_topomap(data, name + '_topo'))
        table_action.triggered.connect(lambda: self.plot_table(data, name + '_table'))

        # Add actions to menu
        menu.addAction(curve_action)
        menu.addAction(bar_action)
        menu.addAction(topo_action)
        menu.addAction(table_action)

        # Display menu at position
        menu.exec_(self.listWidget.viewport().mapToGlobal(position))

    def set_plot_type(self, item, plot_type):
        """
        Set visualization type for list item.

        :param item: List widget item
        :type item: QListWidgetItem
        :param plot_type: Visualization type to assign
        :type plot_type: str
        """
        item.setData(Qt.UserRole, plot_type)
        item.setText(f"{item.text()} - {plot_type}")

    def plot_bar(self, data, name):
        """
        Create bar plot visualization.

        :param data: Input data for visualization
        :type data: dict
        :param name: Tab name for visualization
        :type name: str
        """
        bar_widget = BarPlotWidget(data, data['ch_names'], data['feature'].keys())
        self.drawing_widget.addTab(bar_widget, name)
        self.drawing_widget.setCurrentWidget(bar_widget)

    def plot_feature_curve(self, data, name):
        """
        Create feature curve visualization.

        :param data: Input data for visualization
        :type data: dict
        :param name: Tab name for visualization
        :type name: str
        """
        curve_widget = CurvePlotWidget(data, data['ch_names'], data['feature'].keys())
        self.drawing_widget.addTab(curve_widget, name)
        self.drawing_widget.setCurrentWidget(curve_widget)

    def plot_topomap(self, data, name):
        """
        Create topographic map visualization.

        :param data: Input data for visualization
        :type data: dict
        :param name: Tab name for visualization
        :type name: str
        """
        topomap_widget = TopoMapPlotWidget(data, data['ch_names'], data['feature'].keys())
        self.drawing_widget.addTab(topomap_widget, name)
        self.drawing_widget.setCurrentWidget(topomap_widget)

    def plot_table(self, data, name):
        """
        Create data table visualization.

        :param data: Input data for visualization
        :type data: dict
        :param name: Tab name for visualization
        :type name: str
        """
        table_widget = TablePlotWidget(data, data['ch_names'], data['feature'].keys())
        self.drawing_widget.addTab(table_widget, name)
        self.drawing_widget.setCurrentWidget(table_widget)

    def plot_raw_by_file(self, path=None):
        """
        Create raw data curve visualization.

        :param path: File path for visualization
        :type path: str
        """

        def trans_data(data):
            """Transform fNIRS data structure for visualization."""
            if data['type'] == 'fnirs_preprocessed':
                result = data['data'][0]
                result.extend(data['data'][1])
                channel = data['ch_names'][0]
                channel.extend(data['ch_names'][1])
                data['data'] = result
                data['ch_names'] = channel
            return data

        # Load and plot data
        data, file_path = read_file_by_qt(self, path)
        if data:
            data = trans_data(data)
            self.drawing_widget = RawCurvePlotDialog(data=data, filePath=file_path[0], parent=self)
            self.drawing_widget.plot_data(self.drawing_widget.current_page)

    def open_folder(self):
        """Open folder selection dialog."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory", "")
        if dir_path:
            self.list_files(dir_path)
            self.file_name_lineedit.setText(dir_path)

    def list_files(self, dir_path):
        """
        List supported files in directory.

        :param dir_path: Directory path to scan
        :type dir_path: str
        """
        self.listWidget.clear()  # Clear existing items
        # Add supported file types
        for filename in os.listdir(dir_path):
            if filename.endswith(('.edf', '.bdf', '.mat', '.nirs', '.ecg', '.xlsx', '.json')):
                item = QListWidgetItem(filename)
                item.setData(32, os.path.join(dir_path, filename))  # Store full path
                self.listWidget.addItem(item)

    def list_item_double_clicked(self, index):
        """
        Handle double-click on file item.

        :param index: Index of double-clicked item
        :type index: QModelIndex
        """
        # Get file data
        item = self.listWidget.item(index.row())
        file_path = item.data(32)  # Retrieve stored file path
        name = item.text()

        # Check for existing tab
        for i in range(self.drawing_widget.count()):
            if self.drawing_widget.tabText(i) == name:
                self.drawing_widget.setCurrentIndex(i)
                return

        # Create visualization
        plot_widget = self.plot_by_file_type(path=[file_path])
        if plot_widget:
            self.drawing_widget.addTab(plot_widget, name)
            self.drawing_widget.setCurrentWidget(plot_widget)

    def clearLayout(self):
        """Clear all widgets from drawing layout."""
        while self.drawing_layout.count():
            item = self.drawing_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def plot_by_file_type(self, path):
        """
        Create visualization based on file type.

        :param path: File path for visualization
        :type path: list[str]
        :return: Visualization widget for data
        :rtype: QWidget
        """

        def trans_data(data):
            """Transform fNIRS data structure for visualization."""
            if data['type'] == 'fnirs_preprocessed':
                result = data['data'][0]
                result.extend(data['data'][1])
                channel = data['ch_names'][0]
                channel.extend(data['ch_names'][1])
                data['data'] = result
                data['ch_names'] = channel
            return data

        drawing_widget = None
        file_name = os.path.basename(path[0])

        # Handle test data files
        if 'subject_01_MI_statistic_result' in file_name:
            drawing_widget = TestBoxPlot()
        elif 'subject_01_MI_ml_result' in file_name:
            drawing_widget = TestMLPlot()
        elif 'subject_01_MI_eeg_raw' in file_name:
            drawing_widget = TestEEGPlot()
        else:
            # Load and process data files
            data, file_path = read_file_by_qt(self, path)
            if data:
                # Select visualization based on data type
                if data['type'] in self.curve_type:
                    if data['type'] in ['fnirs_preprocessed', 'fnirs']:
                        data = trans_data(data)
                        drawing_widget = fNIRSPlotDialog(data=data, filePath=file_path[0], parent=self)
                    else:
                        drawing_widget = RawCurvePlotDialog(data=data, filePath=file_path[0], parent=self)
                    drawing_widget.plot_data(drawing_widget.current_page)
                elif data['type'] in self.topomap_type:
                    drawing_widget = TopomapViewer(data=data, parent=self)
                elif data['type'] in self.time_frequency_type:
                    drawing_widget = TimeFrequencyViewer(data=data, parent=self)

        return drawing_widget

    def close_tab(self, index):
        """
        Close visualization tab.

        :param index: Tab index to close
        :type index: int
        """
        self.drawing_widget.removeTab(index)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = BrainFusionViewer()
    main_window.show()
    sys.exit(app.exec_())