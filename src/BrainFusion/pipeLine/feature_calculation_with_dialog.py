# -*- coding: utf-8 -*-
# @Time    : 2024/6/18 14:49
# @Author  : XXX
# @Site    : 
# @File    : feature_calculation_with_dialog.py
# @Software: PyCharm 
# @Comment :
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from BrainFusion.pipeLine.pipeLine_with_dialog import RootMeanSquareDialog
from UI.ui_component import BFNavigation, BFTabNavigation, BFFileListPanel


# class FeatureCalculationDialog(QWidget):
#     def __init__(self):
#         super(FeatureCalculationDialog, self).__init__()
#         self.layout = QVBoxLayout(self)
#         self.time_feature_widget = BFTabNavigation()
#         self.navigation = BFNavigation()
#         self.navigation.add_button_and_page('Time', QWidget())
#         self.navigation.add_button_and_page('Frequency', QWidget())
#         self.navigation.add_button_and_page('Time-\nFrequency', QWidget())
#         self.navigation.add_button_and_page('Nonlinear', QWidget())
#         self.navigation.add_button_and_page('Network', QWidget())
#         self.navigation.add_button_and_page('Microstate', QWidget())
#
#         self.file_list_widget = BFFileListPanel()
#
#         self.layout.addWidget(self.file_list_widget)
#         self.layout.addWidget(self.navigation)

class FeatureCalculationDialog(QWidget):
    """
    Comprehensive feature calculation interface. Provides a tabbed interface for calculating different types of features from
    electrophysiological data. Supports:
    - Time domain features (e.g., RMS, amplitude statistics)
    - Frequency domain features (e.g., spectral power, frequency bands)
    - Time-frequency representations (e.g., wavelet transforms)
    - Nonlinear dynamics
    - Network connectivity features
    - Microstate analysis

    :param parent: Parent widget, defaults to None
    :type parent: QWidget, optional
    """

    def __init__(self, parent=None):
        """
        Initialize the feature calculation interface. Creates a file selection panel and tabbed interface containing:
        1. Time domain features
        2. Frequency domain features
        3. Time-frequency analysis
        4. Nonlinear dynamics
        5. Network features
        6. Microstate analysis

        The interface automatically maximizes on display.
        """
        super().__init__(parent)
        self.setWindowTitle('Feature Calculation Toolbox')
        self._initialize_ui()

    def _initialize_ui(self):
        """Configure the user interface components and layout"""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # File selection panel for input data
        self.file_selection_panel = BFFileListPanel()
        self.file_selection_panel.set_title("Select Data Files")
        main_layout.addWidget(self.file_selection_panel)

        # Create tabbed interface for feature types
        self.feature_tabs = BFTabNavigation()

        # Time domain features
        self.time_features_tab = RootMeanSquareDialog('rms')
        self.feature_tabs.add_tab('Time Domain', self.time_features_tab)

        # Frequency domain features
        self.frequency_features_tab = QWidget()
        self.feature_tabs.add_tab('Frequency Domain', self.frequency_features_tab)

        # Time-frequency analysis
        self.time_frequency_tab = QWidget()
        self.feature_tabs.add_tab('Time-Frequency Analysis', self.time_frequency_tab)

        # Nonlinear dynamics features
        self.nonlinear_features_tab = QWidget()
        self.feature_tabs.add_tab('Nonlinear Dynamics', self.nonlinear_features_tab)

        # Network connectivity features
        self.network_features_tab = QWidget()
        self.feature_tabs.add_tab('Network Connectivity', self.network_features_tab)

        # Microstate analysis
        self.microstate_analysis_tab = QWidget()
        self.feature_tabs.add_tab('Microstate Analysis', self.microstate_analysis_tab)

        # Add tab widget to layout
        main_layout.addWidget(self.feature_tabs)

        # Configure layout stretching
        main_layout.setStretch(0, 2)  # File panel gets 20% space
        main_layout.setStretch(1, 8)  # Feature tabs get 80% space

        # Maximize window
        self.showMaximized()

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    log_widget = FeatureCalculationDialog()
    # log_widget.show()
    sys.exit(app.exec_())
