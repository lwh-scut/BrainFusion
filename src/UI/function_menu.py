# -*- coding: utf-8 -*-
# @Time    : 2024/5/17 8:46
# @Author  : XXX
# @Site    : 
# @File    : menu.py
# @Software: PyCharm 
# @Comment :
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QSizePolicy, QTreeWidget, \
    QTreeWidgetItem, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from BrainFusion.pipeLine.pipeLine_with_dialog import EEGPreprocessingConfigWindow, fNIRSPreprocessingConfigWindow, \
    RootMeanSquareDialog, VarianceDialog, MeanAbsoluteValueDialog, ZeroCrossingDialog, HjorthParametersDialog, \
    EEGPowerSpectralDensityDialog, AperiodicParametersDialog, SampleEntropyDialog, MultiscaleEntropyDialog, \
    ShortTimeFourierTransformDialog, WaveletTransformDialog, ContinuousWaveletTransformDialog, \
    WaveletPacketEnergyDialog, LocalNetworkDialog, GlobalNetworkDialog, EEGMicrostateDialog, \
    CreateSegments, EMGPreprocessingConfigWindow, ECGPreprocessingConfigWindow, EEGPreprocessingByRawConfigWindow
from BrainFusion.utils.BIDS import BIDSConverter


class PreprocessMenu(QWidget):
    """Menu for signal preprocessing options."""

    def __init__(self):
        """Initialize preprocessing menu."""
        super().__init__()
        self.initUI()

    def initUI(self):
        """
        Initialize user interface components.
        """
        # Create vertical layout
        vbox = QVBoxLayout()

        # Add logo image
        self.label = QLabel(self)
        pixmap = QPixmap('resources/logo/brain_fusion(1).png')  # Update with correct path
        if pixmap.isNull():
            print("Failed to load image!")
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)  # Scale image to fit label
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Enable label expansion
        self.label.setAlignment(Qt.AlignCenter)  # Center content
        vbox.addWidget(self.label)
        vbox.addStretch(1)

        # Initialize preprocessing dialogs
        self.eeg_dialog = EEGPreprocessingByRawConfigWindow(feature_name='eeg_preprocessed')
        self.fnirs_dialog = fNIRSPreprocessingConfigWindow(feature_name='fnirs_preprocessed')
        self.emg_dialog = EMGPreprocessingConfigWindow(feature_name='emg_preprocessed')
        self.ecg_dialog = ECGPreprocessingConfigWindow(feature_name='ecg_preprocessed')

        # Create processing option buttons
        self.bnt_eeg = QPushButton("EEG Preprocess Pipeline")
        self.bnt_emg = QPushButton("EMG Preprocess Pipeline")
        self.bnt_ecg = QPushButton("ECG Preprocess Pipeline")
        self.bnt_fnirs = QPushButton("fNIRS Preprocess Pipeline")
        self.bnt_other = QPushButton("Custom Preprocess Pipeline")

        # Connect button signals
        self.bnt_eeg.clicked.connect(self.open_eeg_preprocess_dialog)
        self.bnt_fnirs.clicked.connect(self.open_fnirs_preprocess_dialog)
        self.bnt_emg.clicked.connect(self.open_emg_preprocess_dialog)
        self.bnt_ecg.clicked.connect(self.open_ecg_preprocess_dialog)

        # Add buttons to layout
        vbox.addWidget(self.bnt_eeg)
        vbox.addWidget(self.bnt_emg)
        vbox.addWidget(self.bnt_ecg)
        vbox.addWidget(self.bnt_fnirs)
        vbox.addWidget(self.bnt_other)
        vbox.addStretch(1)

        # Set layout
        self.setLayout(vbox)

        # Configure window
        self.setWindowTitle('Signal Preprocessing Menu')
        self.center_on_screen()

    def open_eeg_preprocess_dialog(self):
        """Display EEG preprocessing configuration dialog."""
        self.eeg_dialog.show()

    def open_fnirs_preprocess_dialog(self):
        """Display fNIRS preprocessing configuration dialog."""
        self.fnirs_dialog.show()

    def open_emg_preprocess_dialog(self):
        """Display EMG preprocessing configuration dialog."""
        self.emg_dialog.show()

    def open_ecg_preprocess_dialog(self):
        """Display ECG preprocessing configuration dialog."""
        self.ecg_dialog.show()

    def center_on_screen(self):
        """
        Center window on screen.
        """
        screen_geometry = QApplication.desktop().screenGeometry()
        screen_center = screen_geometry.center()
        self_geometry = self.geometry()
        self_geometry.moveCenter(screen_center)
        self.setGeometry(self_geometry.x(), self_geometry.y(), 300, 400)


class UtilsMenu(QWidget):
    """Menu for utility functions and tools."""

    def __init__(self):
        """Initialize utilities menu."""
        super().__init__()
        self.initUI()

    def initUI(self):
        """
        Initialize user interface components.
        """
        # Create vertical layout
        vbox = QVBoxLayout()

        # Add logo image
        self.label = QLabel(self)
        pixmap = QPixmap('resources/logo/brain_fusion(1).png')  # Update with correct path
        if pixmap.isNull():
            print("Failed to load image!")
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)  # Scale image to fit label
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Enable label expansion
        self.label.setAlignment(Qt.AlignCenter)  # Center content
        vbox.addWidget(self.label)
        vbox.addStretch(1)

        # Create utility function buttons
        self.bnt_bids = QPushButton("BIDS Converter")
        self.bnt_epoch = QPushButton("Epoch Creator")
        self.bnt_multi_epoch = QPushButton("Multi-Signals Epoch Creator")

        # Connect button signals
        self.bnt_bids.clicked.connect(self.open_bids_converter_dialog)
        self.bnt_epoch.clicked.connect(self.open_epochs_creator_dialog)

        # Add buttons to layout
        vbox.addWidget(self.bnt_bids)
        vbox.addWidget(self.bnt_epoch)
        vbox.addWidget(self.bnt_multi_epoch)
        vbox.addStretch(1)

        # Set layout
        self.setLayout(vbox)

        # Configure window
        self.setWindowTitle('Utilities Menu')
        self.center_on_screen()

    def open_bids_converter_dialog(self):
        """Display BIDS format conversion tool."""
        self.bids_converter_dialog = BIDSConverter()
        self.bids_converter_dialog.show()

    def open_epochs_creator_dialog(self):
        """Display epoch creation tool."""
        self.epochs_dialog = CreateSegments('epochs')
        self.epochs_dialog.show()

    def open_multi_signals_epochs_creator_dialog(self):
        """Reserved for future multi-signal epoch creation."""
        pass

    def center_on_screen(self):
        """
        Center window on screen.
        """
        screen_geometry = QApplication.desktop().screenGeometry()
        screen_center = screen_geometry.center()
        self_geometry = self.geometry()
        self_geometry.moveCenter(screen_center)
        self.setGeometry(self_geometry.x(), self_geometry.y(), 300, 400)


class FeatureMenu(QWidget):
    """Menu for feature calculation options."""

    def __init__(self):
        """Initialize feature calculation menu."""
        super().__init__()
        self.initUI()

    def initUI(self):
        """
        Initialize user interface components.
        """
        # Create vertical layout
        vbox = QVBoxLayout()

        # Add logo image
        self.label = QLabel(self)
        pixmap = QPixmap('resources/logo/brain_fusion(1).png')  # Update with correct path
        if pixmap.isNull():
            print("Failed to load image!")
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)  # Scale image to fit label
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Enable label expansion
        self.label.setAlignment(Qt.AlignCenter)  # Center content
        vbox.addWidget(self.label)
        vbox.addStretch(1)

        # Create feature category buttons
        self.bnt_time = QPushButton("Time Domain")
        self.bnt_frequency = QPushButton("Frequency Domain")
        self.bnt_time_frequency = QPushButton("Time-Frequency")
        self.bnt_nonlinear = QPushButton("Nonlinear Analysis")
        self.bnt_network = QPushButton("Network Analysis")
        self.bnt_microstate = QPushButton("Microstate")

        # Add buttons to layout
        vbox.addWidget(self.bnt_time)
        vbox.addWidget(self.bnt_frequency)
        vbox.addWidget(self.bnt_time_frequency)
        vbox.addWidget(self.bnt_nonlinear)
        vbox.addWidget(self.bnt_network)
        vbox.addWidget(self.bnt_microstate)
        vbox.addStretch(1)

        # Set layout
        self.setLayout(vbox)

        # Configure window
        self.setGeometry(100, 100, 300, 400)
        self.setWindowTitle('Feature Calculation Menu')
        self.center_on_screen()

    def center_on_screen(self):
        """
        Center window on screen.
        """
        screen_geometry = QApplication.desktop().screenGeometry()
        screen_center = screen_geometry.center()
        self_geometry = self.geometry()
        self_geometry.moveCenter(screen_center)
        self.setGeometry(self_geometry.x(), self_geometry.y(), 300, 400)


class NewFeatureMenu(QWidget):
    """Feature selection menu with collapsible tree view."""

    def __init__(self):
        """Initialize feature selection menu."""
        super().__init__()
        self.initUI()

    def initUI(self):
        """
        Initialize user interface components.
        """
        # Create QTreeWidget for feature categories
        self.tree = QTreeWidget(self)
        # Configure widget styling
        self.tree.setStyleSheet("""
            QTreeWidget {
                background-color: #ffffff;
                font-size: 18px;
                border: none;
                outline: 0;
                font-weight: bold;
            }
            QTreeWidget::item {
                padding: 5px;
                border-bottom: 1px solid #dcdcdc;
            }
            QTreeWidget::item:has-children {
                background-color: #0288D1;
                border: 1px solid #000000;
                font-weight: bold;
                color: white;
            }
            QTreeWidget::item:selected {
                background-color: #E1F5FE;
                color: black;
            }
            QTreeWidget::item:hover {
                background-color: #E1F5FE;
                color: black;
                font-weight: bold;
            }
            QHeaderView::section {
                font-size: 20px;
                background-color: #f0f0f0;
                padding: 5px;
                border: none;
                font-weight: bold;
            }
        """)
        self.tree.setHeaderLabels(['Feature Selection'])

        # Create feature category branches
        time_domain_menu = QTreeWidgetItem(self.tree, ['Time Domain'])
        time_domain_menu.setFlags(time_domain_menu.flags() & ~Qt.ItemIsSelectable)
        time_domain_menu.addChild(QTreeWidgetItem(['root mean square']))
        time_domain_menu.addChild(QTreeWidgetItem(['variance']))
        time_domain_menu.addChild(QTreeWidgetItem(['mean absolute value']))
        time_domain_menu.addChild(QTreeWidgetItem(['zero crossing']))
        time_domain_menu.addChild(QTreeWidgetItem(['hjorth']))

        frequency_domain_menu = QTreeWidgetItem(self.tree, ['Frequency Domain'])
        frequency_domain_menu.setFlags(frequency_domain_menu.flags() & ~Qt.ItemIsSelectable)
        frequency_domain_menu.addChild(QTreeWidgetItem(['power spectral density (eeg)']))
        frequency_domain_menu.addChild(QTreeWidgetItem(['power spectral density']))
        frequency_domain_menu.addChild(QTreeWidgetItem(['aperiodic parameters (eeg)']))

        nonlinear_domain_menu = QTreeWidgetItem(self.tree, ['Nonlinear Domain'])
        nonlinear_domain_menu.setFlags(nonlinear_domain_menu.flags() & ~Qt.ItemIsSelectable)
        nonlinear_domain_menu.addChild(QTreeWidgetItem(['sample entropy']))
        nonlinear_domain_menu.addChild(QTreeWidgetItem(['multiscale entropy']))

        time_frequency_menu = QTreeWidgetItem(self.tree, ['Time-Frequency'])
        time_frequency_menu.setFlags(time_frequency_menu.flags() & ~Qt.ItemIsSelectable)
        time_frequency_menu.addChild(QTreeWidgetItem(['short-time Fourier transform']))
        time_frequency_menu.addChild(QTreeWidgetItem(['wavelet transform']))
        time_frequency_menu.addChild(QTreeWidgetItem(['continuous wavelet transform']))
        time_frequency_menu.addChild(QTreeWidgetItem(['wavelet packet energy']))

        network_menu = QTreeWidgetItem(self.tree, ['Network Analysis'])
        network_menu.setFlags(network_menu.flags() & ~Qt.ItemIsSelectable)
        network_menu.addChild(QTreeWidgetItem(['local network']))
        network_menu.addChild(QTreeWidgetItem(['global network']))

        microstate_menu = QTreeWidgetItem(self.tree, ['Microstate'])
        microstate_menu.setFlags(microstate_menu.flags() & ~Qt.ItemIsSelectable)
        microstate_menu.addChild(QTreeWidgetItem(['microstate (eeg)']))

        # Expand all category branches
        self.tree.expandAll()

        # Connect click events to handler
        self.tree.itemClicked.connect(self.onItemClicked)

        # Create layout and add widget
        layout = QVBoxLayout()
        layout.addWidget(self.tree)
        self.setLayout(layout)

        # Initialize feature dialogs
        self.feature_dialog_init()

        # Configure window
        self.setWindowTitle('Feature Selection Tree')
        self.center_on_screen()
        self.resize(600, 800)

    def feature_dialog_init(self):
        """Initialize dialog windows for feature configuration."""
        self.rms_dialog = RootMeanSquareDialog('rms')
        self.variance_dialog = VarianceDialog('var')
        self.mav_dialog = MeanAbsoluteValueDialog('mav')
        self.zc_dialog = ZeroCrossingDialog('zc')
        self.hjorth_dialog = HjorthParametersDialog('hjorth')

        self.psd_eeg_dialog = EEGPowerSpectralDensityDialog('eeg_psd')
        self.psd_dialog = QWidget()
        self.aperiodic_dialog = AperiodicParametersDialog('aperiodic')

        self.sample_entropy_dialog = SampleEntropyDialog('sample_entropy')
        self.multiscale_entropy_dialog = MultiscaleEntropyDialog('multiscale_entropy')

        self.stft_dialog = ShortTimeFourierTransformDialog('stft')
        self.wavelet_transform_dialog = WaveletTransformDialog('wavelet_transform')
        self.cwt_dialog = ContinuousWaveletTransformDialog('cwt')
        self.wavelet_packet_energy_dialog = WaveletPacketEnergyDialog('wavelet_packet_energy')

        self.local_network_dialog = LocalNetworkDialog('local_network')
        self.global_network_dialog = GlobalNetworkDialog('global_network')

        self.microstate = EEGMicrostateDialog('microstate')

    def onItemClicked(self, item, column):
        """
        Handle feature item selection.

        :param item: Selected feature item
        :type item: QTreeWidgetItem
        :param column: Column index of the selection
        :type column: int
        """
        # Get name of selected feature
        algorithm = item.text(0)

        # Open corresponding configuration dialog
        if algorithm == 'root mean square':
            self.rms_dialog.show()
        elif algorithm == 'variance':
            self.variance_dialog.show()
        elif algorithm == 'mean absolute value':
            self.mav_dialog.show()
        elif algorithm == 'zero crossing':
            self.zc_dialog.show()
        elif algorithm == 'hjorth':
            self.hjorth_dialog.show()
        elif algorithm == 'power spectral density (eeg)':
            self.psd_eeg_dialog.show()
        elif algorithm == 'power spectral density':
            self.psd_dialog.show()
        elif algorithm == 'aperiodic parameters (eeg)':
            self.aperiodic_dialog.show()
        elif algorithm == 'sample entropy':
            self.sample_entropy_dialog.show()
        elif algorithm == 'multiscale entropy':
            self.multiscale_entropy_dialog.show()
        elif algorithm == 'short-time Fourier transform':
            self.stft_dialog.show()
        elif algorithm == 'wavelet transform':
            self.wavelet_transform_dialog.show()
        elif algorithm == 'continuous wavelet transform':
            self.cwt_dialog.show()
        elif algorithm == 'wavelet packet energy':
            self.wavelet_packet_energy_dialog.show()
        elif algorithm == 'local network':
            self.local_network_dialog.show()
        elif algorithm == 'global network':
            self.global_network_dialog.show()
        elif algorithm == 'microstate (eeg)':
            self.microstate.show()

    def center_on_screen(self):
        """
        Center window on screen. Calculates screen dimensions and positions window at center.
        """
        screen_geometry = QApplication.desktop().screenGeometry()
        screen_center = screen_geometry.center()
        self_geometry = self.geometry()
        self_geometry.moveCenter(screen_center)
        self.setGeometry(self_geometry.x(), self_geometry.y(), 300, 400)