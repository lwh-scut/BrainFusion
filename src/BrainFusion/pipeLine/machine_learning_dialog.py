import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
                             QHBoxLayout, QLineEdit, QComboBox, QListWidget, QDialog,
                             QCheckBox, QListWidgetItem, QFrame, QGridLayout, QScrollArea,
                             QFormLayout, QGroupBox, QFileDialog, QMessageBox)
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GridSearchCV

from UI.ui_component import BFPushButton


class RocCurveDialog(QDialog):
    """
    Dialog window displaying ROC curve analysis

    :param parent: Parent widget
    :type parent: QWidget, optional
    """

    def __init__(self, parent=None):
        """
        Initialize the ROC curve dialog

        :param parent: Parent widget
        :type parent: QWidget, optional
        """
        super().__init__(parent)
        self.setWindowTitle("ROC Curve")
        self.layout = QVBoxLayout(self)

        # Create matplotlib figure
        self.figure = plt.figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.layout.addWidget(self.canvas)

        # Generate and plot ROC curve
        self.plot_roc_curve()

    def plot_roc_curve(self):
        """
        Generate and plot synthetic ROC curve data
        """
        # Generate synthetic classification data
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

        # Setup SVM model with parameter grid
        svm_model = svm.SVC(probability=True)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf']
        }

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='roc_auc')
        grid_search.fit(X, y)

        # Get best estimator
        best_model = grid_search.best_estimator_

        # Calculate probability scores
        y_scores = best_model.predict_proba(X)[:, 1]

        # Compute ROC curve metrics
        fpr, tpr, _ = roc_curve(y, y_scores)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        # Refresh the canvas
        self.canvas.draw()


class MachineLearningDialog(QWidget):
    """
    Main machine learning dialog widget
    """

    def __init__(self):
        """
        Initialize the machine learning dialog
        """
        super().__init__()
        self.setWindowTitle("Machine Learning Dialog")
        self.resize(1000, 600)

        # Main layout: left and right panels
        main_layout = QHBoxLayout(self)

        # Create Groups panel (left)
        self.create_groups_widget = QFrame()
        self.create_groups_widget.setFrameShape(QFrame.Box)
        self.create_groups_widget.setStyleSheet("background-color: white;")
        self.create_groups_layout = QVBoxLayout()
        self.create_create_groups_widget()
        self.create_groups_widget.setLayout(self.create_groups_layout)

        # ML Designer panel (right)
        self.ml_designer_widget = QFrame()
        self.ml_designer_widget.setFrameShape(QFrame.Box)
        self.ml_designer_widget.setStyleSheet("background-color: white;")
        self.ml_designer_layout = QVBoxLayout()
        self.create_ml_designer_widget()
        self.ml_designer_widget.setLayout(self.ml_designer_layout)

        # Add widgets to main layout
        main_layout.addWidget(self.create_groups_widget)
        main_layout.addWidget(self.ml_designer_widget)
        main_layout.setStretch(0, 3)  # Left panel width ratio
        main_layout.setStretch(1, 7)  # Right panel width ratio

        self.setLayout(main_layout)

    def create_create_groups_widget(self):
        """
        Create the group management panel
        """
        # Group management title
        self.create_groups_label = QLabel("Create Groups")
        self.create_groups_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")
        self.create_groups_layout.addWidget(self.create_groups_label)

        # Button row for group operations
        self.new_group_button = BFPushButton("New Group")
        self.import_group_button = BFPushButton("Import Groups")
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.new_group_button)
        button_layout.addWidget(self.import_group_button)
        self.create_groups_layout.addLayout(button_layout)

        # Group list display
        self.group_list_widget = QListWidget()
        self.create_groups_layout.addWidget(self.group_list_widget)

        # Validation button
        self.group_validation_button = BFPushButton("Groups Validation")
        self.create_groups_layout.addWidget(self.group_validation_button)

        # Connect signals
        self.new_group_button.clicked.connect(self.add_new_group)
        self.group_list_widget.setContextMenuPolicy(3)  # Enable right-click context menu
        self.group_list_widget.customContextMenuRequested.connect(self.show_context_menu)
        self.group_list_widget.itemClicked.connect(self.edit_group)
        self.group_validation_button.clicked.connect(self.validate_groups)

    def add_new_group(self):
        """
        Display dialog for adding a new group
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("New Group")
        layout = QVBoxLayout(dialog)

        # UI elements
        group_name_label = QLabel("Group Name:")
        group_name_edit = QLineEdit()
        group_folder_button = BFPushButton("Select Folder")
        group_folder_label = QLabel("No folder selected")
        confirm_button = BFPushButton("Confirm")

        # Add to layout
        layout.addWidget(group_name_label)
        layout.addWidget(group_name_edit)
        layout.addWidget(group_folder_button)
        layout.addWidget(group_folder_label)
        layout.addWidget(confirm_button)

        def select_folder():
            """Open folder selection dialog"""
            folder = QFileDialog.getExistingDirectory(self, "Select Folder")
            if folder:
                group_folder_label.setText(folder)

        group_folder_button.clicked.connect(select_folder)

        def confirm_group():
            """Validate and save new group"""
            group_name = group_name_edit.text()
            folder = group_folder_label.text()
            if group_name and folder != "No folder selected":
                # Format and add to group list
                item_text = f"{group_name} - {folder}"
                self.group_list_widget.addItem(item_text)
                dialog.accept()
            else:
                QMessageBox.warning(self, "Warning", "Please fill in all fields.")

        confirm_button.clicked.connect(confirm_group)
        dialog.exec_()

    def edit_group(self, item):
        """
        Edit an existing group entry

        :param item: Group item to edit
        :type item: QListWidgetItem
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Group")
        dialog.setFixedSize(300, 150)

        layout = QVBoxLayout(dialog)

        # Parse existing group info
        group_info = item.text().split(' - ')
        existing_name = group_info[0] if len(group_info) > 0 else ""
        existing_folder = group_info[1] if len(group_info) > 1 else "No folder selected"

        # UI elements
        group_name_label = QLabel("Group Name:")
        group_name_edit = QLineEdit(existing_name)
        group_folder_button = BFPushButton("Select Folder")
        group_folder_label = QLabel(existing_folder)
        confirm_button = BFPushButton("Confirm")

        layout.addWidget(group_name_label)
        layout.addWidget(group_name_edit)
        layout.addWidget(group_folder_button)
        layout.addWidget(group_folder_label)
        layout.addWidget(confirm_button)

        def select_folder():
            """Open folder selection dialog"""
            folder = QFileDialog.getExistingDirectory(self, "Select Folder")
            if folder:
                group_folder_label.setText(folder)

        group_folder_button.clicked.connect(select_folder)

        def confirm_group():
            """Validate and save group changes"""
            group_name = group_name_edit.text()
            folder = group_folder_label.text()
            if group_name and folder != "No folder selected":
                # Update list item
                item.setText(f"{group_name} - {folder}")
                dialog.accept()
            else:
                QMessageBox.warning(self, "Warning", "Please fill in all fields.")

        confirm_button.clicked.connect(confirm_group)
        dialog.exec_()

    def show_context_menu(self, position):
        """
        Display context menu for group operations

        :param position: Click position within widget
        :type position: QPoint
        """
        item = self.group_list_widget.itemAt(position)
        if item:
            menu = QMessageBox()
            delete_button = menu.addButton('Delete', QMessageBox.AcceptRole)
            menu.exec_()

            # Handle delete action
            if menu.clickedButton() == delete_button:
                row = self.group_list_widget.row(item)
                self.group_list_widget.takeItem(row)

    def validate_groups(self):
        """
        Validate group configurations
        """
        group_data = {}
        for index in range(self.group_list_widget.count()):
            item_text = self.group_list_widget.item(index).text()
            group_name, group_folder = item_text.split(' - ')
            group_data[group_name] = group_folder
        QMessageBox.information(self, "Validation", f"Groups validated: {group_data}")

    def create_ml_designer_widget(self):
        """
        Create machine learning task designer panel. Contains UI elements for:
        - Creating new ML tasks
        - Configuring task parameters
        - Running analyses
        - Viewing results
        """
        # Title and new task button
        self.ml_designer_label = QLabel("Machine Learning Designer")
        self.ml_designer_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")
        self.ml_designer_layout.addWidget(self.ml_designer_label)

        self.new_task_button = BFPushButton("New Task")
        self.new_task_button.setFixedWidth(150)
        self.ml_designer_layout.addWidget(self.new_task_button)

        # Scrollable task container
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedHeight(600)

        # Content widget for tasks
        scroll_content = QWidget()
        self.task_list_layout = QVBoxLayout(scroll_content)
        scroll_content.setLayout(self.task_list_layout)
        scroll_area.setWidget(scroll_content)

        self.ml_designer_layout.addWidget(scroll_area)
        self.ml_designer_layout.addStretch(1)

        # Connect new task signal
        self.new_task_button.clicked.connect(self.add_new_task)

    def add_new_task(self):
        """
        Add a new machine learning task to the interface. Creates a task widget with configuration options:
        - Task name
        - Channel selection
        - Feature selection
        - Model type
        - Parameters
        - Analysis options
        """
        # Task container frame
        task_widget = QFrame()
        task_widget.setFrameShape(QFrame.Box)
        task_widget.setStyleSheet("background-color: white; margin: 5px; padding: 5px;")

        # Main layout
        task_main_layout = QVBoxLayout(task_widget)
        task_layout = QGridLayout()
        task_main_layout.addLayout(task_layout)

        # 1. Task name
        task_name_label = QLabel("Task Name:")
        task_name_label.setStyleSheet("font-family: 'Times New Roman'; font-weight: bold;")
        task_name_edit = QLineEdit()
        task_layout.addWidget(task_name_label, 0, 0)
        task_layout.addWidget(task_name_edit, 0, 1)

        # 2. Channel selection
        channel_label = QLabel("Channel:")
        channel_label.setStyleSheet("font-family: 'Times New Roman'; font-weight: bold;")
        channel_edit = QLineEdit()
        select_button = BFPushButton("Select")
        task_layout.addWidget(channel_label, 1, 0)
        task_layout.addWidget(channel_edit, 1, 1)
        task_layout.addWidget(select_button, 1, 2)

        # 3. Feature selection
        feature_label = QLabel("Feature:")
        feature_label.setStyleSheet("font-family: 'Times New Roman'; font-weight: bold;")
        feature_edit = QLineEdit()
        feature_select_button = BFPushButton("Select")
        task_layout.addWidget(feature_label, 2, 0)
        task_layout.addWidget(feature_edit, 2, 1)
        task_layout.addWidget(feature_select_button, 2, 2)

        # 4. Model selection
        model_label = QLabel("Model:")
        model_label.setStyleSheet("font-family: 'Times New Roman'; font-weight: bold;")
        model_combobox = QComboBox()
        model_combobox.addItems(["SVM-linear", "SVM-rbf", "RF", "Integrated Model"])
        task_layout.addWidget(model_label, 3, 0)
        task_layout.addWidget(model_combobox, 3, 1)

        # 5. Grid search option
        enable_grid_search = QCheckBox("Enable Grid Search")
        task_layout.addWidget(enable_grid_search, 4, 0, 1, 2)

        # 6. Parameters button
        parameters_button = BFPushButton("Parameters")
        task_layout.addWidget(parameters_button, 5, 0, 1, 2)

        # 7. Run and export buttons
        run_button = BFPushButton("Run")
        export_button = BFPushButton("Export")
        result_label = QLabel("Not run yet")
        result_label.setStyleSheet("font-family: 'Times New Roman'; font-weight: bold;")
        task_layout.addWidget(run_button, 6, 0)
        task_layout.addWidget(result_label, 6, 1)
        task_layout.addWidget(export_button, 6, 2)

        # 8. Analysis result buttons (initially hidden)
        roc_button = BFPushButton("Show ROC Curve")
        feature_importance_button = BFPushButton("Show Feature Importance")
        roc_button.setVisible(False)
        feature_importance_button.setVisible(False)
        task_layout.addWidget(roc_button, 7, 0, 1, 2)
        task_layout.addWidget(feature_importance_button, 7, 2)

        # Store UI elements for later reference
        task_widget.channel_edit = channel_edit
        task_widget.feature_edit = feature_edit

        # Define selection options
        channels = ["FCC5h", "FCC3h", "FCC4h", "FCC6h", "CCP5h", "CCP3h", "CCP4h", "CCP6h"]
        features = ['NVC CSP']

        # Connect button signals
        select_button.clicked.connect(
            lambda: self.open_channel_selection_dialog(task_widget, channels, channel_edit)
        )
        feature_select_button.clicked.connect(
            lambda: self.open_channel_selection_dialog(task_widget, features, feature_edit)
        )
        parameters_button.clicked.connect(
            lambda: self.open_parameters_dialog(model_combobox.currentText(), enable_grid_search.isChecked())
        )
        run_button.clicked.connect(
            lambda: self.run_analysis(result_label, roc_button, feature_importance_button)
        )
        export_button.clicked.connect(self.export_results)
        roc_button.clicked.connect(self.show_roc_curve_dialog)

        # Add task to layout
        self.task_list_layout.addWidget(task_widget)
        self.task_list_layout.addStretch(1)

    def show_roc_curve_dialog(self):
        """
        Display the ROC curve analysis dialog
        """
        dialog = RocCurveDialog(self)
        dialog.exec_()

    def open_channel_selection_dialog(self, task_widget, options, lineedit):
        """
        Open selection dialog for channels or features

        :param task_widget: Parent task widget
        :type task_widget: QWidget
        :param options: Items to choose from
        :type options: list
        :param lineedit: Target line edit to update
        :type lineedit: QLineEdit
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Selection")
        dialog.setFixedSize(300, 400)

        layout = QVBoxLayout(dialog)
        scroll_area = QScrollArea()
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Create checkboxes for each option
        for item in options:
            checkbox = QCheckBox(item)
            scroll_layout.addWidget(checkbox)

        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        confirm_button = BFPushButton("Confirm")
        layout.addWidget(confirm_button)

        def confirm_selection():
            """Collect selected items and update UI"""
            selected_items = []
            for i in range(scroll_layout.count()):
                checkbox = scroll_layout.itemAt(i).widget()
                if checkbox and checkbox.isChecked():
                    selected_items.append(checkbox.text())
            # Update the target line edit
            lineedit.setText(", ".join(selected_items))
            dialog.accept()

        confirm_button.clicked.connect(confirm_selection)
        dialog.exec_()

    def open_parameters_dialog(self, model_name, enable_grid_search):
        """
        Open model-specific parameter configuration dialog

        :param model_name: Selected model type
        :type model_name: str
        :param enable_grid_search: Whether grid search is enabled
        :type enable_grid_search: bool
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Model Parameters")
        layout = QFormLayout(dialog)

        # Dynamically create UI based on model type
        if model_name in ["SVM-linear", "SVM-rbf"]:
            # SVM parameters
            c_label = QLabel("C:")
            c_edit = QLineEdit("1")
            kernel_label = QLabel("Kernel:")
            kernel_combo = QComboBox()
            kernel_combo.addItems(["linear", "RBF"])
            gamma_label = QLabel("Gamma:")
            gamma_edit = QLineEdit("scale")

            layout.addRow(c_label, c_edit)
            layout.addRow(kernel_label, kernel_combo)
            layout.addRow(gamma_label, gamma_edit)

            # Handle grid search mode
            if enable_grid_search:
                # Replace combos with line edits
                layout.replaceWidget(kernel_combo, QLineEdit(kernel_combo.currentText()))

        elif model_name == "RF":
            # Random Forest parameters
            n_estimators_label = QLabel("n_estimators:")
            n_estimators_edit = QLineEdit("100")
            max_depth_label = QLabel("max_depth:")
            max_depth_edit = QLineEdit("None")
            min_samples_split_label = QLabel("min_samples_split:")
            min_samples_split_edit = QLineEdit("2")

            layout.addRow(n_estimators_label, n_estimators_edit)
            layout.addRow(max_depth_label, max_depth_edit)
            layout.addRow(min_samples_split_label, min_samples_split_edit)

        elif model_name == "Integrated Model":
            # Ensemble model parameters
            base_estimator_label = QLabel("Base Estimator:")
            base_estimator_combo = QComboBox()
            base_estimator_combo.addItems(["Estimator 1", "Estimator 2"])
            learning_rate_label = QLabel("Learning Rate:")
            learning_rate_edit = QLineEdit("0.1")
            n_estimators_label = QLabel("n_estimators:")
            n_estimators_edit = QLineEdit("50")

            layout.addRow(base_estimator_label, base_estimator_combo)
            layout.addRow(learning_rate_label, learning_rate_edit)
            layout.addRow(n_estimators_label, n_estimators_edit)

            # Handle grid search mode
            if enable_grid_search:
                layout.replaceWidget(base_estimator_combo, QLineEdit(base_estimator_combo.currentText()))

        # Confirm button
        confirm_button = BFPushButton("Confirm")
        layout.addRow(confirm_button)
        confirm_button.clicked.connect(dialog.accept)

        dialog.exec_()

    def run_analysis(self, result_label, roc_button, feature_importance_button):
        """
        Run the machine learning analysis (simulation)

        :param result_label: UI element to display status
        :type result_label: QLabel
        :param roc_button: ROC curve button (to enable after completion)
        :type roc_button: QPushButton
        :param feature_importance_button: Feature importance button (to enable after completion)
        :type feature_importance_button: QPushButton
        """
        # Update UI for running state
        result_label.setText("Running...")
        QApplication.processEvents()  # Force UI update

        # Simulate analysis work (in a real app this would be replaced with actual ML code)
        # ... [Actual machine learning analysis would go here] ...

        # Update UI after completion
        result_label.setText("Run Completed")
        roc_button.setVisible(True)
        feature_importance_button.setVisible(True)

    def export_results(self):
        """Open dialog to export analysis results"""
        dialog = QFileDialog()
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilter("CSV Files (*.csv)")
        dialog.setDefaultSuffix("csv")
        dialog.exec_()


if __name__ == "__main__":

    # Create Qt application instance
    app = QApplication(sys.argv)

    # Create and show main window
    window = MachineLearningDialog()
    window.show()

    # Execute application
    sys.exit(app.exec_())