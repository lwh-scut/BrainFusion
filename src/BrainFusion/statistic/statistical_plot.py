import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ScatterPlotWindow(QMainWindow):
    """
    PyQt5-based window for visualizing group data using scatter plots.
    """

    def __init__(self):
        """Initialize scatter plot visualization window."""
        super().__init__()
        self.setWindowTitle("Scatter Plot")
        self.setGeometry(100, 100, 800, 600)
        self._initialize_ui()

    def _initialize_ui(self):
        """Set up UI components and layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        self.canvas = FigureCanvas(Figure(figsize=(8, 6)))
        layout.addWidget(self.canvas)

    def plot_scatter(self, data_dict):
        """
        Render group data as a scatter plot.

        :param data_dict: Dictionary of group data values
        :type data_dict: dict
        """
        ax = self.canvas.figure.add_subplot(111)
        ax.clear()

        # Plot group values
        for group_name, values in data_dict.items():
            ax.scatter([group_name] * len(values), values, label=group_name)

        # Configure plot appearance
        ax.set_title("Group Comparison")
        ax.set_xlabel("Groups")
        ax.set_ylabel("Values")
        ax.legend()
        self.canvas.draw()


class DensityHistogramWindow(QMainWindow):
    """
    PyQt5-based window for visualizing density histograms.
    """

    def __init__(self):
        """Initialize density histogram visualization window."""
        super().__init__()
        self.setWindowTitle("Density Histogram")
        self.setGeometry(100, 100, 800, 600)
        self._initialize_ui()

    def _initialize_ui(self):
        """Set up UI components and layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        self.canvas = FigureCanvas(Figure(figsize=(8, 6)))
        layout.addWidget(self.canvas)

    def plot_density_histogram(self, data_dict):
        """
        Render group data as density histograms.

        :param data_dict: Dictionary of group data values
        :type data_dict: dict
        """
        import numpy as np
        ax = self.canvas.figure.add_subplot(111)
        ax.clear()

        # Plot density distributions
        for group_name, values in data_dict.items():
            ax.hist(
                values,
                bins=20,
                alpha=0.5,
                density=True,
                label=f"{group_name} (Density)"
            )

        # Configure plot appearance
        ax.set_title("Density Distribution")
        ax.set_xlabel("Values")
        ax.set_ylabel("Density")
        ax.legend()
        self.canvas.draw()


class SignificanceBoxPlotWindow(QMainWindow):
    """
    PyQt5-based window for boxplots with significance markers.
    """

    def __init__(self):
        """Initialize significance boxplot visualization window."""
        super().__init__()
        self.setWindowTitle("Box Plot with Significance")
        self.setGeometry(100, 100, 800, 600)
        self._initialize_ui()

    def _initialize_ui(self):
        """Set up UI components and layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        self.canvas = FigureCanvas(plt.figure(figsize=(10, 6)))
        layout.addWidget(self.canvas)

    def plot_boxplot_with_significance(self, data_dict, significance_dict):
        """
        Render boxplots with statistical significance markers.

        :param data_dict: Dictionary of group data values
        :type data_dict: dict
        :param significance_dict: Significance mappings between groups
        :type significance_dict: dict
        """
        import numpy as np
        ax = self.canvas.figure.add_subplot(111)
        ax.clear()

        # Extract and position groups
        groups = list(data_dict.keys())
        group_data = [data_dict[group] for group in groups]
        group_positions = np.arange(len(groups))

        # Create boxplots
        box_plot = ax.boxplot(
            group_data,
            positions=group_positions,
            patch_artist=True,
            notch=True
        )

        # Apply group colors
        plot_colors = plt.cm.tab10.colors
        for box, color in zip(box_plot['boxes'], plot_colors[:len(groups)]):
            box.set_facecolor(color)

        # Add significance markers
        significance_markers = {0.0001: '***', 0.001: '**', 0.05: '*'}
        bracket_height = 0.2
        asterisk_offset = 0.01
        line_width = 1.5

        # Process significance indicators
        for (group1, group2), p_value in significance_dict.items():
            if p_value < max(significance_markers.keys()):
                # Determine marker type
                marker = next(
                    v for k, v in significance_markers.items()
                    if p_value < k
                )

                # Calculate marker positions
                idx1, idx2 = groups.index(group1), groups.index(group2)
                pos1, pos2 = group_positions[idx1], group_positions[idx2]
                max_val = max(max(data_dict[group1]), max(data_dict[group2]))
                y_bracket = max_val + bracket_height
                y_asterisk = y_bracket + asterisk_offset

                # Add statistical markers
                ax.text(
                    (pos1 + pos2) / 2,
                    y_asterisk,
                    marker,
                    ha='center',
                    va='bottom',
                    fontsize=14,
                    color='red'
                )

                # Add bracket lines
                ax.plot([pos1, pos2], [y_bracket, y_bracket], color='gray', lw=line_width)
                ax.plot([pos1, pos1], [y_bracket, y_bracket - 0.06], color='gray', lw=line_width)
                ax.plot([pos2, pos2], [y_bracket, y_bracket - 0.06], color='gray', lw=line_width)

        # Add significance legend
        ax.text(
            0.5,
            -0.2,
            'Significance markers: * p < 0.05, ** p < 0.001, *** p < 0.0001',
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=12
        )

        # Configure plot appearance
        ax.set_title("Statistical Group Comparison")
        ax.set_xlabel("Groups")
        ax.set_ylabel("Values")
        ax.set_xticks(group_positions)
        ax.set_xticklabels(groups)
        self.canvas.draw()


class SignificanceViolinPlotWindow(QMainWindow):
    """
    PyQt5-based window for violin plots with significance markers.
    """

    def __init__(self):
        """Initialize significance violin plot visualization window."""
        super().__init__()
        self.setWindowTitle("Violin Plot with Significance")
        self.setGeometry(100, 100, 800, 600)
        self._initialize_ui()

    def _initialize_ui(self):
        """Set up UI components and layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        self.canvas = FigureCanvas(plt.figure(figsize=(10, 6)))
        layout.addWidget(self.canvas)

    def plot_violin_with_significance(self, data_dict, significance_dict):
        """
        Render violin plots with statistical significance markers.

        :param data_dict: Dictionary of group data values
        :type data_dict: dict
        :param significance_dict: Significance mappings between groups
        :type significance_dict: dict
        """
        import numpy as np
        ax = self.canvas.figure.add_subplot(111)
        ax.clear()

        # Extract and position groups
        groups = list(data_dict.keys())
        group_data = [data_dict[group] for group in groups]
        group_positions = np.arange(len(groups))

        # Create violin plots
        violin_plot = ax.violinplot(
            group_data,
            positions=group_positions,
            showmedians=True,
            showmeans=False,
            showextrema=False
        )

        # Apply group styling
        plot_colors = plt.cm.tab10.colors
        for violin, color in zip(violin_plot['bodies'], plot_colors[:len(groups)]):
            violin.set_facecolor(color)
            violin.set_edgecolor('black')
            violin.set_alpha(0.7)

        # Add significance markers
        significance_markers = {0.0001: '***', 0.001: '**', 0.05: '*'}
        bracket_height = 0.2
        asterisk_offset = 0.01
        line_width = 1.5

        # Process significance indicators
        for (group1, group2), p_value in significance_dict.items():
            if p_value < max(significance_markers.keys()):
                # Determine marker type
                marker = next(
                    v for k, v in significance_markers.items()
                    if p_value < k
                )

                # Calculate marker positions
                idx1, idx2 = groups.index(group1), groups.index(group2)
                pos1, pos2 = group_positions[idx1], group_positions[idx2]
                max_val = max(max(data_dict[group1]), max(data_dict[group2]))
                y_bracket = max_val + bracket_height
                y_asterisk = y_bracket + asterisk_offset

                # Add statistical markers
                ax.text(
                    (pos1 + pos2) / 2,
                    y_asterisk,
                    marker,
                    ha='center',
                    va='bottom',
                    fontsize=14,
                    color='red'
                )

                # Add bracket lines
                ax.plot([pos1, pos2], [y_bracket, y_bracket], color='gray', lw=line_width)
                ax.plot([pos1, pos1], [y_bracket, y_bracket - 0.06], color='gray', lw=line_width)
                ax.plot([pos2, pos2], [y_bracket, y_bracket - 0.06], color='gray', lw=line_width)

        # Add significance legend
        ax.text(
            0.5,
            -0.2,
            'Significance markers: * p < 0.05, ** p < 0.001, *** p < 0.0001',
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=12
        )

        # Configure plot appearance
        ax.set_title("Distribution Comparison")
        ax.set_xlabel("Groups")
        ax.set_ylabel("Values")
        ax.set_xticks(group_positions)
        ax.set_xticklabels(groups)
        self.canvas.draw()


if __name__ == "__main__":
    # Demonstration of visualization functionality
    import numpy as np

    sample_data = {
        "Group 1": np.random.normal(5, 1, 30),
        "Group 2": np.random.normal(6, 1, 30),
        "Group 3": np.random.normal(7, 1, 30),
        "Group 4": np.random.normal(7, 1, 30),
        "Group 5": np.random.normal(7, 1, 30),
    }

    sample_significance = {
        ("Group 1", "Group 2"): 0.03,
        ("Group 2", "Group 3"): 0.0005,
        ("Group 1", "Group 3"): 0.2,
        ("Group 1", "Group 4"): 0.00002,
        ("Group 1", "Group 5"): 0.00002,
    }

    # Launch demonstration application
    app = QApplication(sys.argv)
    plot_window = SignificanceViolinPlotWindow()
    plot_window.plot_violin_with_significance(sample_data, sample_significance)
    plot_window.show()
    sys.exit(app.exec_())