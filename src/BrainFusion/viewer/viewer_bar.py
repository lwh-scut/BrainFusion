import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class MultiBarChartApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Excel 数据多子图柱状图")
        self.setGeometry(100, 100, 1200, 800)

        # 主窗口布局
        self.main_widget = QWidget()
        self.layout = QVBoxLayout(self.main_widget)

        # 按钮用于选择Excel文件
        self.button = QPushButton("加载 Excel 文件")
        self.button.clicked.connect(self.load_excel_file)
        self.layout.addWidget(self.button)

        # Matplotlib FigureCanvas
        self.canvas = FigureCanvas(plt.figure(figsize=(10, 8)))
        self.layout.addWidget(self.canvas)

        self.setCentralWidget(self.main_widget)

    def load_excel_file(self):
        # 打开文件对话框选择 Excel 文件
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 Excel 文件", "", "Excel Files (*.xlsx *.xls)")
        if not file_path:
            return

        # 读取 Excel 数据
        try:
            data = pd.read_excel(file_path)
        except Exception as e:
            print(f"读取文件失败: {e}")
            return

        # 检查数据格式是否正确
        if data.shape[1] < 2:
            print("数据格式不正确，请确保至少有两列数据")
            return

        # 使用数据绘制柱状图
        self.plot_bar_charts(data)

    def plot_bar_charts(self, data):
        plt.clf()  # 清空图像

        num_subplots = data.shape[1] // 2  # 每两列对应一个子图
        if num_subplots == 0:
            print("数据格式不正确，请确保至少有两列数据")
            return

        # 创建子图
        fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 8))
        if num_subplots == 1:
            axes = [axes]  # 保证 axes 是一个可迭代的列表

        for i in range(num_subplots):
            ax = axes[i]
            x = data.iloc[:, i * 2]  # 每两个列中的第一列为 x 轴
            y = data.iloc[:, i * 2 + 1]  # 每两个列中的第二列为 y 轴

            # 绘制柱状图
            bars = ax.bar(x, y, width=0.6, color='skyblue', label=f'图 {i + 1}')

            # 添加注释
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

            # 设置图例和标题
            ax.legend()
            ax.set_title(f'子图 {i + 1}')
            ax.set_xlabel("类别")
            ax.set_ylabel("值")

        # 调整布局，避免重叠
        plt.tight_layout()

        # 更新画布
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiBarChartApp()
    window.show()
    sys.exit(app.exec_())
