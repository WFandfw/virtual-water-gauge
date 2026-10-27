import sys
import io
import numpy as np
import requests
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSlider, QFileDialog,
    QButtonGroup, QRadioButton, QGroupBox, QSplitter, QMessageBox
)
from PyQt5.QtCore import Qt, QPoint, QRect, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QFont
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

# 导入SAM3相关模块
from sam3.model.sam3_image_processor import Sam3Processor
import sam3
from sam3 import build_sam3_image_model
import os


class ImageProcessor(QThread):
    """后台处理线程，避免UI冻结"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, processor, func, *args, **kwargs):
        super().__init__()
        self.processor = processor
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib画布"""

    def __init__(self, parent=None, width=12, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # 设置背景色
        self.fig.patch.set_facecolor('#eeeeee')
        super().__init__(self.fig)
        self.setParent(parent)

        # 设置鼠标样式为十字光标
        self.setCursor(Qt.CrossCursor)


class SAM3SegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🖼️ SAM3 Interactive Segmentation")
        self.setGeometry(100, 100, 1400, 900)

        # 初始化SAM3模型
        self.init_sam3_model()

        # 状态变量
        self.state = None
        self.current_image = None
        self.current_image_array = None
        self.box_mode = "positive"
        self.drawing_box = False
        self.box_start = None
        self.current_rect = None
        self.drawing_rect = None

        # 设置UI
        self.setup_ui()

        # 应用全局样式
        self.apply_stylesheet()

    def setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧控制面板
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)

        # 右侧显示区域
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        display_layout.setContentsMargins(20, 20, 20, 20)

        # Matplotlib画布
        self.canvas = MplCanvas(self)
        self.canvas.setSizePolicy(1, 1)
        display_layout.addWidget(self.canvas)

        # 连接鼠标事件
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        splitter.addWidget(display_widget)

        # 设置分割器比例
        splitter.setSizes([400, 1000])

        # 初始状态
        self.status_label.setText("📤 上传图片开始")

    def apply_stylesheet(self):
        """应用全局样式表"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }

            QWidget {
                font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
                font-size: 13px;
            }

            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 18px;
                background-color: white;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 15px;
                padding: 0 8px;
                color: #2c3e50;
                background-color: white;
            }

            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 13px;
            }

            QPushButton:hover {
                background-color: #45a049;
            }

            QPushButton:pressed {
                background-color: #3d8b40;
            }

            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }

            QPushButton#segmentBtn {
                background-color: #2196F3;
            }

            QPushButton#segmentBtn:hover {
                background-color: #1976D2;
            }

            QPushButton#clearBtn {
                background-color: #f44336;
            }

            QPushButton#clearBtn:hover {
                background-color: #d32f2f;
            }

            QPushButton#boxSegmentBtn {
                background-color: #FF9800;
            }

            QPushButton#boxSegmentBtn:hover {
                background-color: #F57C00;
            }

            QLineEdit {
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                padding: 8px 12px;
                background-color: white;
                font-size: 13px;
            }

            QLineEdit:focus {
                border: 2px solid #4CAF50;
            }

            QRadioButton {
                spacing: 8px;
                padding: 5px;
            }

            QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }

            QRadioButton::indicator:unchecked {
                border: 2px solid #bdbdbd;
                border-radius: 9px;
                background-color: white;
            }

            QRadioButton::indicator:checked {
                border: 2px solid #4CAF50;
                border-radius: 9px;
                background-color: #4CAF50;
                background-image: radial-gradient(circle, white 0%, white 40%, transparent 50%);
            }

            QSlider::groove:horizontal {
                border: 1px solid #bdbdbd;
                height: 6px;
                background: #e0e0e0;
                border-radius: 3px;
            }

            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 2px solid #4CAF50;
                width: 18px;
                height: 18px;
                margin: -7px 0;
                border-radius: 9px;
            }

            QSlider::handle:horizontal:hover {
                background: #45a049;
                border: 2px solid #45a049;
            }

            QLabel {
                color: #2c3e50;
            }

            QLabel#statusLabel {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                padding: 12px;
                color: #2c3e50;
                font-size: 13px;
            }
        """)

    def init_sam3_model(self):
        """初始化SAM3模型"""
        try:
            print("正在初始化SAM3模型...")

            # 1. BPE文件路径 - 修改为你的新路径
            bpe_path = r"D:\sam3model\assets\bpe_simple_vocab_16e6.txt.gz"

            # 2. 权重文件路径
            checkpoint_path = r"D:\sam3model\sam3.pt"

            print(f"BPE文件路径: {bpe_path}")
            print(f"权重文件路径: {checkpoint_path}")
            print(f"BPE文件存在: {os.path.exists(bpe_path)}")
            print(f"权重文件存在: {os.path.exists(checkpoint_path)}")

            # 检查BPE文件
            if not os.path.exists(bpe_path):
                QMessageBox.critical(self, "错误",
                                     f"BPE文件不存在:\n{bpe_path}\n\n"
                                     f"请确保文件在正确位置。")
                sys.exit(1)

            # 检查权重文件
            if not os.path.exists(checkpoint_path):
                # 尝试在原位置查找
                alt_checkpoint = r"D:\sam3model\sam3.pt"
                print(f"尝试备选权重路径: {alt_checkpoint}")
                if os.path.exists(alt_checkpoint):
                    checkpoint_path = alt_checkpoint
                    print(f"使用备选权重: {checkpoint_path}")
                else:
                    QMessageBox.critical(self, "错误",
                                         f"权重文件不存在:\n{checkpoint_path}\n"
                                         f"备选路径也不存在:\n{alt_checkpoint}")
                    sys.exit(1)

            # 3. 构建模型并加载权重
            print("构建模型中...")
            model = build_sam3_image_model(bpe_path=bpe_path)
            print("基础模型构建完成")

            # 4. 加载权重
            print("加载模型权重...")
            import torch

            # 检查文件大小
            file_size = os.path.getsize(checkpoint_path)
            print(f"权重文件大小: {file_size / 1024 / 1024:.2f} MB")

            # 加载权重
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            print(f"权重文件类型: {type(checkpoint)}")

            # === 方法1：移除权重键名中的 'detector.' 前缀 ===
            if isinstance(checkpoint, dict):
                print(f"原始权重字典keys (前5个):")
                keys_list = list(checkpoint.keys())
                for i, key in enumerate(keys_list[:5]):
                    print(f"  [{i}] {key}")

                # 检查是否需要移除detector.前缀
                needs_strip = any(key.startswith('detector.') for key in checkpoint.keys())
                print(f"需要移除detector.前缀: {needs_strip}")

                if needs_strip:
                    # 创建新的权重字典
                    new_checkpoint = {}
                    for key, value in checkpoint.items():
                        if key.startswith('detector.'):
                            # 移除detector.前缀
                            new_key = key.replace('detector.', '', 1)
                        else:
                            new_key = key
                        new_checkpoint[new_key] = value

                    print(f"移除前缀后的keys (前5个):")
                    new_keys_list = list(new_checkpoint.keys())
                    for i, key in enumerate(new_keys_list[:5]):
                        print(f"  [{i}] {key}")

                    # 尝试加载
                    model.load_state_dict(new_checkpoint, strict=False)
                    print("✓ 使用移除前缀后的权重加载成功")
                else:
                    # 尝试不同的key加载权重
                    weight_keys = ["state_dict", "model", "weights"]
                    loaded = False

                    for key in weight_keys:
                        if key in checkpoint:
                            model.load_state_dict(checkpoint[key], strict=False)
                            print(f"✓ 从 '{key}' 加载权重成功")
                            loaded = True
                            break

                    if not loaded:
                        # 尝试直接作为state_dict
                        try:
                            model.load_state_dict(checkpoint, strict=False)
                            print("✓ 直接加载权重成功")
                        except Exception as e:
                            print(f"✗ 直接加载失败: {e}")
                            # 尝试再次移除可能的其他前缀
                            print("尝试处理可能的嵌套结构...")
                            # 打印模型结构
                            print("\n模型结构示例 (前10个键):")
                            for i, (name, _) in enumerate(model.state_dict().items()):
                                if i < 10:
                                    print(f"  {name}")
                                else:
                                    break

                            # 尝试找到匹配的权重
                            model_keys = set(model.state_dict().keys())
                            ckpt_keys = set(checkpoint.keys())
                            common_keys = model_keys.intersection(ckpt_keys)
                            print(f"\n匹配的键数量: {len(common_keys)}")

                            # 如果有很多匹配的键，尝试加载
                            if len(common_keys) > 100:  # 假设至少有100个匹配的键
                                model.load_state_dict(checkpoint, strict=False)
                                print("✓ 使用strict=False加载成功")
                            else:
                                raise
            else:
                # 如果不是字典，可能是直接的state_dict
                model.load_state_dict(checkpoint, strict=False)
                print("✓ 直接加载非字典格式权重成功")

            # 5. 设置为评估模式
            model.eval()
            print("模型设置为评估模式")

            # 6. 创建处理器
            print("创建处理器中...")
            self.processor = Sam3Processor(model)
            print("✓ SAM3初始化成功!")

            # 显示模型信息
            total_params = sum(p.numel() for p in model.parameters())
            print(f"模型总参数量: {total_params:,}")

        except Exception as e:
            print(f"初始化详细错误: {e}")
            import traceback
            traceback.print_exc()

            QMessageBox.critical(self, "错误",
                                 f"初始化SAM3模型失败:\n\n{str(e)}\n\n"
                                 f"文件检查:\n"
                                 f"• BPE文件: {os.path.exists(bpe_path)}\n"
                                 f"• 权重文件: {os.path.exists(checkpoint_path)}")
            sys.exit(1)

    def create_control_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        panel.setMaximumWidth(380)
        panel.setMinimumWidth(380)
        # panel.setStyleSheet("background-color: #fafafa;")

        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 状态标签
        self.status_label = QLabel("📤 上传图片开始")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(50)
        layout.addWidget(self.status_label)

        # 图片源组
        image_group = QGroupBox("📁 图片源")
        image_layout = QVBoxLayout()
        image_layout.setSpacing(10)

        self.upload_btn = QPushButton("📂 上传图片")
        self.upload_btn.setMinimumHeight(45)
        self.upload_btn.clicked.connect(self.on_upload_image)
        image_layout.addWidget(self.upload_btn)

        image_group.setLayout(image_layout)
        layout.addWidget(image_group)

        # 分割提示组
        prompt_group = QGroupBox("✨ 分割提示")
        prompt_layout = QVBoxLayout()
        prompt_layout.setSpacing(12)

        # 文本提示
        text_label = QLabel("💬 文本提示:")
        text_label.setStyleSheet("font-weight: bold; color: #555;")
        prompt_layout.addWidget(text_label)

        text_input_layout = QHBoxLayout()
        text_input_layout.setSpacing(8)
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText('输入分割提示 (例如: "person", "dog")')
        self.text_input.setMinimumHeight(40)
        self.text_input.returnPressed.connect(self.on_text_prompt)

        self.segment_btn = QPushButton("🎯 分割")
        self.segment_btn.setObjectName("segmentBtn")
        self.segment_btn.setMinimumHeight(40)
        self.segment_btn.setMinimumWidth(100)
        self.segment_btn.clicked.connect(self.on_text_prompt)
        text_input_layout.addWidget(self.text_input, 3)
        text_input_layout.addWidget(self.segment_btn, 1)
        prompt_layout.addLayout(text_input_layout)

        # 分隔线
        separator1 = QLabel()
        separator1.setStyleSheet("background-color: #e0e0e0; max-height: 1px;")
        separator1.setMaximumHeight(1)
        prompt_layout.addWidget(separator1)

        # 框选模式
        box_mode_label = QLabel("🖱️ 框选模式:")
        box_mode_label.setStyleSheet("font-weight: bold; color: #555; margin-top: 5px;")
        prompt_layout.addWidget(box_mode_label)

        self.box_mode_group = QButtonGroup()
        self.positive_radio = QRadioButton("✅ 正向框选 (包含)")
        self.positive_radio.setStyleSheet("color: #4CAF50; font-weight: 500;")
        self.negative_radio = QRadioButton("❌ 负向框选 (排除)")
        self.negative_radio.setStyleSheet("color: #f44336; font-weight: 500;")
        self.positive_radio.setChecked(True)
        self.positive_radio.toggled.connect(self.on_box_mode_change)

        self.box_mode_group.addButton(self.positive_radio)
        self.box_mode_group.addButton(self.negative_radio)

        prompt_layout.addWidget(self.positive_radio)
        prompt_layout.addWidget(self.negative_radio)

        # 置信度滑块
        confidence_label = QLabel("📊 置信度阈值: 0.50")
        confidence_label.setStyleSheet("font-weight: bold; color: #555; margin-top: 8px;")
        self.confidence_label = confidence_label
        prompt_layout.addWidget(confidence_label)

        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setMinimumHeight(30)
        self.confidence_slider.valueChanged.connect(self.on_confidence_change)
        prompt_layout.addWidget(self.confidence_slider)

        box_layout = QHBoxLayout()
        box_layout.setSpacing(8)

        # 清除按钮
        self.clear_btn = QPushButton("🗑️ 清除所有提示")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.setMinimumHeight(40)
        self.clear_btn.clicked.connect(self.on_clear_prompts)
        box_layout.addWidget(self.clear_btn)

        self.box_fenge_btn = QPushButton("📦 分割")
        self.box_fenge_btn.setObjectName("boxSegmentBtn")
        self.box_fenge_btn.setMinimumHeight(40)
        self.box_fenge_btn.clicked.connect(self.box_fenge)
        box_layout.addWidget(self.box_fenge_btn)

        prompt_layout.addLayout(box_layout)

        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)

        # 显示设置组
        display_group = QGroupBox("⚙️ 显示设置")
        display_layout = QVBoxLayout()
        display_layout.setSpacing(10)

        size_label = QLabel("📐 图片显示尺寸: 960")
        size_label.setStyleSheet("font-weight: bold; color: #555;")
        self.size_label = size_label
        display_layout.addWidget(size_label)

        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(300)
        self.size_slider.setMaximum(2000)
        self.size_slider.setValue(960)
        self.size_slider.setMinimumHeight(30)
        self.size_slider.valueChanged.connect(self.on_size_change)
        display_layout.addWidget(self.size_slider)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        layout.addStretch()

        return panel

    def box_fenge(self):

        if self.state is not None and "prompted_boxes" in self.state:
            prompted_boxes = self.state["prompted_boxes"]

            self.processor.reset_all_prompts(self.state)
            for i in range(len(prompted_boxes)):
                box = prompted_boxes[i].get("box")
                label = prompted_boxes[i].get("label")
                self.state = self.processor.add_geometric_prompt(box, label, self.state)
                self.set_loading(False)
                print(box)
                print(label)
            self.update_display()

    def set_loading(self, is_loading, message="处理中..."):
        """设置加载状态"""
        if is_loading:
            self.status_label.setText(f"⏳ {message}")
            self.upload_btn.setEnabled(False)
            self.segment_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
            self.positive_radio.setEnabled(False)
            self.negative_radio.setEnabled(False)
            self.confidence_slider.setEnabled(False)
            QApplication.processEvents()
        else:
            self.upload_btn.setEnabled(True)
            self.segment_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)
            self.positive_radio.setEnabled(True)
            self.negative_radio.setEnabled(True)
            self.confidence_slider.setEnabled(True)

    def on_upload_image(self):
        """处理图片上传"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            try:
                image = Image.open(file_path).convert("RGB")
                self.set_image(image)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图片失败: {str(e)}")

    def set_image(self, image):
        """设置当前图片"""
        self.set_loading(True, "通过模型处理图片...")

        try:
            self.current_image = image
            self.current_image_array = np.array(image)
            self.state = self.processor.set_image(image)
            self.set_loading(False)
            self.status_label.setText(f"✅ 图片已加载: {image.size[0]}x{image.size[1]} 像素")
            self.resize_figure()
            self.update_display()
        except Exception as e:
            self.set_loading(False)
            QMessageBox.critical(self, "错误", f"处理图片失败: {str(e)}")

    def on_text_prompt(self):
        """处理文本提示"""
        if self.state is None:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return

        prompt = self.text_input.text().strip()
        if not prompt:
            QMessageBox.warning(self, "警告", "请输入提示")
            return

        self.set_loading(True, f'使用提示分割: "{prompt}"...')

        try:
            self.processor.reset_all_prompts(self.state)
            self.state = self.processor.set_text_prompt(prompt, self.state)
            self.set_loading(False)
            self.status_label.setText(f'✅ 已使用提示分割: "{prompt}"')
            self.update_display()
        except Exception as e:
            self.set_loading(False)
            QMessageBox.critical(self, "错误", f"分割失败: {str(e)}")

    def on_box_mode_change(self):
        """处理框选模式变化"""
        self.box_mode = "positive" if self.positive_radio.isChecked() else "negative"

    def on_clear_prompts(self):
        """清除所有提示"""
        if self.current_image is not None:
            self.set_loading(True, "清除提示并重置...")
            # self.state = self.processor.reset_all_prompts(self.state)
            if self.state is not None and "prompted_boxes" in self.state:
                del self.state["prompted_boxes"]
            if self.state is not None and "masks" in self.state:
                del self.state["masks"]
            self.text_input.clear()
            self.set_loading(False)
            self.status_label.setText("✅ 已清除所有提示")
            self.update_display()

    def on_confidence_change(self, value):
        """处理置信度变化"""
        confidence = value / 100.0
        self.confidence_label.setText(f"📊 置信度阈值: {confidence:.2f}")

        if self.state is not None:
            self.state = self.processor.set_confidence_threshold(confidence, self.state)
            self.update_display()

    def on_size_change(self, value):
        """处理尺寸变化"""
        self.size_label.setText(f"📐 图片显示尺寸: {value}")

        if self.current_image is not None:
            self.resize_figure()
            self.update_display()

    def resize_figure(self):
        """调整图形大小"""
        if self.current_image is None:
            return

        img_w, img_h = self.current_image.size
        display_w = float(self.size_slider.value())
        aspect_ratio = img_h / img_w
        display_h = int(display_w * aspect_ratio)

        dpi = self.canvas.fig.dpi
        new_figsize = (display_w / dpi, display_h / dpi)
        self.canvas.fig.set_size_inches(new_figsize, forward=True)

    def on_press(self, event):
        """鼠标按下事件"""
        if event.inaxes != self.canvas.ax:
            return
        self.drawing_box = True
        self.box_start = (event.xdata, event.ydata)

        # 创建新的临时矩形
        self.current_rect = None

    def on_motion(self, event):
        """鼠标移动事件"""
        if not self.drawing_box or event.inaxes != self.canvas.ax or self.box_start is None:
            return

        # 移除旧的临时矩形（如果存在）
        if self.current_rect is not None:
            self.current_rect.remove()

        x0, y0 = self.box_start
        x1, y1 = event.xdata, event.ydata
        width = x1 - x0
        height = y1 - y0

        color = "green" if self.box_mode == "positive" else "red"
        self.current_rect = Rectangle(
            (x0, y0), width, height,
            fill=False, edgecolor=color, linewidth=2, linestyle='--'
        )
        self.canvas.ax.add_patch(self.current_rect)
        self.canvas.draw_idle()

    def on_release(self, event):
        """鼠标释放事件"""
        if not self.drawing_box or event.inaxes != self.canvas.ax or self.box_start is None:
            self.drawing_box = False
            return

        self.drawing_box = False

        if self.current_rect is not None:
            # 将临时矩形转换为永久矩形（改变样式）
            self.current_rect.set_linestyle('-')  # 改为实线
            self.current_rect.set_linewidth(1.5)  # 调整线宽
            self.current_rect = None  # 重置当前矩形，但不移除

        if self.state is None:
            return

        x0, y0 = self.box_start
        x1, y1 = event.xdata, event.ydata

        x_min = min(x0, x1)
        x_max = max(x0, x1)
        y_min = min(y0, y1)
        y_max = max(y0, y1)

        if abs(x_max - x_min) < 5 or abs(y_max - y_min) < 5:
            return

        img_h = self.state["original_height"]
        img_w = self.state["original_width"]

        center_x = (x_min + x_max) / 2.0 / img_w
        center_y = (y_min + y_max) / 2.0 / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h

        box = [center_x, center_y, width, height]
        label = self.box_mode == "positive"
        mode_str = "正向" if label else "负向"

        if "prompted_boxes" not in self.state:
            self.state["prompted_boxes"] = []
        self.state["prompted_boxes"].append({
            "box": box,
            "label": label
        })
        print(self.state["prompted_boxes"])

        # 重置起始点
        self.box_start = None

        # 重绘画布以显示所有矩形
        self.canvas.draw_idle()

    def update_display(self):
        """更新显示"""
        if self.current_image_array is None:
            return

        self.canvas.ax.clear()
        self.canvas.ax.axis('off')
        self.canvas.ax.imshow(self.current_image_array)

        if self.state is not None and "masks" in self.state:
            masks = self.state.get("masks", [])
            boxes = self.state.get("boxes", [])
            scores = self.state.get("scores", [])

            if len(masks) > 0:
                mask_overlay = np.zeros((*self.current_image_array.shape[:2], 4))

                for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                    mask_np = mask[0].cpu().numpy()

                    color = plt.cm.tab10(i % 10)[:3]
                    mask_overlay[mask_np > 0.5] = (*color, 0.5)

                    x0, y0, x1, y1 = box.cpu().numpy()
                    rect = Rectangle(
                        (x0, y0), x1 - x0, y1 - y0,
                        fill=False, edgecolor=color, linewidth=2
                    )
                    self.canvas.ax.add_patch(rect)

                    self.canvas.ax.text(
                        x0, y0 - 5, f"{score:.2f}",
                        color='white', fontsize=10,
                        bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2)
                    )

                self.canvas.ax.imshow(mask_overlay)
                self.status_label.setText(f"✅ 找到 {len(masks)} 个对象")
            else:
                self.status_label.setText("⚠️ 未找到高于置信度阈值的对象")

        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    window = SAM3SegmentationApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()