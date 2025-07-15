# Intel RealSense D435i + YOLO 实时3D目标检测与测距系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-4.8+-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.13+-red.svg)](https://pytorch.org/)

一个基于Intel RealSense D435i深度相机和YOLO目标检测算法的实时3D目标检测与测距系统。系统能够同时进行目标检测、深度估计和3D定位，适用于机器人导航、无人机避障、智能监控等应用场景。

## 🎯 主要特性

- **实时3D目标检测**: 结合RGB图像和深度信息进行精确的3D目标检测
- **多目标跟踪**: 支持多个目标的实时跟踪和轨迹预测
- **深度测距**: 提供精确的距离测量和3D坐标计算
- **多平台支持**: 支持Linux、Windows、macOS，特别优化了Jetson设备
- **灵活配置**: 丰富的配置选项，支持不同的YOLO模型和相机设置
- **可视化界面**: 实时显示检测结果、深度信息和3D可视化
- **性能优化**: 支持GPU加速，多线程处理，优化的数据流水线

## 📋 系统要求

### 硬件要求
- **相机**: Intel RealSense D435i 深度相机
- **计算设备**: 
  - CPU: Intel Core i5 或同等性能的ARM处理器
  - 内存: 8GB RAM（推荐16GB）
  - GPU: NVIDIA GPU（可选，用于加速）
- **接口**: USB 3.0 或更高版本

### 软件要求
- **操作系统**: 
  - Ubuntu 20.04+ (推荐)
  - Windows 10+
  - macOS 10.15+
- **Python**: 3.7+
- **其他**: CUDA 11.0+ (可选，用于GPU加速)

## 🚀 快速开始

### 1. 安装系统

#### 方法一：使用安装脚本（推荐 - Jetson设备）
```bash
# 克隆仓库
git clone https://github.com/your-username/realsense-yolo-system.git
cd realsense-yolo-system

# 运行安装脚本
chmod +x scripts/install_jetson.sh
./scripts/install_jetson.sh
```

#### 方法二：手动安装
```bash
# 克隆仓库
git clone https://github.com/your-username/realsense-yolo-system.git
cd realsense-yolo-system

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

#### 方法三：Docker安装
```bash
# 构建Docker镜像
docker build -t realsense-yolo-system .

# 运行容器
docker run --privileged -v /dev/bus/usb:/dev/bus/usb \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -e DISPLAY=$DISPLAY \
           realsense-yolo-system
```

### 2. 连接设备

1. 将Intel RealSense D435i相机连接到USB 3.0端口
2. 验证设备连接：
   ```bash
   python -c "from src.camera_manager import discover_realsense_devices; print(discover_realsense_devices())"
   ```

### 3. 运行系统

#### 基础用法
```bash
# 启动系统
python main.py

# 或使用命令行工具
realsense-yolo
```

#### 高级用法
```bash
# 自定义配置
python main.py --model yolov8n.pt --confidence 0.6

# 保存结果
python main.py --save-results --output-dir results/

# 性能测试
python main.py --benchmark --benchmark-duration 60

# 查看帮助
python main.py --help
```

## 📖 使用说明

### 基本操作

运行系统后，将显示实时的检测界面：

- **左侧**: RGB图像 + 检测结果
- **右侧**: 深度图像 + 检测结果

### 键盘控制

- `q` 或 `ESC`: 退出程序
- `s`: 保存当前帧
- `r`: 重置目标跟踪
- `d`: 切换深度显示
- `c`: 切换彩色显示
- `i`: 切换信息显示
- `SPACE`: 暂停/继续

### 配置选项

主要配置文件位于 `config/config.py`，包含：

- **相机配置**: 分辨率、帧率、深度范围等
- **YOLO配置**: 模型路径、置信度阈值、IoU阈值等
- **深度配置**: 深度处理参数、滤波器设置等
- **可视化配置**: 显示选项、颜色设置等

## 🔧 API 使用

### 基础API示例

```python
from src.camera_manager import CameraManager
from src.object_detector import ObjectDetector
from src.data_fusion import DataFusion
from config.config import CAMERA_CONFIG, YOLO_CONFIG, FUSION_CONFIG

# 初始化组件
camera = CameraManager(CAMERA_CONFIG)
detector = ObjectDetector(YOLO_CONFIG)
fusion = DataFusion(FUSION_CONFIG)

# 初始化
camera.initialize()
detector.initialize()

# 开始检测
camera.start_streaming()

while True:
    # 获取帧数据
    frame_data = camera.get_frame()
    
    # 目标检测
    detections = detector.detect(frame_data.color_image)
    
    # 深度融合
    fused_results = fusion.fuse_detections_with_depth(
        detections, frame_data.depth_image, camera.intrinsics.to_dict()
    )
    
    # 处理结果
    for result in fused_results:
        print(f"检测到: {result.detection.class_name}")
        print(f"置信度: {result.detection.confidence:.2f}")
        print(f"距离: {result.distance_from_camera:.2f}m")
        print(f"3D位置: {result.world_position}")
```

### 高级API示例

```python
from src.camera_manager import CameraManager
from src.object_detector import ObjectDetector
from src.data_fusion import DataFusion
from src.visualizer import Visualizer

# 创建系统类
class RealSenseYOLOSystem:
    def __init__(self):
        self.camera = CameraManager(CAMERA_CONFIG)
        self.detector = ObjectDetector(YOLO_CONFIG)
        self.fusion = DataFusion(FUSION_CONFIG)
        self.visualizer = Visualizer(VISUALIZATION_CONFIG)
    
    def process_frame(self, frame_data):
        # 处理单帧
        detections = self.detector.detect(frame_data.color_image)
        fused_results = self.fusion.fuse_detections_with_depth(
            detections, frame_data.depth_image, self.camera.intrinsics.to_dict()
        )
        return fused_results

# 使用系统
system = RealSenseYOLOSystem()
system.camera.initialize()
system.detector.initialize()
# ... 继续处理
```

## 🛠️ 开发指南

### 项目结构

```
realsense-yolo-system/
├── src/                    # 源代码
│   ├── camera_manager.py   # 相机管理
│   ├── object_detector.py  # 目标检测
│   ├── depth_estimator.py  # 深度估计
│   ├── data_fusion.py      # 数据融合
│   ├── visualizer.py       # 可视化
│   └── utils.py            # 工具函数
├── config/                 # 配置文件
│   └── config.py          # 主配置
├── tests/                  # 测试代码
├── examples/              # 示例代码
├── scripts/               # 脚本文件
├── docs/                  # 文档
├── main.py                # 主程序
├── requirements.txt       # 依赖列表
├── setup.py              # 安装脚本
├── Dockerfile            # Docker配置
└── README.md             # 说明文档
```

### 添加新功能

1. **添加新的检测模型**:
   ```python
   # 在 object_detector.py 中添加新模型支持
   def load_custom_model(self, model_path):
       # 实现自定义模型加载逻辑
       pass
   ```

2. **添加新的滤波器**:
   ```python
   # 在 depth_estimator.py 中添加新滤波器
   def apply_custom_filter(self, depth_image):
       # 实现自定义滤波逻辑
       pass
   ```

3. **添加新的可视化功能**:
   ```python
   # 在 visualizer.py 中添加新的可视化方法
   def create_custom_visualization(self, data):
       # 实现自定义可视化逻辑
       pass
   ```

### 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_camera_manager.py

# 运行覆盖率测试
pytest --cov=src tests/

# 运行性能测试
python main.py --benchmark
```

### 代码风格

项目使用以下代码风格工具：

```bash
# 格式化代码
black src/ config/ main.py

# 检查代码风格
flake8 src/ config/

# 排序导入
isort src/ config/ main.py
```

## 📊 性能指标

### 典型性能（Jetson Orin Nano）

| 配置 | 分辨率 | FPS | 延迟 | GPU使用率 |
|------|--------|-----|------|-----------|
| YOLOv8n | 640x480 | 25-30 | 35ms | 60% |
| YOLOv8s | 640x480 | 20-25 | 45ms | 75% |
| YOLOv8m | 640x480 | 15-20 | 65ms | 85% |

### 优化建议

1. **提高帧率**: 使用较小的YOLO模型 (YOLOv8n)
2. **提高精度**: 使用较大的YOLO模型 (YOLOv8m/l)
3. **降低延迟**: 减少深度处理滤波器
4. **节省内存**: 降低输入分辨率

## 🔧 故障排除

### 常见问题

1. **相机未检测到**
   ```bash
   # 检查USB连接
   lsusb | grep Intel
   
   # 检查权限
   sudo chmod 666 /dev/bus/usb/*/*
   ```

2. **帧率过低**
   - 检查USB 3.0连接
   - 降低分辨率或帧率
   - 使用较小的YOLO模型

3. **深度数据无效**
   - 检查光照条件
   - 调整深度范围设置
   - 清洁相机镜头

4. **GPU内存不足**
   ```python
   # 在配置中降低批处理大小
   YOLO_CONFIG['batch_size'] = 1
   ```

### 调试模式

```bash
# 启用详细日志
python main.py --verbose

# 保存调试信息
python main.py --save-results --output-dir debug/
```

## 📈 扩展功能

### 支持的扩展

1. **多相机支持**: 同时使用多个RealSense相机
2. **ROS集成**: 发布ROS话题和服务
3. **Web界面**: 基于Flask的Web控制界面
4. **数据记录**: 记录检测数据用于分析
5. **报警系统**: 基于距离和目标类型的报警

### 第三方集成

- **ROS/ROS2**: 完整的ROS节点支持
- **OpenCV**: 高级图像处理功能
- **Open3D**: 3D点云处理和可视化
- **TensorBoard**: 性能监控和可视化

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 开发环境设置

```bash
# 克隆开发版本
git clone https://github.com/your-username/realsense-yolo-system.git
cd realsense-yolo-system

# 安装开发依赖
pip install -e ".[dev]"

# 安装pre-commit钩子
pre-commit install
```

## 📝 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [Intel RealSense](https://github.com/IntelRealSense/librealsense) - 深度相机SDK
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - 目标检测模型
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 📞 联系方式

- 项目主页: [https://github.com/your-username/realsense-yolo-system](https://github.com/your-username/realsense-yolo-system)
- 问题报告: [https://github.com/your-username/realsense-yolo-system/issues](https://github.com/your-username/realsense-yolo-system/issues)
- 邮箱: developer@example.com

## 🗺️ 路线图

- [ ] 支持更多YOLO模型版本
- [ ] 添加目标分割功能
- [ ] 实现实时SLAM
- [ ] 移动端应用支持
- [ ] 云端部署方案
- [ ] 边缘计算优化

---

**如果这个项目对您有帮助，请考虑给它一个 ⭐️！**