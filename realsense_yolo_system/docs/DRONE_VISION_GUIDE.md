# 无人机视觉跟踪系统使用指南

## 概述

这是一个基于Intel RealSense D435i相机、YOLO目标检测和PX4无人机的视觉跟踪系统。系统提供了简单易用的API，可以实现：

- 实时目标检测和跟踪
- 无人机自主跟随目标
- 深度估计和3D定位
- 适配Jetson Orin Nano的性能优化

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    统一无人机视觉API                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   相机管理   │  │   目标检测   │  │   数据融合   │         │
│  │ RealSense   │  │    YOLO     │  │  深度+检测   │         │
│  │   D435i     │  │             │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │   跟踪控制   │  │   无人机控制 │                          │
│  │ Multi-Track │  │  MAVSDK-PX4 │                          │
│  │   算法      │  │             │                          │
│  └─────────────┘  └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 环境配置

#### 基础依赖
```bash
# 安装Python依赖
pip install -r requirements.txt

# 针对Jetson Orin Nano的特殊配置
# PyTorch (ARM64版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# RealSense SDK
sudo apt-get update
sudo apt-get install librealsense2-dkms
sudo apt-get install librealsense2-utils
sudo apt-get install librealsense2-dev
```

#### PX4设置
```bash
# 如果使用真实无人机，确保PX4固件正确配置
# 如果使用仿真，启动Gazebo仿真
make px4_sitl gazebo
```

### 2. 基本使用

#### 仅视觉模式
```python
from src.unified_drone_vision_api import UnifiedDroneVisionAPI

# 配置系统
config = {
    'camera': {
        'width': 640,
        'height': 480,
        'fps': 30
    },
    'yolo': {
        'model_path': 'yolov8n.pt',
        'device': 'cuda',  # 或 'cpu'
        'confidence_threshold': 0.5
    },
    'tracking': {
        'tracker_type': 'csrt',
        'max_lost_frames': 15
    }
}

# 初始化API
api = UnifiedDroneVisionAPI(config)

# 启动系统
api.initialize()
api.start_vision_system()

# 开始跟踪 (x, y, w, h)
target_bbox = (100, 100, 200, 200)
api.start_tracking(target_bbox)
```

#### 无人机跟随模式
```python
import asyncio
from src.unified_drone_vision_api import UnifiedDroneVisionAPI, FollowingParameters

async def drone_follow_demo():
    # 配置包含无人机参数
    config = {
        'camera': {...},
        'yolo': {...},
        'tracking': {...},
        'drone': {
            'max_speed': 3.0,
            'max_altitude': 25.0,
            'follow_distance': 8.0
        }
    }
    
    api = UnifiedDroneVisionAPI(config)
    
    # 初始化并连接无人机
    api.initialize()
    await api.connect_drone("udp://:14540")
    api.start_vision_system()
    
    # 无人机准备
    await api.drone_controller.arm()
    await api.drone_controller.takeoff(altitude=5.0)
    
    # 开始跟随
    target_bbox = (100, 100, 200, 200)
    follow_params = FollowingParameters(
        target_distance=6.0,
        max_speed=2.0,
        min_confidence=0.4
    )
    
    await api.start_following(target_bbox, follow_params)
    
    # 等待跟随完成或手动停止
    await asyncio.sleep(60)
    await api.stop_following()
    await api.drone_controller.land()

# 运行演示
asyncio.run(drone_follow_demo())
```

### 3. 交互式演示

运行完整的交互式演示：

```bash
cd realsense_yolo_system
python examples/drone_vision_demo.py
```

#### 操作说明：
- **鼠标拖拽**: 选择跟踪目标
- **'t'**: 开始跟踪选中目标
- **'f'**: 开始跟随目标 (需要无人机连接)
- **'s'**: 停止跟踪/跟随
- **'a'**: 解锁无人机
- **'u'**: 起飞
- **'l'**: 降落
- **'e'**: 紧急停止
- **'q'**: 退出程序

## 高级配置

### 跟踪算法选择

系统支持多种跟踪算法，针对Jetson Orin Nano推荐配置：

```python
tracking_config = {
    'tracker_type': 'csrt',  # 推荐：准确性高，适合实时跟踪
    # 其他选项：'kcf', 'medianflow', 'mosse', 'boosting', 'tld'
    
    'max_lost_frames': 15,      # 最大丢失帧数
    'confidence_threshold': 0.3, # 置信度阈值
    'search_radius': 150,       # 搜索半径
    'velocity_alpha': 0.8       # 速度滤波系数
}
```

### 跟随参数优化

```python
follow_params = FollowingParameters(
    target_distance=6.0,    # 目标距离(米)
    max_speed=2.0,         # 最大速度(m/s)
    min_confidence=0.4,    # 最小置信度
    height_offset=0.0,     # 高度偏移(米)
    angle_offset=0.0       # 角度偏移(度)
)
```

### 性能优化

#### Jetson Orin Nano优化
```python
# YOLO模型配置
yolo_config = {
    'model_path': 'yolov8n.pt',  # 使用nano版本
    'device': 'cuda',
    'half': True,                # 使用半精度
    'imgsz': 416,               # 降低输入尺寸
    'max_det': 50               # 限制检测数量
}

# 相机配置
camera_config = {
    'width': 640,
    'height': 480,
    'fps': 30,
    'buffer_size': 3,           # 减少缓冲区大小
    'enable_filters': True      # 启用深度滤波
}
```

## API参考

### UnifiedDroneVisionAPI

#### 主要方法

```python
class UnifiedDroneVisionAPI:
    def __init__(self, config: Dict[str, Any])
    def initialize() -> bool
    def start_vision_system() -> bool
    def stop_vision_system()
    
    async def connect_drone(system_address: str) -> bool
    async def disconnect_drone()
    
    def start_tracking(target_bbox: Tuple[int, int, int, int], target_id: int = 0) -> bool
    def stop_tracking()
    
    async def start_following(target_bbox: Tuple[int, int, int, int], 
                             follow_params: FollowingParameters = None) -> bool
    async def stop_following()
    
    async def emergency_stop()
    
    def get_system_status() -> Dict[str, Any]
    def get_latest_result() -> Dict[str, Any]
    
    def register_callback(event: str, callback: Callable)
```

#### 事件回调

```python
# 注册事件回调
api.register_callback('on_detection', lambda detections: print(f"检测到{len(detections)}个目标"))
api.register_callback('on_tracking_update', lambda target: print(f"跟踪更新: {target.confidence}"))
api.register_callback('on_target_lost', lambda: print("目标丢失"))
api.register_callback('on_follow_start', lambda params: print("开始跟随"))
api.register_callback('on_follow_stop', lambda: print("停止跟随"))
api.register_callback('on_emergency', lambda: print("紧急停止"))
```

## 故障排除

### 常见问题

1. **相机连接失败**
   ```bash
   # 检查RealSense设备
   rs-enumerate-devices
   
   # 重新插拔USB连接
   sudo realsense-viewer
   ```

2. **无人机连接失败**
   ```bash
   # 检查MAVLink连接
   mavproxy.py --master=udp:127.0.0.1:14540
   
   # 检查PX4仿真
   make px4_sitl gazebo
   ```

3. **YOLO模型加载失败**
   ```bash
   # 下载预训练模型
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   
   # 检查CUDA支持
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **跟踪性能问题**
   - 降低图像分辨率
   - 使用更快的跟踪算法 (如 'mosse')
   - 减少检测频率

### 性能监控

```python
# 获取系统状态
status = api.get_system_status()
print(f"系统模式: {status['mode']}")
print(f"跟踪状态: {status['tracking_status']}")
print(f"无人机状态: {status['drone_status']}")
```

## 扩展开发

### 自定义跟踪算法

```python
class CustomTracker:
    def __init__(self, config):
        self.config = config
    
    def init(self, frame, bbox):
        # 初始化跟踪器
        pass
    
    def update(self, frame):
        # 更新跟踪
        return success, bbox
```

### 自定义跟随策略

```python
class CustomFollowStrategy:
    def calculate_follow_command(self, target, camera_info):
        # 计算跟随命令
        return {
            'forward': 0.0,
            'right': 0.0,
            'down': 0.0,
            'yaw_rate': 0.0
        }
```

## 安全注意事项

1. **飞行前检查**
   - 确保电池电量充足
   - 检查GPS信号强度
   - 验证遥控器连接

2. **飞行期间**
   - 始终保持视线内飞行
   - 准备随时手动接管
   - 避免在人群上方飞行

3. **紧急情况**
   - 按'e'键紧急停止
   - 使用遥控器紧急开关
   - 确保着陆区域安全

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 贡献

欢迎提交问题和拉取请求。请遵循代码规范和测试要求。

## 联系方式

如有问题，请联系：[your-email@example.com]