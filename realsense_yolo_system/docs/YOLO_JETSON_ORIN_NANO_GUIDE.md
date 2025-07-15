# YOLO 配置指南 - Jetson Orin Nano 专用

## 目录
1. [概述](#概述)
2. [硬件需求](#硬件需求)
3. [系统环境准备](#系统环境准备)
4. [Ultralytics YOLO 安装](#ultralytics-yolo-安装)
5. [模型选择与优化](#模型选择与优化)
6. [配置详解](#配置详解)
7. [TensorRT 优化](#tensorrt-优化)
8. [性能调优](#性能调优)
9. [集成配置](#集成配置)
10. [故障排除](#故障排除)
11. [最佳实践](#最佳实践)

---

## 概述

本指南专为 Jetson Orin Nano 开发者设计，详细介绍如何在该平台上配置和优化 Ultralytics YOLO 模型。通过本指南，您将能够：

- 在 Jetson Orin Nano 上成功部署 YOLO 模型
- 实现最优的推理性能
- 集成到现有的 RealSense 视觉系统中
- 解决常见的配置和性能问题

## 硬件需求

### 最低要求
- **设备**: NVIDIA Jetson Orin Nano (8GB)
- **存储**: 32GB+ MicroSD 卡或 NVMe SSD
- **相机**: Intel RealSense D435i 或兼容设备
- **电源**: 适配器或电池包（至少 15W）

### 推荐配置
- **设备**: Jetson Orin Nano Developer Kit
- **存储**: 128GB+ NVMe SSD
- **散热**: 主动散热风扇
- **电源**: 19V 3.42A 适配器

## 系统环境准备

### 1. JetPack 安装

```bash
# 检查 JetPack 版本
cat /etc/nv_tegra_release

# 推荐版本：JetPack 5.1.2 或更高
# 如需升级，请访问 NVIDIA 官方文档
```

### 2. 系统优化

```bash
# 设置最大性能模式
sudo nvpmodel -m 0
sudo jetson_clocks

# 查看当前功耗模式
sudo nvpmodel -q

# 检查 GPU 状态
sudo tegrastats
```

### 3. 基础依赖安装

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装必要的依赖
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libopencv-dev \
    python3-opencv \
    libfreetype6-dev \
    pkg-config \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran

# 创建虚拟环境
python3 -m venv ~/yolo_env
source ~/yolo_env/bin/activate

# 升级 pip
pip install --upgrade pip setuptools wheel
```

## Ultralytics YOLO 安装

### 1. PyTorch 安装（Jetson 专用）

```bash
# 激活虚拟环境
source ~/yolo_env/bin/activate

# 安装 PyTorch（针对 Jetson 优化的版本）
# 对于 JetPack 5.1.2+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或者使用 NVIDIA 提供的轮子
wget https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl -O torch-2.0.0-cp38-cp38-linux_aarch64.whl
pip install torch-2.0.0-cp38-cp38-linux_aarch64.whl

# 验证 PyTorch 安装
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}')"
```

### 2. Ultralytics 安装

```bash
# 安装 Ultralytics YOLO
pip install ultralytics

# 安装额外依赖
pip install \
    opencv-python \
    pillow \
    pyyaml \
    requests \
    scipy \
    tqdm \
    tensorboard \
    pandas \
    seaborn \
    psutil \
    thop

# 验证安装
python3 -c "import ultralytics; print(f'Ultralytics版本: {ultralytics.__version__}')"
```

### 3. 首次运行测试

```bash
# 创建测试脚本
cat > test_yolo.py << 'EOF'
from ultralytics import YOLO
import cv2
import numpy as np

# 加载模型（会自动下载）
model = YOLO('yolov8n.pt')

# 创建测试图像
test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# 运行推理
results = model(test_image)
print(f"模型加载成功！检测到 {len(results[0].boxes)} 个目标")
print(f"推理设备: {model.device}")
EOF

python3 test_yolo.py
```

## 模型选择与优化

### 1. 模型对比表

| 模型 | 参数量 | 模型大小 | mAP | Jetson推理速度 | 推荐场景 |
|------|--------|----------|-----|----------------|----------|
| YOLOv8n | 3.2M | 6.2MB | 37.3 | ~25 FPS | 实时应用 |
| YOLOv8s | 11.2M | 21.5MB | 44.9 | ~18 FPS | 平衡性能 |
| YOLOv8m | 25.9M | 49.7MB | 50.2 | ~12 FPS | 高精度 |
| YOLOv8l | 43.7M | 83.7MB | 52.9 | ~8 FPS | 最高精度 |
| YOLOv8x | 68.2M | 130.5MB | 53.9 | ~5 FPS | 离线处理 |

### 2. 模型下载与配置

```bash
# 下载推荐模型
python3 -c "
from ultralytics import YOLO
import os

# 创建模型目录
os.makedirs('models', exist_ok=True)

# 下载不同大小的模型
models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
for model_name in models:
    print(f'下载模型: {model_name}')
    model = YOLO(model_name)
    model.save(f'models/{model_name}')
    print(f'模型已保存到: models/{model_name}')
"
```

### 3. 自定义模型训练

```bash
# 创建训练脚本
cat > train_custom.py << 'EOF'
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 训练配置
train_config = {
    'data': 'your_dataset.yaml',  # 数据集配置
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,  # Jetson Orin Nano 推荐批次大小
    'workers': 4,
    'device': 0,  # 使用 GPU
    'patience': 50,
    'save_period': 10,
    'cache': True,
    'amp': True,  # 混合精度训练
    'project': 'jetson_training',
    'name': 'custom_yolo'
}

# 开始训练
results = model.train(**train_config)
EOF
```

## 配置详解

### 1. 基础配置文件

```python
# config/yolo_jetson_config.py
import os
from pathlib import Path

class YOLOJetsonConfig:
    """Jetson Orin Nano 专用 YOLO 配置"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
    
    # 模型配置
    MODEL_CONFIG = {
        # 性能优先配置
        'performance': {
            'model_path': 'models/yolov8n.pt',
            'model_type': 'yolov8n',
            'imgsz': 416,  # 降低输入尺寸提高速度
            'conf': 0.4,   # 降低置信度阈值
            'iou': 0.45,
            'max_det': 50,
            'half': True,  # 启用半精度
            'device': 'cuda:0',
            'verbose': False,
            'workers': 2,
            'batch': 1,
        },
        
        # 精度优先配置
        'accuracy': {
            'model_path': 'models/yolov8s.pt',
            'model_type': 'yolov8s',
            'imgsz': 640,
            'conf': 0.5,
            'iou': 0.45,
            'max_det': 100,
            'half': True,
            'device': 'cuda:0',
            'verbose': False,
            'workers': 4,
            'batch': 1,
        },
        
        # 平衡配置
        'balanced': {
            'model_path': 'models/yolov8n.pt',
            'model_type': 'yolov8n',
            'imgsz': 512,
            'conf': 0.45,
            'iou': 0.45,
            'max_det': 75,
            'half': True,
            'device': 'cuda:0',
            'verbose': False,
            'workers': 3,
            'batch': 1,
        },
        
        # 低功耗配置
        'low_power': {
            'model_path': 'models/yolov8n.pt',
            'model_type': 'yolov8n',
            'imgsz': 320,
            'conf': 0.35,
            'iou': 0.45,
            'max_det': 30,
            'half': True,
            'device': 'cuda:0',
            'verbose': False,
            'workers': 1,
            'batch': 1,
        }
    }
    
    # TensorRT 配置
    TENSORRT_CONFIG = {
        'enabled': True,
        'workspace': 4,  # GB
        'max_batch_size': 1,
        'fp16': True,
        'int8': False,  # 需要校准数据集
        'dynamic_axes': True,
        'simplify': True,
        'opset_version': 11,
    }
    
    # 系统优化配置
    SYSTEM_CONFIG = {
        'cpu_threads': 4,
        'gpu_memory_fraction': 0.8,
        'allow_growth': True,
        'enable_xla': True,
        'mixed_precision': True,
    }
    
    @classmethod
    def get_config(cls, mode='balanced'):
        """获取指定模式的配置"""
        config = cls()
        model_config = cls.MODEL_CONFIG.get(mode, cls.MODEL_CONFIG['balanced'])
        
        return {
            'model': model_config,
            'tensorrt': cls.TENSORRT_CONFIG,
            'system': cls.SYSTEM_CONFIG,
            'mode': mode
        }
```

### 2. 动态配置管理

```python
# config/dynamic_config.py
import psutil
import subprocess
import json
from pathlib import Path

class DynamicYOLOConfig:
    """动态 YOLO 配置管理器"""
    
    def __init__(self):
        self.config_file = Path("config/yolo_runtime_config.json")
        self.load_config()
    
    def detect_hardware_capabilities(self):
        """检测硬件能力"""
        capabilities = {}
        
        # 内存信息
        memory = psutil.virtual_memory()
        capabilities['memory_total'] = memory.total
        capabilities['memory_available'] = memory.available
        
        # GPU 信息
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,nounits,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                capabilities['gpu_name'] = gpu_info[0]
                capabilities['gpu_memory_total'] = int(gpu_info[1])
                capabilities['gpu_memory_free'] = int(gpu_info[2])
        except:
            capabilities['gpu_available'] = False
        
        # CPU 信息
        capabilities['cpu_count'] = psutil.cpu_count()
        capabilities['cpu_freq'] = psutil.cpu_freq().current if psutil.cpu_freq() else 0
        
        return capabilities
    
    def get_adaptive_config(self):
        """根据硬件能力自动调整配置"""
        caps = self.detect_hardware_capabilities()
        
        # 基础配置
        config = {
            'model_path': 'models/yolov8n.pt',
            'imgsz': 416,
            'conf': 0.45,
            'iou': 0.45,
            'max_det': 50,
            'half': True,
            'device': 'cuda:0',
            'batch': 1,
            'workers': 2,
        }
        
        # 根据可用内存调整
        if caps.get('memory_available', 0) > 6 * 1024**3:  # 6GB+
            config['imgsz'] = 512
            config['max_det'] = 100
            config['workers'] = 4
        elif caps.get('memory_available', 0) < 4 * 1024**3:  # <4GB
            config['imgsz'] = 320
            config['max_det'] = 30
            config['workers'] = 1
        
        # 根据 GPU 内存调整
        if caps.get('gpu_memory_free', 0) > 4000:  # 4GB+
            config['model_path'] = 'models/yolov8s.pt'
            config['imgsz'] = 640
        elif caps.get('gpu_memory_free', 0) < 2000:  # <2GB
            config['half'] = True
            config['imgsz'] = 320
        
        return config
    
    def save_config(self, config):
        """保存配置到文件"""
        self.config_file.parent.mkdir(exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self):
        """从文件加载配置"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
```

## TensorRT 优化

### 1. TensorRT 安装

```bash
# 检查 TensorRT 版本
python3 -c "import tensorrt; print(f'TensorRT版本: {tensorrt.__version__}')"

# 如果没有安装，使用 JetPack 安装
sudo apt install nvidia-tensorrt

# 安装 Python 绑定
pip install tensorrt
```

### 2. 模型转换脚本

```python
# scripts/tensorrt_convert.py
from ultralytics import YOLO
import torch
import tensorrt as trt
import numpy as np
import os

class TensorRTConverter:
    """TensorRT 模型转换器"""
    
    def __init__(self, model_path, output_path):
        self.model_path = model_path
        self.output_path = output_path
        self.model = None
    
    def convert_to_tensorrt(self, 
                           imgsz=416, 
                           batch_size=1,
                           workspace=4,
                           fp16=True,
                           int8=False,
                           calibration_data=None):
        """转换模型到 TensorRT"""
        
        print(f"开始转换模型: {self.model_path}")
        
        # 加载 YOLO 模型
        self.model = YOLO(self.model_path)
        
        # 导出配置
        export_config = {
            'format': 'engine',
            'imgsz': imgsz,
            'batch': batch_size,
            'workspace': workspace,
            'verbose': True,
            'half': fp16,
            'int8': int8,
            'dynamic': False,
            'simplify': True,
            'opset': 11,
        }
        
        # 如果使用 INT8，需要校准数据
        if int8 and calibration_data:
            export_config['data'] = calibration_data
        
        # 执行转换
        try:
            engine_path = self.model.export(**export_config)
            print(f"TensorRT 模型已生成: {engine_path}")
            return engine_path
        except Exception as e:
            print(f"转换失败: {e}")
            return None
    
    def benchmark_model(self, engine_path, test_runs=100):
        """基准测试"""
        import time
        
        # 创建测试数据
        test_input = np.random.rand(1, 3, 416, 416).astype(np.float32)
        
        # 加载 TensorRT 模型
        model = YOLO(engine_path)
        
        # 预热
        for _ in range(10):
            model(test_input)
        
        # 基准测试
        start_time = time.time()
        for _ in range(test_runs):
            results = model(test_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / test_runs
        fps = 1.0 / avg_time
        
        print(f"平均推理时间: {avg_time*1000:.2f}ms")
        print(f"FPS: {fps:.2f}")
        
        return fps, avg_time

# 使用示例
if __name__ == "__main__":
    converter = TensorRTConverter(
        model_path="models/yolov8n.pt",
        output_path="models/yolov8n_tensorrt.engine"
    )
    
    # 转换模型
    engine_path = converter.convert_to_tensorrt(
        imgsz=416,
        batch_size=1,
        workspace=4,
        fp16=True,
        int8=False
    )
    
    # 基准测试
    if engine_path:
        fps, avg_time = converter.benchmark_model(engine_path)
```

### 3. 自动优化脚本

```bash
# scripts/auto_optimize.sh
#!/bin/bash

echo "开始自动优化流程..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export TRT_LOGGER_VERBOSITY=1

# 创建优化目录
mkdir -p optimized_models
mkdir -p benchmark_results

# 模型列表
models=("yolov8n" "yolov8s")
sizes=(320 416 512 640)

for model in "${models[@]}"; do
    for size in "${sizes[@]}"; do
        echo "优化模型: ${model}, 尺寸: ${size}"
        
        # 转换为 TensorRT
        python3 scripts/tensorrt_convert.py \
            --model "models/${model}.pt" \
            --output "optimized_models/${model}_${size}_fp16.engine" \
            --imgsz ${size} \
            --fp16 \
            --workspace 4
        
        # 基准测试
        python3 scripts/benchmark.py \
            --model "optimized_models/${model}_${size}_fp16.engine" \
            --output "benchmark_results/${model}_${size}_results.json"
    done
done

echo "优化完成！"
```

## 性能调优

### 1. 系统级优化

```bash
# scripts/system_optimize.sh
#!/bin/bash

echo "开始系统优化..."

# 设置最大性能模式
sudo nvpmodel -m 0
sudo jetson_clocks

# 优化 CPU 调度
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 优化内存
echo 1 | sudo tee /proc/sys/vm/overcommit_memory
echo 50 | sudo tee /proc/sys/vm/swappiness

# 优化 GPU
sudo sh -c 'echo 1 > /sys/kernel/debug/tegra_fan/target_pwm'

# 设置环境变量
export CUDA_CACHE_MAXSIZE=2147483648
export CUDA_CACHE_PATH=/tmp/cuda_cache
export TENSORRT_CACHE_PATH=/tmp/tensorrt_cache

# 创建缓存目录
mkdir -p /tmp/cuda_cache
mkdir -p /tmp/tensorrt_cache

echo "系统优化完成！"
```

### 2. 内存优化

```python
# utils/memory_optimizer.py
import gc
import torch
import psutil
import threading
import time

class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self, check_interval=30):
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
    
    def optimize_torch_memory(self):
        """优化 PyTorch 内存使用"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 强制垃圾回收
        gc.collect()
    
    def get_memory_usage(self):
        """获取内存使用情况"""
        # 系统内存
        sys_mem = psutil.virtual_memory()
        
        # GPU 内存
        gpu_mem = {}
        if torch.cuda.is_available():
            gpu_mem = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated(),
                'max_reserved': torch.cuda.max_memory_reserved(),
            }
        
        return {
            'system': {
                'total': sys_mem.total,
                'available': sys_mem.available,
                'used': sys_mem.used,
                'percent': sys_mem.percent,
            },
            'gpu': gpu_mem
        }
    
    def start_monitoring(self):
        """开始内存监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            memory_info = self.get_memory_usage()
            
            # 检查是否需要清理
            if memory_info['system']['percent'] > 85:
                print("内存使用率过高，执行清理...")
                self.optimize_torch_memory()
            
            if torch.cuda.is_available():
                gpu_usage = memory_info['gpu']['allocated'] / (1024**3)  # GB
                if gpu_usage > 6:  # 6GB
                    print("GPU 内存使用过高，执行清理...")
                    self.optimize_torch_memory()
            
            time.sleep(self.check_interval)
    
    def get_recommendations(self):
        """获取优化建议"""
        memory_info = self.get_memory_usage()
        recommendations = []
        
        # 系统内存建议
        if memory_info['system']['percent'] > 80:
            recommendations.append("系统内存使用率过高，建议降低图像分辨率或批次大小")
        
        # GPU 内存建议
        if torch.cuda.is_available():
            gpu_usage = memory_info['gpu']['allocated'] / (1024**3)
            if gpu_usage > 6:
                recommendations.append("GPU 内存使用过高，建议启用半精度推理或使用更小的模型")
        
        return recommendations
```

### 3. 性能监控

```python
# utils/performance_monitor.py
import time
import threading
import json
import psutil
from collections import deque
import matplotlib.pyplot as plt

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, max_samples=1000):
        self.max_samples = max_samples
        self.fps_history = deque(maxlen=max_samples)
        self.inference_times = deque(maxlen=max_samples)
        self.memory_usage = deque(maxlen=max_samples)
        self.gpu_usage = deque(maxlen=max_samples)
        
        self.monitoring = False
        self.monitor_thread = None
        self.last_frame_time = time.time()
    
    def update_frame(self, inference_time):
        """更新帧信息"""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        fps = 1.0 / frame_time if frame_time > 0 else 0
        
        self.fps_history.append(fps)
        self.inference_times.append(inference_time)
        self.last_frame_time = current_time
    
    def start_monitoring(self):
        """开始系统监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_system(self):
        """监控系统资源"""
        while self.monitoring:
            # 内存使用
            mem = psutil.virtual_memory()
            self.memory_usage.append(mem.percent)
            
            # GPU 使用（如果可用）
            try:
                import nvidia_ml_py3 as nvml
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = nvml.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_usage.append(gpu_util.gpu)
            except:
                self.gpu_usage.append(0)
            
            time.sleep(1)
    
    def get_stats(self):
        """获取性能统计"""
        if not self.fps_history:
            return {}
        
        return {
            'fps': {
                'current': self.fps_history[-1] if self.fps_history else 0,
                'average': sum(self.fps_history) / len(self.fps_history),
                'min': min(self.fps_history),
                'max': max(self.fps_history),
            },
            'inference_time': {
                'current': self.inference_times[-1] if self.inference_times else 0,
                'average': sum(self.inference_times) / len(self.inference_times),
                'min': min(self.inference_times),
                'max': max(self.inference_times),
            },
            'memory_usage': {
                'current': self.memory_usage[-1] if self.memory_usage else 0,
                'average': sum(self.memory_usage) / len(self.memory_usage),
            },
            'gpu_usage': {
                'current': self.gpu_usage[-1] if self.gpu_usage else 0,
                'average': sum(self.gpu_usage) / len(self.gpu_usage),
            }
        }
    
    def save_report(self, filepath):
        """保存性能报告"""
        stats = self.get_stats()
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'stats': stats,
            'raw_data': {
                'fps_history': list(self.fps_history),
                'inference_times': list(self.inference_times),
                'memory_usage': list(self.memory_usage),
                'gpu_usage': list(self.gpu_usage),
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def plot_performance(self, save_path='performance_chart.png'):
        """绘制性能图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # FPS 图表
        axes[0, 0].plot(list(self.fps_history))
        axes[0, 0].set_title('FPS Over Time')
        axes[0, 0].set_ylabel('FPS')
        axes[0, 0].grid(True)
        
        # 推理时间图表
        axes[0, 1].plot(list(self.inference_times))
        axes[0, 1].set_title('Inference Time Over Time')
        axes[0, 1].set_ylabel('Time (s)')
        axes[0, 1].grid(True)
        
        # 内存使用图表
        axes[1, 0].plot(list(self.memory_usage))
        axes[1, 0].set_title('Memory Usage Over Time')
        axes[1, 0].set_ylabel('Usage (%)')
        axes[1, 0].grid(True)
        
        # GPU 使用图表
        axes[1, 1].plot(list(self.gpu_usage))
        axes[1, 1].set_title('GPU Usage Over Time')
        axes[1, 1].set_ylabel('Usage (%)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
```

## 集成配置

### 1. 现有系统集成

```python
# integration/yolo_jetson_integration.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
import torch
import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional

# 导入现有系统组件
from src.object_detector import ObjectDetector, DetectionResult
from config.yolo_jetson_config import YOLOJetsonConfig
from utils.memory_optimizer import MemoryOptimizer
from utils.performance_monitor import PerformanceMonitor

class JetsonYOLODetector(ObjectDetector):
    """Jetson 优化的 YOLO 检测器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Jetson 专用配置
        self.jetson_config = YOLOJetsonConfig.get_config(
            config.get('jetson_mode', 'balanced')
        )
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        self.memory_optimizer = MemoryOptimizer()
        
        # TensorRT 支持
        self.tensorrt_enabled = config.get('use_tensorrt', False)
        self.tensorrt_engine_path = None
        
        logger.info("Jetson YOLO 检测器初始化完成")
    
    def initialize(self) -> bool:
        """初始化检测器"""
        try:
            # 启动内存优化
            self.memory_optimizer.start_monitoring()
            
            # 启动性能监控
            self.performance_monitor.start_monitoring()
            
            # 加载模型
            model_config = self.jetson_config['model']
            model_path = model_config['model_path']
            
            # 检查 TensorRT 引擎
            if self.tensorrt_enabled:
                tensorrt_path = model_path.replace('.pt', '_tensorrt.engine')
                if os.path.exists(tensorrt_path):
                    model_path = tensorrt_path
                    self.tensorrt_engine_path = tensorrt_path
                    logger.info(f"使用 TensorRT 引擎: {tensorrt_path}")
            
            # 加载模型
            self.model = YOLO(model_path)
            
            # 应用配置
            self.model.conf = model_config['conf']
            self.model.iou = model_config['iou']
            self.model.max_det = model_config['max_det']
            
            # 设置设备
            self.device = model_config['device']
            if self.device == 'cuda:0' and not torch.cuda.is_available():
                logger.warning("CUDA 不可用，切换到 CPU")
                self.device = 'cpu'
            
            self.model.to(self.device)
            
            # 预热模型
            self._warmup_jetson_model()
            
            self.is_initialized = True
            logger.info("Jetson YOLO 检测器初始化成功")
            
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return False
    
    def _warmup_jetson_model(self):
        """Jetson 专用模型预热"""
        try:
            model_config = self.jetson_config['model']
            imgsz = model_config['imgsz']
            
            # 创建测试输入
            dummy_input = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            
            # 预热推理
            logger.info("开始模型预热...")
            for i in range(10):
                start_time = time.time()
                self.model(dummy_input, verbose=False)
                warmup_time = time.time() - start_time
                logger.info(f"预热 {i+1}/10: {warmup_time*1000:.2f}ms")
            
            logger.info("模型预热完成")
            
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")
    
    def detect(self, image: np.ndarray, 
               filter_classes: Optional[List[str]] = None) -> List[DetectionResult]:
        """优化的检测方法"""
        if not self.is_initialized:
            return []
        
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 预处理图像
            processed_image = self._preprocess_image(image)
            
            # 运行推理
            results = self.model(processed_image, verbose=False)
            
            # 记录推理时间
            inference_time = time.time() - start_time
            
            # 更新性能监控
            self.performance_monitor.update_frame(inference_time)
            
            # 解析结果
            detections = self._parse_results(results, filter_classes)
            
            # 定期清理内存
            if len(self.inference_times) % 50 == 0:
                self.memory_optimizer.optimize_torch_memory()
            
            return detections
            
        except Exception as e:
            logger.error(f"检测失败: {e}")
            return []
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Jetson 优化的图像预处理"""
        model_config = self.jetson_config['model']
        imgsz = model_config['imgsz']
        
        # 调整图像大小
        if image.shape[:2] != (imgsz, imgsz):
            image = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        
        return image
    
    def _parse_results(self, results, filter_classes=None) -> List[DetectionResult]:
        """解析检测结果"""
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    class_name = self.model.names.get(cls_id, f"class_{cls_id}")
                    
                    if filter_classes and class_name not in filter_classes:
                        continue
                    
                    detection = DetectionResult(
                        bbox=box.tolist(),
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=class_name
                    )
                    
                    detections.append(detection)
        
        return detections
    
    def get_jetson_stats(self) -> Dict[str, Any]:
        """获取 Jetson 专用统计信息"""
        base_stats = self.get_detection_stats()
        jetson_stats = {
            'performance': self.performance_monitor.get_stats(),
            'memory_recommendations': self.memory_optimizer.get_recommendations(),
            'tensorrt_enabled': self.tensorrt_enabled,
            'tensorrt_engine': self.tensorrt_engine_path,
            'jetson_config': self.jetson_config,
        }
        
        return {**base_stats, **jetson_stats}
    
    def save_performance_report(self, filepath: str):
        """保存性能报告"""
        self.performance_monitor.save_report(filepath)
        logger.info(f"性能报告已保存: {filepath}")
    
    def cleanup(self):
        """清理资源"""
        # 停止监控
        self.performance_monitor.stop_monitoring()
        self.memory_optimizer.stop_monitoring()
        
        # 清理模型
        super().cleanup()
        
        logger.info("Jetson YOLO 检测器资源清理完成")

# 工厂函数
def create_jetson_yolo_detector(config: Dict[str, Any]) -> JetsonYOLODetector:
    """创建 Jetson YOLO 检测器"""
    return JetsonYOLODetector(config)
```

### 2. 统一 API 集成

```python
# integration/unified_api_integration.py
from src.unified_drone_vision_api import UnifiedDroneVisionAPI
from integration.yolo_jetson_integration import JetsonYOLODetector
from config.yolo_jetson_config import YOLOJetsonConfig

class JetsonUnifiedDroneVisionAPI(UnifiedDroneVisionAPI):
    """Jetson 优化的统一 API"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # 使用 Jetson 优化的检测器
        self.jetson_mode = config.get('jetson_mode', 'balanced')
        self.detector = None
    
    def initialize(self) -> bool:
        """初始化系统"""
        try:
            # 获取 Jetson 配置
            jetson_config = YOLOJetsonConfig.get_config(self.jetson_mode)
            
            # 创建 Jetson 优化的检测器
            detector_config = {
                **self.config.get('yolo', {}),
                'jetson_mode': self.jetson_mode,
                'use_tensorrt': self.config.get('use_tensorrt', False)
            }
            
            self.detector = JetsonYOLODetector(detector_config)
            
            # 初始化检测器
            if not self.detector.initialize():
                logger.error("Jetson 检测器初始化失败")
                return False
            
            # 初始化其他组件
            success = super().initialize()
            
            # 替换检测器
            if success and hasattr(self, 'object_detector'):
                self.object_detector.cleanup()
                self.object_detector = self.detector
            
            return success
            
        except Exception as e:
            logger.error(f"Jetson API 初始化失败: {e}")
            return False
    
    def get_jetson_performance_stats(self):
        """获取 Jetson 性能统计"""
        if self.detector:
            return self.detector.get_jetson_stats()
        return {}
    
    def save_jetson_performance_report(self, filepath: str):
        """保存 Jetson 性能报告"""
        if self.detector:
            self.detector.save_performance_report(filepath)
    
    def optimize_for_jetson(self, mode: str = 'balanced'):
        """动态优化 Jetson 设置"""
        if mode != self.jetson_mode:
            logger.info(f"切换 Jetson 模式: {self.jetson_mode} -> {mode}")
            self.jetson_mode = mode
            
            # 重新初始化检测器
            if self.detector:
                self.detector.cleanup()
                
                detector_config = {
                    **self.config.get('yolo', {}),
                    'jetson_mode': mode,
                    'use_tensorrt': self.config.get('use_tensorrt', False)
                }
                
                self.detector = JetsonYOLODetector(detector_config)
                self.detector.initialize()
                self.object_detector = self.detector
```

## 故障排除

### 1. 常见问题及解决方案

```bash
# 故障排除脚本
# scripts/troubleshoot.sh

#!/bin/bash

echo "=== YOLO Jetson 故障排除工具 ==="

# 检查硬件
echo "1. 检查硬件状态..."
nvidia-smi
echo "GPU 状态: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)%"

# 检查内存
echo "2. 检查内存使用..."
free -h
echo "GPU 内存: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)"

# 检查 Python 环境
echo "3. 检查 Python 环境..."
python3 --version
pip list | grep -E "(torch|ultralytics|opencv|numpy)"

# 检查 CUDA
echo "4. 检查 CUDA..."
nvcc --version
python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"

# 检查模型文件
echo "5. 检查模型文件..."
ls -la models/
du -sh models/*

# 运行测试
echo "6. 运行基础测试..."
python3 -c "
from ultralytics import YOLO
import torch
import numpy as np

try:
    model = YOLO('yolov8n.pt')
    test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = model(test_input)
    print('✓ 基础测试通过')
except Exception as e:
    print(f'✗ 基础测试失败: {e}')
"

echo "故障排除完成！"
```

### 2. 性能问题诊断

```python
# utils/performance_diagnostics.py
import subprocess
import json
import time
import torch
import numpy as np
from ultralytics import YOLO

class PerformanceDiagnostics:
    """性能诊断工具"""
    
    def __init__(self):
        self.results = {}
    
    def diagnose_all(self):
        """执行所有诊断"""
        print("开始性能诊断...")
        
        self.check_hardware()
        self.check_environment()
        self.check_model_performance()
        self.check_memory_usage()
        self.generate_recommendations()
        
        return self.results
    
    def check_hardware(self):
        """检查硬件状态"""
        print("检查硬件...")
        
        # GPU 信息
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                self.results['gpu'] = {
                    'name': gpu_info[0],
                    'utilization': int(gpu_info[1]),
                    'memory_used': int(gpu_info[2]),
                    'memory_total': int(gpu_info[3]),
                    'temperature': int(gpu_info[4])
                }
        except:
            self.results['gpu'] = {'error': 'GPU 信息获取失败'}
        
        # CPU 信息
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()
            cpu_count = cpu_info.count('processor')
            self.results['cpu'] = {'cores': cpu_count}
        except:
            self.results['cpu'] = {'error': 'CPU 信息获取失败'}
    
    def check_environment(self):
        """检查环境配置"""
        print("检查环境...")
        
        self.results['environment'] = {
            'python_version': subprocess.run(['python3', '--version'], capture_output=True, text=True).stdout.strip(),
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'torch_version': torch.__version__,
        }
        
        # 检查 Ultralytics
        try:
            import ultralytics
            self.results['environment']['ultralytics_version'] = ultralytics.__version__
        except:
            self.results['environment']['ultralytics_version'] = 'Not installed'
    
    def check_model_performance(self):
        """检查模型性能"""
        print("检查模型性能...")
        
        models_to_test = ['yolov8n.pt', 'yolov8s.pt']
        input_sizes = [320, 416, 640]
        
        for model_name in models_to_test:
            if model_name not in self.results:
                self.results[model_name] = {}
            
            try:
                model = YOLO(model_name)
                
                for size in input_sizes:
                    # 创建测试输入
                    test_input = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                    
                    # 预热
                    for _ in range(3):
                        model(test_input, verbose=False)
                    
                    # 基准测试
                    times = []
                    for _ in range(10):
                        start = time.time()
                        model(test_input, verbose=False)
                        times.append(time.time() - start)
                    
                    avg_time = np.mean(times)
                    fps = 1.0 / avg_time
                    
                    self.results[model_name][f'size_{size}'] = {
                        'avg_time': avg_time,
                        'fps': fps,
                        'std_time': np.std(times)
                    }
                    
            except Exception as e:
                self.results[model_name] = {'error': str(e)}
    
    def check_memory_usage(self):
        """检查内存使用"""
        print("检查内存使用...")
        
        # 系统内存
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            mem_total = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1])
            mem_available = int([line for line in meminfo.split('\n') if 'MemAvailable' in line][0].split()[1])
            
            self.results['memory'] = {
                'total_mb': mem_total // 1024,
                'available_mb': mem_available // 1024,
                'usage_percent': (mem_total - mem_available) / mem_total * 100
            }
        except:
            self.results['memory'] = {'error': '内存信息获取失败'}
        
        # GPU 内存
        if torch.cuda.is_available():
            self.results['gpu_memory'] = {
                'allocated_mb': torch.cuda.memory_allocated() // 1024 // 1024,
                'reserved_mb': torch.cuda.memory_reserved() // 1024 // 1024,
                'max_allocated_mb': torch.cuda.max_memory_allocated() // 1024 // 1024,
            }
    
    def generate_recommendations(self):
        """生成优化建议"""
        print("生成建议...")
        
        recommendations = []
        
        # GPU 利用率建议
        if 'gpu' in self.results and 'utilization' in self.results['gpu']:
            gpu_util = self.results['gpu']['utilization']
            if gpu_util < 50:
                recommendations.append("GPU 利用率较低，可以考虑增加批次大小或使用更大的模型")
            elif gpu_util > 95:
                recommendations.append("GPU 利用率过高，可能出现性能瓶颈，建议降低批次大小")
        
        # 内存建议
        if 'memory' in self.results and 'usage_percent' in self.results['memory']:
            mem_usage = self.results['memory']['usage_percent']
            if mem_usage > 85:
                recommendations.append("系统内存使用率过高，建议关闭其他应用程序或降低图像分辨率")
        
        # 性能建议
        best_config = None
        best_fps = 0
        
        for model_name in ['yolov8n.pt', 'yolov8s.pt']:
            if model_name in self.results:
                for size_key in self.results[model_name]:
                    if size_key.startswith('size_') and 'fps' in self.results[model_name][size_key]:
                        fps = self.results[model_name][size_key]['fps']
                        if fps > best_fps:
                            best_fps = fps
                            best_config = f"{model_name} with {size_key.replace('size_', '')}x{size_key.replace('size_', '')}"
        
        if best_config:
            recommendations.append(f"推荐配置: {best_config} (FPS: {best_fps:.1f})")
        
        self.results['recommendations'] = recommendations
    
    def save_report(self, filepath):
        """保存诊断报告"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"诊断报告已保存: {filepath}")

# 使用示例
if __name__ == "__main__":
    diagnostics = PerformanceDiagnostics()
    results = diagnostics.diagnose_all()
    diagnostics.save_report('performance_diagnostics.json')
```

### 3. 自动修复脚本

```bash
# scripts/auto_fix.sh
#!/bin/bash

echo "=== 自动修复脚本 ==="

# 检查并修复权限问题
echo "1. 检查权限..."
sudo usermod -a -G video $USER
sudo usermod -a -G dialout $USER

# 检查并修复 CUDA 环境
echo "2. 检查 CUDA 环境..."
if ! command -v nvcc &> /dev/null; then
    echo "CUDA 工具包未安装，请手动安装 JetPack"
    exit 1
fi

# 检查并修复 Python 环境
echo "3. 检查 Python 环境..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch 未安装，开始安装..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

if ! python3 -c "import ultralytics" 2>/dev/null; then
    echo "Ultralytics 未安装，开始安装..."
    pip install ultralytics
fi

# 检查并修复模型文件
echo "4. 检查模型文件..."
mkdir -p models
if [ ! -f "models/yolov8n.pt" ]; then
    echo "下载 YOLOv8n 模型..."
    python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.save('models/yolov8n.pt')
"
fi

# 检查并修复配置文件
echo "5. 检查配置文件..."
if [ ! -f "config/yolo_jetson_config.py" ]; then
    echo "配置文件缺失，请确保正确安装项目"
    exit 1
fi

# 运行基础测试
echo "6. 运行测试..."
python3 -c "
from ultralytics import YOLO
import torch
import numpy as np

try:
    # 检查 CUDA
    if not torch.cuda.is_available():
        print('警告: CUDA 不可用')
    
    # 加载模型
    model = YOLO('models/yolov8n.pt')
    
    # 运行测试
    test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = model(test_input)
    
    print('✓ 自动修复成功')
    
except Exception as e:
    print(f'✗ 自动修复失败: {e}')
    exit 1
"

echo "自动修复完成！"
```

## 最佳实践

### 1. 开发流程

```python
# 推荐的开发流程
"""
1. 环境准备
   - 安装 JetPack 5.1.2+
   - 配置 Python 虚拟环境
   - 安装必要依赖

2. 模型选择
   - 根据应用场景选择合适的模型
   - 优先考虑 YOLOv8n 或 YOLOv8s
   - 进行基准测试

3. 配置优化
   - 使用推荐的配置文件
   - 根据硬件能力调整参数
   - 启用 TensorRT 加速

4. 性能监控
   - 实时监控 FPS 和内存使用
   - 定期检查系统状态
   - 保存性能报告

5. 部署上线
   - 进行充分的测试
   - 准备故障恢复方案
   - 监控生产环境性能
"""
```

### 2. 配置建议

```python
# 不同场景的推荐配置

# 实时检测 (高帧率)
REALTIME_CONFIG = {
    'model': 'yolov8n.pt',
    'imgsz': 320,
    'conf': 0.35,
    'iou': 0.45,
    'max_det': 30,
    'half': True,
    'batch': 1,
    'workers': 1,
}

# 高精度检测
ACCURACY_CONFIG = {
    'model': 'yolov8s.pt',
    'imgsz': 640,
    'conf': 0.5,
    'iou': 0.45,
    'max_det': 100,
    'half': True,
    'batch': 1,
    'workers': 2,
}

# 平衡配置
BALANCED_CONFIG = {
    'model': 'yolov8n.pt',
    'imgsz': 416,
    'conf': 0.45,
    'iou': 0.45,
    'max_det': 50,
    'half': True,
    'batch': 1,
    'workers': 2,
}
```

### 3. 维护建议

```bash
# 定期维护脚本
# scripts/maintenance.sh

#!/bin/bash

echo "=== 定期维护脚本 ==="

# 清理缓存
echo "1. 清理缓存..."
rm -rf /tmp/cuda_cache/*
rm -rf /tmp/tensorrt_cache/*
rm -rf ~/.cache/torch/*

# 检查磁盘空间
echo "2. 检查磁盘空间..."
df -h

# 检查系统性能
echo "3. 检查系统性能..."
nvidia-smi
tegrastats --interval 1000 --logfile tegrastats.log &
TEGRA_PID=$!
sleep 5
kill $TEGRA_PID

# 更新依赖
echo "4. 检查更新..."
pip list --outdated

# 备份配置
echo "5. 备份配置..."
cp -r config/ backup/config_$(date +%Y%m%d_%H%M%S)/

# 运行诊断
echo "6. 运行诊断..."
python3 utils/performance_diagnostics.py

echo "维护完成！"
```

### 4. 部署检查清单

```markdown
## 部署前检查清单

### 硬件检查
- [ ] Jetson Orin Nano 正常工作
- [ ] 散热系统正常
- [ ] 电源供应稳定
- [ ] 存储空间充足 (>16GB)

### 软件检查
- [ ] JetPack 5.1.2+ 已安装
- [ ] Python 3.8+ 已安装
- [ ] PyTorch 已正确安装
- [ ] Ultralytics 已安装
- [ ] 所有依赖库已安装

### 配置检查
- [ ] 配置文件完整
- [ ] 模型文件存在
- [ ] 权限设置正确
- [ ] 环境变量正确

### 性能检查
- [ ] 基准测试通过
- [ ] 内存使用合理
- [ ] FPS 达到要求
- [ ] 错误率可接受

### 安全检查
- [ ] 用户权限正确
- [ ] 网络配置安全
- [ ] 数据备份完整
- [ ] 监控系统正常

### 文档检查
- [ ] 用户手册完整
- [ ] 故障排除指南
- [ ] 维护文档
- [ ] 联系信息
```

---

## 总结

本指南为在 Jetson Orin Nano 上部署和优化 Ultralytics YOLO 提供了全面的解决方案。通过遵循本指南的建议和最佳实践，您可以：

1. **成功部署**: 在 Jetson Orin Nano 上成功运行 YOLO 模型
2. **性能优化**: 获得最佳的推理性能和能耗比
3. **稳定运行**: 确保系统长期稳定运行
4. **快速故障排除**: 快速定位和解决问题

如果您在使用过程中遇到问题，请参考故障排除部分或联系技术支持。

祝您使用愉快！

---

*文档版本: 1.0*  
*最后更新: 2024年*  
*适用于: Jetson Orin Nano + Ultralytics YOLO*