"""
Intel RealSense D435i + YOLO 目标检测与深度测距系统
版本: 1.0.0
作者: System Developer
"""

__version__ = "1.0.0"
__author__ = "System Developer"
__email__ = "developer@example.com"
__description__ = "Intel RealSense D435i + YOLO 实时3D目标检测与测距系统"

# 导入核心模块
try:
    from .src.camera_manager import CameraManager
    from .src.object_detector import ObjectDetector
    from .src.depth_estimator import DepthEstimator
    from .src.data_fusion import DataFusion
    from .src.visualizer import Visualizer
    from .config.config import *
    
    __all__ = [
        'CameraManager',
        'ObjectDetector', 
        'DepthEstimator',
        'DataFusion',
        'Visualizer',
    ]
except ImportError as e:
    print(f"导入模块时出错: {e}")
    print("请确保已安装所有依赖包")
    __all__ = []

# 系统信息
SYSTEM_INFO = {
    'name': 'RealSense YOLO System',
    'version': __version__,
    'description': __description__,
    'author': __author__,
    'python_requires': '>=3.7',
    'platforms': ['Linux', 'Windows', 'macOS'],
    'hardware_requirements': {
        'camera': 'Intel RealSense D435i',
        'compute': 'CUDA-capable GPU (recommended)',
        'memory': '8GB RAM (minimum)',
        'storage': '5GB (for models and cache)',
    },
    'supported_models': [
        'YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x',
        'YOLOv5n', 'YOLOv5s', 'YOLOv5m', 'YOLOv5l', 'YOLOv5x',
    ],
}

def get_system_info():
    """获取系统信息"""
    return SYSTEM_INFO

def print_banner():
    """打印系统横幅"""
    banner = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    Intel RealSense D435i + YOLO System                       ║
║                           Version {__version__}                                     ║
║                                                                               ║
║  实时3D目标检测与深度测距系统                                                  ║
║  支持硬件: Intel RealSense D435i                                              ║
║  AI模型: YOLO系列 (v5/v8)                                                     ║
║  平台支持: Linux/Windows/macOS                                                ║
║                                                                               ║
║  作者: {__author__}                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

if __name__ == "__main__":
    print_banner()
    print("系统信息:")
    for key, value in get_system_info().items():
        print(f"  {key}: {value}")