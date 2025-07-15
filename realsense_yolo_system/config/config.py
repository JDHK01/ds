"""
Intel RealSense D435i + YOLO 系统配置文件
配置相机参数、YOLO模型设置、系统运行参数等
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 相机配置
CAMERA_CONFIG = {
    'width': 1280,
    'height': 720,
    'fps': 30,
    'depth_format': 'Z16',
    'color_format': 'BGR8',
    'enable_depth': True,
    'enable_color': True,
    'enable_infrared': False,
    'enable_imu': True,
    'align_to_color': True,
    'device_id': None,  # None表示自动检测，或者指定设备序列号
}

# YOLO模型配置
YOLO_CONFIG = {
    'model_path': PROJECT_ROOT / 'models' / 'yolov8n.pt',
    'model_type': 'yolov8n',  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    'confidence_threshold': 0.5,
    'iou_threshold': 0.45,
    'max_detections': 100,
    'device': 'cuda' if os.getenv('CUDA_AVAILABLE', '0') == '1' else 'cpu',
    'half_precision': True,  # 使用半精度推理以提高速度
    'image_size': 640,
    'classes': None,  # None表示检测所有类别，或者指定类别列表
}

# 深度估计配置
DEPTH_CONFIG = {
    'min_depth': 0.3,  # 最小深度（米）
    'max_depth': 10.0,  # 最大深度（米）
    'depth_scale': 0.001,  # 深度缩放因子
    'spatial_filter': {
        'enable': True,
        'magnitude': 2,
        'smooth_alpha': 0.5,
        'smooth_delta': 20,
    },
    'temporal_filter': {
        'enable': True,
        'smooth_alpha': 0.4,
        'smooth_delta': 20,
        'persistence_control': 3,
    },
    'hole_filling': {
        'enable': True,
        'mode': 'farest_from_around',  # farest_from_around, nearest_from_around
    },
    'decimation': {
        'enable': True,
        'magnitude': 2,
    },
}

# 数据融合配置
FUSION_CONFIG = {
    'depth_roi_expansion': 0.1,  # 深度ROI扩展比例
    'depth_aggregation': 'median',  # mean, median, mode
    'coordinate_system': 'camera',  # camera, world
    'min_valid_pixels': 10,  # 最小有效像素数
    'outlier_removal': {
        'enable': True,
        'std_multiplier': 2.0,
    },
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'show_rgb': True,
    'show_depth': True,
    'show_detections': True,
    'show_3d_info': True,
    'color_map': 'jet',  # 深度图颜色映射
    'font_scale': 0.6,
    'font_thickness': 2,
    'bbox_thickness': 2,
    'fps_display': True,
    'save_results': False,
    'output_dir': PROJECT_ROOT / 'output',
}

# 系统配置
SYSTEM_CONFIG = {
    'max_fps': 30,
    'buffer_size': 5,
    'num_threads': 4,
    'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'log_file': PROJECT_ROOT / 'logs' / 'system.log',
    'profile_performance': False,
    'auto_exposure': True,
    'auto_white_balance': True,
}

# ROS配置（可选）
ROS_CONFIG = {
    'enable_ros': False,
    'node_name': 'realsense_yolo_detector',
    'rgb_topic': '/camera/color/image_raw',
    'depth_topic': '/camera/depth/image_rect_raw',
    'detection_topic': '/detections',
    'point_cloud_topic': '/camera/depth/points',
    'camera_info_topic': '/camera/color/camera_info',
}

# 性能优化配置
PERFORMANCE_CONFIG = {
    'use_gpu': True,
    'batch_processing': False,
    'memory_optimization': True,
    'multi_threading': True,
    'frame_skip': 0,  # 跳帧数量，0表示不跳帧
    'roi_processing': False,  # 仅处理感兴趣区域
}

# 调试配置
DEBUG_CONFIG = {
    'save_debug_images': False,
    'debug_output_dir': PROJECT_ROOT / 'debug',
    'profiling': False,
    'verbose_logging': False,
    'benchmark_mode': False,
}

# 创建必要的目录
def create_directories():
    """创建项目所需的目录"""
    directories = [
        PROJECT_ROOT / 'models',
        PROJECT_ROOT / 'output',
        PROJECT_ROOT / 'logs',
        PROJECT_ROOT / 'debug',
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)

# 验证配置
def validate_config():
    """验证配置参数的有效性"""
    errors = []
    
    # 检查相机配置
    if CAMERA_CONFIG['fps'] <= 0:
        errors.append("相机FPS必须大于0")
    
    # 检查YOLO配置
    if YOLO_CONFIG['confidence_threshold'] < 0 or YOLO_CONFIG['confidence_threshold'] > 1:
        errors.append("YOLO置信度阈值必须在0-1之间")
    
    # 检查深度配置
    if DEPTH_CONFIG['min_depth'] >= DEPTH_CONFIG['max_depth']:
        errors.append("最小深度必须小于最大深度")
    
    if errors:
        raise ValueError("配置错误:\n" + "\n".join(errors))
    
    return True

# 获取配置摘要
def get_config_summary():
    """获取配置摘要信息"""
    return {
        'camera': f"{CAMERA_CONFIG['width']}x{CAMERA_CONFIG['height']}@{CAMERA_CONFIG['fps']}fps",
        'yolo_model': YOLO_CONFIG['model_type'],
        'device': YOLO_CONFIG['device'],
        'depth_range': f"{DEPTH_CONFIG['min_depth']}-{DEPTH_CONFIG['max_depth']}m",
        'max_fps': SYSTEM_CONFIG['max_fps'],
    }

if __name__ == "__main__":
    # 创建必要目录
    create_directories()
    
    # 验证配置
    validate_config()
    
    # 打印配置摘要
    summary = get_config_summary()
    print("系统配置摘要:")
    for key, value in summary.items():
        print(f"  {key}: {value}")