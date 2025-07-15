"""
无人机视觉系统配置文件
针对Jetson Orin Nano优化的多场景配置预设
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 基础配置模板
BASE_CONFIG = {
    # 相机配置
    'camera': {
        'width': 640,
        'height': 480,
        'fps': 30,
        'depth_format': 'Z16',
        'color_format': 'BGR8',
        'enable_depth': True,
        'enable_color': True,
        'enable_infrared': False,
        'align_to_color': True,
        'buffer_size': 3,
        'enable_filters': True,
        'depth_units': 0.001,
        'enable_emitter': True,
        'laser_power': 150,
        'accuracy': 'medium',
        'filter_option': 'medium',
        'hole_fill': 'farthest_from_around',
        'preset': 'high_density',
        'device_id': None  # None表示自动检测
    },
    
    # YOLO检测配置 (针对Jetson Orin Nano优化)
    'yolo': {
        'model_path': 'yolov8n.pt',  # 使用nano版本提高速度
        'model_type': 'yolov8n',
        'device': 'cuda' if os.getenv('CUDA_AVAILABLE', '1') == '1' else 'cpu',
        'half': True,  # 使用半精度加速
        'imgsz': 416,  # 降低输入尺寸
        'confidence_threshold': 0.5,
        'iou_threshold': 0.45,
        'max_det': 50,
        'agnostic_nms': True,
        'augment': False,
        'visualize': False,
        'classes': None,  # 检测所有类别
        'retina_masks': False,
        'line_thickness': 2,
        'use_tensorrt': False,  # 设置为True启用TensorRT加速
        'track_thresh': 0.5,
        'track_buffer': 30,
        'match_thresh': 0.8,
        'frame_rate': 30
    },
    
    # 数据融合配置
    'fusion': {
        'enable_depth_fusion': True,
        'depth_scale': 0.001,
        'min_depth': 0.5,
        'max_depth': 10.0,
        'depth_confidence_threshold': 0.7,
        'depth_roi_expansion': 0.1,
        'depth_aggregation': 'median',
        'coordinate_system': 'camera',
        'min_valid_pixels': 10,
        'spatial_filter_magnitude': 2,
        'spatial_filter_alpha': 0.5,
        'temporal_filter_alpha': 0.4,
        'temporal_filter_delta': 20,
        'hole_fill_mode': 'farthest_from_around',
        'enable_decimation': True,
        'decimation_magnitude': 2,
        'outlier_removal': {
            'enable': True,
            'std_multiplier': 2.0,
        }
    },
    
    # 跟踪配置 (针对实时性能优化)
    'tracking': {
        'tracker_type': 'csrt',  # CSRT在准确性和速度间平衡最好
        'max_lost_frames': 15,
        'confidence_threshold': 0.3,
        'search_radius': 150,
        'velocity_smoothing': 0.8,
        'max_targets': 10,
        'prediction_enabled': True,
        'kalman_process_noise': 0.03,
        'kalman_measurement_noise': 0.1,
        'reacquisition_enabled': True,
        'reacquisition_threshold': 0.3,
        'async_mode': False,
        'optimize_for_jetson': True
    },
    
    # 无人机控制配置
    'drone': {
        'max_speed': 3.0,        # 最大速度 (m/s)
        'max_altitude': 25.0,    # 最大高度 (m)
        'min_altitude': 1.0,     # 最小高度 (m)
        'safety_radius': 100.0,  # 安全半径 (m)
        'battery_warning_level': 20.0,
        'battery_critical_level': 10.0,
        'gps_min_satellites': 6,
        'takeoff_altitude': 5.0,
        'land_speed': 1.0,
        'emergency_descent_rate': 2.0,
        'telemetry_rate': 10.0,
        'offboard_rate': 20.0,
        'connection_timeout': 10.0
    },
    
    # 跟随控制配置
    'following': {
        'target_distance': 6.0,
        'max_speed': 2.0,
        'min_confidence': 0.4,
        'height_offset': 0.0,
        'angle_offset': 0.0,
        'position_p_gain': 0.5,
        'velocity_p_gain': 0.3,
        'yaw_p_gain': 0.5,
        'safety_radius': 2.0,
        'max_vertical_speed': 1.0,
        'max_yaw_rate': 30.0,
        'position_filter_alpha': 0.8,
        'velocity_filter_alpha': 0.6
    },
    
    # 可视化配置
    'visualization': {
        'enable_display': True,
        'display_width': 1280,
        'display_height': 720,
        'show_fps': True,
        'show_detections': True,
        'show_tracking': True,
        'show_depth': True,
        'show_statistics': True,
        'save_results': False,
        'output_dir': 'output',
        'video_codec': 'mp4v',
        'image_format': 'jpg',
        'overlay_alpha': 0.7,
        'font_scale': 0.6,
        'line_thickness': 2
    },
    
    # 系统配置
    'system': {
        'max_fps': 30,
        'enable_gpu': True,
        'gpu_memory_growth': True,
        'log_level': 'INFO',
        'enable_profiling': False,
        'profiling_interval': 10.0,
        'memory_limit_mb': 4096,
        'thread_count': 4,
        'enable_multiprocessing': False,
        'buffer_size': 5,
        'timeout_ms': 1000
    }
}

# 场景预设配置
SCENE_PRESETS = {
    'indoor': {
        'name': '室内场景',
        'description': '针对室内环境优化，使用较低分辨率和激光功率',
        'config': {
            'camera': {
                'width': 424,
                'height': 240,
                'fps': 30,
                'laser_power': 100,
                'preset': 'indoor'
            },
            'yolo': {
                'confidence_threshold': 0.4,
                'imgsz': 320,
                'max_det': 30
            },
            'tracking': {
                'tracker_type': 'kcf',
                'max_lost_frames': 20,
                'search_radius': 100
            },
            'drone': {
                'max_speed': 2.0,
                'max_altitude': 15.0,
                'safety_radius': 50.0
            },
            'following': {
                'target_distance': 3.0,
                'max_speed': 1.5,
                'safety_radius': 1.5
            }
        }
    },
    
    'outdoor': {
        'name': '户外场景',
        'description': '针对户外环境优化，使用标准分辨率和激光功率',
        'config': {
            'camera': {
                'width': 640,
                'height': 480,
                'fps': 30,
                'laser_power': 150,
                'preset': 'outdoor'
            },
            'yolo': {
                'confidence_threshold': 0.5,
                'imgsz': 416,
                'max_det': 50
            },
            'tracking': {
                'tracker_type': 'csrt',
                'max_lost_frames': 15,
                'search_radius': 150
            },
            'drone': {
                'max_speed': 3.0,
                'max_altitude': 25.0,
                'safety_radius': 100.0
            },
            'following': {
                'target_distance': 6.0,
                'max_speed': 2.5,
                'safety_radius': 2.0
            }
        }
    },
    
    'high_performance': {
        'name': '高性能模式',
        'description': '追求最高帧率和精度，适合高端硬件',
        'config': {
            'camera': {
                'width': 848,
                'height': 480,
                'fps': 60,
                'laser_power': 150,
                'preset': 'high_density'
            },
            'yolo': {
                'confidence_threshold': 0.6,
                'imgsz': 640,
                'half': False,
                'max_det': 100,
                'use_tensorrt': True
            },
            'tracking': {
                'tracker_type': 'csrt',
                'max_lost_frames': 10,
                'search_radius': 200
            },
            'drone': {
                'max_speed': 4.0,
                'max_altitude': 30.0,
                'safety_radius': 150.0
            },
            'following': {
                'target_distance': 8.0,
                'max_speed': 3.0,
                'position_p_gain': 0.7
            },
            'system': {
                'max_fps': 60,
                'thread_count': 6
            }
        }
    },
    
    'low_power': {
        'name': '低功耗模式',
        'description': '降低功耗和CPU使用，适合长时间运行',
        'config': {
            'camera': {
                'width': 424,
                'height': 240,
                'fps': 15,
                'laser_power': 50,
                'preset': 'low_power'
            },
            'yolo': {
                'confidence_threshold': 0.3,
                'imgsz': 320,
                'max_det': 20
            },
            'tracking': {
                'tracker_type': 'mosse',
                'max_lost_frames': 30,
                'search_radius': 80
            },
            'drone': {
                'max_speed': 1.5,
                'max_altitude': 15.0,
                'safety_radius': 50.0
            },
            'following': {
                'target_distance': 4.0,
                'max_speed': 1.0,
                'safety_radius': 1.5
            },
            'system': {
                'max_fps': 15,
                'thread_count': 2
            }
        }
    },
    
    'precision_tracking': {
        'name': '精确跟踪模式',
        'description': '优化跟踪精度，适合需要高精度跟踪的场景',
        'config': {
            'camera': {
                'width': 640,
                'height': 480,
                'fps': 30,
                'laser_power': 150
            },
            'yolo': {
                'confidence_threshold': 0.7,
                'imgsz': 640,
                'max_det': 30
            },
            'tracking': {
                'tracker_type': 'csrt',
                'max_lost_frames': 5,
                'search_radius': 100,
                'kalman_process_noise': 0.01,
                'kalman_measurement_noise': 0.05
            },
            'drone': {
                'max_speed': 2.0,
                'safety_radius': 75.0
            },
            'following': {
                'target_distance': 5.0,
                'max_speed': 1.5,
                'position_p_gain': 0.3,
                'position_filter_alpha': 0.9
            }
        }
    },
    
    'fast_response': {
        'name': '快速响应模式',
        'description': '优化响应速度，适合快速运动的目标',
        'config': {
            'camera': {
                'width': 640,
                'height': 480,
                'fps': 60,
                'buffer_size': 2
            },
            'yolo': {
                'confidence_threshold': 0.4,
                'imgsz': 416,
                'max_det': 50
            },
            'tracking': {
                'tracker_type': 'mosse',
                'max_lost_frames': 10,
                'search_radius': 200
            },
            'drone': {
                'max_speed': 5.0,
                'safety_radius': 100.0
            },
            'following': {
                'target_distance': 7.0,
                'max_speed': 3.5,
                'position_p_gain': 0.8,
                'position_filter_alpha': 0.6
            },
            'system': {
                'max_fps': 60
            }
        }
    }
}

# 跟随参数预设
FOLLOW_PRESETS = {
    'conservative': {
        'name': '保守跟随',
        'description': '安全优先，较慢的响应速度',
        'params': {
            'target_distance': 8.0,
            'max_speed': 1.5,
            'min_confidence': 0.6,
            'height_offset': 0.0,
            'angle_offset': 0.0,
            'position_p_gain': 0.3,
            'velocity_p_gain': 0.2,
            'yaw_p_gain': 0.3,
            'safety_radius': 3.0,
            'max_vertical_speed': 0.5,
            'max_yaw_rate': 20.0,
            'position_filter_alpha': 0.9,
            'velocity_filter_alpha': 0.8
        }
    },
    
    'balanced': {
        'name': '平衡跟随',
        'description': '平衡安全性和响应速度',
        'params': {
            'target_distance': 6.0,
            'max_speed': 2.5,
            'min_confidence': 0.4,
            'height_offset': 0.0,
            'angle_offset': 0.0,
            'position_p_gain': 0.5,
            'velocity_p_gain': 0.3,
            'yaw_p_gain': 0.5,
            'safety_radius': 2.0,
            'max_vertical_speed': 1.0,
            'max_yaw_rate': 30.0,
            'position_filter_alpha': 0.8,
            'velocity_filter_alpha': 0.6
        }
    },
    
    'aggressive': {
        'name': '激进跟随',
        'description': '快速响应，适合熟练操作者',
        'params': {
            'target_distance': 4.0,
            'max_speed': 4.0,
            'min_confidence': 0.3,
            'height_offset': 0.0,
            'angle_offset': 0.0,
            'position_p_gain': 0.8,
            'velocity_p_gain': 0.5,
            'yaw_p_gain': 0.8,
            'safety_radius': 1.5,
            'max_vertical_speed': 2.0,
            'max_yaw_rate': 45.0,
            'position_filter_alpha': 0.6,
            'velocity_filter_alpha': 0.4
        }
    },
    
    'cinematic': {
        'name': '电影跟随',
        'description': '平滑的跟随，适合拍摄',
        'params': {
            'target_distance': 10.0,
            'max_speed': 1.0,
            'min_confidence': 0.5,
            'height_offset': 1.0,
            'angle_offset': 0.0,
            'position_p_gain': 0.2,
            'velocity_p_gain': 0.1,
            'yaw_p_gain': 0.2,
            'safety_radius': 3.0,
            'max_vertical_speed': 0.3,
            'max_yaw_rate': 15.0,
            'position_filter_alpha': 0.95,
            'velocity_filter_alpha': 0.9
        }
    }
}

# 跟踪算法配置
TRACKER_CONFIGS = {
    'csrt': {
        'name': 'CSRT',
        'description': '最准确的跟踪算法，适合复杂场景',
        'performance': 'medium',
        'accuracy': 'high',
        'config': {
            'tracker_type': 'csrt',
            'max_lost_frames': 15,
            'confidence_threshold': 0.3,
            'search_radius': 150,
            'kalman_process_noise': 0.03,
            'kalman_measurement_noise': 0.1
        }
    },
    
    'kcf': {
        'name': 'KCF',
        'description': '平衡速度和准确性的跟踪算法',
        'performance': 'high',
        'accuracy': 'medium',
        'config': {
            'tracker_type': 'kcf',
            'max_lost_frames': 20,
            'confidence_threshold': 0.4,
            'search_radius': 120,
            'kalman_process_noise': 0.05,
            'kalman_measurement_noise': 0.2
        }
    },
    
    'mosse': {
        'name': 'MOSSE',
        'description': '最快的跟踪算法，适合实时应用',
        'performance': 'very_high',
        'accuracy': 'medium',
        'config': {
            'tracker_type': 'mosse',
            'max_lost_frames': 30,
            'confidence_threshold': 0.5,
            'search_radius': 100,
            'kalman_process_noise': 0.1,
            'kalman_measurement_noise': 0.3
        }
    },
    
    'medianflow': {
        'name': 'MedianFlow',
        'description': '适合稳定跟踪的算法',
        'performance': 'medium',
        'accuracy': 'medium',
        'config': {
            'tracker_type': 'medianflow',
            'max_lost_frames': 10,
            'confidence_threshold': 0.6,
            'search_radius': 80,
            'kalman_process_noise': 0.02,
            'kalman_measurement_noise': 0.1
        }
    },
    
    'yolo_native': {
        'name': 'YOLO Native',
        'description': '使用YOLO原生跟踪，适合多目标',
        'performance': 'medium',
        'accuracy': 'high',
        'config': {
            'tracker_type': 'yolo_native',
            'max_lost_frames': 10,
            'confidence_threshold': 0.5,
            'search_radius': 200,
            'max_targets': 10
        }
    }
}

# Jetson Orin Nano优化配置
JETSON_OPTIMIZATIONS = {
    'performance': {
        'name': '性能优化',
        'config': {
            'camera': {
                'width': 640,
                'height': 480,
                'fps': 30,
                'buffer_size': 3
            },
            'yolo': {
                'imgsz': 416,
                'half': True,
                'use_tensorrt': True
            },
            'tracking': {
                'tracker_type': 'kcf',
                'optimize_for_jetson': True
            },
            'system': {
                'thread_count': 4,
                'enable_gpu': True,
                'gpu_memory_growth': True
            }
        }
    },
    
    'memory': {
        'name': '内存优化',
        'config': {
            'camera': {
                'width': 424,
                'height': 240,
                'buffer_size': 2
            },
            'yolo': {
                'imgsz': 320,
                'max_det': 20
            },
            'tracking': {
                'max_targets': 5
            },
            'system': {
                'memory_limit_mb': 2048,
                'buffer_size': 3
            }
        }
    },
    
    'power': {
        'name': '功耗优化',
        'config': {
            'camera': {
                'fps': 15,
                'laser_power': 50
            },
            'yolo': {
                'imgsz': 320,
                'half': True
            },
            'tracking': {
                'tracker_type': 'mosse'
            },
            'system': {
                'max_fps': 15,
                'thread_count': 2
            }
        }
    }
}

def get_config(scene: str = 'balanced', 
               follow_preset: str = 'balanced',
               tracker: str = 'csrt',
               jetson_optimization: Optional[str] = None,
               custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    获取完整配置
    
    Args:
        scene: 场景预设名称
        follow_preset: 跟随预设名称
        tracker: 跟踪算法名称
        jetson_optimization: Jetson优化类型
        custom_overrides: 自定义覆盖配置
    
    Returns:
        完整配置字典
    """
    # 从基础配置开始
    config = BASE_CONFIG.copy()
    
    # 深度复制配置
    import copy
    config = copy.deepcopy(config)
    
    # 应用场景预设
    if scene in SCENE_PRESETS:
        scene_config = SCENE_PRESETS[scene]['config']
        for section, settings in scene_config.items():
            if section in config:
                config[section].update(settings)
    
    # 应用跟随预设
    if follow_preset in FOLLOW_PRESETS:
        follow_config = FOLLOW_PRESETS[follow_preset]['params']
        config['following'].update(follow_config)
    
    # 应用跟踪算法配置
    if tracker in TRACKER_CONFIGS:
        tracker_config = TRACKER_CONFIGS[tracker]['config']
        config['tracking'].update(tracker_config)
    
    # 应用Jetson优化
    if jetson_optimization and jetson_optimization in JETSON_OPTIMIZATIONS:
        jetson_config = JETSON_OPTIMIZATIONS[jetson_optimization]['config']
        for section, settings in jetson_config.items():
            if section in config:
                config[section].update(settings)
    
    # 应用自定义覆盖
    if custom_overrides:
        for section, settings in custom_overrides.items():
            if section in config:
                config[section].update(settings)
            else:
                config[section] = settings
    
    return config

def get_scene_presets() -> Dict[str, Dict[str, Any]]:
    """获取所有场景预设"""
    return SCENE_PRESETS

def get_follow_presets() -> Dict[str, Dict[str, Any]]:
    """获取所有跟随预设"""
    return FOLLOW_PRESETS

def get_tracker_configs() -> Dict[str, Dict[str, Any]]:
    """获取所有跟踪器配置"""
    return TRACKER_CONFIGS

def get_jetson_optimizations() -> Dict[str, Dict[str, Any]]:
    """获取Jetson优化配置"""
    return JETSON_OPTIMIZATIONS

def create_custom_scene(name: str, description: str, base_scene: str, 
                       overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建自定义场景
    
    Args:
        name: 场景名称
        description: 场景描述
        base_scene: 基础场景
        overrides: 覆盖配置
    
    Returns:
        自定义场景配置
    """
    if base_scene not in SCENE_PRESETS:
        raise ValueError(f"基础场景 '{base_scene}' 不存在")
    
    # 复制基础场景
    import copy
    custom_scene = copy.deepcopy(SCENE_PRESETS[base_scene])
    
    # 更新信息
    custom_scene['name'] = name
    custom_scene['description'] = description
    
    # 应用覆盖
    for section, settings in overrides.items():
        if section in custom_scene['config']:
            custom_scene['config'][section].update(settings)
        else:
            custom_scene['config'][section] = settings
    
    return custom_scene

def save_config(config: Dict[str, Any], filepath: str):
    """保存配置到文件"""
    import json
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(filepath: str) -> Dict[str, Any]:
    """从文件加载配置"""
    import json
    
    with open(filepath, 'r') as f:
        return json.load(f)

def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置有效性"""
    required_sections = ['camera', 'yolo', 'tracking', 'drone', 'following']
    
    for section in required_sections:
        if section not in config:
            return False
    
    return True

def get_recommended_config(hardware_type: str = 'jetson_orin_nano') -> Dict[str, Any]:
    """
    获取推荐配置
    
    Args:
        hardware_type: 硬件类型
    
    Returns:
        推荐配置
    """
    if hardware_type == 'jetson_orin_nano':
        return get_config(
            scene='outdoor',
            follow_preset='balanced',
            tracker='csrt',
            jetson_optimization='performance'
        )
    elif hardware_type == 'jetson_nano':
        return get_config(
            scene='outdoor',
            follow_preset='conservative',
            tracker='kcf',
            jetson_optimization='memory'
        )
    elif hardware_type == 'high_end_pc':
        return get_config(
            scene='high_performance',
            follow_preset='aggressive',
            tracker='csrt'
        )
    else:
        return get_config()

def print_config_summary(config: Dict[str, Any]):
    """打印配置摘要"""
    print("=== 配置摘要 ===")
    print(f"相机分辨率: {config['camera']['width']}x{config['camera']['height']} @ {config['camera']['fps']}fps")
    print(f"YOLO模型: {config['yolo']['model_type']}, 置信度: {config['yolo']['confidence_threshold']}")
    print(f"跟踪算法: {config['tracking']['tracker_type']}")
    print(f"无人机最大速度: {config['drone']['max_speed']}m/s")
    print(f"跟随距离: {config['following']['target_distance']}m")
    print(f"跟随最大速度: {config['following']['max_speed']}m/s")

# 示例用法
if __name__ == "__main__":
    # 获取推荐配置
    config = get_recommended_config('jetson_orin_nano')
    print_config_summary(config)
    
    # 获取所有预设
    scenes = get_scene_presets()
    print(f"\n可用场景: {list(scenes.keys())}")
    
    follow_presets = get_follow_presets()
    print(f"可用跟随预设: {list(follow_presets.keys())}")
    
    trackers = get_tracker_configs()
    print(f"可用跟踪器: {list(trackers.keys())}")
    
    # 创建自定义场景
    custom_scene = create_custom_scene(
        name="我的自定义场景",
        description="针对特定需求的自定义配置",
        base_scene="outdoor",
        overrides={
            'camera': {'fps': 60},
            'yolo': {'confidence_threshold': 0.7}
        }
    )
    
    print(f"\n自定义场景: {custom_scene['name']}")
    print(f"描述: {custom_scene['description']}")