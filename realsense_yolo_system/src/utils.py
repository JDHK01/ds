"""
通用工具函数
包含数据类型转换、图像处理、数学计算等辅助函数
"""

import cv2
import numpy as np
import time
import logging
import os
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
from contextlib import contextmanager
import threading
import queue
from functools import wraps

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Timer:
    """计时器类，用于性能测试"""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        
    def stop(self):
        """停止计时"""
        self.end_time = time.time()
        return self.elapsed()
        
    def elapsed(self) -> float:
        """获取经过的时间（秒）"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
        
    @contextmanager
    def time_it(self):
        """上下文管理器用法"""
        self.start()
        try:
            yield self
        finally:
            self.stop()
            logger.info(f"{self.name}: {self.elapsed():.4f}s")

def timer_decorator(func):
    """装饰器：自动计时函数执行时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} 执行时间: {end_time - start_time:.4f}s")
        return result
    return wrapper

class FPSCounter:
    """FPS计算器"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
        
    def update(self) -> float:
        """更新FPS并返回当前FPS"""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
            
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        return 0.0
        
    def get_fps(self) -> float:
        """获取当前FPS"""
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        return 0.0

class CircularBuffer:
    """循环缓冲区"""
    
    def __init__(self, size: int):
        self.size = size
        self.buffer = [None] * size
        self.head = 0
        self.count = 0
        self.lock = threading.Lock()
        
    def put(self, item):
        """添加元素到缓冲区"""
        with self.lock:
            self.buffer[self.head] = item
            self.head = (self.head + 1) % self.size
            self.count = min(self.count + 1, self.size)
            
    def get_latest(self):
        """获取最新的元素"""
        with self.lock:
            if self.count == 0:
                return None
            index = (self.head - 1) % self.size
            return self.buffer[index]
            
    def get_all(self) -> List:
        """获取所有元素"""
        with self.lock:
            if self.count == 0:
                return []
            items = []
            for i in range(self.count):
                index = (self.head - self.count + i) % self.size
                items.append(self.buffer[index])
            return items
            
    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        return self.count == self.size
        
    def is_empty(self) -> bool:
        """检查缓冲区是否为空"""
        return self.count == 0

# 图像处理工具函数
def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                 keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    调整图像大小
    
    Args:
        image: 输入图像
        target_size: 目标大小 (width, height)
        keep_aspect_ratio: 是否保持宽高比
    
    Returns:
        调整后的图像
    """
    if not keep_aspect_ratio:
        return cv2.resize(image, target_size)
    
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # 计算缩放比例
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 调整大小
    resized = cv2.resize(image, (new_w, new_h))
    
    # 创建目标大小的图像并居中放置
    result = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return result

def apply_colormap(depth_image: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    为深度图应用色彩映射
    
    Args:
        depth_image: 深度图像
        colormap: 色彩映射类型
    
    Returns:
        彩色深度图
    """
    # 归一化深度图到0-255
    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)
    
    # 应用色彩映射
    colored_depth = cv2.applyColorMap(depth_normalized, colormap)
    
    return colored_depth

def draw_text_with_background(image: np.ndarray, text: str, position: Tuple[int, int],
                             font_scale: float = 0.6, thickness: int = 2,
                             text_color: Tuple[int, int, int] = (255, 255, 255),
                             bg_color: Tuple[int, int, int] = (0, 0, 0),
                             alpha: float = 0.7) -> np.ndarray:
    """
    在图像上绘制带背景的文本
    
    Args:
        image: 输入图像
        text: 要绘制的文本
        position: 文本位置 (x, y)
        font_scale: 字体缩放
        thickness: 线条粗细
        text_color: 文本颜色
        bg_color: 背景颜色
        alpha: 背景透明度
    
    Returns:
        绘制了文本的图像
    """
    # 获取文本大小
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # 计算背景矩形
    x, y = position
    bg_x1, bg_y1 = x - 5, y - text_height - 5
    bg_x2, bg_y2 = x + text_width + 5, y + baseline + 5
    
    # 绘制背景
    overlay = image.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # 绘制文本
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, text_color, thickness)
    
    return image

# 数学计算工具函数
def calculate_distance_3d(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    计算两个3D点之间的距离
    
    Args:
        point1: 第一个点 [x, y, z]
        point2: 第二个点 [x, y, z]
    
    Returns:
        距离
    """
    return np.linalg.norm(point1 - point2)

def pixel_to_point(pixel_coords: Tuple[int, int], depth_value: float,
                   intrinsics: Dict[str, float]) -> np.ndarray:
    """
    将像素坐标转换为3D点坐标
    
    Args:
        pixel_coords: 像素坐标 (x, y)
        depth_value: 深度值（米）
        intrinsics: 相机内参 {'fx', 'fy', 'cx', 'cy'}
    
    Returns:
        3D点坐标 [x, y, z]
    """
    x, y = pixel_coords
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    # 转换到3D坐标
    x_3d = (x - cx) * depth_value / fx
    y_3d = (y - cy) * depth_value / fy
    z_3d = depth_value
    
    return np.array([x_3d, y_3d, z_3d])

def point_to_pixel(point_3d: np.ndarray, intrinsics: Dict[str, float]) -> Tuple[int, int]:
    """
    将3D点坐标转换为像素坐标
    
    Args:
        point_3d: 3D点坐标 [x, y, z]
        intrinsics: 相机内参 {'fx', 'fy', 'cx', 'cy'}
    
    Returns:
        像素坐标 (x, y)
    """
    x_3d, y_3d, z_3d = point_3d
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    # 转换到像素坐标
    x = int(x_3d * fx / z_3d + cx)
    y = int(y_3d * fy / z_3d + cy)
    
    return (x, y)

def filter_outliers(data: np.ndarray, std_multiplier: float = 2.0) -> np.ndarray:
    """
    使用标准差方法过滤异常值
    
    Args:
        data: 输入数据
        std_multiplier: 标准差倍数
    
    Returns:
        过滤后的数据
    """
    if len(data) == 0:
        return data
    
    mean = np.mean(data)
    std = np.std(data)
    
    # 过滤异常值
    mask = np.abs(data - mean) < std_multiplier * std
    return data[mask]

def moving_average(data: List[float], window_size: int) -> List[float]:
    """
    计算移动平均
    
    Args:
        data: 输入数据
        window_size: 窗口大小
    
    Returns:
        移动平均结果
    """
    if len(data) < window_size:
        return data
    
    result = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        result.append(sum(window) / window_size)
    
    return result

# 文件和配置工具函数
def ensure_directory(path: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
    
    Returns:
        Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    加载JSON文件
    
    Args:
        filepath: 文件路径
    
    Returns:
        JSON数据
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Dict[str, Any], filepath: Union[str, Path]):
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        filepath: 文件路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_system_info() -> Dict[str, Any]:
    """
    获取系统信息
    
    Returns:
        系统信息字典
    """
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': psutil.disk_usage('/').percent,
    }

# 线程安全的队列处理
class SafeQueue:
    """线程安全的队列"""
    
    def __init__(self, maxsize: int = 0):
        self.queue = queue.Queue(maxsize=maxsize)
        
    def put(self, item, block: bool = True, timeout: Optional[float] = None):
        """添加元素到队列"""
        return self.queue.put(item, block=block, timeout=timeout)
        
    def get(self, block: bool = True, timeout: Optional[float] = None):
        """从队列获取元素"""
        return self.queue.get(block=block, timeout=timeout)
        
    def empty(self) -> bool:
        """检查队列是否为空"""
        return self.queue.empty()
        
    def full(self) -> bool:
        """检查队列是否已满"""
        return self.queue.full()
        
    def qsize(self) -> int:
        """获取队列大小"""
        return self.queue.qsize()

# 配置验证工具
def validate_camera_config(config: Dict[str, Any]) -> bool:
    """
    验证相机配置
    
    Args:
        config: 相机配置字典
    
    Returns:
        验证是否通过
    """
    required_keys = ['width', 'height', 'fps']
    
    for key in required_keys:
        if key not in config:
            logger.error(f"缺少必需的相机配置参数: {key}")
            return False
            
    if config['width'] <= 0 or config['height'] <= 0:
        logger.error("相机分辨率必须大于0")
        return False
        
    if config['fps'] <= 0:
        logger.error("相机帧率必须大于0")
        return False
        
    return True

def validate_yolo_config(config: Dict[str, Any]) -> bool:
    """
    验证YOLO配置
    
    Args:
        config: YOLO配置字典
    
    Returns:
        验证是否通过
    """
    required_keys = ['model_path', 'confidence_threshold', 'iou_threshold']
    
    for key in required_keys:
        if key not in config:
            logger.error(f"缺少必需的YOLO配置参数: {key}")
            return False
            
    if not (0 <= config['confidence_threshold'] <= 1):
        logger.error("置信度阈值必须在0-1之间")
        return False
        
    if not (0 <= config['iou_threshold'] <= 1):
        logger.error("IoU阈值必须在0-1之间")
        return False
        
    return True

if __name__ == "__main__":
    # 测试工具函数
    print("测试工具函数...")
    
    # 测试FPS计算器
    fps_counter = FPSCounter()
    for i in range(10):
        time.sleep(0.1)
        fps = fps_counter.update()
        print(f"FPS: {fps:.2f}")
    
    # 测试循环缓冲区
    buffer = CircularBuffer(5)
    for i in range(7):
        buffer.put(i)
        print(f"Buffer: {buffer.get_all()}")
    
    # 测试计时器
    with Timer("测试计时器").time_it() as timer:
        time.sleep(0.5)
    
    print("工具函数测试完成")