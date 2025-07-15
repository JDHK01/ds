"""
数据融合模块
将目标检测结果与深度信息融合，生成3D定位结果
"""

import numpy as np
import cv2
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import threading

from .object_detector import DetectionResult
from .depth_estimator import DepthInfo, DepthEstimator
from .utils import timer_decorator, calculate_distance_3d, moving_average

logger = logging.getLogger(__name__)

@dataclass
class FusedResult:
    """融合结果数据结构"""
    detection: DetectionResult
    depth_info: DepthInfo
    world_position: Tuple[float, float, float]  # 世界坐标 (x, y, z)
    timestamp: float
    tracking_id: Optional[str] = None
    velocity: Optional[Tuple[float, float, float]] = None
    confidence_3d: float = 0.0
    
    def __post_init__(self):
        """后初始化处理"""
        if self.timestamp <= 0:
            self.timestamp = time.time()
        
        # 计算3D置信度（结合检测置信度和深度置信度）
        self.confidence_3d = (self.detection.confidence * self.depth_info.confidence) ** 0.5
    
    @property
    def distance_from_camera(self) -> float:
        """相机到目标的距离"""
        return self.depth_info.distance
    
    @property
    def is_valid(self) -> bool:
        """检查融合结果是否有效"""
        return (self.detection.confidence > 0 and 
                self.depth_info.is_valid() and
                self.confidence_3d > 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'detection': self.detection.to_dict(),
            'depth_info': self.depth_info.to_dict(),
            'world_position': self.world_position,
            'timestamp': self.timestamp,
            'tracking_id': self.tracking_id,
            'velocity': self.velocity,
            'confidence_3d': self.confidence_3d,
            'distance_from_camera': self.distance_from_camera,
        }

@dataclass
class TrackingInfo:
    """跟踪信息"""
    track_id: str
    positions: deque = field(default_factory=lambda: deque(maxlen=10))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=10))
    last_update: float = 0.0
    class_name: str = ""
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def add_position(self, position: Tuple[float, float, float], 
                    timestamp: float, confidence: float):
        """添加位置记录"""
        self.positions.append(position)
        self.timestamps.append(timestamp)
        self.confidence_history.append(confidence)
        self.last_update = timestamp
    
    def get_velocity(self) -> Optional[Tuple[float, float, float]]:
        """计算速度"""
        if len(self.positions) < 2:
            return None
        
        # 使用最近两个位置计算速度
        pos1 = self.positions[-2]
        pos2 = self.positions[-1]
        t1 = self.timestamps[-2]
        t2 = self.timestamps[-1]
        
        dt = t2 - t1
        if dt <= 0:
            return None
        
        vx = (pos2[0] - pos1[0]) / dt
        vy = (pos2[1] - pos1[1]) / dt
        vz = (pos2[2] - pos1[2]) / dt
        
        return (vx, vy, vz)
    
    def get_smooth_position(self) -> Optional[Tuple[float, float, float]]:
        """获取平滑位置"""
        if not self.positions:
            return None
        
        # 使用加权平均，最近的位置权重更高
        positions = list(self.positions)
        weights = np.linspace(0.5, 1.0, len(positions))
        
        x = np.average([p[0] for p in positions], weights=weights)
        y = np.average([p[1] for p in positions], weights=weights)
        z = np.average([p[2] for p in positions], weights=weights)
        
        return (x, y, z)
    
    def is_stale(self, current_time: float, timeout: float = 1.0) -> bool:
        """检查跟踪是否过期"""
        return current_time - self.last_update > timeout

class DataFusion:
    """数据融合器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据融合器
        
        Args:
            config: 融合配置
        """
        self.config = config
        self.depth_estimator = DepthEstimator(config)
        
        # 跟踪管理
        self.tracking_objects = {}
        self.next_track_id = 1
        self.tracking_lock = threading.Lock()
        
        # 融合参数
        self.roi_expansion = config.get('depth_roi_expansion', 0.1)
        self.min_valid_pixels = config.get('min_valid_pixels', 10)
        self.max_depth_std = config.get('max_depth_std', 0.5)
        
        # 性能监控
        self.fusion_times = []
        self.total_fusions = 0
        
        logger.info("数据融合器初始化完成")
    
    @timer_decorator
    def fuse_detections_with_depth(self, detections: List[DetectionResult],
                                  depth_image: np.ndarray,
                                  intrinsics: Dict[str, float]) -> List[FusedResult]:
        """
        融合检测结果和深度信息
        
        Args:
            detections: 检测结果列表
            depth_image: 深度图像
            intrinsics: 相机内参
            
        Returns:
            融合结果列表
        """
        start_time = time.time()
        
        # 处理深度图像
        processed_depth = self.depth_estimator.process_depth_image(depth_image)
        
        fused_results = []
        
        for detection in detections:
            # 扩展ROI以获得更好的深度估计
            expanded_bbox = self._expand_bbox(detection.bbox, 
                                            processed_depth.shape, 
                                            self.roi_expansion)
            
            # 提取深度信息
            depth_info = self.depth_estimator.extract_depth_info(
                processed_depth, expanded_bbox, intrinsics)
            
            # 检查深度信息质量
            if not self._validate_depth_info(depth_info):
                continue
            
            # 计算世界坐标
            world_position = self._calculate_world_position(
                detection, depth_info, intrinsics)
            
            # 创建融合结果
            fused_result = FusedResult(
                detection=detection,
                depth_info=depth_info,
                world_position=world_position,
                timestamp=time.time()
            )
            
            fused_results.append(fused_result)
        
        # 更新跟踪信息
        self._update_tracking(fused_results)
        
        # 更新性能统计
        fusion_time = time.time() - start_time
        self.fusion_times.append(fusion_time)
        if len(self.fusion_times) > 100:
            self.fusion_times.pop(0)
        
        self.total_fusions += 1
        
        return fused_results
    
    def _expand_bbox(self, bbox: List[float], image_shape: Tuple[int, int], 
                    expansion: float) -> List[int]:
        """
        扩展边界框以获得更好的深度估计
        
        Args:
            bbox: 原始边界框 [x1, y1, x2, y2]
            image_shape: 图像形状 (height, width)
            expansion: 扩展比例
            
        Returns:
            扩展后的边界框 [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        h, w = image_shape
        
        # 计算扩展量
        width = x2 - x1
        height = y2 - y1
        
        dx = width * expansion / 2
        dy = height * expansion / 2
        
        # 扩展边界框
        new_x1 = max(0, int(x1 - dx))
        new_y1 = max(0, int(y1 - dy))
        new_x2 = min(w - 1, int(x2 + dx))
        new_y2 = min(h - 1, int(y2 + dy))
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    def _validate_depth_info(self, depth_info: DepthInfo) -> bool:
        """
        验证深度信息质量
        
        Args:
            depth_info: 深度信息
            
        Returns:
            是否有效
        """
        # 检查基本有效性
        if not depth_info.is_valid():
            return False
        
        # 检查有效像素数量
        if depth_info.valid_pixel_count < self.min_valid_pixels:
            return False
        
        # 检查深度范围
        if not (self.depth_estimator.min_depth <= depth_info.distance <= 
                self.depth_estimator.max_depth):
            return False
        
        # 检查置信度
        if depth_info.confidence < 0.3:
            return False
        
        return True
    
    def _calculate_world_position(self, detection: DetectionResult,
                                depth_info: DepthInfo,
                                intrinsics: Dict[str, float]) -> Tuple[float, float, float]:
        """
        计算世界坐标
        
        Args:
            detection: 检测结果
            depth_info: 深度信息
            intrinsics: 相机内参
            
        Returns:
            世界坐标 (x, y, z)
        """
        # 使用检测框中心点
        center_x, center_y = detection.center
        
        # 计算3D坐标
        x_3d = (center_x - intrinsics['cx']) * depth_info.distance / intrinsics['fx']
        y_3d = (center_y - intrinsics['cy']) * depth_info.distance / intrinsics['fy']
        z_3d = depth_info.distance
        
        # 坐标系转换（如果需要）
        if self.config.get('coordinate_system') == 'world':
            # 这里可以添加相机到世界坐标系的转换
            # 目前使用相机坐标系
            pass
        
        return (x_3d, y_3d, z_3d)
    
    def _update_tracking(self, fused_results: List[FusedResult]):
        """
        更新目标跟踪
        
        Args:
            fused_results: 融合结果列表
        """
        with self.tracking_lock:
            current_time = time.time()
            
            # 移除过期的跟踪
            expired_tracks = []
            for track_id, track_info in self.tracking_objects.items():
                if track_info.is_stale(current_time):
                    expired_tracks.append(track_id)
            
            for track_id in expired_tracks:
                del self.tracking_objects[track_id]
            
            # 匹配当前检测结果与现有跟踪
            for result in fused_results:
                matched_track = self._match_detection_to_track(result)
                
                if matched_track:
                    # 更新现有跟踪
                    track_info = self.tracking_objects[matched_track]
                    track_info.add_position(result.world_position, 
                                          result.timestamp, 
                                          result.confidence_3d)
                    
                    # 更新融合结果
                    result.tracking_id = matched_track
                    result.velocity = track_info.get_velocity()
                    
                    # 使用平滑位置
                    smooth_pos = track_info.get_smooth_position()
                    if smooth_pos:
                        result.world_position = smooth_pos
                else:
                    # 创建新跟踪
                    track_id = f"track_{self.next_track_id}"
                    self.next_track_id += 1
                    
                    track_info = TrackingInfo(
                        track_id=track_id,
                        class_name=result.detection.class_name
                    )
                    track_info.add_position(result.world_position, 
                                          result.timestamp, 
                                          result.confidence_3d)
                    
                    self.tracking_objects[track_id] = track_info
                    result.tracking_id = track_id
    
    def _match_detection_to_track(self, result: FusedResult) -> Optional[str]:
        """
        匹配检测结果到现有跟踪
        
        Args:
            result: 融合结果
            
        Returns:
            匹配的跟踪ID或None
        """
        if not self.tracking_objects:
            return None
        
        best_match = None
        min_distance = float('inf')
        max_distance = 0.5  # 最大匹配距离（米）
        
        for track_id, track_info in self.tracking_objects.items():
            # 类别匹配
            if track_info.class_name != result.detection.class_name:
                continue
            
            # 计算距离
            if track_info.positions:
                last_pos = track_info.positions[-1]
                distance = calculate_distance_3d(result.world_position, last_pos)
                
                if distance < min_distance and distance < max_distance:
                    min_distance = distance
                    best_match = track_id
        
        return best_match
    
    def get_tracking_info(self, track_id: str) -> Optional[TrackingInfo]:
        """
        获取跟踪信息
        
        Args:
            track_id: 跟踪ID
            
        Returns:
            跟踪信息或None
        """
        with self.tracking_lock:
            return self.tracking_objects.get(track_id)
    
    def get_all_tracking_info(self) -> Dict[str, TrackingInfo]:
        """获取所有跟踪信息"""
        with self.tracking_lock:
            return self.tracking_objects.copy()
    
    def filter_fused_results(self, results: List[FusedResult],
                           min_confidence: float = 0.5,
                           min_distance: float = 0.3,
                           max_distance: float = 10.0) -> List[FusedResult]:
        """
        过滤融合结果
        
        Args:
            results: 融合结果列表
            min_confidence: 最小置信度
            min_distance: 最小距离
            max_distance: 最大距离
            
        Returns:
            过滤后的结果
        """
        filtered = []
        
        for result in results:
            # 置信度过滤
            if result.confidence_3d < min_confidence:
                continue
            
            # 距离过滤
            distance = result.distance_from_camera
            if distance < min_distance or distance > max_distance:
                continue
            
            # 有效性检查
            if not result.is_valid:
                continue
            
            filtered.append(result)
        
        return filtered
    
    def calculate_relative_positions(self, results: List[FusedResult]) -> List[Dict[str, Any]]:
        """
        计算目标之间的相对位置
        
        Args:
            results: 融合结果列表
            
        Returns:
            相对位置信息列表
        """
        relative_positions = []
        
        for i, result1 in enumerate(results):
            for j, result2 in enumerate(results):
                if i >= j:
                    continue
                
                # 计算相对距离
                distance = calculate_distance_3d(result1.world_position, 
                                               result2.world_position)
                
                # 计算相对方向
                dx = result2.world_position[0] - result1.world_position[0]
                dy = result2.world_position[1] - result1.world_position[1]
                dz = result2.world_position[2] - result1.world_position[2]
                
                # 计算角度
                angle_xy = np.arctan2(dy, dx) * 180 / np.pi
                angle_xz = np.arctan2(dz, dx) * 180 / np.pi
                
                relative_info = {
                    'object1': {
                        'class': result1.detection.class_name,
                        'tracking_id': result1.tracking_id,
                        'position': result1.world_position,
                    },
                    'object2': {
                        'class': result2.detection.class_name,
                        'tracking_id': result2.tracking_id,
                        'position': result2.world_position,
                    },
                    'distance': distance,
                    'direction': (dx, dy, dz),
                    'angle_xy': angle_xy,
                    'angle_xz': angle_xz,
                }
                
                relative_positions.append(relative_info)
        
        return relative_positions
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """
        获取融合统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_fusions': self.total_fusions,
            'active_tracks': len(self.tracking_objects),
            'avg_fusion_time': np.mean(self.fusion_times) if self.fusion_times else 0,
            'config': self.config,
        }
        
        # 添加跟踪统计
        if self.tracking_objects:
            track_durations = []
            for track_info in self.tracking_objects.values():
                if len(track_info.timestamps) > 1:
                    duration = track_info.timestamps[-1] - track_info.timestamps[0]
                    track_durations.append(duration)
            
            stats['avg_track_duration'] = np.mean(track_durations) if track_durations else 0
            stats['max_track_duration'] = np.max(track_durations) if track_durations else 0
        
        return stats
    
    def save_tracking_history(self, filepath: str):
        """
        保存跟踪历史
        
        Args:
            filepath: 文件路径
        """
        import json
        
        history = {}
        with self.tracking_lock:
            for track_id, track_info in self.tracking_objects.items():
                history[track_id] = {
                    'class_name': track_info.class_name,
                    'positions': list(track_info.positions),
                    'timestamps': list(track_info.timestamps),
                    'confidence_history': list(track_info.confidence_history),
                    'last_update': track_info.last_update,
                }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"跟踪历史已保存: {filepath}")
    
    def reset_tracking(self):
        """重置跟踪"""
        with self.tracking_lock:
            self.tracking_objects.clear()
            self.next_track_id = 1
        
        logger.info("跟踪已重置")
    
    def cleanup(self):
        """清理资源"""
        self.reset_tracking()
        self.depth_estimator.cleanup()
        logger.info("数据融合器资源清理完成")

# 工厂函数
def create_data_fusion(config: Dict[str, Any]) -> DataFusion:
    """
    创建数据融合器
    
    Args:
        config: 融合配置
        
    Returns:
        数据融合器实例
    """
    return DataFusion(config)

# 几何计算工具函数
def calculate_bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    计算两个边界框的IoU
    
    Args:
        bbox1: 第一个边界框 [x1, y1, x2, y2]
        bbox2: 第二个边界框 [x1, y1, x2, y2]
        
    Returns:
        IoU值
    """
    # 计算交集
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算并集
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def project_3d_to_2d(point_3d: Tuple[float, float, float],
                    intrinsics: Dict[str, float]) -> Tuple[int, int]:
    """
    将3D点投影到2D图像平面
    
    Args:
        point_3d: 3D点 (x, y, z)
        intrinsics: 相机内参
        
    Returns:
        2D像素坐标 (u, v)
    """
    x, y, z = point_3d
    
    if z == 0:
        return (0, 0)
    
    u = int(x * intrinsics['fx'] / z + intrinsics['cx'])
    v = int(y * intrinsics['fy'] / z + intrinsics['cy'])
    
    return (u, v)

if __name__ == "__main__":
    # 测试数据融合器
    import sys
    sys.path.append('..')
    from config.config import FUSION_CONFIG
    from src.object_detector import DetectionResult
    
    print("测试数据融合器...")
    
    # 创建融合器
    fusion = create_data_fusion(FUSION_CONFIG)
    
    # 创建测试数据
    test_detections = [
        DetectionResult(
            bbox=[100, 100, 200, 200],
            confidence=0.8,
            class_id=0,
            class_name="person"
        )
    ]
    
    test_depth = np.random.randint(500, 2000, (480, 640)).astype(np.uint16)
    test_intrinsics = {
        'fx': 600, 'fy': 600,
        'cx': 320, 'cy': 240
    }
    
    # 运行融合
    fused_results = fusion.fuse_detections_with_depth(
        test_detections, test_depth, test_intrinsics)
    
    print(f"融合结果数量: {len(fused_results)}")
    
    for result in fused_results:
        print(f"目标: {result.detection.class_name}")
        print(f"3D位置: {result.world_position}")
        print(f"距离: {result.distance_from_camera:.2f}m")
        print(f"置信度: {result.confidence_3d:.2f}")
    
    # 获取统计信息
    stats = fusion.get_fusion_statistics()
    print(f"融合统计: {stats}")
    
    # 清理资源
    fusion.cleanup()
    
    print("测试完成")