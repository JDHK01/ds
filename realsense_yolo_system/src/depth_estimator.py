"""
深度估计模块
处理深度图像，提取深度信息，计算3D坐标
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional, Union
from dataclasses import dataclass
from scipy import ndimage
from scipy.spatial.distance import cdist

from .utils import timer_decorator, filter_outliers, moving_average

logger = logging.getLogger(__name__)

@dataclass
class DepthInfo:
    """深度信息数据结构"""
    distance: float  # 距离（米）
    x: float  # 3D x坐标
    y: float  # 3D y坐标
    z: float  # 3D z坐标
    confidence: float  # 置信度（0-1）
    valid_pixel_count: int  # 有效像素数量
    
    def __post_init__(self):
        """后初始化验证"""
        if self.distance < 0:
            raise ValueError("距离不能为负数")
        if not 0 <= self.confidence <= 1:
            raise ValueError("置信度必须在0-1之间")
    
    def is_valid(self) -> bool:
        """检查深度信息是否有效"""
        return (self.distance > 0 and 
                self.confidence > 0 and 
                self.valid_pixel_count > 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'distance': self.distance,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'confidence': self.confidence,
            'valid_pixel_count': self.valid_pixel_count,
        }

@dataclass
class DepthRegion:
    """深度区域数据结构"""
    bbox: List[int]  # [x1, y1, x2, y2]
    depth_values: np.ndarray  # 深度值数组
    mean_depth: float  # 平均深度
    std_depth: float  # 深度标准差
    valid_ratio: float  # 有效像素比例
    
    @property
    def center(self) -> Tuple[int, int]:
        """获取区域中心点"""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    @property
    def area(self) -> int:
        """获取区域面积"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

class DepthEstimator:
    """深度估计器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化深度估计器
        
        Args:
            config: 深度估计配置
        """
        self.config = config
        self.depth_scale = config.get('depth_scale', 0.001)
        self.min_depth = config.get('min_depth', 0.3)
        self.max_depth = config.get('max_depth', 10.0)
        
        # 历史深度数据用于时间滤波
        self.depth_history = {}
        self.history_size = 10
        
        logger.info("深度估计器初始化完成")
    
    @timer_decorator
    def process_depth_image(self, depth_image: np.ndarray) -> np.ndarray:
        """
        处理深度图像
        
        Args:
            depth_image: 原始深度图像
            
        Returns:
            处理后的深度图像
        """
        # 转换为浮点数并应用深度缩放
        depth_float = depth_image.astype(np.float32) * self.depth_scale
        
        # 过滤异常值
        depth_float = self._filter_depth_outliers(depth_float)
        
        # 空洞填充
        depth_float = self._fill_depth_holes(depth_float)
        
        # 平滑滤波
        depth_float = self._smooth_depth(depth_float)
        
        # 限制深度范围
        depth_float = np.clip(depth_float, self.min_depth, self.max_depth)
        
        # 将无效深度设为0
        depth_float[depth_float < self.min_depth] = 0
        depth_float[depth_float > self.max_depth] = 0
        
        return depth_float
    
    def _filter_depth_outliers(self, depth_image: np.ndarray) -> np.ndarray:
        """过滤深度异常值"""
        # 使用中值滤波去除噪声
        filtered = cv2.medianBlur(depth_image, 5)
        
        # 使用统计方法过滤异常值
        valid_mask = depth_image > 0
        if np.sum(valid_mask) > 0:
            valid_depths = depth_image[valid_mask]
            mean_depth = np.mean(valid_depths)
            std_depth = np.std(valid_depths)
            
            # 保留在3个标准差内的深度值
            outlier_mask = np.abs(depth_image - mean_depth) > 3 * std_depth
            filtered[outlier_mask] = 0
        
        return filtered
    
    def _fill_depth_holes(self, depth_image: np.ndarray) -> np.ndarray:
        """填充深度图中的空洞"""
        # 创建mask标识空洞
        hole_mask = (depth_image <= 0)
        
        if not np.any(hole_mask):
            return depth_image
        
        # 使用插值填充空洞
        filled = depth_image.copy()
        
        # 方法1: 使用最近邻插值
        if self.config.get('hole_filling', {}).get('mode') == 'nearest':
            # 找到最近的有效像素
            indices = np.where(~hole_mask)
            if len(indices[0]) > 0:
                # 使用KD树找最近邻
                from scipy.spatial import cKDTree
                tree = cKDTree(np.column_stack(indices))
                
                hole_indices = np.where(hole_mask)
                if len(hole_indices[0]) > 0:
                    distances, nearest_idx = tree.query(np.column_stack(hole_indices))
                    filled[hole_indices] = depth_image[indices[0][nearest_idx], indices[1][nearest_idx]]
        
        # 方法2: 使用形态学闭运算
        else:
            kernel = np.ones((3, 3), np.uint8)
            filled = cv2.morphologyEx(depth_image, cv2.MORPH_CLOSE, kernel)
        
        return filled
    
    def _smooth_depth(self, depth_image: np.ndarray) -> np.ndarray:
        """平滑深度图"""
        # 使用双边滤波保持边缘
        smoothed = cv2.bilateralFilter(depth_image, 9, 75, 75)
        
        # 使用高斯滤波进一步平滑
        smoothed = cv2.GaussianBlur(smoothed, (5, 5), 0)
        
        return smoothed
    
    def extract_depth_info(self, depth_image: np.ndarray, bbox: List[int],
                          intrinsics: Dict[str, float]) -> DepthInfo:
        """
        从指定区域提取深度信息
        
        Args:
            depth_image: 深度图像
            bbox: 边界框 [x1, y1, x2, y2]
            intrinsics: 相机内参
            
        Returns:
            深度信息
        """
        x1, y1, x2, y2 = bbox
        
        # 确保边界框在图像范围内
        h, w = depth_image.shape
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        # 提取区域
        region = depth_image[y1:y2, x1:x2]
        
        # 获取有效深度值
        valid_depths = region[region > 0]
        
        if len(valid_depths) == 0:
            return DepthInfo(0, 0, 0, 0, 0, 0)
        
        # 计算统计信息
        depth_values = self._aggregate_depth_values(valid_depths)
        distance = depth_values['distance']
        confidence = depth_values['confidence']
        
        # 计算3D坐标（使用边界框中心点）
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        x_3d = (center_x - intrinsics['cx']) * distance / intrinsics['fx']
        y_3d = (center_y - intrinsics['cy']) * distance / intrinsics['fy']
        z_3d = distance
        
        return DepthInfo(
            distance=distance,
            x=x_3d,
            y=y_3d,
            z=z_3d,
            confidence=confidence,
            valid_pixel_count=len(valid_depths)
        )
    
    def _aggregate_depth_values(self, depth_values: np.ndarray) -> Dict[str, float]:
        """
        聚合深度值
        
        Args:
            depth_values: 深度值数组
            
        Returns:
            聚合结果
        """
        if len(depth_values) == 0:
            return {'distance': 0, 'confidence': 0}
        
        # 过滤异常值
        filtered_depths = filter_outliers(depth_values, std_multiplier=2.0)
        
        if len(filtered_depths) == 0:
            filtered_depths = depth_values
        
        # 根据配置选择聚合方法
        aggregation_method = self.config.get('depth_aggregation', 'median')
        
        if aggregation_method == 'mean':
            distance = np.mean(filtered_depths)
        elif aggregation_method == 'median':
            distance = np.median(filtered_depths)
        elif aggregation_method == 'mode':
            # 使用最频繁的深度值
            hist, bin_edges = np.histogram(filtered_depths, bins=50)
            mode_idx = np.argmax(hist)
            distance = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
        else:
            distance = np.median(filtered_depths)
        
        # 计算置信度（基于深度值的一致性）
        std_depth = np.std(filtered_depths)
        confidence = 1.0 / (1.0 + std_depth)  # 标准差越小，置信度越高
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return {
            'distance': float(distance),
            'confidence': float(confidence)
        }
    
    def extract_depth_region(self, depth_image: np.ndarray, bbox: List[int]) -> DepthRegion:
        """
        提取深度区域信息
        
        Args:
            depth_image: 深度图像
            bbox: 边界框 [x1, y1, x2, y2]
            
        Returns:
            深度区域信息
        """
        x1, y1, x2, y2 = bbox
        
        # 确保边界框在图像范围内
        h, w = depth_image.shape
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        # 提取区域
        region = depth_image[y1:y2, x1:x2]
        
        # 获取有效深度值
        valid_depths = region[region > 0]
        total_pixels = region.size
        
        if len(valid_depths) == 0:
            return DepthRegion(
                bbox=bbox,
                depth_values=np.array([]),
                mean_depth=0,
                std_depth=0,
                valid_ratio=0
            )
        
        # 计算统计信息
        mean_depth = np.mean(valid_depths)
        std_depth = np.std(valid_depths)
        valid_ratio = len(valid_depths) / total_pixels
        
        return DepthRegion(
            bbox=bbox,
            depth_values=valid_depths,
            mean_depth=mean_depth,
            std_depth=std_depth,
            valid_ratio=valid_ratio
        )
    
    def temporal_filter(self, depth_info: DepthInfo, object_id: str) -> DepthInfo:
        """
        时间滤波，基于历史深度数据平滑当前深度
        
        Args:
            depth_info: 当前深度信息
            object_id: 对象ID
            
        Returns:
            滤波后的深度信息
        """
        if object_id not in self.depth_history:
            self.depth_history[object_id] = []
        
        # 添加当前深度到历史记录
        self.depth_history[object_id].append(depth_info.distance)
        
        # 保持历史记录大小
        if len(self.depth_history[object_id]) > self.history_size:
            self.depth_history[object_id].pop(0)
        
        # 使用移动平均平滑深度
        smoothed_distances = moving_average(self.depth_history[object_id], 
                                          min(5, len(self.depth_history[object_id])))
        
        if smoothed_distances:
            smoothed_distance = smoothed_distances[-1]
            
            # 创建新的深度信息
            smoothed_info = DepthInfo(
                distance=smoothed_distance,
                x=depth_info.x,
                y=depth_info.y,
                z=smoothed_distance,  # 更新z坐标
                confidence=depth_info.confidence,
                valid_pixel_count=depth_info.valid_pixel_count
            )
            
            return smoothed_info
        
        return depth_info
    
    def calculate_depth_statistics(self, depth_image: np.ndarray) -> Dict[str, Any]:
        """
        计算深度图像的统计信息
        
        Args:
            depth_image: 深度图像
            
        Returns:
            统计信息字典
        """
        valid_depths = depth_image[depth_image > 0]
        
        if len(valid_depths) == 0:
            return {
                'mean_depth': 0,
                'std_depth': 0,
                'min_depth': 0,
                'max_depth': 0,
                'valid_pixel_ratio': 0,
                'total_pixels': depth_image.size,
                'valid_pixels': 0,
            }
        
        stats = {
            'mean_depth': float(np.mean(valid_depths)),
            'std_depth': float(np.std(valid_depths)),
            'min_depth': float(np.min(valid_depths)),
            'max_depth': float(np.max(valid_depths)),
            'valid_pixel_ratio': len(valid_depths) / depth_image.size,
            'total_pixels': depth_image.size,
            'valid_pixels': len(valid_depths),
        }
        
        return stats
    
    def create_depth_mask(self, depth_image: np.ndarray, 
                         min_depth: Optional[float] = None,
                         max_depth: Optional[float] = None) -> np.ndarray:
        """
        创建深度掩码
        
        Args:
            depth_image: 深度图像
            min_depth: 最小深度（可选）
            max_depth: 最大深度（可选）
            
        Returns:
            深度掩码
        """
        mask = np.ones_like(depth_image, dtype=np.uint8) * 255
        
        # 过滤无效深度
        mask[depth_image <= 0] = 0
        
        # 过滤深度范围
        if min_depth is not None:
            mask[depth_image < min_depth] = 0
        if max_depth is not None:
            mask[depth_image > max_depth] = 0
        
        return mask
    
    def generate_point_cloud(self, depth_image: np.ndarray, 
                           color_image: np.ndarray,
                           intrinsics: Dict[str, float]) -> np.ndarray:
        """
        生成点云
        
        Args:
            depth_image: 深度图像
            color_image: 彩色图像
            intrinsics: 相机内参
            
        Returns:
            点云数组 [N, 6] (x, y, z, r, g, b)
        """
        h, w = depth_image.shape
        
        # 创建坐标网格
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # 过滤有效深度
        valid_mask = depth_image > 0
        
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = depth_image[valid_mask]
        
        # 计算3D坐标
        x_3d = (u_valid - intrinsics['cx']) * z_valid / intrinsics['fx']
        y_3d = (v_valid - intrinsics['cy']) * z_valid / intrinsics['fy']
        
        # 获取颜色信息
        if color_image is not None:
            colors = color_image[valid_mask]
            r, g, b = colors[:, 2], colors[:, 1], colors[:, 0]  # BGR to RGB
        else:
            r = g = b = np.ones_like(x_3d) * 255
        
        # 组装点云
        point_cloud = np.column_stack([x_3d, y_3d, z_valid, r, g, b])
        
        return point_cloud
    
    def visualize_depth_distribution(self, depth_image: np.ndarray) -> np.ndarray:
        """
        可视化深度分布
        
        Args:
            depth_image: 深度图像
            
        Returns:
            可视化图像
        """
        # 创建深度直方图
        valid_depths = depth_image[depth_image > 0]
        
        if len(valid_depths) == 0:
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        # 计算直方图
        hist, bin_edges = np.histogram(valid_depths, bins=50)
        
        # 创建可视化图像
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # 绘制直方图
        max_count = np.max(hist)
        if max_count > 0:
            for i, count in enumerate(hist):
                height = int(count * 350 / max_count)
                x = int(i * 600 / len(hist))
                cv2.rectangle(img, (x, 400 - height), (x + 10, 400), (0, 255, 0), -1)
        
        # 添加标签
        cv2.putText(img, f"Min: {np.min(valid_depths):.2f}m", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Max: {np.max(valid_depths):.2f}m", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Mean: {np.mean(valid_depths):.2f}m", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img
    
    def cleanup(self):
        """清理资源"""
        self.depth_history.clear()
        logger.info("深度估计器资源清理完成")

# 工厂函数
def create_depth_estimator(config: Dict[str, Any]) -> DepthEstimator:
    """
    创建深度估计器
    
    Args:
        config: 深度估计配置
        
    Returns:
        深度估计器实例
    """
    return DepthEstimator(config)

# 深度转换工具函数
def convert_depth_units(depth_image: np.ndarray, 
                       from_unit: str, to_unit: str) -> np.ndarray:
    """
    转换深度单位
    
    Args:
        depth_image: 深度图像
        from_unit: 原始单位 ('mm', 'cm', 'm')
        to_unit: 目标单位 ('mm', 'cm', 'm')
        
    Returns:
        转换后的深度图像
    """
    # 单位转换表（转换为米）
    unit_to_meters = {
        'mm': 0.001,
        'cm': 0.01,
        'm': 1.0
    }
    
    # 转换为米
    depth_in_meters = depth_image * unit_to_meters[from_unit]
    
    # 转换为目标单位
    depth_converted = depth_in_meters / unit_to_meters[to_unit]
    
    return depth_converted

def calculate_3d_distance(point1: Tuple[float, float, float], 
                         point2: Tuple[float, float, float]) -> float:
    """
    计算两个3D点之间的距离
    
    Args:
        point1: 第一个点 (x, y, z)
        point2: 第二个点 (x, y, z)
        
    Returns:
        距离
    """
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    dz = point1[2] - point2[2]
    
    return np.sqrt(dx*dx + dy*dy + dz*dz)

if __name__ == "__main__":
    # 测试深度估计器
    import sys
    sys.path.append('..')
    from config.config import DEPTH_CONFIG
    
    print("测试深度估计器...")
    
    # 创建深度估计器
    estimator = create_depth_estimator(DEPTH_CONFIG)
    
    # 创建测试深度图像
    test_depth = np.random.randint(500, 2000, (480, 640)).astype(np.uint16)
    
    # 处理深度图像
    processed_depth = estimator.process_depth_image(test_depth)
    print(f"处理后深度图像形状: {processed_depth.shape}")
    
    # 计算统计信息
    stats = estimator.calculate_depth_statistics(processed_depth)
    print(f"深度统计信息: {stats}")
    
    # 提取深度信息
    bbox = [100, 100, 200, 200]
    intrinsics = {
        'fx': 600, 'fy': 600,
        'cx': 320, 'cy': 240
    }
    
    depth_info = estimator.extract_depth_info(processed_depth, bbox, intrinsics)
    print(f"深度信息: {depth_info}")
    
    # 清理资源
    estimator.cleanup()
    
    print("测试完成")