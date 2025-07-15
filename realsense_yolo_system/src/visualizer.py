"""
可视化模块
提供实时检测结果显示、深度信息可视化和3D点云显示
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
from pathlib import Path

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logging.warning("Open3D不可用，3D可视化功能将被禁用")

from .data_fusion import FusedResult
from .object_detector import DetectionResult
from .depth_estimator import DepthInfo
from .utils import apply_colormap, draw_text_with_background, FPSCounter

logger = logging.getLogger(__name__)

class Visualizer:
    """可视化器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化可视化器
        
        Args:
            config: 可视化配置
        """
        self.config = config
        self.fps_counter = FPSCounter()
        
        # 显示配置
        self.show_rgb = config.get('show_rgb', True)
        self.show_depth = config.get('show_depth', True)
        self.show_detections = config.get('show_detections', True)
        self.show_3d_info = config.get('show_3d_info', True)
        
        # 可视化参数
        self.font_scale = config.get('font_scale', 0.6)
        self.font_thickness = config.get('font_thickness', 2)
        self.bbox_thickness = config.get('bbox_thickness', 2)
        self.color_map = config.get('color_map', 'jet')
        
        # 颜色配置
        self.class_colors = self._generate_class_colors()
        
        # 3D可视化
        self.point_cloud_visualizer = None
        if OPEN3D_AVAILABLE and config.get('enable_3d_visualization', False):
            self._initialize_3d_visualizer()
        
        # 保存配置
        self.save_results = config.get('save_results', False)
        self.output_dir = Path(config.get('output_dir', 'output'))
        if self.save_results:
            self.output_dir.mkdir(exist_ok=True)
        
        self.frame_count = 0
        
        logger.info("可视化器初始化完成")
    
    def _generate_class_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """生成类别颜色映射"""
        # 预定义一些常见类别的颜色
        predefined_colors = {
            'person': (255, 0, 0),      # 红色
            'car': (0, 255, 0),         # 绿色
            'bicycle': (0, 0, 255),     # 蓝色
            'motorcycle': (255, 255, 0), # 黄色
            'bus': (255, 0, 255),       # 洋红
            'truck': (0, 255, 255),     # 青色
            'bottle': (128, 0, 128),    # 紫色
            'chair': (255, 165, 0),     # 橙色
            'dog': (0, 128, 0),         # 深绿
            'cat': (128, 128, 0),       # 橄榄绿
        }
        
        # 为其他类别生成随机颜色
        np.random.seed(42)  # 确保颜色一致
        colors = {}
        
        for i in range(100):  # 支持100个类别
            if i < len(predefined_colors):
                class_name = list(predefined_colors.keys())[i]
                colors[class_name] = predefined_colors[class_name]
            else:
                # 生成随机颜色
                color = tuple(np.random.randint(0, 255, 3).tolist())
                colors[f'class_{i}'] = color
        
        return colors
    
    def _initialize_3d_visualizer(self):
        """初始化3D可视化器"""
        if not OPEN3D_AVAILABLE:
            return
        
        try:
            self.point_cloud_visualizer = o3d.visualization.Visualizer()
            self.point_cloud_visualizer.create_window("3D Point Cloud", 800, 600)
            
            # 设置渲染选项
            render_option = self.point_cloud_visualizer.get_render_option()
            render_option.point_size = 2.0
            render_option.background_color = np.array([0.1, 0.1, 0.1])
            
            logger.info("3D可视化器初始化完成")
        except Exception as e:
            logger.error(f"3D可视化器初始化失败: {e}")
            self.point_cloud_visualizer = None
    
    def draw_detections(self, image: np.ndarray, 
                       detections: List[DetectionResult]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像
            detections: 检测结果列表
            
        Returns:
            绘制后的图像
        """
        if not self.show_detections:
            return image
        
        result_image = image.copy()
        
        for detection in detections:
            # 获取颜色
            color = self.class_colors.get(detection.class_name, (255, 255, 255))
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, detection.bbox)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, self.bbox_thickness)
            
            # 绘制类别和置信度
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                       self.font_scale, self.font_thickness)[0]
            
            # 绘制标签背景
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 30
            cv2.rectangle(result_image, 
                         (x1, label_y - label_size[1] - 5), 
                         (x1 + label_size[0] + 5, label_y + 5), 
                         color, -1)
            
            # 绘制标签文本
            cv2.putText(result_image, label, (x1 + 2, label_y - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 
                       (255, 255, 255), self.font_thickness)
        
        return result_image
    
    def draw_depth_info(self, image: np.ndarray, 
                       fused_results: List[FusedResult]) -> np.ndarray:
        """
        在图像上绘制深度信息
        
        Args:
            image: 输入图像
            fused_results: 融合结果列表
            
        Returns:
            绘制后的图像
        """
        if not self.show_3d_info:
            return image
        
        result_image = image.copy()
        
        for result in fused_results:
            if not result.is_valid:
                continue
            
            # 获取边界框中心
            center_x, center_y = result.detection.center
            
            # 绘制中心点
            cv2.circle(result_image, (center_x, center_y), 5, (0, 255, 0), -1)
            
            # 绘制深度信息
            depth_text = f"D: {result.depth_info.distance:.2f}m"
            pos_text = f"3D: ({result.world_position[0]:.2f}, {result.world_position[1]:.2f}, {result.world_position[2]:.2f})"
            conf_text = f"C: {result.confidence_3d:.2f}"
            
            # 绘制文本
            text_y = center_y + 20
            draw_text_with_background(result_image, depth_text, 
                                    (center_x - 50, text_y), 
                                    self.font_scale, self.font_thickness)
            
            draw_text_with_background(result_image, pos_text, 
                                    (center_x - 50, text_y + 25), 
                                    self.font_scale, self.font_thickness)
            
            draw_text_with_background(result_image, conf_text, 
                                    (center_x - 50, text_y + 50), 
                                    self.font_scale, self.font_thickness)
            
            # 绘制跟踪ID
            if result.tracking_id:
                track_text = f"ID: {result.tracking_id}"
                draw_text_with_background(result_image, track_text, 
                                        (center_x - 50, text_y + 75), 
                                        self.font_scale, self.font_thickness)
            
            # 绘制速度信息
            if result.velocity:
                velocity_mag = np.linalg.norm(result.velocity)
                vel_text = f"V: {velocity_mag:.2f}m/s"
                draw_text_with_background(result_image, vel_text, 
                                        (center_x - 50, text_y + 100), 
                                        self.font_scale, self.font_thickness)
        
        return result_image
    
    def create_depth_visualization(self, depth_image: np.ndarray) -> np.ndarray:
        """
        创建深度图可视化
        
        Args:
            depth_image: 深度图像
            
        Returns:
            可视化的深度图
        """
        if not self.show_depth:
            return np.zeros_like(depth_image)
        
        # 应用颜色映射
        if self.color_map == 'jet':
            colormap = cv2.COLORMAP_JET
        elif self.color_map == 'hot':
            colormap = cv2.COLORMAP_HOT
        elif self.color_map == 'rainbow':
            colormap = cv2.COLORMAP_RAINBOW
        else:
            colormap = cv2.COLORMAP_JET
        
        return apply_colormap(depth_image, colormap)
    
    def draw_fps_info(self, image: np.ndarray) -> np.ndarray:
        """
        绘制FPS信息
        
        Args:
            image: 输入图像
            
        Returns:
            绘制后的图像
        """
        if not self.config.get('fps_display', True):
            return image
        
        fps = self.fps_counter.update()
        fps_text = f"FPS: {fps:.1f}"
        
        # 绘制FPS
        draw_text_with_background(image, fps_text, (10, 30), 
                                self.font_scale, self.font_thickness,
                                text_color=(0, 255, 0))
        
        return image
    
    def create_combined_view(self, color_image: np.ndarray, 
                           depth_image: np.ndarray,
                           fused_results: List[FusedResult]) -> np.ndarray:
        """
        创建组合视图
        
        Args:
            color_image: 彩色图像
            depth_image: 深度图像
            fused_results: 融合结果列表
            
        Returns:
            组合视图图像
        """
        # 处理RGB图像
        rgb_result = color_image.copy()
        if self.show_rgb:
            rgb_result = self.draw_detections(rgb_result, 
                                            [r.detection for r in fused_results])
            rgb_result = self.draw_depth_info(rgb_result, fused_results)
            rgb_result = self.draw_fps_info(rgb_result)
        
        # 处理深度图像
        depth_colored = self.create_depth_visualization(depth_image)
        
        # 在深度图上也绘制检测框
        if self.show_depth and self.show_detections:
            depth_colored = self.draw_detections(depth_colored, 
                                               [r.detection for r in fused_results])
        
        # 组合显示
        if self.show_rgb and self.show_depth:
            # 调整图像大小使其一致
            h, w = color_image.shape[:2]
            depth_colored = cv2.resize(depth_colored, (w, h))
            
            # 水平拼接
            combined = np.hstack([rgb_result, depth_colored])
            
            # 添加分割线
            line_x = w
            cv2.line(combined, (line_x, 0), (line_x, h), (255, 255, 255), 2)
            
            # 添加标签
            cv2.putText(combined, "RGB", (10, h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "Depth", (w + 10, h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return combined
        
        elif self.show_rgb:
            return rgb_result
        elif self.show_depth:
            return depth_colored
        else:
            return color_image
    
    def visualize_3d_point_cloud(self, point_cloud: np.ndarray):
        """
        可视化3D点云
        
        Args:
            point_cloud: 点云数据 [N, 6] (x, y, z, r, g, b)
        """
        if not OPEN3D_AVAILABLE or self.point_cloud_visualizer is None:
            return
        
        try:
            # 创建点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
            
            if point_cloud.shape[1] >= 6:
                # 归一化颜色
                colors = point_cloud[:, 3:6] / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 清除之前的点云
            self.point_cloud_visualizer.clear_geometries()
            
            # 添加新的点云
            self.point_cloud_visualizer.add_geometry(pcd)
            
            # 更新可视化
            self.point_cloud_visualizer.poll_events()
            self.point_cloud_visualizer.update_renderer()
            
        except Exception as e:
            logger.error(f"3D点云可视化失败: {e}")
    
    def save_visualization(self, image: np.ndarray, 
                          fused_results: List[FusedResult],
                          prefix: str = "result"):
        """
        保存可视化结果
        
        Args:
            image: 图像
            fused_results: 融合结果
            prefix: 文件前缀
        """
        if not self.save_results:
            return
        
        # 保存图像
        filename = f"{prefix}_{self.frame_count:06d}.jpg"
        filepath = self.output_dir / filename
        cv2.imwrite(str(filepath), image)
        
        # 保存检测结果
        results_data = {
            'frame': self.frame_count,
            'timestamp': time.time(),
            'detections': [result.to_dict() for result in fused_results]
        }
        
        json_filename = f"{prefix}_{self.frame_count:06d}.json"
        json_filepath = self.output_dir / json_filename
        
        import json
        with open(json_filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.frame_count += 1
    
    def create_statistics_overlay(self, image: np.ndarray, 
                                 stats: Dict[str, Any]) -> np.ndarray:
        """
        创建统计信息覆盖层
        
        Args:
            image: 输入图像
            stats: 统计信息
            
        Returns:
            添加统计信息的图像
        """
        result_image = image.copy()
        
        # 统计信息位置
        start_y = 60
        line_height = 25
        
        # 绘制统计信息
        info_texts = [
            f"Total Objects: {stats.get('total_objects', 0)}",
            f"Active Tracks: {stats.get('active_tracks', 0)}",
            f"Avg Distance: {stats.get('avg_distance', 0):.2f}m",
            f"Fusion Time: {stats.get('fusion_time', 0):.3f}s",
        ]
        
        for i, text in enumerate(info_texts):
            y_pos = start_y + i * line_height
            draw_text_with_background(result_image, text, (10, y_pos), 
                                    self.font_scale, self.font_thickness,
                                    text_color=(255, 255, 255),
                                    bg_color=(0, 0, 0))
        
        return result_image
    
    def create_distance_heatmap(self, fused_results: List[FusedResult], 
                               image_shape: Tuple[int, int]) -> np.ndarray:
        """
        创建距离热图
        
        Args:
            fused_results: 融合结果列表
            image_shape: 图像形状 (height, width)
            
        Returns:
            距离热图
        """
        h, w = image_shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        for result in fused_results:
            if not result.is_valid:
                continue
            
            # 获取边界框
            x1, y1, x2, y2 = map(int, result.detection.bbox)
            
            # 确保边界框在图像范围内
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            
            # 在热图上设置距离值
            distance = result.depth_info.distance
            heatmap[y1:y2, x1:x2] = distance
        
        # 归一化并应用颜色映射
        if np.max(heatmap) > 0:
            heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), 
                                              cv2.COLORMAP_JET)
        else:
            heatmap_colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        return heatmap_colored
    
    def create_trajectory_visualization(self, tracking_info: Dict[str, Any],
                                      image_shape: Tuple[int, int]) -> np.ndarray:
        """
        创建轨迹可视化
        
        Args:
            tracking_info: 跟踪信息
            image_shape: 图像形状
            
        Returns:
            轨迹可视化图像
        """
        h, w = image_shape
        trajectory_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        for track_id, track_info in tracking_info.items():
            if len(track_info.positions) < 2:
                continue
            
            # 获取轨迹点
            positions = list(track_info.positions)
            
            # 绘制轨迹线
            for i in range(1, len(positions)):
                # 这里需要将3D位置投影到2D图像平面
                # 简化处理，直接使用x, y坐标
                pt1 = (int(positions[i-1][0] * 100 + w//2), 
                       int(positions[i-1][1] * 100 + h//2))
                pt2 = (int(positions[i][0] * 100 + w//2), 
                       int(positions[i][1] * 100 + h//2))
                
                # 确保点在图像范围内
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    cv2.line(trajectory_image, pt1, pt2, (0, 255, 0), 2)
        
        return trajectory_image
    
    def show_interactive_controls(self):
        """显示交互控制信息"""
        controls = [
            "交互控制:",
            "q - 退出",
            "s - 保存当前帧",
            "r - 重置跟踪",
            "d - 切换深度显示",
            "c - 切换彩色显示",
            "i - 切换信息显示",
            "SPACE - 暂停/继续",
        ]
        
        print("\\n".join(controls))
    
    def handle_key_input(self, key: int) -> str:
        """
        处理键盘输入
        
        Args:
            key: 键值
            
        Returns:
            动作类型
        """
        if key == ord('q') or key == 27:  # ESC
            return 'quit'
        elif key == ord('s'):
            return 'save'
        elif key == ord('r'):
            return 'reset'
        elif key == ord('d'):
            self.show_depth = not self.show_depth
            return 'toggle_depth'
        elif key == ord('c'):
            self.show_rgb = not self.show_rgb
            return 'toggle_rgb'
        elif key == ord('i'):
            self.show_3d_info = not self.show_3d_info
            return 'toggle_info'
        elif key == ord(' '):
            return 'pause'
        
        return 'none'
    
    def cleanup(self):
        """清理资源"""
        if self.point_cloud_visualizer:
            try:
                self.point_cloud_visualizer.destroy_window()
            except:
                pass
            self.point_cloud_visualizer = None
        
        cv2.destroyAllWindows()
        logger.info("可视化器资源清理完成")

# 工厂函数
def create_visualizer(config: Dict[str, Any]) -> Visualizer:
    """
    创建可视化器
    
    Args:
        config: 可视化配置
        
    Returns:
        可视化器实例
    """
    return Visualizer(config)

# 辅助函数
def draw_grid(image: np.ndarray, grid_size: int = 50) -> np.ndarray:
    """
    绘制网格
    
    Args:
        image: 输入图像
        grid_size: 网格大小
        
    Returns:
        绘制网格的图像
    """
    result = image.copy()
    h, w = image.shape[:2]
    
    # 绘制垂直线
    for x in range(0, w, grid_size):
        cv2.line(result, (x, 0), (x, h), (128, 128, 128), 1)
    
    # 绘制水平线
    for y in range(0, h, grid_size):
        cv2.line(result, (0, y), (w, y), (128, 128, 128), 1)
    
    return result

def create_color_legend(classes: List[str], colors: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
    """
    创建颜色图例
    
    Args:
        classes: 类别列表
        colors: 颜色映射
        
    Returns:
        图例图像
    """
    legend_height = len(classes) * 30 + 20
    legend_width = 200
    
    legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    
    for i, class_name in enumerate(classes):
        y = 20 + i * 30
        color = colors.get(class_name, (255, 255, 255))
        
        # 绘制颜色块
        cv2.rectangle(legend, (10, y), (30, y + 20), color, -1)
        
        # 绘制类别名称
        cv2.putText(legend, class_name, (40, y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return legend

if __name__ == "__main__":
    # 测试可视化器
    import sys
    sys.path.append('..')
    from config.config import VISUALIZATION_CONFIG
    
    print("测试可视化器...")
    
    # 创建可视化器
    visualizer = create_visualizer(VISUALIZATION_CONFIG)
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_depth = np.random.randint(500, 2000, (480, 640), dtype=np.uint16)
    
    # 显示交互控制
    visualizer.show_interactive_controls()
    
    # 测试组合视图
    combined_view = visualizer.create_combined_view(test_image, test_depth, [])
    
    print(f"组合视图形状: {combined_view.shape}")
    
    # 清理资源
    visualizer.cleanup()
    
    print("测试完成")