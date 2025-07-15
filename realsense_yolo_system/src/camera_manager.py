"""
相机管理模块
负责Intel RealSense D435i相机的初始化、配置、数据获取等功能
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from contextlib import contextmanager

try:
    import pyrealsense2 as rs
except ImportError:
    raise ImportError("请安装pyrealsense2: pip install pyrealsense2")

from .utils import Timer, FPSCounter, CircularBuffer

logger = logging.getLogger(__name__)

@dataclass
class CameraIntrinsics:
    """相机内参数据结构"""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            'width': self.width,
            'height': self.height,
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy
        }

@dataclass
class FrameData:
    """帧数据结构"""
    color_image: np.ndarray
    depth_image: np.ndarray
    timestamp: float
    frame_number: int
    
    def is_valid(self) -> bool:
        """检查帧数据是否有效"""
        return (self.color_image is not None and 
                self.depth_image is not None and
                self.color_image.size > 0 and
                self.depth_image.size > 0)

class CameraManager:
    """Intel RealSense D435i相机管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化相机管理器
        
        Args:
            config: 相机配置字典
        """
        self.config = config
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        self.intrinsics = None
        self.is_streaming = False
        self.frame_count = 0
        
        # 性能监控
        self.fps_counter = FPSCounter()
        self.frame_buffer = CircularBuffer(config.get('buffer_size', 5))
        
        # 线程控制
        self.capture_thread = None
        self.capture_lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # 后处理过滤器
        self.filters = {}
        self._setup_filters()
        
        logger.info("相机管理器初始化完成")
    
    def _setup_filters(self):
        """设置深度图后处理过滤器"""
        try:
            # 抽取滤波器（降采样）
            self.filters['decimation'] = rs.decimation_filter()
            self.filters['decimation'].set_option(rs.option.filter_magnitude, 2)
            
            # 空间滤波器
            self.filters['spatial'] = rs.spatial_filter()
            self.filters['spatial'].set_option(rs.option.filter_magnitude, 2)
            self.filters['spatial'].set_option(rs.option.filter_smooth_alpha, 0.5)
            self.filters['spatial'].set_option(rs.option.filter_smooth_delta, 20)
            
            # 时间滤波器
            self.filters['temporal'] = rs.temporal_filter()
            self.filters['temporal'].set_option(rs.option.filter_smooth_alpha, 0.4)
            self.filters['temporal'].set_option(rs.option.filter_smooth_delta, 20)
            
            # 孔洞填充滤波器
            self.filters['hole_filling'] = rs.hole_filling_filter()
            
            logger.info("深度图后处理滤波器设置完成")
        except Exception as e:
            logger.warning(f"设置滤波器时出错: {e}")
    
    def _apply_filters(self, depth_frame):
        """应用深度图后处理滤波器"""
        try:
            # 按顺序应用过滤器
            filtered_frame = depth_frame
            
            if self.config.get('decimation', {}).get('enable', False):
                filtered_frame = self.filters['decimation'].process(filtered_frame)
            
            if self.config.get('spatial_filter', {}).get('enable', False):
                filtered_frame = self.filters['spatial'].process(filtered_frame)
            
            if self.config.get('temporal_filter', {}).get('enable', False):
                filtered_frame = self.filters['temporal'].process(filtered_frame)
            
            if self.config.get('hole_filling', {}).get('enable', False):
                filtered_frame = self.filters['hole_filling'].process(filtered_frame)
            
            return filtered_frame
        except Exception as e:
            logger.warning(f"应用过滤器时出错: {e}")
            return depth_frame
    
    def initialize(self) -> bool:
        """
        初始化相机
        
        Returns:
            是否初始化成功
        """
        try:
            # 检查设备连接
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) == 0:
                logger.error("未检测到RealSense设备")
                return False
            
            # 打印设备信息
            for device in devices:
                logger.info(f"检测到设备: {device.get_info(rs.camera_info.name)}")
                logger.info(f"序列号: {device.get_info(rs.camera_info.serial_number)}")
            
            # 创建pipeline
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # 设置设备ID（如果指定）
            if self.config.get('device_id'):
                config.enable_device(self.config['device_id'])
            
            # 配置流
            if self.config.get('enable_color', True):
                config.enable_stream(
                    rs.stream.color,
                    self.config['width'],
                    self.config['height'],
                    rs.format.bgr8,
                    self.config['fps']
                )
            
            if self.config.get('enable_depth', True):
                config.enable_stream(
                    rs.stream.depth,
                    self.config['width'],
                    self.config['height'],
                    rs.format.z16,
                    self.config['fps']
                )
            
            if self.config.get('enable_infrared', False):
                config.enable_stream(rs.stream.infrared, 1, self.config['width'],
                                   self.config['height'], rs.format.y8, self.config['fps'])
                config.enable_stream(rs.stream.infrared, 2, self.config['width'],
                                   self.config['height'], rs.format.y8, self.config['fps'])
            
            # 启动pipeline
            profile = self.pipeline.start(config)
            
            # 获取深度传感器和比例
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            logger.info(f"深度比例: {self.depth_scale}")
            
            # 创建对齐对象
            if self.config.get('align_to_color', True):
                self.align = rs.align(rs.stream.color)
            
            # 获取相机内参
            self._get_intrinsics()
            
            # 设置相机参数
            self._configure_camera_settings()
            
            logger.info("相机初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"相机初始化失败: {e}")
            return False
    
    def _get_intrinsics(self):
        """获取相机内参"""
        try:
            # 获取color stream的内参
            profile = self.pipeline.get_active_profile()
            color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
            intrinsics = color_profile.get_intrinsics()
            
            self.intrinsics = CameraIntrinsics(
                width=intrinsics.width,
                height=intrinsics.height,
                fx=intrinsics.fx,
                fy=intrinsics.fy,
                cx=intrinsics.ppx,
                cy=intrinsics.ppy
            )
            
            logger.info(f"相机内参: {self.intrinsics}")
            
        except Exception as e:
            logger.error(f"获取相机内参失败: {e}")
    
    def _configure_camera_settings(self):
        """配置相机设置"""
        try:
            # 获取设备
            profile = self.pipeline.get_active_profile()
            device = profile.get_device()
            
            # 配置深度传感器
            depth_sensor = device.first_depth_sensor()
            
            # 配置RGB传感器
            color_sensor = device.first_color_sensor()
            
            # 设置自动曝光
            if self.config.get('auto_exposure', True):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            
            # 设置自动白平衡
            if self.config.get('auto_white_balance', True):
                color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
            
            # 设置激光功率（如果支持）
            if depth_sensor.supports(rs.option.laser_power):
                depth_sensor.set_option(rs.option.laser_power, 150)
            
            # 设置深度单位
            if depth_sensor.supports(rs.option.depth_units):
                depth_sensor.set_option(rs.option.depth_units, 0.001)
            
            logger.info("相机设置配置完成")
            
        except Exception as e:
            logger.warning(f"配置相机设置时出错: {e}")
    
    def start_streaming(self, threaded: bool = True) -> bool:
        """
        开始流式传输
        
        Args:
            threaded: 是否使用线程模式
            
        Returns:
            是否启动成功
        """
        if self.is_streaming:
            logger.warning("相机已经在流式传输中")
            return True
        
        if not self.pipeline:
            logger.error("相机未初始化")
            return False
        
        try:
            self.is_streaming = True
            self.stop_event.clear()
            
            if threaded:
                self.capture_thread = threading.Thread(target=self._capture_loop)
                self.capture_thread.daemon = True
                self.capture_thread.start()
                logger.info("开始线程模式流式传输")
            else:
                logger.info("开始同步模式流式传输")
            
            return True
            
        except Exception as e:
            logger.error(f"启动流式传输失败: {e}")
            self.is_streaming = False
            return False
    
    def stop_streaming(self):
        """停止流式传输"""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        self.stop_event.set()
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        logger.info("停止流式传输")
    
    def _capture_loop(self):
        """捕获循环（线程模式）"""
        while self.is_streaming and not self.stop_event.is_set():
            try:
                frame_data = self.get_frame()
                if frame_data and frame_data.is_valid():
                    self.frame_buffer.put(frame_data)
                    
            except Exception as e:
                logger.error(f"捕获帧时出错: {e}")
                time.sleep(0.01)
    
    def get_frame(self, timeout: float = 1.0) -> Optional[FrameData]:
        """
        获取一帧数据
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            帧数据或None
        """
        if not self.pipeline:
            return None
        
        try:
            # 等待帧数据
            frames = self.pipeline.wait_for_frames(int(timeout * 1000))
            
            if not frames:
                return None
            
            # 对齐帧
            if self.align:
                frames = self.align.process(frames)
            
            # 获取color和depth帧
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None
            
            # 应用深度图过滤器
            depth_frame = self._apply_filters(depth_frame)
            
            # 转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # 创建帧数据
            frame_data = FrameData(
                color_image=color_image,
                depth_image=depth_image,
                timestamp=time.time(),
                frame_number=self.frame_count
            )
            
            self.frame_count += 1
            
            # 更新FPS
            self.fps_counter.update()
            
            return frame_data
            
        except Exception as e:
            logger.error(f"获取帧数据失败: {e}")
            return None
    
    def get_latest_frame(self) -> Optional[FrameData]:
        """
        获取最新的帧数据（线程模式）
        
        Returns:
            最新帧数据或None
        """
        if not self.is_streaming:
            return self.get_frame()
        
        return self.frame_buffer.get_latest()
    
    def get_depth_at_pixel(self, x: int, y: int, depth_image: np.ndarray) -> float:
        """
        获取指定像素的深度值
        
        Args:
            x: 像素x坐标
            y: 像素y坐标
            depth_image: 深度图像
            
        Returns:
            深度值（米）
        """
        if (0 <= x < depth_image.shape[1] and 
            0 <= y < depth_image.shape[0]):
            depth_value = depth_image[y, x] * self.depth_scale
            return depth_value
        return 0.0
    
    def pixel_to_point(self, x: int, y: int, depth: float) -> Tuple[float, float, float]:
        """
        将像素坐标转换为3D坐标
        
        Args:
            x: 像素x坐标
            y: 像素y坐标
            depth: 深度值（米）
            
        Returns:
            3D坐标 (x, y, z)
        """
        if not self.intrinsics:
            return (0.0, 0.0, 0.0)
        
        # 转换到3D坐标系
        x_3d = (x - self.intrinsics.cx) * depth / self.intrinsics.fx
        y_3d = (y - self.intrinsics.cy) * depth / self.intrinsics.fy
        z_3d = depth
        
        return (x_3d, y_3d, z_3d)
    
    def point_to_pixel(self, x: float, y: float, z: float) -> Tuple[int, int]:
        """
        将3D坐标转换为像素坐标
        
        Args:
            x: 3D x坐标
            y: 3D y坐标
            z: 3D z坐标
            
        Returns:
            像素坐标 (x, y)
        """
        if not self.intrinsics or z == 0:
            return (0, 0)
        
        # 转换到像素坐标系
        pixel_x = int(x * self.intrinsics.fx / z + self.intrinsics.cx)
        pixel_y = int(y * self.intrinsics.fy / z + self.intrinsics.cy)
        
        return (pixel_x, pixel_y)
    
    def get_fps(self) -> float:
        """获取当前FPS"""
        return self.fps_counter.get_fps()
    
    def get_camera_info(self) -> Dict[str, Any]:
        """
        获取相机信息
        
        Returns:
            相机信息字典
        """
        info = {
            'is_streaming': self.is_streaming,
            'frame_count': self.frame_count,
            'fps': self.get_fps(),
            'depth_scale': self.depth_scale,
            'config': self.config,
        }
        
        if self.intrinsics:
            info['intrinsics'] = self.intrinsics.to_dict()
        
        return info
    
    def save_frame(self, frame_data: FrameData, output_dir: str = "output"):
        """
        保存帧数据
        
        Args:
            frame_data: 帧数据
            output_dir: 输出目录
        """
        import os
        from pathlib import Path
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存color图像
        color_path = Path(output_dir) / f"color_{frame_data.frame_number:06d}.jpg"
        cv2.imwrite(str(color_path), frame_data.color_image)
        
        # 保存深度图像
        depth_path = Path(output_dir) / f"depth_{frame_data.frame_number:06d}.png"
        cv2.imwrite(str(depth_path), frame_data.depth_image)
        
        logger.info(f"帧数据已保存: {color_path}, {depth_path}")
    
    @contextmanager
    def camera_context(self):
        """相机上下文管理器"""
        success = self.initialize()
        if not success:
            raise RuntimeError("相机初始化失败")
        
        try:
            yield self
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.stop_streaming()
        
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
            self.pipeline = None
        
        logger.info("相机资源清理完成")
    
    def __del__(self):
        """析构函数"""
        self.cleanup()

# 工厂函数
def create_camera_manager(config: Dict[str, Any]) -> CameraManager:
    """
    创建相机管理器
    
    Args:
        config: 相机配置
        
    Returns:
        相机管理器实例
    """
    return CameraManager(config)

# 设备发现函数
def discover_realsense_devices() -> List[Dict[str, str]]:
    """
    发现RealSense设备
    
    Returns:
        设备信息列表
    """
    devices = []
    
    try:
        ctx = rs.context()
        for device in ctx.query_devices():
            device_info = {
                'name': device.get_info(rs.camera_info.name),
                'serial_number': device.get_info(rs.camera_info.serial_number),
                'firmware_version': device.get_info(rs.camera_info.firmware_version),
                'product_id': device.get_info(rs.camera_info.product_id),
            }
            devices.append(device_info)
    except Exception as e:
        logger.error(f"发现设备时出错: {e}")
    
    return devices

if __name__ == "__main__":
    # 测试相机管理器
    from config.config import CAMERA_CONFIG
    
    print("发现的RealSense设备:")
    devices = discover_realsense_devices()
    for device in devices:
        print(f"  - {device['name']} ({device['serial_number']})")
    
    if devices:
        print("\\n测试相机管理器...")
        
        # 创建相机管理器
        camera = create_camera_manager(CAMERA_CONFIG)
        
        with camera.camera_context():
            # 开始流式传输
            camera.start_streaming(threaded=False)
            
            # 获取几帧数据
            for i in range(10):
                frame = camera.get_frame()
                if frame and frame.is_valid():
                    print(f"帧 {i}: {frame.color_image.shape}, {frame.depth_image.shape}")
                    print(f"FPS: {camera.get_fps():.2f}")
                
                time.sleep(0.1)
        
        print("测试完成")
    else:
        print("未找到RealSense设备")