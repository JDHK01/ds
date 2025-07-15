"""
统一无人机视觉API
整合RealSense视觉系统、目标跟踪和PX4无人机控制
"""

import asyncio
import cv2
import numpy as np
import time
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from queue import Queue, Empty
import json
from pathlib import Path

# 导入现有模块
from .camera_manager import CameraManager, discover_realsense_devices
from .object_detector import ObjectDetector
from .data_fusion import DataFusion
from .drone_controller import DroneController, DroneState
from .tracking_controller import TrackingController, TrackingState, TrackingTarget, create_tracking_controller
from .visualizer import Visualizer

logger = logging.getLogger(__name__)

class SystemMode(Enum):
    """系统模式枚举"""
    IDLE = "idle"
    VISION_ONLY = "vision_only"
    DRONE_READY = "drone_ready"
    TRACKING = "tracking"
    FOLLOWING = "following"
    EMERGENCY = "emergency"

class ControlStrategy(Enum):
    """控制策略枚举"""
    POSITION_HOLD = "position_hold"
    VELOCITY_CONTROL = "velocity_control"
    PREDICTIVE_CONTROL = "predictive_control"

@dataclass
class FollowingParameters:
    """跟随参数"""
    target_distance: float = 6.0  # 目标距离(米)
    max_speed: float = 2.0  # 最大速度(m/s)
    min_confidence: float = 0.4  # 最小置信度
    height_offset: float = 0.0  # 高度偏移(米)
    angle_offset: float = 0.0  # 角度偏移(度)
    
    # 控制参数
    position_p_gain: float = 0.5  # 位置P增益
    velocity_p_gain: float = 0.3  # 速度P增益
    yaw_p_gain: float = 0.5  # 偏航P增益
    
    # 安全参数
    safety_radius: float = 2.0  # 安全半径(米)
    max_vertical_speed: float = 1.0  # 最大垂直速度(m/s)
    max_yaw_rate: float = 30.0  # 最大偏航速度(deg/s)
    
    # 滤波参数
    position_filter_alpha: float = 0.8  # 位置滤波系数
    velocity_filter_alpha: float = 0.6  # 速度滤波系数
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'target_distance': self.target_distance,
            'max_speed': self.max_speed,
            'min_confidence': self.min_confidence,
            'height_offset': self.height_offset,
            'angle_offset': self.angle_offset,
            'position_p_gain': self.position_p_gain,
            'velocity_p_gain': self.velocity_p_gain,
            'yaw_p_gain': self.yaw_p_gain,
            'safety_radius': self.safety_radius,
            'max_vertical_speed': self.max_vertical_speed,
            'max_yaw_rate': self.max_yaw_rate,
            'position_filter_alpha': self.position_filter_alpha,
            'velocity_filter_alpha': self.velocity_filter_alpha
        }

class PIDController:
    """PID控制器"""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        
    def update(self, error: float, dt: Optional[float] = None) -> float:
        """更新PID控制器"""
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_time
        
        if dt <= 0:
            return 0.0
        
        # 积分项
        self.integral += error * dt
        
        # 微分项
        derivative = (error - self.last_error) / dt
        
        # PID输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # 更新状态
        self.last_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """重置PID控制器"""
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

class UnifiedDroneVisionAPI:
    """统一无人机视觉API"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化统一API
        
        Args:
            config: 配置字典，包含camera, yolo, drone, tracking等配置
        """
        self.config = config
        self.mode = SystemMode.IDLE
        self.control_strategy = ControlStrategy.VELOCITY_CONTROL
        
        # 核心组件
        self.camera_manager: Optional[CameraManager] = None
        self.object_detector: Optional[ObjectDetector] = None
        self.data_fusion: Optional[DataFusion] = None
        self.drone_controller: Optional[DroneController] = None
        self.tracking_controller: Optional[TrackingController] = None
        self.visualizer: Optional[Visualizer] = None
        
        # 系统状态
        self.is_running = False
        self.is_vision_active = False
        self.is_drone_connected = False
        self.is_following = False
        
        # 跟随控制
        self.following_params = FollowingParameters()
        self.current_target: Optional[TrackingTarget] = None
        self.follow_task: Optional[asyncio.Task] = None
        
        # PID控制器
        self.pid_controllers = {
            'x': PIDController(kp=0.5, ki=0.0, kd=0.1),
            'y': PIDController(kp=0.5, ki=0.0, kd=0.1),
            'z': PIDController(kp=0.3, ki=0.0, kd=0.05),
            'yaw': PIDController(kp=0.5, ki=0.0, kd=0.1)
        }
        
        # 数据流
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=5)
        self.telemetry_queue = Queue(maxsize=10)
        
        # 回调函数
        self.callbacks: Dict[str, List[Callable]] = {
            'on_detection': [],
            'on_tracking_update': [],
            'on_target_lost': [],
            'on_follow_start': [],
            'on_follow_stop': [],
            'on_emergency': [],
            'on_mode_change': [],
            'on_error': []
        }
        
        # 线程控制
        self.processing_thread: Optional[threading.Thread] = None
        self.telemetry_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # 性能监控
        self.performance_stats = {
            'frame_count': 0,
            'detection_count': 0,
            'tracking_count': 0,
            'follow_count': 0,
            'error_count': 0,
            'start_time': time.time()
        }
        
        # 滤波器
        self.position_filter = {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0
        }
        
        self.velocity_filter = {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0
        }
        
        logger.info("统一无人机视觉API初始化完成")
    
    def initialize(self) -> bool:
        """
        初始化所有组件
        
        Returns:
            是否初始化成功
        """
        try:
            logger.info("正在初始化系统组件...")
            
            # 初始化相机管理器
            if 'camera' in self.config:
                self.camera_manager = CameraManager(self.config['camera'])
                if not self.camera_manager.initialize():
                    logger.error("相机初始化失败")
                    return False
                logger.info("相机初始化成功")
            
            # 初始化目标检测器
            if 'yolo' in self.config:
                self.object_detector = ObjectDetector(self.config['yolo'])
                if not self.object_detector.initialize():
                    logger.error("目标检测器初始化失败")
                    return False
                logger.info("目标检测器初始化成功")
            
            # 初始化数据融合器
            if 'fusion' in self.config:
                self.data_fusion = DataFusion(self.config['fusion'])
                logger.info("数据融合器初始化成功")
            
            # 初始化跟踪控制器
            if 'tracking' in self.config:
                self.tracking_controller = create_tracking_controller(self.config['tracking'])
                logger.info("跟踪控制器初始化成功")
            
            # 初始化可视化器
            if 'visualization' in self.config:
                self.visualizer = Visualizer(self.config['visualization'])
                logger.info("可视化器初始化成功")
            
            # 初始化无人机控制器
            if 'drone' in self.config:
                self.drone_controller = DroneController(self.config['drone'])
                # 注册无人机事件回调
                self._register_drone_callbacks()
                logger.info("无人机控制器初始化成功")
            
            # 更新跟随参数
            if 'following' in self.config:
                self._update_following_params(self.config['following'])
            
            self.mode = SystemMode.VISION_ONLY
            self._trigger_callbacks('on_mode_change', self.mode)
            
            logger.info("所有组件初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            self._trigger_callbacks('on_error', f"系统初始化失败: {e}")
            return False
    
    def _register_drone_callbacks(self):
        """注册无人机事件回调"""
        if self.drone_controller is None:
            return
        
        self.drone_controller.register_callback('on_connection_changed', self._on_drone_connection_changed)
        self.drone_controller.register_callback('on_armed_changed', self._on_drone_armed_changed)
        self.drone_controller.register_callback('on_emergency', self._on_drone_emergency)
        self.drone_controller.register_callback('on_safety_warning', self._on_drone_safety_warning)
    
    def _on_drone_connection_changed(self, connected: bool):
        """无人机连接状态改变"""
        self.is_drone_connected = connected
        if connected:
            if self.mode == SystemMode.VISION_ONLY:
                self.mode = SystemMode.DRONE_READY
                self._trigger_callbacks('on_mode_change', self.mode)
        else:
            if self.mode in [SystemMode.DRONE_READY, SystemMode.FOLLOWING]:
                self.mode = SystemMode.VISION_ONLY
                self._trigger_callbacks('on_mode_change', self.mode)
    
    def _on_drone_armed_changed(self, armed: bool):
        """无人机解锁状态改变"""
        logger.info(f"无人机解锁状态: {'已解锁' if armed else '已上锁'}")
    
    def _on_drone_emergency(self, emergency_type: str):
        """无人机紧急事件"""
        logger.critical(f"无人机紧急事件: {emergency_type}")
        self.mode = SystemMode.EMERGENCY
        self._trigger_callbacks('on_emergency', emergency_type)
        
        # 停止跟随
        if self.is_following:
            asyncio.create_task(self.stop_following())
    
    def _on_drone_safety_warning(self, warning_type: str):
        """无人机安全警告"""
        logger.warning(f"无人机安全警告: {warning_type}")
    
    async def connect_drone(self, system_address: str = "udp://:14540") -> bool:
        """
        连接无人机
        
        Args:
            system_address: 系统地址
            
        Returns:
            连接是否成功
        """
        if self.drone_controller is None:
            logger.error("无人机控制器未初始化")
            return False
        
        try:
            success = await self.drone_controller.connect(system_address)
            if success:
                self.is_drone_connected = True
                self.mode = SystemMode.DRONE_READY
                self._trigger_callbacks('on_mode_change', self.mode)
                logger.info("无人机连接成功")
                
                # 启动遥测监控
                self._start_telemetry_monitoring()
                
            return success
            
        except Exception as e:
            logger.error(f"连接无人机失败: {e}")
            self._trigger_callbacks('on_error', f"连接无人机失败: {e}")
            return False
    
    async def disconnect_drone(self):
        """断开无人机连接"""
        if self.drone_controller is None:
            return
        
        try:
            # 停止跟随
            if self.is_following:
                await self.stop_following()
            
            # 断开连接
            await self.drone_controller.disconnect()
            self.is_drone_connected = False
            
            if self.mode in [SystemMode.DRONE_READY, SystemMode.FOLLOWING]:
                self.mode = SystemMode.VISION_ONLY
                self._trigger_callbacks('on_mode_change', self.mode)
            
            logger.info("无人机连接已断开")
            
        except Exception as e:
            logger.error(f"断开无人机连接失败: {e}")
            self._trigger_callbacks('on_error', f"断开无人机连接失败: {e}")
    
    def start_vision_system(self) -> bool:
        """
        启动视觉系统
        
        Returns:
            是否启动成功
        """
        try:
            if self.camera_manager is None:
                logger.error("相机管理器未初始化")
                return False
            
            # 启动相机流
            if not self.camera_manager.start_streaming(threaded=True):
                logger.error("启动相机流失败")
                return False
            
            # 启动处理线程
            self.is_running = True
            self.is_vision_active = True
            self.processing_thread = threading.Thread(target=self._vision_processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            logger.info("视觉系统启动成功")
            return True
            
        except Exception as e:
            logger.error(f"启动视觉系统失败: {e}")
            self._trigger_callbacks('on_error', f"启动视觉系统失败: {e}")
            return False
    
    def stop_vision_system(self):
        """停止视觉系统"""
        try:
            self.is_running = False
            self.is_vision_active = False
            self.stop_event.set()
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)
            
            if self.camera_manager:
                self.camera_manager.cleanup()
            
            logger.info("视觉系统已停止")
            
        except Exception as e:
            logger.error(f"停止视觉系统失败: {e}")
            self._trigger_callbacks('on_error', f"停止视觉系统失败: {e}")
    
    def _vision_processing_loop(self):
        """视觉处理循环"""
        while self.is_running and not self.stop_event.is_set():
            try:
                # 获取帧数据
                frame_data = self.camera_manager.get_latest_frame()
                if not frame_data or not frame_data.is_valid():
                    time.sleep(0.01)
                    continue
                
                # 处理帧
                result = self._process_frame(frame_data)
                
                # 将结果放入队列
                try:
                    self.result_queue.put(result, timeout=0.01)
                except:
                    # 队列满了，丢弃旧结果
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put(result, timeout=0.01)
                    except:
                        pass
                
                # 更新性能统计
                self.performance_stats['frame_count'] += 1
                
            except Exception as e:
                logger.error(f"视觉处理循环错误: {e}")
                self.performance_stats['error_count'] += 1
                self._trigger_callbacks('on_error', f"视觉处理错误: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame_data) -> Dict[str, Any]:
        """处理单帧数据"""
        try:
            # 目标检测
            detections = []
            if self.object_detector:
                detections = self.object_detector.detect(frame_data.color_image)
                self.performance_stats['detection_count'] += len(detections)
            
            # 数据融合
            fused_results = []
            if self.data_fusion and len(detections) > 0:
                fused_results = self.data_fusion.fuse_detections_with_depth(
                    detections, 
                    frame_data.depth_image,
                    self.camera_manager.intrinsics.to_dict()
                )
            
            # 更新跟踪
            tracking_targets = []
            if self.tracking_controller:
                tracking_targets = self.tracking_controller.update_tracking(
                    frame_data.color_image, frame_data.depth_image)
                self.performance_stats['tracking_count'] += len(tracking_targets)
                
                # 如果跟踪丢失，尝试重新初始化
                if len(tracking_targets) == 0 and self.tracking_controller.state == TrackingState.LOST:
                    self._attempt_reinitialization(frame_data, fused_results)
            
            # 调用回调函数
            self._trigger_callbacks('on_detection', detections)
            if tracking_targets:
                for target in tracking_targets:
                    self._trigger_callbacks('on_tracking_update', target)
            
            # 更新当前目标
            if tracking_targets:
                active_target = self.tracking_controller.get_active_target()
                if active_target:
                    self.current_target = active_target
            
            return {
                'frame_data': frame_data,
                'detections': detections,
                'fused_results': fused_results,
                'tracking_targets': tracking_targets,
                'current_target': self.current_target,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"处理帧失败: {e}")
            self.performance_stats['error_count'] += 1
            self._trigger_callbacks('on_error', f"处理帧失败: {e}")
            return {
                'frame_data': frame_data,
                'detections': [],
                'fused_results': [],
                'tracking_targets': [],
                'current_target': None,
                'timestamp': time.time()
            }
    
    def _attempt_reinitialization(self, frame_data, fused_results):
        """尝试重新初始化跟踪"""
        try:
            if not fused_results:
                return
            
            # 选择最佳目标重新初始化
            best_target = max(fused_results, key=lambda x: x.detection.confidence)
            
            detection = best_target.detection
            bbox = (int(detection.bbox[0]), int(detection.bbox[1]), 
                   int(detection.width), int(detection.height))
            
            class_info = {
                'class_id': detection.class_id,
                'class_name': detection.class_name
            }
            
            success = self.tracking_controller.start_tracking(
                frame_data.color_image, bbox, class_info=class_info)
            
            if success:
                logger.info("跟踪重新初始化成功")
            
        except Exception as e:
            logger.error(f"重新初始化跟踪失败: {e}")
    
    def start_tracking(self, target_bbox: Tuple[int, int, int, int], 
                      target_id: Optional[int] = None, 
                      class_info: Optional[Dict] = None) -> bool:
        """
        开始跟踪指定目标
        
        Args:
            target_bbox: 目标边界框 (x, y, w, h)
            target_id: 目标ID
            class_info: 分类信息
            
        Returns:
            是否成功开始跟踪
        """
        try:
            if not self.is_vision_active:
                logger.error("视觉系统未启动")
                return False
            
            if self.tracking_controller is None:
                logger.error("跟踪控制器未初始化")
                return False
            
            # 获取当前帧
            frame_data = self.camera_manager.get_latest_frame()
            if not frame_data or not frame_data.is_valid():
                logger.error("无法获取有效帧")
                return False
            
            # 开始跟踪
            success = self.tracking_controller.start_tracking(
                frame_data.color_image, target_bbox, target_id, class_info)
            
            if success:
                self.mode = SystemMode.TRACKING
                self._trigger_callbacks('on_mode_change', self.mode)
                logger.info(f"开始跟踪目标 {target_id or 'auto'}")
                
            return success
            
        except Exception as e:
            logger.error(f"开始跟踪失败: {e}")
            self._trigger_callbacks('on_error', f"开始跟踪失败: {e}")
            return False
    
    def stop_tracking(self, target_id: Optional[int] = None):
        """停止跟踪"""
        try:
            if self.tracking_controller:
                self.tracking_controller.stop_tracking(target_id)
            
            self.current_target = None
            
            if self.mode == SystemMode.TRACKING:
                self.mode = SystemMode.VISION_ONLY if not self.is_drone_connected else SystemMode.DRONE_READY
                self._trigger_callbacks('on_mode_change', self.mode)
            
            logger.info("停止跟踪")
            
        except Exception as e:
            logger.error(f"停止跟踪失败: {e}")
            self._trigger_callbacks('on_error', f"停止跟踪失败: {e}")
    
    async def start_following(self, target_bbox: Tuple[int, int, int, int], 
                             follow_params: Optional[FollowingParameters] = None,
                             target_id: Optional[int] = None,
                             class_info: Optional[Dict] = None) -> bool:
        """
        开始跟随目标
        
        Args:
            target_bbox: 目标边界框
            follow_params: 跟随参数
            target_id: 目标ID
            class_info: 分类信息
            
        Returns:
            是否成功开始跟随
        """
        try:
            if not self.is_drone_connected:
                logger.error("无人机未连接")
                return False
            
            if self.drone_controller.state != DroneState.FLYING:
                logger.error("无人机未在飞行状态")
                return False
            
            # 更新跟随参数
            if follow_params:
                self.following_params = follow_params
            
            # 开始跟踪
            if not self.start_tracking(target_bbox, target_id, class_info):
                logger.error("开始跟踪失败")
                return False
            
            # 重置PID控制器
            for controller in self.pid_controllers.values():
                controller.reset()
            
            # 启动跟随任务
            self.follow_task = asyncio.create_task(self._follow_target_loop())
            self.is_following = True
            self.mode = SystemMode.FOLLOWING
            
            self._trigger_callbacks('on_follow_start', self.following_params)
            self._trigger_callbacks('on_mode_change', self.mode)
            
            logger.info("开始跟随目标")
            return True
            
        except Exception as e:
            logger.error(f"开始跟随失败: {e}")
            self._trigger_callbacks('on_error', f"开始跟随失败: {e}")
            return False
    
    async def stop_following(self):
        """停止跟随"""
        try:
            if self.follow_task:
                self.follow_task.cancel()
                try:
                    await self.follow_task
                except asyncio.CancelledError:
                    pass
                self.follow_task = None
            
            self.is_following = False
            
            # 停止跟踪
            self.stop_tracking()
            
            # 悬停
            if self.drone_controller and self.drone_controller.is_flying:
                await self.drone_controller.hold_position()
            
            self.mode = SystemMode.DRONE_READY
            self._trigger_callbacks('on_follow_stop', None)
            self._trigger_callbacks('on_mode_change', self.mode)
            
            logger.info("停止跟随")
            
        except Exception as e:
            logger.error(f"停止跟随失败: {e}")
            self._trigger_callbacks('on_error', f"停止跟随失败: {e}")
    
    async def _follow_target_loop(self):
        """跟随目标循环"""
        control_rate = 20  # 20Hz控制频率
        control_interval = 1.0 / control_rate
        
        while self.is_following and self.mode == SystemMode.FOLLOWING:
            try:
                loop_start = time.time()
                
                # 获取跟踪目标
                if self.current_target is None:
                    # 目标丢失，悬停等待
                    await self.drone_controller.hold_position()
                    self._trigger_callbacks('on_target_lost', None)
                    
                    # 等待目标重新出现
                    await asyncio.sleep(0.1)
                    continue
                
                # 检查目标置信度
                if self.current_target.confidence < self.following_params.min_confidence:
                    logger.warning(f"目标置信度过低: {self.current_target.confidence}")
                    await asyncio.sleep(0.1)
                    continue
                
                # 计算控制命令
                control_command = self._calculate_follow_command(self.current_target)
                
                if control_command:
                    # 应用安全限制
                    control_command = self._apply_safety_limits(control_command)
                    
                    # 执行控制命令
                    if self.control_strategy == ControlStrategy.VELOCITY_CONTROL:
                        await self.drone_controller.set_velocity_body(
                            control_command['forward'],
                            control_command['right'],
                            control_command['down'],
                            control_command['yaw_rate']
                        )
                    elif self.control_strategy == ControlStrategy.POSITION_HOLD:
                        # 基于位置的控制
                        await self.drone_controller.goto_position_ned(
                            control_command['north'],
                            control_command['east'],
                            control_command['down'],
                            control_command['yaw']
                        )
                    
                    # 更新性能统计
                    self.performance_stats['follow_count'] += 1
                
                # 控制频率限制
                elapsed = time.time() - loop_start
                if elapsed < control_interval:
                    await asyncio.sleep(control_interval - elapsed)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"跟随目标循环错误: {e}")
                self.performance_stats['error_count'] += 1
                self._trigger_callbacks('on_error', f"跟随控制错误: {e}")
                await asyncio.sleep(0.1)
    
    def _calculate_follow_command(self, target: TrackingTarget) -> Optional[Dict[str, float]]:
        """计算跟随控制命令"""
        try:
            if target.depth <= 0:
                return None
            
            # 获取相机内参
            intrinsics = self.camera_manager.intrinsics
            
            # 图像中心
            image_center_x = intrinsics.width / 2
            image_center_y = intrinsics.height / 2
            
            # 目标在图像中的偏移
            target_x, target_y = target.center
            offset_x = target_x - image_center_x
            offset_y = target_y - image_center_y
            
            # 归一化偏移
            norm_offset_x = offset_x / intrinsics.width
            norm_offset_y = offset_y / intrinsics.height
            
            # 距离误差
            distance_error = target.depth - self.following_params.target_distance
            
            # 使用PID控制器
            forward_speed = self.pid_controllers['x'].update(distance_error)
            right_speed = self.pid_controllers['y'].update(norm_offset_x * target.depth)
            down_speed = self.pid_controllers['z'].update(norm_offset_y * target.depth)
            yaw_rate = self.pid_controllers['yaw'].update(norm_offset_x * 50)  # 偏航控制
            
            # 应用滤波器
            self.position_filter['x'] = (self.following_params.position_filter_alpha * forward_speed + 
                                        (1 - self.following_params.position_filter_alpha) * self.position_filter['x'])
            self.position_filter['y'] = (self.following_params.position_filter_alpha * right_speed + 
                                        (1 - self.following_params.position_filter_alpha) * self.position_filter['y'])
            self.position_filter['z'] = (self.following_params.position_filter_alpha * down_speed + 
                                        (1 - self.following_params.position_filter_alpha) * self.position_filter['z'])
            
            # 添加高度偏移
            down_speed += self.following_params.height_offset
            
            return {
                'forward': self.position_filter['x'],
                'right': self.position_filter['y'],
                'down': self.position_filter['z'],
                'yaw_rate': yaw_rate,
                'north': 0.0,  # 用于位置控制
                'east': 0.0,
                'yaw': 0.0
            }
            
        except Exception as e:
            logger.error(f"计算跟随命令失败: {e}")
            return None
    
    def _apply_safety_limits(self, command: Dict[str, float]) -> Dict[str, float]:
        """应用安全限制"""
        try:
            # 速度限制
            max_speed = self.following_params.max_speed
            command['forward'] = np.clip(command['forward'], -max_speed, max_speed)
            command['right'] = np.clip(command['right'], -max_speed, max_speed)
            command['down'] = np.clip(command['down'], 
                                     -self.following_params.max_vertical_speed, 
                                     self.following_params.max_vertical_speed)
            
            # 偏航速度限制
            command['yaw_rate'] = np.clip(command['yaw_rate'], 
                                         -self.following_params.max_yaw_rate, 
                                         self.following_params.max_yaw_rate)
            
            # 安全距离检查
            if self.current_target and self.current_target.depth < self.following_params.safety_radius:
                # 目标太近，后退
                command['forward'] = min(command['forward'], -0.5)
            
            return command
            
        except Exception as e:
            logger.error(f"应用安全限制失败: {e}")
            return command
    
    def _start_telemetry_monitoring(self):
        """启动遥测监控"""
        try:
            self.telemetry_thread = threading.Thread(target=self._telemetry_monitoring_loop)
            self.telemetry_thread.daemon = True
            self.telemetry_thread.start()
            
        except Exception as e:
            logger.error(f"启动遥测监控失败: {e}")
    
    def _telemetry_monitoring_loop(self):
        """遥测监控循环"""
        while self.is_drone_connected and not self.stop_event.is_set():
            try:
                if self.drone_controller:
                    telemetry = self.drone_controller.get_telemetry()
                    safety_status = self.drone_controller.get_safety_status()
                    
                    telemetry_data = {
                        'telemetry': telemetry,
                        'safety_status': safety_status,
                        'timestamp': time.time()
                    }
                    
                    # 将遥测数据放入队列
                    try:
                        self.telemetry_queue.put(telemetry_data, timeout=0.01)
                    except:
                        try:
                            self.telemetry_queue.get_nowait()
                            self.telemetry_queue.put(telemetry_data, timeout=0.01)
                        except:
                            pass
                
                time.sleep(0.1)  # 10Hz遥测频率
                
            except Exception as e:
                logger.error(f"遥测监控错误: {e}")
                time.sleep(0.1)
    
    def _update_following_params(self, params: Dict[str, Any]):
        """更新跟随参数"""
        try:
            for key, value in params.items():
                if hasattr(self.following_params, key):
                    setattr(self.following_params, key, value)
            
            # 更新PID控制器参数
            if 'position_p_gain' in params:
                for axis in ['x', 'y', 'z']:
                    self.pid_controllers[axis].kp = params['position_p_gain']
            
            if 'velocity_p_gain' in params:
                # 可以添加速度控制器
                pass
            
            logger.info("跟随参数已更新")
            
        except Exception as e:
            logger.error(f"更新跟随参数失败: {e}")
    
    def register_callback(self, event: str, callback: Callable):
        """注册回调函数"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            logger.warning(f"未知事件类型: {event}")
    
    def _trigger_callbacks(self, event: str, data: Any):
        """触发回调函数"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"回调函数错误 ({event}): {e}")
    
    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """获取最新的处理结果"""
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None
    
    def get_latest_telemetry(self) -> Optional[Dict[str, Any]]:
        """获取最新的遥测数据"""
        try:
            return self.telemetry_queue.get_nowait()
        except Empty:
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'mode': self.mode.value,
            'control_strategy': self.control_strategy.value,
            'is_running': self.is_running,
            'is_vision_active': self.is_vision_active,
            'is_drone_connected': self.is_drone_connected,
            'is_following': self.is_following,
            'drone_status': self.drone_controller.get_telemetry() if self.drone_controller else None,
            'tracking_status': self.tracking_controller.get_tracking_status() if self.tracking_controller else None,
            'following_params': self.following_params.to_dict(),
            'current_target': self.current_target.to_dict() if self.current_target else None,
            'performance_stats': self.performance_stats.copy()
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        uptime = time.time() - self.performance_stats['start_time']
        
        return {
            'uptime': uptime,
            'frame_rate': self.performance_stats['frame_count'] / uptime if uptime > 0 else 0,
            'detection_rate': self.performance_stats['detection_count'] / uptime if uptime > 0 else 0,
            'tracking_rate': self.performance_stats['tracking_count'] / uptime if uptime > 0 else 0,
            'follow_rate': self.performance_stats['follow_count'] / uptime if uptime > 0 else 0,
            'error_rate': self.performance_stats['error_count'] / uptime if uptime > 0 else 0,
            'total_frames': self.performance_stats['frame_count'],
            'total_detections': self.performance_stats['detection_count'],
            'total_tracks': self.performance_stats['tracking_count'],
            'total_follows': self.performance_stats['follow_count'],
            'total_errors': self.performance_stats['error_count']
        }
    
    def set_control_strategy(self, strategy: ControlStrategy):
        """设置控制策略"""
        self.control_strategy = strategy
        logger.info(f"控制策略已设置为: {strategy.value}")
    
    def update_following_parameters(self, params: Dict[str, Any]):
        """更新跟随参数"""
        self._update_following_params(params)
    
    async def emergency_stop(self):
        """紧急停止"""
        try:
            logger.critical("执行紧急停止")
            self.mode = SystemMode.EMERGENCY
            
            # 停止跟随
            if self.is_following:
                await self.stop_following()
            
            # 停止跟踪
            self.stop_tracking()
            
            # 无人机紧急停止
            if self.drone_controller:
                await self.drone_controller.emergency_stop()
            
            self._trigger_callbacks('on_emergency', 'system_emergency_stop')
            self._trigger_callbacks('on_mode_change', self.mode)
            
        except Exception as e:
            logger.error(f"紧急停止失败: {e}")
            self._trigger_callbacks('on_error', f"紧急停止失败: {e}")
    
    def save_configuration(self, filepath: str):
        """保存配置"""
        try:
            config_data = {
                'config': self.config,
                'following_params': self.following_params.to_dict(),
                'control_strategy': self.control_strategy.value,
                'performance_stats': self.get_performance_stats()
            }
            
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"配置已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def load_configuration(self, filepath: str) -> bool:
        """加载配置"""
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            # 更新配置
            if 'following_params' in config_data:
                self._update_following_params(config_data['following_params'])
            
            if 'control_strategy' in config_data:
                self.control_strategy = ControlStrategy(config_data['control_strategy'])
            
            logger.info(f"配置已从 {filepath} 加载")
            return True
            
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return False
    
    def cleanup(self):
        """清理资源"""
        try:
            logger.info("正在清理统一API资源...")
            
            # 停止所有活动
            self.stop_vision_system()
            
            # 停止跟随
            if self.is_following:
                # 在异步上下文中运行
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.stop_following())
                loop.close()
            
            # 清理组件
            if self.camera_manager:
                self.camera_manager.cleanup()
            
            if self.object_detector:
                self.object_detector.cleanup()
            
            if self.data_fusion:
                self.data_fusion.cleanup()
            
            if self.tracking_controller:
                self.tracking_controller.cleanup()
            
            if self.visualizer:
                self.visualizer.cleanup()
            
            if self.drone_controller:
                # 异步清理
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.drone_controller.cleanup())
                loop.close()
            
            logger.info("统一API资源清理完成")
            
        except Exception as e:
            logger.error(f"统一API资源清理失败: {e}")

# 便捷函数
def create_unified_api(config: Dict[str, Any]) -> UnifiedDroneVisionAPI:
    """创建统一API实例"""
    return UnifiedDroneVisionAPI(config)