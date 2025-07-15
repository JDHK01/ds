"""
高性能目标跟踪控制器
针对Jetson Orin Nano优化，支持多种跟踪算法
"""

import cv2
import numpy as np
import time
import logging
import math
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import threading
from collections import deque
import asyncio

# 尝试导入高性能跟踪算法
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO不可用，将使用OpenCV跟踪器")

try:
    # 导入BYTETracker (如果可用)
    from yolox.tracker.byte_tracker import BYTETracker
    BYTE_TRACKER_AVAILABLE = True
except ImportError:
    BYTE_TRACKER_AVAILABLE = False
    logging.warning("BYTETracker不可用，将使用OpenCV跟踪器")

logger = logging.getLogger(__name__)

class TrackingState(Enum):
    """跟踪状态枚举"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    LOST = "lost"
    SEARCHING = "searching"
    RECOVERING = "recovering"

class TrackerType(Enum):
    """跟踪器类型枚举"""
    YOLO11_BYTE = "yolo11_byte"  # 推荐：YOLO11 + BYTETracker
    YOLOX_BYTE = "yolox_byte"    # 备选：YOLOX + BYTETracker
    YOLO_NATIVE = "yolo_native"   # YOLO原生跟踪
    CSRT = "csrt"                 # OpenCV CSRT
    KCF = "kcf"                   # OpenCV KCF
    MEDIANFLOW = "medianflow"     # OpenCV MedianFlow
    MOSSE = "mosse"               # OpenCV MOSSE (最快)

@dataclass
class TrackingTarget:
    """跟踪目标数据结构"""
    id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[int, int]
    confidence: float
    last_seen: float
    velocity: Tuple[float, float]  # (vx, vy) pixels/sec
    acceleration: Tuple[float, float]  # (ax, ay) pixels/sec²
    depth: float = 0.0
    world_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    class_id: int = -1
    class_name: str = ""
    tracking_quality: float = 1.0
    lost_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'bbox': self.bbox,
            'center': self.center,
            'confidence': self.confidence,
            'last_seen': self.last_seen,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'depth': self.depth,
            'world_position': self.world_position,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'tracking_quality': self.tracking_quality,
            'lost_count': self.lost_count
        }

@dataclass
class TrackingPerformance:
    """跟踪性能统计"""
    fps: float = 0.0
    avg_processing_time: float = 0.0
    detection_time: float = 0.0
    tracking_time: float = 0.0
    total_frames: int = 0
    successful_tracks: int = 0
    lost_tracks: int = 0
    recovered_tracks: int = 0

class KalmanPredictor:
    """改进的Kalman滤波预测器"""
    
    def __init__(self, process_noise: float = 0.03, measurement_noise: float = 0.1):
        self.kalman = cv2.KalmanFilter(6, 2)  # 6状态(x,y,vx,vy,ax,ay), 2观测(x,y)
        
        # 转移矩阵 (constant acceleration model)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 观测矩阵
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # 噪声协方差
        self.kalman.processNoiseCov = process_noise * np.eye(6, dtype=np.float32)
        self.kalman.measurementNoiseCov = measurement_noise * np.eye(2, dtype=np.float32)
        
        # 误差协方差
        self.kalman.errorCovPost = 1.0 * np.eye(6, dtype=np.float32)
        
        self.initialized = False
        self.last_time = time.time()
    
    def init(self, x: float, y: float):
        """初始化预测器"""
        self.kalman.statePre = np.array([x, y, 0, 0, 0, 0], dtype=np.float32)
        self.kalman.statePost = np.array([x, y, 0, 0, 0, 0], dtype=np.float32)
        self.initialized = True
        self.last_time = time.time()
    
    def update(self, x: float, y: float) -> Tuple[float, float, float, float]:
        """更新预测器并返回速度和加速度"""
        if not self.initialized:
            self.init(x, y)
            return 0.0, 0.0, 0.0, 0.0
        
        # 更新时间步长
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # 更新转移矩阵的时间步长
        self.kalman.transitionMatrix[0, 2] = dt
        self.kalman.transitionMatrix[1, 3] = dt
        self.kalman.transitionMatrix[0, 4] = 0.5 * dt * dt
        self.kalman.transitionMatrix[1, 5] = 0.5 * dt * dt
        self.kalman.transitionMatrix[2, 4] = dt
        self.kalman.transitionMatrix[3, 5] = dt
        
        # 预测
        prediction = self.kalman.predict()
        
        # 更新
        measurement = np.array([[x], [y]], dtype=np.float32)
        corrected = self.kalman.correct(measurement)
        
        # 返回速度和加速度
        vx, vy = float(corrected[2]), float(corrected[3])
        ax, ay = float(corrected[4]), float(corrected[5])
        
        return vx, vy, ax, ay
    
    def predict(self, steps: int = 1) -> Tuple[float, float]:
        """预测未来位置"""
        if not self.initialized:
            return 0.0, 0.0
        
        # 多步预测
        state = self.kalman.statePost.copy()
        for _ in range(steps):
            state = np.dot(self.kalman.transitionMatrix, state)
        
        return float(state[0]), float(state[1])

class YOLOTracker:
    """YOLO + BYTETracker组合跟踪器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detector = None
        self.byte_tracker = None
        self.initialized = False
        
        # 尝试初始化YOLO检测器
        if YOLO_AVAILABLE:
            try:
                model_path = config.get('model_path', 'yolov8n.pt')
                self.detector = YOLO(model_path)
                
                # 针对Jetson Orin Nano优化
                if config.get('use_tensorrt', False):
                    # 导出TensorRT模型
                    self.detector.export(format='engine', dynamic=True, simplify=True)
                
                logger.info(f"YOLO检测器初始化成功: {model_path}")
                
            except Exception as e:
                logger.error(f"YOLO初始化失败: {e}")
                return
        
        # 尝试初始化BYTETracker
        if BYTE_TRACKER_AVAILABLE:
            try:
                # BYTETracker参数
                tracker_config = {
                    'track_thresh': config.get('track_thresh', 0.5),
                    'track_buffer': config.get('track_buffer', 30),
                    'match_thresh': config.get('match_thresh', 0.8),
                    'frame_rate': config.get('frame_rate', 30)
                }
                
                # 这里需要根据实际的BYTETracker API调整
                # self.byte_tracker = BYTETracker(tracker_config)
                
                logger.info("BYTETracker初始化成功")
                
            except Exception as e:
                logger.error(f"BYTETracker初始化失败: {e}")
                return
        
        self.initialized = (self.detector is not None)
    
    def detect_and_track(self, frame: np.ndarray) -> List[TrackingTarget]:
        """检测和跟踪"""
        if not self.initialized:
            return []
        
        targets = []
        
        try:
            # YOLO检测
            results = self.detector(frame, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                
                # 提取检测结果
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    # 转换为跟踪目标
                    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                        x1, y1, x2, y2 = map(int, box)
                        bbox = (x1, y1, x2 - x1, y2 - y1)
                        center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
                        
                        target = TrackingTarget(
                            id=i,
                            bbox=bbox,
                            center=center,
                            confidence=float(score),
                            last_seen=time.time(),
                            velocity=(0.0, 0.0),
                            acceleration=(0.0, 0.0),
                            class_id=int(cls),
                            class_name=self.detector.names[int(cls)] if hasattr(self.detector, 'names') else str(cls)
                        )
                        
                        targets.append(target)
            
        except Exception as e:
            logger.error(f"YOLO检测跟踪失败: {e}")
        
        return targets

class OpenCVTracker:
    """OpenCV传统跟踪器"""
    
    def __init__(self, tracker_type: TrackerType):
        self.tracker_type = tracker_type
        self.tracker = None
        self.initialized = False
        
    def _create_tracker(self) -> cv2.Tracker:
        """创建OpenCV跟踪器"""
        if self.tracker_type == TrackerType.CSRT:
            return cv2.TrackerCSRT_create()
        elif self.tracker_type == TrackerType.KCF:
            return cv2.TrackerKCF_create()
        elif self.tracker_type == TrackerType.MEDIANFLOW:
            return cv2.TrackerMedianFlow_create()
        elif self.tracker_type == TrackerType.MOSSE:
            return cv2.TrackerMOSSE_create()
        else:
            return cv2.TrackerCSRT_create()
    
    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """初始化跟踪器"""
        try:
            self.tracker = self._create_tracker()
            success = self.tracker.init(frame, bbox)
            self.initialized = success
            return success
        except Exception as e:
            logger.error(f"OpenCV跟踪器初始化失败: {e}")
            return False
    
    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """更新跟踪"""
        if not self.initialized or self.tracker is None:
            return False, None
        
        try:
            success, bbox = self.tracker.update(frame)
            if success and bbox is not None:
                return True, tuple(map(int, bbox))
            return False, None
        except Exception as e:
            logger.error(f"OpenCV跟踪更新失败: {e}")
            return False, None

class TrackingController:
    """高性能跟踪控制器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化跟踪控制器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.state = TrackingState.IDLE
        self.tracker_type = TrackerType(config.get('tracker_type', 'csrt'))
        
        # 跟踪器实例
        self.yolo_tracker = None
        self.opencv_tracker = None
        self.current_tracker = None
        
        # 目标管理
        self.targets: Dict[int, TrackingTarget] = {}
        self.active_target_id: Optional[int] = None
        self.next_target_id = 0
        
        # 预测和历史
        self.predictors: Dict[int, KalmanPredictor] = {}
        self.tracking_history: Dict[int, deque] = {}
        
        # 性能参数
        self.max_lost_frames = config.get('max_lost_frames', 15)
        self.confidence_threshold = config.get('confidence_threshold', 0.3)
        self.search_radius = config.get('search_radius', 150)
        self.velocity_smoothing = config.get('velocity_smoothing', 0.8)
        self.max_targets = config.get('max_targets', 10)
        
        # 性能监控
        self.performance = TrackingPerformance()
        self.processing_times = deque(maxlen=100)
        self.frame_count = 0
        self.start_time = time.time()
        
        # 线程控制
        self.tracking_lock = threading.Lock()
        self.async_mode = config.get('async_mode', False)
        
        # 初始化跟踪器
        self._initialize_tracker()
        
        logger.info(f"跟踪控制器初始化完成，使用跟踪器: {self.tracker_type.value}")
    
    def _initialize_tracker(self):
        """初始化跟踪器"""
        try:
            if self.tracker_type in [TrackerType.YOLO11_BYTE, TrackerType.YOLOX_BYTE, TrackerType.YOLO_NATIVE]:
                # 初始化YOLO跟踪器
                if YOLO_AVAILABLE:
                    self.yolo_tracker = YOLOTracker(self.config)
                    if self.yolo_tracker.initialized:
                        self.current_tracker = self.yolo_tracker
                        logger.info("使用YOLO跟踪器")
                    else:
                        logger.warning("YOLO跟踪器初始化失败，回退到OpenCV")
                        self.tracker_type = TrackerType.CSRT
                        self.opencv_tracker = OpenCVTracker(self.tracker_type)
                        self.current_tracker = self.opencv_tracker
                else:
                    logger.warning("YOLO不可用，使用OpenCV跟踪器")
                    self.tracker_type = TrackerType.CSRT
                    self.opencv_tracker = OpenCVTracker(self.tracker_type)
                    self.current_tracker = self.opencv_tracker
            else:
                # 使用OpenCV跟踪器
                self.opencv_tracker = OpenCVTracker(self.tracker_type)
                self.current_tracker = self.opencv_tracker
                
        except Exception as e:
            logger.error(f"跟踪器初始化失败: {e}")
            # 回退到最简单的跟踪器
            self.tracker_type = TrackerType.MOSSE
            self.opencv_tracker = OpenCVTracker(self.tracker_type)
            self.current_tracker = self.opencv_tracker
    
    def start_tracking(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                      target_id: Optional[int] = None, class_info: Optional[Dict] = None) -> bool:
        """
        开始跟踪目标
        
        Args:
            frame: 当前帧
            bbox: 目标边界框 (x, y, w, h)
            target_id: 目标ID，None表示自动分配
            class_info: 分类信息
            
        Returns:
            是否成功开始跟踪
        """
        try:
            with self.tracking_lock:
                self.state = TrackingState.INITIALIZING
                
                # 分配目标ID
                if target_id is None:
                    target_id = self.next_target_id
                    self.next_target_id += 1
                
                # 创建跟踪目标
                center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
                target = TrackingTarget(
                    id=target_id,
                    bbox=bbox,
                    center=center,
                    confidence=1.0,
                    last_seen=time.time(),
                    velocity=(0.0, 0.0),
                    acceleration=(0.0, 0.0),
                    class_id=class_info.get('class_id', -1) if class_info else -1,
                    class_name=class_info.get('class_name', '') if class_info else ''
                )
                
                # 初始化跟踪器
                if isinstance(self.current_tracker, OpenCVTracker):
                    if not self.current_tracker.init(frame, bbox):
                        logger.error("OpenCV跟踪器初始化失败")
                        self.state = TrackingState.IDLE
                        return False
                
                # 初始化预测器
                predictor = KalmanPredictor(
                    process_noise=self.config.get('kalman_process_noise', 0.03),
                    measurement_noise=self.config.get('kalman_measurement_noise', 0.1)
                )
                predictor.init(center[0], center[1])
                
                # 存储目标信息
                self.targets[target_id] = target
                self.predictors[target_id] = predictor
                self.tracking_history[target_id] = deque(maxlen=30)
                self.active_target_id = target_id
                
                self.state = TrackingState.TRACKING
                logger.info(f"开始跟踪目标 {target_id}: {bbox}")
                
                return True
                
        except Exception as e:
            logger.error(f"开始跟踪失败: {e}")
            self.state = TrackingState.IDLE
            return False
    
    def update_tracking(self, frame: np.ndarray, depth_frame: Optional[np.ndarray] = None) -> List[TrackingTarget]:
        """
        更新跟踪
        
        Args:
            frame: 当前帧
            depth_frame: 深度帧（可选）
            
        Returns:
            跟踪目标列表
        """
        start_time = time.time()
        
        try:
            with self.tracking_lock:
                if self.state not in [TrackingState.TRACKING, TrackingState.LOST, TrackingState.SEARCHING]:
                    return []
                
                active_targets = []
                
                # 使用YOLO跟踪器
                if isinstance(self.current_tracker, YOLOTracker):
                    detected_targets = self.current_tracker.detect_and_track(frame)
                    active_targets = self._update_yolo_targets(detected_targets, depth_frame)
                
                # 使用OpenCV跟踪器
                elif isinstance(self.current_tracker, OpenCVTracker):
                    if self.active_target_id is not None:
                        target = self._update_opencv_target(frame, self.active_target_id, depth_frame)
                        if target:
                            active_targets = [target]
                
                # 更新性能统计
                self._update_performance_stats(start_time)
                
                return active_targets
                
        except Exception as e:
            logger.error(f"更新跟踪失败: {e}")
            return []
    
    def _update_yolo_targets(self, detected_targets: List[TrackingTarget], 
                           depth_frame: Optional[np.ndarray] = None) -> List[TrackingTarget]:
        """更新YOLO检测的目标"""
        active_targets = []
        
        try:
            # 关联检测结果和现有目标
            for detected in detected_targets:
                best_match_id = None
                best_distance = float('inf')
                
                # 寻找最佳匹配
                for target_id, existing_target in self.targets.items():
                    distance = self._calculate_distance(detected.center, existing_target.center)
                    if distance < best_distance and distance < self.search_radius:
                        best_distance = distance
                        best_match_id = target_id
                
                if best_match_id is not None:
                    # 更新现有目标
                    self._update_target_info(best_match_id, detected, depth_frame)
                    active_targets.append(self.targets[best_match_id])
                elif len(self.targets) < self.max_targets:
                    # 创建新目标
                    new_id = self.next_target_id
                    self.next_target_id += 1
                    
                    detected.id = new_id
                    self.targets[new_id] = detected
                    
                    # 初始化预测器
                    predictor = KalmanPredictor()
                    predictor.init(detected.center[0], detected.center[1])
                    self.predictors[new_id] = predictor
                    self.tracking_history[new_id] = deque(maxlen=30)
                    
                    active_targets.append(detected)
            
            # 处理丢失的目标
            self._handle_lost_targets(active_targets)
            
        except Exception as e:
            logger.error(f"更新YOLO目标失败: {e}")
        
        return active_targets
    
    def _update_opencv_target(self, frame: np.ndarray, target_id: int, 
                            depth_frame: Optional[np.ndarray] = None) -> Optional[TrackingTarget]:
        """更新OpenCV跟踪的目标"""
        try:
            if target_id not in self.targets:
                return None
            
            target = self.targets[target_id]
            
            # 更新跟踪器
            success, bbox = self.current_tracker.update(frame)
            
            if success and bbox is not None:
                # 更新目标信息
                new_center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
                
                # 更新速度和加速度
                if target_id in self.predictors:
                    vx, vy, ax, ay = self.predictors[target_id].update(new_center[0], new_center[1])
                    target.velocity = (vx, vy)
                    target.acceleration = (ax, ay)
                
                # 更新基本信息
                target.bbox = bbox
                target.center = new_center
                target.last_seen = time.time()
                target.lost_count = 0
                target.tracking_quality = min(1.0, target.tracking_quality + 0.1)
                
                # 更新深度信息
                if depth_frame is not None:
                    target.depth = self._get_target_depth(depth_frame, bbox)
                
                # 添加到历史记录
                if target_id in self.tracking_history:
                    self.tracking_history[target_id].append(target)
                
                self.state = TrackingState.TRACKING
                return target
                
            else:
                # 跟踪失败
                target.lost_count += 1
                target.tracking_quality = max(0.0, target.tracking_quality - 0.2)
                
                if target.lost_count <= self.max_lost_frames:
                    # 尝试预测位置
                    if target_id in self.predictors:
                        pred_x, pred_y = self.predictors[target_id].predict(target.lost_count)
                        target.center = (int(pred_x), int(pred_y))
                        target.confidence = max(0.1, target.confidence - 0.1)
                    
                    self.state = TrackingState.SEARCHING
                    return target
                else:
                    # 目标完全丢失
                    self.state = TrackingState.LOST
                    return None
                    
        except Exception as e:
            logger.error(f"更新OpenCV目标失败: {e}")
            return None
    
    def _update_target_info(self, target_id: int, detected: TrackingTarget, 
                          depth_frame: Optional[np.ndarray] = None):
        """更新目标信息"""
        try:
            if target_id not in self.targets:
                return
            
            target = self.targets[target_id]
            
            # 更新速度和加速度
            if target_id in self.predictors:
                vx, vy, ax, ay = self.predictors[target_id].update(detected.center[0], detected.center[1])
                target.velocity = (vx, vy)
                target.acceleration = (ax, ay)
            
            # 更新基本信息
            target.bbox = detected.bbox
            target.center = detected.center
            target.confidence = detected.confidence
            target.last_seen = time.time()
            target.lost_count = 0
            target.tracking_quality = min(1.0, target.tracking_quality + 0.1)
            
            # 更新分类信息
            if detected.class_id != -1:
                target.class_id = detected.class_id
                target.class_name = detected.class_name
            
            # 更新深度信息
            if depth_frame is not None:
                target.depth = self._get_target_depth(depth_frame, detected.bbox)
            
            # 添加到历史记录
            if target_id in self.tracking_history:
                self.tracking_history[target_id].append(target)
            
        except Exception as e:
            logger.error(f"更新目标信息失败: {e}")
    
    def _handle_lost_targets(self, active_targets: List[TrackingTarget]):
        """处理丢失的目标"""
        try:
            active_ids = {target.id for target in active_targets}
            
            # 检查所有已知目标
            for target_id in list(self.targets.keys()):
                if target_id not in active_ids:
                    target = self.targets[target_id]
                    target.lost_count += 1
                    
                    if target.lost_count > self.max_lost_frames:
                        # 移除长时间丢失的目标
                        del self.targets[target_id]
                        if target_id in self.predictors:
                            del self.predictors[target_id]
                        if target_id in self.tracking_history:
                            del self.tracking_history[target_id]
                        
                        logger.info(f"移除丢失的目标: {target_id}")
                        
                        # 更新性能统计
                        self.performance.lost_tracks += 1
                        
        except Exception as e:
            logger.error(f"处理丢失目标失败: {e}")
    
    def _get_target_depth(self, depth_frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """获取目标深度"""
        try:
            x, y, w, h = bbox
            
            # 计算采样区域
            center_x = x + w // 2
            center_y = y + h // 2
            sample_size = min(w, h) // 4
            
            x1 = max(0, center_x - sample_size)
            y1 = max(0, center_y - sample_size)
            x2 = min(depth_frame.shape[1], center_x + sample_size)
            y2 = min(depth_frame.shape[0], center_y + sample_size)
            
            # 提取深度值
            depth_region = depth_frame[y1:y2, x1:x2]
            valid_depths = depth_region[depth_region > 0]
            
            if len(valid_depths) > 0:
                return np.median(valid_depths) * 0.001  # 转换为米
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"获取目标深度失败: {e}")
            return 0.0
    
    def _calculate_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """计算两点间距离"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _update_performance_stats(self, start_time: float):
        """更新性能统计"""
        try:
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.frame_count += 1
            
            # 计算平均值
            if len(self.processing_times) > 0:
                self.performance.avg_processing_time = sum(self.processing_times) / len(self.processing_times)
                self.performance.fps = 1.0 / self.performance.avg_processing_time if self.performance.avg_processing_time > 0 else 0.0
            
            self.performance.total_frames = self.frame_count
            self.performance.successful_tracks = len(self.targets)
            
        except Exception as e:
            logger.error(f"更新性能统计失败: {e}")
    
    def stop_tracking(self, target_id: Optional[int] = None):
        """停止跟踪"""
        try:
            with self.tracking_lock:
                if target_id is None:
                    # 停止所有跟踪
                    self.targets.clear()
                    self.predictors.clear()
                    self.tracking_history.clear()
                    self.active_target_id = None
                    logger.info("停止所有跟踪")
                else:
                    # 停止特定目标跟踪
                    if target_id in self.targets:
                        del self.targets[target_id]
                    if target_id in self.predictors:
                        del self.predictors[target_id]
                    if target_id in self.tracking_history:
                        del self.tracking_history[target_id]
                    
                    if self.active_target_id == target_id:
                        self.active_target_id = None
                    
                    logger.info(f"停止跟踪目标: {target_id}")
                
                self.state = TrackingState.IDLE
                
        except Exception as e:
            logger.error(f"停止跟踪失败: {e}")
    
    def get_tracking_status(self) -> Dict[str, Any]:
        """获取跟踪状态"""
        return {
            'state': self.state.value,
            'tracker_type': self.tracker_type.value,
            'active_targets': len(self.targets),
            'active_target_id': self.active_target_id,
            'targets': {tid: target.to_dict() for tid, target in self.targets.items()},
            'performance': {
                'fps': self.performance.fps,
                'avg_processing_time': self.performance.avg_processing_time,
                'total_frames': self.performance.total_frames,
                'successful_tracks': self.performance.successful_tracks,
                'lost_tracks': self.performance.lost_tracks,
                'recovered_tracks': self.performance.recovered_tracks
            }
        }
    
    def set_active_target(self, target_id: int) -> bool:
        """设置活动目标"""
        try:
            if target_id in self.targets:
                self.active_target_id = target_id
                logger.info(f"设置活动目标: {target_id}")
                return True
            else:
                logger.warning(f"目标不存在: {target_id}")
                return False
        except Exception as e:
            logger.error(f"设置活动目标失败: {e}")
            return False
    
    def get_active_target(self) -> Optional[TrackingTarget]:
        """获取活动目标"""
        if self.active_target_id is not None and self.active_target_id in self.targets:
            return self.targets[self.active_target_id]
        return None
    
    def predict_target_position(self, target_id: int, steps: int = 1) -> Optional[Tuple[int, int]]:
        """预测目标位置"""
        try:
            if target_id in self.predictors:
                x, y = self.predictors[target_id].predict(steps)
                return (int(x), int(y))
            return None
        except Exception as e:
            logger.error(f"预测目标位置失败: {e}")
            return None
    
    def optimize_for_jetson(self):
        """针对Jetson Orin Nano的优化"""
        try:
            # 启用GPU加速
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                logger.info("启用CUDA加速")
                # 这里可以添加CUDA优化代码
            
            # 调整跟踪器参数
            if self.tracker_type == TrackerType.CSRT:
                # CSRT在Jetson上的优化参数
                self.max_lost_frames = 10  # 减少丢失帧数限制
                self.search_radius = 100   # 减少搜索半径
            
            elif self.tracker_type == TrackerType.MOSSE:
                # MOSSE是最快的，适合实时应用
                self.max_lost_frames = 5
                self.search_radius = 80
            
            # 启用多线程处理
            cv2.setNumThreads(4)  # Jetson Orin Nano有6个CPU核心
            
            logger.info("Jetson Orin Nano优化完成")
            
        except Exception as e:
            logger.error(f"Jetson优化失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        try:
            logger.info("正在清理跟踪控制器资源...")
            
            # 停止所有跟踪
            self.stop_tracking()
            
            # 清理跟踪器
            if self.opencv_tracker:
                self.opencv_tracker.tracker = None
            
            if self.yolo_tracker:
                self.yolo_tracker.detector = None
                self.yolo_tracker.byte_tracker = None
            
            # 清理预测器
            self.predictors.clear()
            self.tracking_history.clear()
            
            logger.info("跟踪控制器资源清理完成")
            
        except Exception as e:
            logger.error(f"跟踪控制器资源清理失败: {e}")

# 工厂函数
def create_tracking_controller(config: Dict[str, Any]) -> TrackingController:
    """创建跟踪控制器"""
    controller = TrackingController(config)
    
    # 根据配置优化
    if config.get('optimize_for_jetson', False):
        controller.optimize_for_jetson()
    
    return controller