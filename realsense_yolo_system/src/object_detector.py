"""
目标检测模块
使用YOLO模型进行实时目标检测和分类
"""

import cv2
import numpy as np
import torch
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("请安装ultralytics: pip install ultralytics")

from .utils import Timer, FPSCounter, timer_decorator

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """检测结果数据结构"""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    
    def __post_init__(self):
        """后初始化验证"""
        if len(self.bbox) != 4:
            raise ValueError("边界框必须包含4个坐标值")
        if not 0 <= self.confidence <= 1:
            raise ValueError("置信度必须在0-1之间")
    
    @property
    def center(self) -> Tuple[int, int]:
        """获取边界框中心点"""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    @property
    def area(self) -> float:
        """获取边界框面积"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    @property
    def width(self) -> float:
        """获取边界框宽度"""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """获取边界框高度"""
        return self.bbox[3] - self.bbox[1]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'center': self.center,
            'area': self.area,
            'width': self.width,
            'height': self.height,
        }

class ObjectDetector:
    """YOLO目标检测器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化目标检测器
        
        Args:
            config: 检测器配置
        """
        self.config = config
        self.model = None
        self.device = None
        self.class_names = []
        self.is_initialized = False
        
        # 性能监控
        self.fps_counter = FPSCounter()
        self.inference_times = []
        
        # 统计信息
        self.total_detections = 0
        self.detection_history = []
        
        logger.info("目标检测器初始化完成")
    
    def initialize(self) -> bool:
        """
        初始化YOLO模型
        
        Returns:
            是否初始化成功
        """
        try:
            # 检查模型文件是否存在
            model_path = Path(self.config['model_path'])
            if not model_path.exists():
                logger.info(f"模型文件不存在，将下载: {model_path}")
                # 如果模型文件不存在，ultralytics会自动下载
            
            # 加载模型
            self.model = YOLO(str(model_path))
            
            # 设置设备
            self.device = self.config.get('device', 'cpu')
            if self.device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA不可用，切换到CPU")
                self.device = 'cpu'
            
            self.model.to(self.device)
            
            # 获取类别名称
            self.class_names = self.model.names
            
            # 设置模型参数
            self.model.conf = self.config.get('confidence_threshold', 0.5)
            self.model.iou = self.config.get('iou_threshold', 0.45)
            self.model.max_det = self.config.get('max_detections', 100)
            
            # 预热模型
            self._warmup_model()
            
            self.is_initialized = True
            logger.info(f"YOLO模型初始化成功，设备: {self.device}")
            logger.info(f"模型类别数: {len(self.class_names)}")
            
            return True
            
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            return False
    
    def _warmup_model(self):
        """预热模型以获得更好的性能"""
        try:
            # 创建dummy输入
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            
            # 运行几次推理
            for _ in range(3):
                self.model(dummy_input, verbose=False)
            
            logger.info("模型预热完成")
            
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")
    
    @timer_decorator
    def detect(self, image: np.ndarray, 
               filter_classes: Optional[List[str]] = None) -> List[DetectionResult]:
        """
        检测图像中的目标
        
        Args:
            image: 输入图像
            filter_classes: 过滤的类别名称列表
            
        Returns:
            检测结果列表
        """
        if not self.is_initialized:
            logger.error("模型未初始化")
            return []
        
        try:
            # 记录推理开始时间
            start_time = time.time()
            
            # 运行推理
            results = self.model(image, verbose=False)
            
            # 记录推理时间
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            # 解析结果
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        # 获取类别名称
                        class_name = self.class_names.get(cls_id, f"class_{cls_id}")
                        
                        # 过滤类别
                        if filter_classes and class_name not in filter_classes:
                            continue
                        
                        # 创建检测结果
                        detection = DetectionResult(
                            bbox=box.tolist(),
                            confidence=float(conf),
                            class_id=int(cls_id),
                            class_name=class_name
                        )
                        
                        detections.append(detection)
            
            # 更新统计信息
            self.total_detections += len(detections)
            self.detection_history.append(len(detections))
            if len(self.detection_history) > 100:
                self.detection_history.pop(0)
            
            # 更新FPS
            self.fps_counter.update()
            
            return detections
            
        except Exception as e:
            logger.error(f"目标检测失败: {e}")
            return []
    
    def detect_batch(self, images: List[np.ndarray], 
                    filter_classes: Optional[List[str]] = None) -> List[List[DetectionResult]]:
        """
        批量检测图像中的目标
        
        Args:
            images: 输入图像列表
            filter_classes: 过滤的类别名称列表
            
        Returns:
            检测结果列表的列表
        """
        if not self.is_initialized:
            logger.error("模型未初始化")
            return []
        
        try:
            # 运行批量推理
            results = self.model(images, verbose=False)
            
            # 解析结果
            all_detections = []
            for result in results:
                detections = []
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        # 获取类别名称
                        class_name = self.class_names.get(cls_id, f"class_{cls_id}")
                        
                        # 过滤类别
                        if filter_classes and class_name not in filter_classes:
                            continue
                        
                        # 创建检测结果
                        detection = DetectionResult(
                            bbox=box.tolist(),
                            confidence=float(conf),
                            class_id=int(cls_id),
                            class_name=class_name
                        )
                        
                        detections.append(detection)
                
                all_detections.append(detections)
            
            return all_detections
            
        except Exception as e:
            logger.error(f"批量目标检测失败: {e}")
            return []
    
    def filter_detections(self, detections: List[DetectionResult], 
                         min_confidence: float = 0.5,
                         min_area: float = 100,
                         max_area: Optional[float] = None,
                         allowed_classes: Optional[List[str]] = None) -> List[DetectionResult]:
        """
        过滤检测结果
        
        Args:
            detections: 原始检测结果
            min_confidence: 最小置信度
            min_area: 最小面积
            max_area: 最大面积
            allowed_classes: 允许的类别列表
            
        Returns:
            过滤后的检测结果
        """
        filtered = []
        
        for detection in detections:
            # 置信度过滤
            if detection.confidence < min_confidence:
                continue
            
            # 面积过滤
            area = detection.area
            if area < min_area:
                continue
            if max_area and area > max_area:
                continue
            
            # 类别过滤
            if allowed_classes and detection.class_name not in allowed_classes:
                continue
            
            filtered.append(detection)
        
        return filtered
    
    def non_max_suppression(self, detections: List[DetectionResult], 
                           iou_threshold: float = 0.45) -> List[DetectionResult]:
        """
        非极大值抑制
        
        Args:
            detections: 检测结果列表
            iou_threshold: IoU阈值
            
        Returns:
            NMS后的检测结果
        """
        if not detections:
            return []
        
        # 按置信度排序
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        # 计算IoU
        def calculate_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        # 执行NMS
        keep = []
        for i, detection in enumerate(detections):
            should_keep = True
            
            for j in keep:
                if calculate_iou(detection.bbox, detections[j].bbox) > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(i)
        
        return [detections[i] for i in keep]
    
    def track_detections(self, detections: List[DetectionResult], 
                        previous_detections: List[DetectionResult],
                        max_distance: float = 100.0) -> List[DetectionResult]:
        """
        简单的检测追踪（基于距离）
        
        Args:
            detections: 当前检测结果
            previous_detections: 前一帧检测结果
            max_distance: 最大匹配距离
            
        Returns:
            匹配后的检测结果
        """
        if not previous_detections:
            return detections
        
        # 简单的距离匹配
        matched_detections = []
        
        for detection in detections:
            current_center = detection.center
            min_distance = float('inf')
            
            for prev_detection in previous_detections:
                prev_center = prev_detection.center
                distance = np.sqrt((current_center[0] - prev_center[0])**2 + 
                                 (current_center[1] - prev_center[1])**2)
                
                if distance < min_distance and distance < max_distance:
                    min_distance = distance
            
            # 如果找到匹配，保留检测结果
            if min_distance < max_distance:
                matched_detections.append(detection)
        
        return matched_detections
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """
        获取检测统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_detections': self.total_detections,
            'fps': self.fps_counter.get_fps(),
            'avg_detections_per_frame': np.mean(self.detection_history) if self.detection_history else 0,
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'model_info': {
                'model_path': str(self.config['model_path']),
                'device': self.device,
                'num_classes': len(self.class_names),
                'confidence_threshold': self.config.get('confidence_threshold', 0.5),
                'iou_threshold': self.config.get('iou_threshold', 0.45),
            }
        }
        
        return stats
    
    def get_class_names(self) -> List[str]:
        """获取所有类别名称"""
        return list(self.class_names.values())
    
    def get_class_id(self, class_name: str) -> Optional[int]:
        """
        根据类别名称获取ID
        
        Args:
            class_name: 类别名称
            
        Returns:
            类别ID或None
        """
        for cls_id, name in self.class_names.items():
            if name == class_name:
                return cls_id
        return None
    
    def save_model_info(self, filepath: str):
        """
        保存模型信息
        
        Args:
            filepath: 文件路径
        """
        import json
        
        model_info = {
            'model_path': str(self.config['model_path']),
            'device': self.device,
            'class_names': self.class_names,
            'config': self.config,
            'stats': self.get_detection_stats(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"模型信息已保存: {filepath}")
    
    def cleanup(self):
        """清理资源"""
        if self.model:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.model = None
        
        self.is_initialized = False
        logger.info("目标检测器资源清理完成")

# 工厂函数
def create_object_detector(config: Dict[str, Any]) -> ObjectDetector:
    """
    创建目标检测器
    
    Args:
        config: 检测器配置
        
    Returns:
        目标检测器实例
    """
    return ObjectDetector(config)

# 模型下载函数
def download_yolo_model(model_name: str, save_path: str = "models") -> str:
    """
    下载YOLO模型
    
    Args:
        model_name: 模型名称 (如 'yolov8n', 'yolov8s', 等)
        save_path: 保存路径
        
    Returns:
        模型文件路径
    """
    from pathlib import Path
    
    save_dir = Path(save_path)
    save_dir.mkdir(exist_ok=True)
    
    model_path = save_dir / f"{model_name}.pt"
    
    try:
        # 使用ultralytics自动下载
        model = YOLO(model_name)
        
        # 保存模型
        model.save(str(model_path))
        
        logger.info(f"模型已下载: {model_path}")
        return str(model_path)
        
    except Exception as e:
        logger.error(f"下载模型失败: {e}")
        raise

# 支持的YOLO模型
SUPPORTED_MODELS = {
    'yolov8n': 'YOLOv8 Nano',
    'yolov8s': 'YOLOv8 Small',
    'yolov8m': 'YOLOv8 Medium',
    'yolov8l': 'YOLOv8 Large',
    'yolov8x': 'YOLOv8 Extra Large',
    'yolov5n': 'YOLOv5 Nano',
    'yolov5s': 'YOLOv5 Small',
    'yolov5m': 'YOLOv5 Medium',
    'yolov5l': 'YOLOv5 Large',
    'yolov5x': 'YOLOv5 Extra Large',
}

def get_supported_models() -> Dict[str, str]:
    """获取支持的模型列表"""
    return SUPPORTED_MODELS

if __name__ == "__main__":
    # 测试目标检测器
    import sys
    sys.path.append('..')
    from config.config import YOLO_CONFIG
    
    print("测试目标检测器...")
    
    # 创建检测器
    detector = create_object_detector(YOLO_CONFIG)
    
    # 初始化
    if detector.initialize():
        print("检测器初始化成功")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # 运行检测
        detections = detector.detect(test_image)
        print(f"检测到 {len(detections)} 个目标")
        
        # 打印统计信息
        stats = detector.get_detection_stats()
        print(f"统计信息: {stats}")
        
        # 清理资源
        detector.cleanup()
    else:
        print("检测器初始化失败")
    
    print("测试完成")