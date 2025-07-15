"""
工具函数测试
"""

import unittest
import numpy as np
import time
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    Timer, FPSCounter, CircularBuffer, 
    resize_image, calculate_distance_3d, pixel_to_point, point_to_pixel,
    filter_outliers, moving_average, SafeQueue,
    validate_camera_config, validate_yolo_config
)

class TestTimer(unittest.TestCase):
    """计时器测试"""
    
    def test_timer_basic_functionality(self):
        """测试计时器基本功能"""
        timer = Timer("测试计时器")
        
        # 测试开始和停止
        timer.start()
        time.sleep(0.1)  # 休眠100ms
        elapsed = timer.stop()
        
        # 检查经过时间
        self.assertGreaterEqual(elapsed, 0.1)
        self.assertLess(elapsed, 0.2)  # 不应该超过200ms
    
    def test_timer_context_manager(self):
        """测试计时器上下文管理器"""
        with Timer("上下文测试").time_it() as timer:
            time.sleep(0.05)  # 休眠50ms
        
        elapsed = timer.elapsed()
        self.assertGreaterEqual(elapsed, 0.05)
        self.assertLess(elapsed, 0.1)

class TestFPSCounter(unittest.TestCase):
    """FPS计数器测试"""
    
    def test_fps_counter_basic(self):
        """测试FPS计数器基本功能"""
        fps_counter = FPSCounter(window_size=5)
        
        # 模拟帧更新
        for _ in range(10):
            fps = fps_counter.update()
            time.sleep(0.01)  # 10ms间隔
        
        # FPS应该在合理范围内
        self.assertGreater(fps, 0)
        self.assertLess(fps, 1000)  # 不应该超过1000FPS
    
    def test_fps_counter_get_fps(self):
        """测试获取FPS功能"""
        fps_counter = FPSCounter()
        
        # 初始FPS应该为0
        self.assertEqual(fps_counter.get_fps(), 0)
        
        # 更新后应该有值
        fps_counter.update()
        time.sleep(0.01)
        fps_counter.update()
        
        fps = fps_counter.get_fps()
        self.assertGreater(fps, 0)

class TestCircularBuffer(unittest.TestCase):
    """循环缓冲区测试"""
    
    def test_circular_buffer_basic(self):
        """测试循环缓冲区基本功能"""
        buffer = CircularBuffer(3)
        
        # 测试空状态
        self.assertTrue(buffer.is_empty())
        self.assertFalse(buffer.is_full())
        self.assertIsNone(buffer.get_latest())
        
        # 添加元素
        buffer.put(1)
        buffer.put(2)
        buffer.put(3)
        
        # 测试满状态
        self.assertTrue(buffer.is_full())
        self.assertFalse(buffer.is_empty())
        self.assertEqual(buffer.get_latest(), 3)
        
        # 获取所有元素
        all_items = buffer.get_all()
        self.assertEqual(all_items, [1, 2, 3])
    
    def test_circular_buffer_overflow(self):
        """测试循环缓冲区溢出"""
        buffer = CircularBuffer(2)
        
        # 添加超过容量的元素
        buffer.put(1)
        buffer.put(2)
        buffer.put(3)  # 应该覆盖第一个元素
        
        all_items = buffer.get_all()
        self.assertEqual(all_items, [2, 3])
        self.assertEqual(buffer.get_latest(), 3)

class TestImageProcessing(unittest.TestCase):
    """图像处理测试"""
    
    def test_resize_image(self):
        """测试图像调整大小"""
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 测试不保持宽高比
        resized = resize_image(test_image, (320, 240), keep_aspect_ratio=False)
        self.assertEqual(resized.shape[:2], (240, 320))
        
        # 测试保持宽高比
        resized_aspect = resize_image(test_image, (320, 240), keep_aspect_ratio=True)
        self.assertEqual(resized_aspect.shape[:2], (240, 320))

class TestMathFunctions(unittest.TestCase):
    """数学函数测试"""
    
    def test_calculate_distance_3d(self):
        """测试3D距离计算"""
        point1 = np.array([0, 0, 0])
        point2 = np.array([3, 4, 0])
        
        distance = calculate_distance_3d(point1, point2)
        self.assertAlmostEqual(distance, 5.0, places=5)  # 3-4-5三角形
    
    def test_pixel_to_point(self):
        """测试像素到3D点转换"""
        intrinsics = {
            'fx': 600, 'fy': 600,
            'cx': 320, 'cy': 240
        }
        
        # 测试中心点
        point_3d = pixel_to_point((320, 240), 1.0, intrinsics)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(point_3d, expected)
    
    def test_point_to_pixel(self):
        """测试3D点到像素转换"""
        intrinsics = {
            'fx': 600, 'fy': 600,
            'cx': 320, 'cy': 240
        }
        
        # 测试中心点
        pixel_coords = point_to_pixel(np.array([0.0, 0.0, 1.0]), intrinsics)
        self.assertEqual(pixel_coords, (320, 240))
    
    def test_filter_outliers(self):
        """测试异常值过滤"""
        # 创建包含异常值的数据
        data = np.array([1, 2, 3, 4, 5, 100])  # 100是异常值
        
        filtered = filter_outliers(data, std_multiplier=2.0)
        
        # 异常值应该被过滤掉
        self.assertNotIn(100, filtered)
        self.assertIn(1, filtered)
    
    def test_moving_average(self):
        """测试移动平均"""
        data = [1, 2, 3, 4, 5]
        result = moving_average(data, window_size=3)
        
        expected = [2.0, 3.0, 4.0]  # (1+2+3)/3, (2+3+4)/3, (3+4+5)/3
        self.assertEqual(result, expected)

class TestSafeQueue(unittest.TestCase):
    """安全队列测试"""
    
    def test_safe_queue_basic(self):
        """测试安全队列基本功能"""
        queue = SafeQueue(maxsize=2)
        
        # 测试空状态
        self.assertTrue(queue.empty())
        self.assertFalse(queue.full())
        self.assertEqual(queue.qsize(), 0)
        
        # 添加元素
        queue.put("item1")
        queue.put("item2")
        
        # 测试满状态
        self.assertTrue(queue.full())
        self.assertFalse(queue.empty())
        self.assertEqual(queue.qsize(), 2)
        
        # 获取元素
        item = queue.get()
        self.assertEqual(item, "item1")
        self.assertEqual(queue.qsize(), 1)

class TestValidationFunctions(unittest.TestCase):
    """验证函数测试"""
    
    def test_validate_camera_config(self):
        """测试相机配置验证"""
        # 有效配置
        valid_config = {
            'width': 640,
            'height': 480,
            'fps': 30
        }
        self.assertTrue(validate_camera_config(valid_config))
        
        # 无效配置 - 缺少必需键
        invalid_config = {
            'width': 640,
            'height': 480
            # 缺少fps
        }
        self.assertFalse(validate_camera_config(invalid_config))
        
        # 无效配置 - 无效值
        invalid_config2 = {
            'width': 0,  # 无效宽度
            'height': 480,
            'fps': 30
        }
        self.assertFalse(validate_camera_config(invalid_config2))
    
    def test_validate_yolo_config(self):
        """测试YOLO配置验证"""
        # 有效配置
        valid_config = {
            'model_path': 'yolov8n.pt',
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45
        }
        self.assertTrue(validate_yolo_config(valid_config))
        
        # 无效配置 - 置信度超出范围
        invalid_config = {
            'model_path': 'yolov8n.pt',
            'confidence_threshold': 1.5,  # 超出范围
            'iou_threshold': 0.45
        }
        self.assertFalse(validate_yolo_config(invalid_config))

class TestUtilsIntegration(unittest.TestCase):
    """工具函数集成测试"""
    
    def test_timer_with_fps_counter(self):
        """测试计时器与FPS计数器集成"""
        timer = Timer("集成测试")
        fps_counter = FPSCounter()
        
        # 模拟处理循环
        for _ in range(5):
            timer.start()
            time.sleep(0.02)  # 模拟处理时间
            timer.stop()
            fps_counter.update()
        
        # 验证结果
        self.assertGreater(timer.elapsed(), 0)
        self.assertGreater(fps_counter.get_fps(), 0)
    
    def test_circular_buffer_with_moving_average(self):
        """测试循环缓冲区与移动平均集成"""
        buffer = CircularBuffer(5)
        
        # 添加数据
        values = [1, 2, 3, 4, 5]
        for value in values:
            buffer.put(value)
        
        # 获取数据并计算移动平均
        data = buffer.get_all()
        avg = moving_average(data, 3)
        
        # 验证结果
        self.assertEqual(len(avg), 3)
        self.assertEqual(avg, [2.0, 3.0, 4.0])

if __name__ == '__main__':
    unittest.main()