"""
配置模块测试
"""

import unittest
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (
    CAMERA_CONFIG, YOLO_CONFIG, DEPTH_CONFIG, 
    FUSION_CONFIG, VISUALIZATION_CONFIG, SYSTEM_CONFIG,
    validate_config, create_directories, get_config_summary
)

class TestConfiguration(unittest.TestCase):
    """配置测试类"""
    
    def test_camera_config_validity(self):
        """测试相机配置有效性"""
        # 检查必需的键
        required_keys = ['width', 'height', 'fps', 'enable_depth', 'enable_color']
        for key in required_keys:
            self.assertIn(key, CAMERA_CONFIG, f"缺少必需的相机配置键: {key}")
        
        # 检查数值范围
        self.assertGreater(CAMERA_CONFIG['width'], 0, "相机宽度必须大于0")
        self.assertGreater(CAMERA_CONFIG['height'], 0, "相机高度必须大于0")
        self.assertGreater(CAMERA_CONFIG['fps'], 0, "相机帧率必须大于0")
        
        # 检查布尔值
        self.assertIsInstance(CAMERA_CONFIG['enable_depth'], bool)
        self.assertIsInstance(CAMERA_CONFIG['enable_color'], bool)
    
    def test_yolo_config_validity(self):
        """测试YOLO配置有效性"""
        # 检查必需的键
        required_keys = ['model_path', 'confidence_threshold', 'iou_threshold']
        for key in required_keys:
            self.assertIn(key, YOLO_CONFIG, f"缺少必需的YOLO配置键: {key}")
        
        # 检查阈值范围
        self.assertGreaterEqual(YOLO_CONFIG['confidence_threshold'], 0)
        self.assertLessEqual(YOLO_CONFIG['confidence_threshold'], 1)
        self.assertGreaterEqual(YOLO_CONFIG['iou_threshold'], 0)
        self.assertLessEqual(YOLO_CONFIG['iou_threshold'], 1)
    
    def test_depth_config_validity(self):
        """测试深度配置有效性"""
        # 检查必需的键
        required_keys = ['min_depth', 'max_depth', 'depth_scale']
        for key in required_keys:
            self.assertIn(key, DEPTH_CONFIG, f"缺少必需的深度配置键: {key}")
        
        # 检查深度范围
        self.assertGreater(DEPTH_CONFIG['min_depth'], 0)
        self.assertGreater(DEPTH_CONFIG['max_depth'], DEPTH_CONFIG['min_depth'])
        self.assertGreater(DEPTH_CONFIG['depth_scale'], 0)
    
    def test_fusion_config_validity(self):
        """测试融合配置有效性"""
        # 检查必需的键
        required_keys = ['depth_roi_expansion', 'min_valid_pixels']
        for key in required_keys:
            self.assertIn(key, FUSION_CONFIG, f"缺少必需的融合配置键: {key}")
        
        # 检查数值范围
        self.assertGreaterEqual(FUSION_CONFIG['depth_roi_expansion'], 0)
        self.assertGreater(FUSION_CONFIG['min_valid_pixels'], 0)
    
    def test_visualization_config_validity(self):
        """测试可视化配置有效性"""
        # 检查必需的键
        required_keys = ['show_rgb', 'show_depth', 'font_scale']
        for key in required_keys:
            self.assertIn(key, VISUALIZATION_CONFIG, f"缺少必需的可视化配置键: {key}")
        
        # 检查布尔值
        self.assertIsInstance(VISUALIZATION_CONFIG['show_rgb'], bool)
        self.assertIsInstance(VISUALIZATION_CONFIG['show_depth'], bool)
        
        # 检查字体缩放
        self.assertGreater(VISUALIZATION_CONFIG['font_scale'], 0)
    
    def test_system_config_validity(self):
        """测试系统配置有效性"""
        # 检查必需的键
        required_keys = ['max_fps', 'log_level']
        for key in required_keys:
            self.assertIn(key, SYSTEM_CONFIG, f"缺少必需的系统配置键: {key}")
        
        # 检查FPS
        self.assertGreaterEqual(SYSTEM_CONFIG['max_fps'], 0)
        
        # 检查日志级别
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        self.assertIn(SYSTEM_CONFIG['log_level'], valid_log_levels)
    
    def test_validate_config_function(self):
        """测试配置验证函数"""
        # 应该不抛出异常
        self.assertTrue(validate_config())
    
    def test_create_directories_function(self):
        """测试目录创建函数"""
        # 应该不抛出异常
        create_directories()
    
    def test_get_config_summary_function(self):
        """测试配置摘要函数"""
        summary = get_config_summary()
        
        # 检查返回类型
        self.assertIsInstance(summary, dict)
        
        # 检查必需的键
        required_keys = ['camera', 'yolo_model', 'device']
        for key in required_keys:
            self.assertIn(key, summary, f"配置摘要缺少键: {key}")
    
    def test_config_consistency(self):
        """测试配置一致性"""
        # 检查相机和深度配置的一致性
        # 例如，确保深度范围与相机规格匹配
        self.assertGreater(DEPTH_CONFIG['max_depth'], DEPTH_CONFIG['min_depth'])
        
        # 检查融合和可视化配置的一致性
        if FUSION_CONFIG.get('enable_3d_visualization'):
            self.assertTrue(VISUALIZATION_CONFIG.get('show_3d_info', True))

class TestConfigurationIntegration(unittest.TestCase):
    """配置集成测试"""
    
    def test_all_configs_loadable(self):
        """测试所有配置可加载"""
        configs = [
            CAMERA_CONFIG,
            YOLO_CONFIG,
            DEPTH_CONFIG,
            FUSION_CONFIG,
            VISUALIZATION_CONFIG,
            SYSTEM_CONFIG,
        ]
        
        for config in configs:
            self.assertIsInstance(config, dict)
            self.assertGreater(len(config), 0)
    
    def test_config_types(self):
        """测试配置类型"""
        # 相机配置类型检查
        self.assertIsInstance(CAMERA_CONFIG['width'], int)
        self.assertIsInstance(CAMERA_CONFIG['height'], int)
        self.assertIsInstance(CAMERA_CONFIG['fps'], int)
        
        # YOLO配置类型检查
        self.assertIsInstance(YOLO_CONFIG['confidence_threshold'], (int, float))
        self.assertIsInstance(YOLO_CONFIG['iou_threshold'], (int, float))
        
        # 深度配置类型检查
        self.assertIsInstance(DEPTH_CONFIG['min_depth'], (int, float))
        self.assertIsInstance(DEPTH_CONFIG['max_depth'], (int, float))

if __name__ == '__main__':
    unittest.main()