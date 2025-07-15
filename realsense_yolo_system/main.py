"""
主程序
整合所有模块，提供完整的RealSense D435i + YOLO系统功能
"""

import cv2
import numpy as np
import argparse
import logging
import time
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import threading
from queue import Queue, Empty

# 导入自定义模块
from src.camera_manager import CameraManager, discover_realsense_devices
from src.object_detector import ObjectDetector
from src.data_fusion import DataFusion
from src.visualizer import Visualizer
from src.utils import Timer, FPSCounter
from config.config import (
    CAMERA_CONFIG, YOLO_CONFIG, DEPTH_CONFIG, 
    FUSION_CONFIG, VISUALIZATION_CONFIG, SYSTEM_CONFIG
)

# 设置日志
logging.basicConfig(
    level=getattr(logging, SYSTEM_CONFIG['log_level']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealSenseYOLOSystem:
    """RealSense D435i + YOLO 系统主类"""
    
    def __init__(self, config_overrides: Optional[Dict[str, Any]] = None):
        """
        初始化系统
        
        Args:
            config_overrides: 配置覆盖
        """
        self.config = self._merge_configs(config_overrides or {})
        
        # 核心组件
        self.camera_manager = None
        self.object_detector = None
        self.data_fusion = None
        self.visualizer = None
        
        # 系统状态
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0
        
        # 性能监控
        self.fps_counter = FPSCounter()
        self.processing_times = []
        
        # 线程管理
        self.processing_thread = None
        self.result_queue = Queue(maxsize=10)
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("RealSense YOLO 系统初始化完成")
    
    def _merge_configs(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置"""
        config = {
            'camera': CAMERA_CONFIG.copy(),
            'yolo': YOLO_CONFIG.copy(),
            'depth': DEPTH_CONFIG.copy(),
            'fusion': FUSION_CONFIG.copy(),
            'visualization': VISUALIZATION_CONFIG.copy(),
            'system': SYSTEM_CONFIG.copy(),
        }
        
        # 应用覆盖
        for key, value in overrides.items():
            if key in config:
                config[key].update(value)
        
        return config
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"接收到信号 {signum}，正在关闭系统...")
        self.shutdown()
        sys.exit(0)
    
    def initialize(self) -> bool:
        """
        初始化所有组件
        
        Returns:
            是否初始化成功
        """
        try:
            # 检查设备
            devices = discover_realsense_devices()
            if not devices:
                logger.error("未找到RealSense设备")
                return False
            
            logger.info(f"发现设备: {devices[0]['name']}")
            
            # 初始化相机管理器
            self.camera_manager = CameraManager(self.config['camera'])
            if not self.camera_manager.initialize():
                logger.error("相机初始化失败")
                return False
            
            # 初始化目标检测器
            self.object_detector = ObjectDetector(self.config['yolo'])
            if not self.object_detector.initialize():
                logger.error("目标检测器初始化失败")
                return False
            
            # 初始化数据融合器
            self.data_fusion = DataFusion(self.config['fusion'])
            
            # 初始化可视化器
            self.visualizer = Visualizer(self.config['visualization'])
            
            logger.info("所有组件初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            return False
    
    def start(self):
        """启动系统"""
        if not self.initialize():
            return False
        
        # 启动相机流
        if not self.camera_manager.start_streaming(threaded=True):
            logger.error("启动相机流失败")
            return False
        
        self.is_running = True
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # 启动主循环
        self._main_loop()
        
        return True
    
    def _processing_loop(self):
        """处理循环（在单独线程中运行）"""
        while self.is_running:
            try:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
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
                
            except Exception as e:
                logger.error(f"处理循环错误: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame_data) -> Dict[str, Any]:
        """
        处理单帧数据
        
        Args:
            frame_data: 帧数据
            
        Returns:
            处理结果
        """
        with Timer("帧处理") as timer:
            # 目标检测
            detections = self.object_detector.detect(frame_data.color_image)
            
            # 数据融合
            fused_results = self.data_fusion.fuse_detections_with_depth(
                detections, 
                frame_data.depth_image,
                self.camera_manager.intrinsics.to_dict()
            )
            
            # 过滤结果
            filtered_results = self.data_fusion.filter_fused_results(fused_results)
            
            # 更新性能统计
            self.processing_times.append(timer.elapsed())
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            return {
                'frame_data': frame_data,
                'detections': detections,
                'fused_results': filtered_results,
                'processing_time': timer.elapsed(),
                'frame_number': self.frame_count
            }
    
    def _main_loop(self):
        """主显示循环"""
        logger.info("系统启动成功，按 'q' 退出")
        self.visualizer.show_interactive_controls()
        
        while self.is_running:
            try:
                # 从队列获取处理结果
                try:
                    result = self.result_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                if self.is_paused:
                    continue
                
                # 创建可视化
                combined_view = self.visualizer.create_combined_view(
                    result['frame_data'].color_image,
                    result['frame_data'].depth_image,
                    result['fused_results']
                )
                
                # 添加统计信息
                stats = self._get_system_stats(result)
                combined_view = self.visualizer.create_statistics_overlay(
                    combined_view, stats)
                
                # 显示结果
                cv2.imshow('RealSense YOLO System', combined_view)
                
                # 保存结果（如果需要）
                if self.config['visualization']['save_results']:
                    self.visualizer.save_visualization(
                        combined_view, result['fused_results'])
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                action = self.visualizer.handle_key_input(key)
                
                if action == 'quit':
                    break
                elif action == 'save':
                    self._save_current_frame(result)
                elif action == 'reset':
                    self.data_fusion.reset_tracking()
                elif action == 'pause':
                    self.is_paused = not self.is_paused
                    logger.info(f"系统{'暂停' if self.is_paused else '继续'}")
                
                # 更新帧计数
                self.frame_count += 1
                
                # 限制帧率
                if self.config['system']['max_fps'] > 0:
                    time.sleep(1.0 / self.config['system']['max_fps'])
                
            except Exception as e:
                logger.error(f"主循环错误: {e}")
                time.sleep(0.1)
        
        self.shutdown()
    
    def _get_system_stats(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            'total_objects': len(result['fused_results']),
            'active_tracks': len(self.data_fusion.get_all_tracking_info()),
            'avg_distance': np.mean([r.distance_from_camera for r in result['fused_results']]) if result['fused_results'] else 0,
            'fusion_time': result['processing_time'],
            'fps': self.camera_manager.get_fps(),
            'frame_count': self.frame_count,
        }
    
    def _save_current_frame(self, result: Dict[str, Any]):
        """保存当前帧"""
        timestamp = int(time.time())
        output_dir = Path(f"saved_frames_{timestamp}")
        output_dir.mkdir(exist_ok=True)
        
        # 保存原始图像
        cv2.imwrite(str(output_dir / "color.jpg"), result['frame_data'].color_image)
        
        # 保存深度图
        depth_colored = self.visualizer.create_depth_visualization(result['frame_data'].depth_image)
        cv2.imwrite(str(output_dir / "depth.jpg"), depth_colored)
        
        # 保存检测结果
        detection_image = self.visualizer.draw_detections(
            result['frame_data'].color_image, 
            [r.detection for r in result['fused_results']]
        )
        cv2.imwrite(str(output_dir / "detections.jpg"), detection_image)
        
        # 保存数据
        import json
        data = {
            'timestamp': timestamp,
            'frame_number': result['frame_number'],
            'camera_info': self.camera_manager.get_camera_info(),
            'detection_stats': self.object_detector.get_detection_stats(),
            'fusion_stats': self.data_fusion.get_fusion_statistics(),
            'results': [r.to_dict() for r in result['fused_results']]
        }
        
        with open(output_dir / "data.json", 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"当前帧已保存到: {output_dir}")
    
    def run_benchmark(self, duration: int = 60):
        """
        运行性能基准测试
        
        Args:
            duration: 测试持续时间（秒）
        """
        logger.info(f"开始性能基准测试，持续时间: {duration}秒")
        
        if not self.initialize():
            return False
        
        self.camera_manager.start_streaming(threaded=False)
        
        start_time = time.time()
        frame_times = []
        detection_times = []
        fusion_times = []
        
        while time.time() - start_time < duration:
            frame_start = time.time()
            
            # 获取帧
            frame_data = self.camera_manager.get_frame()
            if not frame_data:
                continue
            
            # 检测
            detect_start = time.time()
            detections = self.object_detector.detect(frame_data.color_image)
            detect_time = time.time() - detect_start
            
            # 融合
            fusion_start = time.time()
            fused_results = self.data_fusion.fuse_detections_with_depth(
                detections, frame_data.depth_image,
                self.camera_manager.intrinsics.to_dict()
            )
            fusion_time = time.time() - fusion_start
            
            frame_time = time.time() - frame_start
            
            frame_times.append(frame_time)
            detection_times.append(detect_time)
            fusion_times.append(fusion_time)
            
            if len(frame_times) % 30 == 0:
                print(f"已处理 {len(frame_times)} 帧...")
        
        # 输出基准测试结果
        self._print_benchmark_results(frame_times, detection_times, fusion_times)
        
        self.shutdown()
        return True
    
    def _print_benchmark_results(self, frame_times, detection_times, fusion_times):
        """打印基准测试结果"""
        print("\\n=== 性能基准测试结果 ===")
        print(f"总帧数: {len(frame_times)}")
        print(f"平均FPS: {1.0 / np.mean(frame_times):.2f}")
        print(f"最大FPS: {1.0 / np.min(frame_times):.2f}")
        print(f"最小FPS: {1.0 / np.max(frame_times):.2f}")
        print(f"平均帧处理时间: {np.mean(frame_times) * 1000:.2f}ms")
        print(f"平均检测时间: {np.mean(detection_times) * 1000:.2f}ms")
        print(f"平均融合时间: {np.mean(fusion_times) * 1000:.2f}ms")
        print(f"检测时间占比: {np.mean(detection_times) / np.mean(frame_times) * 100:.1f}%")
        print(f"融合时间占比: {np.mean(fusion_times) / np.mean(frame_times) * 100:.1f}%")
    
    def shutdown(self):
        """关闭系统"""
        logger.info("正在关闭系统...")
        
        self.is_running = False
        
        # 等待处理线程结束
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # 清理组件
        if self.camera_manager:
            self.camera_manager.cleanup()
        
        if self.object_detector:
            self.object_detector.cleanup()
        
        if self.data_fusion:
            self.data_fusion.cleanup()
        
        if self.visualizer:
            self.visualizer.cleanup()
        
        logger.info("系统关闭完成")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RealSense D435i + YOLO 系统')
    
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--model', type=str, help='YOLO模型路径')
    parser.add_argument('--device', type=str, default='auto', 
                       help='设备类型 (cpu/cuda/auto)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='检测置信度阈值')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示可视化界面')
    parser.add_argument('--save-results', action='store_true',
                       help='保存检测结果')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='输出目录')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行性能基准测试')
    parser.add_argument('--benchmark-duration', type=int, default=60,
                       help='基准测试持续时间（秒）')
    parser.add_argument('--list-devices', action='store_true',
                       help='列出可用设备')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 列出设备
    if args.list_devices:
        print("可用的RealSense设备:")
        devices = discover_realsense_devices()
        for i, device in enumerate(devices):
            print(f"  {i}: {device['name']} (SN: {device['serial_number']})")
        return
    
    # 准备配置覆盖
    config_overrides = {}
    
    if args.model:
        config_overrides['yolo'] = {'model_path': args.model}
    
    if args.device != 'auto':
        config_overrides['yolo'] = config_overrides.get('yolo', {})
        config_overrides['yolo']['device'] = args.device
    
    if args.confidence != 0.5:
        config_overrides['yolo'] = config_overrides.get('yolo', {})
        config_overrides['yolo']['confidence_threshold'] = args.confidence
    
    if args.save_results:
        config_overrides['visualization'] = {'save_results': True, 'output_dir': args.output_dir}
    
    # 创建系统
    system = RealSenseYOLOSystem(config_overrides)
    
    # 运行系统
    try:
        if args.benchmark:
            system.run_benchmark(args.benchmark_duration)
        else:
            system.start()
    except KeyboardInterrupt:
        logger.info("接收到中断信号")
    except Exception as e:
        logger.error(f"系统运行错误: {e}")
    finally:
        system.shutdown()

if __name__ == "__main__":
    main()