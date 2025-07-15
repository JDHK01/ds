"""
基础使用示例
演示如何使用RealSense D435i + YOLO系统进行3D目标检测
"""

import cv2
import numpy as np
import time
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.camera_manager import CameraManager, discover_realsense_devices
from src.object_detector import ObjectDetector
from src.data_fusion import DataFusion
from src.visualizer import Visualizer
from config.config import CAMERA_CONFIG, YOLO_CONFIG, FUSION_CONFIG, VISUALIZATION_CONFIG

def basic_detection_example():
    """基础检测示例"""
    print("=== 基础检测示例 ===")
    
    # 1. 检查设备
    print("1. 检查RealSense设备...")
    devices = discover_realsense_devices()
    
    if not devices:
        print("❌ 未找到RealSense设备")
        return False
    
    print(f"✓ 发现设备: {devices[0]['name']}")
    
    # 2. 初始化组件
    print("2. 初始化组件...")
    
    # 相机管理器
    camera = CameraManager(CAMERA_CONFIG)
    if not camera.initialize():
        print("❌ 相机初始化失败")
        return False
    
    # 目标检测器
    detector = ObjectDetector(YOLO_CONFIG)
    if not detector.initialize():
        print("❌ 目标检测器初始化失败")
        camera.cleanup()
        return False
    
    # 数据融合器
    fusion = DataFusion(FUSION_CONFIG)
    
    # 可视化器
    visualizer = Visualizer(VISUALIZATION_CONFIG)
    
    print("✓ 所有组件初始化成功")
    
    # 3. 开始检测
    print("3. 开始检测（按'q'退出）...")
    
    camera.start_streaming(threaded=False)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # 获取帧数据
            frame_data = camera.get_frame()
            if not frame_data or not frame_data.is_valid():
                continue
            
            # 目标检测
            detections = detector.detect(frame_data.color_image)
            
            # 数据融合
            fused_results = fusion.fuse_detections_with_depth(
                detections, 
                frame_data.depth_image,
                camera.intrinsics.to_dict()
            )
            
            # 可视化
            combined_view = visualizer.create_combined_view(
                frame_data.color_image,
                frame_data.depth_image,
                fused_results
            )
            
            # 显示结果
            cv2.imshow('Basic Detection Example', combined_view)
            
            # 打印检测结果
            if fused_results:
                print(f"帧 {frame_count}: 检测到 {len(fused_results)} 个目标")
                for i, result in enumerate(fused_results):
                    print(f"  目标 {i+1}: {result.detection.class_name} "
                          f"({result.detection.confidence:.2f}) "
                          f"距离: {result.distance_from_camera:.2f}m")
            
            # 检查退出条件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            
            # 每100帧输出一次统计信息
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"统计信息: {frame_count} 帧, 平均FPS: {fps:.2f}")
    
    except KeyboardInterrupt:
        print("\\n接收到中断信号")
    
    finally:
        # 4. 清理资源
        print("4. 清理资源...")
        camera.cleanup()
        detector.cleanup()
        fusion.cleanup()
        visualizer.cleanup()
        cv2.destroyAllWindows()
    
    print("✓ 基础检测示例完成")
    return True

def batch_processing_example():
    """批量处理示例"""
    print("\\n=== 批量处理示例 ===")
    
    # 创建测试图像
    test_images = []
    for i in range(5):
        # 生成随机图像作为测试数据
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_images.append(image)
    
    print(f"✓ 创建了 {len(test_images)} 张测试图像")
    
    # 初始化检测器
    detector = ObjectDetector(YOLO_CONFIG)
    if not detector.initialize():
        print("❌ 目标检测器初始化失败")
        return False
    
    # 批量检测
    print("正在进行批量检测...")
    start_time = time.time()
    
    batch_results = detector.detect_batch(test_images)
    
    processing_time = time.time() - start_time
    
    # 输出结果
    print(f"✓ 批量处理完成")
    print(f"  处理时间: {processing_time:.2f}秒")
    print(f"  平均每张: {processing_time/len(test_images):.3f}秒")
    
    for i, results in enumerate(batch_results):
        print(f"  图像 {i+1}: 检测到 {len(results)} 个目标")
    
    detector.cleanup()
    return True

def performance_benchmark_example():
    """性能基准测试示例"""
    print("\\n=== 性能基准测试示例 ===")
    
    # 检查设备
    devices = discover_realsense_devices()
    if not devices:
        print("❌ 未找到RealSense设备，使用模拟数据")
        return simulate_benchmark()
    
    # 初始化组件
    camera = CameraManager(CAMERA_CONFIG)
    if not camera.initialize():
        print("❌ 相机初始化失败")
        return False
    
    detector = ObjectDetector(YOLO_CONFIG)
    if not detector.initialize():
        print("❌ 目标检测器初始化失败")
        camera.cleanup()
        return False
    
    fusion = DataFusion(FUSION_CONFIG)
    
    print("✓ 开始性能基准测试（30秒）...")
    
    camera.start_streaming(threaded=False)
    
    # 性能统计
    frame_times = []
    detection_times = []
    fusion_times = []
    
    start_time = time.time()
    test_duration = 30  # 30秒
    
    try:
        while time.time() - start_time < test_duration:
            frame_start = time.time()
            
            # 获取帧
            frame_data = camera.get_frame()
            if not frame_data:
                continue
            
            # 检测
            detect_start = time.time()
            detections = detector.detect(frame_data.color_image)
            detect_time = time.time() - detect_start
            
            # 融合
            fusion_start = time.time()
            fused_results = fusion.fuse_detections_with_depth(
                detections, frame_data.depth_image,
                camera.intrinsics.to_dict()
            )
            fusion_time = time.time() - fusion_start
            
            frame_time = time.time() - frame_start
            
            frame_times.append(frame_time)
            detection_times.append(detect_time)
            fusion_times.append(fusion_time)
            
            if len(frame_times) % 30 == 0:
                print(f"已处理 {len(frame_times)} 帧...")
    
    except KeyboardInterrupt:
        print("\\n基准测试被中断")
    
    finally:
        camera.cleanup()
        detector.cleanup()
        fusion.cleanup()
    
    # 输出性能报告
    print_performance_report(frame_times, detection_times, fusion_times)
    return True

def simulate_benchmark():
    """模拟基准测试（无相机）"""
    print("使用模拟数据进行基准测试...")
    
    detector = ObjectDetector(YOLO_CONFIG)
    if not detector.initialize():
        print("❌ 目标检测器初始化失败")
        return False
    
    fusion = DataFusion(FUSION_CONFIG)
    
    # 模拟数据
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_depth = np.random.randint(500, 3000, (480, 640), dtype=np.uint16)
    test_intrinsics = {
        'fx': 600, 'fy': 600,
        'cx': 320, 'cy': 240
    }
    
    # 性能测试
    detection_times = []
    fusion_times = []
    
    print("运行100次模拟检测...")
    
    for i in range(100):
        # 检测
        detect_start = time.time()
        detections = detector.detect(test_image)
        detect_time = time.time() - detect_start
        
        # 融合
        fusion_start = time.time()
        fused_results = fusion.fuse_detections_with_depth(
            detections, test_depth, test_intrinsics
        )
        fusion_time = time.time() - fusion_start
        
        detection_times.append(detect_time)
        fusion_times.append(fusion_time)
        
        if (i + 1) % 20 == 0:
            print(f"已完成 {i + 1}/100")
    
    detector.cleanup()
    fusion.cleanup()
    
    # 输出结果
    print(f"\\n模拟基准测试结果:")
    print(f"  平均检测时间: {np.mean(detection_times)*1000:.2f}ms")
    print(f"  平均融合时间: {np.mean(fusion_times)*1000:.2f}ms")
    print(f"  检测FPS: {1.0/np.mean(detection_times):.2f}")
    
    return True

def print_performance_report(frame_times, detection_times, fusion_times):
    """打印性能报告"""
    print("\\n=== 性能基准测试报告 ===")
    print(f"总帧数: {len(frame_times)}")
    print(f"平均FPS: {1.0/np.mean(frame_times):.2f}")
    print(f"最大FPS: {1.0/np.min(frame_times):.2f}")
    print(f"最小FPS: {1.0/np.max(frame_times):.2f}")
    print(f"平均帧处理时间: {np.mean(frame_times)*1000:.2f}ms")
    print(f"平均检测时间: {np.mean(detection_times)*1000:.2f}ms")
    print(f"平均融合时间: {np.mean(fusion_times)*1000:.2f}ms")
    print(f"检测时间占比: {np.mean(detection_times)/np.mean(frame_times)*100:.1f}%")
    print(f"融合时间占比: {np.mean(fusion_times)/np.mean(frame_times)*100:.1f}%")

def configuration_example():
    """配置示例"""
    print("\\n=== 配置示例 ===")
    
    # 自定义配置
    custom_camera_config = CAMERA_CONFIG.copy()
    custom_camera_config['fps'] = 15  # 降低帧率
    custom_camera_config['width'] = 640
    custom_camera_config['height'] = 480
    
    custom_yolo_config = YOLO_CONFIG.copy()
    custom_yolo_config['confidence_threshold'] = 0.3  # 降低置信度阈值
    custom_yolo_config['model_type'] = 'yolov8n'  # 使用小模型
    
    print("✓ 自定义配置:")
    print(f"  相机FPS: {custom_camera_config['fps']}")
    print(f"  相机分辨率: {custom_camera_config['width']}x{custom_camera_config['height']}")
    print(f"  YOLO置信度: {custom_yolo_config['confidence_threshold']}")
    print(f"  YOLO模型: {custom_yolo_config['model_type']}")
    
    return True

def main():
    """主函数"""
    print("RealSense D435i + YOLO 系统示例")
    print("=" * 50)
    
    try:
        # 1. 基础检测示例
        if not basic_detection_example():
            print("❌ 基础检测示例失败")
            return
        
        # 2. 批量处理示例
        if not batch_processing_example():
            print("❌ 批量处理示例失败")
            return
        
        # 3. 性能基准测试示例
        if not performance_benchmark_example():
            print("❌ 性能基准测试示例失败")
            return
        
        # 4. 配置示例
        if not configuration_example():
            print("❌ 配置示例失败")
            return
        
        print("\\n✓ 所有示例运行完成")
        
    except Exception as e:
        print(f"❌ 运行示例时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()