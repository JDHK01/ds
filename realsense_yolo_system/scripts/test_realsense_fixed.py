#!/usr/bin/env python3
"""
修复后的RealSense D435i测试代码
包含错误处理和兼容性检查
"""

import sys
import numpy as np
import cv2

def check_realsense_installation():
    """检查RealSense安装"""
    try:
        import pyrealsense2 as rs
        print("✅ pyrealsense2 导入成功")
        
        # 检查版本
        try:
            version = getattr(rs, '__version__', '未知版本')
            print(f"版本: {version}")
        except:
            print("无法获取版本信息")
        
        # 检查核心对象
        if not hasattr(rs, 'pipeline'):
            print("❌ 错误: rs.pipeline 不存在")
            return False, None
            
        if not hasattr(rs, 'config'):
            print("❌ 错误: rs.config 不存在")
            return False, None
            
        print("✅ 核心对象检查通过")
        return True, rs
        
    except ImportError as e:
        print(f"❌ 无法导入 pyrealsense2: {e}")
        return False, None

def check_devices(rs):
    """检查设备连接"""
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("❌ 未检测到RealSense设备")
            return False
        
        print(f"✅ 检测到 {len(devices)} 个设备:")
        for i, device in enumerate(devices):
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            print(f"  设备 {i+1}: {name} (序列号: {serial})")
        
        return True
        
    except Exception as e:
        print(f"❌ 设备检查失败: {e}")
        return False

def main():
    """主函数"""
    print("RealSense D435i 测试程序")
    print("=" * 40)
    
    # 检查安装
    success, rs = check_realsense_installation()
    if not success:
        print("\\n请先解决 pyrealsense2 安装问题")
        print("运行诊断脚本: python scripts/diagnose_realsense.py")
        return
    
    # 检查设备
    if not check_devices(rs):
        print("\\n请检查设备连接:")
        print("1. 确保D435i已连接到USB 3.0端口")
        print("2. 检查USB权限")
        print("3. 运行: sudo chmod 666 /dev/bus/usb/*/*")
        return
    
    # 创建pipeline
    print("\\n初始化相机...")
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        
        # 配置流
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 启动pipeline
        profile = pipeline.start(config)
        
        # 获取深度传感器和比例
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"✅ 深度比例: {depth_scale}")
        
        # 创建对齐对象
        align = rs.align(rs.stream.color)
        
        # 全局变量用于鼠标回调
        global depth_image
        depth_image = None
        
        # 鼠标回调函数
        def mouse_callback(event, x, y, flags, param):
            global depth_image
            if event == cv2.EVENT_LBUTTONDOWN and depth_image is not None:
                if 0 <= y < depth_image.shape[0] and 0 <= x < depth_image.shape[1]:
                    distance = depth_image[y, x] * depth_scale
                    print(f"坐标 ({x}, {y}) 距离: {distance:.3f} 米")
        
        # 创建窗口
        cv2.namedWindow('RealSense RGB', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('RealSense Depth', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('RealSense Depth', mouse_callback)
        
        print("\\n✅ 相机启动成功！")
        print("操作说明:")
        print("- 在深度图像上点击鼠标左键查看距离")
        print("- 按 ESC 退出")
        print("- 按 's' 保存当前帧")
        
        frame_count = 0
        
        while True:
            # 获取帧
            frames = pipeline.wait_for_frames()
            
            # 对齐帧
            aligned_frames = align.process(frames)
            
            # 获取对齐后的帧
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # 转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 创建深度彩色图
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            # 显示图像
            cv2.imshow('RealSense RGB', color_image)
            cv2.imshow('RealSense Depth', depth_colormap)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('s'):  # 保存
                cv2.imwrite(f'rgb_frame_{frame_count}.jpg', color_image)
                cv2.imwrite(f'depth_frame_{frame_count}.jpg', depth_colormap)
                print(f"已保存帧 {frame_count}")
                frame_count += 1
    
    except Exception as e:
        print(f"❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        try:
            pipeline.stop()
            cv2.destroyAllWindows()
            print("\\n✅ 程序正常退出")
        except:
            pass

if __name__ == "__main__":
    main()