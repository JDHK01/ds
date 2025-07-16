import pyrealsense2 as rs
import numpy as np
import cv2

'''
修改后的RealSense D435i代码
同时显示RGB图像和深度图像，支持测距功能
'''

# 创建管道 + 创建配置 + 启动流
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 获取转换比例
depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# 创建对齐对象: 将深度对齐到彩色
align_to_color = rs.align(rs.stream.color)

# 全局变量
depth_image = None
color_image = None

# 鼠标回调函数 - 用于测距
def mouse_callback(event, x, y, flags, param):
    global depth_image, color_image
    if event == cv2.EVENT_LBUTTONDOWN:
        if depth_image is not None:
            # 确保点击位置在图像范围内
            if 0 <= y < depth_image.shape[0] and 0 <= x < depth_image.shape[1]:
                distance = depth_image[y, x] * depth_scale
                print(f"点击位置 ({x}, {y}): 距离 {distance:.3f} 米")
                
                # 在RGB图像上绘制点击点和距离信息
                if color_image is not None:
                    # cv2.circle(<帧>, <x,y坐标>, <圆的半径>, <颜色>, <粗细>)
                    # cv2.putText(<帧>, <文本>, <文本位置>, <字体>, <字体大小>, <颜色>, <字体粗细>)
                    cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(color_image, f"{distance:.2f}m", (x+10, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

try:
    # 创建窗口 + 回调
    cv2.namedWindow('RealSense RGB', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('RealSense Depth', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('RealSense RGB', mouse_callback)
    cv2.setMouseCallback('RealSense Depth', mouse_callback)
    
    print("操作说明:")
    print("在RGB或深度图像窗口中点击鼠标左键进行测距")
    print("按 ESC 键退出程序")
    print("按 's' 键保存当前帧")
    print("按 'r' 键重置RGB图像上的标记")
    
    frame_count = 0
    
    while True:
        # 获取原始帧 + 对齐 + 获取对齐帧
        frames = pipeline.wait_for_frames()
        aligned_frames = align_to_color.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()   

        # 将对齐帧转换为可读信息(每个像素包含一个大小)
        if not depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # 将深度图像的深度进行映射, 确定颜色
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        
        # 在深度图像上添加距离信息显示
        # 在图像上显示一些基本信息
        info_text = f"Frame: {frame_count} | Scale: {depth_scale:.6f}"
        cv2.putText(depth_colormap, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # 在RGB图像上显示信息
        cv2.putText(color_image, "Click to measure distance", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(color_image, f"Frame: {frame_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示图像
        cv2.imshow('RealSense RGB', color_image)
        cv2.imshow('RealSense Depth', depth_colormap)
        


        # test
        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC键退出
            break
        elif key == ord('s'):  # 保存当前帧
            cv2.imwrite(f'rgb_frame_{frame_count:04d}.jpg', color_image)
            cv2.imwrite(f'depth_frame_{frame_count:04d}.jpg', depth_colormap)
            print(f"已保存帧 {frame_count}")
        elif key == ord('r'):  # 重置RGB图像标记
            color_image = np.asanyarray(color_frame.get_data())
            print("已重置RGB图像标记")
        elif key == ord('h'):  # 显示帮助
            print("\n键盘快捷键:")
            print("ESC - 退出程序")
            print("s   - 保存当前帧")
            print("r   - 重置RGB图像标记")
            print("h   - 显示此帮助信息")
            print("鼠标左键 - 测距")
        
        frame_count += 1

except Exception as e:
    print(f"发生错误: {e}")
    import traceback
    traceback.print_exc()

finally:
    # 停止流并关闭窗口
    pipeline.stop()
    cv2.destroyAllWindows()
    print("程序已退出")