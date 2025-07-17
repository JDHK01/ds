import pyrealsense2 as rs
import numpy as np
import cv2


'''
使用rs的两个对象 .pipeline 和 .enable_stream
    创建管道对象, 管理rs的数据流
    想要从pipeline中获取的参数 深度数据 640*480 30帧 像素格式是16位无符号
    启动流
'''
pipeline = rs.pipeline()
config = rs.config();config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

'''
获取传感器模块 + 获取转换比例
PS:这里获得的值每个像素表示一个深度, 但是这里的深度不是距离, 需要乘以一个系数
'''
depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor();
depth_scale = depth_sensor.get_depth_scale()

'''
使用rs的一个方法, .align方法
'''
align_to = rs.stream.depth
align = rs.align(align_to)

# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global depth_image
    if event == cv2.EVENT_LBUTTONDOWN:
        distance = depth_image[y, x] * depth_scale
        print(f"Distance at ({x}, {y}): {distance:.3f} meters")

try:
    # cv2中对窗口的定义: 窗口名 + 在窗口中按下的回掉函数
    cv2.namedWindow('RealSense Depth', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('RealSense Depth', mouse_callback)

    while True:
        '''
        pipeline的.wait_for_frames()方法, 获得数据帧
        algin的.process()方法,对齐一下
        用.get_depth_frame()获得深度专用帧
        '''
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        cv2.imshow('RealSense Depth', depth_colormap)
        key = cv2.waitKey(1)
        if key == 27:  # 按下 Esc 退出
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
