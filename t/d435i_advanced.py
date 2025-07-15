import pyrealsense2 as rs
import numpy as np
import cv2
import time

'''
增强版RealSense D435i代码
同时显示RGB图像和深度图像，支持多种测距模式和可视化功能
'''

class RealSenseViewer:
    def __init__(self):
        self.pipeline = None
        self.depth_scale = None
        self.align = None
        self.depth_image = None
        self.color_image = None
        self.original_color_image = None
        self.frame_count = 0
        self.measurement_points = []
        self.show_crosshair = True
        self.show_info = True
        self.measurement_mode = 'single'  # 'single', 'multiple', 'line'
        
        # 颜色定义
        self.colors = {
            'point': (0, 255, 0),
            'line': (255, 0, 0),
            'text': (0, 255, 255),
            'crosshair': (255, 255, 255)
        }
    
    def initialize_camera(self):
        """初始化相机"""
        try:
            # 创建管道对象和配置
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # 配置深度流和彩色流
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # 启动流
            profile = self.pipeline.start(config)
            
            # 获取深度传感器和比例
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # 创建对齐对象
            self.align = rs.align(rs.stream.color)
            
            print(f"相机初始化成功！深度比例: {self.depth_scale}")
            return True
            
        except Exception as e:
            print(f"相机初始化失败: {e}")
            return False
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.depth_image is not None:
                distance = self.get_distance_at_point(x, y)
                if distance > 0:
                    if self.measurement_mode == 'single':
                        self.measurement_points = [(x, y, distance)]
                    elif self.measurement_mode == 'multiple':
                        self.measurement_points.append((x, y, distance))
                    elif self.measurement_mode == 'line':
                        self.measurement_points.append((x, y, distance))
                        if len(self.measurement_points) > 2:
                            self.measurement_points.pop(0)
                    
                    print(f"测距点 ({x}, {y}): {distance:.3f} 米")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键清除测量点
            self.measurement_points.clear()
            print("已清除所有测量点")
    
    def get_distance_at_point(self, x, y):
        """获取指定点的距离"""
        if (self.depth_image is not None and 
            0 <= y < self.depth_image.shape[0] and 
            0 <= x < self.depth_image.shape[1]):
            return self.depth_image[y, x] * self.depth_scale
        return 0
    
    def draw_crosshair(self, image, x, y, size=20):
        """绘制十字线"""
        if self.show_crosshair:
            cv2.line(image, (x-size, y), (x+size, y), self.colors['crosshair'], 1)
            cv2.line(image, (x, y-size), (x, y+size), self.colors['crosshair'], 1)
    
    def draw_measurements(self, image):
        """绘制测量结果"""
        for i, (x, y, distance) in enumerate(self.measurement_points):
            # 绘制点
            cv2.circle(image, (x, y), 5, self.colors['point'], -1)
            
            # 绘制距离文本
            text = f"{distance:.2f}m"
            cv2.putText(image, text, (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            
            # 绘制点编号
            if self.measurement_mode == 'multiple':
                cv2.putText(image, str(i+1), (x-10, y+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        # 如果是线段测量模式，绘制线段和距离
        if self.measurement_mode == 'line' and len(self.measurement_points) == 2:
            p1 = self.measurement_points[0]
            p2 = self.measurement_points[1]
            
            # 绘制线段
            cv2.line(image, (p1[0], p1[1]), (p2[0], p2[1]), self.colors['line'], 2)
            
            # 计算3D距离
            distance_3d = self.calculate_3d_distance(p1, p2)
            
            # 在线段中点显示距离
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            cv2.putText(image, f"3D: {distance_3d:.2f}m", (mid_x, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['line'], 2)
    
    def calculate_3d_distance(self, p1, p2):
        """计算两点间的3D距离"""
        # 简化的3D距离计算
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = (p2[2] - p1[2]) * 1000  # 转换为毫米
        
        return np.sqrt(dx*dx + dy*dy + dz*dz) / 1000  # 转换回米
    
    def draw_info_panel(self, image):
        """绘制信息面板"""
        if not self.show_info:
            return
        
        # 信息面板背景
        panel_height = 120
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 显示信息
        info_lines = [
            f"Frame: {self.frame_count}",
            f"Mode: {self.measurement_mode}",
            f"Points: {len(self.measurement_points)}",
            f"Depth Scale: {self.depth_scale:.6f}",
            f"FPS: {self.get_fps():.1f}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(image, line, (10, 20 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def get_fps(self):
        """获取帧率"""
        if hasattr(self, 'last_time'):
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_time)
            self.last_time = current_time
            return fps
        else:
            self.last_time = time.time()
            return 0
    
    def process_frame(self):
        """处理单帧"""
        # 获取数据帧
        frames = self.pipeline.wait_for_frames()
        
        # 对齐帧
        aligned_frames = self.align.process(frames)
        
        # 获取对齐后的深度帧和彩色帧
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None
        
        # 转换为numpy数组
        self.depth_image = np.asanyarray(depth_frame.get_data())
        self.color_image = np.asanyarray(color_frame.get_data())
        self.original_color_image = self.color_image.copy()
        
        # 创建深度图像的彩色可视化
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(self.depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        
        return self.color_image, depth_colormap
    
    def run(self):
        """运行主循环"""
        if not self.initialize_camera():
            return
        
        # 创建窗口
        cv2.namedWindow('RealSense RGB', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('RealSense Depth', cv2.WINDOW_AUTOSIZE)
        
        # 设置鼠标回调
        cv2.setMouseCallback('RealSense RGB', self.mouse_callback)
        cv2.setMouseCallback('RealSense Depth', self.mouse_callback)
        
        print("程序启动成功！")
        print("操作说明:")
        print("鼠标左键 - 测距")
        print("鼠标右键 - 清除测量点")
        print("键盘快捷键:")
        print("  ESC - 退出程序")
        print("  s   - 保存当前帧")
        print("  c   - 切换十字线显示")
        print("  i   - 切换信息面板显示")
        print("  1   - 单点测量模式")
        print("  2   - 多点测量模式")
        print("  3   - 线段测量模式")
        print("  r   - 重置所有测量点")
        print("  h   - 显示帮助")
        
        try:
            while True:
                # 处理帧
                color_image, depth_colormap = self.process_frame()
                
                if color_image is None or depth_colormap is None:
                    continue
                
                # 绘制测量结果
                self.draw_measurements(color_image)
                
                # 绘制信息面板
                self.draw_info_panel(color_image)
                
                # 在深度图上也绘制测量点
                self.draw_measurements(depth_colormap)
                
                # 显示图像
                cv2.imshow('RealSense RGB', color_image)
                cv2.imshow('RealSense Depth', depth_colormap)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('s'):  # 保存
                    cv2.imwrite(f'rgb_frame_{self.frame_count:04d}.jpg', color_image)
                    cv2.imwrite(f'depth_frame_{self.frame_count:04d}.jpg', depth_colormap)
                    print(f"已保存帧 {self.frame_count}")
                elif key == ord('c'):  # 切换十字线
                    self.show_crosshair = not self.show_crosshair
                    print(f"十字线显示: {'开' if self.show_crosshair else '关'}")
                elif key == ord('i'):  # 切换信息面板
                    self.show_info = not self.show_info
                    print(f"信息面板显示: {'开' if self.show_info else '关'}")
                elif key == ord('1'):  # 单点模式
                    self.measurement_mode = 'single'
                    print("切换到单点测量模式")
                elif key == ord('2'):  # 多点模式
                    self.measurement_mode = 'multiple'
                    print("切换到多点测量模式")
                elif key == ord('3'):  # 线段模式
                    self.measurement_mode = 'line'
                    print("切换到线段测量模式")
                elif key == ord('r'):  # 重置
                    self.measurement_points.clear()
                    print("已重置所有测量点")
                elif key == ord('h'):  # 帮助
                    print("\n=== 帮助信息 ===")
                    print("当前模式:", self.measurement_mode)
                    print("测量点数:", len(self.measurement_points))
                    print("十字线显示:", self.show_crosshair)
                    print("信息面板显示:", self.show_info)
                
                self.frame_count += 1
        
        except Exception as e:
            print(f"运行时错误: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 清理资源
            if self.pipeline:
                self.pipeline.stop()
            cv2.destroyAllWindows()
            print("程序已退出")

def main():
    """主函数"""
    viewer = RealSenseViewer()
    viewer.run()

if __name__ == "__main__":
    main()