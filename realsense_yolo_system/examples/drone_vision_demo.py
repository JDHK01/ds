#!/usr/bin/env python3
"""
无人机视觉系统交互式演示程序
集成RealSense D435i + YOLO + PX4无人机控制的完整演示

功能特性:
- 多场景配置选择
- 实时目标检测和跟踪
- 无人机控制和跟随
- 交互式界面和控制
- 安全监控和紧急停止
- 性能监控和统计

使用方法:
    python drone_vision_demo.py [--config scene_name] [--simulation]

控制键:
    ESC/Q: 退出程序
    SPACE: 紧急停止
    C: 连接/断开无人机
    A: 解锁/上锁无人机
    T: 起飞/降落
    S: 开始/停止跟踪
    F: 开始/停止跟随
    H: 悬停/位置保持
    R: 返回起飞点
    1-6: 切换场景配置
    M: 显示/隐藏菜单
    I: 显示系统信息
    
鼠标控制:
    左键点击: 选择跟踪目标
    右键点击: 取消跟踪
"""

import asyncio
import cv2
import numpy as np
import time
import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import threading
from dataclasses import dataclass
from enum import Enum
import json

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入系统组件
from src.unified_drone_vision_api import UnifiedDroneVisionAPI, SystemMode, ControlStrategy
from config.drone_vision_config import (
    get_config, get_scene_presets, get_follow_presets, 
    get_tracker_configs, get_recommended_config, print_config_summary
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoMode(Enum):
    """演示模式枚举"""
    VISION_ONLY = "vision_only"
    SIMULATION = "simulation"
    FULL_SYSTEM = "full_system"

@dataclass
class DemoSettings:
    """演示设置"""
    mode: DemoMode = DemoMode.VISION_ONLY
    config_scene: str = "outdoor"
    follow_preset: str = "balanced"
    tracker_type: str = "csrt"
    jetson_optimization: str = "performance"
    show_menu: bool = True
    show_statistics: bool = True
    auto_connect: bool = False
    system_address: str = "udp://:14540"
    fullscreen: bool = False

class MouseController:
    """鼠标控制器"""
    
    def __init__(self, demo_app):
        self.demo_app = demo_app
        self.selection_start = None
        self.selection_end = None
        self.is_selecting = False
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selection_start = (x, y)
            self.is_selecting = True
            
        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_selecting and self.selection_start:
                self.selection_end = (x, y)
                self._handle_selection()
                self.is_selecting = False
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键停止跟踪
            self.demo_app.stop_tracking()

    def _handle_selection(self):
        """处理选择区域"""
        if not self.selection_start or not self.selection_end:
            return
            
        x1, y1 = self.selection_start
        x2, y2 = self.selection_end
        
        # 计算边界框
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        
        # 最小选择区域
        if w < 20 or h < 20:
            print("选择区域太小，请重新选择")
            return
            
        # 开始跟踪
        bbox = (x, y, w, h)
        self.demo_app.start_tracking(bbox)

class DroneVisionDemo:
    """无人机视觉演示应用"""
    
    def __init__(self, settings: DemoSettings):
        """初始化演示应用"""
        self.settings = settings
        self.api: Optional[UnifiedDroneVisionAPI] = None
        self.mouse_controller = MouseController(self)
        
        # 界面状态
        self.window_name = "无人机视觉系统演示"
        self.display_image = None
        self.current_result = None
        self.is_running = False
        
        # 性能监控
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # 按键状态
        self.key_handlers = {
            ord('q'): self.quit_demo,
            ord('Q'): self.quit_demo,
            27: self.quit_demo,  # ESC
            ord(' '): self.emergency_stop,
            ord('c'): self.toggle_drone_connection,
            ord('C'): self.toggle_drone_connection,
            ord('a'): self.toggle_arm,
            ord('A'): self.toggle_arm,
            ord('t'): self.toggle_takeoff_land,
            ord('T'): self.toggle_takeoff_land,
            ord('s'): self.toggle_tracking,
            ord('S'): self.toggle_tracking,
            ord('f'): self.toggle_following,
            ord('F'): self.toggle_following,
            ord('h'): self.hold_position,
            ord('H'): self.hold_position,
            ord('r'): self.return_to_launch,
            ord('R'): self.return_to_launch,
            ord('m'): self.toggle_menu,
            ord('M'): self.toggle_menu,
            ord('i'): self.show_system_info,
            ord('I'): self.show_system_info,
            ord('1'): lambda: self.switch_config('indoor'),
            ord('2'): lambda: self.switch_config('outdoor'),
            ord('3'): lambda: self.switch_config('high_performance'),
            ord('4'): lambda: self.switch_config('low_power'),
            ord('5'): lambda: self.switch_config('precision_tracking'),
            ord('6'): lambda: self.switch_config('fast_response'),
        }
        
        # 状态信息
        self.status_messages = []
        self.max_status_messages = 10
        
    def add_status_message(self, message: str, level: str = "INFO"):
        """添加状态消息"""
        timestamp = time.strftime("%H:%M:%S")
        status_msg = f"[{timestamp}] {level}: {message}"
        self.status_messages.append(status_msg)
        
        # 限制消息数量
        if len(self.status_messages) > self.max_status_messages:
            self.status_messages.pop(0)
            
        print(status_msg)
        
    async def initialize(self) -> bool:
        """初始化系统"""
        try:
            self.add_status_message("正在初始化无人机视觉系统...")
            
            # 获取配置
            config = get_config(
                scene=self.settings.config_scene,
                follow_preset=self.settings.follow_preset,
                tracker=self.settings.tracker_type,
                jetson_optimization=self.settings.jetson_optimization
            )
            
            # 打印配置摘要
            print_config_summary(config)
            
            # 创建API实例
            self.api = UnifiedDroneVisionAPI(config)
            
            # 注册回调函数
            self._register_callbacks()
            
            # 初始化组件
            if not self.api.initialize():
                self.add_status_message("系统初始化失败", "ERROR")
                return False
            
            self.add_status_message("系统初始化成功")
            
            # 启动视觉系统
            if not self.api.start_vision_system():
                self.add_status_message("视觉系统启动失败", "ERROR")
                return False
            
            self.add_status_message("视觉系统启动成功")
            
            # 自动连接无人机
            if self.settings.auto_connect and self.settings.mode != DemoMode.VISION_ONLY:
                await self.connect_drone()
            
            return True
            
        except Exception as e:
            self.add_status_message(f"初始化失败: {e}", "ERROR")
            logger.error(f"初始化失败: {e}")
            return False
    
    def _register_callbacks(self):
        """注册系统回调"""
        if not self.api:
            return
            
        self.api.register_callback('on_detection', self._on_detection)
        self.api.register_callback('on_tracking_update', self._on_tracking_update)
        self.api.register_callback('on_target_lost', self._on_target_lost)
        self.api.register_callback('on_follow_start', self._on_follow_start)
        self.api.register_callback('on_follow_stop', self._on_follow_stop)
        self.api.register_callback('on_emergency', self._on_emergency)
        self.api.register_callback('on_mode_change', self._on_mode_change)
        self.api.register_callback('on_error', self._on_error)
    
    def _on_detection(self, detections):
        """检测回调"""
        pass  # 在主显示循环中处理
    
    def _on_tracking_update(self, target):
        """跟踪更新回调"""
        pass  # 在主显示循环中处理
    
    def _on_target_lost(self, data):
        """目标丢失回调"""
        self.add_status_message("目标丢失", "WARNING")
    
    def _on_follow_start(self, params):
        """跟随开始回调"""
        self.add_status_message("开始跟随目标")
    
    def _on_follow_stop(self, data):
        """跟随停止回调"""
        self.add_status_message("停止跟随")
    
    def _on_emergency(self, emergency_type):
        """紧急事件回调"""
        self.add_status_message(f"紧急事件: {emergency_type}", "CRITICAL")
    
    def _on_mode_change(self, new_mode):
        """模式变化回调"""
        self.add_status_message(f"系统模式: {new_mode.value}")
    
    def _on_error(self, error_msg):
        """错误回调"""
        self.add_status_message(f"错误: {error_msg}", "ERROR")
    
    async def run(self):
        """运行演示"""
        try:
            # 初始化系统
            if not await self.initialize():
                return
            
            # 创建显示窗口
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.window_name, self.mouse_controller.mouse_callback)
            
            self.is_running = True
            self.add_status_message("演示程序启动成功，按'M'显示帮助菜单")
            
            # 主循环
            while self.is_running:
                await self._update_display()
                await self._handle_input()
                await asyncio.sleep(0.01)  # 让出控制权
                
        except KeyboardInterrupt:
            self.add_status_message("接收到中断信号")
        except Exception as e:
            self.add_status_message(f"运行时错误: {e}", "ERROR")
            logger.error(f"运行时错误: {e}")
        finally:
            await self.cleanup()
    
    async def _update_display(self):
        """更新显示"""
        try:
            if not self.api:
                return
                
            # 获取最新结果
            result = self.api.get_latest_result()
            if result:
                self.current_result = result
                
            # 创建显示图像
            if self.current_result:
                self.display_image = self._create_display_image(self.current_result)
            
            # 显示图像
            if self.display_image is not None:
                cv2.imshow(self.window_name, self.display_image)
                
                # 更新FPS
                self.fps_counter += 1
                if self.fps_counter % 30 == 0:
                    elapsed = time.time() - self.fps_start_time
                    self.current_fps = 30 / elapsed
                    self.fps_start_time = time.time()
                    
        except Exception as e:
            logger.error(f"更新显示失败: {e}")
    
    def _create_display_image(self, result: Dict[str, Any]) -> np.ndarray:
        """创建显示图像"""
        try:
            frame_data = result['frame_data']
            detections = result['detections']
            tracking_targets = result['tracking_targets']
            current_target = result['current_target']
            
            # 获取彩色图像
            display_img = frame_data.color_image.copy()
            
            # 绘制检测结果
            for detection in detections:
                bbox = detection.bbox
                x, y, w, h = bbox
                
                # 绘制边界框
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 绘制标签
                label = f"{detection.class_name} {detection.confidence:.2f}"
                cv2.putText(display_img, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制跟踪目标
            for target in tracking_targets:
                bbox = target.bbox
                x, y, w, h = bbox
                
                # 跟踪框颜色
                color = (0, 0, 255) if target == current_target else (255, 0, 0)
                thickness = 3 if target == current_target else 2
                
                # 绘制跟踪框
                cv2.rectangle(display_img, (x, y), (x + w, y + h), color, thickness)
                
                # 绘制中心点
                center = target.center
                cv2.circle(display_img, center, 5, color, -1)
                
                # 绘制目标信息
                info_text = f"ID:{target.id} Conf:{target.confidence:.2f}"
                if target.depth > 0:
                    info_text += f" Dist:{target.depth:.1f}m"
                
                cv2.putText(display_img, info_text, (x, y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 绘制速度向量
                if target.velocity != (0.0, 0.0):
                    vx, vy = target.velocity
                    end_point = (int(center[0] + vx * 10), int(center[1] + vy * 10))
                    cv2.arrowedLine(display_img, center, end_point, color, 2)
            
            # 绘制选择框
            if self.mouse_controller.is_selecting and self.mouse_controller.selection_start:
                start = self.mouse_controller.selection_start
                current_pos = cv2.getMousePos()
                if current_pos != (-1, -1):
                    cv2.rectangle(display_img, start, current_pos, (255, 255, 255), 2)
            
            # 添加信息叠加
            self._add_info_overlay(display_img)
            
            return display_img
            
        except Exception as e:
            logger.error(f"创建显示图像失败: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def _add_info_overlay(self, image: np.ndarray):
        """添加信息叠加"""
        try:
            h, w = image.shape[:2]
            
            # 系统状态
            if self.api:
                status = self.api.get_system_status()
                mode = status['mode']
                
                # 模式信息
                mode_text = f"Mode: {mode}"
                cv2.putText(image, mode_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # FPS信息
                fps_text = f"FPS: {self.current_fps:.1f}"
                cv2.putText(image, fps_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 无人机状态
                if status['is_drone_connected']:
                    drone_text = f"Drone: {'Flying' if status['drone_status'] and status['drone_status']['is_flying'] else 'Connected'}"
                    cv2.putText(image, drone_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(image, "Drone: Disconnected", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 跟踪状态
                if status['current_target']:
                    target_text = f"Target: ID {status['current_target']['id']}"
                    cv2.putText(image, target_text, (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 显示帮助菜单
            if self.settings.show_menu:
                self._draw_help_menu(image)
                
            # 显示状态消息
            self._draw_status_messages(image)
            
        except Exception as e:
            logger.error(f"添加信息叠加失败: {e}")
    
    def _draw_help_menu(self, image: np.ndarray):
        """绘制帮助菜单"""
        try:
            h, w = image.shape[:2]
            
            # 菜单背景
            menu_x = w - 350
            menu_y = 10
            menu_w = 340
            menu_h = 400
            
            # 半透明背景
            overlay = image.copy()
            cv2.rectangle(overlay, (menu_x, menu_y), (menu_x + menu_w, menu_y + menu_h), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            # 菜单标题
            cv2.putText(image, "Control Menu", (menu_x + 10, menu_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 菜单项
            menu_items = [
                ("ESC/Q: Quit", (255, 255, 255)),
                ("SPACE: Emergency Stop", (255, 100, 100)),
                ("C: Connect/Disconnect Drone", (100, 255, 100)),
                ("A: Arm/Disarm", (100, 255, 100)),
                ("T: Takeoff/Land", (100, 255, 100)),
                ("S: Start/Stop Tracking", (100, 255, 255)),
                ("F: Start/Stop Following", (100, 255, 255)),
                ("H: Hold Position", (100, 255, 100)),
                ("R: Return to Launch", (100, 255, 100)),
                ("1-6: Switch Config", (255, 255, 100)),
                ("M: Toggle Menu", (255, 255, 255)),
                ("I: System Info", (255, 255, 255)),
                ("Left Click: Select Target", (255, 255, 100)),
                ("Right Click: Cancel Track", (255, 255, 100)),
            ]
            
            y_offset = 60
            for item, color in menu_items:
                cv2.putText(image, item, (menu_x + 10, menu_y + y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += 25
                
        except Exception as e:
            logger.error(f"绘制帮助菜单失败: {e}")
    
    def _draw_status_messages(self, image: np.ndarray):
        """绘制状态消息"""
        try:
            h, w = image.shape[:2]
            
            # 状态消息区域
            msg_x = 10
            msg_y = h - 30 * len(self.status_messages) - 10
            
            for i, msg in enumerate(self.status_messages):
                y_pos = msg_y + i * 25
                
                # 根据消息级别设置颜色
                if "ERROR" in msg or "CRITICAL" in msg:
                    color = (0, 0, 255)
                elif "WARNING" in msg:
                    color = (0, 255, 255)
                else:
                    color = (255, 255, 255)
                
                cv2.putText(image, msg, (msg_x, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
        except Exception as e:
            logger.error(f"绘制状态消息失败: {e}")
    
    async def _handle_input(self):
        """处理输入"""
        try:
            key = cv2.waitKey(1) & 0xFF
            if key in self.key_handlers:
                handler = self.key_handlers[key]
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
                    
        except Exception as e:
            logger.error(f"处理输入失败: {e}")
    
    def quit_demo(self):
        """退出演示"""
        self.add_status_message("正在退出演示程序...")
        self.is_running = False
    
    async def emergency_stop(self):
        """紧急停止"""
        self.add_status_message("执行紧急停止", "CRITICAL")
        if self.api:
            await self.api.emergency_stop()
    
    async def toggle_drone_connection(self):
        """切换无人机连接"""
        if not self.api:
            return
            
        if self.api.is_drone_connected:
            await self.api.disconnect_drone()
            self.add_status_message("无人机已断开连接")
        else:
            self.add_status_message("正在连接无人机...")
            success = await self.api.connect_drone(self.settings.system_address)
            if success:
                self.add_status_message("无人机连接成功")
            else:
                self.add_status_message("无人机连接失败", "ERROR")
    
    async def connect_drone(self):
        """连接无人机"""
        if not self.api:
            return
            
        self.add_status_message("正在连接无人机...")
        success = await self.api.connect_drone(self.settings.system_address)
        if success:
            self.add_status_message("无人机连接成功")
        else:
            self.add_status_message("无人机连接失败", "ERROR")
    
    async def toggle_arm(self):
        """切换解锁状态"""
        if not self.api or not self.api.drone_controller:
            self.add_status_message("无人机未连接", "ERROR")
            return
            
        if self.api.drone_controller.is_armed:
            await self.api.drone_controller.disarm()
            self.add_status_message("无人机已上锁")
        else:
            success = await self.api.drone_controller.arm()
            if success:
                self.add_status_message("无人机已解锁")
            else:
                self.add_status_message("无人机解锁失败", "ERROR")
    
    async def toggle_takeoff_land(self):
        """切换起飞/降落"""
        if not self.api or not self.api.drone_controller:
            self.add_status_message("无人机未连接", "ERROR")
            return
            
        if self.api.drone_controller.is_flying:
            success = await self.api.drone_controller.land()
            if success:
                self.add_status_message("无人机已降落")
            else:
                self.add_status_message("降落失败", "ERROR")
        else:
            success = await self.api.drone_controller.takeoff()
            if success:
                self.add_status_message("无人机已起飞")
            else:
                self.add_status_message("起飞失败", "ERROR")
    
    def toggle_tracking(self):
        """切换跟踪状态"""
        if not self.api:
            return
            
        if self.api.tracking_controller and self.api.tracking_controller.get_active_target():
            self.api.stop_tracking()
            self.add_status_message("停止跟踪")
        else:
            self.add_status_message("请用鼠标选择跟踪目标", "INFO")
    
    def start_tracking(self, bbox: Tuple[int, int, int, int]):
        """开始跟踪"""
        if not self.api:
            return
            
        success = self.api.start_tracking(bbox)
        if success:
            self.add_status_message(f"开始跟踪目标: {bbox}")
        else:
            self.add_status_message("开始跟踪失败", "ERROR")
    
    def stop_tracking(self):
        """停止跟踪"""
        if not self.api:
            return
            
        self.api.stop_tracking()
        self.add_status_message("停止跟踪")
    
    async def toggle_following(self):
        """切换跟随状态"""
        if not self.api:
            return
            
        if self.api.is_following:
            await self.api.stop_following()
            self.add_status_message("停止跟随")
        else:
            # 需要有活动目标才能开始跟随
            if not self.api.current_target:
                self.add_status_message("请先选择跟踪目标", "WARNING")
                return
                
            # 需要无人机在飞行状态
            if not self.api.drone_controller or not self.api.drone_controller.is_flying:
                self.add_status_message("无人机未在飞行状态", "WARNING")
                return
                
            # 使用当前目标的边界框开始跟随
            bbox = self.api.current_target.bbox
            success = await self.api.start_following(bbox)
            if success:
                self.add_status_message("开始跟随目标")
            else:
                self.add_status_message("开始跟随失败", "ERROR")
    
    async def hold_position(self):
        """位置保持"""
        if not self.api or not self.api.drone_controller:
            self.add_status_message("无人机未连接", "ERROR")
            return
            
        success = await self.api.drone_controller.hold_position()
        if success:
            self.add_status_message("无人机进入位置保持模式")
        else:
            self.add_status_message("位置保持失败", "ERROR")
    
    async def return_to_launch(self):
        """返回起飞点"""
        if not self.api or not self.api.drone_controller:
            self.add_status_message("无人机未连接", "ERROR")
            return
            
        success = await self.api.drone_controller.return_to_launch()
        if success:
            self.add_status_message("无人机返回起飞点")
        else:
            self.add_status_message("返回起飞点失败", "ERROR")
    
    def toggle_menu(self):
        """切换菜单显示"""
        self.settings.show_menu = not self.settings.show_menu
        self.add_status_message(f"菜单显示: {'开启' if self.settings.show_menu else '关闭'}")
    
    def show_system_info(self):
        """显示系统信息"""
        if not self.api:
            return
            
        status = self.api.get_system_status()
        performance = self.api.get_performance_stats()
        
        info_lines = [
            "=== 系统信息 ===",
            f"模式: {status['mode']}",
            f"视觉系统: {'运行中' if status['is_vision_active'] else '已停止'}",
            f"无人机: {'已连接' if status['is_drone_connected'] else '未连接'}",
            f"跟随状态: {'激活' if status['is_following'] else '未激活'}",
            f"当前目标: {status['current_target']['id'] if status['current_target'] else '无'}",
            f"运行时间: {performance['uptime']:.1f}秒",
            f"帧率: {performance['frame_rate']:.1f}fps",
            f"总帧数: {performance['total_frames']}",
            f"检测数: {performance['total_detections']}",
            f"跟踪数: {performance['total_tracks']}",
            f"错误数: {performance['total_errors']}",
        ]
        
        for line in info_lines:
            self.add_status_message(line)
    
    def switch_config(self, scene: str):
        """切换配置场景"""
        self.settings.config_scene = scene
        self.add_status_message(f"已切换到场景: {scene}")
        # 注意：实际切换需要重新初始化系统
        self.add_status_message("配置切换需要重启系统", "WARNING")
    
    async def cleanup(self):
        """清理资源"""
        try:
            self.add_status_message("正在清理资源...")
            
            if self.api:
                await self.api.cleanup()
                
            cv2.destroyAllWindows()
            
            self.add_status_message("资源清理完成")
            
        except Exception as e:
            logger.error(f"清理资源失败: {e}")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='无人机视觉系统演示程序')
    
    parser.add_argument('--config', type=str, default='outdoor',
                       choices=['indoor', 'outdoor', 'high_performance', 'low_power', 
                               'precision_tracking', 'fast_response'],
                       help='选择配置场景')
    
    parser.add_argument('--follow-preset', type=str, default='balanced',
                       choices=['conservative', 'balanced', 'aggressive', 'cinematic'],
                       help='跟随预设')
    
    parser.add_argument('--tracker', type=str, default='csrt',
                       choices=['csrt', 'kcf', 'mosse', 'medianflow', 'yolo_native'],
                       help='跟踪算法')
    
    parser.add_argument('--jetson-optimization', type=str, default='performance',
                       choices=['performance', 'memory', 'power'],
                       help='Jetson优化模式')
    
    parser.add_argument('--simulation', action='store_true',
                       help='仿真模式（不连接真实无人机）')
    
    parser.add_argument('--vision-only', action='store_true',
                       help='仅视觉模式（不使用无人机功能）')
    
    parser.add_argument('--auto-connect', action='store_true',
                       help='自动连接无人机')
    
    parser.add_argument('--system-address', type=str, default='udp://:14540',
                       help='无人机系统地址')
    
    parser.add_argument('--fullscreen', action='store_true',
                       help='全屏显示')
    
    parser.add_argument('--no-menu', action='store_true',
                       help='隐藏控制菜单')
    
    return parser.parse_args()

def print_welcome():
    """打印欢迎信息"""
    print("=" * 60)
    print("    无人机视觉系统交互式演示程序")
    print("    RealSense D435i + YOLO + PX4 Drone Control")
    print("=" * 60)
    print()
    print("功能特性:")
    print("  • 实时目标检测和跟踪")
    print("  • PX4无人机控制和跟随")
    print("  • 多场景配置和优化")
    print("  • 交互式控制界面")
    print("  • 安全监控和紧急停止")
    print("  • 性能监控和统计")
    print()
    print("控制说明:")
    print("  • ESC/Q: 退出程序")
    print("  • SPACE: 紧急停止")
    print("  • C: 连接/断开无人机")
    print("  • A: 解锁/上锁无人机")
    print("  • T: 起飞/降落")
    print("  • S: 开始/停止跟踪")
    print("  • F: 开始/停止跟随")
    print("  • M: 显示/隐藏菜单")
    print("  • 左键点击: 选择跟踪目标")
    print("  • 右键点击: 取消跟踪")
    print()

async def main():
    """主函数"""
    print_welcome()
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建演示设置
    settings = DemoSettings(
        config_scene=args.config,
        follow_preset=args.follow_preset,
        tracker_type=args.tracker,
        jetson_optimization=args.jetson_optimization,
        auto_connect=args.auto_connect,
        system_address=args.system_address,
        fullscreen=args.fullscreen,
        show_menu=not args.no_menu
    )
    
    # 确定演示模式
    if args.vision_only:
        settings.mode = DemoMode.VISION_ONLY
    elif args.simulation:
        settings.mode = DemoMode.SIMULATION
    else:
        settings.mode = DemoMode.FULL_SYSTEM
    
    print(f"演示模式: {settings.mode.value}")
    print(f"配置场景: {settings.config_scene}")
    print(f"跟随预设: {settings.follow_preset}")
    print(f"跟踪算法: {settings.tracker_type}")
    print(f"Jetson优化: {settings.jetson_optimization}")
    print()
    
    # 创建并运行演示应用
    demo = DroneVisionDemo(settings)
    
    try:
        await demo.run()
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在退出...")
    except Exception as e:
        print(f"演示程序错误: {e}")
        logger.error(f"演示程序错误: {e}")
    finally:
        await demo.cleanup()

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())