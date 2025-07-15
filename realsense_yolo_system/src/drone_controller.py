"""
无人机控制器
集成MAVSDK-Python实现PX4无人机控制功能
"""

import asyncio
import logging
import time
import math
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass
from enum import Enum

try:
    from mavsdk import System
    from mavsdk.offboard import PositionNedYaw, VelocityBodyYawspeed, VelocityNedYaw
    from mavsdk.telemetry import Position, Quaternion, VelocityNed, Battery, Gps
    from mavsdk.action import ActionError
    from mavsdk.offboard import OffboardError
except ImportError:
    raise ImportError("请安装MAVSDK-Python: pip install mavsdk")

logger = logging.getLogger(__name__)

class DroneState(Enum):
    """无人机状态枚举"""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    ARMED = "armed"
    TAKING_OFF = "taking_off"
    FLYING = "flying"
    LANDING = "landing"
    LANDED = "landed"
    FOLLOWING = "following"
    EMERGENCY = "emergency"

class ControlMode(Enum):
    """控制模式枚举"""
    MANUAL = "manual"
    POSITION = "position"
    VELOCITY = "velocity"
    OFFBOARD = "offboard"

@dataclass
class DronePosition:
    """无人机位置信息"""
    latitude: float
    longitude: float
    altitude: float
    relative_altitude: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'relative_altitude': self.relative_altitude
        }

@dataclass
class DroneVelocity:
    """无人机速度信息"""
    north: float
    east: float
    down: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'north': self.north,
            'east': self.east,
            'down': self.down
        }

@dataclass
class DroneAttitude:
    """无人机姿态信息"""
    roll: float
    pitch: float
    yaw: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'roll': self.roll,
            'pitch': self.pitch,
            'yaw': self.yaw
        }

@dataclass
class SafetyLimits:
    """安全限制参数"""
    max_speed: float = 5.0
    max_altitude: float = 30.0
    min_altitude: float = 1.0
    safety_radius: float = 100.0
    battery_warning_level: float = 20.0
    battery_critical_level: float = 10.0
    gps_min_satellites: int = 6

class DroneController:
    """PX4无人机控制器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化无人机控制器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.drone = System()
        self.state = DroneState.DISCONNECTED
        self.control_mode = ControlMode.MANUAL
        
        # 连接状态
        self.is_connected = False
        self.is_armed = False
        self.is_flying = False
        self.is_offboard_active = False
        
        # 遥测数据
        self.position: Optional[DronePosition] = None
        self.velocity: Optional[DroneVelocity] = None
        self.attitude: Optional[DroneAttitude] = None
        self.battery: Optional[Battery] = None
        self.gps_info: Optional[Gps] = None
        
        # 安全限制
        self.safety_limits = SafetyLimits(
            max_speed=config.get('max_speed', 5.0),
            max_altitude=config.get('max_altitude', 30.0),
            min_altitude=config.get('min_altitude', 1.0),
            safety_radius=config.get('safety_radius', 100.0),
            battery_warning_level=config.get('battery_warning_level', 20.0),
            battery_critical_level=config.get('battery_critical_level', 10.0),
            gps_min_satellites=config.get('gps_min_satellites', 6)
        )
        
        # 控制参数
        self.takeoff_altitude = config.get('takeoff_altitude', 5.0)
        self.land_speed = config.get('land_speed', 1.0)
        self.emergency_descent_rate = config.get('emergency_descent_rate', 2.0)
        
        # 任务管理
        self.current_task: Optional[asyncio.Task] = None
        self.telemetry_task: Optional[asyncio.Task] = None
        
        # 回调函数
        self.callbacks: Dict[str, List[Callable]] = {
            'on_connection_changed': [],
            'on_armed_changed': [],
            'on_flight_mode_changed': [],
            'on_position_changed': [],
            'on_battery_changed': [],
            'on_emergency': [],
            'on_safety_warning': []
        }
        
        # 性能监控
        self.last_heartbeat = time.time()
        self.telemetry_rate = config.get('telemetry_rate', 10.0)
        
        logger.info("无人机控制器初始化完成")
    
    async def connect(self, system_address: str = "udp://:14540") -> bool:
        """
        连接无人机
        
        Args:
            system_address: 系统地址
            
        Returns:
            连接是否成功
        """
        try:
            logger.info(f"正在连接无人机: {system_address}")
            await self.drone.connect(system_address=system_address)
            
            # 等待连接建立
            logger.info("等待无人机连接...")
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    self.is_connected = True
                    self.state = DroneState.CONNECTED
                    logger.info("无人机连接成功")
                    
                    # 启动遥测监控
                    self.telemetry_task = asyncio.create_task(self._telemetry_monitor())
                    
                    # 触发连接回调
                    self._trigger_callbacks('on_connection_changed', True)
                    
                    return True
                    
        except Exception as e:
            logger.error(f"连接无人机失败: {e}")
            return False
        
        return False
    
    async def disconnect(self):
        """断开连接"""
        try:
            logger.info("正在断开无人机连接...")
            
            # 停止遥测监控
            if self.telemetry_task:
                self.telemetry_task.cancel()
                try:
                    await self.telemetry_task
                except asyncio.CancelledError:
                    pass
            
            # 如果正在飞行，先紧急降落
            if self.is_flying:
                await self.emergency_land()
            
            # 如果已解锁，先上锁
            if self.is_armed:
                await self.disarm()
            
            self.is_connected = False
            self.state = DroneState.DISCONNECTED
            
            # 触发断开回调
            self._trigger_callbacks('on_connection_changed', False)
            
            logger.info("无人机连接已断开")
            
        except Exception as e:
            logger.error(f"断开连接失败: {e}")
    
    async def arm(self) -> bool:
        """
        解锁无人机
        
        Returns:
            解锁是否成功
        """
        try:
            if not self.is_connected:
                logger.error("无人机未连接")
                return False
            
            # 安全检查
            if not await self._pre_arm_checks():
                logger.error("解锁前安全检查失败")
                return False
            
            logger.info("正在解锁无人机...")
            await self.drone.action.arm()
            
            # 等待解锁完成
            await asyncio.sleep(1)
            
            # 验证解锁状态
            async for is_armed in self.drone.telemetry.armed():
                if is_armed:
                    self.is_armed = True
                    self.state = DroneState.ARMED
                    logger.info("无人机解锁成功")
                    
                    # 触发解锁回调
                    self._trigger_callbacks('on_armed_changed', True)
                    
                    return True
                break
            
            return False
            
        except ActionError as e:
            logger.error(f"解锁无人机失败: {e}")
            return False
        except Exception as e:
            logger.error(f"解锁过程中发生错误: {e}")
            return False
    
    async def disarm(self) -> bool:
        """
        上锁无人机
        
        Returns:
            上锁是否成功
        """
        try:
            if not self.is_connected:
                return False
            
            # 如果正在飞行，不能上锁
            if self.is_flying:
                logger.warning("无人机正在飞行，无法上锁")
                return False
            
            logger.info("正在上锁无人机...")
            await self.drone.action.disarm()
            
            # 等待上锁完成
            await asyncio.sleep(1)
            
            # 验证上锁状态
            async for is_armed in self.drone.telemetry.armed():
                if not is_armed:
                    self.is_armed = False
                    self.state = DroneState.CONNECTED
                    logger.info("无人机上锁成功")
                    
                    # 触发上锁回调
                    self._trigger_callbacks('on_armed_changed', False)
                    
                    return True
                break
            
            return False
            
        except ActionError as e:
            logger.error(f"上锁无人机失败: {e}")
            return False
        except Exception as e:
            logger.error(f"上锁过程中发生错误: {e}")
            return False
    
    async def takeoff(self, altitude: Optional[float] = None) -> bool:
        """
        起飞
        
        Args:
            altitude: 起飞高度(米)，None使用默认高度
            
        Returns:
            起飞是否成功
        """
        try:
            if not self.is_armed:
                logger.error("无人机未解锁")
                return False
            
            if self.is_flying:
                logger.warning("无人机已在飞行中")
                return True
            
            takeoff_alt = altitude or self.takeoff_altitude
            
            # 高度安全检查
            if takeoff_alt > self.safety_limits.max_altitude:
                logger.error(f"起飞高度超过安全限制: {takeoff_alt}m > {self.safety_limits.max_altitude}m")
                return False
            
            logger.info(f"正在起飞到高度: {takeoff_alt}m")
            self.state = DroneState.TAKING_OFF
            
            # 设置起飞高度
            await self.drone.action.set_takeoff_altitude(takeoff_alt)
            await self.drone.action.takeoff()
            
            # 等待起飞完成
            start_time = time.time()
            timeout = 30  # 30秒超时
            
            while time.time() - start_time < timeout:
                if self.position and self.position.relative_altitude > takeoff_alt * 0.9:
                    self.is_flying = True
                    self.state = DroneState.FLYING
                    logger.info("起飞完成")
                    return True
                await asyncio.sleep(0.1)
            
            logger.error("起飞超时")
            return False
            
        except ActionError as e:
            logger.error(f"起飞失败: {e}")
            return False
        except Exception as e:
            logger.error(f"起飞过程中发生错误: {e}")
            return False
    
    async def land(self) -> bool:
        """
        降落
        
        Returns:
            降落是否成功
        """
        try:
            if not self.is_flying:
                logger.error("无人机未在飞行状态")
                return False
            
            logger.info("正在降落...")
            self.state = DroneState.LANDING
            
            # 停止offboard模式
            if self.is_offboard_active:
                await self.stop_offboard()
            
            await self.drone.action.land()
            
            # 等待降落完成
            start_time = time.time()
            timeout = 60  # 60秒超时
            
            while time.time() - start_time < timeout:
                if self.position and self.position.relative_altitude < 0.5:
                    self.is_flying = False
                    self.state = DroneState.LANDED
                    logger.info("降落完成")
                    return True
                await asyncio.sleep(0.1)
            
            logger.error("降落超时")
            return False
            
        except ActionError as e:
            logger.error(f"降落失败: {e}")
            return False
        except Exception as e:
            logger.error(f"降落过程中发生错误: {e}")
            return False
    
    async def emergency_land(self):
        """紧急降落"""
        try:
            logger.warning("执行紧急降落")
            self.state = DroneState.EMERGENCY
            
            # 停止offboard模式
            if self.is_offboard_active:
                await self.stop_offboard()
            
            # 紧急降落
            await self.drone.action.land()
            
            # 触发紧急回调
            self._trigger_callbacks('on_emergency', 'emergency_land')
            
        except Exception as e:
            logger.error(f"紧急降落失败: {e}")
    
    async def start_offboard(self) -> bool:
        """
        启动offboard模式
        
        Returns:
            是否成功启动
        """
        try:
            if not self.is_flying:
                logger.error("无人机未在飞行状态")
                return False
            
            logger.info("启动offboard模式...")
            
            # 发送初始位置设定点
            if self.position:
                await self.drone.offboard.set_position_ned(
                    PositionNedYaw(0.0, 0.0, -self.position.relative_altitude, 0.0))
            
            await self.drone.offboard.start()
            self.is_offboard_active = True
            self.control_mode = ControlMode.OFFBOARD
            
            logger.info("offboard模式启动成功")
            return True
            
        except OffboardError as e:
            logger.error(f"启动offboard模式失败: {e}")
            return False
        except Exception as e:
            logger.error(f"启动offboard模式时发生错误: {e}")
            return False
    
    async def stop_offboard(self):
        """停止offboard模式"""
        try:
            if self.is_offboard_active:
                logger.info("停止offboard模式...")
                await self.drone.offboard.stop()
                self.is_offboard_active = False
                self.control_mode = ControlMode.MANUAL
                logger.info("offboard模式已停止")
                
        except OffboardError as e:
            logger.error(f"停止offboard模式失败: {e}")
        except Exception as e:
            logger.error(f"停止offboard模式时发生错误: {e}")
    
    async def goto_position_ned(self, north: float, east: float, down: float, yaw: float = 0.0) -> bool:
        """
        飞向指定位置 (NED坐标系)
        
        Args:
            north: 北向距离(米)
            east: 东向距离(米)
            down: 下向距离(米，负值表示向上)
            yaw: 偏航角(度)
            
        Returns:
            是否成功
        """
        try:
            if not self.is_flying:
                logger.error("无人机未在飞行状态")
                return False
            
            # 安全检查
            if abs(down) > self.safety_limits.max_altitude:
                logger.warning(f"高度超过限制: {abs(down)}m > {self.safety_limits.max_altitude}m")
                return False
            
            # 距离安全检查
            distance = math.sqrt(north**2 + east**2)
            if distance > self.safety_limits.safety_radius:
                logger.warning(f"距离超过安全半径: {distance}m > {self.safety_limits.safety_radius}m")
                return False
            
            # 启动offboard模式
            if not self.is_offboard_active:
                if not await self.start_offboard():
                    return False
            
            # 发送位置命令
            await self.drone.offboard.set_position_ned(
                PositionNedYaw(north, east, down, yaw))
            
            logger.debug(f"飞向位置: N={north:.2f}, E={east:.2f}, D={down:.2f}, Y={yaw:.2f}")
            return True
            
        except OffboardError as e:
            logger.error(f"飞向位置失败: {e}")
            return False
        except Exception as e:
            logger.error(f"飞向位置时发生错误: {e}")
            return False
    
    async def set_velocity_body(self, forward: float, right: float, down: float, yaw_rate: float = 0.0) -> bool:
        """
        设置速度 (Body坐标系)
        
        Args:
            forward: 前进速度(m/s)
            right: 右移速度(m/s)
            down: 下降速度(m/s)
            yaw_rate: 偏航角速度(deg/s)
            
        Returns:
            是否成功
        """
        try:
            if not self.is_flying:
                logger.error("无人机未在飞行状态")
                return False
            
            # 速度限制
            max_speed = self.safety_limits.max_speed
            forward = max(-max_speed, min(max_speed, forward))
            right = max(-max_speed, min(max_speed, right))
            down = max(-max_speed, min(max_speed, down))
            
            # 启动offboard模式
            if not self.is_offboard_active:
                if not await self.start_offboard():
                    return False
            
            # 发送速度命令
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(forward, right, down, yaw_rate))
            
            logger.debug(f"设置速度: F={forward:.2f}, R={right:.2f}, D={down:.2f}, YR={yaw_rate:.2f}")
            return True
            
        except OffboardError as e:
            logger.error(f"设置速度失败: {e}")
            return False
        except Exception as e:
            logger.error(f"设置速度时发生错误: {e}")
            return False
    
    async def set_velocity_ned(self, north: float, east: float, down: float, yaw: float = 0.0) -> bool:
        """
        设置速度 (NED坐标系)
        
        Args:
            north: 北向速度(m/s)
            east: 东向速度(m/s)
            down: 下向速度(m/s)
            yaw: 偏航角(度)
            
        Returns:
            是否成功
        """
        try:
            if not self.is_flying:
                logger.error("无人机未在飞行状态")
                return False
            
            # 速度限制
            max_speed = self.safety_limits.max_speed
            north = max(-max_speed, min(max_speed, north))
            east = max(-max_speed, min(max_speed, east))
            down = max(-max_speed, min(max_speed, down))
            
            # 启动offboard模式
            if not self.is_offboard_active:
                if not await self.start_offboard():
                    return False
            
            # 发送速度命令
            await self.drone.offboard.set_velocity_ned(
                VelocityNedYaw(north, east, down, yaw))
            
            logger.debug(f"设置NED速度: N={north:.2f}, E={east:.2f}, D={down:.2f}, Y={yaw:.2f}")
            return True
            
        except OffboardError as e:
            logger.error(f"设置NED速度失败: {e}")
            return False
        except Exception as e:
            logger.error(f"设置NED速度时发生错误: {e}")
            return False
    
    async def hold_position(self) -> bool:
        """
        保持当前位置
        
        Returns:
            是否成功
        """
        try:
            if not self.is_flying:
                logger.error("无人机未在飞行状态")
                return False
            
            # 停止offboard模式，让无人机进入hold模式
            if self.is_offboard_active:
                await self.stop_offboard()
            
            # 发送hold命令
            await self.drone.action.hold()
            
            logger.info("无人机进入位置保持模式")
            return True
            
        except ActionError as e:
            logger.error(f"位置保持失败: {e}")
            return False
        except Exception as e:
            logger.error(f"位置保持时发生错误: {e}")
            return False
    
    async def return_to_launch(self) -> bool:
        """
        返回起飞点
        
        Returns:
            是否成功
        """
        try:
            if not self.is_flying:
                logger.error("无人机未在飞行状态")
                return False
            
            logger.info("返回起飞点...")
            
            # 停止offboard模式
            if self.is_offboard_active:
                await self.stop_offboard()
            
            await self.drone.action.return_to_launch()
            
            logger.info("返回起飞点命令已发送")
            return True
            
        except ActionError as e:
            logger.error(f"返回起飞点失败: {e}")
            return False
        except Exception as e:
            logger.error(f"返回起飞点时发生错误: {e}")
            return False
    
    async def _telemetry_monitor(self):
        """遥测监控任务"""
        try:
            # 创建遥测监控任务
            tasks = [
                asyncio.create_task(self._monitor_position()),
                asyncio.create_task(self._monitor_velocity()),
                asyncio.create_task(self._monitor_attitude()),
                asyncio.create_task(self._monitor_battery()),
                asyncio.create_task(self._monitor_gps()),
                asyncio.create_task(self._monitor_armed_state()),
                asyncio.create_task(self._monitor_flight_mode())
            ]
            
            # 等待所有任务完成
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"遥测监控错误: {e}")
    
    async def _monitor_position(self):
        """位置监控"""
        try:
            async for position in self.drone.telemetry.position():
                self.position = DronePosition(
                    latitude=position.latitude_deg,
                    longitude=position.longitude_deg,
                    altitude=position.absolute_altitude_m,
                    relative_altitude=position.relative_altitude_m
                )
                
                # 触发位置更新回调
                self._trigger_callbacks('on_position_changed', self.position)
                
                # 更新心跳
                self.last_heartbeat = time.time()
                
        except Exception as e:
            logger.error(f"位置监控错误: {e}")
    
    async def _monitor_velocity(self):
        """速度监控"""
        try:
            async for velocity in self.drone.telemetry.velocity_ned():
                self.velocity = DroneVelocity(
                    north=velocity.north_m_s,
                    east=velocity.east_m_s,
                    down=velocity.down_m_s
                )
                
        except Exception as e:
            logger.error(f"速度监控错误: {e}")
    
    async def _monitor_attitude(self):
        """姿态监控"""
        try:
            async for attitude in self.drone.telemetry.attitude_euler():
                self.attitude = DroneAttitude(
                    roll=attitude.roll_deg,
                    pitch=attitude.pitch_deg,
                    yaw=attitude.yaw_deg
                )
                
        except Exception as e:
            logger.error(f"姿态监控错误: {e}")
    
    async def _monitor_battery(self):
        """电池监控"""
        try:
            async for battery in self.drone.telemetry.battery():
                self.battery = battery
                
                # 电池电量警告
                if battery.remaining_percent < self.safety_limits.battery_critical_level:
                    logger.critical(f"电池电量严重不足: {battery.remaining_percent:.1f}%")
                    self._trigger_callbacks('on_emergency', 'battery_critical')
                elif battery.remaining_percent < self.safety_limits.battery_warning_level:
                    logger.warning(f"电池电量不足: {battery.remaining_percent:.1f}%")
                    self._trigger_callbacks('on_safety_warning', 'battery_low')
                
                # 触发电池更新回调
                self._trigger_callbacks('on_battery_changed', battery)
                
        except Exception as e:
            logger.error(f"电池监控错误: {e}")
    
    async def _monitor_gps(self):
        """GPS监控"""
        try:
            async for gps_info in self.drone.telemetry.gps_info():
                self.gps_info = gps_info
                
                # GPS信号检查
                if gps_info.num_satellites < self.safety_limits.gps_min_satellites:
                    logger.warning(f"GPS卫星数量不足: {gps_info.num_satellites}")
                    self._trigger_callbacks('on_safety_warning', 'gps_weak')
                
        except Exception as e:
            logger.error(f"GPS监控错误: {e}")
    
    async def _monitor_armed_state(self):
        """解锁状态监控"""
        try:
            async for is_armed in self.drone.telemetry.armed():
                if self.is_armed != is_armed:
                    self.is_armed = is_armed
                    self._trigger_callbacks('on_armed_changed', is_armed)
                
        except Exception as e:
            logger.error(f"解锁状态监控错误: {e}")
    
    async def _monitor_flight_mode(self):
        """飞行模式监控"""
        try:
            async for flight_mode in self.drone.telemetry.flight_mode():
                self._trigger_callbacks('on_flight_mode_changed', flight_mode)
                
        except Exception as e:
            logger.error(f"飞行模式监控错误: {e}")
    
    async def _pre_arm_checks(self) -> bool:
        """解锁前安全检查"""
        try:
            logger.info("执行解锁前安全检查...")
            
            # 检查电池电量
            if self.battery and self.battery.remaining_percent < self.safety_limits.battery_warning_level:
                logger.error(f"电池电量过低: {self.battery.remaining_percent:.1f}%")
                return False
            
            # 检查GPS信号
            if self.gps_info and self.gps_info.num_satellites < self.safety_limits.gps_min_satellites:
                logger.error(f"GPS卫星数量不足: {self.gps_info.num_satellites}")
                return False
            
            # 检查系统健康状态
            async for health in self.drone.telemetry.health():
                if not health.is_global_position_ok:
                    logger.error("全球定位不可用")
                    return False
                if not health.is_home_position_ok:
                    logger.error("起飞点位置不可用")
                    return False
                if not health.is_armable:
                    logger.error("无人机无法解锁")
                    return False
                break
            
            logger.info("解锁前安全检查通过")
            return True
            
        except Exception as e:
            logger.error(f"解锁前安全检查失败: {e}")
            return False
    
    def register_callback(self, event: str, callback: Callable):
        """注册回调函数"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            logger.warning(f"未知事件类型: {event}")
    
    def _trigger_callbacks(self, event: str, data: Any):
        """触发回调函数"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"回调函数错误 ({event}): {e}")
    
    def get_telemetry(self) -> Dict[str, Any]:
        """
        获取遥测数据
        
        Returns:
            遥测数据字典
        """
        return {
            'state': self.state.value,
            'control_mode': self.control_mode.value,
            'is_connected': self.is_connected,
            'is_armed': self.is_armed,
            'is_flying': self.is_flying,
            'is_offboard_active': self.is_offboard_active,
            'position': self.position.to_dict() if self.position else None,
            'velocity': self.velocity.to_dict() if self.velocity else None,
            'attitude': self.attitude.to_dict() if self.attitude else None,
            'battery_level': self.battery.remaining_percent if self.battery else 0,
            'gps_satellites': self.gps_info.num_satellites if self.gps_info else 0,
            'last_heartbeat': self.last_heartbeat
        }
    
    def get_safety_status(self) -> Dict[str, Any]:
        """
        获取安全状态
        
        Returns:
            安全状态字典
        """
        warnings = []
        
        # 电池检查
        if self.battery:
            if self.battery.remaining_percent < self.safety_limits.battery_critical_level:
                warnings.append("电池电量严重不足")
            elif self.battery.remaining_percent < self.safety_limits.battery_warning_level:
                warnings.append("电池电量不足")
        
        # 高度检查
        if self.position:
            if self.position.relative_altitude > self.safety_limits.max_altitude * 0.9:
                warnings.append("高度接近限制")
            elif self.position.relative_altitude < self.safety_limits.min_altitude:
                warnings.append("高度过低")
        
        # GPS检查
        if self.gps_info:
            if self.gps_info.num_satellites < self.safety_limits.gps_min_satellites:
                warnings.append("GPS信号弱")
        
        # 连接检查
        if time.time() - self.last_heartbeat > 5.0:
            warnings.append("遥测连接中断")
        
        return {
            'safe': len(warnings) == 0,
            'warnings': warnings,
            'battery_level': self.battery.remaining_percent if self.battery else 0,
            'altitude': self.position.relative_altitude if self.position else 0,
            'gps_satellites': self.gps_info.num_satellites if self.gps_info else 0,
            'limits': {
                'max_altitude': self.safety_limits.max_altitude,
                'max_speed': self.safety_limits.max_speed,
                'safety_radius': self.safety_limits.safety_radius
            }
        }
    
    async def emergency_stop(self):
        """紧急停止"""
        try:
            logger.critical("执行紧急停止")
            self.state = DroneState.EMERGENCY
            
            # 停止offboard模式
            if self.is_offboard_active:
                await self.stop_offboard()
            
            # 悬停
            await self.drone.action.hold()
            
            # 如果高度过低，紧急降落
            if self.position and self.position.relative_altitude < self.safety_limits.min_altitude:
                await self.emergency_land()
            
            # 触发紧急回调
            self._trigger_callbacks('on_emergency', 'emergency_stop')
            
        except Exception as e:
            logger.error(f"紧急停止失败: {e}")
    
    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("正在清理无人机控制器资源...")
            
            # 停止当前任务
            if self.current_task:
                self.current_task.cancel()
                try:
                    await self.current_task
                except asyncio.CancelledError:
                    pass
            
            # 断开连接
            if self.is_connected:
                await self.disconnect()
            
            logger.info("无人机控制器资源清理完成")
            
        except Exception as e:
            logger.error(f"无人机控制器资源清理失败: {e}")