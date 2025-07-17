import asyncio
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, OffboardError
from mavsdk.telemetry import LandedState

async def concurrent_operations():
    drone = System()
    await drone.connect()
    
    # 创建并发任务
    telemetry_task = asyncio.create_task(monitor_position(drone))
    status_task = asyncio.create_task(monitor_velocity(drone))
    
    # 主要飞行逻辑
    await drone.action.arm()
    await drone.action.takeoff()
    await asyncio.sleep(30)
    await drone.action.land()
    
    # 清理后台任务
    telemetry_task.cancel()
    status_task.cancel()
    
    # 等待任务完成
    await asyncio.gather(telemetry_task, status_task, return_exceptions=True)

async def run():
    """ Does Offboard control using position NED coordinates. """

    drone = System()
    await drone.connect(system_address="udp://127.0.0.1:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    print("Waiting for drone to have a local position estimate...")
    async for health in drone.telemetry.health():
        if health.is_local_position_ok:
            print("-- Global position estimate OK")
            break

    # ======================创建并发任务==============================
    telemetry_task = asyncio.create_task(monitor_position(drone))
    status_task = asyncio.create_task(monitor_speed(drone))

    # ======================解锁=========================================
    print("-- Arming")
    await drone.action.arm()

    # ===================设置基准点=====================================
    print("-- Setting initial setpoint")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 90.0))


    # ===================开始飞行=========================================
    print("-- Starting offboard")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed \
                with error code: {error._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        return


    # 也可以直接使用起飞的api
    # await drone.action.set_takeoff_altitude(10.0)
    # await drone.action.takeoff()
    # await asyncio.sleep(10)


    print("-- Go 0m North, 0m East, -5m Down \
            within local coordinate system")
    await drone.offboard.set_position_ned(
        PositionNedYaw(0.0, 0.0, -1.3, 90.0))
    await asyncio.sleep(20)

    await drone.offboard.set_position_ned(
        PositionNedYaw(1, 0, -1.3, 90.0))
    await asyncio.sleep(10)

    await drone.offboard.set_position_ned(
        PositionNedYaw(1, 1, -1.3, 90.0))
    await asyncio.sleep(10)

    await drone.offboard.set_position_ned(
        PositionNedYaw(1, 1, -1.3 ,90.0))
    await asyncio.sleep(10)


   # ==============================着陆========================================
    print("-- Landing")
    await drone.action.land()

    async for state in drone.telemetry.landed_state():
        if state == LandedState.ON_GROUND:
            break

    # ==============================停止offboard模式========================================
    print("-- Stopping offboard")
    try:
        await drone.offboard.stop()
    except Exception as error:
        print(f"Stopping offboard mode failed with error: {error}")
    # ==============================解锁========================================
    print("-- Disarming")
    await drone.action.disarm()

    # ==========================清理并发任务==================================
    telemetry_task.cancel()
    status_task.cancel()
    await asyncio.gather(telemetry_task, status_task, return_exceptions=True)

# 并发任务一
async def monitor_position(drone):
    last_print_time = 0
    PRINT_INTERVAL = 0.5  # 每0.5秒打印一次
    async for position in drone.telemetry.position():
        current_time = asyncio.get_event_loop().time()
        if current_time - last_print_time >= PRINT_INTERVAL:
            print(f"Position: {position.latitude_deg:.6f}, {position.longitude_deg:.6f}")
            last_print_time = current_time


# 并发任务二
async def monitor_velocity(drone):
    last_print_time = 0
    PRINT_INTERVAL = 0.5  # 每0.5秒打印一次
    async for velocity in drone.telemetry.velocity_ned():
        current_time = asyncio.get_event_loop().time()
        if current_time - last_print_time >= PRINT_INTERVAL:
            horizontal_speed = (velocity.north_m_s**2 + velocity.east_m_s**2)**0.5
            print(f"速度 - 北:{velocity.north_m_s:.2f} 东:{velocity.east_m_s:.2f} 下:{velocity.down_m_s:.2f} m/s")
            print(f"速度: {horizontal_speed:.2f} m/s")
            last_print_time = current_time

if __name__ == "__main__":
    # Run the asyncio loop
    asyncio.run(run())