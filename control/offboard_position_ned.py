#!/usr/bin/env python3

"""
Caveat when attempting to run the examples in non-gps environments:

`drone.offboard.stop()` will return a `COMMAND_DENIED` result because it
requires a mode switch to HOLD, something that is currently not supported in a
non-gps environment.
"""
'''
        位置 NED 坐标系
        打印NED坐标系下的参数
    async for pvn in drone.telemetry.position_velocity_ned():
        # NED 位置
        north = pvn.position.north_m  # X (NED)
        east = pvn.position.east_m    # Y (NED)  
        down = pvn.position.down_m    # Z (NED)
'''



import asyncio

from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw)
from mavsdk.telemetry import LandedState


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

    print("-- Arming")
    await drone.action.arm()

    print("-- Setting initial setpoint")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 90.0))

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
'''
    print("-- Go 5m North, 0m East, -5m Down \
            within local coordinate system, turn to face East")
    await drone.offboard.set_position_ned(
            PositionNedYaw(5.0, 0.0, -5.0, 90.0))
    await asyncio.sleep(10)

    print("-- Go 5m North, 10m East, -5m Down \
            within local coordinate system")
    await drone.offboard.set_position_ned(
            PositionNedYaw(5.0, 10.0, -5.0, 90.0))
    await asyncio.sleep(15)

    print("-- Go 0m North, 10m East, 0m Down \
            within local coordinate system, turn to face South")
    await drone.offboard.set_position_ned(
            PositionNedYaw(0.0, 10.0, 0.0, 180.0))
    await asyncio.sleep(10)
'''



if __name__ == "__main__":
    
    # Run the asyncio loop
    asyncio.run(run())
