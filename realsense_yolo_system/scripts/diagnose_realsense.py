#!/usr/bin/env python3
"""
RealSense 安装诊断脚本
检查 pyrealsense2 安装状态并提供修复建议
"""

import sys
import subprocess
import importlib.util

def check_python_version():
    """检查Python版本"""
    print(f"Python版本: {sys.version}")
    if sys.version_info < (3, 6):
        print("❌ Python版本过低，需要3.6+")
        return False
    print("✅ Python版本符合要求")
    return True

def check_pyrealsense2_installation():
    """检查pyrealsense2安装"""
    print("\n=== 检查 pyrealsense2 安装 ===")
    
    try:
        import pyrealsense2 as rs
        print("✅ pyrealsense2 已安装")
        
        # 检查版本
        try:
            version = rs.__version__
            print(f"✅ 版本: {version}")
        except:
            print("⚠️  无法获取版本信息")
        
        # 检查核心功能
        try:
            pipeline = rs.pipeline()
            print("✅ pipeline 对象可用")
            
            config = rs.config()
            print("✅ config 对象可用")
            
            context = rs.context()
            devices = context.query_devices()
            print(f"✅ 检测到 {len(devices)} 个RealSense设备")
            
            return True
            
        except Exception as e:
            print(f"❌ pyrealsense2 功能测试失败: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ pyrealsense2 导入失败: {e}")
        return False

def check_system_info():
    """检查系统信息"""
    print("\n=== 系统信息 ===")
    
    import platform
    print(f"操作系统: {platform.system()}")
    print(f"架构: {platform.machine()}")
    print(f"平台: {platform.platform()}")
    
    # 检查是否为ARM架构
    if platform.machine() in ['aarch64', 'armv7l', 'arm64']:
        print("⚠️  检测到ARM架构，可能需要从源码编译")
        return 'arm'
    
    return 'x86'

def check_librealsense():
    """检查librealsense库"""
    print("\n=== 检查 librealsense 库 ===")
    
    try:
        result = subprocess.run(['rs-enumerate-devices'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ librealsense 命令行工具可用")
            print("设备列表:")
            print(result.stdout)
            return True
        else:
            print("❌ librealsense 命令行工具不可用")
            return False
    except FileNotFoundError:
        print("❌ 未找到 rs-enumerate-devices 命令")
        return False
    except subprocess.TimeoutExpired:
        print("❌ 命令执行超时")
        return False
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def provide_solutions(arch):
    """提供解决方案"""
    print("\n=== 解决方案 ===")
    
    if arch == 'arm':
        print("ARM架构解决方案:")
        print("1. 从源码编译安装 (推荐)")
        print("   # 下载源码")
        print("   git clone https://github.com/IntelRealSense/librealsense.git")
        print("   cd librealsense")
        print("   git checkout v2.54.1")
        print("")
        print("   # 安装依赖")
        print("   sudo apt update")
        print("   sudo apt install -y build-essential cmake pkg-config")
        print("   sudo apt install -y libusb-1.0-0-dev libgtk-3-dev")
        print("")
        print("   # 编译安装")
        print("   mkdir build && cd build")
        print("   cmake .. -DBUILD_PYTHON_BINDINGS=true -DPYTHON_EXECUTABLE=/usr/bin/python3")
        print("   make -j$(nproc)")
        print("   sudo make install")
        print("")
        print("2. 使用预编译包 (如果可用)")
        print("   pip install pyrealsense2")
        print("")
        
    else:
        print("x86架构解决方案:")
        print("1. 重新安装 pyrealsense2")
        print("   pip uninstall pyrealsense2")
        print("   pip install pyrealsense2")
        print("")
        print("2. 尝试特定版本")
        print("   pip install pyrealsense2==2.54.1")
        print("")
        print("3. 从源码安装")
        print("   按照ARM架构的步骤执行")
        print("")
    
    print("通用解决方案:")
    print("1. 检查USB权限")
    print("   sudo usermod -a -G dialout $USER")
    print("   sudo chmod 666 /dev/bus/usb/*/*")
    print("")
    print("2. 安装udev规则")
    print("   wget https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules")
    print("   sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/")
    print("   sudo udevadm control --reload-rules")
    print("   sudo udevadm trigger")
    print("")
    print("3. 重启系统")
    print("   sudo reboot")

def main():
    """主函数"""
    print("RealSense 安装诊断工具")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        return
    
    # 检查系统信息
    arch = check_system_info()
    
    # 检查pyrealsense2
    pyrealsense2_ok = check_pyrealsense2_installation()
    
    # 检查librealsense
    librealsense_ok = check_librealsense()
    
    # 提供解决方案
    if not pyrealsense2_ok or not librealsense_ok:
        provide_solutions(arch)
    else:
        print("\n✅ 所有检查通过！RealSense环境正常")

if __name__ == "__main__":
    main()