#!/bin/bash

# RealSense 自动修复脚本
# 解决 pyrealsense2 安装问题

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查系统架构
check_architecture() {
    ARCH=$(uname -m)
    log_info "系统架构: $ARCH"
    
    if [[ "$ARCH" == "aarch64" || "$ARCH" == "armv7l" || "$ARCH" == "arm64" ]]; then
        echo "arm"
    else
        echo "x86"
    fi
}

# 检查Python版本
check_python() {
    log_step "检查Python版本..."
    
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    log_info "Python版本: $PYTHON_VERSION"
    
    # 检查是否为3.6+
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 6) else 1)"; then
        log_info "✅ Python版本符合要求"
        return 0
    else
        log_error "❌ Python版本过低，需要3.6+"
        return 1
    fi
}

# 卸载旧版本
uninstall_old_version() {
    log_step "卸载旧版本..."
    
    pip3 uninstall -y pyrealsense2 2>/dev/null || true
    
    log_info "✅ 旧版本已卸载"
}

# 安装系统依赖
install_system_deps() {
    log_step "安装系统依赖..."
    
    sudo apt update
    sudo apt install -y \
        build-essential \
        cmake \
        pkg-config \
        libusb-1.0-0-dev \
        libgtk-3-dev \
        libglfw3-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        python3-dev \
        python3-pip
    
    log_info "✅ 系统依赖安装完成"
}

# 方法1：尝试pip安装
try_pip_install() {
    log_step "尝试pip安装..."
    
    if pip3 install pyrealsense2; then
        log_info "✅ pip安装成功"
        return 0
    else
        log_warn "❌ pip安装失败"
        return 1
    fi
}

# 方法2：从源码编译
compile_from_source() {
    log_step "从源码编译安装..."
    
    # 创建临时目录
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    # 下载源码
    log_info "下载librealsense源码..."
    git clone https://github.com/IntelRealSense/librealsense.git
    cd librealsense
    
    # 检出稳定版本
    git checkout v2.54.1
    
    # 安装udev规则
    log_info "安装udev规则..."
    sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    
    # 创建构建目录
    mkdir build
    cd build
    
    # 配置cmake
    log_info "配置cmake..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_EXAMPLES=false \
        -DBUILD_GRAPHICAL_EXAMPLES=false \
        -DBUILD_PYTHON_BINDINGS=true \
        -DPYTHON_EXECUTABLE=$(which python3) \
        -DBUILD_WITH_CUDA=OFF
    
    # 编译
    log_info "编译中（这可能需要一些时间）..."
    make -j$(nproc)
    
    # 安装
    log_info "安装..."
    sudo make install
    
    # 更新链接库
    sudo ldconfig
    
    # 清理
    cd /
    rm -rf "$TEMP_DIR"
    
    log_info "✅ 源码编译安装完成"
}

# 设置权限
setup_permissions() {
    log_step "设置USB权限..."
    
    # 添加用户到dialout组
    sudo usermod -a -G dialout $USER
    
    # 设置USB权限
    sudo chmod 666 /dev/bus/usb/*/* 2>/dev/null || true
    
    log_info "✅ 权限设置完成"
}

# 验证安装
verify_installation() {
    log_step "验证安装..."
    
    # 测试导入
    if python3 -c "import pyrealsense2 as rs; print('✅ pyrealsense2导入成功')"; then
        log_info "✅ 导入测试通过"
    else
        log_error "❌ 导入测试失败"
        return 1
    fi
    
    # 测试核心功能
    if python3 -c "import pyrealsense2 as rs; rs.pipeline(); print('✅ pipeline对象创建成功')"; then
        log_info "✅ 核心功能测试通过"
    else
        log_error "❌ 核心功能测试失败"
        return 1
    fi
    
    # 检查设备
    log_info "检查设备连接..."
    python3 -c "
import pyrealsense2 as rs
ctx = rs.context()
devices = ctx.query_devices()
print(f'检测到 {len(devices)} 个设备')
for i, device in enumerate(devices):
    name = device.get_info(rs.camera_info.name)
    serial = device.get_info(rs.camera_info.serial_number)
    print(f'  设备 {i+1}: {name} (序列号: {serial})')
"
    
    return 0
}

# 主函数
main() {
    log_info "RealSense 自动修复脚本启动..."
    
    # 检查Python
    if ! check_python; then
        log_error "Python版本不符合要求，请升级Python"
        exit 1
    fi
    
    # 检查架构
    ARCH=$(check_architecture)
    log_info "检测到架构: $ARCH"
    
    # 卸载旧版本
    uninstall_old_version
    
    # 安装系统依赖
    install_system_deps
    
    # 尝试不同的安装方法
    if [[ "$ARCH" == "x86" ]]; then
        # x86架构先尝试pip
        if try_pip_install; then
            log_info "使用pip安装成功"
        else
            log_warn "pip安装失败，尝试源码编译"
            compile_from_source
        fi
    else
        # ARM架构直接源码编译
        log_info "ARM架构，使用源码编译"
        compile_from_source
    fi
    
    # 设置权限
    setup_permissions
    
    # 验证安装
    if verify_installation; then
        log_info "🎉 修复完成！"
        log_info "请重新启动终端或运行: source ~/.bashrc"
        log_info "然后重新运行您的程序"
    else
        log_error "❌ 修复失败"
        exit 1
    fi
}

# 运行主函数
main "$@"