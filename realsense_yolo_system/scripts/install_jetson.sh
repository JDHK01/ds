#!/bin/bash

# RealSense D435i + YOLO 系统安装脚本
# 适用于 Jetson Orin Nano Ubuntu 20.04

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# 检查是否为root用户
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "请勿使用root用户运行此脚本"
        exit 1
    fi
}

# 检查系统信息
check_system() {
    log_info "检查系统信息..."
    
    # 检查操作系统
    if [[ ! -f /etc/os-release ]]; then
        log_error "无法检测操作系统"
        exit 1
    fi
    
    source /etc/os-release
    log_info "操作系统: $NAME $VERSION"
    
    # 检查架构
    ARCH=$(uname -m)
    log_info "系统架构: $ARCH"
    
    if [[ "$ARCH" != "aarch64" ]]; then
        log_warn "此脚本主要针对ARM64架构，当前架构可能不完全支持"
    fi
    
    # 检查CUDA
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \\([0-9]*\\.[0-9]*\\).*/\\1/')
        log_info "CUDA版本: $CUDA_VERSION"
    else
        log_warn "未检测到CUDA，将使用CPU模式"
    fi
}

# 更新系统
update_system() {
    log_info "更新系统包..."
    sudo apt update
    sudo apt upgrade -y
}

# 安装基础依赖
install_base_dependencies() {
    log_info "安装基础依赖..."
    
    sudo apt install -y \\
        build-essential \\
        cmake \\
        pkg-config \\
        libjpeg-dev \\
        libtiff5-dev \\
        libpng-dev \\
        libavcodec-dev \\
        libavformat-dev \\
        libswscale-dev \\
        libv4l-dev \\
        libxvidcore-dev \\
        libx264-dev \\
        libgtk-3-dev \\
        libatlas-base-dev \\
        gfortran \\
        python3-dev \\
        python3-pip \\
        python3-venv \\
        git \\
        wget \\
        curl \\
        unzip \\
        libssl-dev \\
        libffi-dev \\
        libhdf5-dev \\
        libhdf5-serial-dev \\
        libhdf5-103 \\
        libqtgui4 \\
        libqtwebkit4 \\
        libqt4-test \\
        python3-pyqt5 \\
        libusb-1.0-0-dev \\
        libglfw3-dev \\
        libgl1-mesa-dev \\
        libglu1-mesa-dev
}

# 安装RealSense SDK
install_realsense_sdk() {
    log_info "安装Intel RealSense SDK..."
    
    # 创建工作目录
    mkdir -p ~/realsense_build
    cd ~/realsense_build
    
    # 下载librealsense源码
    if [[ ! -d "librealsense" ]]; then
        log_info "下载librealsense源码..."
        git clone https://github.com/IntelRealSense/librealsense.git
    fi
    
    cd librealsense
    
    # 检出稳定版本
    git checkout v2.54.1
    
    # 安装依赖
    sudo apt install -y libusb-1.0-0-dev pkg-config libgtk-3-dev
    
    # 安装udev规则
    sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    
    # 构建并安装
    mkdir -p build
    cd build
    
    log_info "配置构建..."
    cmake .. \\
        -DCMAKE_BUILD_TYPE=Release \\
        -DBUILD_EXAMPLES=true \\
        -DBUILD_GRAPHICAL_EXAMPLES=true \\
        -DBUILD_PYTHON_BINDINGS=bool:true \\
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \\
        -DBUILD_WITH_CUDA=ON
    
    log_info "编译RealSense SDK（这可能需要较长时间）..."
    make -j$(nproc)
    
    log_info "安装RealSense SDK..."
    sudo make install
    
    # 更新库路径
    echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    
    log_info "RealSense SDK安装完成"
}

# 安装Python依赖
install_python_dependencies() {
    log_info "创建Python虚拟环境..."
    
    # 创建虚拟环境
    python3 -m venv ~/realsense_yolo_env
    source ~/realsense_yolo_env/bin/activate
    
    # 升级pip
    pip install --upgrade pip
    
    log_info "安装Python依赖包..."
    
    # 安装基础包
    pip install numpy scipy matplotlib
    
    # 安装OpenCV（ARM64预编译版本）
    pip install opencv-python
    
    # 安装PyTorch（ARM64版本）
    log_info "安装PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # 安装其他依赖
    pip install \\
        ultralytics \\
        pillow \\
        scikit-image \\
        psutil \\
        tqdm \\
        colorama \\
        pyyaml \\
        loguru \\
        memory-profiler
    
    # 尝试安装Open3D
    log_info "尝试安装Open3D..."
    pip install open3d || log_warn "Open3D安装失败，3D可视化功能将不可用"
    
    log_info "Python依赖安装完成"
}

# 验证安装
verify_installation() {
    log_info "验证安装..."
    
    # 激活虚拟环境
    source ~/realsense_yolo_env/bin/activate
    
    # 检查RealSense
    if command -v realsense-viewer &> /dev/null; then
        log_info "✓ RealSense SDK安装成功"
    else
        log_error "✗ RealSense SDK安装失败"
        return 1
    fi
    
    # 检查Python包
    python3 -c "import pyrealsense2; print('✓ pyrealsense2 可用')" || log_error "✗ pyrealsense2 不可用"
    python3 -c "import cv2; print('✓ OpenCV 可用')" || log_error "✗ OpenCV 不可用"
    python3 -c "import torch; print('✓ PyTorch 可用')" || log_error "✗ PyTorch 不可用"
    python3 -c "import ultralytics; print('✓ Ultralytics 可用')" || log_error "✗ Ultralytics 不可用"
    
    log_info "安装验证完成"
}

# 创建启动脚本
create_launch_script() {
    log_info "创建启动脚本..."
    
    cat > ~/start_realsense_yolo.sh << 'EOF'
#!/bin/bash

# RealSense YOLO 系统启动脚本

# 激活虚拟环境
source ~/realsense_yolo_env/bin/activate

# 设置环境变量
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# 进入项目目录
cd ~/realsense_yolo_system

# 启动系统
python3 main.py "$@"
EOF

    chmod +x ~/start_realsense_yolo.sh
    
    log_info "启动脚本创建完成: ~/start_realsense_yolo.sh"
}

# 创建桌面快捷方式
create_desktop_shortcut() {
    log_info "创建桌面快捷方式..."
    
    cat > ~/Desktop/RealSense_YOLO.desktop << 'EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=RealSense YOLO System
Comment=Intel RealSense D435i + YOLO 3D Object Detection
Exec=/home/$USER/start_realsense_yolo.sh
Icon=camera-video
Terminal=true
Categories=Development;Science;
EOF

    chmod +x ~/Desktop/RealSense_YOLO.desktop
    
    log_info "桌面快捷方式创建完成"
}

# 主函数
main() {
    log_info "开始安装RealSense D435i + YOLO系统..."
    
    check_root
    check_system
    
    read -p "是否继续安装? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "安装已取消"
        exit 0
    fi
    
    update_system
    install_base_dependencies
    install_realsense_sdk
    install_python_dependencies
    verify_installation
    create_launch_script
    create_desktop_shortcut
    
    log_info "安装完成！"
    log_info "使用方法："
    log_info "  1. 连接RealSense D435i相机"
    log_info "  2. 运行: ~/start_realsense_yolo.sh"
    log_info "  3. 或双击桌面快捷方式"
    log_info ""
    log_info "注意事项："
    log_info "  - 首次运行会自动下载YOLO模型"
    log_info "  - 确保相机有足够的USB 3.0带宽"
    log_info "  - 建议使用专用的USB 3.0端口"
    log_info ""
    log_info "如需手动激活环境："
    log_info "  source ~/realsense_yolo_env/bin/activate"
}

# 运行主函数
main "$@"