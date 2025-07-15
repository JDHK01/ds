# RealSense D435i + YOLO 系统 Docker 镜像
# 基于 Ubuntu 20.04 with CUDA 支持

# 基础镜像 - 选择合适的CUDA版本
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 维护者信息
LABEL maintainer="developer@example.com"
LABEL description="Intel RealSense D435i + YOLO 3D Object Detection System"
LABEL version="1.0.0"

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1
ENV PYTHON_VERSION=3.8

# 设置工作目录
WORKDIR /app

# 更新系统并安装基础依赖
RUN apt-get update && apt-get install -y \
    # 系统工具
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    wget \
    curl \
    unzip \
    git \
    cmake \
    build-essential \
    pkg-config \
    # Python相关
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    # 图像处理库
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    # 数学库
    libatlas-base-dev \
    gfortran \
    # USB支持
    libusb-1.0-0-dev \
    # OpenGL支持
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    # 清理
    && rm -rf /var/lib/apt/lists/*

# 创建符号链接
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# 升级pip
RUN python -m pip install --upgrade pip

# 安装Intel RealSense SDK
RUN mkdir -p /tmp/realsense && cd /tmp/realsense \
    && git clone https://github.com/IntelRealSense/librealsense.git \
    && cd librealsense \
    && git checkout v2.54.1 \
    && mkdir build && cd build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_EXAMPLES=false \
        -DBUILD_GRAPHICAL_EXAMPLES=false \
        -DBUILD_PYTHON_BINDINGS=true \
        -DPYTHON_EXECUTABLE=/usr/bin/python3.8 \
        -DBUILD_WITH_CUDA=ON \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd / && rm -rf /tmp/realsense

# 设置udev规则（用于USB设备访问）
RUN echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", MODE="0666"' > /etc/udev/rules.d/99-realsense-libusb.rules
RUN echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b07", MODE="0666"' >> /etc/udev/rules.d/99-realsense-libusb.rules

# 复制requirements文件
COPY requirements.txt /app/

# 安装Python依赖
RUN pip install -r requirements.txt

# 安装PyTorch（CUDA版本）
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他特定依赖
RUN pip install \
    ultralytics \
    open3d \
    pyrealsense2

# 复制项目文件
COPY . /app/

# 设置Python路径
ENV PYTHONPATH=/app:$PYTHONPATH

# 创建必要的目录
RUN mkdir -p /app/models /app/output /app/logs /app/debug

# 设置权限
RUN chmod +x /app/scripts/*.sh

# 创建非root用户
RUN useradd -m -s /bin/bash realsense && \
    usermod -aG video realsense && \
    chown -R realsense:realsense /app

# 切换到非root用户
USER realsense

# 暴露端口（如果需要Web界面）
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import pyrealsense2; print('RealSense SDK OK')" || exit 1

# 设置环境变量
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# 默认命令
CMD ["python", "main.py"]

# 构建参数
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# 添加标签
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="realsense-yolo-system" \
      org.label-schema.description="Intel RealSense D435i + YOLO 3D Object Detection System" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/your-username/realsense-yolo-system" \
      org.label-schema.vendor="Your Organization" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"