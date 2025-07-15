#!/usr/bin/env python3
"""
RealSense D435i + YOLO 系统安装配置
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# 读取README文件
README_PATH = Path(__file__).parent / "README.md"
if README_PATH.exists():
    with open(README_PATH, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "Intel RealSense D435i + YOLO 实时3D目标检测与测距系统"

# 读取版本号
def get_version():
    version_file = Path(__file__).parent / "realsense_yolo_system" / "__init__.py"
    if version_file.exists():
        with open(version_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return "1.0.0"

# 基础依赖
install_requires = [
    "numpy>=1.21.0",
    "opencv-python>=4.8.0",
    "matplotlib>=3.5.0",
    "scikit-image>=0.19.0",
    "scipy>=1.9.0",
    "psutil>=5.9.0",
    "tqdm>=4.64.0",
    "colorama>=0.4.5",
    "pyyaml>=6.0",
    "loguru>=0.6.0",
    "Pillow>=9.0.0",
]

# 可选依赖
extras_require = {
    # 深度学习依赖
    'ai': [
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "ultralytics>=8.0.0",
    ],
    
    # 3D处理依赖
    '3d': [
        "open3d>=0.17.0",
    ],
    
    # RealSense依赖（需要从源码编译）
    'realsense': [
        # 注意：pyrealsense2需要在ARM64上从源码编译
        # "pyrealsense2>=2.54.1",
    ],
    
    # 开发依赖
    'dev': [
        "pytest>=7.0.0",
        "black>=22.0.0",
        "flake8>=5.0.0",
        "isort>=5.10.0",
        "memory-profiler>=0.60.0",
        "line-profiler>=4.0.0",
    ],
    
    # 可视化增强
    'vis': [
        "plotly>=5.0.0",
        "seaborn>=0.11.0",
    ],
    
    # Jupyter支持
    'jupyter': [
        "jupyter>=1.0.0",
        "ipywidgets>=8.0.0",
    ],
}

# 全部依赖
extras_require['all'] = []
for deps in extras_require.values():
    extras_require['all'].extend(deps)

# 移除重复
extras_require['all'] = list(set(extras_require['all']))

# 控制台脚本
console_scripts = [
    "realsense-yolo=realsense_yolo_system.main:main",
    "realsense-yolo-benchmark=realsense_yolo_system.main:main",
]

setup(
    name="realsense-yolo-system",
    version=get_version(),
    author="System Developer",
    author_email="developer@example.com",
    description="Intel RealSense D435i + YOLO 实时3D目标检测与测距系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/realsense-yolo-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": console_scripts,
    },
    include_package_data=True,
    package_data={
        "realsense_yolo_system": [
            "config/*.py",
            "models/*.pt",
            "examples/*.py",
            "docs/*.md",
        ],
    },
    data_files=[
        ("share/realsense-yolo-system/scripts", ["scripts/install_jetson.sh"]),
        ("share/realsense-yolo-system/examples", ["examples/basic_usage.py"]),
    ],
    zip_safe=False,
    keywords="realsense yolo 3d object detection depth camera computer vision",
    project_urls={
        "Documentation": "https://github.com/your-username/realsense-yolo-system/wiki",
        "Source": "https://github.com/your-username/realsense-yolo-system",
        "Tracker": "https://github.com/your-username/realsense-yolo-system/issues",
    },
)

# 安装后的提示信息
def print_installation_info():
    print("\\n" + "="*60)
    print("RealSense YOLO 系统安装完成!")
    print("="*60)
    print("\\n快速开始:")
    print("  1. 连接Intel RealSense D435i相机")
    print("  2. 运行: realsense-yolo")
    print("  3. 或使用Python: python -m realsense_yolo_system.main")
    print("\\n配置:")
    print("  - 配置文件位置: realsense_yolo_system/config/config.py")
    print("  - 模型自动下载到: models/")
    print("  - 输出保存到: output/")
    print("\\n注意事项:")
    print("  - 首次运行会自动下载YOLO模型")
    print("  - 确保相机连接到USB 3.0端口")
    print("  - 在ARM64平台上，pyrealsense2需要从源码编译")
    print("\\n获取帮助:")
    print("  realsense-yolo --help")
    print("\\n" + "="*60)

if __name__ == "__main__":
    print_installation_info()