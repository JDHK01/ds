"""
测试套件入口
运行所有单元测试
"""

import pytest
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_tests():
    """运行所有测试"""
    print("RealSense D435i + YOLO 系统测试套件")
    print("=" * 50)
    
    # 测试参数
    test_args = [
        str(Path(__file__).parent),  # 测试目录
        "-v",  # 详细输出
        "--tb=short",  # 简短回溯
        "--color=yes",  # 彩色输出
        "--durations=10",  # 显示最慢的10个测试
        "--strict-markers",  # 严格标记模式
        "--strict-config",  # 严格配置模式
    ]
    
    # 运行测试
    exit_code = pytest.main(test_args)
    
    return exit_code

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)