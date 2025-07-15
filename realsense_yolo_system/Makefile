# RealSense D435i + YOLO 系统 Makefile
# 简化构建、测试和部署流程

# 项目配置
PROJECT_NAME := realsense-yolo-system
VERSION := 1.0.0
DOCKER_IMAGE := $(PROJECT_NAME):$(VERSION)
DOCKER_REGISTRY := your-registry.com
PYTHON_VERSION := 3.8

# 颜色定义
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# 默认目标
.DEFAULT_GOAL := help

# 帮助信息
.PHONY: help
help: ## 显示帮助信息
	@echo "$(BLUE)RealSense D435i + YOLO 系统 - 构建工具$(NC)"
	@echo ""
	@echo "$(GREEN)可用目标:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# 环境设置
.PHONY: setup
setup: ## 设置开发环境
	@echo "$(GREEN)设置开发环境...$(NC)"
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -e ".[dev,all]"
	@echo "$(GREEN)开发环境设置完成$(NC)"

.PHONY: setup-jetson
setup-jetson: ## 在Jetson设备上设置环境
	@echo "$(GREEN)在Jetson设备上设置环境...$(NC)"
	chmod +x scripts/install_jetson.sh
	./scripts/install_jetson.sh
	@echo "$(GREEN)Jetson环境设置完成$(NC)"

# 依赖管理
.PHONY: install
install: ## 安装依赖
	@echo "$(GREEN)安装依赖...$(NC)"
	pip install -r requirements.txt

.PHONY: install-dev
install-dev: ## 安装开发依赖
	@echo "$(GREEN)安装开发依赖...$(NC)"
	pip install -e ".[dev]"

.PHONY: update-deps
update-deps: ## 更新依赖
	@echo "$(GREEN)更新依赖...$(NC)"
	pip install --upgrade -r requirements.txt

# 代码质量
.PHONY: lint
lint: ## 代码风格检查
	@echo "$(GREEN)运行代码风格检查...$(NC)"
	flake8 src/ config/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ config/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

.PHONY: format
format: ## 格式化代码
	@echo "$(GREEN)格式化代码...$(NC)"
	black src/ config/ main.py setup.py
	isort src/ config/ main.py setup.py

.PHONY: type-check
type-check: ## 类型检查
	@echo "$(GREEN)运行类型检查...$(NC)"
	mypy src/ --ignore-missing-imports

# 测试
.PHONY: test
test: ## 运行测试
	@echo "$(GREEN)运行测试...$(NC)"
	python -m pytest tests/ -v

.PHONY: test-coverage
test-coverage: ## 运行测试并生成覆盖率报告
	@echo "$(GREEN)运行测试并生成覆盖率报告...$(NC)"
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

.PHONY: test-benchmark
test-benchmark: ## 运行性能基准测试
	@echo "$(GREEN)运行性能基准测试...$(NC)"
	python main.py --benchmark --benchmark-duration 60

# 运行
.PHONY: run
run: ## 运行系统
	@echo "$(GREEN)启动RealSense YOLO系统...$(NC)"
	python main.py

.PHONY: run-dev
run-dev: ## 开发模式运行
	@echo "$(GREEN)开发模式启动系统...$(NC)"
	python main.py --verbose

.PHONY: run-demo
run-demo: ## 演示模式运行
	@echo "$(GREEN)演示模式启动系统...$(NC)"
	python main.py --save-results --output-dir demo_output

# 构建
.PHONY: build
build: ## 构建Python包
	@echo "$(GREEN)构建Python包...$(NC)"
	python setup.py build

.PHONY: build-wheel
build-wheel: ## 构建wheel包
	@echo "$(GREEN)构建wheel包...$(NC)"
	python setup.py bdist_wheel

.PHONY: build-sdist
build-sdist: ## 构建源码包
	@echo "$(GREEN)构建源码包...$(NC)"
	python setup.py sdist

.PHONY: build-all
build-all: build-wheel build-sdist ## 构建所有包

# Docker
.PHONY: docker-build
docker-build: ## 构建Docker镜像
	@echo "$(GREEN)构建Docker镜像...$(NC)"
	docker build -t $(DOCKER_IMAGE) .

.PHONY: docker-build-dev
docker-build-dev: ## 构建开发Docker镜像
	@echo "$(GREEN)构建开发Docker镜像...$(NC)"
	docker build -t $(DOCKER_IMAGE)-dev --target development .

.PHONY: docker-run
docker-run: ## 运行Docker容器
	@echo "$(GREEN)运行Docker容器...$(NC)"
	docker run --rm -it --privileged \
		-v /dev/bus/usb:/dev/bus/usb \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY=$$DISPLAY \
		--name $(PROJECT_NAME) \
		$(DOCKER_IMAGE)

.PHONY: docker-compose-up
docker-compose-up: ## 启动Docker Compose服务
	@echo "$(GREEN)启动Docker Compose服务...$(NC)"
	docker-compose up -d

.PHONY: docker-compose-down
docker-compose-down: ## 停止Docker Compose服务
	@echo "$(GREEN)停止Docker Compose服务...$(NC)"
	docker-compose down

.PHONY: docker-compose-logs
docker-compose-logs: ## 查看Docker Compose日志
	@echo "$(GREEN)查看Docker Compose日志...$(NC)"
	docker-compose logs -f

# 部署
.PHONY: deploy
deploy: ## 部署到生产环境
	@echo "$(GREEN)部署到生产环境...$(NC)"
	# 这里添加具体的部署命令
	@echo "$(YELLOW)部署功能待实现$(NC)"

.PHONY: deploy-jetson
deploy-jetson: ## 部署到Jetson设备
	@echo "$(GREEN)部署到Jetson设备...$(NC)"
	scp -r . jetson@jetson-device:~/realsense_yolo_system/
	ssh jetson@jetson-device "cd ~/realsense_yolo_system && make install"

# 清理
.PHONY: clean
clean: ## 清理构建文件
	@echo "$(GREEN)清理构建文件...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

.PHONY: clean-all
clean-all: clean ## 清理所有生成文件
	@echo "$(GREEN)清理所有生成文件...$(NC)"
	rm -rf output/
	rm -rf logs/
	rm -rf debug/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf venv/

# 文档
.PHONY: docs
docs: ## 生成文档
	@echo "$(GREEN)生成文档...$(NC)"
	# 这里可以添加Sphinx或其他文档生成工具
	@echo "$(YELLOW)文档生成功能待实现$(NC)"

# 发布
.PHONY: release
release: clean build-all ## 发布新版本
	@echo "$(GREEN)发布新版本...$(NC)"
	twine upload dist/*

# 系统信息
.PHONY: info
info: ## 显示系统信息
	@echo "$(BLUE)系统信息:$(NC)"
	@echo "  项目名称: $(PROJECT_NAME)"
	@echo "  版本号: $(VERSION)"
	@echo "  Python版本: $(PYTHON_VERSION)"
	@echo "  Docker镜像: $(DOCKER_IMAGE)"
	@echo ""
	@echo "$(BLUE)环境信息:$(NC)"
	@python --version
	@echo -n "  GPU支持: "
	@python -c "import torch; print('是' if torch.cuda.is_available() else '否')" 2>/dev/null || echo "未知"
	@echo -n "  RealSense SDK: "
	@python -c "import pyrealsense2; print('已安装')" 2>/dev/null || echo "未安装"

# 安全检查
.PHONY: security
security: ## 安全检查
	@echo "$(GREEN)运行安全检查...$(NC)"
	safety check
	bandit -r src/

# 性能分析
.PHONY: profile
profile: ## 性能分析
	@echo "$(GREEN)运行性能分析...$(NC)"
	python -m cProfile -o profile.stats main.py --benchmark --benchmark-duration 30
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# 检查设备
.PHONY: check-devices
check-devices: ## 检查RealSense设备
	@echo "$(GREEN)检查RealSense设备...$(NC)"
	python -c "from src.camera_manager import discover_realsense_devices; devices = discover_realsense_devices(); print(f'发现 {len(devices)} 个设备'); [print(f'  - {d[\"name\"]} ({d[\"serial_number\"]})') for d in devices]"

# 快速检查
.PHONY: check
check: lint type-check test ## 快速检查（格式+类型+测试）

# 全面检查
.PHONY: check-all
check-all: check security test-coverage ## 全面检查

# CI/CD
.PHONY: ci
ci: check-all build-all ## CI流水线

# 监控
.PHONY: monitor
monitor: ## 监控系统运行
	@echo "$(GREEN)监控系统运行状态...$(NC)"
	watch -n 5 'ps aux | grep python | grep -v grep'

# 日志查看
.PHONY: logs
logs: ## 查看系统日志
	@echo "$(GREEN)查看系统日志...$(NC)"
	tail -f logs/system.log

# 交互式设置
.PHONY: interactive-setup
interactive-setup: ## 交互式环境设置
	@echo "$(GREEN)交互式环境设置...$(NC)"
	@echo "请选择安装类型:"
	@echo "  1) 标准安装"
	@echo "  2) 开发环境"
	@echo "  3) Jetson设备"
	@read -p "选择 [1-3]: " choice; \
	case $$choice in \
		1) make install ;; \
		2) make setup ;; \
		3) make setup-jetson ;; \
		*) echo "无效选择" ;; \
	esac

# 卸载
.PHONY: uninstall
uninstall: ## 卸载系统
	@echo "$(RED)卸载系统...$(NC)"
	pip uninstall -y $(PROJECT_NAME)
	rm -rf ~/realsense_yolo_env

# 版本检查
.PHONY: version
version: ## 显示版本信息
	@echo "$(BLUE)版本信息:$(NC)"
	@echo "  当前版本: $(VERSION)"
	@git describe --tags --dirty --always 2>/dev/null || echo "  Git版本: 未知"

# 备份
.PHONY: backup
backup: ## 备份重要文件
	@echo "$(GREEN)备份重要文件...$(NC)"
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz \
		src/ config/ main.py setup.py requirements.txt \
		--exclude-vcs --exclude='*.pyc' --exclude='__pycache__'

# 更新
.PHONY: update
update: ## 更新系统
	@echo "$(GREEN)更新系统...$(NC)"
	git pull origin main
	make install
	@echo "$(GREEN)更新完成$(NC)"

# 重置
.PHONY: reset
reset: clean-all ## 重置到初始状态
	@echo "$(RED)重置到初始状态...$(NC)"
	git checkout -- .
	git clean -fd

# 状态检查
.PHONY: status
status: ## 检查系统状态
	@echo "$(BLUE)系统状态:$(NC)"
	@echo -n "  项目目录: "; pwd
	@echo -n "  Git状态: "; git status --porcelain | wc -l | xargs -I {} echo "{} 个修改"
	@echo -n "  虚拟环境: "; [ -d "venv" ] && echo "存在" || echo "不存在"
	@echo -n "  依赖状态: "; pip check > /dev/null 2>&1 && echo "正常" || echo "有冲突"