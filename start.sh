#!/bin/bash
# MSG平台 一键启动 (Linux/Mac)
set -e

echo "============================================"
echo "  MSG-Embedding 5G信道仿真平台"
echo "============================================"
echo

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MSG_REPO_ROOT="$SCRIPT_DIR"
echo "项目目录: $MSG_REPO_ROOT"

# 检测 Python
if ! command -v python3 &>/dev/null; then
    echo "[错误] 未检测到 python3"
    exit 1
fi
python3 --version

# 创建虚拟环境
if [ ! -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    echo
    echo "[1/3] 创建虚拟环境..."
    python3 -m venv "$SCRIPT_DIR/.venv"

    echo "[2/3] 安装依赖..."
    "$SCRIPT_DIR/.venv/bin/pip" install -e "$SCRIPT_DIR[platform]" --quiet
    echo "依赖安装完成"
else
    echo "虚拟环境已存在，跳过安装"
fi

# 初始化数据库
echo
echo "[3/3] 初始化数据库..."
"$SCRIPT_DIR/.venv/bin/python" -c "
import sys; sys.path.insert(0,'src')
from platform.backend.db import init_db; init_db()
" 2>/dev/null || true
echo "数据库就绪"

echo
echo "============================================"
echo "  启动中... 浏览器访问 http://localhost:8000"
echo "  按 Ctrl+C 停止"
echo "============================================"
echo

"$SCRIPT_DIR/.venv/bin/python" -m uvicorn platform.backend.main:app --host 0.0.0.0 --port 8000
