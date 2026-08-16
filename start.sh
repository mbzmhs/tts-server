#!/usr/bin/env bash
# TTS 本地服务启动脚本（Linux/macOS；Windows 用 start.bat）
# 自动检测 GPU/CPU，自动定位 Python 运行时
# 用法:  ./start.sh                 启动服务（默认 127.0.0.1:9880）
#        ./start.sh -p 9881         指定端口
#        ./start.sh --device cpu    强制 CPU
set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo " TTS Local Server (GPT-SoVITS V4)"
echo "============================================================"

# 1) engine 必须存在（按显卡型号自行下载解压/clone 到 engine/）
if [ ! -d "engine/GPT_SoVITS" ]; then
  echo "[TTS] ERROR: engine/GPT_SoVITS not found."
  echo "       请先按显卡型号下载 GPT-SoVITS 引擎放到本目录的 engine/:"
  echo "         - Windows: 官方整合包整体解压为 engine/（含 engine/runtime/python.exe）"
  echo "         - Linux  : git clone https://github.com/RVC-Boss/GPT-SoVITS engine"
  echo "         RTX 50 系显卡下载 ...-nvidia50 版本（CUDA 12.8），其余用标准版（CUDA 12.4）。"
  echo "       详见 README.md 的「安装引擎」一节。"
  exit 1
fi

# 2) 定位带 numpy/torch/fastapi 的 Python（优先引擎自带 runtime，其次 conda 环境，最后系统 python）
has_deps() {
  [ -x "$1" ] && "$1" -c "import numpy, torch, fastapi" >/dev/null 2>&1
}

PY=""
# 引擎自带运行时（整合包 / 内置 runtime）
if [ -f "engine/runtime/python" ] && has_deps "engine/runtime/python"; then
  PY="engine/runtime/python"
fi
# conda 环境（gptsovits 或当前激活环境）
if [ -z "$PY" ] && [ -n "${CONDA_PREFIX:-}" ] && has_deps "$CONDA_PREFIX/bin/python"; then
  PY="$CONDA_PREFIX/bin/python"
fi
if [ -z "$PY" ]; then
  for env in "$HOME/miniconda3/envs/gptsovits" "$HOME/anaconda3/envs/gptsovits" \
             "/opt/conda/envs/gptsovits"; do
    if has_deps "$env/bin/python"; then PY="$env/bin/python"; break; fi
  done
fi
# 系统 python3 / python
if [ -z "$PY" ] && has_deps "python3"; then PY="python3"; fi
if [ -z "$PY" ] && has_deps "python"; then PY="python"; fi

if [ -z "$PY" ]; then
  echo "[TTS] 未找到带 numpy/torch/fastapi 的 Python。"
  echo "       请安装依赖（conda create -n gptsovits python=3.9 ...）或激活环境后重试。"
  exit 1
fi

# 3) 某些 torch 版本需要 npp 库路径
NPP_LIB=$("$PY" -c "import site, os; print(os.path.join(site.getsitepackages()[0], 'nvidia', 'npp', 'lib'))" 2>/dev/null || true)
if [ -n "$NPP_LIB" ] && [ -d "$NPP_LIB" ]; then
  export LD_LIBRARY_PATH="$NPP_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "[TTS] LD_LIBRARY_PATH += $NPP_LIB"
fi

echo "[TTS] using python: $PY"
exec "$PY" tts_server.py "$@"
