#!/usr/bin/env bash
# TTS 本地服务启动脚本（Linux/macOS；Windows 用 start.bat）
# 自动检测 GPU/CPU，自动定位 Python 运行时
# 缺失 engine/ 时交互引导 git clone 安装（可选用 HTTP 代理）
# 用法:  ./start.sh                 启动服务（默认 127.0.0.1:9880）
#        ./start.sh -p 9881         指定端口
#        ./start.sh --device cpu    强制 CPU
set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo " TTS Local Server (GPT-SoVITS V4)"
echo "============================================================"

# 1) engine 必须存在；缺失时交互引导克隆安装
if [ ! -d "engine/GPT_SoVITS" ]; then
  echo "[TTS] 未找到引擎: engine/GPT_SoVITS"
  echo "       Linux 通过 git clone 官方仓库安装（约 1~2GB）。"
  echo "       （Windows 请使用 start.bat，会按显卡型号下载整合包；RTX 50 系用"
  echo "         -nvidia50/CUDA 12.8 版，其余用标准版/CUDA 12.4，见 README.md）"
  printf "是否现在克隆引擎？[y/N] "
  read -r ans
  case "$ans" in
    y|Y|yes|YES)
      printf "HTTP 代理（如 http://host:port，直接回车则直连）: "
      read -r proxy
      if [ -n "$proxy" ]; then
        echo "[TTS] 使用代理 $proxy 克隆 ..."
        git -c http.proxy="$proxy" \
          -c https.proxy="$proxy" \
          clone https://github.com/RVC-Boss/GPT-SoVITS engine
      else
        echo "[TTS] 直连克隆 ..."
        git clone https://github.com/RVC-Boss/GPT-SoVITS engine
      fi
      if [ ! -d "engine/GPT_SoVITS" ]; then
        echo "[TTS] 克隆失败。请检查网络后重试，或手动 git clone 到 engine/。"
        exit 1
      fi
      echo "[TTS] 引擎已克隆到 engine/。"
      echo "       请安装 Python 依赖后再运行本脚本，例如："
      echo "         conda create -n gptsovits python=3.9 -y && conda activate gptsovits"
      echo "         pip install -r engine/requirements.txt"
      exit 0
      ;;
    *)
      echo "[TTS] 已取消。请手动 git clone https://github.com/RVC-Boss/GPT-SoVITS engine 后重试。"
      exit 1
      ;;
  esac
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
