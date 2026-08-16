# 本地 TTS 服务（GPT-SoVITS V4）

基于 GPT-SoVITS V4 的本地语音合成 HTTP 服务（只适用于单角色无并发）。支持中 / 日 / 英多种语言、多音色管理
（含自定义训练音色），GPU / CPU 自动选择，以及 FIFO 顺序合成队列（请求按提交顺序
逐条处理，合成在后台线程执行，服务永不因单次请求卡死）。每个音色可配置任意数量的
「情绪」（名称自定），通过参考音频迁移实现；比如某音色 xxx 含 8 档
（neutral / happy / sad / angry / surprised / afraid / shy / confused）。为
[desktop-pet-ui](https://github.com/mbzmhs/desktop-pet-ui) 等桌面应用提供音频接口。

## 目录结构

```
tts-server/
├── tts_server.py            # FastAPI 服务（核心）
├── start.bat                # Windows 启动脚本
├── start.sh                 # Linux/macOS 启动脚本
├── server_config.json       # 服务/引擎配置（端口、设备、底模路径，可编辑）
├── voices.json              # 音色注册表（模型路径 / 参考音频 / 参考文本）
├── GPT_weights_v4/          # 定制音色的 GPT 训练权重（*.ckpt）
├── SoVITS_weights_v4/       # 定制音色的 SoVITS 训练权重（*.pth）
├── ref/                     # 情绪参考音频（ref/<voice>/<emotion>.wav）
├── check/                   # 训练 / 对照音频输出
└── engine/                  # GPT-SoVITS 引擎（自行下载，见「安装引擎」）
```

模型权重、参考音频与配置文件统一放在项目根目录；`engine/` 只保留 GPT-SoVITS
引擎源码与预训练底模（BERT / HuBERT / s1v3 / s2Gv4），由引擎代码相对定位，无需改动。

> `engine/` 与所有模型权重**不入 git 库**，需按显卡型号自行下载，见下节。

## 安装引擎（按显卡下载）

本项目不包含任何模型文件。首次使用需先安装 GPT-SoVITS V4 引擎（含预训练底模），
解压 / clone 到项目根目录下的 `engine/` 即可；服务自动定位，**不修改引擎任何文件**。
引擎缺失时 `start.bat` / `start.sh` 会**交互式引导下载**（`start.bat` 会先问显卡
型号、下载源、是否走 HTTP 代理（可手动输入代理地址，或回车直连），然后自动下载并尝试解压
重命名为 `engine`）。

**Windows**（推荐官方整合包，自带 `engine\runtime\python.exe`，无独立 Python 也能跑）：

| 显卡 | 下载文件 |
|------|----------|
| 非 50 系（CUDA 12.4，如 20/30/40 系） | `GPT-SoVITS-v4-20250529.7z` |
| RTX 50 系（CUDA 12.8） | `GPT-SoVITS-v4-20250529-nvidia50.7z` |

下载地址（约 7GB，.7z 格式）：
- HuggingFace: `https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-v4-20250529.7z?download=true`
- 魔搭（国内满速）: `https://www.modelscope.cn/models/FlowerCry/gpt-sovits-7z-pacakges/resolve/master/GPT-SoVITS-v4-20250529.7z`

RTX 50 系把文件名换成 `GPT-SoVITS-v4-20250529-nvidia50.7z`（路径同上面两个站点）。
用 7-Zip 解压，把解压出的整个文件夹放入项目根目录并**重命名为 `engine`**，确保存在
`engine\runtime\python.exe` 与 `engine\GPT_SoVITS\` 两个路径。注意：目录名**必须为
`engine`**，否则启动脚本找不到引擎。

**Linux / macOS**：clone 源码到 `engine/` 并安装依赖：

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS engine
cd engine
pip install -r requirements.txt    # 或按官方文档用 conda 建环境
```

## 配置（server_config.json）

模型与服务相关配置放在项目根目录 `server_config.json`，用户可直接编辑：

| 字段 | 默认 | 说明 |
|------|------|------|
| `host` | `127.0.0.1` | 监听地址 |
| `port` | `9880` | 监听端口 |
| `device` | `auto` | `auto`（有 CUDA 用 GPU）/ `cuda` / `cpu` |
| `default_emotion` | `neutral` | 请求未指定情绪时的默认情绪名称 |
| `bert_base_path` | `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large` | 语义 BERT 底模 |
| `cnhuhbert_base_path` | `GPT_SoVITS/pretrained_models/chinese-hubert-base` | 语音特征 HuBERT 底模 |

命令行参数优先级高于配置文件（如 `start.bat -p 9881`、`--device cpu`）。
音色与情绪参考（模型路径 / 参考 wav / 参考文本）在 `voices.json` 中编辑。

### 情绪配置

每个音色的 `emotions` 是一个「情绪名 → 参考音频配置」的映射，数量与名称均不限
（无需固定 6 档）；情绪在启动时从 `voices.json` 载入并预计算参考特征。实际使用的
默认情绪按以下优先级解析：

```
请求显式指定 emotion > 音色 default_emotion（voices.json）> server_config 的
default_emotion > 该音色定义的第一个情绪
```

也可给单个音色配置专属默认情绪（缺省走全局配置），在 `voices.json` 该音色下加
`"default_emotion"` 字段即可，如 `"default_emotion": "happy"`。

## 启动

Windows（整合包放于 `engine/`，自带 `engine\runtime\python.exe`，无卡也能用）：

```
start.bat                  # 自动检测 GPU/CPU，监听 127.0.0.1:9880
start.bat -p 9881          # 指定端口
start.bat --device cpu     # 强制 CPU
```

Linux/macOS：

```
./start.sh [--device cpu] [-p 9881]
```

服务启动时自动检测 CUDA：有 GPU 用 fp16（快），无 GPU 回退 CPU（慢但可用）。

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/health` | 运行状态、设备、当前音色、队列积压（`queue_pending`） |
| GET  | `/voices` | 音色列表与情绪 |
| POST | `/voices/{id}/activate` | 切换到指定音色 |
| POST | `/tts` | 语音合成，返回音频 |

### 合成请求

```json
POST /tts
{
  "text": "你好，今天过得怎么样？",
  "text_lang": "zh",
  "voice_id": "default",
  "emotion": "happy",
  "speed_factor": 1.0,
  "media_type": "wav"
}
```

- `text_lang`: `auto`（自动检测，首次慢）/ `zh` / `en` / `ja`
- `emotion`: 该音色已配置的情绪之一（内置音色 default 有 `neutral` `happy` `sad` `angry` `surprised` `afraid`）；缺省用默认情绪
- `voice_id`: 缺省用当前激活音色；指定后会临时切换（下次请求仍用激活音色）
- `speed_factor`: 语速，默认 1.0
- `media_type`: `wav`（默认）/ `ogg` / `aac` / `raw`（16k 裸 PCM）
- `streaming`: `true` 开启流式合成（默认 `false`）。逐句完成后立刻以 chunked
  HTTP 下发，首句延迟大幅降低（多句文本明显）。响应头 `X-Audio-Sample-Rate`
  为采样率（wav/raw 均可用）。**仅支持 `wav` / `raw`**；`ogg` / `aac` 走流式
  会返回 400。

  流式 `wav` 的响应体是**一串独立的 WAV 段**（每句一个完整 WAV，各自带
  44 字节 RIFF 头），客户端按序解码即可边收边播；流式 `raw` 响应体是逐句的
  16-bit 单声道 PCM，需配合响应头的采样率解码。

  **抢占**：新的 `/tts` 请求（流式或普通）到达时，会立即终止当前正在进行的
  流式合成——引擎完成正在合成的当前句后停止，已产出的音频照常下发，旧请求
  以 HTTP 200 正常结束（不报错），新请求随后立即开始。适合「说话被打断、
  马上换下一句」的场景，无需等整段生成完。

多个并发请求会进入 FIFO 队列按提交顺序逐条合成；`/health` 的 `queue_pending`
可查看当前排队数量。流式请求串行执行（复用同一把合成锁），与普通请求互斥；
流式请求会被新请求抢占，普通（非流式）请求仍按 FIFO 排队。

### 音色与情绪参考路径

`gpt_path` / `sovits_path` / `wav` 为相对路径：优先在项目根目录下解析；若不存在
则回退到 `engine/` 下解析。

内置的 `default` 音色是**零样本预训练音色**（无需训练），其 `gpt_path` /
`sovits_path` 直接指向引擎内置底模，路径明确写为 `engine/GPT_SoVITS/pretrained_models/...`
（`engine/` 即整合包所在目录），一眼可辨，不会与项目根目录下的自定义权重混淆。

自定义训练音色则把权重放在项目根目录：`GPT_weights_v4/xxx.ckpt`、
`SoVITS_weights_v4/xxx.pth`，参考音频放 `ref/<voice>/<emotion>.wav`。
情绪参考音频建议使用训练集内的音频，文本需与实际发音一致（V4 推理音色由参考音频
强力控制）。接入自定义音色的训练流程见
[GPT-SoVITS 文档](https://github.com/RVC-Boss/GPT-SoVITS)。

