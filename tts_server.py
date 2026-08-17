# -*- coding: utf-8 -*-
"""
TTS 本地服务（基于 GPT-SoVITS V4）
==================================
- 自动选择 GPU / CPU 推理（启动时检测 CUDA）
- 多音色管理：注册 / 列出 / 切换（按音色懒加载引擎实例）
- 每音色可配置任意数量情绪（名称自定，如 neutral / happy / sad / ...），
  情绪通过「参考音频」迁移实现（V3/V4 情绪表达能力最佳）；
  情绪从 voices.json 初始载入，请求未指定时用默认情绪
  （优先级：请求参数 > 音色 default_emotion > server_config default_emotion > 该音色第一个情绪）
- **并发**：流式与非流式请求各自持有一个轻量引擎实例（共享同一音色的权重模型，
  各自独立的 prompt_cache / stop_flag），任意请求（不同音色 / 不同情绪 / 相同情绪）
  均可并行合成，不再有 FIFO 队列。
- `stop_prev`：新请求带 `stop_prev=true` 时，终止所有在途的流式请求（句级抢占，
  当前句播完后返回）；false（默认）则完全不打断，全部并发。
- HTTP 接口返回音频（wav / ogg / raw / aac）

启动:
    python tts_server.py [--host 127.0.0.1] [--port 9880] [--voices voices.json]
接口:
    GET  /health                  探针（含设备信息）
    GET  /voices                  列出音色 + 各自情绪 + 当前激活音色
    POST /voices/{voice_id}/activate  激活音色（校验 + 预热）
    POST /tts                     合成语音，返回音频流
"""

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

# 关闭引擎的 tqdm 进度条输出：Windows 控制台在窗口最小化 / 被遮挡时渲染跟不上，
# 高频刷新的进度条写入会阻塞线程，导致 GPU 空转、合成变慢。本环境 tqdm 4.70 的
# disable 参数默认是 False（非 None），TQDM_DISABLE 环境变量会被忽略，因此用子类
# 强制 disable=True；须在引擎 import 之前完成（引擎各处均为 from tqdm import tqdm，
# 且 pytorch_lightning 会继承 tqdm.tqdm，故必须是类而非函数）。
import tqdm as _tqdm


class _SilentTqdm(_tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


_tqdm.tqdm = _SilentTqdm


class _SilentStdout:
    """线程感知的 stdout 包装：调用 mute() 后，仅丢弃当前线程的 write。

    引擎在合成/加载时向 stdout 打印大量诊断（切分文本、BERT 特征、解码进度、
    合成用时等），这些对用户是噪音；但错误 traceback 走 stderr，不受影响。
    多个并发请求各自在独立线程合成，用 thread-local 标记即可互不干扰地静音。
    """

    def __init__(self):
        self._real = sys.stdout
        self._local = threading.local()

    def _muted(self) -> bool:
        return bool(getattr(self._local, "muted", False))

    def mute(self):
        self._local.muted = True

    def unmute(self):
        self._local.muted = False

    def write(self, s):
        if not self._muted():
            self._real.write(s)

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._real, name)


@contextlib.contextmanager
def _silence_engine():
    """在当前线程静音引擎 stdout 噪音（加载/合成期间的 print），退出后恢复。"""
    sys.stdout.mute()
    try:
        yield
    finally:
        sys.stdout.unmute()

# ----------------------------------------------------------------------------
# 定位引擎目录（包含 GPT_SoVITS 的目录）并把它加入 sys.path
# ----------------------------------------------------------------------------
def find_engine_dir(start: Path) -> Path:
    candidates = [Path.cwd(), start]
    d = start.resolve()
    for _ in range(6):
        candidates.append(d)
        if d.parent == d:
            break
        d = d.parent
    for c in candidates:
        if (c / "GPT_SoVITS").is_dir():
            return c
    for c in candidates:
        for sub in c.iterdir():
            if sub.is_dir() and (sub / "GPT_SoVITS").is_dir():
                return sub
    return Path.cwd()

ENGINE_DIR = find_engine_dir(Path(__file__).resolve().parent)
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(ENGINE_DIR)
sys.path.append(str(ENGINE_DIR))
sys.path.append(str(ENGINE_DIR / "GPT_SoVITS"))

import numpy as np
import torch
import torchaudio
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, StreamingResponse
from pydantic import BaseModel

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import splits

# ----------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------
MEDIA_TYPES = ["wav", "ogg", "raw", "aac"]
LANGUAGES = ["auto", "auto_yue", "zh", "ja", "en", "yue", "ko",
             "all_zh", "all_ja", "all_yue", "all_ko"]

# parallel_infer 恒为 True：保证所有并发请求在共享的 t2s_model 上落到同一个
# infer_panel（batch_infer），避免并发改写模型属性互相干扰。
DEFAULT_REQ = {
    "top_k": 15,
    "top_p": 1.0,
    "temperature": 1.0,
    "text_split_method": "cut5",
    "batch_size": 1,
    "batch_threshold": 0.75,
    "split_bucket": True,
    "speed_factor": 1.0,
    "fragment_interval": 0.3,
    "seed": -1,
    "parallel_infer": True,
    "repetition_penalty": 1.35,
    "sample_steps": 32,
    "super_sampling": False,
    "streaming_mode": False,
    "overlap_length": 2,
    "min_chunk_length": 16,
}

# 服务/引擎配置默认值（可被 server_config.json 覆盖，命令行参数优先级最高）
DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 9880,
    "device": "auto",   # auto / cuda（无 GPU 时拒绝启动）
    "default_emotion": "neutral",
    "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
}


def load_server_config() -> dict:
    """读取项目根目录 server_config.json（用户可编辑）；缺失/损坏时返回 {}。"""
    p = PROJECT_ROOT / "server_config.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[config] 读取 server_config.json 失败，使用默认值: {e}")
        return {}


def _sanitize_text(text: str) -> str:
    """归一化引擎无法处理的字符，防止个别符号导致合成崩溃。

    引擎 zh_normalization 只认识部分文字系统的数字（DIGITS 表不含泰文数字等），
    遇到生僻数字会抛 KeyError。这里把「所有 Unicode 十进制数字」（Nd 类，
    如泰文 ๑、阿拉伯-印度数字 ٥）统一转成 ASCII 数字——语义完全等价。
    其余字符原样保留。
    """
    out = []
    for ch in text:
        if unicodedata.category(ch) == "Nd":
            try:
                out.append(str(unicodedata.digit(ch)))
            except ValueError:
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


async def _is_disconnected(request: Request) -> bool:
    """检测客户端是否断开。外层 wait_for 兼容旧版 Starlette（is_disconnected
    无内部超时、无消息时会一直阻塞）与新版（自带极短超时）。"""
    try:
        return await asyncio.wait_for(request.is_disconnected(), timeout=0.1)
    except asyncio.TimeoutError:
        return False


async def _aq_or_disconnect(q: asyncio.Queue, request: Request):
    """等待队列下一项；客户端断开则返回 (True, None)，否则 (False, item)。"""
    get_task = asyncio.create_task(q.get())
    while True:
        if await _is_disconnected(request):
            get_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await get_task
            return True, None
        done, _ = await asyncio.wait({get_task}, timeout=0.5)
        if done:
            return False, get_task.result()

# ----------------------------------------------------------------------------
# 音频打包
# ----------------------------------------------------------------------------
def pack_wav(data: np.ndarray, rate: int) -> bytes:
    buf = BytesIO()
    sf.write(buf, data, rate, format="wav")
    return buf.getvalue()


def pack_ogg(data: np.ndarray, rate: int) -> bytes:
    buf = BytesIO()
    sf.write(buf, data, rate, format="ogg")
    return buf.getvalue()


def pack_raw(data: np.ndarray) -> bytes:
    return data.astype(np.int16).tobytes()


def pack_aac(data: np.ndarray, rate: int) -> bytes:
    pcm = data.astype(np.int16).tobytes()
    proc = subprocess.Popen(
        ["ffmpeg", "-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", "pipe:0",
         "-c:a", "aac", "-b:a", "192k", "-vn", "-f", "adts", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    out, _ = proc.communicate(input=pcm)
    return out


def pack_audio(data: np.ndarray, rate: int, media_type: str) -> bytes:
    if media_type == "ogg":
        return pack_ogg(data, rate)
    if media_type == "aac":
        return pack_aac(data, rate)
    if media_type == "raw":
        return pack_raw(data)
    return pack_wav(data, rate)


MEDIA_MIME = {
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "raw": "audio/x-raw",
    "aac": "audio/aac",
}

# ----------------------------------------------------------------------------
# 音色注册表
# ----------------------------------------------------------------------------
class VoiceRegistry:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"version": "1.0", "active_voice": None, "voices": {}}
        data.setdefault("voices", {})
        data.setdefault("active_voice", None)
        return data

    def save(self):
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    def voices(self) -> dict:
        return self.data["voices"]

    def active(self) -> Optional[str]:
        return self.data.get("active_voice")

    def set_active(self, voice_id: str):
        self.data["active_voice"] = voice_id
        self.save()

    def get(self, voice_id: str) -> Optional[dict]:
        return self.voices().get(voice_id)

    def validate(self, voice: dict) -> Optional[str]:
        gpt = voice.get("gpt_path", "")
        sovits = voice.get("sovits_path", "")
        if not gpt or not os.path.isfile(self.resolve(gpt)):
            return f"gpt_path 不存在: {gpt}"
        if not sovits or not os.path.isfile(self.resolve(sovits)):
            return f"sovits_path 不存在: {sovits}"
        emotions = voice.get("emotions", {})
        if not isinstance(emotions, dict):
            return "emotions 必须是对象"
        for emo, ref in emotions.items():
            if not isinstance(ref, dict):
                return f"情绪 [{emo}] 配置格式错误"
            wav = ref.get("wav", "")
            if not wav or not os.path.isfile(self.resolve(wav)):
                return f"情绪 [{emo}] 参考音频不存在: {wav}"
            if not ref.get("text", "").strip():
                return f"情绪 [{emo}] 缺少参考音频文本 (text)"
        return None

    def resolve(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        r = Path(p)
        if (PROJECT_ROOT / r).exists():
            return str(PROJECT_ROOT / r)
        return str(ENGINE_DIR / r)

# ----------------------------------------------------------------------------
# TTS 引擎封装（并发版）
# ----------------------------------------------------------------------------
class _LockedTextPreprocessor:
    """对共享 text_preprocessor 的调用加锁。

    并发请求共用同一个 text_preprocessor（内部走 pyopenjtalk g2p / BERT 特征
    提取），这些 C/模型调用并非线程安全，用一把全局锁串行化文本预处理部分；
    文本预处理只占推理的一小部分，不影响并行的 GPU 合成主流程。
    """

    def __init__(self, inner, lock: threading.Lock):
        self._inner = inner
        self._lock = lock

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def preprocess(self, *args, **kwargs):
        with self._lock:
            return self._inner.preprocess(*args, **kwargs)

    def segment_and_extract_feature_for_text(self, *args, **kwargs):
        with self._lock:
            return self._inner.segment_and_extract_feature_for_text(*args, **kwargs)


class EmotionTTS(TTS):
    """共享基础引擎模型、拥有独立 prompt_cache / stop_flag 的轻量实例。

    不加载任何权重（避免重复占用显存）；每个请求用后即弃。
    参考特征在创建时注入 prompt_cache，run() 期间不再改写共享状态。
    """

    def __init__(self, base: TTS):
        # 复制 base 的全部实例属性（模型、配置等，均为共享引用），仅 prompt_cache /
        # stop_flag 用本实例独立副本。不硬编码属性名：不同引擎版本（官方整合包 vs
        # GitHub 源码）属性集合可能不同（如 sv_model 仅部分版本存在），复制 __dict__
        # 天然兼容。run() 在推理期间只会改写 stop_flag / prompt_cache（已独立）与
        # infer_panel / t2s_model（仅重绑定本实例，不污染共享的 base）。
        self.configs = base.configs
        for k, v in base.__dict__.items():
            if k in ("configs", "prompt_cache", "stop_flag"):
                continue
            setattr(self, k, v)
        self.stop_flag = False
        self.prompt_cache = {
            "ref_audio_path": None,
            "prompt_semantic": None,
            "refer_spec": [],
            "prompt_text": None,
            "prompt_lang": None,
            "phones": None,
            "bert_features": None,
            "norm_text": None,
            "aux_ref_audio_paths": [],
        }


class TTSBackend:
    def __init__(self, registry: VoiceRegistry, initial_voice_id: Optional[str],
                 device: str = "auto",
                 default_emotion: Optional[str] = None,
                 bert_base_path: Optional[str] = None,
                 cnhuhbert_base_path: Optional[str] = None):
        self.registry = registry
        self.default_emotion = default_emotion or DEFAULT_CONFIG["default_emotion"]
        self.sample_rate = 48000

        # 安装线程感知的 stdout 静音包装（压制引擎加载/合成的诊断噪音）
        if not isinstance(sys.stdout, _SilentStdout):
            sys.stdout = _SilentStdout()

        use_cuda = torch.cuda.is_available()
        if device == "cuda":
            if not use_cuda:
                print("[backend] 请求使用 CUDA 但不可用，回退到 CPU")
                device = "cpu"
        elif device == "cpu":
            pass
        else:
            device = "cuda" if use_cuda else "cpu"
        self.device = device
        self.is_half = device == "cuda"

        self.bert_base_path = bert_base_path or DEFAULT_CONFIG["bert_base_path"]
        self.cnhuhbert_base_path = cnhuhbert_base_path or DEFAULT_CONFIG["cnhuhbert_base_path"]

        self._bases: dict = {}
        self._base_lock = threading.Lock()
        self._ref_cache: dict = {}
        self._ref_lock = threading.RLock()
        self._text_lock = threading.Lock()

        if initial_voice_id and not registry.get(initial_voice_id):
            raise ValueError(f"音色不存在: {initial_voice_id}")
        vid = self._pick_initial(initial_voice_id)
        if registry.active() is None:
            registry.set_active(vid)
        voice = registry.get(vid)
        self.engine_version = voice.get("engine", "v4") if voice else "v4"
        print(f"[backend] 引擎就绪 device={self.device} is_half={self.is_half} "
              f"默认音色={vid}（按需加载，流式/非流式可并发）")
        self.prime_refs(vid)

    def _pick_initial(self, requested: Optional[str]) -> str:
        if requested and self.registry.get(requested):
            return requested
        active = self.registry.active()
        if active and self.registry.get(active):
            return active
        if self.registry.voices():
            return next(iter(self.registry.voices()))
        raise RuntimeError("voices.json 中没有可用的音色，请先注册音色")

    def _build_base(self, voice: dict) -> TTS:
        cfg = {
            "custom": {
                "bert_base_path": self.bert_base_path,
                "cnhuhbert_base_path": self.cnhuhbert_base_path,
                "device": self.device,
                "is_half": self.is_half,
                "t2s_weights_path": self.registry.resolve(voice["gpt_path"]),
                "vits_weights_path": self.registry.resolve(voice["sovits_path"]),
                "version": voice.get("engine", "v4"),
            }
        }
        return TTS(TTS_Config(cfg))

    def _get_base(self, voice_id: str) -> TTS:
        with self._base_lock:
            base = self._bases.get(voice_id)
            if base is not None:
                return base
            voice = self.registry.get(voice_id)
            if not voice:
                raise ValueError(f"音色不存在: {voice_id}")
            err = self.registry.validate(voice)
            if err:
                raise ValueError(err)
            t0 = time.time()
            with _silence_engine():
                base = self._build_base(voice)
            self._bases[voice_id] = base
            print(f"[backend] 音色 [{voice_id}] 引擎实例加载 ({(time.time()-t0):.1f}s)")
            return base

    def ensure_voice(self, voice_id: str) -> str:
        """校验并预热音色（加载权重 + 预计算情绪参考特征），供激活/切换使用。"""
        self._get_base(voice_id)
        self.prime_refs(voice_id)
        self.registry.set_active(voice_id)
        return voice_id

    def _get_ref_features(self, voice_id: str, emotion: str, ref_wav: str,
                          prompt_text: str, prompt_lang: str) -> dict:
        key = (voice_id, emotion)
        try:
            st = os.stat(ref_wav)
            mtime_sz = (st.st_mtime, st.st_size)
        except OSError:
            mtime_sz = None
        with self._ref_lock:
            entry = self._ref_cache.get(key)
            if entry and entry["mtime_sz"] == mtime_sz and entry["path"] == ref_wav:
                return entry["features"]

            base = self._get_base(voice_id)
            with self._text_lock:
                with _silence_engine():
                    if base.prompt_cache.get("ref_audio_path") != ref_wav:
                        base.set_ref_audio(ref_wav)
                    pt = prompt_text.strip("\n")
                    if pt and pt[-1] not in splits:
                        pt += "。" if prompt_lang != "en" else "."
                    if base.prompt_cache.get("prompt_text") != pt:
                        phones, bert_features, norm_text = (
                            base.text_preprocessor.segment_and_extract_feature_for_text(
                                pt, prompt_lang, base.configs.version))
                        base.prompt_cache["prompt_text"] = pt
                        base.prompt_cache["prompt_lang"] = prompt_lang
                        base.prompt_cache["phones"] = phones
                        base.prompt_cache["bert_features"] = bert_features
                        base.prompt_cache["norm_text"] = norm_text

                features = {
                    "raw_audio": base.prompt_cache["raw_audio"],
                    "raw_sr": base.prompt_cache["raw_sr"],
                    "refer_spec": base.prompt_cache["refer_spec"][:],
                    "prompt_semantic": base.prompt_cache["prompt_semantic"],
                    "prompt_text": base.prompt_cache["prompt_text"],
                    "prompt_lang": base.prompt_cache["prompt_lang"],
                    "phones": base.prompt_cache["phones"],
                    "bert_features": base.prompt_cache["bert_features"],
                    "norm_text": base.prompt_cache["norm_text"],
                    "ref_audio_path": base.prompt_cache["ref_audio_path"],
                    "aux_ref_audio_paths": base.prompt_cache["aux_ref_audio_paths"],
                }
            self._ref_cache[key] = {"mtime_sz": mtime_sz, "path": ref_wav, "features": features}
            return features

    def prime_refs(self, voice_id: str):
        """启动时预计算该音色所有情绪的参考特征，消除首个请求 / 情绪切换的额外耗时。"""
        voice = self.registry.get(voice_id)
        if not voice:
            return
        t0 = time.time()
        n = 0
        emotions = voice.get("emotions", {})
        for emotion, ref in emotions.items():
            try:
                self._get_ref_features(
                    voice_id, emotion, self.registry.resolve(ref["wav"]),
                    ref["text"], ref["lang"])
                n += 1
            except Exception as e:
                print(f"[refcache] 预计算情绪 [{emotion}] 失败: {e}")
        print(f"[refcache] 音色 {voice_id} 参考特征预计算 {n}/{len(emotions)} 个 ({(time.time()-t0):.1f}s)")

    def resolve_emotion(self, voice: dict, requested: Optional[str]) -> str:
        """解析实际使用的情绪：
        请求显式指定 > 音色 default_emotion > server_config default_emotion > 该音色第一个情绪。"""
        emos = list(voice.get("emotions", {}))
        if requested:
            return requested
        for cand in (voice.get("default_emotion"), self.default_emotion):
            if cand and cand in voice.get("emotions", {}):
                return cand
        return emos[0] if emos else self.default_emotion

    def new_instance(self, voice_id: str, features: dict) -> EmotionTTS:
        """为单个请求创建轻量引擎实例（共享该音色权重模型，注入参考特征）。"""
        base = self._get_base(voice_id)
        inst = EmotionTTS(base)
        inst.text_preprocessor = _LockedTextPreprocessor(base.text_preprocessor, self._text_lock)
        inst.prompt_cache.update(features)
        return inst

    def _build_req(self, text: str, text_lang: str, ref: dict, extra: dict,
                   return_fragment: bool) -> dict:
        req = dict(DEFAULT_REQ)
        req.update(extra)
        req.update({
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": self.registry.resolve(ref["wav"]),
            "prompt_text": ref["text"],
            "prompt_lang": ref["lang"],
            "return_fragment": return_fragment,
        })
        return req

    def run_non_stream(self, inst: EmotionTTS, text: str, text_lang: str,
                       ref: dict, media_type: str, extra: dict):
        """非流式合成（在线程池中执行，实例用后即弃，可并发）。"""
        try:
            req = self._build_req(text, text_lang, ref, extra, return_fragment=False)
            with _silence_engine():
                gen = inst.run(req)
                sr, audio = next(gen)
            audio = np.asarray(audio)
            return pack_audio(audio, sr, media_type), sr
        finally:
            try:
                del inst
            except Exception:
                pass

    def run_stream(self, inst: EmotionTTS, req: dict, emotion: str):
        """流式合成：逐句 yield (sample_rate, audio_int16)。"""
        for sr, audio in inst.run(req):
            yield sr, np.asarray(audio)

    def cancel_instance(self, inst: EmotionTTS):
        """句级终止指定实例的流式合成（当前句结束后返回）。"""
        try:
            inst.stop_flag = True
        except Exception:
            pass

# ----------------------------------------------------------------------------
# FastAPI 应用
# ----------------------------------------------------------------------------
def create_app(backend: TTSBackend, registry: VoiceRegistry) -> FastAPI:
    app = FastAPI(title="TTS 本地服务", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    active = {"lock": threading.Lock(), "streams": {}}

    def _stop_all_streams():
        """终止所有在途的流式请求（句级抢占，供 stop_prev 使用）。"""
        with active["lock"]:
            streams = list(active["streams"].values())
        for inst in streams:
            backend.cancel_instance(inst)

    class TTSRequest(BaseModel):
        text: str
        text_lang: str = "auto"
        voice_id: Optional[str] = None
        emotion: Optional[str] = None
        media_type: str = "wav"
        streaming: bool = False
        stop_prev: bool = False
        speed_factor: float = 1.0
        top_k: int = 15
        top_p: float = 1.0
        temperature: float = 1.0
        seed: int = -1
        repetition_penalty: float = 1.35

    @app.get("/health")
    async def health():
        with active["lock"]:
            n_streams = len(active["streams"])
        return {
            "code": 0,
            "message": "ok",
            "data": {
                "status": "running",
                "device": backend.device,
                "is_half": backend.is_half,
                "cuda": torch.cuda.is_available(),
                "engine_version": backend.engine_version,
                "active_voice": registry.active(),
                "voice_count": len(registry.voices()),
                "active_streams": n_streams,
                "concurrency": "per-request-instance",
            },
        }

    @app.get("/voices")
    async def list_voices():
        voices = {}
        for vid, v in registry.voices().items():
            voices[vid] = {
                "name": v.get("name", vid),
                "engine": v.get("engine", "v4"),
                "gpt_path": v.get("gpt_path"),
                "sovits_path": v.get("sovits_path"),
                "default_emotion": backend.resolve_emotion(v, None),
                "emotions": {
                    emo: {
                        "wav": ref.get("wav"),
                        "lang": ref.get("lang"),
                        "text": ref.get("text"),
                    }
                    for emo, ref in v.get("emotions", {}).items()
                },
            }
        return {"code": 0, "message": "ok",
                "data": {"active_voice": registry.active(), "voices": voices}}

    @app.post("/voices/{voice_id}/activate")
    async def activate_voice(voice_id: str):
        try:
            backend.ensure_voice(voice_id)
        except (ValueError, RuntimeError) as e:
            return JSONResponse(status_code=400, content={"code": 400, "message": str(e)})
        return {"code": 0, "message": "ok", "data": {"voice_id": voice_id, "device": backend.device}}

    @app.post("/tts")
    async def tts(req: TTSRequest, request: Request):
        text = _sanitize_text(req.text)
        if not text.strip():
            return JSONResponse(status_code=400, content={"code": 400, "message": "text 不能为空"})
        if req.text_lang not in LANGUAGES:
            return JSONResponse(status_code=400,
                                content={"code": 400, "message": f"text_lang 不支持: {req.text_lang}"})
        if req.media_type not in MEDIA_TYPES:
            return JSONResponse(status_code=400,
                                content={"code": 400, "message": f"media_type 不支持: {req.media_type}"})
        try:
            voice_id = req.voice_id or registry.active()
            voice = registry.get(voice_id)
            if not voice:
                raise ValueError(f"音色不存在: {voice_id}")
            emotion = backend.resolve_emotion(voice, req.emotion)
            ref = voice["emotions"].get(emotion)
            if not ref:
                # 请求的情绪不存在：回退到默认情绪链（音色 default_emotion >
                # 全局 default_emotion > 该音色第一个情绪）
                fallback = backend.resolve_emotion(voice, None)
                print(f"[tts] 情绪 [{emotion}] 不存在，回退到 [{fallback}]")
                emotion = fallback
                ref = voice["emotions"].get(emotion)
                if not ref:
                    raise ValueError(f"音色 [{voice_id}] 没有可用情绪，可选: {list(voice['emotions'].keys())}")

            mode = "流式" if req.streaming else "合成"
            shown = text if len(text) <= 60 else text[:60] + "…"
            print(f"[tts] [{mode}] {voice_id}/{emotion} ({req.text_lang}): {shown}")

            ref_wav = registry.resolve(ref["wav"])
            features = backend._get_ref_features(voice_id, emotion, ref_wav,
                                                 ref["text"], ref["lang"])
            extra = {
                "speed_factor": float(req.speed_factor),
                "top_k": int(req.top_k),
                "top_p": float(req.top_p),
                "temperature": float(req.temperature),
                "seed": int(req.seed),
                "repetition_penalty": float(req.repetition_penalty),
            }

            if req.streaming:
                if req.media_type not in ("wav", "raw"):
                    return JSONResponse(status_code=400, content={
                        "code": 400,
                        "message": f"streaming 仅支持 wav / raw，不支持 {req.media_type}"})

                if req.stop_prev:
                    _stop_all_streams()

                inst = backend.new_instance(voice_id, features)
                rid = id(inst)
                with active["lock"]:
                    active["streams"][rid] = inst
                body = backend._build_req(text, req.text_lang, ref, extra, return_fragment=True)
                aq: asyncio.Queue = asyncio.Queue()

                def producer():
                    try:
                        with _silence_engine():
                            for sr, audio in backend.run_stream(inst, body, emotion):
                                if audio.size and not audio.any():
                                    continue  # 引擎停止/出错时的静音段，不下发
                                aq.put_nowait((sr, pack_audio(audio, sr, req.media_type)))
                        aq.put_nowait(None)
                    except Exception as e:
                        aq.put_nowait(e)
                    finally:
                        # 注意：不要对 inst 做任何赋值/del（否则会把 inst 变成 producer
                        # 的局部变量，导致上方 run_stream(inst, ...) 抛 UnboundLocalError）。
                        # 线程结束后 run_stream 生成器被回收，inst 随之释放。
                        with active["lock"]:
                            active["streams"].pop(rid, None)

                threading.Thread(target=producer, daemon=True).start()
                disconnected, first = await _aq_or_disconnect(aq, request)
                if disconnected:
                    # 首块到达前客户端断开：停止合成，避免孤儿请求继续占 GPU
                    print(f"[tts] 客户端断开，取消流式合成 ({voice_id}/{emotion})")
                    backend.cancel_instance(inst)
                    return Response(content=b"", status_code=499,
                                    media_type=MEDIA_MIME[req.media_type])
                if first is None:
                    return Response(content=b"", media_type=MEDIA_MIME[req.media_type])
                if isinstance(first, Exception):
                    if isinstance(first, (ValueError, RuntimeError, FileNotFoundError)):
                        return JSONResponse(status_code=400, content={"code": 400, "message": str(first)})
                    traceback.print_exc()
                    return JSONResponse(status_code=500, content={"code": 500, "message": str(first)})
                sr0, chunk0 = first
                headers = {"X-Audio-Sample-Rate": str(sr0), "Cache-Control": "no-cache"}

                async def stream():
                    try:
                        yield chunk0
                        while True:
                            disconnected, item = await _aq_or_disconnect(aq, request)
                            if disconnected:
                                break
                            if item is None or isinstance(item, Exception):
                                break
                            yield item[1]
                    finally:
                        # 正常结束：清注册；客户端断开：终止后台合成，尽快释放实例
                        backend.cancel_instance(inst)

                return StreamingResponse(stream(), media_type=MEDIA_MIME[req.media_type],
                                         headers=headers)

            if req.stop_prev:
                _stop_all_streams()
            inst = backend.new_instance(voice_id, features)
            task = asyncio.create_task(asyncio.to_thread(
                backend.run_non_stream, inst, text, req.text_lang,
                ref, req.media_type, extra))
            try:
                while True:
                    done, _ = await asyncio.wait({task}, timeout=0.5)
                    if done:
                        break
                    if await _is_disconnected(request):
                        # 客户端断开：句级取消合成，避免孤儿请求继续占 GPU
                        print(f"[tts] 客户端断开，取消合成 ({voice_id}/{emotion})")
                        backend.cancel_instance(inst)
                        try:
                            await asyncio.wait_for(task, timeout=30)
                        except asyncio.TimeoutError:
                            task.cancel()  # 线程无法强杀，但 stop_flag 已置位，会尽快退出
                        return Response(content=b"", status_code=499,
                                        media_type=MEDIA_MIME[req.media_type])
                data, sr = task.result()
            except asyncio.CancelledError:
                backend.cancel_instance(inst)
                raise
            return Response(content=data, media_type=MEDIA_MIME[req.media_type])
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            return JSONResponse(status_code=400, content={"code": 400, "message": str(e)})
        except Exception as e:
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"code": 500, "message": str(e)})

    return app


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------
if sys.platform == "win32":
    import ctypes

    class _PowerThrottling(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_uint32),
                    ("ControlMask", ctypes.c_uint32),
                    ("StateMask", ctypes.c_uint32)]


def _apply_windows_perf_boost():
    """Windows 下静默防止窗口最小化/后台时被系统功耗节流而降速。

    实测（RTX4090）：窗口后台时合成慢约 2 倍（GPU 利用率仅 ~40%），
    关掉 ExecutionSpeed 功耗节流 + 禁用 priority boost + 提到 AboveNormal 后
    后台性能恢复到接近前台。仅对当前进程生效（进程退出即自动还原系统原本设定，
    不会持久改变 Windows 设置）。失败不影响启动，全部静默处理。
    """
    if sys.platform != "win32":
        return
    try:
        k32 = ctypes.windll.kernel32
        h = k32.GetCurrentProcess()
        k32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                              ctypes.c_void_p, ctypes.c_uint32]
        k32.SetProcessInformation.restype = ctypes.c_bool
        k32.SetProcessPriorityBoost.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        k32.SetProcessPriorityBoost.restype = ctypes.c_bool
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass.restype = ctypes.c_bool

        st = _PowerThrottling(1, 1, 0)  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 1，置 0 即禁用
        k32.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st))
        k32.SetProcessPriorityBoost(h, True)
        k32.SetPriorityClass(h, 0x00008000)  # ABOVE_NORMAL_PRIORITY_CLASS
    except Exception:
        pass


def main():
    _apply_windows_perf_boost()
    cfg = load_server_config()
    parser = argparse.ArgumentParser(description="TTS 本地服务 (GPT-SoVITS V4)")
    parser.add_argument("-a", "--host", type=str, default=None)
    parser.add_argument("-p", "--port", type=int, default=None)
    parser.add_argument("--voices", type=str, default=str(PROJECT_ROOT / "voices.json"))
    parser.add_argument("--initial-voice", type=str, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="auto / cuda（默认取 server_config.json，再取 auto；无 GPU 时拒绝启动）")
    args = parser.parse_args()

    host = args.host or cfg.get("host") or DEFAULT_CONFIG["host"]
    port = int(args.port or cfg.get("port") or DEFAULT_CONFIG["port"])
    device = args.device or cfg.get("device") or DEFAULT_CONFIG["device"]

    # 无 GPU 直接拒绝启动：CPU 合成速度不可接受（快速结束的前提就是有 GPU 实时合成）
    if not torch.cuda.is_available() or device == "cpu":
        print("=" * 60)
        print("[tts] 未检测到可用的 NVIDIA GPU（CUDA），拒绝启动。")
        print("      GPT-SoVITS 的 CPU 合成速度无法支持实时使用，本服务仅在 GPU 上运行。")
        print("      请确认 NVIDIA 显卡驱动 / CUDA 环境正常后重试。")
        print("=" * 60)
        sys.exit(1)

    registry = VoiceRegistry(args.voices)
    backend = TTSBackend(registry, args.initial_voice, device=device,
                         default_emotion=cfg.get("default_emotion"),
                         bert_base_path=cfg.get("bert_base_path"),
                         cnhuhbert_base_path=cfg.get("cnhuhbert_base_path"))
    app = create_app(backend, registry)

    print("=" * 60)
    print(f"  TTS 本地服务  http://{host}:{port}")
    print(f"  设备: {backend.device}  半精度: {backend.is_half}  引擎: {backend.engine_version}")
    print(f"  默认情绪: {backend.default_emotion}（请求未指定时使用；可按音色在 voices.json 覆盖）")
    print(f"  接口: GET /health | GET /voices | POST /voices/{{id}}/activate | POST /tts")
    print("  并发: 每请求独立轻量引擎实例，流式/非流式可并行；stop_prev 可句级抢占在途流式请求")
    print("=" * 60)
    # log_level=warning：去掉 uvicorn 的逐请求访问日志，保留错误/警告
    try:
        uvicorn.run(app, host=host, port=port, workers=1, log_level="warning")
        print("\n[TTS] 服务已正常退出")
    except KeyboardInterrupt:
        print("\n[TTS] 服务已正常退出（Ctrl+C）")


if __name__ == "__main__":
    main()