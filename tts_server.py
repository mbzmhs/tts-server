# -*- coding: utf-8 -*-
"""
TTS 本地服务（基于 GPT-SoVITS V4）
==================================
- 自动选择 GPU / CPU 推理（启动时检测 CUDA）
- 多音色管理：注册 / 列出 / 切换（模型热加载）
- 每音色可配置任意数量情绪（名称自定，如 neutral / happy / sad / ...），
  情绪通过「参考音频」迁移实现（V3/V4 情绪表达能力最佳）；
  情绪从 voices.json 初始载入，请求未指定时用默认情绪
  （优先级：请求参数 > 音色 default_emotion > server_config default_emotion > 该音色第一个情绪）
- HTTP 接口返回音频（wav / ogg / raw / aac）

启动:
    python tts_server.py [--host 127.0.0.1] [--port 9880] [--voices voices.json]
接口:
    GET  /health                  探针（含设备信息）
    GET  /voices                  列出音色 + 各自情绪 + 当前激活音色
    POST /voices                  注册 / 更新音色
    DELETE /voices/{voice_id}     删除音色
    POST /voices/{voice_id}/activate  切换激活音色（加载模型）
    POST /tts                     合成语音，返回音频流
"""

import argparse
import asyncio
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
from queue import Queue
from typing import Optional, Union

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
    "device": "auto",   # auto / cuda / cpu
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
# TTS 引擎封装
# ----------------------------------------------------------------------------
class TTSBackend:
    def __init__(self, registry: VoiceRegistry, initial_voice_id: Optional[str],
                 device: str = "auto",
                 default_emotion: Optional[str] = None,
                 bert_base_path: Optional[str] = None,
                 cnhuhbert_base_path: Optional[str] = None):
        self.registry = registry
        self.lock = threading.RLock()
        self.default_emotion = default_emotion or DEFAULT_CONFIG["default_emotion"]

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
        self.sample_rate = 48000

        bert_base_path = bert_base_path or DEFAULT_CONFIG["bert_base_path"]
        cnhuhbert_base_path = cnhuhbert_base_path or DEFAULT_CONFIG["cnhuhbert_base_path"]

        voice_id, voice = self._pick_initial(initial_voice_id)
        cfg = {
            "custom": {
                "bert_base_path": bert_base_path,
                "cnhuhbert_base_path": cnhuhbert_base_path,
                "device": self.device,
                "is_half": self.is_half,
                "t2s_weights_path": self.registry.resolve(voice["gpt_path"]),
                "vits_weights_path": self.registry.resolve(voice["sovits_path"]),
                "version": voice.get("engine", "v4"),
            }
        }
        self.config = TTS_Config(cfg)
        self.tts = TTS(self.config)
        self.current_voice_id = voice_id
        self._ref_cache: dict = {}
        if self.registry.active() is None:
            self.registry.set_active(voice_id)
        print(f"[backend] 引擎已加载 device={self.device} is_half={self.is_half} voice={voice_id}")
        self.prime_refs(voice_id)

    def _pick_initial(self, requested: Optional[str]):
        if requested and self.registry.get(requested):
            return requested, self.registry.get(requested)
        active = self.registry.active()
        if active and self.registry.get(active):
            return active, self.registry.get(active)
        if self.registry.voices():
            vid = next(iter(self.registry.voices()))
            return vid, self.registry.get(vid)
        raise RuntimeError("voices.json 中没有可用的音色，请先注册音色")

    def switch_voice(self, voice_id: str) -> str:
        if voice_id == self.current_voice_id:
            return voice_id
        voice = self.registry.get(voice_id)
        if not voice:
            raise ValueError(f"音色不存在: {voice_id}")
        err = self.registry.validate(voice)
        if err:
            raise ValueError(err)
        t0 = time.time()
        with self.lock:
            self.tts.init_t2s_weights(self.registry.resolve(voice["gpt_path"]))
            self.tts.init_vits_weights(self.registry.resolve(voice["sovits_path"]))
            self.current_voice_id = voice_id
            self.registry.set_active(voice_id)
        print(f"[backend] 切换音色 -> {voice_id} ({(time.time()-t0):.1f}s)")
        return voice_id

    def _get_ref_features(self, voice_id: str, emotion: str, ref_wav: str,
                          prompt_text: str, prompt_lang: str) -> dict:
        """取某情绪参考音频的已缓存特征；文件变动时重建。返回可直接注入 prompt_cache 的 dict。"""
        key = (voice_id, emotion)
        try:
            st = os.stat(ref_wav)
            mtime_sz = (st.st_mtime, st.st_size)
        except OSError:
            mtime_sz = None
        entry = self._ref_cache.get(key)
        if entry and entry["mtime_sz"] == mtime_sz and entry["path"] == ref_wav:
            return entry["features"]

        if self.tts.prompt_cache.get("ref_audio_path") != ref_wav:
            self.tts.set_ref_audio(ref_wav)

        print(f"[refcache] 重建特征 {voice_id}/{emotion} <- {os.path.basename(ref_wav)}")

        pt = prompt_text.strip("\n")
        if pt and pt[-1] not in splits:
            pt += "。" if prompt_lang != "en" else "."
        if self.tts.prompt_cache.get("prompt_text") != pt:
            phones, bert_features, norm_text = (
                self.tts.text_preprocessor.segment_and_extract_feature_for_text(
                    pt, prompt_lang, self.tts.configs.version
                )
            )
            self.tts.prompt_cache["prompt_text"] = pt
            self.tts.prompt_cache["prompt_lang"] = prompt_lang
            self.tts.prompt_cache["phones"] = phones
            self.tts.prompt_cache["bert_features"] = bert_features
            self.tts.prompt_cache["norm_text"] = norm_text

        features = {
            "raw_audio": self.tts.prompt_cache["raw_audio"],
            "raw_sr": self.tts.prompt_cache["raw_sr"],
            "refer_spec": self.tts.prompt_cache["refer_spec"][:],
            "prompt_semantic": self.tts.prompt_cache["prompt_semantic"],
            "prompt_text": self.tts.prompt_cache["prompt_text"],
            "prompt_lang": self.tts.prompt_cache["prompt_lang"],
            "phones": self.tts.prompt_cache["phones"],
            "bert_features": self.tts.prompt_cache["bert_features"],
            "norm_text": self.tts.prompt_cache["norm_text"],
            "ref_audio_path": self.tts.prompt_cache["ref_audio_path"],
            "aux_ref_audio_paths": self.tts.prompt_cache["aux_ref_audio_paths"],
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

    def _prepare(self, voice_id: Optional[str], emotion: Optional[str],
                 text: str, text_lang: str, extra: dict, streaming: bool):
        """加锁状态下准备一次合成：切音色、解析情绪、准备参考特征、构造请求并启动生成器。
        返回 (generator, voice_id, emotion)。调用方必须已持有 self.lock。"""
        text = _sanitize_text(text)
        voice_id = self.switch_voice(voice_id or self.registry.active())
        voice = self.registry.get(voice_id)
        emotion = self.resolve_emotion(voice, emotion)
        ref = voice["emotions"].get(emotion)
        if not ref:
            raise ValueError(f"音色 [{voice_id}] 没有情绪 [{emotion}]，可选: {list(voice['emotions'].keys())}")

        ref_wav = self.registry.resolve(ref["wav"])
        features = self._get_ref_features(voice_id, emotion, ref_wav,
                                          ref["text"], ref["lang"])
        self.tts.prompt_cache.update(features)

        req = dict(DEFAULT_REQ)
        req.update(extra)
        req.update({
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref_wav,
            "prompt_text": ref["text"],
            "prompt_lang": ref["lang"],
            "return_fragment": streaming,
        })
        gen = self.tts.run(req)
        return gen, voice_id, emotion

    def synthesize(self, voice_id: Optional[str], emotion: Optional[str],
                   text: str, text_lang: str, media_type: str, extra: dict):
        with self.lock:
            gen, voice_id, emotion = self._prepare(
                voice_id, emotion, text, text_lang, extra, streaming=False)
            t0 = time.time()
            sr, audio = next(gen)
            audio = np.asarray(audio)
            print(f"[tts] voice={voice_id} emotion={emotion} lang={text_lang} "
                  f"dur={len(audio)/sr:.2f}s infer={(time.time()-t0):.2f}s")
            return pack_audio(audio, sr, media_type), sr

    def synthesize_stream(self, voice_id: Optional[str], emotion: Optional[str],
                          text: str, text_lang: str, media_type: str, extra: dict):
        """流式合成：逐句 yield (sample_rate, audio_int16)。

        全程持有 self.lock，与其它合成互斥，避免并发请求改写 prompt_cache。
        每句已完成音频后立即产出，客户端可边收边播，显著降低首句延迟。
        """
        with self.lock:
            gen, voice_id, emotion = self._prepare(
                voice_id, emotion, text, text_lang, extra, streaming=True)
            for sr, audio in gen:
                audio = np.asarray(audio)
                print(f"[tts][stream] voice={voice_id} emotion={emotion} "
                      f"chunk={len(audio)/sr:.2f}s")
                yield sr, audio

    def cancel_stream(self):
        """终止当前流式合成。

        引擎在完成正在合成的当前句后停止（句级抢占），已产出的音频
        仍正常下发，流式请求随后正常结束。
        """
        if self.tts is not None:
            self.tts.stop_flag = True


# ----------------------------------------------------------------------------
# 顺序合成队列
# ----------------------------------------------------------------------------
class _SynthJob:
    __slots__ = ("fut", "voice_id", "emotion", "text", "text_lang",
                 "media_type", "extra")


class SynthesisQueue:
    """FIFO 顺序合成队列。

    请求按提交顺序逐条处理（单 worker），保证「一句话接着一句话」。
    合成在后台线程进行，事件循环不阻塞；任何异常只影响当次请求，
    队列继续处理后续任务。
    """

    def __init__(self, backend: TTSBackend, loop: asyncio.AbstractEventLoop):
        self.backend = backend
        self.loop = loop
        self._q: Queue = Queue()
        self._worker = threading.Thread(target=self._run, name="synth-worker",
                                        daemon=True)
        self._worker.start()

    def _run(self):
        while True:
            job = self._q.get()
            try:
                data, sr = self.backend.synthesize(
                    job.voice_id, job.emotion, job.text, job.text_lang,
                    job.media_type, job.extra)
                self.loop.call_soon_threadsafe(job.fut.set_result, (data, sr))
            except Exception as e:
                traceback.print_exc()
                self.loop.call_soon_threadsafe(job.fut.set_exception, e)
            finally:
                self._q.task_done()

    def submit(self, voice_id: Optional[str], emotion: str, text: str,
               text_lang: str, media_type: str, extra: dict) -> asyncio.Future:
        job = _SynthJob()
        job.fut = self.loop.create_future()
        job.voice_id = voice_id
        job.emotion = emotion
        job.text = text
        job.text_lang = text_lang
        job.media_type = media_type
        job.extra = extra
        self._q.put(job)
        return job.fut

    @property
    def pending(self) -> int:
        return self._q.qsize()


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

    synth_queue: Optional[SynthesisQueue] = None

    active = {"cancel": None}

    def _cancel_active_stream():
        """终止正在进行的流式合成（若有），供新请求抢占。"""
        fn = active.get("cancel")
        if fn:
            active["cancel"] = None
            fn()

    @app.on_event("startup")
    def _startup():
        nonlocal synth_queue
        synth_queue = SynthesisQueue(backend, asyncio.get_running_loop())
        print("[backend] 合成队列已启动 (FIFO 顺序合成)")

    class TTSRequest(BaseModel):
        text: str
        text_lang: str = "auto"
        voice_id: Optional[str] = None
        emotion: Optional[str] = None
        media_type: str = "wav"
        streaming: bool = False
        speed_factor: float = 1.0
        top_k: int = 15
        top_p: float = 1.0
        temperature: float = 1.0
        seed: int = -1
        repetition_penalty: float = 1.35

    @app.get("/health")
    async def health():
        return {
            "code": 0,
            "message": "ok",
            "data": {
                "status": "running",
                "device": backend.device,
                "is_half": backend.is_half,
                "cuda": torch.cuda.is_available(),
                "engine_version": backend.config.version,
                "active_voice": registry.active(),
                "voice_count": len(registry.voices()),
                "queue_pending": synth_queue.pending if synth_queue else 0,
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
            backend.switch_voice(voice_id)
        except (ValueError, RuntimeError) as e:
            return JSONResponse(status_code=400, content={"code": 400, "message": str(e)})
        return {"code": 0, "message": "ok", "data": {"voice_id": voice_id, "device": backend.device}}

    @app.post("/tts")
    async def tts(req: TTSRequest):
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

                _cancel_active_stream()
                aq: asyncio.Queue = asyncio.Queue()

                def cancel_fn():
                    backend.cancel_stream()

                active["cancel"] = cancel_fn

                def producer():
                    try:
                        for sr, audio in backend.synthesize_stream(
                                req.voice_id, req.emotion, text, req.text_lang,
                                req.media_type, extra):
                            if audio.size and not audio.any():
                                continue  # 引擎停止/出错时的静音段，不下发
                            aq.put_nowait((sr, pack_audio(audio, sr, req.media_type)))
                        aq.put_nowait(None)
                    except Exception as e:
                        aq.put_nowait(e)

                threading.Thread(target=producer, daemon=True).start()
                first = await aq.get()
                if first is None:
                    active["cancel"] = None
                    return Response(content=b"", media_type=MEDIA_MIME[req.media_type])
                if isinstance(first, Exception):
                    if active.get("cancel") is cancel_fn:
                        active["cancel"] = None
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
                            item = await aq.get()
                            if item is None or isinstance(item, Exception):
                                break
                            yield item[1]
                    finally:
                        # 正常结束：清注册；客户端断开：终止后台合成，尽快释放合成锁
                        if active.get("cancel") is cancel_fn:
                            active["cancel"] = None
                            cancel_fn()

                return StreamingResponse(stream(), media_type=MEDIA_MIME[req.media_type],
                                         headers=headers)

            _cancel_active_stream()
            fut = synth_queue.submit(req.voice_id, req.emotion,
                                     text, req.text_lang,
                                     req.media_type, extra)
            data, sr = await fut
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
def main():
    cfg = load_server_config()
    parser = argparse.ArgumentParser(description="TTS 本地服务 (GPT-SoVITS V4)")
    parser.add_argument("-a", "--host", type=str, default=None)
    parser.add_argument("-p", "--port", type=int, default=None)
    parser.add_argument("--voices", type=str, default=str(PROJECT_ROOT / "voices.json"))
    parser.add_argument("--initial-voice", type=str, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="auto / cuda / cpu（默认取 server_config.json，再取 auto）")
    args = parser.parse_args()

    host = args.host or cfg.get("host") or DEFAULT_CONFIG["host"]
    port = int(args.port or cfg.get("port") or DEFAULT_CONFIG["port"])
    device = args.device or cfg.get("device") or DEFAULT_CONFIG["device"]

    registry = VoiceRegistry(args.voices)
    backend = TTSBackend(registry, args.initial_voice, device=device,
                         default_emotion=cfg.get("default_emotion"),
                         bert_base_path=cfg.get("bert_base_path"),
                         cnhuhbert_base_path=cfg.get("cnhuhbert_base_path"))
    app = create_app(backend, registry)

    print("=" * 60)
    print(f"  TTS 本地服务  http://{host}:{port}")
    print(f"  设备: {backend.device}  半精度: {backend.is_half}  引擎: {backend.config.version}")
    print(f"  默认情绪: {backend.default_emotion}（请求未指定时使用；可按音色在 voices.json 覆盖）")
    print(f"  接口: GET /health | GET /voices | POST /voices/{{id}}/activate | POST /tts")
    print("  队列: FIFO 顺序合成（请求按提交顺序逐条处理）")
    print(f"  配置: server_config.json{'（已加载）' if cfg else '（不存在，使用默认值）'}")
    print("=" * 60)
    uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == "__main__":
    main()