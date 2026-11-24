import os
import sys
import re
import io
import csv
import html
import json
import time
import uuid
import wave
import shutil
import inspect
import tempfile
import threading
import subprocess
from pathlib import Path
from urllib.parse import quote
from typing import Optional, List, Dict, Any, Tuple


def _register_cuda_dlls():
    if os.name != "nt":
        return
    try:
        site_dirs = [p for p in sys.path if "site-packages" in p.lower() and os.path.isdir(p)]
        for sp in site_dirs:
            nvidia_root = os.path.join(sp, "nvidia")
            if not os.path.isdir(nvidia_root):
                continue
            for root, dirs, files in os.walk(nvidia_root):
                if os.path.basename(root).lower() in ("bin", "lib"):
                    if any(f.lower().endswith(".dll") for f in files):
                        try:
                            os.add_dll_directory(root)
                        except Exception:
                            pass
                        os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


_register_cuda_dlls()

import httpx
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    import numpy as np
except Exception:
    np = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_BIN = shutil.which("ffmpeg") or ""

try:
    import pykakasi
    _KAKASI = pykakasi.kakasi()
except Exception:
    _KAKASI = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MEDIA_DIR = DATA_DIR / "media"
EXPORT_DIR = DATA_DIR / "exports"
JOBS_DIR = DATA_DIR / "jobs"
MODELS_DIR = DATA_DIR / "models"
VOCAB_FILE = DATA_DIR / "vocab.json"
STATIC_DIR = BASE_DIR / "static"

for _d in (MEDIA_DIR, EXPORT_DIR, JOBS_DIR, MODELS_DIR, STATIC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MinhSub Studio v5.6 — Engineered by Lê Đình Hoàng Minh")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
WHISPER_CACHE: Dict[str, Any] = {}
WHISPER_LOCK = threading.Lock()


# ===================== MULTI-LANGUAGE CATALOG =====================
LANGUAGES: List[Dict[str, str]] = [
    {"code": "auto", "name": "Tự phát hiện (Mọi ngôn ngữ)", "flag": "🌐"},
    {"code": "vi", "name": "Tiếng Việt", "flag": "🇻🇳"},
    {"code": "en", "name": "Tiếng Anh (English)", "flag": "🇬🇧"},
    {"code": "ja", "name": "Tiếng Nhật (日本語)", "flag": "🇯🇵"},
    {"code": "zh-CN", "name": "Tiếng Trung Giản thể (简体)", "flag": "🇨🇳"},
    {"code": "zh-TW", "name": "Tiếng Trung Phồn thể (繁體)", "flag": "🇹🇼"},
    {"code": "ko", "name": "Tiếng Hàn (한국어)", "flag": "🇰🇷"},
    {"code": "fr", "name": "Tiếng Pháp (Français)", "flag": "🇫🇷"},
    {"code": "de", "name": "Tiếng Đức (Deutsch)", "flag": "🇩🇪"},
    {"code": "es", "name": "Tiếng Tây Ban Nha (Español)", "flag": "🇪🇸"},
    {"code": "ru", "name": "Tiếng Nga (Русский)", "flag": "🇷🇺"},
    {"code": "th", "name": "Tiếng Thái (ไทย)", "flag": "🇹🇭"},
    {"code": "id", "name": "Tiếng Indonesia", "flag": "🇮🇩"},
    {"code": "pt", "name": "Tiếng Bồ Đào Nha", "flag": "🇵🇹"},
    {"code": "it", "name": "Tiếng Ý (Italiano)", "flag": "🇮🇹"},
    {"code": "ar", "name": "Tiếng Ả Rập (العربية)", "flag": "🇸🇦"},
    {"code": "hi", "name": "Tiếng Hindi (हिन्दी)", "flag": "🇮🇳"},
    {"code": "nl", "name": "Tiếng Hà Lan", "flag": "🇳🇱"},
    {"code": "pl", "name": "Tiếng Ba Lan", "flag": "🇵🇱"},
    {"code": "tr", "name": "Tiếng Thổ Nhĩ Kỳ", "flag": "🇹🇷"},
    {"code": "uk", "name": "Tiếng Ukraina", "flag": "🇺🇦"},
    {"code": "sv", "name": "Tiếng Thụy Điển", "flag": "🇸🇪"},
    {"code": "ms", "name": "Tiếng Malaysia", "flag": "🇲🇾"},
    {"code": "tl", "name": "Tiếng Philippines", "flag": "🇵🇭"},
    {"code": "lo", "name": "Tiếng Lào", "flag": "🇱🇦"},
    {"code": "km", "name": "Tiếng Khmer", "flag": "🇰🇭"},
]

LANG_NAME_MAP: Dict[str, str] = {x["code"]: x["name"] for x in LANGUAGES}
LANG_NAME_MAP["zh"] = "Tiếng Trung"
NO_SPACE_LANGS = {"ja", "zh", "zh-cn", "zh-tw", "th", "lo", "km", "my"}

POS_VI_MAP = {
    "noun": "Danh từ (n)", "verb": "Động từ (v)", "adjective": "Tính từ (adj)",
    "adverb": "Trạng từ (adv)", "pronoun": "Đại từ (pron)", "preposition": "Giới từ (prep)",
    "conjunction": "Liên từ (conj)", "interjection": "Thán từ (interj)", "exclamation": "Thán từ",
    "article": "Mạo từ", "determiner": "Hạn định từ", "auxiliary verb": "Trợ động từ",
    "abbreviation": "Từ viết tắt", "prefix": "Tiền tố", "suffix": "Hậu tố",
    "phrase": "Cụm từ", "idiom": "Thành ngữ",
}

STYLE_LABELS = {
    "auto": "Tự động theo giọng & ngữ cảnh",
    "couple": "Tình cảm (Anh – Em)",
    "family": "Gia đình (Con – Bố/Mẹ)",
    "casual": "Đời thường / Bạn bè (Mình – Cậu)",
    "anime": "Phim / Anime (Gần gũi tự nhiên)",
    "formal": "Công việc / TED (Tôi – Các bạn)",
}


def norm_whisper_lang(code: Optional[str]) -> Optional[str]:
    c = (code or "").strip()
    if not c or c.lower() == "auto":
        return None
    cl = c.lower()
    if cl in ("zh-cn", "zh-tw", "zh-hans", "zh-hant"):
        return "zh"
    if cl == "fil":
        return "tl"
    return cl.split("-")[0]


def norm_gtrans_lang(code: Optional[str], default: str = "auto") -> str:
    c = (code or "").strip()
    if not c:
        return default
    cl = c.lower()
    if cl == "auto":
        return "auto"
    if cl in ("zh", "zh-cn", "zh-hans"):
        return "zh-CN"
    if cl in ("zh-tw", "zh-hant"):
        return "zh-TW"
    if cl == "fil":
        return "tl"
    return cl


def lang_label(code: str) -> str:
    if not code:
        return "AUTO"
    return LANG_NAME_MAP.get(code, LANG_NAME_MAP.get(code.split("-")[0], code.upper()))


def safe_filename(name: str, fallback: str = "video") -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\r\n]+', "_", (name or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._")
    return cleaned[:70] if cleaned else fallback


# ===================== HARDWARE & WHISPER LOADER =====================
def detect_compute_device() -> str:
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_whisper_model(model_size: str, force_cpu: bool = False):
    from faster_whisper import WhisperModel

    device = "cpu" if force_cpu else detect_compute_device()
    compute_type = "float16" if device == "cuda" else "int8"
    cache_key = f"{model_size}:{device}:{compute_type}"

    with WHISPER_LOCK:
        if cache_key in WHISPER_CACHE:
            return WHISPER_CACHE[cache_key], device

        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            WHISPER_CACHE.clear()
            WHISPER_CACHE[cache_key] = model
            return model, device
        except Exception:
            if device == "cuda":
                fallback_key = f"{model_size}:cpu:int8"
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
                WHISPER_CACHE.clear()
                WHISPER_CACHE[fallback_key] = model
                return model, "cpu"
            raise


# ===================== AI VOICE GENDER ENGINE (ECAPA-TDNN + WAV2VEC2 ONNX) =====================
def _softmax_2(logits) -> Tuple[float, float]:
    arr = np.asarray(logits, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        return 0.5, 0.5
    m = float(np.max(arr[:2]))
    e0 = float(np.exp(arr[0] - m))
    e1 = float(np.exp(arr[1] - m))
    s = e0 + e1
    return (e0 / s, e1 / s) if s > 0 else (0.5, 0.5)


def _download_file_atomic(url: str, dest: Path, timeout: float = 240.0) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with httpx.Client(headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True, timeout=timeout) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=131072):
                        if chunk:
                            f.write(chunk)
        if tmp.exists() and tmp.stat().st_size > 1000:
            tmp.replace(dest)
            return True
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
    return False


class VoiceGenderEngine:
    """
    Nhận diện giới tính giọng nói & giọng hát bằng mô hình AI Deep Learning (ONNX Runtime):
    - Primary: ECAPA-TDNN (VoxCeleb2, 98.7% acc, 15.8 MB) — phân tích âm sắc cổ họng, miễn nhiễm nốt cao khi hát.
    - Verifier: Wav2Vec2-Base (Common Voice đa ngôn ngữ, 98.46% acc, 95.4 MB) — kiểm chứng các câu sát ranh giới.
    """

    ECAPA_ONNX_URL = "https://huggingface.co/Alice-Sabrina-Ivy/voice-gender-classifier-onnx-q8/resolve/main/onnx/model_quantized.onnx"
    ECAPA_CFG_URL = "https://huggingface.co/Alice-Sabrina-Ivy/voice-gender-classifier-onnx-q8/resolve/main/config.json"

    W2V2_ONNX_URL = "https://huggingface.co/prithivMLmods/Common-Voice-Gender-Detection-ONNX/resolve/main/onnx/model_quantized.onnx"
    W2V2_CFG_URL = "https://huggingface.co/prithivMLmods/Common-Voice-Gender-Detection-ONNX/resolve/main/config.json"

    def __init__(self):
        self._lock = threading.Lock()
        self.ecapa_sess = None
        self.ecapa_in = ""
        self.ecapa_male_idx = 0  # verified: {"0": "male", "1": "female"}

        self.w2v2_sess = None
        self.w2v2_in = ""
        self.w2v2_male_idx = 1   # verified: {"0": "female", "1": "male"}

        self._ecapa_tried = False
        self._w2v2_tried = False

    def _parse_male_idx(self, cfg_path: Path, default_idx: int) -> int:
        try:
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                id2label = cfg.get("id2label") or {}
                for k, v in id2label.items():
                    if str(v).strip().lower() in ("male", "man", "masculine"):
                        return int(k)
        except Exception:
            pass
        return default_idx

    def _make_session(self, onnx_path: Path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 2))
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

    def ensure_ecapa(self, status_cb=None) -> bool:
        if ort is None or np is None:
            return False
        with self._lock:
            if self.ecapa_sess is not None:
                return True
            if self._ecapa_tried:
                return False
            self._ecapa_tried = True

            onnx_path = MODELS_DIR / "ecapa_gender_q8.onnx"
            cfg_path = MODELS_DIR / "ecapa_gender_config.json"
            if not onnx_path.exists():
                if status_cb:
                    status_cb("Đang tải AI nhận diện giọng ECAPA-TDNN (15.8 MB, chỉ tải 1 lần)...")
                _download_file_atomic(self.ECAPA_CFG_URL, cfg_path, timeout=30.0)
                if not _download_file_atomic(self.ECAPA_ONNX_URL, onnx_path, timeout=180.0):
                    return False

            try:
                self.ecapa_male_idx = self._parse_male_idx(cfg_path, 0)
                self.ecapa_sess = self._make_session(onnx_path)
                self.ecapa_in = self.ecapa_sess.get_inputs()[0].name
                return True
            except Exception:
                self.ecapa_sess = None
                return False

    def ensure_w2v2(self, status_cb=None) -> bool:
        if ort is None or np is None:
            return False
        with self._lock:
            if self.w2v2_sess is not None:
                return True
            if self._w2v2_tried:
                return False
            self._w2v2_tried = True

            onnx_path = MODELS_DIR / "w2v2_gender_q8.onnx"
            cfg_path = MODELS_DIR / "w2v2_gender_config.json"
            if not onnx_path.exists():
                if status_cb:
                    status_cb("Đang tải AI kiểm chứng Wav2Vec2 (95 MB, chỉ tải 1 lần)...")
                _download_file_atomic(self.W2V2_CFG_URL, cfg_path, timeout=30.0)
                if not _download_file_atomic(self.W2V2_ONNX_URL, onnx_path, timeout=300.0):
                    return False

            try:
                self.w2v2_male_idx = self._parse_male_idx(cfg_path, 1)
                self.w2v2_sess = self._make_session(onnx_path)
                self.w2v2_in = self.w2v2_sess.get_inputs()[0].name
                return True
            except Exception:
                self.w2v2_sess = None
                return False

    def predict_ecapa_male_prob(self, clip: "np.ndarray") -> Optional[float]:
        if self.ecapa_sess is None or clip.size < 4000:
            return None
        try:
            # ECAPA-TDNN yêu cầu raw float32 KHÔNG z-score normalize (do_normalize=False)
            wav = clip.astype(np.float32, copy=True)
            peak = float(np.max(np.abs(wav)))
            if peak > 1e-5 and peak < 0.08:
                wav = wav * (0.45 / peak)
            elif peak > 1.0:
                wav = wav / peak
            inp = wav[np.newaxis, :]
            out = self.ecapa_sess.run(None, {self.ecapa_in: inp})[0]
            probs = _softmax_2(out[0])
            return probs[self.ecapa_male_idx]
        except Exception:
            return None

    def predict_w2v2_male_prob(self, clip: "np.ndarray") -> Optional[float]:
        if self.w2v2_sess is None or clip.size < 4000:
            return None
        try:
            # Wav2Vec2 yêu cầu chuẩn hóa zero-mean unit-variance (do_normalize=True)
            wav = clip.astype(np.float32, copy=True)
            mean = float(np.mean(wav))
            std = float(np.sqrt(np.var(wav) + 1e-7))
            wav = (wav - mean) / std
            inp = wav[np.newaxis, :]
            out = self.w2v2_sess.run(None, {self.w2v2_in: inp})[0]
            probs = _softmax_2(out[0])
            return probs[self.w2v2_male_idx]
        except Exception:
            return None

    def status_label(self) -> str:
        if ort is None or np is None:
            return "Chưa có ONNX"
        has_e = (MODELS_DIR / "ecapa_gender_q8.onnx").exists() or self.ecapa_sess is not None
        has_w = (MODELS_DIR / "w2v2_gender_q8.onnx").exists() or self.w2v2_sess is not None
        if has_e and has_w:
            return "ECAPA + Wav2Vec2"
        if has_e:
            return "ECAPA-TDNN AI"
        return "AI Sẵn sàng (Auto-tải)"


GENDER_AI = VoiceGenderEngine()


def _extract_analysis_clip(audio_np: "np.ndarray", start: float, end: float, sr: int = 16000) -> "np.ndarray":
    """Lấy đoạn audio tối ưu dài 1.8s – 5.0s quanh câu thoại để mạng AI đọc đủ âm sắc giọng."""
    dur = max(0.2, end - start)
    center = (start + end) * 0.5
    win_dur = min(5.0, max(1.8, dur))
    s = max(0.0, center - win_dur * 0.5)
    e = min(audio_np.size / float(sr), s + win_dur)
    i0 = max(0, int(s * sr))
    i1 = min(audio_np.size, int(e * sr))
    return audio_np[i0:i1]


def annotate_cues_gender(
    audio_np,
    cues: List[Dict[str, Any]],
    music_mode: bool = False,
    voice_override: str = "auto",
    status_cb=None,
) -> List[Dict[str, Any]]:
    if not cues:
        return cues

    vo = (voice_override or "auto").strip().lower()
    if vo in ("male", "female"):
        for c in cues:
            c["gender"] = vo
            c["gender_conf"] = 1.0
            c["gender_src"] = "Khóa thủ công"
        return cues

    if np is None or not isinstance(audio_np, np.ndarray) or audio_np.size < 4000:
        for c in cues:
            c.setdefault("gender", "unknown")
            c.setdefault("gender_conf", 0.5)
            c.setdefault("gender_src", "none")
        return cues

    has_ecapa = GENDER_AI.ensure_ecapa(status_cb=status_cb)
    has_w2v2 = GENDER_AI.ensure_w2v2(status_cb=status_cb)

    if not has_ecapa and not has_w2v2:
        for c in cues:
            c["gender"] = "unknown"
            c["gender_conf"] = 0.5
            c["gender_src"] = "offline"
        return cues

    # Bước 1: Chấm điểm ECAPA-TDNN siêu nhanh cho toàn bộ câu
    ecapa_probs: List[Optional[float]] = []
    clips: List["np.ndarray"] = []
    for c in cues:
        clip = _extract_analysis_clip(audio_np, float(c["start"]), float(c["end"]))
        clips.append(clip)
        p_e = GENDER_AI.predict_ecapa_male_prob(clip) if has_ecapa else None
        ecapa_probs.append(p_e)

    # Bước 2: Chạy Wav2Vec2 kiểm chứng (toàn bộ nếu <= 45 câu; hoặc câu phân vân + mẫu đại diện nếu dài)
    n = len(cues)
    w2v2_indices = set()
    if has_w2v2:
        if n <= 45:
            w2v2_indices = set(range(n))
        else:
            # Kiểm chứng tất cả câu phân vân (0.18..0.82) + tối đa 18 câu trải đều để hiệu chuẩn bài hát/video
            for i, pe in enumerate(ecapa_probs):
                if pe is None or (0.18 <= pe <= 0.82):
                    w2v2_indices.add(i)
            step = max(1, n // 18)
            for i in range(0, n, step):
                w2v2_indices.add(i)

    final_male_probs: List[float] = []
    sources: List[str] = []

    for i in range(n):
        pe = ecapa_probs[i]
        pw = GENDER_AI.predict_w2v2_male_prob(clips[i]) if (i in w2v2_indices) else None

        if pe is not None and pw is not None:
            # Trong nhạc nền dày, ECAPA (VoxCeleb2) chịu nhiễu tốt hơn Wav2Vec2 nên lấy trọng số 0.65/0.35
            w_e = 0.65 if music_mode else 0.55
            p = w_e * pe + (1.0 - w_e) * pw
            src = "ECAPA+W2V2"
        elif pe is not None:
            p = pe
            src = "ECAPA-TDNN"
        elif pw is not None:
            p = pw
            src = "Wav2Vec2"
        else:
            p = 0.5
            src = "unknown"

        final_male_probs.append(float(p))
        sources.append(src)

    # Bước 3: Làm mượt theo ngữ cảnh ca sĩ / người nói (chống nhảy nhãn sai ở câu điệp khúc cao)
    probs = list(final_male_probs)
    if n >= 3:
        # Làm mượt cửa sổ trượt 3 câu cho các câu có điểm số gần ngưỡng
        smoothed = list(probs)
        for i in range(n):
            if 0.28 <= probs[i] <= 0.72:
                lo = max(0, i - 1)
                hi = min(n, i + 2)
                smoothed[i] = float(np.mean(probs[lo:hi]))
        probs = smoothed

    # Kiểm tra bài hát solo (1 ca sĩ chính): nếu >72% thời lượng nghiêng về 1 giới tính và không có đoạn song ca rõ nét
    weights = [max(0.4, float(c["end"]) - float(c["start"])) for c in cues]
    total_w = sum(weights) or 1.0
    male_w = sum(w for p, w in zip(probs, weights) if p >= 0.52)
    female_w = sum(w for p, w in zip(probs, weights) if p <= 0.48)
    male_ratio = male_w / total_w
    female_ratio = female_w / total_w

    # Phát hiện xem có chuỗi song ca thật (>= 2 câu liên tiếp cực chắc chắn của giới tính còn lại) hay không
    def _has_strong_counter_run(target_male: bool) -> bool:
        streak = 0
        for p in probs:
            is_strong_other = (p <= 0.16) if target_male else (p >= 0.84)
            if is_strong_other:
                streak += 1
                if streak >= 2:
                    return True
            else:
                streak = 0
        return False

    dominant: Optional[str] = None
    if (music_mode and male_ratio >= 0.68 and not _has_strong_counter_run(True)) or (male_ratio >= 0.82 and not _has_strong_counter_run(True)):
        dominant = "male"
    elif (music_mode and female_ratio >= 0.68 and not _has_strong_counter_run(False)) or (female_ratio >= 0.82 and not _has_strong_counter_run(False)):
        dominant = "female"

    for i, c in enumerate(cues):
        p = probs[i]
        if dominant == "male" and p > 0.18:
            c["gender"] = "male"
            c["gender_conf"] = round(max(p, 0.72), 2)
            c["gender_src"] = sources[i]
        elif dominant == "female" and p < 0.82:
            c["gender"] = "female"
            c["gender_conf"] = round(max(1.0 - p, 0.72), 2)
            c["gender_src"] = sources[i]
        else:
            if p >= 0.50:
                c["gender"] = "male"
                c["gender_conf"] = round(p, 2)
            else:
                c["gender"] = "female"
                c["gender_conf"] = round(1.0 - p, 2)
            c["gender_src"] = sources[i]

    return cues


# ===================== FFMPEG AUDIO DECODER & SPEECH TRIM =====================
def decode_audio_ffmpeg_np(media_path: str, vocal_boost: bool = False):
    if not FFMPEG_BIN or np is None:
        return media_path

    tmp_wav = None
    try:
        fd, tmp_wav = tempfile.mkstemp(prefix="ms_audio_", suffix=".wav")
        os.close(fd)

        cmd = [FFMPEG_BIN, "-y", "-i", media_path, "-vn", "-ac", "1", "-ar", "16000"]
        if vocal_boost:
            # Giữ nguyên dải trầm giọng nam (từ 75 Hz thay vì 160 Hz như bản cũ)
            cmd += [
                "-af",
                "highpass=f=75,lowpass=f=4800,acompressor=threshold=-20dB:ratio=2.5:attack=5:release=60,dynaudnorm=f=150:g=15",
            ]
        cmd += ["-f", "wav", "-acodec", "pcm_s16le", tmp_wav]

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        proc = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            creationflags=creationflags, timeout=300,
        )
        if proc.returncode != 0 or not os.path.exists(tmp_wav) or os.path.getsize(tmp_wav) < 100:
            return media_path

        with wave.open(tmp_wav, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_np if audio_np.size > 0 else media_path
    except Exception:
        return media_path
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass


def trim_cues_to_speech(
    audio_np,
    cues: List[Dict[str, Any]],
    music_mode: bool = False,
    sr: int = 16000,
) -> List[Dict[str, Any]]:
    if not cues or np is None or not isinstance(audio_np, np.ndarray) or audio_np.size < sr // 2:
        return cues

    frame_sec = 0.04
    frame_len = int(sr * frame_sec)
    n_total = audio_np.size // frame_len
    if n_total < 5:
        return cues

    sample_step = max(1, n_total // 1500)
    sample_rms = [
        float(np.sqrt(np.mean(audio_np[k * frame_len: (k + 1) * frame_len] ** 2)))
        for k in range(0, n_total, sample_step)
    ]
    noise_floor = float(np.percentile(sample_rms, 15)) if sample_rms else 0.008

    trimmed: List[Dict[str, Any]] = []
    for c in cues:
        start, end = float(c["start"]), float(c["end"])
        if end <= start:
            continue

        i0, i1 = max(0, int(start * sr)), min(audio_np.size, int(end * sr))
        seg = audio_np[i0:i1]
        nf = seg.size // frame_len
        if nf < 2:
            trimmed.append(c)
            continue

        frames = seg[: nf * frame_len].reshape(nf, frame_len)
        rms = np.sqrt(np.mean(frames * frames, axis=1))
        peak = float(np.max(rms))

        if not music_mode and peak < max(0.007, noise_floor * 1.35):
            continue

        thr = max(max(0.011, min(0.045, noise_floor * 2.1)), peak * (0.16 if not music_mode else 0.14))
        active = np.where(rms >= thr)[0]
        if active.size == 0:
            if music_mode:
                trimmed.append(c)
            continue

        new_start = start + max(0.0, int(active[0]) * frame_sec - 0.05)
        new_end = start + min(end - start, (int(active[-1]) + 1) * frame_sec + 0.10)

        txt_len = max(1, len((c.get("text") or "").strip()))
        is_cjk = (c.get("lang") or "").lower() in NO_SPACE_LANGS
        max_reasonable = max(2.2, (txt_len * 0.45 if is_cjk else txt_len * 0.14) + 1.4)
        if not music_mode and (new_end - new_start) > max_reasonable:
            new_end = new_start + max_reasonable

        if new_end - new_start < 0.30:
            new_end = min(end, new_start + 0.35)

        c["start"] = round(new_start, 2)
        c["end"] = round(max(new_start + 0.25, new_end), 2)
        trimmed.append(c)

    for i in range(len(trimmed) - 1):
        if trimmed[i]["end"] > trimmed[i + 1]["start"]:
            trimmed[i]["end"] = round(max(trimmed[i]["start"] + 0.2, trimmed[i + 1]["start"] - 0.02), 2)
    return trimmed or cues


# ===================== LANGUAGE & READING HELPERS =====================
def has_kanji(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9faf]", text or ""))


def has_kana(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff]", text or ""))


def get_japanese_reading(text: str) -> str:
    if not text or not _KAKASI or not (has_kanji(text) or has_kana(text)):
        return ""
    try:
        items = _KAKASI.convert(text)
        hira = "".join(it.get("hira", "") for it in items).strip()
        hepburn = " ".join(it.get("hepburn", "").strip() for it in items if it.get("hepburn", "").strip()).strip()
        if hira and hira != text.strip():
            return f"{hira} ({hepburn})" if hepburn else hira
        return hepburn
    except Exception:
        return ""


def detect_text_lang(text: str) -> str:
    if not text:
        return "en"
    if has_kana(text):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    if has_kanji(text):
        return "zh-CN"
    if re.search(r"[\u0e00-\u0e7f]", text):
        return "th"
    if re.search(r"[\u0400-\u04ff]", text):
        return "ru"
    if re.search(r"[\u0600-\u06ff]", text):
        return "ar"
    return "auto"


# ===================== VIETNAMESE CONTEXT & PRONOUN NATURALIZER =====================
RE_MOM = re.compile(r"\b(mom|mommy|mum|mummy|mother|mama|ma'am|mẹ|má)\b|お母さん|ママ|おかあさん|妈妈|妈|엄마|어머니", re.IGNORECASE)
RE_DAD = re.compile(r"\b(dad|daddy|father|papa|bố|ba|cha)\b|お父さん|パパ|おとうさん|爸爸|爸|아빠|아버지", re.IGNORECASE)
RE_PARENT_TO_CHILD = re.compile(r"\b(my son|my daughter|my child|sweetheart|kiddo|con trai|con gái|con yêu)\b|息子|娘|儿子|女儿", re.IGNORECASE)
RE_ROMANTIC = re.compile(
    r"\b(love you|miss you|heart|honey|darling|babe|baby|marry me|kiss me|my love|fell in love|hold you|without you|yêu bạn|nhớ bạn|thích bạn|trái tim|ôm bạn|lấy tôi|hôn tôi|người yêu)\b"
    r"|愛してる|大好き|好きだ|好きよ|会いたい|恋|君|きみ|キミ|結婚|사랑해|보고 싶어|너를|我爱你|想你|亲爱的",
    re.IGNORECASE,
)
RE_PLURAL_YOU = re.compile(r"\b(you guys|you all|everyone|everybody|folks|ladies and gentlemen|all of you)\b|皆さん|みなさん|大家|여러분", re.IGNORECASE)


def _match_case(orig: str, repl: str) -> str:
    if not orig or not repl:
        return repl
    if orig.isupper() and len(orig) > 1:
        return repl.upper()
    if orig[0].isupper():
        return repl[0].upper() + repl[1:]
    return repl


def _replace_vn_pronouns(text: str, self_p: Optional[str], other_p: Optional[str]) -> str:
    out = text
    if self_p:
        out = re.sub(
            r"(?<!chúng\s)(?<!bọn\s)(?<!tụi\s)(?<!cái\s)\btôi\b(?!\s+luyện|\s+tớ)",
            lambda m: _match_case(m.group(0), self_p), out, flags=re.IGNORECASE,
        )
    if other_p:
        out = re.sub(
            r"(?<!các\s)(?<!người\s)(?<!tình\s)(?<!kết\s)(?<!làm\s)(?<!đôi\s)(?<!bầu\s)\bbạn\b"
            r"(?!\s+bè|\s+thân|\s+học|\s+gái|\s+trai|\s+đời|\s+đồng|\s+cùng|\s+hữu|\s+nhậu|\s+diễn|\s+nhảy|\s+lòng)",
            lambda m: _match_case(m.group(0), other_p), out, flags=re.IGNORECASE,
        )
    return out


def naturalize_vietnamese_cues(cues: List[Dict[str, Any]], style: str = "auto", music_mode: bool = False) -> List[Dict[str, Any]]:
    if not cues:
        return cues

    style = (style or "auto").lower()
    genders = [c.get("gender", "unknown") for c in cues]
    has_both_genders = ("male" in genders) and ("female" in genders)
    total_romantic = sum(
        1 for c in cues
        if RE_ROMANTIC.search(c.get("text", "")) or RE_ROMANTIC.search(c.get("raw_vi", c.get("vi", "")))
    )
    is_love_song = (music_mode and total_romantic >= 1) or (total_romantic >= 2)

    for i, c in enumerate(cues):
        raw_vi = (c.get("raw_vi") or c.get("vi") or c.get("translated") or "").strip()
        c["raw_vi"] = raw_vi
        if not raw_vi:
            continue

        orig = c.get("text", "")
        gender = c.get("gender", "unknown")

        win_slice = cues[max(0, i - 2): min(len(cues), i + 3)]
        win_orig = " ".join(x.get("text", "") for x in win_slice)
        win_vi = " ".join(x.get("raw_vi", x.get("vi", "")) for x in win_slice)
        win_genders = {x.get("gender", "unknown") for x in win_slice}
        local_pair = ("male" in win_genders) and ("female" in win_genders)

        cue_to_mom = bool(RE_MOM.search(orig) or RE_MOM.search(raw_vi))
        cue_to_dad = bool(RE_DAD.search(orig) or RE_DAD.search(raw_vi))
        win_to_mom = bool(RE_MOM.search(win_orig) or RE_MOM.search(win_vi))
        win_to_dad = bool(RE_DAD.search(win_orig) or RE_DAD.search(win_vi))
        is_parent_line = bool(RE_PARENT_TO_CHILD.search(orig) or RE_PARENT_TO_CHILD.search(raw_vi))
        is_romantic_line = bool(RE_ROMANTIC.search(orig) or RE_ROMANTIC.search(raw_vi))
        is_romantic_win = bool(RE_ROMANTIC.search(win_orig) or RE_ROMANTIC.search(win_vi))
        is_plural_you = bool(RE_PLURAL_YOU.search(orig))

        self_p: Optional[str] = None
        other_p: Optional[str] = None
        vi = raw_vi

        if is_parent_line and style in ("auto", "family", "anime"):
            self_p = "bố" if gender == "male" else ("mẹ" if gender == "female" else None)
            other_p = "con"
        elif (cue_to_mom or (style == "family" and win_to_mom)) and style != "formal":
            self_p, other_p = "con", "mẹ"
            vi = re.sub(r"^(xin\s+)?chào\s+mẹ\b", "Con chào mẹ", vi, flags=re.IGNORECASE)
            vi = re.sub(r"^chào\s+buổi\s+sáng[,\s]+mẹ\b", "Con chào buổi sáng mẹ", vi, flags=re.IGNORECASE)
            vi = re.sub(r"^tạm\s+biệt[,\s]+mẹ\b", "Con tạm biệt mẹ", vi, flags=re.IGNORECASE)
            vi = re.sub(r"^cảm\s+ơn[,\s]+mẹ\b", "Con cảm ơn mẹ", vi, flags=re.IGNORECASE)
        elif (cue_to_dad or (style == "family" and win_to_dad)) and style != "formal":
            self_p, other_p = "con", "bố"
            vi = re.sub(r"^(xin\s+)?chào\s+(bố|ba|cha)\b", "Con chào bố", vi, flags=re.IGNORECASE)
            vi = re.sub(r"^chào\s+buổi\s+sáng[,\s]+(bố|ba|cha)\b", "Con chào buổi sáng bố", vi, flags=re.IGNORECASE)
            vi = re.sub(r"^tạm\s+biệt[,\s]+(bố|ba|cha)\b", "Con tạm biệt bố", vi, flags=re.IGNORECASE)
            vi = re.sub(r"^cảm\s+ơn[,\s]+(bố|ba|cha)\b", "Con cảm ơn bố", vi, flags=re.IGNORECASE)
        elif style == "couple" or (
            style in ("auto", "anime")
            and (is_romantic_line or is_love_song or (is_romantic_win and (local_pair or has_both_genders)))
        ):
            if gender == "male":
                self_p, other_p = "anh", "em"
            elif gender == "female":
                self_p, other_p = "em", "anh"
            elif is_romantic_line:
                self_p, other_p = "anh", "em"
        elif style == "family":
            self_p = "con"
            other_p = "mẹ" if win_to_mom else ("bố" if win_to_dad else None)
        elif style == "casual":
            self_p, other_p = ("anh", "em") if (local_pair and gender == "male") else (
                ("em", "anh") if (local_pair and gender == "female") else ("mình", "cậu")
            )
        elif style == "anime":
            self_p, other_p = ("anh", "em") if (local_pair and gender == "male") else (
                ("em", "anh") if (local_pair and gender == "female") else ("tớ", "cậu")
            )
        elif style == "formal":
            self_p = "tôi"
            other_p = "các bạn" if is_plural_you else None
        elif style == "auto":
            if is_plural_you:
                other_p = "các bạn"
            elif music_mode and gender in ("male", "female"):
                self_p, other_p = ("anh", "em") if gender == "male" else ("em", "anh")

        if is_plural_you and other_p in ("em", "anh", "cậu"):
            other_p = "mọi người"

        vi = _replace_vn_pronouns(vi, self_p, other_p)
        vi = re.sub(r"\b(Anh|Em|Con|Mình|Tớ)\s+(anh|em|con|mình|tớ)\b", r"\1", vi)
        vi = re.sub(r"\s{2,}", " ", vi).strip()
        c["vi"] = vi
        c["translated"] = vi

    return cues


# ===================== CONTEXT-AWARE TRANSLATOR =====================
def _google_translate_request(
    client: httpx.Client, text: str, sl: str = "auto", tl: str = "vi", include_dict: bool = False
) -> Dict[str, Any]:
    clean = (text or "").strip()
    if not clean:
        return {"translated": "", "reading": "", "dict": [], "detected_lang": ""}

    params = [
        ("client", "gtx"), ("sl", norm_gtrans_lang(sl, "auto")),
        ("tl", norm_gtrans_lang(tl, "vi")), ("dt", "t"), ("dt", "rm"),
    ]
    if include_dict:
        params.append(("dt", "bd"))

    url = "https://translate.googleapis.com/translate_a/single"
    resp = None
    if len(clean) <= 600:
        try:
            resp = client.get(url, params=params + [("q", clean)], timeout=20.0)
            resp.raise_for_status()
        except Exception:
            resp = None

    if resp is None:
        resp = client.post(url, params=params, data={"q": clean}, timeout=25.0)
        resp.raise_for_status()

    data = resp.json()
    translated, reading = "", ""
    if isinstance(data, list) and data and isinstance(data[0], list):
        translated = "".join((seg[0] or "") for seg in data[0] if isinstance(seg, list) and seg and seg[0])
        for seg in reversed(data[0]):
            if isinstance(seg, list) and len(seg) > 3 and isinstance(seg[3], str) and seg[3].strip():
                reading = seg[3].strip()
                break

    detected = data[2] if isinstance(data, list) and len(data) > 2 and isinstance(data[2], str) else ""
    dict_groups = []
    if include_dict and isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
        for g in data[1]:
            if isinstance(g, list) and len(g) >= 2 and isinstance(g[0], str) and isinstance(g[1], list):
                raw_pos = g[0].strip().lower()
                dict_groups.append({
                    "pos": POS_VI_MAP.get(raw_pos, g[0]),
                    "pos_raw": raw_pos,
                    "terms": [str(x) for x in g[1][:6]],
                })

    return {"translated": translated.strip(), "reading": reading.strip(), "dict": dict_groups, "detected_lang": detected}


def _fallback_translate_mymemory(client: httpx.Client, text: str, sl: str, tl: str) -> str:
    src = norm_gtrans_lang(sl, "en")
    if src == "auto":
        src = "en"
    resp = client.get(
        "https://api.mymemory.translated.net/get",
        params={"q": text[:450], "langpair": f"{src}|{norm_gtrans_lang(tl, 'vi')}"},
        timeout=15.0,
    )
    resp.raise_for_status()
    return ((resp.json().get("responseData") or {}).get("translatedText") or "").strip()


def _translate_block_aligned(client: httpx.Client, lines: List[str], sl: str, tl: str, prev_context: str = "") -> List[str]:
    n = len(lines)
    if n == 0:
        return []
    try:
        res = _google_translate_request(client, "\n".join(lines), sl=sl, tl=tl)
        out_lines = [ln.strip() for ln in res["translated"].split("\n")]
        if len(out_lines) != n:
            non_empty = [ln for ln in out_lines if ln]
            if len(non_empty) == n:
                out_lines = non_empty
        if len(out_lines) == n:
            return out_lines
    except Exception:
        pass

    if n > 1:
        mid = n // 2
        return (
            _translate_block_aligned(client, lines[:mid], sl, tl, prev_context=prev_context)
            + _translate_block_aligned(client, lines[mid:], sl, tl, prev_context=lines[mid - 1])
        )

    txt = lines[0]
    if prev_context:
        try:
            res = _google_translate_request(client, f"{prev_context}\n{txt}", sl=sl, tl=tl)
            parts = [p.strip() for p in res["translated"].split("\n") if p.strip()]
            if len(parts) >= 2:
                return [parts[-1]]
        except Exception:
            pass

    for fn in (
        lambda: _google_translate_request(client, txt, sl=sl, tl=tl)["translated"],
        lambda: _fallback_translate_mymemory(client, txt, sl, tl),
    ):
        try:
            out = fn()
            if out:
                return [out]
        except Exception:
            pass
    return [txt]


def _translate_batch_with_llm(
    client: httpx.Client, batch: List[Dict[str, Any]], tl: str, style: str,
    llm_url: str, llm_model: str, llm_key: str, prev_lines: List[str]
) -> Optional[List[str]]:
    endpoint = llm_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"

    numbered = []
    for idx, c in enumerate(batch, 1):
        g = c.get("gender", "unknown")
        gtag = "[Giọng Nam]" if g == "male" else ("[Giọng Nữ]" if g == "female" else "[Chưa rõ]")
        numbered.append(f"{idx}. {gtag} {c['text']}")

    ctx = ("Ngữ cảnh trước:\n" + "\n".join(prev_lines[-2:]) + "\n\n") if prev_lines else ""
    sys_prompt = (
        f"Bạn là biên dịch viên phụ đề phim/nhạc chuyên nghiệp sang {lang_label(tl)}.\n"
        f"Phong cách: {STYLE_LABELS.get(style, STYLE_LABELS['auto'])}.\n"
        "- Chọn đại từ xưng hô tiếng Việt cực tự nhiên dựa trên nhãn [Giọng Nam]/[Giọng Nữ] và mạch cảm xúc.\n"
        "- Chỉ trả về JSON array đúng số dòng, không giải thích thêm."
    )
    headers = {"Content-Type": "application/json"}
    if llm_key:
        headers["Authorization"] = f"Bearer {llm_key}"

    try:
        resp = client.post(
            endpoint,
            headers=headers,
            json={
                "model": llm_model,
                "temperature": 0.35,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": ctx + "Dịch:\n" + "\n".join(numbered)},
                ],
            },
            timeout=45.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        m = re.search(r"\[.*\]", content, flags=re.DOTALL)
        if m:
            arr = json.loads(m.group(0))
            if isinstance(arr, list) and len(arr) == len(batch):
                return [str(x).strip() for x in arr]
    except Exception:
        return None
    return None


def translate_cues_context(
    cues: List[Dict[str, Any]], sl: str, tl: str = "vi", style: str = "auto",
    music_mode: bool = False, llm_url: str = "", llm_model: str = "", llm_key: str = "", progress_cb=None
) -> List[Dict[str, Any]]:
    if not cues:
        return cues

    src = norm_gtrans_lang(sl, "auto")
    tgt = norm_gtrans_lang(tl, "vi")

    for c in cues:
        c["text"] = re.sub(r"[\r\n]+", " ", (c.get("text") or "")).strip()
        c["reading"] = get_japanese_reading(c["text"]) if (c.get("lang") == "ja" or has_kana(c["text"])) else c.get("reading", "")
        c.setdefault("gender", "unknown")

    if src != "auto" and src.split("-")[0].lower() == tgt.split("-")[0].lower():
        for c in cues:
            c["raw_vi"] = c["vi"] = c["translated"] = c["text"]
        if progress_cb:
            progress_cb(100, len(cues), len(cues))
        return cues

    use_llm = bool((llm_url or "").strip() and (llm_model or "").strip())
    batch_limit = 10 if use_llm else 7
    char_limit = 900 if use_llm else 380

    batches: List[List[Dict[str, Any]]] = []
    cur_batch: List[Dict[str, Any]] = []
    cur_chars = 0
    for c in cues:
        ln = c["text"]
        if cur_batch and (len(cur_batch) >= batch_limit or cur_chars + len(ln) > char_limit):
            batches.append(cur_batch)
            cur_batch, cur_chars = [], 0
        cur_batch.append(c)
        cur_chars += len(ln) + 1
    if cur_batch:
        batches.append(cur_batch)

    total, done_cnt, prev_tail = len(cues), 0, ""
    prev_lines: List[str] = []
    used_llm_all = use_llm

    with httpx.Client(headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}) as client:
        for batch in batches:
            lines = [c["text"] for c in batch]
            translated_lines = None
            if use_llm:
                translated_lines = _translate_batch_with_llm(
                    client, batch, tl=tgt, style=style,
                    llm_url=llm_url.strip(), llm_model=llm_model.strip(), llm_key=(llm_key or "").strip(),
                    prev_lines=prev_lines,
                )
                if translated_lines is None:
                    used_llm_all = False

            if translated_lines is None:
                translated_lines = _translate_block_aligned(client, lines, sl=src, tl=tgt, prev_context=prev_tail)

            for idx, tr_txt in enumerate(translated_lines):
                batch[idx]["raw_vi"] = batch[idx]["vi"] = batch[idx]["translated"] = tr_txt

            prev_tail = lines[-1] if lines else prev_tail
            prev_lines.extend(lines)
            done_cnt += len(batch)
            if progress_cb:
                progress_cb(int((done_cnt / total) * 100), done_cnt, total)
            if not use_llm:
                time.sleep(0.08)

    if tgt.lower().startswith("vi") and not used_llm_all:
        naturalize_vietnamese_cues(cues, style=style, music_mode=music_mode)
    return cues


# ===================== SEGMENT MERGING & WORD-LEVEL SPLIT =====================
HALLUCINATION_PHRASES = (
    "ご視聴ありがとうございました", "ご視聴ありがとう", "チャンネル登録", "高評価",
    "thank you for watching", "thanks for watching", "please subscribe",
    "subtitles by", "amara.org", "mbc 뉴스",
)


def merge_fragmented_segments(raw_segments: List[Dict[str, Any]], lang: str) -> List[Dict[str, Any]]:
    if not raw_segments:
        return []

    lang_l = (lang or "en").lower()
    is_no_space = lang_l in NO_SPACE_LANGS
    max_chars = 44 if is_no_space else 105
    joiner = "" if is_no_space else " "
    end_punct = (".", "!", "?", "。", "！", "？", "…")

    merged: List[Dict[str, Any]] = []
    buf = None
    repeat_streak, last_raw = 0, ""

    for seg in raw_segments:
        txt = re.sub(r"\s+", " ", (seg.get("text") or "")).strip()
        if not txt:
            continue
        if txt == last_raw:
            repeat_streak += 1
            if repeat_streak >= 2:
                continue
        else:
            repeat_streak, last_raw = 0, txt

        start = round(float(seg["start"]), 2)
        end = round(max(start + 0.30, float(seg["end"])), 2)

        if buf is None:
            buf = {
                "start": start, "end": end, "text": txt, "lang": lang_l,
                "reading": "", "vi": "", "raw_vi": "", "translated": "",
                "gender": "unknown", "gender_conf": 0.5, "gender_src": "",
            }
            continue

        gap = start - buf["end"]
        combined_text = (buf["text"] + joiner + txt).strip()
        combined_dur = end - buf["start"]
        prev_ended = buf["text"].endswith(end_punct)

        if (not prev_ended) and gap <= 0.40 and combined_dur <= 5.8 and len(combined_text) <= max_chars:
            buf["end"] = end
            buf["text"] = combined_text
        else:
            merged.append(buf)
            buf = {
                "start": start, "end": end, "text": txt, "lang": lang_l,
                "reading": "", "vi": "", "raw_vi": "", "translated": "",
                "gender": "unknown", "gender_conf": 0.5, "gender_src": "",
            }

    if buf is not None:
        merged.append(buf)

    for i in range(len(merged) - 1):
        if merged[i]["end"] > merged[i + 1]["start"]:
            merged[i]["end"] = round(max(merged[i]["start"] + 0.25, merged[i + 1]["start"] - 0.02), 2)
    return merged


def _build_transcribe_kwargs(model, lang_arg: Optional[str], music_mode: bool) -> Dict[str, Any]:
    sig_params = inspect.signature(model.transcribe).parameters
    kwargs: Dict[str, Any] = {
        "language": lang_arg, "task": "transcribe", "beam_size": 5, "best_of": 5,
        "temperature": [0.0, 0.2, 0.4, 0.6], "condition_on_previous_text": False,
        "word_timestamps": True, "vad_filter": not music_mode,
    }
    if not music_mode:
        kwargs["vad_parameters"] = {"min_silence_duration_ms": 320, "speech_pad_ms": 180, "threshold": 0.35}
        kwargs["no_speech_threshold"] = 0.72
        kwargs["log_prob_threshold"] = -1.2
    else:
        kwargs["no_speech_threshold"] = 0.94
        kwargs["log_prob_threshold"] = -2.0
        kwargs["compression_ratio_threshold"] = 2.6

    if "repetition_penalty" in sig_params:
        kwargs["repetition_penalty"] = 1.1
    if "no_repeat_ngram_size" in sig_params:
        kwargs["no_repeat_ngram_size"] = 3
    return {k: v for k, v in kwargs.items() if k in sig_params}


def _split_segment_by_word_pauses(seg, pause_gap: float = 0.70) -> List[Dict[str, Any]]:
    words = getattr(seg, "words", None) or []
    valid = [
        w for w in words
        if (getattr(w, "word", "") or "").strip() and getattr(w, "start", None) is not None and getattr(w, "end", None) is not None
    ]
    if not valid:
        return [{"start": round(float(seg.start), 2), "end": round(float(seg.end), 2), "text": (seg.text or "").strip()}]

    chunks, cur = [], [valid[0]]
    for w in valid[1:]:
        if float(w.start) - float(cur[-1].end) >= pause_gap:
            txt = "".join(x.word for x in cur).strip()
            if txt:
                chunks.append({
                    "start": round(float(cur[0].start), 2),
                    "end": round(max(float(cur[0].start) + 0.25, float(cur[-1].end)), 2),
                    "text": txt,
                })
            cur = [w]
        else:
            cur.append(w)
    if cur:
        txt = "".join(x.word for x in cur).strip()
        if txt:
            chunks.append({
                "start": round(float(cur[0].start), 2),
                "end": round(max(float(cur[0].start) + 0.25, float(cur[-1].end)), 2),
                "text": txt,
            })
    return chunks


def _collect_whisper_segments(segments_gen, info, job_id: str, used_device: str, source_lang: str, music_mode: bool):
    detected_lang = (getattr(info, "language", "") or norm_whisper_lang(source_lang) or "en").lower()
    total_dur = float(getattr(info, "duration", 0) or 0)
    raw_segments: List[Dict[str, Any]] = []

    for seg in segments_gen:
        txt = (seg.text or "").strip()
        if not txt:
            continue
        no_speech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
        avg_logprob = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
        low_txt = txt.lower().strip(" .!。！♪♫")
        if not low_txt:
            continue
        if any(p in low_txt for p in HALLUCINATION_PHRASES) and (no_speech > 0.35 or float(seg.start) < 6.0):
            continue
        if not music_mode and no_speech > 0.85 and avg_logprob < -1.1:
            continue
        if music_mode and no_speech > 0.96 and avg_logprob < -1.8:
            continue

        raw_segments.extend(_split_segment_by_word_pauses(seg, pause_gap=0.65 if not music_mode else 0.75))
        if total_dur > 0:
            pct = 38 + int(min(1.0, float(seg.end) / total_dur) * 32)
            m_cur, s_cur = divmod(int(seg.end), 60)
            m_tot, s_tot = divmod(int(total_dur), 60)
            update_job(
                job_id, progress=pct, detected_lang=detected_lang,
                stage=f"Bước 2/4: Whisper AI ({used_device.upper()} · {lang_label(detected_lang)}) đang nghe {m_cur:02d}:{s_cur:02d}/{m_tot:02d}:{s_tot:02d} ({len(raw_segments)} câu)...",
            )
    return raw_segments, detected_lang, total_dur


# ===================== SUBTITLE PARSER & EXPORTER =====================
def parse_vtt_or_srt(raw: str, default_lang: str = "en") -> List[Dict[str, Any]]:
    raw = raw.lstrip("\ufeff")
    blocks = re.split(r"\r?\n\s*\r?\n", raw)
    cues = []

    def to_sec(ts: str) -> float:
        parts = [float(x) for x in ts.strip().replace(",", ".").split(":")]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0]

    for b in blocks:
        lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        t_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), -1)
        if t_idx < 0:
            continue
        s_str, e_str = lines[t_idx].split("-->", 1)
        try:
            start, end = to_sec(s_str), to_sec(e_str.strip().split()[0])
        except Exception:
            continue

        txt_lines = [html.unescape(re.sub(r"<[^>]+>", "", ln)).strip() for ln in lines[t_idx + 1:]]
        txt = re.sub(r"\s+", " ", " ".join(x for x in txt_lines if x)).strip()
        if not txt:
            continue
        if cues:
            prev = cues[-1]
            if prev["text"] == txt and abs(prev["start"] - start) < 1.5:
                prev["end"] = max(prev["end"], end)
                continue
            if txt.startswith(prev["text"]) and (start - prev["start"]) < 2.5:
                prev["text"], prev["end"] = txt, max(prev["end"], end)
                continue

        cues.append({
            "start": round(start, 2), "end": round(max(start + 0.35, end), 2), "text": txt,
            "lang": default_lang, "reading": "", "vi": "", "raw_vi": "", "translated": "",
            "gender": "unknown", "gender_conf": 0.5, "gender_src": "",
        })
    return cues


def fetch_best_existing_subtitle(info: Dict[str, Any], source_lang: str):
    subs_map = info.get("subtitles") or {}
    if not subs_map:
        return None, ""
    wanted = norm_whisper_lang(source_lang)
    video_lang = (info.get("language") or "").split("-")[0].lower()
    candidates = []
    if wanted:
        candidates = [k for k in subs_map if k.lower() in (source_lang.lower(), wanted) or k.lower().startswith(wanted + "-")]
    elif video_lang:
        candidates = [k for k in subs_map if k.lower() == video_lang or k.lower().startswith(video_lang + "-")]
    if not candidates and not wanted:
        for pref in ("en", "ja", "zh", "ko", "fr", "de", "es", "vi"):
            for k in subs_map:
                if k.lower() == pref or k.lower().startswith(pref + "-"):
                    candidates.append(k)
                    break
            if candidates:
                break
    if not candidates:
        return None, ""

    chosen_lang = candidates[0]
    chosen_url = ""
    for ext_pref in ("vtt", "srt"):
        for tr in (subs_map.get(chosen_lang) or []):
            if (tr.get("ext") or "").lower() == ext_pref and tr.get("url"):
                chosen_url = tr["url"]
                break
        if chosen_url:
            break
    if not chosen_url:
        return None, ""

    try:
        with httpx.Client(headers={"User-Agent": "Mozilla/5.0"}, timeout=15.0, follow_redirects=True) as client:
            r = client.get(chosen_url)
            r.raise_for_status()
            lang_short = chosen_lang.split("-")[0].lower()
            parsed = parse_vtt_or_srt(r.text, default_lang=lang_short)
            if len(parsed) >= 3:
                return merge_fragmented_segments(parsed, lang_short), lang_short
    except Exception:
        return None, ""
    return None, ""


def _fmt_ass_time(sec: float) -> str:
    cs = max(0, int(round(sec * 100)))
    return f"{cs // 360000}:{(cs % 360000) // 6000:02d}:{(cs % 6000) // 100:02d}.{cs % 100:02d}"


def _fmt_srt_time(sec: float) -> str:
    ms = max(0, int(round(sec * 1000)))
    return f"{ms // 3600000:02d}:{(ms % 3600000) // 60000:02d}:{(ms % 60000) // 1000:02d},{ms % 1000:03d}"


def build_ass_content(cues: List[Dict[str, Any]], include_orig=True, include_read=True, include_trans=True) -> str:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Dual,Arial,34,&H00FFFFFF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,3,2,0,2,36,36,28,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    esc = lambda t: (t or "").replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", " ")
    lines = [header]
    for c in cues:
        parts = []
        if include_read and c.get("reading"):
            parts.append(r"{\fs22\c&HFDCA93&}" + esc(c["reading"]))
        if include_orig and c.get("text"):
            parts.append(r"{\fs34\c&HFFFFFF&}" + esc(c["text"]))
        tr = c.get("translated") or c.get("vi") or ""
        if include_trans and tr:
            parts.append(r"{\fs31\c&H15CCFA&}" + esc(tr))
        if parts:
            lines.append(f"Dialogue: 0,{_fmt_ass_time(c['start'])},{_fmt_ass_time(c['end'])},Dual,,0,0,0,," + r"\N".join(parts))
    return "\n".join(lines) + "\n"


def build_srt_content(cues: List[Dict[str, Any]], include_orig=True, include_trans=True) -> str:
    blocks = []
    for i, c in enumerate(cues, 1):
        lines = []
        if include_orig and c.get("text"):
            lines.append(c["text"])
        tr = c.get("translated") or c.get("vi") or ""
        if include_trans and tr:
            lines.append(tr)
        if lines:
            blocks.append(f"{i}\n{_fmt_srt_time(c['start'])} --> {_fmt_srt_time(c['end'])}\n" + "\n".join(lines))
    return "\n\n".join(blocks) + "\n"


# ===================== JOB STORAGE =====================
def save_job_meta(job: Dict[str, Any]):
    with open(JOBS_DIR / f"{job['id']}.json", "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)


def load_saved_jobs() -> List[Dict[str, Any]]:
    items = []
    for p in sorted(JOBS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            if j.get("media_path") and os.path.exists(j["media_path"]):
                items.append(j)
        except Exception:
            continue
    return items


def update_job(job_id: str, **kwargs):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)
            JOBS[job_id]["updated_at"] = time.time()


def get_job_by_id(job_id: str) -> Optional[Dict[str, Any]]:
    with JOBS_LOCK:
        if job_id in JOBS:
            return JOBS[job_id]
    meta_path = JOBS_DIR / f"{job_id}.json"
    if meta_path.exists():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        with JOBS_LOCK:
            JOBS[job_id] = data
        return data
    return None


# ===================== WORKER PIPELINE =====================
def run_download_and_transcribe_job(
    job_id: str, url: Optional[str], local_file_path: Optional[str], title_hint: str,
    source_lang: str, target_lang: str, trans_style: str, whisper_model_size: str,
    prefer_existing_sub: bool, music_mode: bool = False, voice_override: str = "auto",
    llm_url: str = "", llm_model: str = "", llm_key: str = "",
):
    try:
        media_path = local_file_path
        video_title = title_hint or "Video"
        existing_sub_cues = None
        detected_lang = norm_whisper_lang(source_lang) or ""
        tgt_lang = norm_gtrans_lang(target_lang, "vi")

        if url:
            import yt_dlp
            update_job(job_id, status="downloading", progress=5, stage="Bước 1/4: Đang phân tích & tải video...")
            out_template = str(MEDIA_DIR / f"{job_id}.%(ext)s")

            def progress_hook(d):
                if d.get("status") == "downloading":
                    total_b = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    dl_b = d.get("downloaded_bytes") or 0
                    pct = int((dl_b / total_b) * 100) if total_b > 0 else 0
                    spd = d.get("_speed_str", "").strip()
                    update_job(job_id, progress=min(30, 5 + int(pct * 0.25)), stage=f"Bước 1/4: Đang tải video ({pct}% {spd})...")
                elif d.get("status") == "finished":
                    update_job(job_id, progress=32, stage="Bước 1/4: Đang ghép hình & tiếng chuẩn MP4...")

            ydl_opts: Dict[str, Any] = {
                "format": "bestvideo[ext=mp4][vcodec^=avc1][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/best[height<=1080]/best",
                "merge_output_format": "mp4", "outtmpl": out_template, "noplaylist": True,
                "quiet": True, "no_warnings": True, "progress_hooks": [progress_hook],
            }
            if FFMPEG_BIN:
                ydl_opts["ffmpeg_location"] = FFMPEG_BIN

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_title = info.get("title") or video_title
                candidates = [
                    p for p in MEDIA_DIR.glob(f"{job_id}.*")
                    if p.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov", ".m4a", ".mp3", ".ogg", ".wav")
                ]
                if not candidates:
                    raise RuntimeError("Không tìm thấy file video sau khi tải.")
                candidates.sort(key=lambda x: x.stat().st_size, reverse=True)
                media_path = str(candidates[0])

                if prefer_existing_sub:
                    update_job(job_id, progress=34, stage="Đang kiểm tra phụ đề chính thức sẵn có...")
                    sub_cues, sub_lang = fetch_best_existing_subtitle(info, source_lang)
                    if sub_cues:
                        existing_sub_cues, detected_lang = sub_cues, sub_lang

        if not media_path or not os.path.exists(media_path):
            raise RuntimeError("Không tìm thấy file media để xử lý.")

        update_job(job_id, title=video_title, media_path=media_path, media_url=f"/media/{job_id}")

        # Luôn giải mã 1 bản audio gốc SẠCH (không cắt bass) để AI ECAPA+Wav2Vec2 phân tích âm sắc chuẩn xác
        clean_audio_np = decode_audio_ffmpeg_np(media_path, vocal_boost=False)

        if existing_sub_cues:
            cues = existing_sub_cues
            update_job(job_id, status="analyzing", progress=70, detected_lang=detected_lang, stage=f"Đã lấy phụ đề gốc ({len(cues)} câu). Đang cắt khoảng lặng & chạy AI phân tích giọng...")
        else:
            update_job(job_id, status="transcribing", progress=35, stage=f"Bước 2/4: Nạp Whisper AI ({whisper_model_size})...")
            whisper_audio = decode_audio_ffmpeg_np(media_path, vocal_boost=True) if music_mode else clean_audio_np
            model, used_device = get_whisper_model(whisper_model_size)
            lang_arg = norm_whisper_lang(source_lang)

            def _run_whisper(m, dev, inp, is_music):
                kw = _build_transcribe_kwargs(m, lang_arg, music_mode=is_music)
                seg_gen, inf = m.transcribe(inp, **kw)
                return _collect_whisper_segments(seg_gen, inf, job_id, dev, source_lang, is_music)

            try:
                raw_segments, detected_lang, total_dur = _run_whisper(model, used_device, whisper_audio, music_mode)
            except Exception as gpu_err:
                if used_device == "cuda":
                    update_job(job_id, stage="GPU CUDA lỗi — chuyển sang CPU (int8)...")
                    model, used_device = get_whisper_model(whisper_model_size, force_cpu=True)
                    raw_segments, detected_lang, total_dur = _run_whisper(model, used_device, whisper_audio, music_mode)
                else:
                    raise gpu_err

            if not music_mode and total_dur > 25 and len(raw_segments) < 3:
                update_job(job_id, progress=42, stage="Nền nhạc lớn — kích hoạt bộ lọc giọng hát (Vocal Boost)...")
                boosted = decode_audio_ffmpeg_np(media_path, vocal_boost=True)
                raw2, lang2, _ = _run_whisper(model, used_device, boosted, True)
                if len(raw2) > len(raw_segments):
                    raw_segments, detected_lang, music_mode = raw2, lang2, True

            cues = merge_fragmented_segments(raw_segments, detected_lang)
            if not cues:
                raise RuntimeError("Whisper AI không nhận diện được lời nói. Thử bật '🎵 Nhạc lớn / MV' hoặc chọn model 'small'.")

        # ---------- STEP 3: SPEECH TRIM + AI VOICE GENDER (ECAPA-TDNN + WAV2VEC2) ----------
        update_job(job_id, status="analyzing", progress=72, stage=f"Bước 3/4: Cắt khoảng lặng & chạy AI phân biệt giọng Nam/Nữ ({len(cues)} câu)...")
        cues = trim_cues_to_speech(clean_audio_np, cues, music_mode=music_mode)
        cues = annotate_cues_gender(
            clean_audio_np, cues, music_mode=music_mode, voice_override=voice_override,
            status_cb=lambda msg: update_job(job_id, progress=74, stage=f"Bước 3/4: {msg}"),
        )
        male_cnt = sum(1 for c in cues if c.get("gender") == "male")
        female_cnt = sum(1 for c in cues if c.get("gender") == "female")

        # ---------- STEP 4: CONTEXTUAL TRANSLATION ----------
        update_job(
            job_id, status="translating", progress=76, detected_lang=detected_lang,
            target_lang=tgt_lang, trans_style=trans_style, music_mode=music_mode, voice_override=voice_override,
            stage=f"Bước 4/4: Dịch ngữ cảnh ({male_cnt}♂ / {female_cnt}♀ · {lang_label(detected_lang)} → {lang_label(tgt_lang)})...",
        )

        def on_tr_progress(pct, done_cnt, total_cnt):
            update_job(job_id, progress=76 + int(pct * 0.23), stage=f"Bước 4/4: Đang dịch ({done_cnt}/{total_cnt} câu → {lang_label(tgt_lang)})...")

        sl_for_trans = source_lang if (source_lang and source_lang != "auto") else (detected_lang or "auto")
        cues = translate_cues_context(
            cues, sl=sl_for_trans, tl=tgt_lang, style=trans_style, music_mode=music_mode,
            llm_url=llm_url, llm_model=llm_model, llm_key=llm_key, progress_cb=on_tr_progress,
        )

        update_job(
            job_id, status="done", progress=100, detected_lang=detected_lang,
            target_lang=tgt_lang, trans_style=trans_style, music_mode=music_mode,
            voice_override=voice_override, cues=cues,
            stage=f"Hoàn tất! {len(cues)} câu ({male_cnt}♂ / {female_cnt}♀ · {GENDER_AI.status_label()}).",
        )
        with JOBS_LOCK:
            save_job_meta(JOBS[job_id])

    except Exception as exc:
        update_job(job_id, status="error", error=str(exc), stage=f"Lỗi: {exc}")


# ===================== API ROUTES =====================
@app.get("/", response_class=HTMLResponse)
def serve_home():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/system")
def api_system_info():
    return {
        "app_name": "MinhSub Studio",
        "author": "Lê Đình Hoàng Minh",
        "version": "5.6",
        "device": detect_compute_device(),
        "ffmpeg": bool(FFMPEG_BIN),
        "kakasi": bool(_KAKASI),
        "numpy": np is not None,
        "gender_ai": GENDER_AI.status_label(),
        "languages": LANGUAGES,
    }


class UrlJobRequest(BaseModel):
    url: str
    source_lang: str = "auto"
    target_lang: str = "vi"
    trans_style: str = "auto"
    whisper_model: str = "small"
    prefer_existing_sub: bool = True
    music_mode: bool = False
    voice_override: str = "auto"
    llm_url: str = ""
    llm_model: str = ""
    llm_key: str = ""


@app.post("/api/jobs/url")
def api_create_url_job(req: UrlJobRequest):
    url = (req.url or "").strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Đường link không hợp lệ (cần bắt đầu bằng http:// hoặc https://).")

    job_id = uuid.uuid4().hex[:10]
    job = {
        "id": job_id, "source_type": "url", "source_url": url, "title": url,
        "status": "queued", "progress": 2, "stage": "Đang khởi tạo...",
        "detected_lang": req.source_lang, "target_lang": req.target_lang,
        "trans_style": req.trans_style, "music_mode": req.music_mode,
        "voice_override": req.voice_override,
        "media_path": "", "media_url": "", "cues": [],
        "created_at": time.time(), "updated_at": time.time(), "error": "",
    }
    with JOBS_LOCK:
        JOBS[job_id] = job

    threading.Thread(
        target=run_download_and_transcribe_job,
        args=(
            job_id, url, None, url, req.source_lang, req.target_lang, req.trans_style,
            req.whisper_model, req.prefer_existing_sub, req.music_mode, req.voice_override,
            req.llm_url, req.llm_model, req.llm_key,
        ),
        daemon=True,
    ).start()
    return {"ok": True, "job_id": job_id}


@app.post("/api/jobs/upload")
async def api_create_upload_job(
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("vi"),
    trans_style: str = Form("auto"),
    whisper_model: str = Form("small"),
    music_mode: bool = Form(False),
    voice_override: str = Form("auto"),
    llm_url: str = Form(""),
    llm_model: str = Form(""),
    llm_key: str = Form(""),
):
    job_id = uuid.uuid4().hex[:10]
    ext = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    dest_path = MEDIA_DIR / f"{job_id}{ext}"

    with open(dest_path, "wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    job = {
        "id": job_id, "source_type": "file", "source_url": "", "title": file.filename or f"Video {job_id}",
        "status": "queued", "progress": 10, "stage": "Đã nạp file! Đang chuẩn bị Whisper AI...",
        "detected_lang": source_lang, "target_lang": target_lang, "trans_style": trans_style,
        "music_mode": music_mode, "voice_override": voice_override,
        "media_path": str(dest_path), "media_url": f"/media/{job_id}", "cues": [],
        "created_at": time.time(), "updated_at": time.time(), "error": "",
    }
    with JOBS_LOCK:
        JOBS[job_id] = job

    threading.Thread(
        target=run_download_and_transcribe_job,
        args=(
            job_id, None, str(dest_path), job["title"], source_lang, target_lang,
            trans_style, whisper_model, False, music_mode, voice_override, llm_url, llm_model, llm_key,
        ),
        daemon=True,
    ).start()
    return {"ok": True, "job_id": job_id}


class RetranslateRequest(BaseModel):
    target_lang: str = "vi"
    source_lang: Optional[str] = None
    trans_style: str = "auto"
    music_mode: bool = False
    voice_override: str = "auto"
    llm_url: str = ""
    llm_model: str = ""
    llm_key: str = ""


@app.post("/api/jobs/{job_id}/retranslate")
def api_retranslate_job(job_id: str, req: RetranslateRequest):
    job = get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Không tìm thấy video này.")
    cues = job.get("cues") or []
    if not cues:
        raise HTTPException(status_code=400, detail="Video chưa có phụ đề gốc để dịch lại.")

    # Chạy lại AI phân tích giọng Nam/Nữ (ECAPA+Wav2Vec2) hoặc áp dụng khóa giọng Nam/Nữ mới
    if job.get("media_path") and os.path.exists(job["media_path"]):
        clean_audio = decode_audio_ffmpeg_np(job["media_path"], vocal_boost=False)
        cues = trim_cues_to_speech(clean_audio, cues, music_mode=req.music_mode)
        cues = annotate_cues_gender(clean_audio, cues, music_mode=req.music_mode, voice_override=req.voice_override)
    elif req.voice_override in ("male", "female"):
        cues = annotate_cues_gender(None, cues, music_mode=req.music_mode, voice_override=req.voice_override)

    sl = req.source_lang or job.get("detected_lang") or "auto"
    tl = norm_gtrans_lang(req.target_lang, "vi")

    translated_cues = translate_cues_context(
        cues, sl=sl, tl=tl, style=req.trans_style, music_mode=req.music_mode,
        llm_url=req.llm_url, llm_model=req.llm_model, llm_key=req.llm_key,
    )
    male_cnt = sum(1 for c in translated_cues if c.get("gender") == "male")
    female_cnt = sum(1 for c in translated_cues if c.get("gender") == "female")
    update_job(
        job_id, target_lang=tl, trans_style=req.trans_style, music_mode=req.music_mode,
        voice_override=req.voice_override, cues=translated_cues,
        stage=f"Đã phân tích lại ({male_cnt}♂ / {female_cnt}♀) & dịch sang {lang_label(tl)}!",
    )
    with JOBS_LOCK:
        save_job_meta(JOBS[job_id])
        return JOBS[job_id]


class EditCueRequest(BaseModel):
    text: Optional[str] = None
    vi: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    gender: Optional[str] = None


@app.put("/api/jobs/{job_id}/cues/{cue_idx}")
def api_edit_cue(job_id: str, cue_idx: int, req: EditCueRequest):
    job = get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Không tìm thấy video này.")
    cues = job.get("cues") or []
    if cue_idx < 0 or cue_idx >= len(cues):
        raise HTTPException(status_code=404, detail="Không tìm thấy dòng phụ đề.")

    cue = cues[cue_idx]
    if req.text is not None:
        cue["text"] = req.text.strip()
        if cue.get("lang") == "ja" or has_kana(cue["text"]):
            cue["reading"] = get_japanese_reading(cue["text"])
    if req.vi is not None:
        cue["vi"] = cue["translated"] = req.vi.strip()
    if req.start is not None:
        cue["start"] = round(max(0.0, float(req.start)), 2)
    if req.end is not None:
        cue["end"] = round(max(cue["start"] + 0.2, float(req.end)), 2)
    if req.gender in ("male", "female", "unknown"):
        cue["gender"] = req.gender
        cue["gender_conf"] = 1.0
        cue["gender_src"] = "Sửa thủ công"
        # Tự cập nhật lại xưng hô tiếng Việt cho riêng câu này nếu đang dịch sang tiếng Việt
        if (job.get("target_lang") or "vi").startswith("vi") and req.vi is None:
            naturalize_vietnamese_cues(cues, style=job.get("trans_style", "auto"), music_mode=bool(job.get("music_mode")))

    with JOBS_LOCK:
        save_job_meta(job)
    return {"ok": True, "cue": cue}


@app.post("/api/jobs/{job_id}/import-sub")
async def api_import_sub(
    job_id: str,
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("vi"),
    trans_style: str = Form("auto"),
    music_mode: bool = Form(False),
    voice_override: str = Form("auto"),
):
    job = get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Không tìm thấy video này.")

    raw = (await file.read()).decode("utf-8", errors="ignore")
    lang_guess = norm_whisper_lang(source_lang) or detect_text_lang(raw[:500]) or "en"
    parsed = parse_vtt_or_srt(raw, default_lang=lang_guess)
    if not parsed:
        raise HTTPException(status_code=400, detail="Không đọc được dòng phụ đề nào từ file SRT/VTT này.")

    if job.get("media_path") and os.path.exists(job["media_path"]):
        clean_audio = decode_audio_ffmpeg_np(job["media_path"], vocal_boost=False)
        parsed = trim_cues_to_speech(clean_audio, parsed, music_mode=music_mode)
        parsed = annotate_cues_gender(clean_audio, parsed, music_mode=music_mode, voice_override=voice_override)

    tl = norm_gtrans_lang(target_lang, "vi")
    cues = translate_cues_context(
        parsed, sl=source_lang if source_lang != "auto" else lang_guess,
        tl=tl, style=trans_style, music_mode=music_mode,
    )
    update_job(
        job_id, status="done", progress=100, detected_lang=lang_guess, target_lang=tl,
        trans_style=trans_style, music_mode=music_mode, voice_override=voice_override,
        cues=cues, stage=f"Đã nạp & dịch {len(cues)} câu từ file phụ đề.",
    )
    with JOBS_LOCK:
        save_job_meta(JOBS[job_id])
        return JOBS[job_id]


@app.get("/api/jobs/{job_id}/download-video")
def api_download_raw_video(job_id: str):
    job = get_job_by_id(job_id)
    if not job or not job.get("media_path") or not os.path.exists(job["media_path"]):
        raise HTTPException(status_code=404, detail="Không tìm thấy file video gốc.")
    path = Path(job["media_path"])
    dl_name = f"{safe_filename(job.get('title', job_id))}{path.suffix}"
    return FileResponse(str(path), media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(dl_name)}"})


class ExportVideoRequest(BaseModel):
    mode: str = "hardsub"
    include_orig: bool = True
    include_read: bool = True
    include_trans: bool = True


@app.post("/api/jobs/{job_id}/export-video")
def api_export_subtitled_video(job_id: str, req: ExportVideoRequest):
    if not FFMPEG_BIN:
        raise HTTPException(status_code=500, detail="Không tìm thấy FFmpeg để xuất video.")
    job = get_job_by_id(job_id)
    if not job or not job.get("media_path") or not os.path.exists(job["media_path"]):
        raise HTTPException(status_code=404, detail="Không tìm thấy file video gốc.")
    cues = job.get("cues") or []
    if not cues:
        raise HTTPException(status_code=400, detail="Video chưa có phụ đề để gắn.")

    mode = "softsub" if req.mode == "softsub" else "hardsub"
    out_file = EXPORT_DIR / f"{job_id}_{mode}.mp4"
    tmp_dir = tempfile.mkdtemp(prefix="ms_exp_")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

    try:
        src_abs, out_abs = str(Path(job["media_path"]).resolve()), str(out_file.resolve())
        if mode == "hardsub":
            (Path(tmp_dir) / "subs.ass").write_text(
                build_ass_content(cues, req.include_orig, req.include_read, req.include_trans),
                encoding="utf-8-sig",
            )
            cmd = [
                FFMPEG_BIN, "-y", "-i", src_abs, "-vf", "ass=subs.ass",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_abs,
            ]
        else:
            (Path(tmp_dir) / "subs.srt").write_text(
                build_srt_content(cues, req.include_orig, req.include_trans),
                encoding="utf-8-sig",
            )
            cmd = [
                FFMPEG_BIN, "-y", "-i", src_abs, "-i", "subs.srt",
                "-map", "0:v", "-map", "0:a?", "-map", "1:0",
                "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text",
                "-metadata:s:s:0", "language=vie", "-movflags", "+faststart", out_abs,
            ]

        proc = subprocess.run(cmd, cwd=tmp_dir, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=creationflags, timeout=900)
        if proc.returncode != 0 or not out_file.exists():
            raise RuntimeError(f"FFmpeg xuất video thất bại: {proc.stderr.decode('utf-8', errors='ignore')[-400:]}")
        return {"ok": True, "mode": mode, "download_url": f"/api/jobs/{job_id}/download-exported?mode={mode}"}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/api/jobs/{job_id}/download-exported")
def api_download_exported_video(job_id: str, mode: str = "hardsub"):
    job = get_job_by_id(job_id)
    mode = "softsub" if mode == "softsub" else "hardsub"
    out_file = EXPORT_DIR / f"{job_id}_{mode}.mp4"
    if not out_file.exists():
        raise HTTPException(status_code=404, detail="Chưa có file video xuất.")
    dl_name = f"{safe_filename((job or {}).get('title', job_id))}_MinhSub_{mode}.mp4"
    return FileResponse(str(out_file), media_type="video/mp4", headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(dl_name)}"})


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str):
    job = get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Không tìm thấy video này.")
    return job


@app.get("/api/history")
def api_history():
    return [
        {
            "id": j["id"], "title": j.get("title", "Video"),
            "detected_lang": j.get("detected_lang", ""), "target_lang": j.get("target_lang", "vi"),
            "cue_count": len(j.get("cues", [])), "created_at": j.get("created_at", 0),
        }
        for j in load_saved_jobs()[:30]
    ]


@app.delete("/api/history/{job_id}")
def api_delete_history(job_id: str):
    with JOBS_LOCK:
        JOBS.pop(job_id, None)
    (JOBS_DIR / f"{job_id}.json").unlink(missing_ok=True)
    for folder in (MEDIA_DIR, EXPORT_DIR):
        for f in folder.glob(f"{job_id}*"):
            try:
                f.unlink()
            except Exception:
                pass
    return {"ok": True}


@app.get("/media/{job_id}")
def serve_media(job_id: str, request: Request):
    job = get_job_by_id(job_id)
    if not job or not job.get("media_path") or not os.path.exists(job["media_path"]):
        raise HTTPException(status_code=404, detail="Media not found")

    path = job["media_path"]
    file_size = os.path.getsize(path)
    mime_map = {
        ".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska",
        ".mov": "video/quicktime", ".mp3": "audio/mpeg", ".wav": "audio/wav",
        ".m4a": "audio/mp4", ".ogg": "audio/ogg",
    }
    content_type = mime_map.get(Path(path).suffix.lower(), "video/mp4")

    range_header = request.headers.get("range")
    if range_header:
        m = re.match(r"bytes=(\d*)-(\d*)", range_header)
        if m:
            start = int(m.group(1)) if m.group(1) else 0
            end = min(int(m.group(2)) if m.group(2) else start + 2 * 1024 * 1024 - 1, file_size - 1)
            chunk_len = end - start + 1

            def iter_range():
                with open(path, "rb") as f:
                    f.seek(start)
                    rem = chunk_len
                    while rem > 0:
                        data = f.read(min(65536, rem))
                        if not data:
                            break
                        rem -= len(data)
                        yield data

            return StreamingResponse(
                iter_range(), status_code=206, media_type=content_type,
                headers={"Content-Range": f"bytes {start}-{end}/{file_size}", "Accept-Ranges": "bytes", "Content-Length": str(chunk_len)},
            )

    def iter_full():
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                yield chunk

    return StreamingResponse(iter_full(), media_type=content_type, headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)})


# ===================== DICTIONARY LOOKUP =====================
@app.get("/api/lookup")
def api_lookup(word: str, sentence: str = "", lang: str = "auto", target_lang: str = "vi"):
    clean = re.sub(r"\s+", " ", (word or "")).strip()
    if not clean:
        raise HTTPException(status_code=400, detail="Empty word")

    is_phrase = (" " in clean) or (len(clean) >= 4 and (has_kanji(clean) or has_kana(clean)))
    looks_ja = lang == "ja" or has_kana(clean) or (lang == "auto" and has_kanji(clean))
    sl = "ja" if looks_ja else norm_gtrans_lang(lang, "auto")
    tl = norm_gtrans_lang(target_lang, "vi")

    with httpx.Client(headers={"User-Agent": "Mozilla/5.0"}, timeout=10.0) as client:
        vi_data = {"translated": "", "reading": "", "dict": [], "detected_lang": sl}
        sent_vi = ""
        try:
            vi_data = _google_translate_request(client, clean, sl=sl, tl=tl, include_dict=True)
        except Exception:
            pass
        if sentence:
            try:
                sent_vi = _google_translate_request(client, sentence, sl=sl, tl=tl)["translated"]
            except Exception:
                pass

        detected = "ja" if looks_ja else (vi_data.get("detected_lang") or detect_text_lang(clean))
        phonetic = get_japanese_reading(clean) if looks_ja else (vi_data.get("reading") or "")
        audio_url, extra_note = "", ""
        pos_tags: List[str] = [g["pos"] for g in vi_data.get("dict", []) if g.get("pos")]

        if looks_ja:
            try:
                jr = client.get("https://jisho.org/api/v1/search/words", params={"keyword": clean})
                if jr.status_code == 200 and jr.json().get("data"):
                    entry = jr.json()["data"][0]
                    jp0 = (entry.get("japanese") or [{}])[0]
                    if jp0.get("reading"):
                        phonetic = jp0["reading"] + (f" ({phonetic})" if phonetic and jp0["reading"] not in phonetic else "")
                    jlpt = entry.get("jlpt") or []
                    jlpt_tag = f"[{jlpt[0].upper()}] " if jlpt else ""
                    s0 = (entry.get("senses") or [{}])[0]
                    pos_list = (s0.get("parts_of_speech") or [])[:2]
                    if pos_list and not pos_tags:
                        pos_tags = pos_list
                    eng = "; ".join((s0.get("english_definitions") or [])[:4])
                    extra_note = f"{jlpt_tag}{', '.join(pos_list)}: {eng}".strip(": ")
            except Exception:
                pass
        elif detected.startswith("en") and " " not in clean:
            try:
                dr = client.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{clean.lower()}")
                if dr.status_code == 200 and isinstance(dr.json(), list) and dr.json():
                    e0 = dr.json()[0]
                    phonetic = e0.get("phonetic") or phonetic
                    for ph in e0.get("phonetics") or []:
                        phonetic = phonetic or ph.get("text") or ""
                        audio_url = audio_url or ph.get("audio") or ""
                    meanings = e0.get("meanings") or []
                    for m in meanings[:3]:
                        mapped = POS_VI_MAP.get((m.get("partOfSpeech") or "").lower(), m.get("partOfSpeech") or "")
                        if mapped and mapped not in pos_tags:
                            pos_tags.append(mapped)
                    if meanings:
                        m0 = meanings[0]
                        d0 = (m0.get("definitions") or [{}])[0].get("definition", "")
                        if d0:
                            p_vi = POS_VI_MAP.get((m0.get("partOfSpeech") or "").lower(), m0.get("partOfSpeech", ""))
                            extra_note = f"{p_vi}: {d0}"
            except Exception:
                pass

    if is_phrase and not pos_tags:
        pos_tags = ["Cụm từ / Thành ngữ"]

    return {
        "word": clean, "is_phrase": is_phrase, "lang": detected, "target_lang": tl,
        "phonetic": phonetic, "pos_tags": pos_tags[:3], "audio_url": audio_url,
        "vi_main": vi_data.get("translated", ""), "vi_dict": vi_data.get("dict", []),
        "extra_note": extra_note, "sentence_vi": sent_vi,
    }


# ===================== VOCABULARY NOTEBOOK =====================
def _read_vocab() -> List[Dict[str, Any]]:
    if not VOCAB_FILE.exists():
        return []
    try:
        return json.loads(VOCAB_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_vocab(items: List[Dict[str, Any]]):
    VOCAB_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/api/vocab")
def api_get_vocab():
    return _read_vocab()


@app.post("/api/vocab")
async def api_save_vocab(item: Dict[str, Any]):
    word = re.sub(r"\s+", " ", str(item.get("word") or "")).strip()
    if not word:
        raise HTTPException(status_code=400, detail="Từ vựng không được để trống.")

    items = [x for x in _read_vocab() if str(x.get("word") or "").strip().lower() != word.lower()]
    now = time.time()
    meaning = str(item.get("vi_main") or item.get("meaning") or "").strip()
    entry: Dict[str, Any] = {
        **item,
        "id": str(item.get("id") or uuid.uuid4().hex[:10]),
        "word": word,
        "phonetic": str(item.get("phonetic") or item.get("reading") or "").strip(),
        "vi_main": meaning, "meaning": meaning,
        "pos_tags": item.get("pos_tags") if isinstance(item.get("pos_tags"), list) else [],
        "sentence": str(item.get("sentence") or "").strip(),
        "sentence_vi": str(item.get("sentence_vi") or "").strip(),
        "lang": str(item.get("lang") or "auto").strip(),
        "target_lang": str(item.get("target_lang") or "vi").strip(),
        "extra_note": str(item.get("extra_note") or "").strip(),
        "audio_url": str(item.get("audio_url") or "").strip(),
        "created_at": item.get("created_at") or now,
        "updated_at": now,
    }
    items.insert(0, entry)
    _write_vocab(items)
    return {"ok": True, "item": entry, "total": len(items)}


@app.delete("/api/vocab/{item_id}")
def api_delete_vocab(item_id: str):
    items = _read_vocab()
    target = item_id.strip().lower()
    new_items = [
        x for x in items
        if str(x.get("id") or "").strip().lower() != target and str(x.get("word") or "").strip().lower() != target
    ]
    _write_vocab(new_items)
    return {"ok": True, "deleted": len(items) - len(new_items), "total": len(new_items)}


@app.delete("/api/vocab")
def api_clear_vocab():
    _write_vocab([])
    return {"ok": True, "total": 0}


@app.get("/api/vocab/export")
def api_export_vocab():
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Word", "Phonetic", "Meaning", "POS", "Sentence", "Sentence Translation", "Note", "Lang"])
    for x in _read_vocab():
        pos_str = ", ".join(x.get("pos_tags") or []) if isinstance(x.get("pos_tags"), list) else str(x.get("pos_tags") or "")
        writer.writerow([
            x.get("word", ""), x.get("phonetic", ""), x.get("vi_main") or x.get("meaning", ""),
            pos_str, x.get("sentence", ""), x.get("sentence_vi", ""), x.get("extra_note", ""), x.get("lang", ""),
        ])
    return StreamingResponse(
        iter([buf.getvalue().encode("utf-8-sig")]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=minhsub_vocab.csv"},
    )


if __name__ == "__main__":
    host = os.environ.get("DUALSUB_HOST", "127.0.0.1")
    port = int(os.environ.get("DUALSUB_PORT", os.environ.get("PORT", "8000")))
    print(f"MinhSub Studio v5.6 (by Lê Đình Hoàng Minh) đang chạy tại: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")