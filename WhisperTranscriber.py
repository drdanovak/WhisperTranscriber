import os
import io
import tempfile
import subprocess
import warnings
from typing import List, Tuple

import streamlit as st

# --- load .env (OPENAI_API_KEY etc.) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass  # optional; we handle missing dotenv gracefully

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# --- Optional browser recorder ---
try:
    from audio_recorder_streamlit import audio_recorder
    HAS_RECORDER = True
except Exception:
    HAS_RECORDER = False

# --- Silence the harmless CPU notice from whisper on streamlit ---
warnings.filterwarnings(
    "ignore", message="FP16 is not supported on CPU; using FP32 instead"
)

# --- Be nice to small CPUs when PyTorch is present ---
try:
    import torch
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
except Exception:
    torch = None

# ---- Ensure ffmpeg is available (bundled, no system package needed) ----
try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] = os.path.dirname(FFMPEG_EXE) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    FFMPEG_EXE = None

# ---- Whisper (local) + utils ----
import whisper
from whisper.utils import format_timestamp

# ---- Light features for diarization ----
import numpy as np
import librosa
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ---- (Optional) OpenAI Whisper API client (pin old API for stability) ----
# The tutorial syntax uses the classic openai client. We keep it for compatibility.
try:
    import openai  # pinned in requirements
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False


# ========================= UI =========================
st.set_page_config(page_title="Whisper Diarized Transcriber", layout="centered")
st.title("🎙️ Whisper Diarized Transcriber")

st.markdown(
    """
Upload or record audio/video. The app will:
1) Convert media to audio via a **bundled ffmpeg**  
2) Transcribe with **local Whisper** or **OpenAI Whisper API**  
3) **Diarize** by clustering per‑segment MFCC features  
4) Let you **download** a timestamped, speaker‑labeled transcript  
    """
)

with st.expander("⚙️ Settings"):
    backend = st.selectbox(
        "Transcription backend",
        ["Local Whisper (CPU)", "OpenAI Whisper API (needs OPENAI_API_KEY)"],
        index=0,
        help="API can be faster on CPU-only hosts. Set OPENAI_API_KEY in your .env."
    )
    model_size = st.selectbox(
        "Local Whisper model (used if 'Local Whisper' selected)",
        ["base", "small", "medium"],  # modest sizes for Streamlit CPU
        index=0,
    )
    language_hint = st.text_input(
        "Language hint (optional)",
        value="",
        help="e.g., 'en' or 'English'. Leave blank to auto‑detect."
    )
    translate_to_en = st.checkbox("Translate to English (if supported)", value=False)
    max_speakers = st.slider(
        "Max speakers to try (auto‑estimate within 1..N)",
        min_value=1, max_value=6, value=3,
        help="Tries 1..N clusters; picks the best silhouette score."
    )
    min_segment_sec = st.slider(
        "Merge very short segments (seconds, 0 = off)", 0.0, 2.0, 0.3, 0.1,
        help="Post‑process to reduce spurious switches."
    )

st.markdown("### Input")

rec_bytes = None
if HAS_RECORDER:
    with st.expander("🎤 Record in browser", expanded=False):
        st.caption("Click to start/stop recording; preview appears below.")
        rec_bytes = audio_recorder(pause_threshold=1.0)  # returns WAV bytes
        if rec_bytes:
            st.audio(rec_bytes, format="audio/wav")

uploaded = st.file_uploader(
    "…or upload audio/video",
    type=["mp3","mp4","mpeg","mpga","m4a","wav","webm","flac","ogg","opus","mkv","mov","avi"],
)

if uploaded and (uploaded.type or "").startswith("video/"):
    st.video(uploaded)
elif uploaded:
    st.audio(uploaded)


# ========================= helpers =========================
def _safe_filename(name: str) -> str:
    keep = "-_.() "
    name = name or "file"
    safe = "".join(c for c in name if c.isalnum() or c in keep)
    return (safe.strip().replace(" ", "_")) or "file"

@st.cache_resource(show_spinner=False)
def load_whisper(model_name: str):
    return whisper.load_model(model_name, device="cpu")

def _to_tmp_file_from_bytes(b: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(b)
    tmp.flush()
    tmp.close()
    return tmp.name

def _to_tmp_file_from_uploader(file) -> str:
    suffix = os.path.splitext(file.name)[1].lower() or ".bin"
    data = file.getbuffer()
    return _to_tmp_file_from_bytes(data, suffix)

def _run_cmd(cmd: list) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stderr: {proc.stderr.decode(errors='ignore')[:8000]}"
        )

def _media_to_wav(input_path: str, sr: int = 16000) -> str:
    if not FFMPEG_EXE or not os.path.exists(FFMPEG_EXE):
        raise RuntimeError("Bundled ffmpeg not found. Ensure `imageio-ffmpeg` is installed.")
    out_path = tempfile.mktemp(suffix=".wav")
    cmd = [FFMPEG_EXE, "-y", "-i", input_path, "-ac", "1", "-ar", str(sr), out_path]
    _run_cmd(cmd)
    if not os.path.exists(out_path):
        raise RuntimeError("ffmpeg failed to produce WAV output.")
    return out_path

def _merge_short_segments(segments, threshold_sec: float):
    if threshold_sec <= 0 or len(segments) <= 1:
        return segments
    merged = []
    cur = segments[0].copy()
    for nxt in segments[1:]:
        dur = max(0.0, cur["end"] - cur["start"])
        if dur < threshold_sec and nxt["start"] >= cur["start"]:
            nxt = nxt.copy()
            nxt["start"] = cur["start"]
            nxt["text"] = (cur.get("text", "").strip() + " " + nxt.get("text", "").strip()).strip()
            cur = nxt
        else:
            merged.append(cur)
            cur = nxt.copy()
    merged.append(cur)
    return merged

def _mfcc_stats(y: np.ndarray, sr: int, start: float, end: float, n_mfcc=20) -> np.ndarray:
    start_samp = max(0, int(start * sr))
    end_samp = min(len(y), int(max(start + 0.1, end) * sr))  # ≥0.1s
    clip = y[start_samp:end_samp]
    if clip.size == 0:
        left = max(0, start_samp - int(0.05*sr))
        right = min(len(y), start_samp + int(0.05*sr))
        clip = y[left:right]
    if clip.size == 0:
        clip = y[: min(len(y), int(0.2 * sr))]
    mfcc = librosa.feature.mfcc(y=clip.astype(np.float32), sr=sr, n_mfcc=n_mfcc)
    mu = mfcc.mean(axis=1)
    sd = mfcc.std(axis=1)
    return np.concatenate([mu, sd])

def diarize_by_clustering(wav_path: str,
                          segments: List[dict],
                          max_speakers_try: int = 3) -> Tuple[List[dict], int]:
    if not segments:
        return segments, 1
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    feats = [_mfcc_stats(y, sr, float(s["start"]), float(s["end"])) for s in segments]
    X = np.vstack(feats)
    if len(segments) == 1 or max_speakers_try <= 1:
        for s in segments:
            s["speaker"] = "SPEAKER 1"
        return segments, 1
    best_labels, best_score, best_k = None, -1.0, 1
    for k in range(1, max_speakers_try + 1):
        if k == 1:
            labels, score = np.zeros(len(segments), dtype=int), -1.0
        else:
            try:
                labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
                score = silhouette_score(X, labels, metric="euclidean") if len(np.unique(labels)) > 1 else -1.0
            except Exception:
                labels, score = np.zeros(len(segments), dtype=int), -1.0
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels
    for i, s in enumerate(segments):
        s["speaker"] = f"SPEAKER {int(best_labels[i]) + 1}"
    return segments, best_k

def segments_to_text(segments: List[dict]) -> str:
    out = io.StringIO()
    last_speaker = None
    for seg in segments:
        spk = seg.get("speaker", "SPEAKER 1")
        if spk != last_speaker:
            out.write(f"\n{spk} {format_timestamp(seg['start'])}\n")
            last_speaker = spk
        out.write(seg.get("text", "").lstrip() + " ")
    return out.getvalue().strip() + "\n"

def build_download_name(original_name: str) -> str:
    base = os.path.splitext(_safe_filename(original_name or "transcript"))[0]
    return f"{base}_diarized.txt"


# ========================= Transcription backends =========================
def transcribe_local_whisper(wav_path: str, model_name: str, translate: bool, lang_hint: str):
    model = load_whisper(model_name)
    decode_opts = dict(
        temperature=0.0,
        beam_size=1,
        condition_on_previous_text=False,
        fp16=False,  # CPU-safe
    )
    if translate:
        decode_opts["task"] = "translate"
    if lang_hint.strip():
        decode_opts["language"] = lang_hint.strip()
    result = model.transcribe(wav_path, **decode_opts)
    segs = result.get("segments", []) or []
    # normalize segment dicts
    segments = [
        {"start": float(s["start"]), "end": float(s["end"]), "text": (s.get("text") or "").strip()}
        for s in segs
    ]
    return segments

def transcribe_openai_api(wav_path: str, translate: bool, lang_hint: str):
    if not (HAS_OPENAI and OPENAI_API_KEY):
        raise RuntimeError("OpenAI API backend selected, but OPENAI_API_KEY or 'openai' package is missing.")
    openai.api_key = OPENAI_API_KEY
    # Classic API used in the tutorial; request verbose_json for timestamps + segments
    with open(wav_path, "rb") as f:
        resp = openai.Audio.transcriptions.create(  # type: ignore[attr-defined]
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            temperature=0.0,
            language=(lang_hint.strip() or None),
            translate=bool(translate),
        )
    # resp has .segments with start/end/text when verbose_json
    segs = getattr(resp, "segments", None) or resp.get("segments", [])  # support dict-like responses
    if not segs:
        # Fallback: single segment if a host returns only 'text'
        duration = librosa.get_duration(path=wav_path)
        text = getattr(resp, "text", None) or resp.get("text", "")
        segs = [{"start": 0.0, "end": duration, "text": text}]
    segments = [
        {"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "text": (s.get("text") or "").strip()}
        for s in segs
    ]
    return segments


# ========================= Main action =========================
btn = st.button("🔎 Transcribe & Diarize")
if btn:
    try:
        if rec_bytes:
            in_path = _to_tmp_file_from_bytes(rec_bytes, ".wav")  # recorder gives wav bytes
        elif uploaded is not None:
            in_path = _to_tmp_file_from_uploader(uploaded)
        else:
            st.warning("Record or upload a file to continue.")
            st.stop()

        with st.spinner("Converting media to WAV…"):
            wav_path = in_path
            # only convert non-wav inputs (recorder already WAV)
            if not in_path.lower().endswith(".wav"):
                wav_path = _media_to_wav(in_path, sr=16000)

        # --- choose backend ---
        if backend.startswith("Local"):
            with st.spinner("Transcribing locally with Whisper…"):
                segments = transcribe_local_whisper(
                    wav_path, model_name=model_size, translate=translate_to_en, lang_hint=language_hint
                )
        else:
            if not OPENAI_API_KEY:
                st.error("OPENAI_API_KEY is not set. Put it in a local .env or Streamlit secret.")
                st.stop()
            with st.spinner("Transcribing with OpenAI Whisper API…"):
                segments = transcribe_openai_api(
                    wav_path, translate=translate_to_en, lang_hint=language_hint
                )

        if min_segment_sec > 0:
            segments = _merge_short_segments(segments, float(min_segment_sec))

        with st.spinner("Diarizing (clustering per‑segment features)…"):
            segments, chosen_k = diarize_by_clustering(
                wav_path, segments, max_speakers_try=int(max_speakers)
            )

        st.success(f"Done! Estimated speakers: {chosen_k}")

        st.subheader("Transcript (preview)")
        preview = segments_to_text(segments)
        st.text_area("Transcript", preview, height=320)

        fname = build_download_name((uploaded.name if uploaded else "recording.wav"))
        st.download_button(
            "⬇️ Download diarized transcript (.txt)",
            data=preview.encode("utf-8"),
            file_name=fname,
            mime="text/plain",
        )

    except Exception as e:
        st.error(f"Processing failed: {e}")
    finally:
        # best-effort cleanup
        for p in [locals().get("in_path"), locals().get("wav_path")]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
else:
    st.info("Record or upload an audio/video file, then click **Transcribe & Diarize**.")
