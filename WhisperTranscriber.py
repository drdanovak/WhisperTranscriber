import io
import os
import sys
import gc
import tempfile
import subprocess
from typing import List, Tuple

import streamlit as st

# ------------------ Environment hardening for small CPU hosts ------------------
# Use the statically-linked ffmpeg from imageio-ffmpeg (no system ffmpeg needed)
import imageio_ffmpeg
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
os.environ.setdefault("IMAGEIO_FFMPEG_EXE", FFMPEG_EXE)
os.environ["PATH"] = os.path.dirname(FFMPEG_EXE) + os.pathsep + os.environ.get("PATH", "")

# Torch & Whisper runtime knobs (reduce RAM & thread contention)
os.environ.setdefault("WHISPER_DISABLE_FP16", "1")      # don’t try FP16 on CPU
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))

import whisper
from whisper.utils import format_timestamp

import numpy as np
import librosa
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Optional: detect available RAM and adapt model choices
def _available_memory_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        # Fallback if psutil isn’t present (Streamlit Cloud often has ~2–4 GB)
        return 2.0

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Whisper Diarized Transcriber", layout="centered")
st.title("🎙️ Whisper Diarized Transcriber")

st.markdown(
    """
Upload **audio/video**, we’ll:

1. Convert to mono 16 kHz WAV (bundled `ffmpeg`)  
2. Transcribe with Whisper (local CPU)  
3. **Diarize** by clustering MFCC features  
4. Let you **download** a timestamped transcript with **speaker labels**
    """
)

# Choose safe model set based on RAM
mem_gb = _available_memory_gb()
if mem_gb >= 5:
    allowed_models = ["base", "small"]  # “medium” intentionally disabled for Cloud stability
else:
    allowed_models = ["base"]

with st.expander("⚙️ Settings"):
    model_size = st.selectbox(
        "Whisper model",
        allowed_models,
        index=0,
        help="‘base’ is lightest and recommended for CPU hosts. ‘small’ needs more RAM."
    )
    language_hint = st.text_input("Language hint (optional)", value="")
    translate_to_en = st.checkbox("Translate to English (if supported)", value=False)
    max_speakers = st.slider("Max speakers to try (auto-select 1..N)", 1, 6, 3)
    min_segment_sec = st.slider("Merge very short segments (sec, 0=off)", 0.0, 2.0, 0.5, 0.1)

uploaded = st.file_uploader(
    "Upload audio/video",
    type=["mp3","mp4","mpeg","mpga","m4a","wav","webm","flac","ogg","opus","mkv","mov","avi"],
)

# ------------------ Helpers ------------------
def _safe_filename(name: str) -> str:
    keep = "-_.() "
    return "".join(c for c in name if c.isalnum() or c in keep).strip().replace(" ", "_")

@st.cache_resource(show_spinner=False)
def load_whisper(model_name: str):
    # Force CPU device and deterministic options to keep memory down
    return whisper.load_model(model_name, device="cpu", in_memory=True)

def _save_to_tmp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower() or ".bin"
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(uploaded_file.read())
    f.flush()
    f.close()
    return f.name

def _media_to_wav(input_path: str, sr: int = 16000) -> str:
    out_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        FFMPEG_EXE, "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", str(sr),
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode(errors='ignore')[:600]}") from e
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("ffmpeg produced no output WAV.")
    return out_path

def _merge_short_segments(segments: List[dict], threshold_sec: float) -> List[dict]:
    if threshold_sec <= 0 or len(segments) <= 1:
        return segments
    merged = []
    cur = segments[0].copy()
    for nxt in segments[1:]:
        dur = cur["end"] - cur["start"]
        if dur < threshold_sec:
            cur["end"] = nxt["start"]
            cur["text"] = (cur.get("text","") + " " + nxt.get("text","")).strip()
            nxt["start"] = cur["start"]
            cur = nxt
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)
    return merged

def _mfcc_stats(y: np.ndarray, sr: int, start: float, end: float, n_mfcc=20) -> np.ndarray:
    s0 = max(0, int(start * sr))
    s1 = min(len(y), int(end * sr))
    if s1 <= s0:
        s1 = min(len(y), s0 + int(0.1 * sr))
    clip = y[s0:s1]
    if clip.size == 0:
        clip = y[max(0, s0 - int(0.1*sr)):min(len(y), s0 + int(0.1*sr))]
    mfcc = librosa.feature.mfcc(y=clip.astype(np.float32), sr=sr, n_mfcc=n_mfcc)
    mu = mfcc.mean(axis=1)
    sd = mfcc.std(axis=1)
    return np.concatenate([mu, sd])

def diarize_by_clustering(wav_path: str, segments: List[dict], max_k: int) -> Tuple[List[dict], int]:
    if not segments:
        return segments, 1
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    X = np.vstack([_mfcc_stats(y, sr, s["start"], s["end"]) for s in segments])

    best_labels, best_score, best_k = None, -2.0, 1
    for k in range(1, max_k + 1):
        if k == 1:
            labels = np.zeros(len(segments), dtype=int)
            score = -1.0
        else:
            try:
                model = AgglomerativeClustering(n_clusters=k, linkage="ward")
                labels = model.fit_predict(X)
                score = silhouette_score(X, labels, metric="euclidean") if len(np.unique(labels)) > 1 else -1.0
            except Exception:
                labels, score = np.zeros(len(segments), dtype=int), -1.0
        if score > best_score:
            best_labels, best_score, best_k = labels, score, k

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

def _download_name(original: str) -> str:
    base = os.path.splitext(_safe_filename(original or "transcript"))[0]
    return f"{base}_diarized.txt"

# ------------------ Main flow ------------------
if uploaded is not None:
    if uploaded.type.startswith("video/"):
        st.video(uploaded)
    else:
        st.audio(uploaded)

    if st.button("🔎 Transcribe & Diarize"):
        tmp_in = _save_to_tmp(uploaded)
        wav_path = None
        try:
            with st.spinner("Converting media to WAV…"):
                wav_path = _media_to_wav(tmp_in, sr=16000)

            with st.spinner(f"Loading Whisper model ({model_size})…"):
                model = load_whisper(model_size)

            # Keep decoding options lean for CPU
            decode_opts = dict(temperature=0.0, beam_size=1, condition_on_previous_text=False)
            if translate_to_en:
                decode_opts["task"] = "translate"
            if language_hint.strip():
                decode_opts["language"] = language_hint.strip()

            with st.spinner("Transcribing with Whisper…"):
                result = model.transcribe(wav_path, **decode_opts)
                segments = result.get("segments", []) or []

            if min_segment_sec > 0:
                segments = _merge_short_segments(segments, min_segment_sec)

            with st.spinner("Diarizing (clustering per‑segment features)…"):
                segments, chosen_k = diarize_by_clustering(wav_path, segments, max_k=max_speakers)

            st.success(f"Done! Estimated speakers: {chosen_k}")

            st.subheader("Transcript (preview)")
            text_out = segments_to_text(segments)
            st.text_area("Transcript", text_out, height=320)

            st.download_button(
                "⬇️ Download diarized transcript (.txt)",
                data=text_out.encode("utf-8"),
                file_name=_download_name(uploaded.name),
                mime="text/plain",
            )

        except Exception as e:
            st.error(f"Processing failed: {e}")
        finally:
            # free memory aggressively on small hosts
            try:
                del result, segments  # type: ignore
            except Exception:
                pass
            gc.collect()
            for p in [tmp_in, wav_path]:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
else:
    st.info("Upload an audio or video file to get started.")
