import os
import io
import math
import tempfile
import subprocess
import warnings
from datetime import timedelta
from typing import List, Tuple

import streamlit as st

# --- Silence harmless Whisper CPU warning ---
warnings.filterwarnings(
    "ignore",
    message="FP16 is not supported on CPU; using FP32 instead"
)

# --- Make PyTorch well-behaved on small CPUs ---
try:
    import torch
    # Cap threads to avoid overwhelming Streamlit Cloud CPU
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
except Exception:
    torch = None  # continue without torch-level tweaks

# ---- Ensure ffmpeg is available (bundled) ----
# We rely on imageio-ffmpeg's statically linked ffmpeg so we don't need system packages.
try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    # Prepend ffmpeg's folder to PATH, just in case other libs call `ffmpeg`
    os.environ["PATH"] = os.path.dirname(FFMPEG_EXE) + os.pathsep + os.environ.get("PATH", "")
except Exception as e:
    FFMPEG_EXE = None

# ---- Whisper and audio utils ----
import whisper
from whisper.utils import format_timestamp

# Lightweight feature extraction for diarization
import numpy as np
import librosa
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# ---------- UI ----------
st.set_page_config(page_title="Whisper Diarized Transcriber", layout="centered")
st.title("🎙️ Whisper Diarized Transcriber")

st.markdown(
    """
Upload **any common audio/video** file.  
The app will:
1) Convert your media to audio (via a bundled `ffmpeg`)  
2) Transcribe with OpenAI Whisper (local, CPU by default)  
3) **Diarize** by clustering per‑segment audio features (MFCC statistics)  
4) Let you **download** a timestamped plain‑text transcript with **speaker labels**  
    """
)

with st.expander("⚙️ Settings"):
    model_size = st.selectbox(
        "Whisper model",
        ["base", "small", "medium"],  # keep sizes modest for Streamlit CPU
        index=0,
        help="Larger models are more accurate but slower. 'base' is the most reliable on small CPUs."
    )
    language_hint = st.text_input(
        "Language hint (optional)",
        value="",
        help="e.g., 'en' or 'English'. Leave blank to auto-detect."
    )
    translate_to_en = st.checkbox(
        "Translate to English (if supported by model)", value=False
    )
    max_speakers = st.slider(
        "Max speakers to try (auto-estimate within 1..N)",
        min_value=1, max_value=6, value=3,
        help="The app will try 1..N speakers and pick the best clustering by silhouette score."
    )
    min_segment_sec = st.slider(
        "Merge very short segments (seconds, 0 = off)", 0.0, 2.0, 0.3, 0.1,
        help="Post-process to merge very short turns with neighbors, reducing spurious switches."
    )

uploaded = st.file_uploader(
    "Upload audio/video",
    type=[
        "mp3","mp4","mpeg","mpga","m4a","wav","webm","flac","ogg","opus","mkv","mov","avi"
    ],
)

def _safe_filename(name: str) -> str:
    keep = "-_.() "
    return "".join(c for c in name if c.isalnum() or c in keep).strip(
