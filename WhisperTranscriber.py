import os
import io
import math
import tempfile
from datetime import timedelta
from typing import List, Tuple, Optional

import streamlit as st

# ---- Ensure ffmpeg is available (bundled) ----
# We rely on imageio-ffmpeg's statically linked ffmpeg so we don't need system packages.
import imageio_ffmpeg
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")

# ---- Whisper and audio utils ----
import whisper
from whisper.utils import format_timestamp

# Lightweight feature extraction for diarization
import numpy as np
import librosa
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Optional: torchaudio can help ensure the CPU-only stack is available,
# but we don't import it unless needed to avoid extra overhead.

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
4) Let you **download** a timestamped plain-text transcript with **speaker labels**  
    """
)

with st.expander("⚙️ Settings"):
    model_size = st.selectbox(
        "Whisper model",
        ["base", "small", "medium"],  # keep sizes modest for Streamlit CPU
        index=1,
        help="Larger models are more accurate but slower. 'small' is a good default."
    )
    language_hint = st.text_input(
        "Language hint (optional)",
        value="",
        help="e.g., 'en' or 'English'. Leave blank to auto-detect."
    )
    enable_translate = st.checkbox(
        "Translate to English (if supported by model)", value=False
    )
    max_speakers = st.slider(
        "Max speakers to try (auto-estimate within 1..N)",
        min_value=1, max_value=6, value=3,
        help="The app will try 1..N speakers and pick the best clustering by silhouette score."
    )
    min_segment_sec = st.slider(
        "Merge very short segments (seconds, 0 = off)", 0.0, 2.0, 0.5, 0.1,
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
    return "".join(c for c in name if c.isalnum() or c in keep).strip().replace(" ", "_")

@st.cache_resource(show_spinner=False)
def load_whisper(model_name: str):
    # Whisper will download model weights on first run and cache them
    return whisper.load_model(model_name)

def _to_tmp_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name

def _media_to_wav(input_path: str, sr: int = 16000) -> str:
    """
    Converts any media to mono WAV 16k using bundled ffmpeg for consistent processing.
    """
    out_path = tempfile.mktemp(suffix=".wav")
    # -ac 1 mono, -ar 16000 sample rate
    cmd = f'"{ffmpeg_path}" -y -i "{input_path}" -ac 1 -ar {sr} "{out_path}"'
    code = os.system(cmd)
    if code != 0 or not os.path.exists(out_path):
        raise RuntimeError("ffmpeg failed to convert media to WAV.")
    return out_path

def _merge_short_segments(segments, threshold_sec: float):
    if threshold_sec <= 0 or len(segments) <= 1:
        return segments
    merged = []
    cur = segments[0].copy()
    for nxt in segments[1:]:
        if (cur["end"] - cur["start"]) < threshold_sec:
            # merge into next
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
    """
    Extract MFCCs over [start,end] seconds and return mean+std as a compact embedding.
    """
    start_samp = max(0, int(start * sr))
    end_samp = min(len(y), int(end * sr))
    if end_samp <= start_samp:
        end_samp = min(len(y), start_samp + int(0.1*sr))  # ensure tiny window
    clip = y[start_samp:end_samp]
    if clip.size == 0:
        clip = y[max(0, start_samp - int(0.1*sr)):min(len(y), start_samp + int(0.1*sr))]
    mfcc = librosa.feature.mfcc(y=clip.astype(np.float32), sr=sr, n_mfcc=n_mfcc)
    mu = mfcc.mean(axis=1)
    sd = mfcc.std(axis=1)
    return np.concatenate([mu, sd])

def diarize_by_clustering(wav_path: str,
                          segments: List[dict],
                          max_speakers_try: int = 3) -> Tuple[List[dict], int]:
    """
    Build MFCC-based embeddings for each Whisper segment and cluster.
    Try k=1..max_speakers_try; pick the one with best silhouette score.
    Returns updated segments with 'speaker' and the chosen n_speakers.
    """
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    if not segments:
        return segments, 1

    feats = []
    for s in segments:
        feats.append(_mfcc_stats(y, sr, s["start"], s["end"]))
    X = np.vstack(feats)

    # Edge case: only one segment
    if len(segments) == 1 or max_speakers_try <= 1:
        for s in segments:
            s["speaker"] = "SPEAKER 1"
        return segments, 1

    best_labels, best_score, best_k = None, -1.0, 1
    for k in range(1, max_speakers_try + 1):
        if k == 1:
            labels = np.zeros(len(segments), dtype=int)
            score = -1.0
        else:
            try:
                clustering = AgglomerativeClustering(n_clusters=k, linkage="ward")
                labels = clustering.fit_predict(X)
                # Ward uses Euclidean distance; silhouette on Euclidean:
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X, labels, metric="euclidean")
                else:
                    score = -1.0
            except Exception:
                score = -1.0
                labels = np.zeros(len(segments), dtype=int)
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels

    # Assign speakers
    for i, s in enumerate(segments):
        s["speaker"] = f"SPEAKER {int(best_labels[i]) + 1}"
    return segments, best_k

def segments_to_text(segments: List[dict]) -> str:
    """
    Build a timestamped, speaker-labeled plain text transcript.
    """
    out = io.StringIO()
    last_speaker = None
    for seg in segments:
        spk = seg.get("speaker", "SPEAKER 1")
        if spk != last_speaker:
            out.write(f"\n{spk} {format_timestamp(seg['start'])}\n")
            last_speaker = spk
        # Whisper sometimes includes leading space/punctuations
        text = seg.get("text", "").lstrip()
        out.write(text + " ")
    return out.getvalue().strip() + "\n"

def build_download_name(original_name: str) -> str:
    base = os.path.splitext(_safe_filename(original_name or "transcript"))[0]
    return f"{base}_diarized.txt"

# ---------- Main action ----------
if uploaded is not None:
    st.video(uploaded) if uploaded.type.startswith("video/") else st.audio(uploaded)

    do_transcribe = st.button("🔎 Transcribe & Diarize")
    if do_transcribe:
        with st.spinner("Loading model…"):
            model = load_whisper(model_size)

        # Save upload to disk
        in_path = _to_tmp_file(uploaded)

        try:
            with st.spinner("Converting media to WAV…"):
                wav_path = _media_to_wav(in_path, sr=16000)

            decode_opts = {"task": "translate"} if enable_translate else {}
            if language_hint.strip():
                # Whisper accepts either language name or code; we’ll pass as-is
                decode_opts["language"] = language_hint.strip()

            with st.spinner("Transcribing with Whisper…"):
                result = model.transcribe(wav_path, **decode_opts)
                segments = result.get("segments", []) or []

            if min_segment_sec > 0:
                segments = _merge_short_segments(segments, min_segment_sec)

            with st.spinner("Diarizing (clustering per-segment features)…"):
                segments, chosen_k = diarize_by_clustering(
                    wav_path, segments, max_speakers_try=max_speakers
                )

            st.success(f"Done! Estimated speakers: {chosen_k}")

            # Show a quick preview
            st.subheader("Transcript (preview)")
            preview = segments_to_text(segments)
            st.text_area("Transcript", preview, height=300)

            # Provide a download
            fname = build_download_name(uploaded.name)
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
            try:
                if os.path.exists(in_path): os.remove(in_path)
            except Exception:
                pass
