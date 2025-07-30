import io
import os
import tempfile
import subprocess
from datetime import timedelta
from typing import List, Tuple

import streamlit as st

# ---------- Ensure a working ffmpeg (bundled) ----------
# imageio-ffmpeg wheels ship a statically linked ffmpeg binary.
import imageio_ffmpeg  # installs an ffmpeg exe inside the wheel
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
# Make the binary discoverable for anything that shells out to "ffmpeg"
os.environ.setdefault("IMAGEIO_FFMPEG_EXE", FFMPEG_EXE)
os.environ["PATH"] = os.path.dirname(FFMPEG_EXE) + os.pathsep + os.environ.get("PATH", "")

# ---------- Whisper ----------
import whisper
from whisper.utils import format_timestamp

# ---------- Diarization: lightweight, no external model ----------
import numpy as np
import librosa
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ---------- Page ----------
st.set_page_config(page_title="Whisper Diarized Transcriber", layout="centered")
st.title("🎙️ Whisper Diarized Transcriber")
st.caption("Upload audio/video → transcribe with Whisper → diarize → download timestamped transcript")

with st.expander("⚙️ Settings"):
    model_size = st.selectbox("Whisper model", ["base", "small", "medium"], index=1)
    language_hint = st.text_input("Language hint (optional)", value="")
    translate_to_en = st.checkbox("Translate to English (if supported)", value=False)
    max_speakers = st.slider("Max speakers to try (auto-select 1..N)", 1, 6, 3)
    min_segment_sec = st.slider("Merge very short segments (sec, 0=off)", 0.0, 2.0, 0.5, 0.1)

uploaded = st.file_uploader(
    "Upload audio/video",
    type=["mp3","mp4","mpeg","mpga","m4a","wav","webm","flac","ogg","opus","mkv","mov","avi"],
)

# ---------- Helpers ----------
def _safe_filename(name: str) -> str:
    keep = "-_.() "
    return "".join(c for c in name if c.isalnum() or c in keep).strip().replace(" ", "_")

@st.cache_resource(show_spinner=False)
def load_whisper(model_name: str):
    return whisper.load_model(model_name)

def _save_to_tmp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower() or ".bin"
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(uploaded_file.read())
    f.flush()
    f.close()
    return f.name

def _media_to_wav(input_path: str, sr: int = 16000) -> str:
    """
    Convert any media to mono 16 kHz WAV via the bundled ffmpeg.
    """
    out_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        FFMPEG_EXE, "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", str(sr),
        out_path,
    ]
    # Use subprocess for reliability and surfacing stderr on failure
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode(errors='ignore')[:500]}") from e
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
            # merge with next
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
    Extract MFCCs in [start, end] and return mean+std vector.
    """
    s0 = max(0, int(start * sr))
    s1 = min(len(y), int(end * sr))
    if s1 <= s0:
        s1 = min(len(y), s0 + int(0.1 * sr))
    clip = y[s0:s1]
    if clip.size == 0:
        # fallback tiny window
        clip = y[max(0, s0 - int(0.1*sr)):min(len(y), s0 + int(0.1*sr))]
    mfcc = librosa.feature.mfcc(y=clip.astype(np.float32), sr=sr, n_mfcc=n_mfcc)
    mu = mfcc.mean(axis=1)
    sd = mfcc.std(axis=1)
    return np.concatenate([mu, sd])

def diarize_by_clustering(wav_path: str, segments: List[dict], max_k: int) -> Tuple[List[dict], int]:
    """
    Build MFCC embeddings per Whisper segment and cluster.
    Try k=1..max_k and select the one with the best silhouette score (Euclidean).
    """
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
        out.write(seg.get("text", "").lstrip() + " ")
    return out.getvalue().strip() + "\n"

def _download_name(original: str) -> str:
    base = os.path.splitext(_safe_filename(original or "transcript"))[0]
    return f"{base}_diarized.txt"

# ---------- Main ----------
if uploaded is not None:
    # Preview
    if uploaded.type.startswith("video/"):
        st.video(uploaded)
    else:
        st.audio(uploaded)

    if st.button("🔎 Transcribe & Diarize"):
        tmp_in = _save_to_tmp(uploaded)
        try:
            with st.spinner("Converting media to WAV…"):
                wav_path = _media_to_wav(tmp_in, sr=16000)

            with st.spinner("Loading Whisper model…"):
                model = load_whisper(model_size)

            decode_opts = {"task": "translate"} if translate_to_en else {}
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

            # Preview & download
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
            for p in [tmp_in, locals().get("wav_path")]:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
else:
    st.info("Upload an audio or video file to get started.")
