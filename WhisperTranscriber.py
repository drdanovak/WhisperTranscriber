# app.py
# Whisper diarized transcription in Streamlit, designed to be foolproof on Streamlit Cloud.

import os
import io
import json
import tempfile
import subprocess
import datetime
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import streamlit as st

# ASR
import torch
import whisper

# Audio I/O / conversion
import soundfile as sf
import imageio_ffmpeg


# ------------------------------ UI --------------------------------
st.set_page_config(page_title="Whisper Diarized Transcriber", layout="centered")
st.title("Whisper Diarized Transcriber")
st.caption("Upload audio/video → transcribe with Whisper → cluster speakers → download a timestamped .txt transcript.")

with st.sidebar:
    st.header("Settings")
    model_size = st.selectbox(
        "Whisper model",
        ["small", "medium", "large"],
        index=0,
        help="Larger models are more accurate but slower."
    )
    language_mode = st.selectbox(
        "Language",
        ["Auto-detect", "English", "Specify code…"],
        index=1,
        help="Use 'Auto-detect' or specify a language code (e.g., en, es, fr)."
    )
    language_code: Optional[str] = None
    if language_mode == "English":
        language_code = "en"
    elif language_mode == "Specify code…":
        language_code = st.text_input("Language code (BCP-47)", value="en").strip() or None

    auto_speakers = st.checkbox(
        "Auto-detect number of speakers",
        value=True,
        help="Tries 1–6 and picks the best clustering. Uncheck to set it manually."
    )
    max_auto = st.slider("Auto-detect: max speakers to try", 2, 8, 6, disabled=not auto_speakers)
    num_speakers = st.number_input(
        "Speakers (manual)", min_value=1, max_value=20, value=2, step=1, disabled=auto_speakers
    )

    show_timestamps = st.checkbox("Show timestamps in TXT", value=True)

uploaded = st.file_uploader(
    "Upload audio or video",
    type=[
        "wav", "mp3", "m4a", "mp4", "mov", "aac", "flac", "ogg", "webm",
        "wma", "mkv", "avi", "m4v", "3gp"
    ],
    accept_multiple_files=False
)
if uploaded:
    try:
        st.audio(uploaded)
    except Exception:
        pass

# --------------------------- Helpers & Cache -----------------------
@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size, device=device)
    return model, device

@st.cache_resource(show_spinner=False)
def load_ecapa(device: str):
    # Lazy import to keep top-level import list minimal
    from speechbrain.pretrained import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )

def ffmpeg_bin() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()

def to_wav_mono16k(tmpdir: str, infile_path: str) -> str:
    """
    Convert any media to 16kHz mono WAV using a bundled ffmpeg binary.
    """
    wav_path = os.path.join(tmpdir, "audio_16k_mono.wav")
    cmd = [
        ffmpeg_bin(), "-y",
        "-i", infile_path,
        "-ac", "1",     # mono
        "-ar", "16000", # 16 kHz
        wav_path
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not os.path.exists(wav_path):
        err = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg conversion failed:\n{err[:2000]}")
    return wav_path

def transcribe_whisper(model, wav_path: str, device: str, language_code: Optional[str]):
    kwargs = {
        "language": language_code,
        "verbose": False,
    }
    if device == "cpu":
        kwargs["fp16"] = False
    return model.transcribe(wav_path, **kwargs)

def compute_ecapa_embeddings(wav_path: str, segments: List[Dict[str, Any]], device: str):
    """
    Compute an ECAPA embedding per Whisper segment using SpeechBrain.
    - Uses soundfile to slice the 16k mono WAV.
    - Returns (N, D) float32 array.
    """
    import torch as _torch
    ecapa = load_ecapa(device)

    # Load mono waveform (float32)
    wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # safety

    embs = []
    for seg in segments:
        start_s = max(0.0, float(seg["start"]))
        end_s = max(start_s, float(seg["end"]))
        s_idx = int(round(start_s * sr))
        e_idx = min(len(wav), int(round(end_s * sr)))
        if e_idx <= s_idx:  # guard against zero-length
            e_idx = min(len(wav), s_idx + int(0.2 * sr))
        clip = _torch.from_numpy(wav[s_idx:e_idx]).unsqueeze(0)  # (1, T)
        with _torch.no_grad():
            emb = ecapa.encode_batch(clip)  # (1, 1, 192)
        embs.append(emb.squeeze().cpu().numpy())  # (192,)
    embs = np.vstack(embs).astype(np.float32)
    return np.nan_to_num(embs)

def pick_num_speakers(embs: np.ndarray, max_k: int) -> int:
    """
    Choose K in [1..max_k] by maximizing silhouette score (if K>1).
    If all poor, fall back to 1.
    """
    if embs.shape[0] < 3:
        return 1
    best_k, best_score = 1, -1.0
    for k in range(2, max(2, min(max_k, embs.shape[0])) + 1):
        labels = AgglomerativeClustering(k).fit_predict(embs)
        # Need at least 2 clusters and less than n_samples unique labels
        if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
            continue
        try:
            score = silhouette_score(embs, labels)
            if score > best_score:
                best_score, best_k = score, k
        except Exception:
            # Occasionally fails for degenerate embeddings; ignore and continue
            pass
    return best_k

def apply_speaker_labels(segments: List[Dict[str, Any]], labels: np.ndarray):
    for i, seg in enumerate(segments):
        seg["speaker"] = f"SPEAKER {int(labels[i]) + 1}"
    return segments

def srt_time(s: float) -> str:
    ms_total = int(round(s * 1000))
    ms = ms_total % 1000
    sec = (ms_total // 1000) % 60
    minute = (ms_total // 60000) % 60
    hour = ms_total // 3600000
    return f"{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}"

def to_txt(segments: List[Dict[str, Any]], show_ts: bool) -> str:
    def hhmmss(s: float) -> str:
        return str(datetime.timedelta(seconds=round(s)))
    out = []
    last = None
    for seg in segments:
        who = seg.get("speaker")
        if who and who != last:
            line = who
            if show_ts:
                line += f" {hhmmss(float(seg['start']))}"
            out.append("\n" + line)
            last = who
        text = (seg.get("text") or "").strip()
        if text:
            out.append(text + " ")
    return "".join(out).strip()

def timestamped_filename(base: str, ext: str) -> str:
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(base))[0] or "transcript"
    return f"{base}_transcript_{ts}.{ext}"


# ------------------------------- Run -------------------------------
if uploaded and st.button("Transcribe"):
    try:
        with st.spinner("Loading Whisper…"):
            model, device = load_whisper(model_size)
        st.info(f"Using device: **{device.upper()}**")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save upload to disk
            in_path = os.path.join(tmpdir, uploaded.name)
            with open(in_path, "wb") as f:
                f.write(uploaded.read())

            # 1) Convert to 16k mono WAV
            with st.spinner("Converting to WAV…"):
                wav_path = to_wav_mono16k(tmpdir, in_path)

            # 2) Transcribe with Whisper
            with st.spinner("Transcribing…"):
                result = transcribe_whisper(model, wav_path, device, language_code)
                segments = result.get("segments", [])
                if not segments:
                    st.error("No segments returned by Whisper.")
                    st.stop()

            # 3) Diarize (cluster embeddings)
            with st.spinner("Labeling speakers…"):
                labels_used = False
                try:
                    embs = compute_ecapa_embeddings(wav_path, segments, device)
                    k = (pick_num_speakers(embs, max_auto) if auto_speakers else int(num_speakers))
                    if k > 1:
                        labels = AgglomerativeClustering(k).fit_predict(embs)
                        apply_speaker_labels(segments, labels)
                        labels_used = True
                except Exception as e:
                    st.warning(
                        "Speaker labeling failed; continuing with plain transcript.\n\n"
                        f"Details: {e}"
                    )

            # 4) Render + download
            txt = to_txt(segments, show_timestamps)
            st.subheader("Transcript")
            st.write(txt if txt else "(empty)")

            fname = timestamped_filename(uploaded.name, "txt")
            st.download_button("Download transcript (.txt)", data=txt.encode("utf-8"), file_name=fname)

            if labels_used:
                st.success("Done! Transcription + speaker labels ready.")
            else:
                st.success("Done! Transcription ready.")

    except Exception as e:
        st.error(f"Unexpected error: {e}")
