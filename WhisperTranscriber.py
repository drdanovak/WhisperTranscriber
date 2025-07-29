# WhisperTranscriber.py
# Streamlit app for transcription with optional speaker labeling
# - OpenAI Whisper for ASR
# - Optional "speaker labels" by clustering ECAPA embeddings (SpeechBrain)
# - Outputs: TXT, SRT, VTT, JSON
# - Uses imageio-ffmpeg binary so no system ffmpeg is required

import os
import io
import json
import tempfile
import subprocess
import datetime
from typing import List, Dict, Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

import streamlit as st

# Core ASR
import torch
import whisper

# WAV conversion without system ffmpeg
import imageio_ffmpeg

# Audio I/O for segment slicing
import soundfile as sf

# SpeechBrain ECAPA speaker embeddings
from speechbrain.pretrained import EncoderClassifier


# ------------------------------ UI --------------------------------
st.set_page_config(page_title="Whisper Transcriber", layout="wide")
st.title("Whisper Transcriber (Streamlit)")
st.caption("Transcribe audio/video with Whisper. Optionally add simple speaker labels by clustering ECAPA embeddings.")

with st.sidebar:
    st.header("Settings")
    model_size = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small", "medium", "large"],
        index=4,
        help="Larger models are more accurate but slower."
    )

    language_choice = st.selectbox(
        "Language",
        ["auto", "English", "Specify code…"],
        index=1,
        help="Use 'auto' for language detection, or specify a BCP‑47 code (e.g., 'es', 'fr')."
    )
    language_code = None
    if language_choice == "English":
        language_code = "en"
    elif language_choice == "Specify code…":
        language_code = st.text_input("Language code (e.g., en, es, fr, de, pt-BR)", value="en").strip() or None

    use_speaker_labels = st.checkbox(
        "Add speaker labels (cluster ECAPA embeddings)",
        value=True,
        help="Approximates diarization by clustering per‑segment embeddings."
    )
    num_speakers = st.number_input(
        "Number of speakers (if labeling)",
        min_value=1, max_value=20, value=2, step=1
    )
    show_timestamps = st.checkbox("Show timestamps in TXT output", value=True)

uploaded = st.file_uploader(
    "Upload an audio or video file",
    type=["wav", "mp3", "m4a", "mp4", "mov", "aac", "flac", "ogg", "webm", "wma", "mkv", "avi"]
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
    # SpeechBrain ECAPA-voxceleb embedding model
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )

def get_ffmpeg_bin() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()

def ensure_wav(tmp_dir: str, input_path: str) -> str:
    """
    Convert the input file to mono 16 kHz WAV via bundled ffmpeg.
    """
    if input_path.lower().endswith(".wav"):
        # still standardize to 16k mono to simplify downstream
        wav_path = os.path.join(tmp_dir, "audio.wav")
        ffmpeg_bin = get_ffmpeg_bin()
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", input_path, "-ac", "1", "-ar", "16000", wav_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
        return wav_path if os.path.exists(wav_path) else input_path

    wav_path = os.path.join(tmp_dir, "audio.wav")
    ffmpeg_bin = get_ffmpeg_bin()
    completed = subprocess.run(
        [ffmpeg_bin, "-y", "-i", input_path, "-ac", "1", "-ar", "16000", wav_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if completed.returncode != 0 or not os.path.exists(wav_path):
        stderr = completed.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed to convert to WAV:\n{stderr[:2000]}")
    return wav_path

def transcribe_file(model, wav_path: str, device: str, language_code: str | None):
    kwargs = {
        "language": language_code,
        "verbose": False,
    }
    if device == "cpu":
        kwargs["fp16"] = False
    return model.transcribe(wav_path, **kwargs)

def compute_ecapa_embeddings(wav_path: str, segments: List[Dict[str, Any]], ecapa: EncoderClassifier):
    """
    Compute ECAPA embeddings for each Whisper segment using SpeechBrain directly.
    Assumes wav_path is mono 16kHz (we enforce this in ensure_wav).
    Returns an array of shape (N, D).
    """
    # Load full waveform once for efficient slicing
    waveform, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        # ensure mono
        waveform = waveform.mean(axis=1)
    # SpeechBrain expects torch tensors with shape (batch, time)
    embeddings = []
    for seg in segments:
        start = max(0, int(round(float(seg["start"]) * sr)))
        end = min(len(waveform), int(round(float(seg["end"]) * sr)))
        if end <= start:
            # tiny/zero-length segment: pad a small slice
            end = min(len(waveform), start + int(0.2 * sr))
        clip = torch.from_numpy(waveform[start:end]).unsqueeze(0)  # (1, T)
        with torch.no_grad():
            emb = ecapa.encode_batch(clip)  # (1, 192)
        embeddings.append(emb.squeeze(0).squeeze(0).cpu().numpy())
    embs = np.vstack(embeddings).astype(np.float32)
    # Avoid NaNs in clustering
    return np.nan_to_num(embs)

def label_speakers(segments: List[Dict[str, Any]], labels: np.ndarray):
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

def vtt_time(s: float) -> str:
    ms_total = int(round(s * 1000))
    ms = ms_total % 1000
    sec = (ms_total // 1000) % 60
    minute = (ms_total // 60000) % 60
    hour = ms_total // 3600000
    return f"{hour:02d}:{minute:02d}:{sec:02d}.{ms:03d}"

def to_srt(segments: List[Dict[str, Any]]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = srt_time(float(seg["start"]))
        end = srt_time(float(seg["end"]))
        who = f"{seg.get('speaker', '').strip()} " if seg.get("speaker") else ""
        text = (seg.get("text") or "").strip()
        lines.append(f"{i}\n{start} --> {end}\n{who}{text}\n")
    return "\n".join(lines)

def to_vtt(segments: List[Dict[str, Any]]) -> str:
    out = ["WEBVTT\n"]
    for seg in segments:
        out.append(f"{vtt_time(float(seg['start']))} --> {vtt_time(float(seg['end']))}")
        who = f"{seg.get('speaker', '').strip()} " if seg.get("speaker") else ""
        text = (seg.get("text") or "").strip()
        out.append(f"{who}{text}\n")
    return "\n".join(out)

def to_txt(segments: List[Dict[str, Any]], show_timestamps: bool) -> str:
    def ts(s: float) -> str:
        return str(datetime.timedelta(seconds=round(s)))
    buf = []
    last_speaker = None
    for seg in segments:
        speaker = seg.get("speaker")
        if speaker and speaker != last_speaker:
            buf.append(f"\n{speaker}")
            if show_timestamps:
                buf.append(f" {ts(float(seg['start']))}")
            buf.append("\n")
            last_speaker = speaker
        text = (seg.get("text") or "").strip()
        if text:
            buf.append(text + " ")
    return "".join(buf).strip()

def to_json(segments: List[Dict[str, Any]]) -> str:
    cleaned = [
        {
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": (s.get("text") or "").strip(),
            "speaker": s.get("speaker"),
        }
        for s in segments
    ]
    return json.dumps(cleaned, indent=2)


# ------------------------------- Run -------------------------------
if uploaded and st.button("Transcribe"):
    try:
        with st.spinner("Loading Whisper…"):
            model, device = load_whisper(model_size)
        st.info(f"Using device: **{device.upper()}**")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save upload
            in_path = os.path.join(tmpdir, uploaded.name)
            with open(in_path, "wb") as f:
                f.write(uploaded.read())

            # Convert to WAV (16 kHz mono)
            with st.spinner("Converting to WAV…"):
                wav_path = ensure_wav(tmpdir, in_path)

            # Transcribe
            with st.spinner("Transcribing… (large files/models may take a while)"):
                result = transcribe_file(model, wav_path, device, language_code)
                segments = result.get("segments", [])
                if not segments:
                    st.error("No segments returned by Whisper.")
                    st.stop()

            # Optional speaker labeling
            if use_speaker_labels and len(segments) > 1:
                try:
                    with st.spinner("Computing speaker embeddings…"):
                        ecapa = load_ecapa(device)
                        embs = compute_ecapa_embeddings(wav_path, segments, ecapa)

                    with st.spinner("Clustering speakers…"):
                        clustering = AgglomerativeClustering(num_speakers).fit(embs)
                        labels = clustering.labels_
                        segments = label_speakers(segments, labels)
                except Exception as e:
                    st.warning(
                        "Speaker labeling failed; continuing with plain transcript.\n\n"
                        f"Details: {e}"
                    )

            # ---------------- Outputs ----------------
            txt = to_txt(segments, show_timestamps=show_timestamps)
            srt = to_srt(segments)
            vtt = to_vtt(segments)
            js = to_json(segments)

            st.subheader("Transcript")
            st.write(txt if txt else "(empty)")

            st.download_button("Download TXT", data=txt.encode("utf-8"), file_name="transcript.txt")
            st.download_button("Download SRT", data=srt.encode("utf-8"), file_name="transcript.srt")
            st.download_button("Download VTT", data=vtt.encode("utf-8"), file_name="transcript.vtt")
            st.download_button("Download JSON", data=js.encode("utf-8"), file_name="segments.json")

    except Exception as e:
        st.error(f"Unexpected error: {e}")
