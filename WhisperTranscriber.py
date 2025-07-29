# WhisperTranscriber.py
# Streamlit app for transcription with optional speaker labeling
# - Uses OpenAI Whisper for ASR
# - Optional "speaker labels" by clustering Whisper segments with ECAPA embeddings
# - Generates TXT, SRT, VTT for download
# - Converts any input to WAV using imageio-ffmpeg's bundled binary (no system ffmpeg needed)

import io
import os
import json
import tempfile
import subprocess
import datetime
import wave
import contextlib
from typing import List, Dict, Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

import streamlit as st

# Torch/Whisper
import torch
import whisper

# For WAV conversion without system ffmpeg
import imageio_ffmpeg

# Optional speaker labeling deps (lightweight usage)
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding


# ------------------------------ UI --------------------------------
st.set_page_config(page_title="Whisper Transcriber", layout="wide")
st.title("Whisper Transcriber (Streamlit)")
st.caption(
    "Transcribe audio/video with Whisper. Optionally add simple speaker labels by clustering segments."
)

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
        help="Choose 'auto' for language detection, or specify a BCP-47 code (e.g., 'es', 'fr')."
    )
    language_code = None
    if language_choice == "English":
        language_code = "en"
    elif language_choice == "Specify code…":
        language_code = st.text_input("Language code (e.g., en, es, fr, de, pt-BR)", value="en").strip() or None

    use_diarization = st.checkbox(
        "Add speaker labels (cluster Whisper segments)",
        value=True,
        help="Approximates diarization by clustering per-segment embeddings."
    )
    num_speakers = st.number_input(
        "Number of speakers (if labeling)",
        min_value=1, max_value=20, value=2, step=1
    )
    show_timestamps = st.checkbox(
        "Show timestamps in TXT output",
        value=True
    )

uploaded = st.file_uploader(
    "Upload an audio or video file",
    type=["wav", "mp3", "m4a", "mp4", "mov", "aac", "flac", "ogg", "webm", "wma", "mkv", "avi"]
)
if uploaded:
    # Streamlit can preview audio; video previews depend on format support
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
def load_speaker_embedding_model(device: str):
    # Uses SpeechBrain ECAPA model via pyannote wrapper
    return PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=device)

def get_ffmpeg_bin() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()

def ensure_wav(tmp_dir: str, input_path: str) -> str:
    """
    Convert the input file to WAV using the bundled ffmpeg binary (if not already a WAV).
    """
    if input_path.lower().endswith(".wav"):
        return input_path
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

def read_duration_seconds(wav_path: str) -> float:
    with contextlib.closing(wave.open(wav_path, "rb")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def transcribe_file(model, wav_path: str, device: str, language_code: str | None):
    """
    Call Whisper with sane defaults for CPU/GPU. Disable fp16 on CPU.
    """
    kwargs = {
        "language": language_code,
        "verbose": False,
    }
    if device == "cpu":
        kwargs["fp16"] = False
    # If user chose English and model is not 'large', you could optionally switch to .en variant
    # but we’ll keep the exact model selection from the UI for clarity.
    result = model.transcribe(wav_path, **kwargs)
    return result

def compute_segment_embeddings(
    wav_path: str,
    segments: List[Dict[str, Any]],
    duration: float,
    embedding_model
):
    audio = Audio()
    def segment_embedding(seg):
        # Whisper sometimes overshoots last segment end; clamp to file duration
        start = float(seg["start"])
        end = min(duration, float(seg["end"]))
        clip = Segment(start, end)
        waveform, _sr = audio.crop(wav_path, clip)
        emb = embedding_model(waveform[None]).detach().cpu().numpy()  # (1, 192)
        return emb.squeeze()

    embs = np.zeros((len(segments), 192), dtype=np.float32)
    for i, seg in enumerate(segments):
        embs[i] = segment_embedding(seg)
    return np.nan_to_num(embs)

def label_speakers(segments: List[Dict[str, Any]], labels: np.ndarray):
    for i, seg in enumerate(segments):
        seg["speaker"] = f"SPEAKER {int(labels[i]) + 1}"
    return segments

def srt_time(s: float) -> str:
    # SRT time: HH:MM:SS,mmm
    ms_total = int(round(s * 1000))
    ms = ms_total % 1000
    sec = (ms_total // 1000) % 60
    minute = (ms_total // 60000) % 60
    hour = ms_total // 3600000
    return f"{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}"

def vtt_time(s: float) -> str:
    # VTT time: HH:MM:SS.mmm
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
            if show_timestamps:
                buf.append(f"\n{speaker} {ts(float(seg['start']))}\n")
            else:
                buf.append(f"\n{speaker}\n")
            last_speaker = speaker
        text = (seg.get("text") or "").strip()
        if text:
            buf.append(text + " ")
    return "".join(buf).strip()

def to_json(segments: List[Dict[str, Any]]) -> str:
    # Handy for downstream processing
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
            # Persist uploaded file to disk for ffmpeg/pyannote to read
            in_path = os.path.join(tmpdir, uploaded.name)
            with open(in_path, "wb") as f:
                f.write(uploaded.read())

            # Convert to WAV
            with st.spinner("Converting to WAV…"):
                wav_path = ensure_wav(tmpdir, in_path)

            # Transcribe
            with st.spinner("Transcribing… this can take a while for large models/files"):
                result = transcribe_file(model, wav_path, device, language_code)
                segments = result.get("segments", [])
                if not segments:
                    st.error("No segments returned by Whisper.")
                    st.stop()

            # Optional "speaker labeling" by clustering embeddings
            if use_diarization and len(segments) > 1:
                try:
                    with st.spinner("Computing speaker embeddings…"):
                        embed_model = load_speaker_embedding_model(device)
                        duration = read_duration_seconds(wav_path)
                        embs = compute_segment_embeddings(wav_path, segments, duration, embed_model)

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
