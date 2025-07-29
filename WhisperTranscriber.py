# streamlit_app.py
import io
import os
import tempfile
import subprocess
import datetime
import wave
import contextlib

import numpy as np
from sklearn.cluster import AgglomerativeClustering

import torch
import whisper

import streamlit as st

# ---- Optional speaker labeling deps (same as your script) ----
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

# -------------------------- UI --------------------------------
st.set_page_config(page_title="Whisper Transcriber", layout="wide")
st.title("Whisper Transcription (with optional speaker labels)")

with st.sidebar:
    st.header("Settings")
    model_size = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small", "medium", "large"],
        index=4,
    )
    language = st.selectbox("Language", ["auto", "English"], index=1)
    use_diarization = st.checkbox("Add speaker labels (clustering over Whisper segments)", value=True)
    num_speakers = st.number_input("Number of speakers (if labeling)", min_value=1, max_value=20, value=2, step=1)
    show_timestamps = st.checkbox("Show segment timestamps in text output", value=True)

uploaded = st.file_uploader("Upload an audio or video file", type=[
    "wav", "mp3", "m4a", "mp4", "mov", "aac", "flac", "ogg", "webm"
])
if uploaded:
    st.audio(uploaded, format=None)

# --------------------- Helpers & Caching -----------------------
@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str):
    # GPU if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size, device=device)
    return model, device

@st.cache_resource(show_spinner=False)
def load_speaker_embedding_model(device: str):
    # same model you used: speechbrain/ecapa via pyannote wrapper
    return PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=device)

def ensure_wav(tmp_dir: str, input_path: str) -> str:
    # Whisper and pyannote pipeline both happy with wav; convert when necessary (requires ffmpeg)
    if input_path.lower().endswith(".wav"):
        return input_path
    wav_path = os.path.join(tmp_dir, "audio.wav")
    _ = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, wav_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return wav_path

def transcribe_file(model, wav_path: str, language: str):
    if language == "English" and not model.model_name.endswith(".en") and model.model_name != "large":
        # If user asked for English and model isn't large, you can hint/override
        pass
    result = model.transcribe(wav_path, language=None if language == "auto" else "en")
    return result

def read_duration_seconds(wav_path: str) -> float:
    with contextlib.closing(wave.open(wav_path, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def compute_segment_embeddings(wav_path: str, segments, duration: float, embedding_model):
    audio = Audio()
    def segment_embedding(seg):
        start = seg["start"]
        end = min(duration, seg["end"])  # whisper sometimes overshoots final segment end
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(wav_path, clip)
        return embedding_model(waveform[None]).detach().cpu().numpy()

    embs = np.zeros((len(segments), 192), dtype=np.float32)
    for i, seg in enumerate(segments):
        embs[i] = segment_embedding(seg).squeeze()
    return np.nan_to_num(embs)

def label_speakers(segments, labels):
    for i, seg in enumerate(segments):
        seg["speaker"] = f"SPEAKER {labels[i] + 1}"
    return segments

def to_srt(segments):
    def srt_time(s):
        td = datetime.timedelta(seconds=round(s, 3))
        # SRT format: HH:MM:SS,mmm
        total_ms = int(td.total_seconds() * 1000)
        ms = total_ms % 1000
        sec = (total_ms // 1000) % 60
        minute = (total_ms // 60000) % 60
        hour = total_ms // 3600000
        return f"{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(segments, 1):
        start = srt_time(seg["start"])
        end = srt_time(seg["end"])
        who = f"{seg.get('speaker', '').strip()} " if seg.get("speaker") else ""
        text = seg["text"].strip()
        lines.append(f"{i}\n{start} --> {end}\n{who}{text}\n")
    return "\n".join(lines)

def to_vtt(segments):
    out = ["WEBVTT\n"]
    def vtt_time(s):
        td = datetime.timedelta(seconds=round(s, 3))
        total_ms = int(td.total_seconds() * 1000)
        ms = total_ms % 1000
        sec = (total_ms // 1000) % 60
        minute = (total_ms // 60000) % 60
        hour = total_ms // 3600000
        return f"{hour:02d}:{minute:02d}:{sec:02d}.{ms:03d}"
    for seg in segments:
        out.append(f"{vtt_time(seg['start'])} --> {vtt_time(seg['end'])}")
        who = f"{seg.get('speaker', '').strip()} " if seg.get("speaker") else ""
        text = seg["text"].strip()
        out.append(f"{who}{text}\n")
    return "\n".join(out)

def to_text(segments, show_timestamps=True):
    def ts(s): 
        return str(datetime.timedelta(seconds=round(s))) 
    lines = []
    last_speaker = None
    for i, seg in enumerate(segments):
        speaker = seg.get("speaker")
        if speaker and speaker != last_speaker:
            if show_timestamps:
                lines.append(f"\n{speaker} {ts(seg['start'])}")
            else:
                lines.append(f"\n{speaker}")
            last_speaker = speaker
        lines.append(seg["text"].strip() + " ")
    return "".join(lines).strip()

# -------------------------- Run -------------------------------
if uploaded and st.button("Transcribe"):
    with st.spinner("Loading model…"):
        model, device = load_whisper(model_size)

    with tempfile.TemporaryDirectory() as tmpdir:
        # persist upload to disk for ffmpeg/pyannote
        in_path = os.path.join(tmpdir, uploaded.name)
        with open(in_path, "wb") as f:
            f.write(uploaded.read())
        wav_path = ensure_wav(tmpdir, in_path)

        with st.spinner("Transcribing…"):
            result = transcribe_file(model, wav_path, language)
            segments = result.get("segments", [])
            if not segments:
                st.error("No segments returned by Whisper.")
                st.stop()

        if use_diarization and len(segments) > 1:
            with st.spinner("Computing speaker embeddings…"):
                embed_model = load_speaker_embedding_model(device)
                duration = read_duration_seconds(wav_path)
                embs = compute_segment_embeddings(wav_path, segments, duration, embed_model)
            with st.spinner("Clustering speakers…"):
                clustering = AgglomerativeClustering(num_speakers).fit(embs)
                labels = clustering.labels_
                segments = label_speakers(segments, labels)

        st.success("Done!")

        # ---- Outputs ----
        txt = to_text(segments, show_timestamps=show_timestamps)
        srt = to_srt(segments)
        vtt = to_vtt(segments)

        st.subheader("Transcript")
        st.write(txt)

        st.download_button("Download TXT", data=txt.encode("utf-8"), file_name="transcript.txt")
        st.download_button("Download SRT", data=srt.encode("utf-8"), file_name="transcript.srt")
        st.download_button("Download VTT", data=vtt.encode("utf-8"), file_name="transcript.vtt")
