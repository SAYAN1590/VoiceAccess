"""Accessible voice recording and audio-to-LLM Streamlit application."""

import os
import shutil
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

TEMP_AUDIO_DIR = Path("temp_audio")
MAX_AUDIO_SIZE_MB = 25
LOGGER = logging.getLogger(__name__)


def get_optional_secret(name: str) -> str | None:
    """Read a secret without failing when secrets.toml has not been created."""
    try:
        return st.secrets.get(name, os.getenv(name))
    except Exception:
        # A project can run fully in mock mode without a secrets.toml file.
        return os.getenv(name)


def configure_ffmpeg() -> None:
    """Configure pydub with explicit FFmpeg paths when available.

    This avoids relying solely on the PATH inherited by a virtual environment.
    Set FFMPEG_BINARY and FFPROBE_BINARY to full executable paths when needed.
    """
    ffmpeg = (
        get_optional_secret("FFMPEG_BINARY")
        or shutil.which("ffmpeg")
    )
    ffprobe = (
        get_optional_secret("FFPROBE_BINARY")
        or shutil.which("ffprobe")
    )

    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg was not found by this virtual environment. Activate the venv "
            "from a terminal where 'ffmpeg -version' works, or set FFMPEG_BINARY."
        )

    AudioSegment.converter = ffmpeg
    if ffprobe:
        AudioSegment.ffprobe = ffprobe


def initialize_session_state() -> None:
    """Set defaults for values retained between Streamlit reruns."""
    defaults = {
        "recording_state": "idle",
        "last_audio_signature": None,
        "mp3_path": None,
        "llm_response": None,
        "error_message": None,
        "notice_message": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def convert_audio_to_mp3(audio_file) -> Path:
    """Save the recording locally and encode it as a compact mono MP3."""
    configure_ffmpeg()
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_bytes = audio_file.getvalue()

    if not audio_bytes:
        raise ValueError("The recording is empty. Please record your voice again.")
    if len(audio_bytes) > MAX_AUDIO_SIZE_MB * 1024 * 1024:
        raise ValueError(f"Recordings must be smaller than {MAX_AUDIO_SIZE_MB} MB.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    wav_path = TEMP_AUDIO_DIR / f"recording_{timestamp}.wav"
    mp3_path = TEMP_AUDIO_DIR / f"recording_{timestamp}.mp3"

    try:
        wav_path.write_bytes(audio_bytes)
        audio = AudioSegment.from_file(wav_path, format="wav")
        audio.export(
            mp3_path,
            format="mp3",
            bitrate="128k",
            parameters=["-ac", "1", "-ar", "16000"],
        )
    except CouldntDecodeError as exc:
        raise RuntimeError("The recording could not be decoded. Confirm FFmpeg is installed.") from exc
    except Exception as exc:
        raise RuntimeError(f"MP3 conversion failed: {exc}") from exc
    finally:
        if wav_path.exists():
            wav_path.unlink()

    if not mp3_path.exists() or mp3_path.stat().st_size == 0:
        raise RuntimeError("MP3 conversion did not create a valid audio file.")
    return mp3_path


def send_audio_to_llm(mp3_path: Path) -> str:
    """Mock integration point for an audio-capable LLM provider."""
    api_key = get_optional_secret("OPENAI_API_KEY")
    if not api_key:
        return (
            "Mock response: MP3 created successfully. Add OPENAI_API_KEY to "
            ".streamlit/secrets.toml to connect a real audio-capable LLM."
        )

    # Example implementation with the OpenAI Python SDK:
    # from openai import OpenAI
    # client = OpenAI(api_key=api_key)
    # with mp3_path.open("rb") as audio_file:
    #     return client.audio.transcriptions.create(
    #         model="gpt-4o-mini-transcribe", file=audio_file
    #     ).text
    return f"Mock LLM response: {mp3_path.name} is ready to be uploaded."


def reset_recording() -> None:
    """Clear UI state; existing local MP3 files remain available in temp_audio."""
    for key, value in {
        "recording_state": "idle",
        "last_audio_signature": None,
        "mp3_path": None,
        "llm_response": None,
        "error_message": None,
        "notice_message": "Current recording cleared. You can now make a new recording.",
    }.items():
        st.session_state[key] = value


def main() -> None:
    st.set_page_config(page_title="Voice Accessibility", page_icon="🎙️", layout="centered")
    initialize_session_state()

    st.title("🎙️ Voice Accessibility Assistant")
    if st.session_state.notice_message:
        st.info(st.session_state.notice_message, icon="ℹ️")
        st.session_state.notice_message = None
   
    st.markdown(
        """
        <div role="region" aria-label="Voice input instructions">
          <strong>Voice input:</strong> Select the microphone, speak clearly, then stop
          recording. Your audio is converted to MP3 locally before it is sent for processing.
        </div>
        
        """,
        unsafe_allow_html=True,
    )

    try:
        recording = st.audio_input(
            "Record your voice",
            help="Use the microphone control to start and stop your voice recording.",
        )
    except AttributeError:
        st.error("st.audio_input requires a newer Streamlit version. Run pip install -U streamlit.")
        return
    except Exception as exc:
        st.error(f"Microphone setup failed. Check browser permissions and retry. Details: {exc}")
        return

    if recording is not None:
        st.audio(recording, format="audio/wav")
        signature = f"{recording.name}:{len(recording.getvalue())}:{hash(recording.getvalue())}"

        if signature != st.session_state.last_audio_signature:
            st.session_state.recording_state = "processing"
            st.session_state.error_message = None
            try:
                with st.spinner("Converting recording to MP3 and processing it..."):
                    mp3_path = convert_audio_to_mp3(recording)
                    st.session_state.mp3_path = str(mp3_path)
                    st.session_state.llm_response = send_audio_to_llm(mp3_path)
                st.session_state.last_audio_signature = signature
                st.session_state.recording_state = "complete"
            except (ValueError, RuntimeError) as exc:
                st.session_state.recording_state = "error"
                st.session_state.error_message = str(exc)
            except Exception as exc:
                LOGGER.exception("Unexpected audio-processing error")
                st.session_state.recording_state = "error"
                st.session_state.error_message = (
                    "Audio processing failed unexpectedly. "
                    f"Details: {type(exc).__name__}: {exc}"
                )

    if st.session_state.recording_state == "complete":
        st.success("Recording converted and processed successfully.")
        mp3_path = Path(st.session_state.mp3_path)
        if mp3_path.exists():
            st.audio(mp3_path.read_bytes(), format="audio/mpeg")
            st.download_button("Download MP3", mp3_path.read_bytes(), mp3_path.name, "audio/mpeg")
        st.subheader("LLM response")
        st.write(st.session_state.llm_response)
    elif st.session_state.recording_state == "error":
        st.error(st.session_state.error_message)

    if st.button("Clear current recording"):
        reset_recording()
        st.rerun()


if __name__ == "__main__":
    main()
