"""Accessible voice recording and audio-to-LLM Streamlit application."""

import os
import shutil
import logging
import html
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


def get_provider_api_key(provider: str) -> str | None:
    """Retrieve API key from secrets or env for a given provider, handling key name variations."""
    if provider == "Gemini":
        return (
            get_optional_secret("GEMINI_API_KEY")
            or get_optional_secret("GOOGLE_API_KEY")
        )
    elif provider == "OpenAI":
        return get_optional_secret("OPENAI_API_KEY")
    return None


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
        "transcript": None,
        "llm_response": None,
        "tts_path": None,
        "error_message": None,
        "provider": "Gemini",
        "api_key": None,
        "analysis_prompt": "Provide a detailed summary, sentiment analysis, and list of key action items from the audio transcription.",
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


def transcribe_and_analyze_with_gemini(mp3_path: Path, api_key: str, analysis_prompt: str) -> tuple[str, str]:
    """Transcribe and analyze audio using Google GenAI SDK (Gemini)."""
    from google import genai
    from google.genai import types
    import time
    
    client = genai.Client(api_key=api_key)
    file_size_mb = mp3_path.stat().st_size / (1024 * 1024)
    
    # Transcription prompt instructions
    transcription_prompt = (
        "You are a precise audio transcription tool. Please transcribe the provided audio "
        "word-for-word exactly as spoken. Do not add any commentary, summaries, or introductions. "
        "Only output the transcription text."
    )
    
    if file_size_mb < 20:
        audio_bytes = mp3_path.read_bytes()
        audio_part = types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/mp3"
        )
        
        tx_response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[audio_part, transcription_prompt]
        )
        transcript = tx_response.text or ""
        
        analysis_response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[audio_part, analysis_prompt]
        )
        analysis = analysis_response.text or ""
        
        return transcript.strip(), analysis.strip()
    else:
        uploaded_file = client.files.upload(file=mp3_path)
        try:
            start_time = time.time()
            while True:
                file_info = client.files.get(name=uploaded_file.name)
                if file_info.state.name == "ACTIVE":
                    break
                elif file_info.state.name == "FAILED":
                    raise RuntimeError("Audio file processing failed on Gemini servers.")
                if time.time() - start_time > 120:
                    raise RuntimeError("Gemini file processing timed out.")
                time.sleep(2)
                
            tx_response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[uploaded_file, transcription_prompt]
            )
            transcript = tx_response.text or ""
            
            analysis_response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[uploaded_file, analysis_prompt]
            )
            analysis = analysis_response.text or ""
            
            return transcript.strip(), analysis.strip()
        finally:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception as exc:
                LOGGER.warning(f"Failed to delete remote file {uploaded_file.name}: {exc}")


def transcribe_and_analyze_with_openai(mp3_path: Path, api_key: str, analysis_prompt: str) -> tuple[str, str]:
    """Transcribe and analyze audio using OpenAI's Whisper and GPT-4o-mini."""
    
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    
    # 1. Transcribe with Whisper
    with mp3_path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        ).text or ""
    
    # 2. Analyze transcript with GPT-4o-mini
    chat_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant analyzing speech transcriptions."},
            {"role": "user", "content": f"{analysis_prompt}\n\nTranscript:\n{transcript}"}
        ]
    )
    analysis = chat_response.choices[0].message.content or ""
    
    return transcript.strip(), analysis.strip()


def transcribe_and_analyze(mp3_path: Path, provider: str, api_key: str, analysis_prompt: str) -> tuple[str, str]:
    """Route transcription and analysis based on the selected provider."""
    if not api_key:
        raise ValueError(f"Please provide a valid API key for {provider}.")
        
    if provider == "Gemini":
        return transcribe_and_analyze_with_gemini(mp3_path, api_key, analysis_prompt)
    elif provider == "OpenAI":
        return transcribe_and_analyze_with_openai(mp3_path, api_key, analysis_prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def clean_markdown_for_tts(text: str) -> str:
    """Remove markdown syntax (stars, hashes, lists, backticks) to make the text clean for TTS readback."""
    import re
    # 1. Remove bold/italics markers (asterisks, underscores)
    text = re.sub(r'[*_]{1,3}', '', text)
    
    # 2. Remove headers (e.g. ### Header -> Header)
    text = re.sub(r'#+\s+', '', text)
    
    # 3. Remove list item indicators (bullet points and numbering) at the beginning of lines
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # 4. Remove code blocks and inline code markers (backticks)
    text = re.sub(r'`{1,3}', '', text)
    
    # 5. Remove HTML tags if any
    text = re.sub(r'<[^>]*>', '', text)
    
    # 6. Normalize multiple whitespaces and newlines
    text = re.sub(r'\n+', '\n', text)
    
    return text.strip()


def generate_tts(text: str, provider: str, api_key: str | None) -> Path:
    """Generate TTS audio file from text response, omitting markdown tags for clean readback."""
    clean_text = clean_markdown_for_tts(text)
    
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tts_path = TEMP_AUDIO_DIR / f"tts_{timestamp}.mp3"
    
    if provider == "OpenAI" and api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=clean_text
            )
            response.write_to_file(tts_path)
            return tts_path
        except Exception as exc:
            LOGGER.warning(f"OpenAI TTS failed, falling back to gTTS: {exc}")
            
    # Fallback: gTTS
    from gtts import gTTS
    tts = gTTS(text=clean_text, lang="en")
    tts.save(str(tts_path))
    return tts_path


def reset_recording() -> None:
    """Clear UI state; existing local MP3 files remain available in temp_audio."""
    for key, value in {
        "recording_state": "idle",
        "last_audio_signature": None,
        "mp3_path": None,
        "transcript": None,
        "llm_response": None,
        "tts_path": None,
        "error_message": None,
    }.items():
        st.session_state[key] = value


def main() -> None:
    st.set_page_config(page_title="Voice Accessibility", page_icon="🎙️", layout="centered")
    initialize_session_state()

    # Custom styling
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            color: #f1f5f9;
        }
        h1 {
            background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800 !important;
            letter-spacing: -0.025em;
            margin-bottom: 0.5rem;
        }
        .helper-box {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 24px;
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }
        .result-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            margin-top: 16px;
            backdrop-filter: blur(16px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .result-card:hover {
            transform: translateY(-2px);
            border-color: rgba(168, 85, 247, 0.4);
        }
        .card-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .card-content {
            color: #cbd5e1;
            line-height: 1.6;
            font-size: 0.95rem;
            white-space: pre-wrap;
        }
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            background-color: rgba(15, 23, 42, 0.4) !important;
            color: #f1f5f9 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        div.stButton > button {
            background: linear-gradient(90deg, #a855f7 0%, #6366f1 100%) !important;
            color: white !important;
            border: none !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(168, 85, 247, 0.2) !important;
        }
        div.stButton > button:hover {
            transform: scale(1.03) !important;
            box-shadow: 0 6px 20px rgba(168, 85, 247, 0.4) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Settings")
        
        provider = st.selectbox(
            "AI Provider",
            options=["Gemini", "OpenAI"],
            index=0,
            help="Choose the model provider for Speech-to-Text and Analysis."
        )
        st.session_state.provider = provider
        
        # API Key input
        secret_key_name = "GEMINI_API_KEY/GOOGLE_API_KEY" if provider == "Gemini" else "OPENAI_API_KEY"
        configured_secret = get_provider_api_key(provider)
        
        if configured_secret:
            display_name = "GOOGLE_API_KEY" if (provider == "Gemini" and get_optional_secret("GOOGLE_API_KEY") and not get_optional_secret("GEMINI_API_KEY")) else secret_key_name
            st.success(f"✓ {display_name} loaded from secrets.")
            api_key = st.text_input(
                "API Key (Overrides Secret)",
                type="password",
                placeholder="Enter new key to override...",
                help="Optional if key is already set in your secrets.toml."
            )
        else:
            st.warning(f"⚠️ {secret_key_name} is missing in secrets.")
            api_key = st.text_input(
                "API Key",
                type="password",
                placeholder=f"Enter {provider} API Key...",
                help="Required to run Speech-to-Text and Analysis."
            )
            
        if api_key:
            st.session_state.api_key = api_key
        else:
            st.session_state.api_key = configured_secret

        # Analysis prompt input
        analysis_prompt = st.text_area(
            "LLM Analysis Prompt",
            value="Provide a detailed summary, sentiment analysis, and list of key action items from the audio transcription.",
            help="Customize how you want the LLM to analyze the transcribed audio."
        )
        st.session_state.analysis_prompt = analysis_prompt

    st.title("🎙️ Voice Accessibility Assistant")
    st.markdown(
        """
        <div class="helper-box" role="region" aria-label="Voice input instructions">
          <strong>Voice input:</strong> Select the microphone, speak clearly, then stop
          recording. Your audio is converted to MP3 locally before it is processed by the selected LLM.
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
                with st.spinner("Converting recording to MP3 and analyzing speech..."):
                    mp3_path = convert_audio_to_mp3(recording)
                    st.session_state.mp3_path = str(mp3_path)
                    
                    provider = st.session_state.provider
                    api_key = st.session_state.api_key
                    analysis_prompt = st.session_state.analysis_prompt
                    
                    transcript, analysis = transcribe_and_analyze(
                        mp3_path, provider, api_key, analysis_prompt
                    )
                    st.session_state.transcript = transcript
                    st.session_state.llm_response = analysis
                    
                    # Generate read-back audio
                    tts_path = generate_tts(analysis, provider, api_key)
                    st.session_state.tts_path = str(tts_path)
                    
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
        st.success("Recording processed successfully!")
        mp3_path = Path(st.session_state.mp3_path)
        if mp3_path.exists():
            st.audio(mp3_path.read_bytes(), format="audio/mpeg")
            st.download_button("Download MP3", mp3_path.read_bytes(), mp3_path.name, "audio/mpeg")
            
        # Audio read-back player
        if st.session_state.get("tts_path"):
            tts_path = Path(st.session_state.tts_path)
            if tts_path.exists():
                st.markdown("🔊 **Reading back response:**")
                st.audio(tts_path.read_bytes(), format="audio/mpeg", autoplay=True)
        
        # Display transcript and analysis in beautiful columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="card-title">📝 Transcribed Text</div>
                    <div class="card-content">{html.escape(st.session_state.transcript)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="card-title">🤖 LLM Analysis</div>
                    <div class="card-content">{html.escape(st.session_state.llm_response)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    elif st.session_state.recording_state == "error":
        st.error(st.session_state.error_message)

    if st.button("Clear current recording"):
        reset_recording()
        st.rerun()


if __name__ == "__main__":
    main()
