# Voice Accessibility Assistant

Accessible Streamlit microphone capture, local WAV-to-MP3 conversion, and a mock audio-capable LLM integration.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Install [FFmpeg](https://ffmpeg.org/) and make it available on your system PATH.
4. Optionally copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add an API key.
5. Run `streamlit run app.py`.

Converted recordings are stored locally in `temp_audio/` and are ignored by Git.

If a virtual environment cannot discover FFmpeg, add the full executable paths
as `FFMPEG_BINARY` and `FFPROBE_BINARY` in `.streamlit/secrets.toml`.
