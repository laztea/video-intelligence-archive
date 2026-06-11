# archive/transcribe.py
from pathlib import Path
from archive import config

def transcribe(audio_path: Path) -> list[dict]:
    client = config.get_openai()
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1", file=f, response_format="verbose_json",
            timestamp_granularities=["segment"])
    return [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
            for s in result.segments]
