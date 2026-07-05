"""Transcribes audio files (mp3/wav) locally via faster-whisper.
Timestamps are preserved for retrieval citations."""
from typing import TypedDict


class TranscriptSegment(TypedDict):
    start: float
    end: float
    text: str


class AudioTranscriber:
    def __init__(self, model_size: str = "base"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, file_path: str) -> list[TranscriptSegment]:
        segments, _ = self.model.transcribe(file_path, word_timestamps=False)
        return [
            {"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in segments
        ]
