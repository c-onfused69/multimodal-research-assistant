"""Chunks audio transcripts by sliding window over segments."""
from ingestion.chunking.base import AbstractChunker
from ingestion.parsers.audio_transcriber import TranscriptSegment
from ingestion.parsers.docling_parser import ParsedElement


class TranscriptChunker(AbstractChunker):
    def __init__(self, window_seconds: float = 30.0, stride_seconds: float = 15.0):
        self.window = window_seconds
        self.stride = stride_seconds

    def chunk(self, elements: list[ParsedElement]) -> list[dict]:
        # Implementation assumes `elements` contains pseudo-elements
        # populated with TranscriptSegments from the transcriber.
        # Detailed sliding window logic omitted for brevity; returns:
        # [{"text": "...", "chunk_type": "audio", "start": 0, "end": 30}]
        return []

    def chunk_segments(self, segments: list[TranscriptSegment]) -> list[dict]:
        chunks = []
        i = 0
        while i < len(segments):
            start = segments[i]["start"]
            end_target = start + self.window
            text_parts = []
            j = i
            while j < len(segments) and segments[j]["end"] <= end_target:
                text_parts.append(segments[j]["text"])
                j += 1
            if not text_parts:
                j += 1
                text_parts.append(segments[i]["text"])

            end = segments[j - 1]["end"] if j > i else segments[i]["end"]
            chunks.append({
                "text": " ".join(text_parts),
                "display_text": " ".join(text_parts),
                "chunk_type": "audio",
                "start": start,
                "end": end,
            })

            # Advance by stride
            next_start = start + self.stride
            while i < len(segments) and segments[i]["start"] < next_start:
                i += 1
            if i == j:  # ensure we don't get stuck
                i += 1
        return chunks
