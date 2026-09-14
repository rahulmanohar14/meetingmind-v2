"""Corpus loading for MeetingMind v2. No LLM calls."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TXT_SPEAKER = re.compile(r"^(?:\d+\.\s*)?([^:]+):\s*(.*)$")
_VTT_TIMESTAMP = re.compile(
    r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}"
)
_VTT_VOICE_TAG = re.compile(r"<v\s+([^>]+)>(.*)$", re.DOTALL)
_VTT_NAME_PREFIX = re.compile(r"^([^:]+):\s*(.*)$", re.DOTALL)
_VTT_TAG = re.compile(r"</?[^>]+>")


@dataclass
class Turn:
    meeting_id: str
    turn_index: int
    speaker: str
    text: str
    line_number: int


def turn_id(turn: Turn) -> str:
    return f"{turn.meeting_id}:{turn.turn_index}"


def load_transcript(path: str | Path) -> list[Turn]:
    path = Path(path)
    meeting_id = path.stem
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _load_txt(path, meeting_id)
    if suffix == ".vtt":
        return _load_vtt(path, meeting_id)
    raise ValueError(f"Unsupported transcript format: {path.suffix} ({path})")


def load_corpus(directory: str | Path) -> list[Turn]:
    """Load .txt/.vtt files directly under `directory` (not subfolders)."""
    directory = Path(directory)
    paths = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in {".txt", ".vtt"}
    )
    turns: list[Turn] = []
    for path in paths:
        turns.extend(load_transcript(path))
    return turns


def load_full_corpus(
    data_dir: str | Path | None = None,
    uploads_dir: str | Path | None = None,
) -> list[Turn]:
    """Load demo transcripts from data/ plus any files in data/uploads/."""
    root = Path(__file__).resolve().parents[1]
    data_dir = Path(data_dir) if data_dir is not None else root / "data"
    uploads_dir = (
        Path(uploads_dir) if uploads_dir is not None else root / "data" / "uploads"
    )
    turns = load_corpus(data_dir)
    if uploads_dir.is_dir():
        turns.extend(load_corpus(uploads_dir))
    return turns


def _load_txt(path: Path, meeting_id: str) -> list[Turn]:
    turns: list[Turn] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        match = _TXT_SPEAKER.match(stripped)
        if match:
            speaker = match.group(1).strip() or "UNKNOWN"
            text = match.group(2).strip()
            turns.append(
                Turn(
                    meeting_id=meeting_id,
                    turn_index=len(turns),
                    speaker=speaker,
                    text=text,
                    line_number=line_number,
                )
            )
        elif turns:
            prev = turns[-1]
            turns[-1] = Turn(
                meeting_id=meeting_id,
                turn_index=prev.turn_index,
                speaker=prev.speaker,
                text=f"{prev.text} {stripped}".strip(),
                line_number=prev.line_number,
            )
    return turns


def _load_vtt(path: Path, meeting_id: str) -> list[Turn]:
    lines = path.read_text(encoding="utf-8").splitlines()
    cues: list[tuple[str, str, int]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.upper().startswith("WEBVTT") or line.startswith("NOTE"):
            i += 1
            continue
        if _VTT_TIMESTAMP.match(line):
            cue_line_number = i + 1
            i += 1
            cue_lines: list[str] = []
            while i < len(lines) and lines[i].strip():
                cue_lines.append(lines[i])
                i += 1
            speaker, text = _parse_vtt_cue_text("\n".join(cue_lines))
            if text:
                cues.append((speaker, text, cue_line_number))
            continue
        i += 1

    turns: list[Turn] = []
    for speaker, text, line_number in cues:
        if turns and turns[-1].speaker == speaker:
            prev = turns[-1]
            turns[-1] = Turn(
                meeting_id=meeting_id,
                turn_index=prev.turn_index,
                speaker=prev.speaker,
                text=f"{prev.text} {text}".strip(),
                line_number=prev.line_number,
            )
        else:
            turns.append(
                Turn(
                    meeting_id=meeting_id,
                    turn_index=len(turns),
                    speaker=speaker,
                    text=text,
                    line_number=line_number,
                )
            )
    return turns


def _parse_vtt_cue_text(raw: str) -> tuple[str, str]:
    cleaned = raw.strip()
    voice = _VTT_VOICE_TAG.search(cleaned)
    if voice:
        speaker = voice.group(1).strip()
        text = _VTT_TAG.sub("", voice.group(2)).strip()
        return speaker or "UNKNOWN", text

    plain = _VTT_TAG.sub("", cleaned).strip()
    name = _VTT_NAME_PREFIX.match(plain)
    if name:
        return name.group(1).strip() or "UNKNOWN", name.group(2).strip()
    return "UNKNOWN", plain
