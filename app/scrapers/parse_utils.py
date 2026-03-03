from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

try:
    from langdetect import DetectorFactory, detect
except ImportError:  # pragma: no cover - optional dependency for language hints
    DetectorFactory = None

    def detect(_: str) -> str | None:  # type: ignore[override]
        return None

if DetectorFactory is not None:
    DetectorFactory.seed = 0


CHFS_RE = re.compile(r"CHF\s*([\d'’]+)")
ROOMS_RE = re.compile(r"(?P<rooms>\d+(?:[.,]\d+)?)\s*(?:room|rooms|Zimmer|Z)\b", re.IGNORECASE)
ROOMS_HALF_RE = re.compile(r"(?P<rooms>\d+)\s*[½]", re.IGNORECASE)
AREA_RE = re.compile(r"(?P<area>\d+)\s*m\s*[²2]", re.IGNORECASE)
POSTAL_RE = re.compile(r"\b\d{4}\b")


def normalize_lines(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line]


def extract_section(lines: List[str], heading: str, stop_headings: Iterable[str]) -> List[str]:
    start = None
    heading_lower = heading.lower()
    stop_set = {h.lower() for h in stop_headings}
    for idx, line in enumerate(lines):
        if line.lower() == heading_lower:
            start = idx + 1
            break
    if start is None:
        return []
    end = len(lines)
    for idx in range(start, len(lines)):
        if lines[idx].lower() in stop_set:
            end = idx
            break
    return [line for line in lines[start:end] if line]


def parse_chf_amount(text: str) -> Optional[int]:
    match = CHFS_RE.search(text)
    if not match:
        return None
    raw = match.group(1).replace("'", "").replace("’", "")
    try:
        return int(raw)
    except ValueError:
        return None


def parse_rooms(text: str) -> Optional[float]:
    half_match = ROOMS_HALF_RE.search(text)
    if half_match:
        return float(half_match.group("rooms")) + 0.5
    match = ROOMS_RE.search(text)
    if match:
        return float(match.group("rooms").replace(",", "."))
    return None


def parse_area(text: str) -> Optional[int]:
    match = AREA_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group("area"))
    except ValueError:
        return None


def split_description(lines: List[str]) -> List[Tuple[str, Optional[str]]]:
    paragraphs: List[str] = []
    current: List[str] = []
    for line in lines:
        if not line.strip():
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
        else:
            current.append(line.strip())
    if current:
        paragraphs.append(" ".join(current).strip())

    if len(paragraphs) == 1 and len(lines) > 1:
        paragraphs = [line.strip() for line in lines if line.strip()]

    blocks: List[Tuple[str, Optional[str]]] = []
    for para in paragraphs:
        if not para:
            continue
        try:
            lang = detect(para)
        except Exception:
            lang = None
        blocks.append((para, lang))
    return blocks


def find_address_line(lines: List[str]) -> Optional[str]:
    for line in lines:
        if POSTAL_RE.search(line) and any(char.isalpha() for char in line):
            return line
    for line in lines:
        if "zürich" in line.lower() or "zurich" in line.lower():
            return line
    return None
