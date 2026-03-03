from __future__ import annotations

import re
from typing import Optional

from app.llm import llm_enabled, parse_prompt_with_llm
from app.models import FilterSpec


ROOMS_RANGE_RE = re.compile(
    r"(?P<min>\d+(?:[.,]\d+)?)\s*(?:to|\-|–)\s*(?P<max>\d+(?:[.,]\d+)?)\s*rooms",
    re.IGNORECASE,
)
ROOMS_SINGLE_RE = re.compile(r"(?P<val>\d+(?:[.,]\d+)?)\s*rooms", re.IGNORECASE)
COMMUTE_RE = re.compile(r"within\s+(?P<min>\d+)\s*minutes?\s+to\s+the\s+(?P<target>\w+)", re.IGNORECASE)
NEAR_RE = re.compile(r"near\s+(?P<place>[^,.;]+)", re.IGNORECASE)
IN_RE = re.compile(r"in\s+(?P<place>[^,.;]+)", re.IGNORECASE)


def _to_float(value: str) -> float:
    return float(value.replace(",", "."))


def _extract_location(prompt: str) -> Optional[str]:
    match = NEAR_RE.search(prompt)
    if match:
        return match.group("place").strip()
    match = IN_RE.search(prompt)
    if match:
        return match.group("place").strip()
    return None


def heuristic_parse_prompt(prompt: str) -> FilterSpec:
    min_rooms = None
    max_rooms = None

    range_match = ROOMS_RANGE_RE.search(prompt)
    if range_match:
        min_rooms = _to_float(range_match.group("min"))
        max_rooms = _to_float(range_match.group("max"))
    else:
        single_match = ROOMS_SINGLE_RE.search(prompt)
        if single_match:
            min_rooms = _to_float(single_match.group("val"))

    max_commute_minutes = None
    commute_target = None
    commute_match = COMMUTE_RE.search(prompt)
    if commute_match:
        max_commute_minutes = int(commute_match.group("min"))
        commute_target = commute_match.group("target")

    location_query = _extract_location(prompt)

    keywords = []
    lower_prompt = prompt.lower()
    if "green" in lower_prompt:
        keywords.append("green")
    if "leafy" in lower_prompt:
        keywords.append("leafy")
    if "suburb" in lower_prompt:
        keywords.append("suburb")
    if "kreis" in lower_prompt:
        keywords.append("kreis")
    if "lake" in lower_prompt:
        keywords.append("lake")

    location_tags = []
    if "kreis 7" in lower_prompt or "kreis7" in lower_prompt:
        location_tags.append("Kreis 7")
    if "zurich" in lower_prompt or "zürich" in lower_prompt or "zurich" in lower_prompt:
        location_tags.append("Zürich")

    return FilterSpec(
        raw_prompt=prompt,
        min_rooms=min_rooms,
        max_rooms=max_rooms,
        location_query=location_query,
        location_tags=location_tags,
        max_commute_minutes=max_commute_minutes,
        commute_target=commute_target,
        keywords=keywords,
    )


def parse_prompt(prompt: str, use_llm: bool, api_key: Optional[str] = None) -> FilterSpec:
    if use_llm and llm_enabled(api_key):
        data = parse_prompt_with_llm(prompt, api_key=api_key)
        return FilterSpec(raw_prompt=prompt, **data)
    return heuristic_parse_prompt(prompt)
