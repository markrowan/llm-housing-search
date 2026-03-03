from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def llm_enabled(api_key: Optional[str] = None) -> bool:
    if api_key:
        return True
    return bool(os.getenv("OPENAI_API_KEY"))


def parse_prompt_with_llm(prompt: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    system = (
        "You are a strict JSON generator. Extract housing search filters from the user prompt. "
        "Return only JSON with keys: min_rooms, max_rooms, min_price_chf, max_price_chf, "
        "location_query, location_tags, max_commute_minutes, commute_target, keywords. "
        "Use null for unknown values and arrays for tags/keywords."
    )
    response = client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    text = response.output_text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = _extract_json(text)
        if cleaned is None:
            raise ValueError(f"LLM did not return valid JSON: {text}")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM did not return valid JSON: {text}") from exc


def _extract_json(text: str) -> Optional[str]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return stripped[start : end + 1]


INTENT_KEYS = (
    "location",
    "proximity",
    "views",
    "quietness",
    "transit",
    "outdoor",
    "light",
    "amenities",
    "other",
)


def extract_search_intent_struct(prompt: str, api_key: Optional[str] = None) -> Dict[str, str]:
    system = (
        "You are a strict JSON generator. Summarize the user's housing intent into a structured schema "
        "for semantic matching. Emphasize location specificity (city, quarter, zip codes, proximity to "
        "landmarks such as the lake, commute constraints). Return only JSON with keys: "
        "location, proximity, views, quietness, transit, outdoor, light, amenities, other. "
        "Values should be short comma-separated phrases. Keep the total under 45 words."
    )
    return _extract_intent_struct_from_text(prompt, system, api_key=api_key)


def extract_listing_intent_struct(text: str, api_key: Optional[str] = None) -> Dict[str, str]:
    system = (
        "You are a strict JSON generator. Summarize a housing listing into a structured schema for "
        "semantic matching with a user intent. Emphasize location specificity (city, quarter, zip "
        "codes, proximity to lake/landmarks), view direction, quietness, transit access, outdoor "
        "space, daylight, and distinctive amenities. Ignore generic rental marketing language. "
        "Return only JSON with keys: location, proximity, views, quietness, transit, outdoor, "
        "light, amenities, other. Values should be short comma-separated phrases. Keep the total under 45 words."
    )
    return _extract_intent_struct_from_text(text, system, api_key=api_key)


def _normalize_intent_struct(data: Dict[str, Any]) -> Dict[str, str]:
    normalized = {key: "" for key in INTENT_KEYS}
    if not isinstance(data, dict):
        return normalized
    for key in INTENT_KEYS:
        value = data.get(key, "")
        if isinstance(value, str):
            normalized[key] = value.strip()
    return normalized


def _extract_intent_struct_from_text(
    text: str, system: str, api_key: Optional[str]
) -> Dict[str, str]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    response = client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    raw = response.output_text
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = _extract_json(raw)
        if cleaned is None:
            raise ValueError(f"LLM did not return valid JSON: {raw}")
        data = json.loads(cleaned)
    normalized = _normalize_intent_struct(data if isinstance(data, dict) else {})
    if not any(normalized.values()):
        raise ValueError(f"LLM did not return intent: {raw}")
    return normalized
