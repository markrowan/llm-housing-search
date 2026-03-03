from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlencode

from app.models import FilterSpec


@dataclass(frozen=True)
class BoundingBox:
    north: float
    south: float
    east: float
    west: float


LOCATION_BOXES: Dict[str, BoundingBox] = {
    "zürich": BoundingBox(north=47.43, south=47.32, east=8.64, west=8.45),
    "zurich": BoundingBox(north=47.43, south=47.32, east=8.64, west=8.45),
    "kreis 7": BoundingBox(north=47.42, south=47.37, east=8.58, west=8.53),
}

HOMEGATE_CITY_PATHS: Dict[str, str] = {
    "zürich": "city-zurich",
    "zurich": "city-zurich",
}


def _normalize(text: str) -> str:
    return text.strip().lower()


def resolve_bounding_box(filters: FilterSpec) -> Optional[BoundingBox]:
    if filters.location_tags:
        for tag in filters.location_tags:
            key = _normalize(tag)
            if key in LOCATION_BOXES:
                return LOCATION_BOXES[key]
    if filters.location_query:
        key = _normalize(filters.location_query)
        if key in LOCATION_BOXES:
            return LOCATION_BOXES[key]
    return None


def build_flatfox_search_url(filters: FilterSpec, take: int = 50) -> str:
    base = "https://flatfox.ch/en/search"
    params = {}
    bbox = resolve_bounding_box(filters)
    if bbox:
        params.update(
            {
                "north": bbox.north,
                "south": bbox.south,
                "east": bbox.east,
                "west": bbox.west,
            }
        )
    if filters.min_rooms is not None:
        params["min_rooms"] = filters.min_rooms
    if filters.max_rooms is not None:
        params["max_rooms"] = filters.max_rooms
    if filters.min_price_chf is not None:
        params["min_price"] = filters.min_price_chf
    if filters.max_price_chf is not None:
        params["max_price"] = filters.max_price_chf
    params["take"] = take
    return f"{base}?{urlencode(params)}"


def build_homegate_search_url(filters: FilterSpec) -> str:
    base = "https://www.homegate.ch/rent/apartment"
    if filters.location_tags:
        for tag in filters.location_tags:
            key = _normalize(tag)
            if key in HOMEGATE_CITY_PATHS:
                return f"{base}/{HOMEGATE_CITY_PATHS[key]}/matching-list"
    if filters.location_query:
        key = _normalize(filters.location_query)
        if key in HOMEGATE_CITY_PATHS:
            return f"{base}/{HOMEGATE_CITY_PATHS[key]}/matching-list"
    return f"{base}/matching-list"
