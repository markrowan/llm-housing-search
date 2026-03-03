from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FilterSpec(BaseModel):
    raw_prompt: str
    min_rooms: Optional[float] = None
    max_rooms: Optional[float] = None
    min_price_chf: Optional[int] = None
    max_price_chf: Optional[int] = None
    location_query: Optional[str] = None
    location_tags: List[str] = Field(default_factory=list)
    max_commute_minutes: Optional[int] = None
    commute_target: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)


class DescriptionBlock(BaseModel):
    text: str
    language: Optional[str] = None


class Listing(BaseModel):
    id: Optional[str] = None
    url: str
    title: Optional[str] = None
    price_chf: Optional[int] = None
    rooms: Optional[float] = None
    area_sqm: Optional[int] = None
    address: Optional[str] = None
    relevance_score: Optional[float] = None
    location_score: Optional[float] = None
    intent_summary: Optional[str] = None
    match_label: Optional[str] = None
    duplicate_sites: List[str] = Field(default_factory=list)
    duplicate_urls: List[str] = Field(default_factory=list)
    stats: Dict[str, str] = Field(default_factory=dict)
    descriptions: List[DescriptionBlock] = Field(default_factory=list)


class SiteResult(BaseModel):
    site: str
    search_url: str
    listings: List[Listing] = Field(default_factory=list)


class SearchResponse(BaseModel):
    filters: FilterSpec
    results: List[SiteResult] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
