from __future__ import annotations

import math
import os
import re
from typing import Callable, Dict, Iterable, List, Sequence

from app.models import FilterSpec, Listing, SearchResponse, SiteResult
from app.llm import extract_listing_intent_struct, extract_search_intent_struct, llm_enabled
from app.prompt_parser import heuristic_parse_prompt, parse_prompt
from app.scrapers.flatfox import search_flatfox
from app.scrapers.homegate import search_homegate


def _matches_filters(listing: Listing, filters: FilterSpec) -> bool:
    if filters.min_rooms is not None and listing.rooms is not None:
        if listing.rooms < filters.min_rooms:
            return False
    if filters.max_rooms is not None and listing.rooms is not None:
        if listing.rooms > filters.max_rooms:
            return False
    if filters.min_price_chf is not None and listing.price_chf is not None:
        if listing.price_chf < filters.min_price_chf:
            return False
    if filters.max_price_chf is not None and listing.price_chf is not None:
        if listing.price_chf > filters.max_price_chf:
            return False
    return True


def _filter_listings(listings: Iterable[Listing], filters: FilterSpec) -> List[Listing]:
    return [listing for listing in listings if _matches_filters(listing, filters)]


def _format_llm_error(exc: Exception) -> str:
    try:
        import openai

        if isinstance(exc, openai.AuthenticationError):
            return "OpenAI API authentication failed; check your API key. Using heuristic parsing instead."
        if isinstance(exc, openai.RateLimitError):
            code = None
            body = getattr(exc, "body", None)
            if isinstance(body, dict):
                code = body.get("error", {}).get("code")
            if code == "insufficient_quota":
                return (
                    "OpenAI API quota exceeded or billing not enabled; using heuristic parsing instead."
                )
            return "OpenAI API rate limit reached; using heuristic parsing instead."
    except Exception:
        pass

    if isinstance(exc, ValueError) and "LLM did not return valid JSON" in str(exc):
        return "LLM returned invalid JSON; using heuristic parsing instead."
    return "LLM parsing failed; using heuristic parsing instead."


def _format_embedding_error(exc: Exception) -> str:
    try:
        import openai

        if isinstance(exc, openai.AuthenticationError):
            return "OpenAI API authentication failed; using heuristic relevance ranking instead."
        if isinstance(exc, openai.RateLimitError):
            code = None
            body = getattr(exc, "body", None)
            if isinstance(body, dict):
                code = body.get("error", {}).get("code")
            if code == "insufficient_quota":
                return (
                    "OpenAI API quota exceeded or billing not enabled; "
                    "using heuristic relevance ranking instead."
                )
            return "OpenAI API rate limit reached; using heuristic relevance ranking instead."
    except Exception:
        pass
    return "Embedding-based relevance ranking failed; using heuristic ranking instead."


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "near",
    "within",
    "minutes",
    "min",
    "room",
    "rooms",
    "sqm",
    "m2",
    "m²",
    "und",
    "oder",
    "mit",
    "nahe",
    "bei",
    "im",
    "in",
    "der",
    "die",
    "das",
}



def _tokenize(text: str) -> List[str]:
    tokens = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def _listing_text(listing: Listing) -> str:
    parts: List[str] = []
    if listing.title:
        parts.append(listing.title)
    if listing.address:
        parts.append(listing.address)
    if listing.descriptions:
        parts.extend(block.text for block in listing.descriptions)
    if listing.stats:
        parts.extend(listing.stats.values())
    return " ".join(parts)




def _score_relevance(prompt_tokens: Sequence[str], listing: Listing) -> float:
    if not prompt_tokens:
        return 0.0
    listing_tokens = set(_tokenize(_listing_text(listing)))
    if not listing_tokens:
        return 0.0
    intersection = listing_tokens.intersection(prompt_tokens)
    return len(intersection) / max(len(set(prompt_tokens)), 1)


def _sort_by_relevance(listings: List[Listing], prompt: str) -> List[Listing]:
    prompt_tokens = _tokenize(prompt)
    scored = []
    for index, listing in enumerate(listings):
        score = _score_relevance(prompt_tokens, listing)
        listing.relevance_score = score
        scored.append((score, index, listing))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [listing for _, _, listing in scored]


def _truncate_text(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    denom = math.sqrt(left_norm) * math.sqrt(right_norm)
    if denom == 0.0:
        return 0.0
    return dot / denom


def _intent_location_text(intent: Dict[str, str]) -> str:
    parts = []
    for key in ("location", "proximity", "transit"):
        value = intent.get(key)
        if value:
            parts.append(value)
    return ", ".join(parts).strip()


def _intent_full_text(intent: Dict[str, str]) -> str:
    ordered_keys = (
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
    parts = []
    for key in ordered_keys:
        value = intent.get(key)
        if value:
            parts.append(f"{key}: {value}")
    return "; ".join(parts)


def _get_embeddings(texts: List[str], api_key: str | None) -> List[List[float]]:
    from openai import OpenAI

    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    response = client.embeddings.create(
        model=model,
        input=texts,
        encoding_format="float",
    )
    data = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in data]


def _sort_by_embedding_relevance(
    listings: List[Listing],
    prompt: str,
    api_key: str | None,
) -> List[Listing]:
    prompt_text = prompt.strip()
    if not prompt_text:
        return listings
    indexed_texts = []
    for index, listing in enumerate(listings):
        text = _listing_text(listing).strip()
        if not text:
            continue
        indexed_texts.append((index, _truncate_text(text)))
    if not indexed_texts:
        return listings

    inputs = [prompt_text] + [text for _, text in indexed_texts]
    embeddings = _get_embeddings(inputs, api_key=api_key)
    if len(embeddings) != len(inputs):
        return listings

    prompt_embedding = embeddings[0]
    scores = [-1.0 for _ in listings]
    for (index, _), embedding in zip(indexed_texts, embeddings[1:]):
        scores[index] = _cosine_similarity(prompt_embedding, embedding)

    ordered = sorted(range(len(listings)), key=lambda idx: (-scores[idx], idx))
    for idx, score in enumerate(scores):
        listings[idx].relevance_score = None if score < 0 else score
    return [listings[idx] for idx in ordered]


DUPLICATE_SIMILARITY = 0.88
PRICE_TOLERANCE_CHF = 200
ROOM_TOLERANCE = 0.5
AREA_TOLERANCE_SQM = 15
LOCATION_WEIGHT = 0.7
FULL_WEIGHT = 0.3

STRONG_MATCH_FLOOR = 0.74
STRONG_MATCH_GAP = 0.06
STRONG_MATCH_FALLBACK_COUNT = 3
STRONG_LOCATION_FLOOR = 0.65


def _extract_zip(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"\b\d{4}\b", text)
    return match.group(0) if match else None


def _metadata_compatible(left: Listing, right: Listing) -> bool:
    left_zip = _extract_zip(left.address)
    right_zip = _extract_zip(right.address)
    if left_zip and right_zip and left_zip != right_zip:
        return False
    if left.price_chf is not None and right.price_chf is not None:
        if abs(left.price_chf - right.price_chf) > PRICE_TOLERANCE_CHF:
            return False
    if left.rooms is not None and right.rooms is not None:
        if abs(left.rooms - right.rooms) > ROOM_TOLERANCE:
            return False
    if left.area_sqm is not None and right.area_sqm is not None:
        if abs(left.area_sqm - right.area_sqm) > AREA_TOLERANCE_SQM:
            return False
    return True


def _rank_and_dedupe_results(
    results: List[SiteResult],
    prompt_intent: Dict[str, str],
    api_key: str | None,
    use_embeddings: bool,
    notes: List[str],
) -> None:
    records: List[dict] = []
    for site_result in results:
        for listing in site_result.listings:
            records.append({"site": site_result.site, "listing": listing})

    if not records:
        return

    embeddings: List[List[float]] | None = None
    if use_embeddings:
        try:
            prompt_location = _intent_location_text(prompt_intent) or " "
            prompt_full = _intent_full_text(prompt_intent) or " "
            inputs = [prompt_location, prompt_full]
            intent_failed = False
            for record in records:
                raw_text = _listing_text(record["listing"]).strip() or " "
                listing_text = raw_text
                if llm_enabled(api_key):
                    try:
                        listing_intent = extract_listing_intent_struct(
                            listing_text, api_key=api_key
                        )
                    except Exception as exc:
                        intent_failed = True
                        listing_text = raw_text
                        listing_intent = {"other": listing_text}
                else:
                    listing_intent = {"other": listing_text}
                record["listing"].intent_summary = _intent_full_text(listing_intent)
                record["intent_text"] = listing_intent
                inputs.append(_truncate_text(_intent_location_text(listing_intent) or " "))
                inputs.append(_truncate_text(_intent_full_text(listing_intent) or " "))
            if intent_failed:
                notes.append(
                    "Listing intent extraction failed for some results; using raw listing text instead."
                )
            embeddings = _get_embeddings(inputs, api_key=api_key)
            if len(embeddings) == len(inputs):
                prompt_location_embedding = embeddings[0]
                prompt_full_embedding = embeddings[1]
                offset = 2
                for record in records:
                    location_embedding = embeddings[offset]
                    full_embedding = embeddings[offset + 1]
                    offset += 2
                    record["location_embedding"] = location_embedding
                    record["embedding"] = full_embedding
                    location_score = _cosine_similarity(
                        prompt_location_embedding, location_embedding
                    )
                    full_score = _cosine_similarity(prompt_full_embedding, full_embedding)
                    record["listing"].location_score = location_score
                    record["listing"].relevance_score = (
                        LOCATION_WEIGHT * location_score + FULL_WEIGHT * full_score
                    )
            else:
                embeddings = None
        except Exception as exc:
            notes.append(_format_embedding_error(exc))
            embeddings = None

    if embeddings is None:
        prompt_tokens = _tokenize(_intent_full_text(prompt_intent))
        for record in records:
            listing = record["listing"]
            listing.relevance_score = _score_relevance(prompt_tokens, listing)

    for site_result in results:
        site_result.listings = sorted(
            site_result.listings,
            key=lambda listing: (-(listing.relevance_score or -1), listing.url),
        )

    if embeddings is None:
        notes.append("Duplicate detection requires embeddings; duplicates were not collapsed.")
        return

    records_sorted = sorted(
        records,
        key=lambda record: (-(record["listing"].relevance_score or -1), record["listing"].url),
    )
    dropped: set[int] = set()
    for index, record in enumerate(records_sorted):
        listing = record["listing"]
        if id(listing) in dropped:
            continue
        for other in records_sorted[index + 1 :]:
            other_listing = other["listing"]
            if id(other_listing) in dropped:
                continue
            if not _metadata_compatible(listing, other_listing):
                continue
            similarity = _cosine_similarity(record["embedding"], other["embedding"])
            if similarity < DUPLICATE_SIMILARITY:
                continue
            dropped.add(id(other_listing))
            listing.duplicate_sites.append(other["site"])
            listing.duplicate_urls.append(other_listing.url)

    for site_result in results:
        site_result.listings = [
            listing for listing in site_result.listings if id(listing) not in dropped
        ]

    require_location = bool(_intent_location_text(prompt_intent).strip())
    _label_match_strength(results, require_location=require_location)


def _label_match_strength(results: List[SiteResult], require_location: bool) -> None:
    listings = [listing for site in results for listing in site.listings]
    scores = [listing.relevance_score for listing in listings if listing.relevance_score is not None]
    if not listings or not scores:
        return
    top_score = max(scores)
    threshold = max(STRONG_MATCH_FLOOR, top_score - STRONG_MATCH_GAP)
    strong_matches = []
    for listing in listings:
        location_ok = True
        if require_location:
            location_ok = (listing.location_score or 0.0) >= STRONG_LOCATION_FLOOR
        if (
            listing.relevance_score is not None
            and listing.relevance_score >= threshold
            and location_ok
        ):
            listing.match_label = "strong"
            strong_matches.append(listing)
        else:
            listing.match_label = "other"
    if strong_matches:
        return
    ranked = sorted(
        [listing for listing in listings if listing.relevance_score is not None],
        key=lambda item: item.relevance_score or 0.0,
        reverse=True,
    )
    fallback_count = 0
    for listing in ranked:
        if require_location and (listing.location_score or 0.0) < STRONG_LOCATION_FLOOR:
            continue
        listing.match_label = "strong"
        fallback_count += 1
        if fallback_count >= STRONG_MATCH_FALLBACK_COUNT:
            break


def run_search(
    prompt: str,
    use_llm: bool = False,
    max_listings: int = 5,
    min_price_chf: int | None = None,
    max_price_chf: int | None = None,
    llm_api_key: str | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> SearchResponse:
    notes = []
    def report(message: str, progress: float) -> None:
        if progress_callback:
            progress_callback(message, progress)

    def site_tracker(label: str, start: float, end: float) -> Callable[[int, int], None]:
        def _tracker(done: int, total: int) -> None:
            if total <= 0:
                report(label, start)
                return
            ratio = min(1.0, done / total)
            report(f"{label} ({done}/{total})", start + (end - start) * ratio)

        return _tracker

    if progress_callback:
        report("Parsing prompt", 0.05)
    intent_struct: Dict[str, str] = {"other": prompt}
    if use_llm:
        try:
            filters = parse_prompt(prompt, use_llm=True, api_key=llm_api_key)
        except Exception as exc:
            filters = heuristic_parse_prompt(prompt)
            notes.append(_format_llm_error(exc))
        if llm_enabled(llm_api_key):
            try:
                intent_struct = extract_search_intent_struct(prompt, api_key=llm_api_key)
            except Exception as exc:
                intent_struct = {"other": prompt}
                notes.append(_format_llm_error(exc))
    else:
        filters = parse_prompt(prompt, use_llm=False, api_key=llm_api_key)
    if progress_callback:
        report("Preparing filters", 0.15)
    if min_price_chf is not None:
        filters.min_price_chf = min_price_chf
    if max_price_chf is not None:
        filters.max_price_chf = max_price_chf
    if use_llm and not llm_enabled(llm_api_key):
        notes.append("LLM parsing requested but no OpenAI API key was provided.")
    if not prompt:
        notes.append("Empty prompt provided.")

    use_embeddings = llm_enabled(llm_api_key)
    if not use_embeddings:
        notes.append("Embedding-based ranking requires an OpenAI API key; using heuristic relevance.")

    results = []
    try:
        report("Fetching Flatfox listings", 0.25)
        flatfox_result = search_flatfox(
            filters,
            max_listings=max_listings,
            progress_callback=site_tracker("Parsing Flatfox listings", 0.25, 0.5),
        )
        flatfox_result.listings = _filter_listings(flatfox_result.listings, filters)
        results.append(flatfox_result)
        report("Parsed Flatfox listings", 0.5)
    except PermissionError as exc:
        notes.append(str(exc))
        report("Flatfox blocked", 0.5)

    try:
        report("Fetching Homegate listings", 0.55)
        homegate_result = search_homegate(
            filters,
            max_listings=max_listings,
            progress_callback=site_tracker("Parsing Homegate listings", 0.55, 0.85),
        )
        homegate_result.listings = _filter_listings(homegate_result.listings, filters)
        results.append(homegate_result)
        report("Parsed Homegate listings", 0.85)
    except PermissionError as exc:
        notes.append(str(exc))
        report("Homegate blocked", 0.85)

    report("Ranking and de-duplicating", 0.9)
    _rank_and_dedupe_results(
        results,
        intent_struct,
        api_key=llm_api_key,
        use_embeddings=use_embeddings,
        notes=notes,
    )
    report("Finalizing results", 0.97)

    return SearchResponse(filters=filters, results=results, notes=notes)
