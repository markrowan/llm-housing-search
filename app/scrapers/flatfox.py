from __future__ import annotations

import re
from typing import Callable, List
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from app.models import DescriptionBlock, FilterSpec, Listing, SiteResult
from app.query_builder import build_flatfox_search_url
from app.scrapers.common import fetch_html
from app.scrapers.parse_utils import (
    extract_section,
    find_address_line,
    normalize_lines,
    parse_area,
    parse_chf_amount,
    parse_rooms,
    split_description,
)


FLATFOX_DOMAIN = "flatfox.ch"


def _is_listing_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc and FLATFOX_DOMAIN not in parsed.netloc:
        return False
    return "/flat/" in parsed.path or "/listing/" in parsed.path


def extract_listing_urls(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if not href:
            continue
        if href.startswith("/"):
            href = urljoin("https://flatfox.ch", href)
        if _is_listing_url(href):
            links.append(href.split("?")[0])
    if not links:
        for match in re.findall(r"https?://[\\w.-]*flatfox\\.ch[^\"'\\s<>]+", html):
            if _is_listing_url(match):
                links.append(match.split("?")[0])
        for match in re.findall(r"/[^\"'\\s<>]*(?:/flat/|/listing/)[^\"'\\s<>]+", html):
            candidate = urljoin("https://flatfox.ch", match)
            if _is_listing_url(candidate):
                links.append(candidate.split("?")[0])
    seen = []
    for link in links:
        if link not in seen:
            seen.append(link)
    return seen


def parse_listing_detail(html: str, url: str) -> Listing:
    soup = BeautifulSoup(html, "html.parser")
    title = None
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)

    lines = normalize_lines(soup.get_text("\n"))
    price = None
    rooms = None
    area = None
    address = None
    stats = {}

    for line in lines:
        if price is None:
            price = parse_chf_amount(line)
        if rooms is None and ("room" in line.lower() or "zimmer" in line.lower()):
            rooms = parse_rooms(line) or rooms
        if area is None and "m" in line:
            area = parse_area(line) or area
        if ":" in line:
            key, value = [part.strip() for part in line.split(":", 1)]
            if key and value:
                stats[key] = value

    address = find_address_line(lines)

    description_lines = extract_section(
        lines,
        heading="Description",
        stop_headings=(
            "Contact advertiser",
            "Subscribe to offer",
            "Similar objects",
            "Surroundings",
            "Beschreibung",
        ),
    )
    if not description_lines:
        description_lines = extract_section(
            lines,
            heading="Beschreibung",
            stop_headings=(
                "Kontakt",
                "Contact advertiser",
                "Subscribe to offer",
                "Similar objects",
                "Surroundings",
                "Description",
            ),
        )
    descriptions = [
        DescriptionBlock(text=text, language=lang)
        for text, lang in split_description(description_lines)
    ]

    return Listing(
        url=url,
        title=title,
        price_chf=price,
        rooms=rooms,
        area_sqm=area,
        address=address,
        stats=stats,
        descriptions=descriptions,
    )


def search_flatfox(
    filters: FilterSpec,
    max_listings: int = 5,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SiteResult:
    search_url = build_flatfox_search_url(filters, take=max_listings * 2)
    html = fetch_html(search_url)
    listing_urls = extract_listing_urls(html)
    if not listing_urls:
        try:
            html = fetch_html(search_url, render_js=True)
        except RuntimeError as exc:
            raise PermissionError(str(exc)) from exc
        listing_urls = extract_listing_urls(html)
    listing_urls = listing_urls[:max_listings]
    listings: List[Listing] = []
    total = len(listing_urls)
    for index, url in enumerate(listing_urls, start=1):
        try:
            detail_html = fetch_html(url)
        except PermissionError:
            continue
        listings.append(parse_listing_detail(detail_html, url))
        if progress_callback:
            progress_callback(index, total)
    return SiteResult(site="flatfox", search_url=search_url, listings=listings)
