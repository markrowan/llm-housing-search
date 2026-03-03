from __future__ import annotations

from typing import Callable, List
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from app.models import DescriptionBlock, FilterSpec, Listing, SiteResult
from app.query_builder import build_homegate_search_url
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


HOMEGATE_DOMAIN = "www.homegate.ch"


def extract_listing_urls(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if not href:
            continue
        if href.startswith("/rent/"):
            href = urljoin(f"https://{HOMEGATE_DOMAIN}", href)
        if href.startswith(f"https://{HOMEGATE_DOMAIN}/rent/"):
            if href.split("/rent/")[1].split("/")[0].isdigit():
                links.append(href.split("?")[0])
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
            "Contact",
            "Contact advertiser",
            "Visit",
            "Downloads",
            "Application",
            "Similar listings",
            "Beschreibung",
        ),
    )
    if not description_lines:
        description_lines = extract_section(
            lines,
            heading="Beschreibung",
            stop_headings=(
                "Kontakt",
                "Contact",
                "Downloads",
                "Application",
                "Similar listings",
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


def search_homegate(
    filters: FilterSpec,
    max_listings: int = 5,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SiteResult:
    search_url = build_homegate_search_url(filters)
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
    return SiteResult(site="homegate", search_url=search_url, listings=listings)
