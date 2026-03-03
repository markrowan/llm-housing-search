from pathlib import Path

from app.scrapers.homegate import extract_listing_urls, parse_listing_detail


def test_extract_homegate_listing_urls():
    html = Path("tests/fixtures/homegate_search.html").read_text()
    urls = extract_listing_urls(html)
    assert "https://www.homegate.ch/rent/4002694120" in urls
    assert "https://www.homegate.ch/rent/4002404372" in urls


def test_parse_homegate_listing_detail():
    html = Path("tests/fixtures/homegate_listing.html").read_text()
    listing = parse_listing_detail(html, "https://www.homegate.ch/rent/4002694120")
    assert listing.title == "3.5 room apartment in Zurich"
    assert listing.price_chf == 2650
    assert listing.rooms == 3.5
    assert listing.area_sqm == 85
    assert listing.address == "8005 Zürich"
    assert len(listing.descriptions) >= 2
