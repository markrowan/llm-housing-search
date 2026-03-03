from pathlib import Path

from app.scrapers.flatfox import extract_listing_urls, parse_listing_detail


def test_extract_flatfox_listing_urls():
    html = Path("tests/fixtures/flatfox_search.html").read_text()
    urls = extract_listing_urls(html)
    assert "https://flatfox.ch/en/flat/8048-zurich/85778483/" in urls
    assert "https://p-4259d4.flatfox.ch/en/listing/1220424/" in urls


def test_parse_flatfox_listing_detail():
    html = Path("tests/fixtures/flatfox_listing.html").read_text()
    listing = parse_listing_detail(html, "https://flatfox.ch/en/flat/8048-zurich/85778483/")
    assert listing.title == "Apartment on the 7th floor"
    assert listing.price_chf == 1577
    assert listing.rooms == 1.5
    assert listing.area_sqm == 35
    assert listing.address == "8048 Zürich - CHF 1’577 incl. utilities per month"
    assert len(listing.descriptions) >= 2
