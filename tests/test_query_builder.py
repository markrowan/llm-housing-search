from urllib.parse import parse_qs, urlparse

from app.models import FilterSpec
from app.query_builder import build_flatfox_search_url, build_homegate_search_url


def test_flatfox_url_includes_rooms_and_bbox():
    filters = FilterSpec(
        raw_prompt="test",
        min_rooms=2.5,
        max_rooms=3.5,
        location_tags=["Kreis 7"],
    )
    url = build_flatfox_search_url(filters, take=10)
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    assert qs["min_rooms"] == ["2.5"]
    assert qs["max_rooms"] == ["3.5"]
    assert "north" in qs and "south" in qs and "east" in qs and "west" in qs
    assert qs["take"] == ["10"]


def test_homegate_city_path():
    filters = FilterSpec(raw_prompt="test", location_tags=["Zürich"])
    url = build_homegate_search_url(filters)
    assert url.endswith("/rent/apartment/city-zurich/matching-list")
