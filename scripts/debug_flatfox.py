from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.prompt_parser import parse_prompt
from app.query_builder import build_flatfox_search_url
from app.scrapers.common import fetch_html
from app.scrapers.flatfox import extract_listing_urls


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg and arg.startswith("http"):
        url = arg
    else:
        prompt = arg or (
            "2.5 to 3.5 rooms, in a green and leafy suburb near Kreis 7 "
            "in Zürich and within 20 minutes to the lake"
        )
        filters = parse_prompt(prompt, use_llm=False)
        url = build_flatfox_search_url(filters, take=40)

    print(f"Search URL: {url}")
    html = fetch_html(url, render_js=True)
    output_path = Path("/tmp/flatfox_rendered.html")
    output_path.write_text(html, encoding="utf-8")
    urls = extract_listing_urls(html)
    print(f"Listings found: {len(urls)}")
    for listing in urls[:10]:
        print(f"- {listing}")
    print(f"Saved rendered HTML to {output_path}")


if __name__ == "__main__":
    main()
