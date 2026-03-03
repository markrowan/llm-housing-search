from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

USER_AGENT = "HousingSearchBot/0.1 (respecting robots.txt)"


class RobotsCache:
    def __init__(self) -> None:
        self._cache: Dict[str, RobotFileParser] = {}

    def get_parser(self, base_url: str, client: httpx.Client) -> RobotFileParser:
        parsed = urlparse(base_url)
        key = f"{parsed.scheme}://{parsed.netloc}"
        if key in self._cache:
            return self._cache[key]
        robots_url = f"{key}/robots.txt"
        parser = RobotFileParser()
        try:
            response = client.get(robots_url, headers={"User-Agent": USER_AGENT}, timeout=10)
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            else:
                parser.parse("")
        except httpx.HTTPError:
            parser.parse("")
        self._cache[key] = parser
        return parser

    def can_fetch(self, url: str, client: httpx.Client) -> bool:
        parser = self.get_parser(url, client)
        return parser.can_fetch(USER_AGENT, url)


ROBOTS_CACHE = RobotsCache()


def _render_html_with_playwright(url: str) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Install it with `pip install playwright` "
            "and run `python -m playwright install chromium` to enable JS rendering."
        ) from exc
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(user_agent=USER_AGENT)
        page = context.new_page()
        response = page.goto(url, wait_until="domcontentloaded", timeout=30000)
        status = response.status if response else None
        if status in {401, 403, 429}:
            raise PermissionError(
                f"{url} returned {status} (likely blocking automated requests)."
            )
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        try:
            page.wait_for_selector("a[href*='/flat/'], a[href*='/listing/']", timeout=8000)
        except Exception:
            pass
        html = page.content()
        context.close()
        browser.close()
        time.sleep(0.5)
        return html


def _render_html_with_playwright_threaded(url: str) -> str:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_render_html_with_playwright, url)
        return future.result()


def fetch_html(
    url: str,
    client: Optional[httpx.Client] = None,
    respect_robots: bool = True,
    render_js: bool = False,
) -> str:
    close_client = False
    if client is None:
        client = httpx.Client(follow_redirects=True, timeout=20)
        close_client = True
    try:
        if respect_robots and not ROBOTS_CACHE.can_fetch(url, client):
            raise PermissionError(f"robots.txt disallows fetching {url}")
        if render_js:
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    return _render_html_with_playwright_threaded(url)
            except RuntimeError:
                pass
            return _render_html_with_playwright(url)
        response = client.get(url, headers={"User-Agent": USER_AGENT})
        if response.status_code in {401, 403, 429}:
            raise PermissionError(
                f"{url} returned {response.status_code} (likely blocking automated requests)."
            )
        response.raise_for_status()
        time.sleep(0.5)
        return response.text
    finally:
        if close_client:
            client.close()
