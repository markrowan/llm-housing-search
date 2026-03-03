# Housing Search

Simple FastAPI app that turns a plain-text housing prompt into a targeted scrape of Flatfox and Homegate, with optional LLM-backed parsing.

## Features
- Prompt parsing with heuristics or OpenAI (server-side) when `OPENAI_API_KEY` is set.
- Site-specific URL builders (Flatfox bounding-box queries, Homegate city paths).
- Polite scraping with `robots.txt` checks and a small delay.
- Minimal UI to submit prompts and inspect extracted stats and descriptions.
- Tests for prompt parsing, URL building, and HTML parsing with fixtures.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

## LLM Parsing
Set the OpenAI API key before starting the server:
```bash
export OPENAI_API_KEY="..."
```
Or provide an OpenAI API key in the UI per request.
Optional:
```bash
export OPENAI_MODEL="gpt-4.1-mini"
```

## Headless Rendering (Optional)
Some sites render listings with JavaScript. To enable headless rendering:
```bash
pip install playwright
python -m playwright install chromium
```

## Tests
```bash
python3 -m pytest -q
```

## Notes and caveats
- The app respects `robots.txt` and will skip scraping if disallowed.
- The Homegate URL builder intentionally avoids query parameters and only uses city paths, because `robots.txt` disallows many query-string variants.
- Flatfox query parameters are best-effort (bounding boxes and room limits). If they change, update `app/query_builder.py`.
- The scraper does not attempt to collect contact details or personal data.
