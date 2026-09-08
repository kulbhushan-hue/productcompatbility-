# z/OS Product Compatibility Bot (RAG)

Asks for a **z/OS version** and optional **product name**, and returns
which IBM / Broadcom products are compatible — using data scraped from
each vendor's **public** compatibility pages, a structured lookup for the
exact facts, and Claude for a readable summary (RAG on top of the facts).

## Why there's no login in this bot
You asked for a login step (user id + password) to IBM/Broadcom. After
checking both vendors:

- **IBM's** Software Product Compatibility Reports (SPCR) and "detailed
  system requirements" pages are public — no IBM ID required.
- **Broadcom's** z/OS compatibility matrices
  (`ftpdocs.broadcom.com/.../COMPAT/*.HTML`) are also public pages. Only
  Broadcom's *full support-ticket portal* sits behind a login, and that
  isn't needed for compatibility lookups.

So this bot **never collects a username or password**. That's intentional,
not a shortcut:
- Neither vendor publishes an official "login API" for third parties —
  automating a login form with scraped/user-supplied credentials usually
  breaks the vendor's Terms of Service, even for your own account.
- Having a chatbot ask for and hold a raw password (even briefly, even in
  memory/logs) is a security anti-pattern, full stop.

**If you later need content that genuinely requires login** (e.g. an
entitled Broadcom support-portal PDF), the right pattern is:
1. Get official API credentials from the vendor for your account (an API
   key or OAuth token issued by them), not raw username/password automation, **or**
2. Have a person manually download the entitled document through their
   own logged-in browser session and feed the file into the ingestion
   pipeline (`ingest/`) — no credentials touch the bot at all.

## Architecture
```
scrapers/broadcom_scraper.py   -> scrapes public Broadcom compatibility HTML tables
scrapers/ibm_scraper.py        -> scrapes public IBM support pages (+ optional
                                   Playwright-based deep check against the
                                   live SPCR tool for exact z/OS version detail)
data/*.csv                     -> scraped output (sample/demo data included)
core/compatibility_db.py       -> structured (pandas) exact YES/NO lookup
core/rag_engine.py             -> TF-IDF retrieval over notes/context + Claude synthesis
app.py                         -> Streamlit dashboard (chat-style Q&A)
```

**Why structured lookup + RAG, not RAG alone:** "Is product X compatible
with z/OS Y" is exact tabular data — a vector search over free text is the
wrong tool for that and can hallucinate compatibility. So the structured
DB is the ground truth for the YES/NO facts, and the RAG/LLM layer only
adds the narrative wrapper (grouping, PTF caveats, phrasing) — the system
prompt in `rag_engine.py` explicitly tells Claude never to state a product
is compatible unless it's in the structured matches.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # then edit .env with your real key
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```
The app ships with **sample/demo data** in `data/*_sample.csv` so it runs
out of the box. Run the scrapers below to get real, current data.

## Refreshing the data (run periodically, e.g. weekly via cron/CI)
```bash
cd scrapers
python broadcom_scraper.py --out ../data/broadcom_compat.csv
python ibm_scraper.py --query "detailed system requirements z/OS" --out ../data/ibm_compat.csv
```
- Both scrapers rate-limit themselves (1 req/sec) and send a descriptive
  User-Agent. Check `robots.txt` and each vendor's Terms of Use before
  running at scale or on a schedule, and back off immediately if you see
  403s / CAPTCHAs — that means the page owner doesn't want automated
  access and you should switch to manual export instead.
- For exact z/OS-version-level detail on a specific IBM product, use
  `ibm_scraper.deep_check_spcr(product_name, zos_version)` — this drives
  the live SPCR tool with Playwright, on demand, for one product at a
  time (see docstring for setup and caveats: IBM can change the UI at any
  time, so the selectors are best-effort).

## Deploying (GitHub + Streamlit Community Cloud)
```bash
git init
git add .
git commit -m "z/OS compatibility RAG bot"
git remote add origin <your-repo-url>
git push -u origin main
```
On [share.streamlit.io](https://share.streamlit.io): **New app** → point
to `app.py` → add `ANTHROPIC_API_KEY` under **Secrets** → Deploy.
(Never commit your `.env` file or real API key to git — `.env.example`
is the template only.)

## Using the dashboard
1. Enter a z/OS version (e.g. `z/OS V2R5` or just `2.5`).
2. Optionally narrow to a product name (e.g. `Endevor`) or vendor.
3. Click **Ask** — see a natural-language answer, the raw structured
   matches (downloadable as CSV), and a coverage chart by vendor.

## Known limitations
- Sample data included is illustrative (clearly marked "(sample)" in
  product names) — run the scrapers for real current data before using
  this for real change/upgrade decisions.
- IBM's exact per-version z/OS requirement often needs the dynamic SPCR
  tool (`deep_check_spcr`) rather than the static support pages alone.
- Always cross-check anything safety/production-critical against the
  live vendor page — this tool accelerates lookup, it isn't a substitute
  for the vendor's official current compatibility statement.
