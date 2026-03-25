"""
IIT Jodhpur Deep Scraper — HTML + PDF
Crawls seed URLs, follows internal links, AND downloads + extracts
text from PDFs found on those pages. PDFs are the fastest way to
get a large token corpus.

Requirements:
    pip install requests beautifulsoup4 pypdf pdfplumber
"""

import re
import time
import os
from urllib.parse import urljoin, urldefrag, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import deque
import io

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pdfplumber   # better text extraction from PDFs than pypdf

# ── Config ────────────────────────────────────────────────────────────────────
SEED_URLS = [
    # 1. Academic Regulations (MUST)
    "https://iitj.ac.in/office-of-academics/en/academic-regulations",
    # 2. Academic Programs
    "https://iitj.ac.in/office-of-academics/en/academic-programs?ep=fw",
    # 3. Program Structure
    "https://iitj.ac.in/office-of-academics/en/program-Structure",
    # 4. Curriculum
    "https://iitj.ac.in/office-of-academics/en/curriculum",
]

ALLOWED_DOMAIN  = "iitj.ac.in"
MAX_HTML_PAGES  = 200      # HTML pages crawled per seed URL
MAX_PDFS        = 40      # max PDFs downloaded across ALL seeds combined
MAX_WORKERS     = 10      # concurrent threads
REQUEST_TIMEOUT = 10
DELAY_BETWEEN   = 0.2     # seconds between batches (be polite)
OUTPUT_DIR      = "corpus"

NON_LATIN_RE = re.compile(r"[\u0900-\uFFFF]+")  # strips Devanagari etc.

SKIP_EXT = re.compile(
    r"\.(jpg|jpeg|png|gif|svg|zip|rar|xls|xlsx|ppt|pptx"
    r"|mp4|mp3|avi|mov|css|js|ico|woff|woff2|ttf)(\?.*)?$",
    re.IGNORECASE,
)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AcademicScraper/1.0)"}
# ─────────────────────────────────────────────────────────────────────────────


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    retry = Retry(total=3, backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry,
                          pool_connections=MAX_WORKERS,
                          pool_maxsize=MAX_WORKERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ── URL helpers ───────────────────────────────────────────────────────────────

def is_valid_html_url(url: str) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return False
    if ALLOWED_DOMAIN not in p.netloc:
        return False
    if SKIP_EXT.search(p.path):
        return False
    if p.path.lower().endswith(".pdf"):
        return False
    return True


def is_pdf_url(url: str) -> bool:
    p = urlparse(url)
    return (
        ALLOWED_DOMAIN in p.netloc
        and p.path.lower().endswith(".pdf")
        and p.scheme in ("http", "https")
    )


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    text = NON_LATIN_RE.sub(" ", text)          # remove non-Latin scripts
    text = re.sub(r"https?://\S+", " ", text)   # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)        # remove emails
    text = re.sub(r"[ \t]+", " ", text)         # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)      # collapse blank lines
    return text.strip()


# ── HTML scraping ─────────────────────────────────────────────────────────────

def extract_html_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "head",
                     "nav", "footer", "header", "aside", "iframe"]):
        tag.decompose()
    main = (
        soup.find("main")
        or soup.find(id=re.compile(r"content|main", re.I))
        or soup.find("div", class_=re.compile(r"content|main|body", re.I))
        or soup.body or soup
    )
    return clean(main.get_text(separator=" "))


def extract_links(soup: BeautifulSoup, base_url: str):
    """Return (html_links, pdf_links) found on the page."""
    html_links, pdf_links = [], []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        full, _ = urldefrag(urljoin(base_url, href))
        if is_pdf_url(full):
            pdf_links.append(full)
        elif is_valid_html_url(full):
            html_links.append(full)
    return list(set(html_links)), list(set(pdf_links))


def scrape_html(url: str, session: requests.Session):
    """Returns (url, text, html_links, pdf_links) or (url, None, [], [])."""
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        if "html" not in resp.headers.get("Content-Type", ""):
            return url, None, [], []
        soup = BeautifulSoup(resp.text, "html.parser")
        text = extract_html_text(soup)
        html_links, pdf_links = extract_links(soup, url)
        return url, text, html_links, pdf_links
    except Exception as e:
        print(f"  [html-skip] {url} -> {e}")
        return url, None, [], []


# ── PDF scraping ──────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    try:
        pages_text = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total = len(pdf.pages)
            print(f"({total} pages)", end=" ", flush=True)
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
        return clean("\n".join(pages_text))
    except Exception as e:
        print(f"\n  [pdf-parse-err] {e}")
        return ""


def scrape_pdf(url: str, session: requests.Session):
    """Download and extract text from a PDF. Returns (url, text)."""
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT * 3)  # PDFs are bigger
        resp.raise_for_status()
        if len(resp.content) < 1000:
            return url, ""
        print(f"  PDF  {url} ... ", end="", flush=True)
        text = extract_pdf_text(resp.content)
        tokens_est = len(text.split())
        kb = len(resp.content) // 1024
        print(f"~{tokens_est:,} tokens  ({kb} KB)")
        return url, text
    except Exception as e:
        print(f"\n  [pdf-skip] {url} -> {e}")
        return url, ""


# ── Crawl one seed ────────────────────────────────────────────────────────────

def crawl_seed(seed_url: str, session: requests.Session,
               global_pdf_seen: set, pdf_lock: Lock):
    """
    BFS crawl from seed_url.
    Returns (html_pages dict, pdf_pages dict).
    """
    visited_html = set()
    queue        = deque([seed_url])
    html_results = {}
    pdf_results  = {}
    lock         = Lock()

    print(f"\n{'─'*65}")
    print(f"  Seed : {seed_url}")
    print(f"{'─'*65}")

    while queue and len(visited_html) < MAX_HTML_PAGES:
        batch = []
        while queue and len(batch) < MAX_WORKERS:
            url = queue.popleft()
            if url not in visited_html:
                visited_html.add(url)
                batch.append(url)
        if not batch:
            break

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(scrape_html, u, session): u for u in batch}
            for future in as_completed(futures):
                url, text, html_links, pdf_links = future.result()

                if text:
                    html_results[url] = text
                    print(f"  HTML [{len(html_results):>3}]  {url}")

                with lock:
                    for link in html_links:
                        if link not in visited_html:
                            queue.append(link)

                # Download any newly discovered PDFs
                with pdf_lock:
                    for pdf_url in pdf_links:
                        if (pdf_url not in global_pdf_seen
                                and len(global_pdf_seen) < MAX_PDFS):
                            global_pdf_seen.add(pdf_url)
                            _, pdf_text = scrape_pdf(pdf_url, session)
                            if pdf_text:
                                pdf_results[pdf_url] = pdf_text

        time.sleep(DELAY_BETWEEN)

    print(f"\n  -> HTML: {len(html_results)} pages | "
          f"PDFs discovered: {len(pdf_results)} files")
    return html_results, pdf_results


# ── Save ──────────────────────────────────────────────────────────────────────

def save(pages: dict, filepath: str, source_type: str):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for url, text in pages.items():
            f.write(f"\n{'='*70}\n")
            f.write(f"SOURCE [{source_type}]: {url}\n")
            f.write(f"{'='*70}\n\n")
            f.write(text)
            f.write("\n")
    tokens = sum(len(t.split()) for t in pages.values())
    kb     = round(sum(len(t) for t in pages.values()) / 1024, 1)
    print(f"  Saved -> {filepath}  ({len(pages)} docs, ~{tokens:,} tokens, {kb} KB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session         = make_session()
    global_pdf_seen = set(KNOWN_PDF_URLS)
    pdf_lock        = Lock()

    all_html: dict = {}
    all_pdfs: dict = {}

    # ── Step 1: Download known PDFs directly (fast, high token yield) ─────────
    print("\n" + "="*65)
    print("  STEP 1 — Downloading known PDFs (fastest tokens)")
    print("="*65)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(scrape_pdf, url, session): url
            for url in KNOWN_PDF_URLS
        }
        for future in as_completed(futures):
            url, text = future.result()
            if text:
                all_pdfs[url] = text

    # ── Step 2: Crawl HTML seeds and discover more PDFs ───────────────────────
    print("\n" + "="*65)
    print("  STEP 2 — Crawling HTML pages + discovering linked PDFs")
    print("="*65)
    for i, seed in enumerate(SEED_URLS, 1):
        html_pages, pdf_pages = crawl_seed(
            seed, session, global_pdf_seen, pdf_lock
        )
        if html_pages:
            save(html_pages, f"{OUTPUT_DIR}/html_doc{i}.txt", "HTML")
            all_html.update(html_pages)
        if pdf_pages:
            save(pdf_pages, f"{OUTPUT_DIR}/pdf_doc{i}.txt", "PDF")
            all_pdfs.update(pdf_pages)

    # ── Step 3: Save combined corpus files ────────────────────────────────────
    print("\n" + "="*65)
    print("  STEP 3 — Saving combined corpus")
    print("="*65)
    if all_html:
        save(all_html, f"{OUTPUT_DIR}/ALL_html.txt", "HTML")
    if all_pdfs:
        save(all_pdfs, f"{OUTPUT_DIR}/ALL_pdf.txt",  "PDF")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_tokens = (
        sum(len(t.split()) for t in all_html.values())
        + sum(len(t.split()) for t in all_pdfs.values())
    )
    print(f"""
{'='*65}
  SCRAPING COMPLETE
  HTML pages scraped : {len(all_html)}
  PDFs downloaded    : {len(all_pdfs)}
  Total documents    : {len(all_html) + len(all_pdfs)}
  Estimated tokens   : ~{total_tokens:,}
  Output folder      : {OUTPUT_DIR}/
{'='*65}
""")


if __name__ == "__main__":
    main()