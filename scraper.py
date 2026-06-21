
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from utils import extract_detail_features, parse_area_marla, parse_price

BASE_URL = "https://www.zameen.com/Houses_Property/Islamabad-3-{page}.html"
CITY = "Islamabad"
PROPERTY_TYPE = "House"
TARGET_MIN = 300
TARGET_MAX = 400
OUTPUT_CSV = Path("data/islamabad_properties.csv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

FIELDNAMES = [
    "price",
    "area_marla",
    "city",
    "bedrooms",
    "bathrooms",
    "location",
    "property_type",
    "built_year",
    "parking_spaces",
    "servant_quarters",
    "store_rooms",
    "kitchens",
    "drawing_rooms",
    "listing_url",
]


def fetch(session: requests.Session, url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=35)
            r.raise_for_status()
            return r.text
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
            print(f"  Retry {attempt + 1} for {url}: {exc}")
    return ""


def parse_listing_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for card in soup.select('li[aria-label="Listing"]'):
        price_el = card.select_one('[aria-label="Price"]')
        beds_el = card.select_one('[aria-label="Beds"]')
        baths_el = card.select_one('[aria-label="Baths"]')
        loc_el = card.select_one('[aria-label="Location"]')
        area_el = card.select_one('[aria-label="Area"]')
        link_el = card.find("a", href=True)

        price = parse_price(price_el.get_text(strip=True) if price_el else "")
        if not price:
            continue

        href = link_el["href"] if link_el else ""
        listing_url = urljoin("https://www.zameen.com", href) if href else ""

        def _int(el):
            if not el:
                return 0
            try:
                return int(re.sub(r"\D", "", el.get_text(strip=True)) or 0)
            except ValueError:
                return 0

        rows.append(
            {
                "price": int(price),
                "area_marla": parse_area_marla(area_el.get_text(strip=True) if area_el else "") or 0,
                "city": CITY,
                "bedrooms": _int(beds_el),
                "bathrooms": _int(baths_el),
                "location": loc_el.get_text(strip=True) if loc_el else "",
                "property_type": PROPERTY_TYPE,
                "built_year": None,
                "parking_spaces": None,
                "servant_quarters": None,
                "store_rooms": None,
                "kitchens": None,
                "drawing_rooms": None,
                "listing_url": listing_url,
            }
        )
    return rows


def enrich_from_detail(session: requests.Session, row: dict) -> dict:
    url = row.get("listing_url")
    if not url:
        return row
    try:
        html = fetch(session, url)
        text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
        feats = extract_detail_features(text)
        for k, v in feats.items():
            if v is not None:
                row[k] = v
        type_el = BeautifulSoup(html, "html.parser").select_one('[aria-label="Type"]')
        if type_el:
            row["property_type"] = type_el.get_text(strip=True) or PROPERTY_TYPE
    except Exception as exc:
        print(f"  Detail skip: {url[:60]}... ({exc})")
    return row


def scrape(target_count: int = TARGET_MAX, fetch_details: bool = True) -> list[dict]:
    session = requests.Session()
    all_rows: list[dict] = []
    seen_urls: set[str] = set()
    page = 1

    print(f"Scraping Islamabad houses (target: {target_count})...")
    while len(all_rows) < target_count and page <= 25:
        url = BASE_URL.format(page=page)
        print(f"Page {page}: {url}")
        html = fetch(session, url)
        page_rows = parse_listing_page(html)

        if not page_rows:
            print("  No listings on page — stopping.")
            break

        for row in page_rows:
            u = row["listing_url"]
            if u and u in seen_urls:
                continue
            if u:
                seen_urls.add(u)
            all_rows.append(row)
            if len(all_rows) >= target_count:
                break

        print(f"  Collected {len(page_rows)} | Total: {len(all_rows)}")
        page += 1
        time.sleep(1.2)

    if fetch_details:
        print(f"\nFetching detail pages for amenities ({len(all_rows)} listings)...")

        def _enrich(row):
            s = requests.Session()
            return enrich_from_detail(s, row)

        done = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_enrich, row): row for row in all_rows}
            for fut in as_completed(futures):
                fut.result()
                done += 1
                if done % 50 == 0:
                    print(f"  Details: {done}/{len(all_rows)}")

    return all_rows[:target_count]


def save_csv(rows: list[dict], path: Path = OUTPUT_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} records -> {path.resolve()}")


if __name__ == "__main__":
    data = scrape(target_count=TARGET_MAX, fetch_details=True)
    if len(data) < TARGET_MIN:
        print(f"Warning: only {len(data)} records (minimum {TARGET_MIN} required).")
    save_csv(data)
