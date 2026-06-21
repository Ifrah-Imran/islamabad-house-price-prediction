"""Shared helpers for scraping and ML."""
import re
from typing import Optional


def parse_price(text: str) -> Optional[float]:
    """Convert Zameen price text to PKR (float)."""
    if not text:
        return None
    t = text.strip().replace(",", "").replace("PKR", "").strip()
    m = re.search(r"([\d.]+)\s*(Crore|Lakh|Arab|Million|Thousand)?", t, re.I)
    if not m:
        return None
    value = float(m.group(1))
    unit = (m.group(2) or "").lower()
    multipliers = {
        "crore": 10_000_000,
        "lakh": 100_000,
        "arab": 1_000_000_000,
        "million": 1_000_000,
        "thousand": 1_000,
    }
    return value * multipliers.get(unit, 1)


def parse_area_marla(text: str) -> Optional[float]:
    """Normalize area to marla."""
    if not text:
        return None
    t = text.strip().replace(",", "")
    m = re.search(r"([\d.]+)\s*(Kanal|Marla|Sq\.?\s*Ft\.?|Sq\.?\s*Yd\.?)?", t, re.I)
    if not m:
        return None
    value = float(m.group(1))
    unit = (m.group(2) or "marla").lower().replace(".", "").replace(" ", "")
    if "kanal" in unit:
        return value * 20
    if "marla" in unit or unit == "":
        return value
    if "sqft" in unit or "sqft" in t.lower():
        return value / 225
    if "sqyd" in unit:
        return (value * 9) / 225
    return value


def extract_detail_features(text: str) -> dict:
    """Parse amenity counts from property detail page text."""
    patterns = {
        "built_year": r"Built\s+in\s+year\s*:?\s*(\d{4})",
        "parking_spaces": r"Parking\s+Spaces?\s*:?\s*(\d+)",
        "servant_quarters": r"Servant\s+Quarters?\s*:?\s*(\d+)",
        "store_rooms": r"Store\s+Rooms?\s*:?\s*(\d+)",
        "kitchens": r"Kitchens?\s*:?\s*(\d+)",
        "drawing_rooms": r"Drawing\s+Rooms?\s*:?\s*(\d+)",
    }
    out = {}
    for key, pat in patterns.items():
        m = re.search(pat, text, re.I)
        out[key] = int(m.group(1)) if m else None
    if out.get("drawing_rooms") is None and re.search(r"Drawing\s+Room", text, re.I):
        out["drawing_rooms"] = 1
    return out


def format_price_pkr(value: float) -> str:
    """Human-readable PKR for UI."""
    if value >= 10_000_000:
        return f"PKR {value / 10_000_000:.2f} Crore"
    if value >= 100_000:
        return f"PKR {value / 100_000:.2f} Lakh"
    return f"PKR {value:,.0f}"
