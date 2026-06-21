"""
Islamabad house price benchmarks (PKR per marla), aligned with:
- Zameen.com sale trends (May 2026): 5 marla ~22L–1.5Cr, 10 marla ~95L–3Cr+, 1 kanal ~1–11.5Cr+
- Scraped training dataset medians by area band and location tier
"""
from __future__ import annotations

import re

# Area bands used across training and prediction
AREA_BANDS = ["5-6", "7-9", "10-12", "14-20", "20+"]

# PKR per marla ranges [typical, lo, hi] by market tier and area band
# Sources: zameen.com trends + islamabad_properties.csv analysis
MARKET_PPM = {
    "budget": {
        "5-6": {"typical": 3_400_000, "lo": 2_200_000, "hi": 5_000_000},
        "7-9": {"typical": 4_200_000, "lo": 3_000_000, "hi": 6_000_000},
        "10-12": {"typical": 4_800_000, "lo": 3_500_000, "hi": 6_500_000},
        "14-20": {"typical": 5_500_000, "lo": 4_000_000, "hi": 7_500_000},
        "20+": {"typical": 6_500_000, "lo": 5_000_000, "hi": 9_000_000},
    },
    "mid": {
        "5-6": {"typical": 5_600_000, "lo": 4_000_000, "hi": 7_500_000},
        "7-9": {"typical": 5_400_000, "lo": 4_200_000, "hi": 7_000_000},
        "10-12": {"typical": 5_800_000, "lo": 4_500_000, "hi": 8_500_000},
        "14-20": {"typical": 6_200_000, "lo": 5_000_000, "hi": 9_500_000},
        "20+": {"typical": 7_500_000, "lo": 6_000_000, "hi": 12_000_000},
    },
    "premium": {
        "5-6": {"typical": 6_500_000, "lo": 5_000_000, "hi": 9_000_000},
        "7-9": {"typical": 6_800_000, "lo": 5_500_000, "hi": 9_500_000},
        "10-12": {"typical": 7_200_000, "lo": 5_800_000, "hi": 10_500_000},
        "14-20": {"typical": 7_000_000, "lo": 5_500_000, "hi": 11_000_000},
        "20+": {"typical": 9_500_000, "lo": 7_500_000, "hi": 14_000_000},
    },
    "ultra": {
        "5-6": {"typical": 9_000_000, "lo": 6_000_000, "hi": 14_000_000},
        "7-9": {"typical": 10_500_000, "lo": 8_000_000, "hi": 16_000_000},
        "10-12": {"typical": 12_000_000, "lo": 9_000_000, "hi": 18_000_000},
        "14-20": {"typical": 13_500_000, "lo": 10_000_000, "hi": 20_000_000},
        "20+": {"typical": 15_000_000, "lo": 11_000_000, "hi": 25_000_000},
    },
}

ULTRA_PATTERNS = [
    r"\bF-6\b", r"\bF-7\b", r"\bF-8\b", r"\bE-7\b", r"\bE-11\b",
    r"Diplomatic",
]
PREMIUM_PATTERNS = [
    r"DHA Defence", r"DHA Phase", r"Bahria Enclave", r"Park View",
    r"Naval Anchorage",
]
BUDGET_PATTERNS = [
    r"Bani Gala", r"Airport Enclave", r"Soan Garden", r"CBR Town",
    r"Tarlai",
]
FAISAL_PATTERNS = [r"Faisal", r"Canary"]


def area_band(area_marla: float) -> str:
    if area_marla <= 6:
        return "5-6"
    if area_marla <= 9:
        return "7-9"
    if area_marla <= 12:
        return "10-12"
    if area_marla <= 20:
        return "14-20"
    return "20+"


def infer_tier(location: str) -> str:
    loc = location or ""
    for pat in ULTRA_PATTERNS:
        if re.search(pat, loc, re.I):
            return "ultra"
    for pat in PREMIUM_PATTERNS:
        if re.search(pat, loc, re.I):
            return "premium"
    for pat in BUDGET_PATTERNS:
        if re.search(pat, loc, re.I):
            return "budget"
    for pat in FAISAL_PATTERNS:
        if re.search(pat, loc, re.I):
            return "mid"
    return "mid"


def tier_from_median_ppm(ppm: float) -> str:
    if ppm >= 11_000_000:
        return "ultra"
    if ppm >= 7_500_000:
        return "premium"
    if ppm <= 4_000_000:
        return "budget"
    return "mid"


def market_ppm_for(tier: str, band: str) -> dict:
    tier = tier if tier in MARKET_PPM else "mid"
    band = band if band in AREA_BANDS else "10-12"
    return MARKET_PPM[tier][band]
