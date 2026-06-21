"""Market-aware prediction with location + area-band calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd

from market_rates import (
    MARKET_PPM,
    area_band,
    infer_tier,
    market_ppm_for,
    tier_from_median_ppm,
)


def build_location_profiles(df: pd.DataFrame) -> dict:
    """Per-location stats including area-band PKR/marla from training data."""
    profiles = {}
    for loc, g in df.groupby("location"):
        g = g[g["area_marla"] > 0].copy()
        if len(g) == 0:
            continue
        g["ppm"] = g["price"] / g["area_marla"]
        g["band"] = g["area_marla"].apply(area_band)

        bands = {}
        for band, bg in g.groupby("band"):
            ppm = bg["ppm"]
            bands[band] = {
                "count": int(len(bg)),
                "median_ppm": float(ppm.median()),
                "median_price": float(bg["price"].median()),
            }

        ppm_all = g["ppm"]
        med_ppm = float(ppm_all.median())
        tier = tier_from_median_ppm(med_ppm)
        # Keyword override for known premium/ultra areas with few samples
        tier_kw = infer_tier(loc)
        if tier_kw == "ultra" or (tier_kw == "premium" and tier == "mid"):
            tier = tier_kw
        if tier_kw == "budget" and tier in ("mid", "premium"):
            tier = "budget"

        profiles[loc] = {
            "count": int(len(g)),
            "tier": tier,
            "median_price": float(g["price"].median()),
            "median_price_per_marla": med_ppm,
            "p10_price": float(g["price"].quantile(0.15)),
            "p90_price": float(g["price"].quantile(0.85)),
            "median_area_marla": float(g["area_marla"].median()),
            "median_bedrooms": int(g["bedrooms"].median()),
            "median_bathrooms": int(g["bathrooms"].median()),
            "median_parking_spaces": float(g["parking_spaces"].median()),
            "median_servant_quarters": float(g["servant_quarters"].median()),
            "median_store_rooms": float(g["store_rooms"].median()),
            "median_kitchens": float(g["kitchens"].median()),
            "median_drawing_rooms": float(g["drawing_rooms"].median()),
            "bands": bands,
        }
    return profiles


def _band_baseline(row: dict, prof: dict) -> float | None:
    band = area_band(row["area_marla"])
    band_info = prof.get("bands", {}).get(band)
    if band_info and band_info["count"] >= 1:
        return band_info["median_ppm"] * row["area_marla"]
    if prof.get("count", 0) >= 2:
        return prof["median_price_per_marla"] * row["area_marla"]
    return None


def _market_tier_baseline(row: dict, tier: str) -> float:
    band = area_band(row["area_marla"])
    rates = market_ppm_for(tier, band)
    ppm = rates["typical"]
    base = ppm * row["area_marla"]
    # Slight adjustment for extra bedrooms vs typical 4 bed house
    bed_adj = 1 + 0.025 * max(0, row.get("bedrooms", 4) - 4)
    bath_adj = 1 + 0.015 * max(0, row.get("bathrooms", 3) - 3)
    return base * bed_adj * bath_adj


def _clamp_to_market(price: float, row: dict, prof: dict | None, tier: str) -> float:
    band = area_band(row["area_marla"])
    rates = market_ppm_for(tier, band)
    area = row["area_marla"]
    lo = rates["lo"] * area * 0.90
    hi = rates["hi"] * area * 1.15

    if prof:
        if prof.get("p10_price"):
            lo = max(lo, prof["p10_price"] * 0.75)
        if prof.get("p90_price"):
            hi = min(hi, prof["p90_price"] * 1.35)
        if prof.get("count", 0) >= 3:
            lo = max(lo, prof["median_price"] * 0.55)
            hi = min(hi, prof["median_price"] * 1.65)

    return float(np.clip(price, lo, hi))


def calibration_weight(sample_count: int, has_band_data: bool) -> float:
    """Trust market comps more when we have few or band-specific listings."""
    if has_band_data and sample_count >= 3:
        return 0.72
    if sample_count <= 2:
        return 0.82
    if sample_count <= 5:
        return 0.68
    if sample_count <= 12:
        return 0.52
    if sample_count <= 25:
        return 0.38
    return 0.28


def _row_for_ml(row: dict, meta: dict) -> dict:
    """Build feature dict for sklearn model from UI row."""
    location_map = meta.get("location_map", {})
    loc = row.get("location", "")
    out = {
        "area_marla": float(row["area_marla"]),
        "bedrooms": float(row["bedrooms"]),
        "bathrooms": float(row["bathrooms"]),
        "location_model": location_map.get(loc, "Other Islamabad"),
        "property_type": row.get("property_type", "House"),
        "area_band": area_band(float(row["area_marla"])),
        "total_rooms": float(row.get("bedrooms", 0))
        + float(row.get("bathrooms", 0))
        + float(row.get("kitchens", 2))
        + float(row.get("drawing_rooms", 1)),
        "parking_spaces": float(row.get("parking_spaces", 2)),
        "servant_quarters": float(row.get("servant_quarters", 1)),
        "store_rooms": float(row.get("store_rooms", 1)),
        "kitchens": float(row.get("kitchens", 2)),
        "drawing_rooms": float(row.get("drawing_rooms", 1)),
    }
    return out


def calibrated_predict(model, meta: dict, row: dict) -> tuple[float, dict]:
    """
    Blend ML with location/area-band market baseline, then clamp to realistic PKR range.
    """
    feature_cols = meta["feature_columns"]
    ml_row = _row_for_ml(row, meta)
    X = pd.DataFrame([{c: ml_row[c] for c in feature_cols}])
    ml_price = float(model.predict(X)[0])

    loc = row["location"]
    profiles = meta.get("location_profiles", {})
    prof = profiles.get(loc)
    band = area_band(row["area_marla"])
    has_band = bool(prof and prof.get("bands", {}).get(band))

    tier = prof["tier"] if prof else infer_tier(loc)
    info = {
        "ml_price": ml_price,
        "calibrated": False,
        "baseline_price": None,
        "market_baseline": None,
        "blend_weight": 0.0,
        "tier": tier,
        "area_band": band,
    }

    local_base = _band_baseline(row, prof) if prof else None
    market_base = _market_tier_baseline(row, tier)
    info["market_baseline"] = market_base

    if local_base is not None:
        baseline = local_base
        n = prof["count"]
    else:
        baseline = market_base
        n = 0

    info["baseline_price"] = baseline
    w = calibration_weight(n, has_band)

    # When ML diverges heavily from market comps, lean harder on baseline
    if baseline > 0:
        ratio = ml_price / baseline
        if ratio > 1.45 or ratio < 0.55:
            w = min(0.88, w + 0.15)

    final = w * baseline + (1 - w) * ml_price

    # When ML disagrees with local market comps, trust comps more
    if baseline > 0:
        ratio = ml_price / baseline
        if ratio < 0.65 or ratio > 1.50:
            final = 0.78 * baseline + 0.22 * ml_price
        elif ratio < 0.82 or ratio > 1.28:
            final = 0.65 * baseline + 0.35 * ml_price

    final = _clamp_to_market(final, row, prof, tier)

    info.update(
        {
            "calibrated": True,
            "blend_weight": w,
            "location_samples": n,
            "clamped": True,
        }
    )
    return max(final, 0), info


def defaults_for_location(meta: dict, location: str) -> dict:
    global_defaults = meta.get("imputation_defaults", {})
    prof = meta.get("location_profiles", {}).get(location)
    if not prof:
        return dict(global_defaults)
    return {
        "area_marla": prof["median_area_marla"],
        "bedrooms": prof["median_bedrooms"],
        "bathrooms": prof["median_bathrooms"],
        "parking_spaces": int(prof["median_parking_spaces"]),
        "servant_quarters": int(prof["median_servant_quarters"]),
        "store_rooms": int(prof["median_store_rooms"]),
        "kitchens": int(prof["median_kitchens"]),
        "drawing_rooms": int(prof["median_drawing_rooms"]),
        "property_type": global_defaults.get("property_type", "House"),
    }
