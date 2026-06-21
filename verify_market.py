"""Verify predictions against known Zameen market ranges."""
import json
import joblib
from predict_utils import calibrated_predict

model = joblib.load("models/best_model.joblib")
with open("models/training_meta.json", encoding="utf-8") as f:
    meta = json.load(f)

# (label, row, expected_lo, expected_hi) in PKR
CASES = [
    (
        "5 Marla Faisal Hills",
        {
            "area_marla": 5,
            "bedrooms": 4,
            "bathrooms": 4,
            "location": "Canary Residency, Faisal Hills",
            "property_type": "House",
            "parking_spaces": 2,
            "servant_quarters": 1,
            "store_rooms": 1,
            "kitchens": 2,
            "drawing_rooms": 1,
        },
        19_000_000,
        28_000_000,
    ),
    (
        "10 Marla DHA Ph2",
        {
            "area_marla": 10,
            "bedrooms": 5,
            "bathrooms": 5,
            "location": "DHA Defence Phase 2, DHA Defence",
            "property_type": "House",
            "parking_spaces": 4,
            "servant_quarters": 2,
            "store_rooms": 2,
            "kitchens": 2,
            "drawing_rooms": 1,
        },
        55_000_000,
        120_000_000,
    ),
    (
        "5 Marla B-17",
        {
            "area_marla": 5,
            "bedrooms": 3,
            "bathrooms": 2,
            "location": "B-17, Islamabad",
            "property_type": "House",
            "parking_spaces": 2,
            "servant_quarters": 1,
            "store_rooms": 1,
            "kitchens": 1,
            "drawing_rooms": 1,
        },
        18_000_000,
        40_000_000,
    ),
    (
        "10 Marla G-13",
        {
            "area_marla": 10,
            "bedrooms": 5,
            "bathrooms": 5,
            "location": "G-13, Islamabad",
            "property_type": "House",
            "parking_spaces": 3,
            "servant_quarters": 1,
            "store_rooms": 1,
            "kitchens": 2,
            "drawing_rooms": 1,
        },
        60_000_000,
        95_000_000,
    ),
    (
        "1 Kanal F-7",
        {
            "area_marla": 20,
            "bedrooms": 6,
            "bathrooms": 5,
            "location": "F-7, Islamabad",
            "property_type": "House",
            "parking_spaces": 5,
            "servant_quarters": 2,
            "store_rooms": 2,
            "kitchens": 2,
            "drawing_rooms": 2,
        },
        250_000_000,
        550_000_000,
    ),
]

print(f"Model: {meta.get('best_model')}\n")
ok = 0
for label, row, lo, hi in CASES:
    p, d = calibrated_predict(model, meta, row)
    good = lo <= p <= hi
    ok += good
    status = "OK" if good else "CHECK"
    print(f"[{status}] {label}")
    print(f"       Predicted: {p/1e7:.2f} Cr  (market band {lo/1e7:.1f}-{hi/1e7:.1f} Cr)")
    print(f"       ML: {d['ml_price']/1e7:.2f} Cr | Base: {d['baseline_price']/1e7:.2f} Cr | tier: {d['tier']}")
    print()
print(f"Passed {ok}/{len(CASES)} market band checks")
