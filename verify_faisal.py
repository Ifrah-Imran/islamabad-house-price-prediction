import json
import joblib
from predict_utils import calibrated_predict

model = joblib.load("models/best_model.joblib")
with open("models/training_meta.json") as f:
    meta = json.load(f)

row = {
    "area_marla": 5.0,
    "bedrooms": 4,
    "bathrooms": 4,
    "location": "Canary Residency, Faisal Hills",
    "property_type": "House",
    "parking_spaces": 2,
    "servant_quarters": 1,
    "store_rooms": 1,
    "kitchens": 2,
    "drawing_rooms": 1,
}
price, dbg = calibrated_predict(model, meta, row)
print("Faisal Hills 5 marla:")
print(f"  Final: {price:,.0f} ({price/1e7:.2f} Cr)")
print(f"  ML only: {dbg['ml_price']:,.0f}")
print(f"  Baseline: {dbg.get('baseline_price', 0):,.0f}")
print(f"  Blend: {dbg.get('blend_weight', 0)*100:.0f}% baseline")
print(f"  Actual medians in data: 21-25M")
