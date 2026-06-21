import json
import joblib
import pandas as pd
from pathlib import Path

model = joblib.load("models/best_model.joblib")
with open("models/training_meta.json") as f:
    meta = json.load(f)

defaults = meta["imputation_defaults"]
samples = [
    {"area_marla": 10, "bedrooms": 4, "bathrooms": 3, "location": "DHA Defence Phase 2, DHA Defence"},
    {"area_marla": 5, "bedrooms": 3, "bathrooms": 2, "location": "F-7, Islamabad"},
    {"area_marla": 1, "bedrooms": 1, "bathrooms": 1, "location": "G-13, Islamabad"},
]

for s in samples:
    row = {**defaults, **s, "property_type": defaults.get("property_type", "House")}
    X = pd.DataFrame([row])[meta["feature_columns"]]
    p = model.predict(X)[0]
    print(f"{s['area_marla']} marla, {s['bedrooms']} bed, {s['location'][:30]}... -> PKR {p:,.0f}")
