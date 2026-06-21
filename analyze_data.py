import pandas as pd
import joblib
import json

df = pd.read_csv("data/islamabad_properties.csv")
print("built_year null%", df["built_year"].isna().mean() * 100)
print(df["built_year"].describe())
print("year range", df["built_year"].min(), df["built_year"].max())

fh = df[df["location"].str.contains("Faisal", case=False, na=False)]
print("\nFaisal rows:", len(fh))
print(fh.groupby("location")["price"].agg(["median", "count", "mean"]))
print(fh[["location", "price", "area_marla", "bedrooms", "built_year"]].head(10))

model = joblib.load("models/best_model.joblib")
with open("models/training_meta.json") as f:
    meta = json.load(f)

# Test Faisal Hills predictions vs actual medians
for loc in fh["location"].unique()[:5]:
    sub = fh[fh["location"] == loc]
    med_area = sub["area_marla"].median()
    med_beds = sub["bedrooms"].median()
    med_baths = sub["bathrooms"].median()
    row = {
        "area_marla": med_area,
        "bedrooms": int(med_beds),
        "bathrooms": int(med_baths),
        "location": loc,
        "property_type": "House",
        "built_year": sub["built_year"].median(),
        "parking_spaces": sub["parking_spaces"].median(),
        "servant_quarters": sub["servant_quarters"].median(),
        "store_rooms": sub["store_rooms"].median(),
        "kitchens": sub["kitchens"].median(),
        "drawing_rooms": sub["drawing_rooms"].median(),
    }
    X = pd.DataFrame([row])[meta["feature_columns"]]
    pred = model.predict(X)[0]
    actual = sub["price"].median()
    print(f"\n{loc[:50]}")
    print(f"  actual median: {actual:,.0f} pred: {pred:,.0f} err: {(pred-actual)/actual*100:+.1f}%")
