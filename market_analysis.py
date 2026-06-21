import pandas as pd
import numpy as np

df = pd.read_csv("data/islamabad_properties.csv")
df = df[(df.price > 0) & (df.area_marla > 0)]
df["ppm"] = df.price / df.area_marla

print("=== OVERALL ===")
print(f"rows {len(df)}")
print(f"price median {df.price.median():,.0f} mean {df.price.mean():,.0f}")
print(f"ppm median {df.ppm.median():,.0f} mean {df.ppm.mean():,.0f}")

print("\n=== BY AREA BAND ===")
bins = [0, 3, 6, 9, 12, 20, 100]
labels = ["1-3", "3-6", "6-9", "9-12", "12-20", "20+"]
df["band"] = pd.cut(df.area_marla, bins=bins, labels=labels)
print(df.groupby("band", observed=True).agg(
    n=("price", "count"),
    med_price=("price", "median"),
    med_ppm=("ppm", "median"),
))

print("\n=== TOP LOCATIONS BY LISTINGS ===")
loc = df.groupby("location").agg(
    n=("price", "count"),
    med_price=("price", "median"),
    med_ppm=("ppm", "median"),
    med_area=("area_marla", "median"),
).sort_values("n", ascending=False)
print(loc.head(20).to_string())

print("\n=== FAISAL / DHA / F-7 ===")
for kw in ["Faisal", "DHA", "F-7", "Bahria", "B-17", "G-13"]:
    sub = df[df.location.str.contains(kw, case=False, na=False)]
    if len(sub):
        print(f"\n{kw}: n={len(sub)} price_med={sub.price.median():,.0f} ppm_med={sub.ppm.median():,.0f}")
