"""
Preprocess data, train regression models, evaluate, save best pipeline.

"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from market_rates import MARKET_PPM, area_band, infer_tier
from predict_utils import build_location_profiles, calibrated_predict

DATA_PATH = Path("data/islamabad_properties.csv")
MODELS_DIR = Path("models")
RESULTS_PATH = Path("results/model_comparison.json")
INSIGHTS_PATH = Path("results/insights.json")
BEST_MODEL_PATH = Path("models/best_model.joblib")
META_PATH = Path("models/training_meta.json")

TOP_LOCATIONS = 40
PRICE_CAP_PPM = 18_000_000  # drop extreme luxury outliers from training

FEATURE_COLS = [
    "area_marla",
    "bedrooms",
    "bathrooms",
    "location_model",
    "property_type",
    "area_band",
    "total_rooms",
    "parking_spaces",
    "servant_quarters",
    "store_rooms",
    "kitchens",
    "drawing_rooms",
]
TARGET = "price"


def load_and_preprocess(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df[df[TARGET] > 0]
    df = df[df["area_marla"] > 0]

    numeric_cols = [
        "area_marla",
        "bedrooms",
        "bathrooms",
        "parking_spaces",
        "servant_quarters",
        "store_rooms",
        "kitchens",
        "drawing_rooms",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in numeric_cols:
        if col == "area_marla":
            continue
        loc_median = df.groupby("location")[col].transform("median")
        df[col] = df[col].fillna(loc_median)
        df[col] = df[col].fillna(df[col].median())

    df["location"] = df["location"].fillna("Unknown").astype(str).str.strip()
    df["property_type"] = df["property_type"].fillna("House").astype(str).str.strip()

    # Remove extreme outliers that skew the model (ultra-luxury one-offs)
    df["ppm"] = df["price"] / df["area_marla"]
    is_ultra_loc = df["location"].str.contains(r"F-[678]|E-7", case=False, na=False, regex=True)
    cap_mask = (df["ppm"] <= PRICE_CAP_PPM) | is_ultra_loc
    df = df[cap_mask].copy()

    top_locs = df["location"].value_counts().head(TOP_LOCATIONS).index.tolist()
    df["location_model"] = df["location"].where(df["location"].isin(top_locs), "Other Islamabad")
    df["area_band"] = df["area_marla"].apply(area_band)
    df["total_rooms"] = (
        df["bedrooms"] + df["bathrooms"] + df["kitchens"] + df["drawing_rooms"]
    ).astype(float)

    df = df.dropna(subset=["area_marla", "bedrooms", "bathrooms", "location"])
    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "area_marla",
        "bedrooms",
        "bathrooms",
        "total_rooms",
        "parking_spaces",
        "servant_quarters",
        "store_rooms",
        "kitchens",
        "drawing_rooms",
    ]
    categorical_features = ["location_model", "property_type", "area_band"]

    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def evaluate_model(name: str, y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    within_20 = float(np.mean(np.abs(y_true - y_pred) / y_true <= 0.20) * 100)
    return {
        "model": name,
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE_pct": round(mape, 2),
        "within_20pct": round(within_20, 2),
    }


def get_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=4, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.85, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42,
            objective="reg:squarederror",
        ),
        "CatBoost": CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.04,
            l2_leaf_reg=5,
            verbose=0,
            random_state=42,
        ),
    }


def make_regressor(estimator, preprocessor) -> TransformedTargetRegressor:
    base = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    return TransformedTargetRegressor(
        regressor=base,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def train_and_evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, object, dict]:
    X = df[FEATURE_COLS]
    y = df[TARGET]
    full_locations = df["location"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    loc_train = full_locations.loc[X_train.index]
    loc_test = full_locations.loc[X_test.index]
    train_df = df.loc[X_train.index]

    location_profiles = build_location_profiles(train_df)
    location_map = {loc: loc for loc in df["location"].unique()}
    for loc in df["location"].unique():
        if loc not in train_df["location"].values:
            pass
    top = df["location"].value_counts().head(TOP_LOCATIONS).index
    for loc in df["location"].unique():
        location_map[loc] = loc if loc in top else "Other Islamabad"

    preprocessor = build_preprocessor()
    results = []
    best_model = None
    best_cal_mape = float("inf")
    best_name = ""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    imputation_defaults = {"property_type": str(df["property_type"].mode().iloc[0])}
    for col in [
        "area_marla",
        "bedrooms",
        "bathrooms",
        "parking_spaces",
        "servant_quarters",
        "store_rooms",
        "kitchens",
        "drawing_rooms",
    ]:
        imputation_defaults[col] = float(df[col].median())

    meta = {
        "best_model": "",
        "feature_columns": FEATURE_COLS,
        "ui_columns": [
            "area_marla",
            "bedrooms",
            "bathrooms",
            "location",
            "property_type",
            "parking_spaces",
            "servant_quarters",
            "store_rooms",
            "kitchens",
            "drawing_rooms",
        ],
        "imputation_defaults": imputation_defaults,
        "location_profiles": location_profiles,
        "location_map": location_map,
        "market_rates": MARKET_PPM,
        "locations": sorted(df["location"].unique().tolist()),
        "property_types": sorted(df["property_type"].unique().tolist()),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    for name, estimator in get_models().items():
        reg = make_regressor(estimator, preprocessor)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        metrics = evaluate_model(name, y_test, preds)
        results.append(metrics)
        print(
            f"{name:20s} | MAE: {metrics['MAE']:,.0f} | RMSE: {metrics['RMSE']:,.0f} | "
            f"R²: {metrics['R2']:.4f} | MAPE: {metrics['MAPE_pct']:.1f}%"
        )
        joblib.dump(reg, MODELS_DIR / f"{name.replace(' ', '_').lower()}.joblib")

        # Pick model by calibrated test MAPE
        cal_preds = []
        for idx, row in X_test.iterrows():
            ui = {
                "area_marla": row["area_marla"],
                "bedrooms": int(row["bedrooms"]),
                "bathrooms": int(row["bathrooms"]),
                "location": loc_test.loc[idx],
                "property_type": row["property_type"],
                "parking_spaces": int(row["parking_spaces"]),
                "servant_quarters": int(row["servant_quarters"]),
                "store_rooms": int(row["store_rooms"]),
                "kitchens": int(row["kitchens"]),
                "drawing_rooms": int(row["drawing_rooms"]),
            }
            p, _ = calibrated_predict(reg, meta, ui)
            cal_preds.append(p)
        cal_mape = float(np.mean(np.abs((y_test - cal_preds) / y_test)) * 100)
        if cal_mape < best_cal_mape:
            best_cal_mape = cal_mape
            best_model = reg
            best_name = name

    meta["best_model"] = best_name

    cal_preds_final = []
    for idx, row in X_test.iterrows():
        ui = {
            "area_marla": row["area_marla"],
            "bedrooms": int(row["bedrooms"]),
            "bathrooms": int(row["bathrooms"]),
            "location": loc_test.loc[idx],
            "property_type": row["property_type"],
            "parking_spaces": int(row["parking_spaces"]),
            "servant_quarters": int(row["servant_quarters"]),
            "store_rooms": int(row["store_rooms"]),
            "kitchens": int(row["kitchens"]),
            "drawing_rooms": int(row["drawing_rooms"]),
        }
        p, _ = calibrated_predict(best_model, meta, ui)
        cal_preds_final.append(p)
    cal_metrics = evaluate_model("Calibrated (market-aware)", y_test, np.array(cal_preds_final))
    results.append(cal_metrics)
    print(
        f"\nCalibrated ({best_name}) | MAE: {cal_metrics['MAE']:,.0f} | RMSE: {cal_metrics['RMSE']:,.0f} | "
        f"R²: {cal_metrics['R2']:.4f} | MAPE: {cal_metrics['MAPE_pct']:.1f}%"
    )

    loc_errors = []
    for loc in loc_test.unique():
        mask = loc_test == loc
        if mask.sum() == 0:
            continue
        sub_y = y_test[mask]
        preds = []
        for idx in sub_y.index:
            row = X_test.loc[idx]
            ui = {
                "area_marla": row["area_marla"],
                "bedrooms": int(row["bedrooms"]),
                "bathrooms": int(row["bathrooms"]),
                "location": loc_test.loc[idx],
                "property_type": row["property_type"],
                "parking_spaces": int(row["parking_spaces"]),
                "servant_quarters": int(row["servant_quarters"]),
                "store_rooms": int(row["store_rooms"]),
                "kitchens": int(row["kitchens"]),
                "drawing_rooms": int(row["drawing_rooms"]),
            }
            p, _ = calibrated_predict(best_model, meta, ui)
            preds.append(p)
        mape = np.mean(np.abs(sub_y - preds) / sub_y) * 100
        loc_errors.append({"location": loc, "mape_pct": round(mape, 1), "test_count": int(mask.sum())})
    loc_errors.sort(key=lambda x: x["mape_pct"], reverse=True)

    insights = {
        "best_model": best_name,
        "calibrated_metrics": cal_metrics,
        "interpretation": {
            "rmse_crore": round(cal_metrics["RMSE"] / 10_000_000, 2),
            "typical_error_pct": cal_metrics["MAPE_pct"],
            "predictions_within_20pct": f"{cal_metrics['within_20pct']}% of test listings",
        },
        "market_reference": {
            "5_marla_mid_typical": "PKR 2.5 - 3.2 Crore (Faisal Hills / mid societies)",
            "10_marla_dha_typical": "PKR 6 - 9 Crore",
            "1_kanal_f7_typical": "PKR 30 - 50+ Crore",
            "source": "Zameen.com Islamabad trends + scraped dataset",
        },
        "hardest_locations": loc_errors[:8],
        "features_used": FEATURE_COLS,
    }

    joblib.dump(best_model, BEST_MODEL_PATH)
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with INSIGHTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)

    print(f"\nBest model (by calibrated MAPE): {best_name}")
    return pd.DataFrame(results).sort_values("RMSE"), best_model, meta


if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run: python scraper.py")
    dataframe = load_and_preprocess(DATA_PATH)
    print(f"Records for training (after outlier filter): {len(dataframe)}")
    comparison, _, _ = train_and_evaluate(dataframe)
    print("\nModel comparison:")
    print(comparison.to_string(index=False))
