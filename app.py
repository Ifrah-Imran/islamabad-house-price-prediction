"""
Streamlit UI 
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from predict_utils import calibrated_predict, defaults_for_location
from utils import format_price_pkr

BEST_MODEL_PATH = Path("models/best_model.joblib")
META_PATH = Path("models/training_meta.json")
RESULTS_PATH = Path("results/model_comparison.json")
INSIGHTS_PATH = Path("results/insights.json")
DATA_PATH = Path("data/islamabad_properties.csv")

st.set_page_config(
    page_title="Islamabad House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(120deg, #0f4c75 0%, #1b6ca8 50%, #3282b8 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white !important; margin: 0; font-size: 1.75rem; }
    .main-header p { color: #e8f4fc; margin: 0.35rem 0 0 0; font-size: 0.95rem; }
    .price-card {
        background: linear-gradient(135deg, #1a5f4a 0%, #2d8f6f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .price-card .big { font-size: 2rem; font-weight: 700; }
    .price-card .sub { font-size: 1rem; opacity: 0.9; }
    .insight-box {
        background: #f0f7ff;
        border-left: 4px solid #1b6ca8;
        padding: 0.85rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PRESETS = {
    "5 Marla — Faisal Hills (Canary)": {
        "area_marla": 5.0,
        "bedrooms": 4,
        "bathrooms": 4,
        "location": "Canary Residency, Faisal Hills",
        "parking_spaces": 2,
        "servant_quarters": 1,
        "store_rooms": 1,
        "kitchens": 2,
        "drawing_rooms": 1,
    },
    "5 Marla — Faisal Town Block C": {
        "area_marla": 5.0,
        "bedrooms": 3,
        "bathrooms": 2,
        "location": "Faisal Town Phase 1 - Block C, Faisal Town Phase 1",
        "parking_spaces": 2,
        "servant_quarters": 1,
        "store_rooms": 1,
        "kitchens": 1,
        "drawing_rooms": 1,
    },
    "10 Marla — DHA Phase 2": {
        "area_marla": 10.0,
        "bedrooms": 5,
        "bathrooms": 4,
        "location": "DHA Defence Phase 2, DHA Defence",
        "parking_spaces": 4,
        "servant_quarters": 2,
        "store_rooms": 2,
        "kitchens": 2,
        "drawing_rooms": 1,
    },
    "1 Kanal — F-7": {
        "area_marla": 20.0,
        "bedrooms": 6,
        "bathrooms": 5,
        "location": "F-7, Islamabad",
        "parking_spaces": 6,
        "servant_quarters": 2,
        "store_rooms": 3,
        "kitchens": 2,
        "drawing_rooms": 2,
    },
}


@st.cache_resource
def load_artifacts():
    if not BEST_MODEL_PATH.exists() or not META_PATH.exists():
        return None, None, None, None
    model = joblib.load(BEST_MODEL_PATH)
    with META_PATH.open(encoding="utf-8") as f:
        meta = json.load(f)
    results = None
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open(encoding="utf-8") as f:
            results = pd.DataFrame(json.load(f))
    insights = None
    if INSIGHTS_PATH.exists():
        with INSIGHTS_PATH.open(encoding="utf-8") as f:
            insights = json.load(f)
    return model, meta, results, insights


@st.cache_data
def load_market_stats():
    if not DATA_PATH.exists():
        return None, None
    df = pd.read_csv(DATA_PATH)
    df = df[df["price"] > 0]
    by_loc = df.groupby("location")["price"].agg(["median", "count", "mean"]).reset_index()
    by_loc.columns = ["location", "median_price", "listings", "mean_price"]
    return df, by_loc


def apply_preset(name: str) -> None:
    if name and name in PRESETS:
        st.session_state.update(PRESETS[name])
        st.session_state["preset_applied"] = name


def init_session_defaults(meta: dict) -> None:
    if "form_initialized" not in st.session_state:
        st.session_state["form_initialized"] = True
        locs = meta.get("locations", ["Islamabad"])
        loc = locs[0]
        loc_defaults = defaults_for_location(meta, loc)
        st.session_state["location"] = loc
        for k, v in loc_defaults.items():
            st.session_state[k] = v


def location_reference(by_loc: pd.DataFrame, location: str) -> dict | None:
    if by_loc is None:
        return None
    match = by_loc[by_loc["location"] == location]
    if match.empty:
        return None
    r = match.iloc[0]
    return {"median": r["median_price"], "mean": r["mean_price"], "count": int(r["listings"])}


model, meta, results_df, insights = load_artifacts()

if model is None:
    st.error("Models not trained yet. Run `python train_models.py` then refresh.")
    st.stop()

init_session_defaults(meta)
df_market, by_loc = load_market_stats()

with st.sidebar:
    st.markdown("### Settings")
    st.caption(f"**Model:** {meta.get('best_model', 'Gradient Boosting')}")
    st.caption(f"**Training samples:** {meta.get('train_size', '-')}")
    st.divider()
    st.markdown("**Quick presets**")
    preset_choice = st.selectbox(
        "Load example property",
        ["— Custom —"] + list(PRESETS.keys()),
        label_visibility="collapsed",
    )
    if st.button("Apply preset", use_container_width=True):
        apply_preset(preset_choice)
        st.rerun()
    if st.session_state.get("preset_applied"):
        st.success(f"Loaded: {st.session_state['preset_applied']}")
    st.divider()
    st.caption("AIC354 · COMSATS Islamabad · Spring 2026")

st.markdown(
    """
    <div class="main-header">
        <h1>Islamabad House Price Predictor</h1>
        <p>Market-calibrated estimates · Zameen.com Islamabad trends + 400 scraped listings</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_compare, tab_about = st.tabs(
    ["Predict Price", "Model Comparison", "About Project"]
)

with tab_predict:
    col_form, col_result = st.columns([1.35, 1], gap="large")

    with col_form:
        with st.form("prediction_form", border=True):
            st.markdown("##### Property basics")
            c1, c2 = st.columns(2)
            with c1:
                area = st.number_input(
                    "Area (Marla)",
                    min_value=0.5,
                    max_value=500.0,
                    value=float(st.session_state.get("area_marla", 10.0)),
                    step=0.5,
                    help="1 Kanal = 20 Marla",
                )
            with c2:
                prop_type = st.selectbox(
                    "Property type",
                    meta.get("property_types", ["House"]),
                    index=0,
                )

            loc_search = st.text_input(
                "Search location",
                placeholder="e.g. Faisal Hills, DHA, F-7, Bahria...",
            )
            locations = meta.get("locations", [])
            filtered = (
                [l for l in locations if loc_search.lower() in l.lower()]
                if loc_search.strip()
                else locations
            )
            if not filtered:
                filtered = locations
            default_loc = st.session_state.get("location", filtered[0] if filtered else "")
            loc_index = filtered.index(default_loc) if default_loc in filtered else 0
            location = st.selectbox("Location", options=filtered, index=loc_index)

            prof = meta.get("location_profiles", {}).get(location)
            if prof:
                st.caption(
                    f"Training data: **{prof['count']}** listings here · "
                    f"typical **{prof['median_price_per_marla']/1e6:.2f}M PKR/marla**"
                )
            else:
                st.warning("Few or no training listings for this location — estimate may be less accurate.")

            st.markdown("##### Rooms & layout")
            loc_defaults = defaults_for_location(meta, location)
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                bedrooms = st.number_input(
                    "Bedrooms", 0, 25, int(st.session_state.get("bedrooms", loc_defaults.get("bedrooms", 4)))
                )
            with r2:
                bathrooms = st.number_input(
                    "Bathrooms", 0, 25, int(st.session_state.get("bathrooms", loc_defaults.get("bathrooms", 3)))
                )
            with r3:
                kitchens = st.number_input(
                    "Kitchens", 0, 10, int(st.session_state.get("kitchens", loc_defaults.get("kitchens", 2)))
                )
            with r4:
                drawing_rooms = st.number_input(
                    "Drawing rooms",
                    0,
                    10,
                    int(st.session_state.get("drawing_rooms", loc_defaults.get("drawing_rooms", 1))),
                )

            st.markdown("##### Amenities")
            a1, a2, a3 = st.columns(3)
            with a1:
                parking = st.number_input(
                    "Parking spaces",
                    0,
                    20,
                    int(st.session_state.get("parking_spaces", loc_defaults.get("parking_spaces", 2))),
                )
            with a2:
                servant = st.number_input(
                    "Servant quarters",
                    0,
                    10,
                    int(st.session_state.get("servant_quarters", loc_defaults.get("servant_quarters", 1))),
                )
            with a3:
                store = st.number_input(
                    "Store rooms",
                    0,
                    10,
                    int(st.session_state.get("store_rooms", loc_defaults.get("store_rooms", 1))),
                )

            submitted = st.form_submit_button("Estimate price", type="primary", use_container_width=True)

        if submitted:
            state = {
                "area_marla": area,
                "bedrooms": int(bedrooms),
                "bathrooms": int(bathrooms),
                "location": location,
                "property_type": prop_type,
                "parking_spaces": int(parking),
                "servant_quarters": int(servant),
                "store_rooms": int(store),
                "kitchens": int(kitchens),
                "drawing_rooms": int(drawing_rooms),
            }
            price, dbg = calibrated_predict(model, meta, state)
            st.session_state["last_prediction"] = price
            st.session_state["last_debug"] = dbg
            st.session_state["last_state"] = state

    with col_result:
        st.markdown("##### Estimate")
        if "last_prediction" in st.session_state:
            price = st.session_state["last_prediction"]
            state = st.session_state.get("last_state", {})
            dbg = st.session_state.get("last_debug", {})

            st.markdown(
                f"""
                <div class="price-card">
                    <div class="sub">Estimated market value</div>
                    <div class="big">{format_price_pkr(price)}</div>
                    <div class="sub">PKR {price:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if dbg.get("calibrated"):
                st.caption(
                    f"Market tier: **{dbg.get('tier', 'mid')}** · Area band: **{dbg.get('area_band', '')}** · "
                    f"Blend **{dbg['blend_weight']*100:.0f}%** local comps / **{(1-dbg['blend_weight'])*100:.0f}%** ML"
                )
                with st.expander("How this estimate was built"):
                    st.write(f"ML model: {format_price_pkr(dbg['ml_price'])}")
                    st.write(f"Local comps (location + size): {format_price_pkr(dbg['baseline_price'])}")
                    if dbg.get("market_baseline"):
                        st.write(f"Islamabad tier benchmark: {format_price_pkr(dbg['market_baseline'])}")
                    st.caption("Aligned with Zameen.com Islamabad price trends and scraped listings.")

            m1, m2, m3 = st.columns(3)
            m1.metric("Area", f"{state.get('area_marla', 0)} Marla")
            m2.metric("Beds / Baths", f"{state.get('bedrooms', 0)} / {state.get('bathrooms', 0)}")
            m3.metric("Kitchens", state.get("kitchens", "-"))

            ref = location_reference(by_loc, state.get("location", ""))
            if ref:
                st.markdown("##### Scraped market (same location)")
                diff = price - ref["median"]
                pct = (diff / ref["median"]) * 100 if ref["median"] else 0
                st.metric(
                    "Median listing in dataset",
                    format_price_pkr(ref["median"]),
                    delta=f"{pct:+.1f}% vs estimate",
                    delta_color="inverse",
                )
                st.caption(f"From {ref['count']} Zameen listings (not live market).")

            with st.expander("All inputs used"):
                st.json(state)
        else:
            st.info("Fill in the form and click **Estimate price**.")

        if df_market is not None and "last_state" in st.session_state:
            loc_df = df_market[df_market["location"] == st.session_state["last_state"]["location"]]
            if not loc_df.empty:
                st.markdown("##### Listings in this location (dataset)")
                st.scatter_chart(
                    loc_df[["area_marla", "price"]],
                    x="area_marla",
                    y="price",
                    height=200,
                )

with tab_compare:
    st.subheader("Model performance & insights")

    if insights:
        interp = insights.get("interpretation", {})
        cal = insights.get("calibrated_metrics", {})
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Best model", insights.get("best_model", "-"))
        i2.metric("Test R² (calibrated)", f"{cal.get('R2', 0):.3f}")
        i3.metric("Typical error (MAPE)", f"{interp.get('typical_error_pct', '-')}%")
        i4.metric("Within ±20% of actual", interp.get("predictions_within_20pct", "-"))

        st.markdown(
            f'<div class="insight-box">'
            f"<b>What this means:</b> On held-out test listings, the calibrated system is wrong by about "
            f"<b>{interp.get('typical_error_pct', '?')}%</b> on average (MAPE). "
            f"RMSE is about <b>{interp.get('rmse_crore', '?')} crore</b> PKR. "
            f"Sparse areas (e.g. Faisal Hills) rely more on local PKR/marla from scraped comps."
            f"</div>",
            unsafe_allow_html=True,
        )

        mref = insights.get("market_reference", {})
        if mref:
            st.markdown("##### Islamabad market reference (Zameen.com)")
            st.markdown(
                f"- **5 Marla (mid):** {mref.get('5_marla_mid_typical', '')}  \n"
                f"- **10 Marla (DHA):** {mref.get('10_marla_dha_typical', '')}  \n"
                f"- **1 Kanal (F-7):** {mref.get('1_kanal_f7_typical', '')}  \n"
                f"- *Source:* {mref.get('source', 'Zameen.com')}"
            )

        hardest = insights.get("hardest_locations", [])
        if hardest:
            st.markdown("##### Hardest locations on test set (highest % error)")
            st.dataframe(pd.DataFrame(hardest), use_container_width=True, hide_index=True)

        removed = insights.get("features_removed", [])
        if removed:
            st.caption("Removed feature: " + "; ".join(removed))

    if results_df is not None and not results_df.empty:
        st.divider()
        ml_only = results_df[~results_df["model"].str.contains("Calibrated", na=False)]
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("##### All models (test set)")
            show_cols = ["model", "MAE", "RMSE", "R2", "MAPE_pct", "within_20pct"]
            show_cols = [c for c in show_cols if c in results_df.columns]
            st.dataframe(
                results_df[show_cols].sort_values("RMSE"),
                use_container_width=True,
                hide_index=True,
            )
        with c2:
            chart_df = ml_only.set_index("model")[["R2"]].sort_values("R2")
            st.bar_chart(chart_df, height=280)
            st.caption("R² — higher is better (raw ML, before location blend)")
        cal_row = results_df[results_df["model"].str.contains("Calibrated", na=False)]
        if not cal_row.empty:
            st.success(
                "The app uses **location-calibrated** predictions (row above), which improves "
                "accuracy in areas like Faisal Hills with few listings."
            )
    else:
        st.warning("Run `python train_models.py` to generate metrics.")

with tab_about:
    st.markdown(
        """
        ### House Price Prediction System

        | Item | Detail |
        |------|--------|
        | **Course** | AIC354 — Machine Learning Fundamentals Lab |
        | **Data** | 400 Islamabad listings from [Zameen.com](https://www.zameen.com) |
        | **Best model** | Gradient Boosting (log-price target) + location calibration |

        **Features:** area, bedrooms, bathrooms, location, property type, parking,
        servant quarters, store rooms, kitchens, drawing rooms.

        **Not used:** built year (too many missing/wrong values from scraping).

        **Pipeline:** `scraper.py` → `train_models.py` → `streamlit run app.py`
        """
    )