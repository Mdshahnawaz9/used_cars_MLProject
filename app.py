import warnings
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "used_car_Knn_model.pkl"
DATA_PATH = BASE_DIR / "used_cars_cleaned.csv"

CAT_COLS = ["brand", "fuel_type", "engine", "transmission", "ext_col", "int_col", "accident"]
NUM_COLS = ["model_year", "milage"]  # scaled with the saved MinMaxScaler

# Colours / copy for each prediction class
CATEGORY_STYLE = {
    "Budget": {
        "emoji": "💸",
        "color": "#22c55e",
        "grad": "linear-gradient(135deg,#052e1a 0%,#0f5132 100%)",
        "text": "Easy on the wallet. Typically an older and/or higher-mileage car.",
    },
    "Mid-Range": {
        "emoji": "🚗",
        "color": "#f59e0b",
        "grad": "linear-gradient(135deg,#3a2606 0%,#6b4410 100%)",
        "text": "A balanced pick. Good mix of age, mileage and brand value.",
    },
    "Premium": {
        "emoji": "💎",
        "color": "#a78bfa",
        "grad": "linear-gradient(135deg,#1e1442 0%,#4c2a9a 100%)",
        "text": "Top tier. Newer, lower-mileage or high-value brand/engine.",
    },
}

# --------------------------------------------------------------------------
# Page config + CSS
# --------------------------------------------------------------------------
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚘", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Poppins', sans-serif; }

.stApp {
    background:
        radial-gradient(1100px 500px at 10% -10%, rgba(124,92,255,.22), transparent 60%),
        radial-gradient(900px 450px at 100% 0%, rgba(34,211,238,.14), transparent 55%),
        #0b1020;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

/* ---------- hero ---------- */
.hero {
    position: relative; overflow: hidden;
    padding: 2.2rem 2.4rem; border-radius: 22px; margin-bottom: 1.4rem;
    background: linear-gradient(120deg, #7c5cff 0%, #3b82f6 55%, #22d3ee 100%);
    box-shadow: 0 18px 50px rgba(59,130,246,.28);
}
.hero::after {
    content: "🚘"; position: absolute; right: 2.2rem; top: 50%;
    transform: translateY(-50%); font-size: 6.5rem; opacity: .22;
}
.hero h1 { margin: 0; font-size: 2.3rem; font-weight: 800; color: #fff; letter-spacing: -.5px; }
.hero p  { margin: .5rem 0 0; color: rgba(255,255,255,.9); font-size: 1.02rem; max-width: 640px; }
.chips   { margin-top: 1rem; display: flex; gap: .5rem; flex-wrap: wrap; }
.chip {
    background: rgba(255,255,255,.18); backdrop-filter: blur(6px);
    color: #fff; padding: .28rem .8rem; border-radius: 999px;
    font-size: .78rem; font-weight: 500; border: 1px solid rgba(255,255,255,.25);
}

/* ---------- section titles / cards ---------- */
.section-title { font-size: 1.05rem; font-weight: 600; color: #cfd6f6; margin: .2rem 0 .6rem; }
.card {
    background: rgba(20,27,52,.75); border: 1px solid rgba(124,140,255,.18);
    border-radius: 18px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    box-shadow: 0 8px 28px rgba(0,0,0,.25);
}
.card h4 { margin: 0 0 .6rem; color: #fff; font-weight: 600; font-size: 1rem; }

/* ---------- form ---------- */
div[data-testid="stForm"] {
    background: rgba(20,27,52,.75); border: 1px solid rgba(124,140,255,.18);
    border-radius: 18px; padding: 1.4rem 1.5rem 1.2rem;
    box-shadow: 0 8px 28px rgba(0,0,0,.25);
}
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    font-weight: 500 !important; color: #b9c2ea !important; font-size: .86rem !important;
}
div[data-baseweb="select"] > div, .stNumberInput input {
    background: #0f1630 !important; border-radius: 12px !important;
    border: 1px solid rgba(124,140,255,.25) !important;
}

/* ---------- buttons ---------- */
.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    border-radius: 12px; font-weight: 600; padding: .6rem 1.2rem;
    border: 1px solid rgba(124,140,255,.35); transition: all .2s ease;
}
div[data-testid="stFormSubmitButton"] > button {
    width: 100%; color: #fff; border: none; font-size: 1.05rem; padding: .8rem 1rem;
    background: linear-gradient(90deg, #7c5cff, #3b82f6);
    box-shadow: 0 10px 26px rgba(124,92,255,.38);
}
div[data-testid="stFormSubmitButton"] > button:hover { transform: translateY(-2px); filter: brightness(1.1); }
.stButton > button:hover { border-color: #7c5cff; color: #fff; transform: translateY(-1px); }

/* ---------- result ---------- */
.result {
    border-radius: 22px; padding: 1.8rem 1.6rem; text-align: center; color: #fff;
    border: 1px solid rgba(255,255,255,.14); animation: pop .5s ease both;
    box-shadow: 0 18px 46px rgba(0,0,0,.4);
}
.result .emoji { font-size: 3.2rem; line-height: 1; }
.result .label { font-size: .8rem; letter-spacing: .18em; text-transform: uppercase; opacity: .75; margin-top: .7rem; }
.result .value { font-size: 2.4rem; font-weight: 800; margin: .1rem 0 .4rem; }
.result .desc  { font-size: .92rem; opacity: .88; }
@keyframes pop { from { opacity: 0; transform: translateY(14px) scale(.97); } to { opacity: 1; transform: none; } }

.kpi {
    background: rgba(20,27,52,.8); border: 1px solid rgba(124,140,255,.18);
    border-radius: 16px; padding: 1rem 1.1rem; text-align: center;
}
.kpi .k { font-size: .75rem; color: #98a3d6; text-transform: uppercase; letter-spacing: .1em; }
.kpi .v { font-size: 1.55rem; font-weight: 700; color: #fff; margin-top: .15rem; }

/* probability bars */
.prob-row { display: flex; align-items: center; gap: .8rem; margin: .55rem 0; }
.prob-name { width: 90px; font-size: .86rem; color: #d5dbf7; font-weight: 500; }
.prob-track { flex: 1; height: 12px; background: #0f1630; border-radius: 999px; overflow: hidden; }
.prob-fill  { height: 100%; border-radius: 999px; }
.prob-pct   { width: 48px; text-align: right; font-size: .86rem; color: #fff; font-weight: 600; }

/* tabs */
button[data-baseweb="tab"] { font-weight: 600; font-size: .95rem; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#10173a 0%,#0b1020 100%);
    border-right: 1px solid rgba(124,140,255,.14);
}
.side-stat { display:flex; justify-content:space-between; padding:.45rem 0;
    border-bottom: 1px dashed rgba(124,140,255,.16); font-size:.88rem; color:#c5cdf0; }
.side-stat b { color:#fff; }

.footer { text-align:center; color:#7f89b8; font-size:.8rem; margin-top:2rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    art = joblib.load(MODEL_PATH)  # saved with joblib
    return art["model"], art["OHE"], art["scaler"], art["LE"]


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def preprocess(df_raw: pd.DataFrame, model, ohe, scaler) -> pd.DataFrame:
    """Raw columns -> the exact 77-column frame the model expects."""
    enc = pd.DataFrame(
        ohe.transform(df_raw[CAT_COLS]),
        columns=ohe.get_feature_names_out(CAT_COLS),
        index=df_raw.index,
    )
    num = pd.DataFrame(
        scaler.transform(df_raw[NUM_COLS]), columns=NUM_COLS, index=df_raw.index
    )
    return pd.concat([enc, num], axis=1)[list(model.feature_names_in_)]


@st.cache_data
def transform_dataset(_model, _ohe, _scaler, df: pd.DataFrame) -> pd.DataFrame:
    return preprocess(df, _model, _ohe, _scaler)


# --------------------------------------------------------------------------
# Guard rails
# --------------------------------------------------------------------------
missing = [p.name for p in (MODEL_PATH, DATA_PATH) if not p.exists()]
if missing:
    st.error(f"Missing file(s): {', '.join(missing)}. Place them in the same folder as app.py.")
    st.stop()

model, ohe, scaler, le = load_artifacts()
df = load_data()

# class code -> label  (LabelEncoder classes_ = ['Budget', 'Mid-Range', 'Premium'])
DECODE = {int(c): str(n).strip() for c, n in zip(le.transform(le.classes_), le.classes_)}

CATS = dict(zip(CAT_COLS, ohe.categories_))
YEAR_MIN, YEAR_MAX = int(df["model_year"].min()), int(df["model_year"].max())
KM_MAX = int(df["milage"].max())

# --------------------------------------------------------------------------
# Session-state defaults + "random car" helper
# --------------------------------------------------------------------------
DEFAULTS = {
    "brand": "Toyota", "fuel_type": "Gasoline", "engine": "Other",
    "transmission": "Automatic", "ext_col": "Black", "int_col": "Black",
    "accident": "No Accident", "model_year": 2018, "milage": 45_000,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def fill_random_car():
    row = df.sample(1).iloc[0]
    for k in DEFAULTS:
        val = row[k]
        st.session_state[k] = int(val) if k in NUM_COLS else str(val)
    st.session_state["milage"] = min(int(row["milage"]), 500_000)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚘 About the model")
    st.markdown(
        f"""
        <div class="side-stat"><span>Algorithm</span><b>K-Nearest Neighbors</b></div>
        <div class="side-stat"><span>Neighbors (k)</span><b>{model.n_neighbors}</b></div>
        <div class="side-stat"><span>Input features</span><b>{model.n_features_in_}</b></div>
        <div class="side-stat"><span>Classes</span><b>{len(model.classes_)}</b></div>
        <div class="side-stat"><span>Dataset rows</span><b>{len(df):,}</b></div>
        <div class="side-stat"><span>Years covered</span><b>{YEAR_MIN}–{YEAR_MAX}</b></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 🎯 How it works")
    st.caption(
        "Your car is encoded and scaled exactly like the training data. "
        "The model then looks at the closest cars in the dataset and "
        "votes for a price category."
    )
    st.markdown("### 🎲 Quick start")
    st.button("Load a random car", on_click=fill_random_car, use_container_width=True)

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Used Car Price Predictor</h1>
        <p>Describe a used car and get an instant price-category prediction —
           <b>Budget</b>, <b>Mid-Range</b> or <b>Premium</b> — powered by a KNN model.</p>
        <div class="chips">
            <span class="chip">⚡ Instant prediction</span>
            <span class="chip">📊 Confidence breakdown</span>
            <span class="chip">🔍 Similar cars</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_insights, tab_data = st.tabs(["🔮 Predict", "📊 Market insights", "🗂️ Dataset"])

# ==========================================================================
# TAB 1 - PREDICT
# ==========================================================================
with tab_predict:
    with st.form("car_form"):
        st.markdown('<div class="section-title">🚗 Vehicle details</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            brand = st.selectbox("Brand", CATS["brand"], key="brand")
            fuel_type = st.selectbox("Fuel type", CATS["fuel_type"], key="fuel_type")
            transmission = st.selectbox("Transmission", CATS["transmission"], key="transmission")
        with c2:
            model_year = st.slider("Model year", YEAR_MIN, YEAR_MAX, key="model_year")
            milage = st.number_input(
                "Mileage (miles)", min_value=0, max_value=500_000, step=1_000, key="milage"
            )
            accident = st.selectbox("Accident history", CATS["accident"], key="accident")
        with c3:
            engine = st.selectbox("Engine", CATS["engine"], key="engine")
            ext_col = st.selectbox("Exterior colour", CATS["ext_col"], key="ext_col")
            int_col = st.selectbox("Interior colour", CATS["int_col"], key="int_col")

        submitted = st.form_submit_button("🔮 Predict price category")

    if submitted:
        raw = pd.DataFrame([{
            "brand": brand, "fuel_type": fuel_type, "engine": engine,
            "transmission": transmission, "ext_col": ext_col, "int_col": int_col,
            "accident": accident, "model_year": model_year, "milage": milage,
        }])
        X = preprocess(raw, model, ohe, scaler)

        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        label = DECODE[pred]
        style = CATEGORY_STYLE[label]
        confidence = proba[list(model.classes_).index(pred)]

        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns([1, 1.15], gap="large")

        with left:
            st.markdown(
                f"""
                <div class="result" style="background:{style['grad']};">
                    <div class="emoji">{style['emoji']}</div>
                    <div class="label">Predicted category</div>
                    <div class="value" style="color:{style['color']};">{label}</div>
                    <div class="desc">{style['text']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            k1, k2 = st.columns(2)
            k1.markdown(
                f'<div class="kpi" style="margin-top:1rem"><div class="k">Confidence</div>'
                f'<div class="v">{confidence:.0%}</div></div>', unsafe_allow_html=True)
            k2.markdown(
                f'<div class="kpi" style="margin-top:1rem"><div class="k">Neighbors voting</div>'
                f'<div class="v">{int(round(confidence * model.n_neighbors))}/{model.n_neighbors}</div></div>',
                unsafe_allow_html=True)

        with right:
            bars = ""
            for code, p in zip(model.classes_, proba):
                name = DECODE[int(code)]
                col = CATEGORY_STYLE[name]["color"]
                bars += (
                    f'<div class="prob-row"><div class="prob-name">{name}</div>'
                    f'<div class="prob-track"><div class="prob-fill" '
                    f'style="width:{p*100:.0f}%;background:{col};"></div></div>'
                    f'<div class="prob-pct">{p:.0%}</div></div>'
                )
            st.markdown(
                f'<div class="card"><h4>📊 Probability by category</h4>{bars}</div>',
                unsafe_allow_html=True,
            )

            # nearest neighbours from the dataset
            X_all = transform_dataset(model, ohe, scaler, df)
            dist, idx = model.kneighbors(X, n_neighbors=model.n_neighbors)
            sim = df.iloc[idx[0]][
                ["brand", "model_year", "milage", "fuel_type", "accident", "price_category"]
            ].copy()
            sim.insert(0, "distance", dist[0].round(3))
            st.markdown('<div class="card"><h4>🔍 Most similar cars in the dataset</h4>', unsafe_allow_html=True)
            st.dataframe(sim.reset_index(drop=True), use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🧪 Processed features sent to the model (non-zero only)"):
            nz = X.T.rename(columns={0: "value"})
            st.dataframe(nz[nz["value"] != 0], use_container_width=True)
    else:
        st.info("Fill in the car details above (or load a random car from the sidebar) and press **Predict**.")

# ==========================================================================
# TAB 2 - INSIGHTS
# ==========================================================================
PALETTE = alt.Scale(
    domain=["Budget", "Mid-Range", "Premium"],
    range=[CATEGORY_STYLE[c]["color"] for c in ["Budget", "Mid-Range", "Premium"]],
)

with tab_insights:
    a, b, c = st.columns(3)
    a.markdown(f'<div class="kpi"><div class="k">Cars analysed</div><div class="v">{len(df):,}</div></div>', unsafe_allow_html=True)
    b.markdown(f'<div class="kpi"><div class="k">Avg. mileage</div><div class="v">{df["milage"].mean():,.0f}</div></div>', unsafe_allow_html=True)
    c.markdown(f'<div class="kpi"><div class="k">Avg. model year</div><div class="v">{df["model_year"].mean():.0f}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown('<div class="section-title">Average mileage by category</div>', unsafe_allow_html=True)
        d = df.groupby("price_category", as_index=False)["milage"].mean()
        st.altair_chart(
            alt.Chart(d).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
                x=alt.X("price_category:N", title=None, sort=["Budget", "Mid-Range", "Premium"]),
                y=alt.Y("milage:Q", title="Avg. mileage"),
                color=alt.Color("price_category:N", scale=PALETTE, legend=None),
                tooltip=["price_category", alt.Tooltip("milage:Q", format=",.0f")],
            ).properties(height=300),
            use_container_width=True,
        )
    with g2:
        st.markdown('<div class="section-title">Average model year by category</div>', unsafe_allow_html=True)
        d = df.groupby("price_category", as_index=False)["model_year"].mean()
        st.altair_chart(
            alt.Chart(d).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
                x=alt.X("price_category:N", title=None, sort=["Budget", "Mid-Range", "Premium"]),
                y=alt.Y("model_year:Q", title="Avg. model year", scale=alt.Scale(zero=False)),
                color=alt.Color("price_category:N", scale=PALETTE, legend=None),
                tooltip=["price_category", alt.Tooltip("model_year:Q", format=".1f")],
            ).properties(height=300),
            use_container_width=True,
        )

    st.markdown('<div class="section-title">Category mix by brand (top 12 brands)</div>', unsafe_allow_html=True)
    top = df["brand"].value_counts().head(12).index
    d = df[df["brand"].isin(top)].groupby(["brand", "price_category"], as_index=False).size()
    st.altair_chart(
        alt.Chart(d).mark_bar().encode(
            y=alt.Y("brand:N", sort="-x", title=None),
            x=alt.X("size:Q", title="Cars", stack="normalize", axis=alt.Axis(format="%")),
            color=alt.Color("price_category:N", scale=PALETTE, title="Category"),
            tooltip=["brand", "price_category", "size"],
        ).properties(height=380),
        use_container_width=True,
    )

    g3, g4 = st.columns(2, gap="large")
    with g3:
        st.markdown('<div class="section-title">Accident history vs category</div>', unsafe_allow_html=True)
        d = df.groupby(["accident", "price_category"], as_index=False).size()
        st.altair_chart(
            alt.Chart(d).mark_bar().encode(
                x=alt.X("accident:N", title=None),
                y=alt.Y("size:Q", stack="normalize", title=None, axis=alt.Axis(format="%")),
                color=alt.Color("price_category:N", scale=PALETTE, title="Category"),
                tooltip=["accident", "price_category", "size"],
            ).properties(height=300),
            use_container_width=True,
        )
    with g4:
        st.markdown('<div class="section-title">Fuel type vs category</div>', unsafe_allow_html=True)
        d = df.groupby(["fuel_type", "price_category"], as_index=False).size()
        st.altair_chart(
            alt.Chart(d).mark_bar().encode(
                x=alt.X("fuel_type:N", title=None, axis=alt.Axis(labelAngle=-30)),
                y=alt.Y("size:Q", stack="normalize", title=None, axis=alt.Axis(format="%")),
                color=alt.Color("price_category:N", scale=PALETTE, title="Category"),
                tooltip=["fuel_type", "price_category", "size"],
            ).properties(height=300),
            use_container_width=True,
        )

# ==========================================================================
# TAB 3 - DATA
# ==========================================================================
with tab_data:
    f1, f2, f3 = st.columns(3)
    sel_brand = f1.multiselect("Brand", sorted(df["brand"].unique()))
    sel_cat = f2.multiselect("Price category", ["Budget", "Mid-Range", "Premium"])
    yr = f3.slider("Model year", YEAR_MIN, YEAR_MAX, (YEAR_MIN, YEAR_MAX), key="yr_filter")

    view = df[df["model_year"].between(*yr)]
    if sel_brand:
        view = view[view["brand"].isin(sel_brand)]
    if sel_cat:
        view = view[view["price_category"].isin(sel_cat)]

    st.caption(f"Showing {len(view):,} of {len(df):,} cars")
    st.dataframe(view, use_container_width=True, hide_index=True, height=460)
    st.download_button(
        "⬇️ Download filtered data (CSV)",
        view.to_csv(index=False).encode("utf-8"),
        "used_cars_filtered.csv",
        "text/csv",
    )

st.markdown(
    '<div class="footer">Built with Streamlit · KNN classifier · '
    'Predictions are estimates based on the training dataset, not a valuation.</div>',
    unsafe_allow_html=True,
)
