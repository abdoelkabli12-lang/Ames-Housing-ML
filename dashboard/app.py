import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Page Config ──
st.set_page_config(
    page_title="🏠 Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Artifacts (cached) ──
@st.cache_resource
def load_artifacts():
    base = Path(__file__).parent.parent
    pipe = joblib.load(base / "models" / "best_pipeline.pkl")
    with open(base / "models" / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(base / "models" / "target_info.json") as f:
        target_info = json.load(f)
    return pipe, feature_names, target_info

pipeline, feature_names, target_info = load_artifacts()

ALL_FEATURES = feature_names
MODEL_NAME = target_info["model_name"]
TARGET = target_info["target"]
TEST_R2 = target_info["test_r2"]
TEST_MAE = target_info["test_mae"]
TEST_RMSE = target_info["test_rmse"]
LOG_TARGET = TARGET == "LogSalePrice"

# ── Mappings ──
ORDINAL_LABELS = {
    "ExterQual":    ["Po", "Fa", "Gd", "Ex"],
    "ExterCond":    ["Po", "Fa", "Gd", "Ex"],
    "KitchenQual":  ["Po", "Fa", "Gd", "Ex"],
    "HeatingQC":    ["Po", "Fa", "Gd", "Ex"],
    "BsmtQual":     ["No Basement", "Po", "Fa", "Gd", "Ex"],
    "BsmtCond":     ["No Basement", "Po", "Fa", "Gd", "Ex"],
    "GarageQual":   ["No Garage", "Po", "Fa", "Gd", "Ex"],
    "GarageCond":   ["No Garage", "Po", "Fa", "Gd", "Ex"],
}

def ordinal_index(col, label):
    return ORDINAL_LABELS[col].index(label)

# ── Default values for every feature ──
def default_feature_dict():
    d = {}
    for col in ALL_FEATURES:
        if col == "Id":
            d[col] = 0.0
        elif col == "MSSubClass":
            d[col] = 20.0
        elif col == "LotFrontage":
            d[col] = 60.0
        elif col == "LotArea":
            d[col] = 8000.0
        elif col == "OverallCond":
            d[col] = 5.0
        elif col == "YearRemodAdd":
            d[col] = 0.0
        elif col == "MasVnrArea":
            d[col] = 0.0
        elif col == "BsmtFinSF1":
            d[col] = 0.0
        elif col == "BsmtUnfSF":
            d[col] = 0.0
        elif col == "BsmtHalfBath":
            d[col] = 0.0
        elif col == "HalfBath":
            d[col] = 1.0
        elif col == "BedroomAbvGr":
            d[col] = 3.0
        elif col == "TotRmsAbvGrd":
            d[col] = 7.0
        elif col == "GarageYrBlt":
            d[col] = 2000.0
        elif col == "GarageArea":
            d[col] = 400.0
        elif col == "WoodDeckSF":
            d[col] = 0.0
        elif col == "OpenPorchSF":
            d[col] = 0.0
        elif col == "EnclosedPorch":
            d[col] = 0.0
        elif col == "3SsnPorch":
            d[col] = 0.0
        elif col == "ScreenPorch":
            d[col] = 0.0
        elif col == "MoSold":
            d[col] = 6.0
        elif col == "YrSold":
            d[col] = 2008.0
        elif col == "Fireplaces":
            d[col] = 1.0
        elif col in ORDINAL_LABELS:
            labels = ORDINAL_LABELS[col]
            d[col] = float(ordinal_index(col, labels[len(labels) // 2]))
        elif col in ("HasGarage", "HasBasement", "HasFireplace", "HasPool", "HasMasonryVeneer"):
            d[col] = 1.0
        elif col == "IsNew":
            d[col] = 0.0
        elif col.startswith("MSZoning_"):
            d[col] = 1.0 if col == "MSZoning_RL" else 0.0
        elif col.startswith("BldgType_"):
            d[col] = 1.0 if col == "BldgType_1Fam" else 0.0
        elif col.startswith("HouseStyle_"):
            d[col] = 1.0 if col == "HouseStyle_1Story" else 0.0
        elif col.startswith("Neighborhood_"):
            d[col] = 1.0 if col == "Neighborhood_NAmes" else 0.0
        elif col.startswith("RoofStyle_"):
            d[col] = 1.0 if col == "RoofStyle_Gable" else 0.0
        elif col.startswith("Foundation_"):
            d[col] = 1.0 if col == "Foundation_PConc" else 0.0
        elif col.startswith("SaleType_"):
            d[col] = 1.0 if col == "SaleType_WD" else 0.0
        elif col.startswith("SaleCondition_"):
            d[col] = 1.0 if col == "SaleCondition_Normal" else 0.0
        elif col.startswith("CentralAir_"):
            d[col] = 1.0 if col == "CentralAir_Y" else 0.0
        elif col.startswith("PavedDrive_"):
            d[col] = 1.0 if col == "PavedDrive_Y" else 0.0
        elif col.startswith("Electrical_"):
            d[col] = 1.0 if col == "Electrical_SBrkr" else 0.0
        elif col.startswith("Functional_"):
            d[col] = 1.0 if col == "Functional_Typ" else 0.0
        elif col.startswith("Exterior1st_"):
            d[col] = 1.0 if col == "Exterior1st_VinylSd" else 0.0
        elif col.startswith("Exterior2nd_"):
            d[col] = 1.0 if col == "Exterior2nd_VinylSd" else 0.0
        elif col.startswith("MasVnrType_"):
            d[col] = 1.0 if col == "MasVnrType_None" else 0.0
        else:
            d[col] = 0.0
    return d

def compute_derived(d):
    total_sf = d["GrLivArea"] + d["TotalBsmtSF"]
    age = 2010 - d["YearBuilt"]
    remod = d.get("YearRemodAdd", 0)
    remod_age = (2010 - remod) if remod > 0 else age
    total_bath = d["FullBath"] + d.get("BsmtFullBath", 0) + 0.5 * d.get("BsmtHalfBath", 0)
    total_porch = (d.get("WoodDeckSF", 0) + d.get("OpenPorchSF", 0)
                   + d.get("EnclosedPorch", 0) + d.get("3SsnPorch", 0) + d.get("ScreenPorch", 0))
    is_new = 1.0 if d["YearBuilt"] >= 2007 else 0.0
    d["TotalSF"] = total_sf
    d["Age"] = max(0, age)
    d["RemodAge"] = max(0, remod_age)
    d["TotalBath"] = total_bath
    d["TotalPorchSF"] = total_porch
    d["IsNew"] = is_new

def build_input_df(inputs):
    d = default_feature_dict()
    for col in [
        "OverallQual", "OverallCond", "GrLivArea", "TotalBsmtSF",
        "1stFlrSF", "2ndFlrSF", "LotArea", "LotFrontage",
        "GarageCars", "GarageArea", "MasVnrArea", "YearBuilt",
        "YearRemodAdd", "FullBath", "BsmtFullBath", "BsmtHalfBath",
        "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd", "Fireplaces",
        "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
        "ScreenPorch", "PoolArea", "GarageYrBlt",
    ]:
        if col in inputs:
            d[col] = float(inputs[col])
    for col in ORDINAL_LABELS:
        if col in inputs:
            d[col] = float(ordinal_index(col, inputs[col]))
    for col in ["HasGarage", "HasBasement", "HasFireplace", "HasPool", "HasMasonryVeneer"]:
        if col in inputs:
            d[col] = 1.0 if inputs[col] else 0.0
    cat_map = {
        "MSZoning": "MSZoning", "BldgType": "BldgType", "HouseStyle": "HouseStyle",
        "Neighborhood": "Neighborhood", "RoofStyle": "RoofStyle", "Foundation": "Foundation",
        "SaleType": "SaleType", "SaleCondition": "SaleCondition",
        "CentralAir": "CentralAir", "PavedDrive": "PavedDrive",
        "Electrical": "Electrical", "Functional": "Functional",
        "Exterior1st": "Exterior1st", "Exterior2nd": "Exterior2nd",
        "MasVnrType": "MasVnrType",
    }
    for cat_col, prefix in cat_map.items():
        if cat_col in inputs:
            chosen = inputs[cat_col]
            for c in ALL_FEATURES:
                if c.startswith(f"{prefix}_"):
                    d[c] = 0.0
            target_col = f"{prefix}_{chosen}"
            if target_col in d:
                d[target_col] = 1.0
    compute_derived(d)
    return pd.DataFrame({col: [d[col]] for col in ALL_FEATURES})

# ── Sidebar ──
st.sidebar.title("🏠 Property Features")
st.sidebar.caption("Adjust features below, then click **Predict Price**.")
st.sidebar.markdown("---")

# ── Overview ──
st.sidebar.subheader("Overall")
oq = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 7, 1, key="oq")
oc = st.sidebar.slider("Overall Condition (1-10)", 1, 10, 5, 1, key="oc")
yb = st.sidebar.number_input("Year Built", 1872, 2010, 2000, 1, key="yb")
yr = st.sidebar.number_input("Year Remodeled (0 = never)", 0, 2010, 0, 1, key="yr")

# ── Size ──
st.sidebar.subheader("Size")
c1, c2 = st.sidebar.columns(2)
with c1:
    grliv = st.number_input("Living Area (sq ft)", 300, 5000, 1500, 50, key="grliv")
    bsmt = st.number_input("Basement Area (sq ft)", 0, 5000, 800, 50, key="bsmt")
with c2:
    fl1 = st.number_input("1st Floor Area (sq ft)", 300, 4000, 1000, 50, key="fl1")
    fl2 = st.number_input("2nd Floor Area (sq ft)", 0, 2000, 500, 50, key="fl2")

c1, c2, c3 = st.sidebar.columns(3)
with c1:
    lot = st.number_input("Lot Area (sq ft)", 1000, 20000, 8000, 500, key="lot")
with c2:
    gcars = st.number_input("Garage Capacity (cars)", 0, 4, 2, 1, key="gcars")
with c3:
    garea = st.number_input("Garage Area (sq ft)", 0, 1500, 400, 50, key="garea")

mv = st.sidebar.number_input("Masonry Veneer Area (sq ft)", 0, 1200, 0, 25, key="mv")

# ── Bathrooms & Rooms ──
st.sidebar.subheader("Bathrooms & Rooms")
c1, c2, c3 = st.sidebar.columns(3)
with c1:
    fb = st.number_input("Full Baths", 0, 4, 2, 1, key="fb")
with c2:
    bfb = st.number_input("Basement Full Baths", 0, 2, 0, 1, key="bfb")
with c3:
    bhb = st.number_input("Basement Half Baths", 0, 2, 0, 1, key="bhb")

c1, c2 = st.sidebar.columns(2)
with c1:
    hb = st.number_input("Half Baths", 0, 2, 1, 1, key="hb2")
with c2:
    beds = st.number_input("Bedrooms", 0, 6, 3, 1, key="bed")

rooms = st.number_input("Rooms Above Ground", 2, 14, 7, 1, key="rooms")

# ── Quality Ratings ──
st.sidebar.subheader("Quality Ratings")
qc1, qc2 = st.sidebar.columns(2)
for i, (col_name, labels) in enumerate(ORDINAL_LABELS.items()):
    with qc1 if i % 2 == 0 else qc2:
        default_idx = len(labels) // 2
        label_key = f"ord_{col_name}"
        st.selectbox(
            col_name.replace("Qual", " Quality").replace("Cond", " Condition"),
            options=labels,
            index=default_idx,
            key=label_key,
        )

# ── Basement, Garage & Outdoor ──
st.sidebar.subheader("Basement, Garage & Outdoor")
c1, c2, c3 = st.sidebar.columns(3)
with c1:
    hg = st.checkbox("Has Garage", value=True, key="hg")
with c2:
    hbsmt = st.checkbox("Has Basement", value=True, key="hbsmt")
with c3:
    hfp = st.checkbox("Has Fireplace", value=True, key="hfp")

gyr = st.sidebar.number_input("Garage Built Year", 0, 2010, 2000, 1, key="gyr")

c1, c2, c3 = st.sidebar.columns(3)
with c1:
    pool = st.number_input("Pool Area (sq ft)", 0, 800, 0, 25, key="pool")
with c2:
    hp2 = st.checkbox("Has Pool", value=False, key="hp2")
with c3:
    hmv = st.checkbox("Has Masonry Veneer", value=False, key="hmv")

st.sidebar.markdown("— Outdoor —")
c1, c2, c3, c4, c5 = st.sidebar.columns(5)
with c1:
    wd = st.number_input("Wood Deck SF", 0, 1500, 0, 25, key="wd")
with c2:
    op = st.number_input("Open Porch SF", 0, 600, 0, 25, key="op")
with c3:
    ep = st.number_input("Enclosed Porch SF", 0, 800, 0, 25, key="ep")
with c4:
    p3 = st.number_input("3-Season Porch SF", 0, 600, 0, 25, key="p3")
with c5:
    sp = st.number_input("Screen Porch SF", 0, 600, 0, 25, key="sp")

# ── Location & Building ──
st.sidebar.subheader("Location & Building")
msz_opts = sorted([c.replace("MSZoning_", "") for c in ALL_FEATURES if c.startswith("MSZoning_")])
bldg_opts = sorted([c.replace("BldgType_", "") for c in ALL_FEATURES if c.startswith("BldgType_")])
house_opts = sorted([c.replace("HouseStyle_", "") for c in ALL_FEATURES if c.startswith("HouseStyle_")])
nh_opts = sorted([c.replace("Neighborhood_", "") for c in ALL_FEATURES if c.startswith("Neighborhood_")])
roof_opts = sorted([c.replace("RoofStyle_", "") for c in ALL_FEATURES if c.startswith("RoofStyle_")])
found_opts = sorted([c.replace("Foundation_", "") for c in ALL_FEATURES if c.startswith("Foundation_")])

st.sidebar.selectbox("MS Zoning", options=msz_opts, index=msz_opts.index("RL"), key="msz")
st.sidebar.selectbox("Building Type", options=bldg_opts, index=bldg_opts.index("1Fam"), key="bldg")
st.sidebar.selectbox("House Style", options=house_opts, index=house_opts.index("1Story"), key="house")
st.sidebar.selectbox("Neighborhood", options=nh_opts, index=nh_opts.index("NAmes"), key="nh")
st.sidebar.selectbox("Roof Style", options=roof_opts, index=roof_opts.index("Gable"), key="roof")
st.sidebar.selectbox("Foundation", options=found_opts, index=found_opts.index("PConc"), key="found")

# ── Exterior ──
st.sidebar.subheader("Exterior")
ext1_opts = sorted([c.replace("Exterior1st_", "") for c in ALL_FEATURES if c.startswith("Exterior1st_")])
ext2_opts = sorted([c.replace("Exterior2nd_", "") for c in ALL_FEATURES if c.startswith("Exterior2nd_")])
mvt_opts = sorted([c.replace("MasVnrType_", "") for c in ALL_FEATURES if c.startswith("MasVnrType_")])

c1, c2 = st.sidebar.columns(2)
with c1:
    st.selectbox("Exterior 1st", options=ext1_opts, index=ext1_opts.index("VinylSd"), key="ext1")
with c2:
    st.selectbox("Exterior 2nd", options=ext2_opts, index=ext2_opts.index("VinylSd"), key="ext2")
st.selectbox("Masonry Veneer Type", options=mvt_opts, index=mvt_opts.index("None"), key="mvt")

# ── Utilities & Sale ──
st.sidebar.subheader("Utilities & Sale")
ca_opts = sorted([c.replace("CentralAir_", "") for c in ALL_FEATURES if c.startswith("CentralAir_")])
pd_opts = sorted([c.replace("PavedDrive_", "") for c in ALL_FEATURES if c.startswith("PavedDrive_")])
el_opts = sorted([c.replace("Electrical_", "") for c in ALL_FEATURES if c.startswith("Electrical_")])
func_opts = sorted([c.replace("Functional_", "") for c in ALL_FEATURES if c.startswith("Functional_")])
stype_opts = sorted([c.replace("SaleType_", "") for c in ALL_FEATURES if c.startswith("SaleType_")])
scond_opts = sorted([c.replace("SaleCondition_", "") for c in ALL_FEATURES if c.startswith("SaleCondition_")])

c1, c2 = st.sidebar.columns(2)
with c1:
    st.selectbox("Central Air", options=ca_opts, index=ca_opts.index("Y"), key="ca")
    st.selectbox("Electrical", options=el_opts, index=el_opts.index("SBrkr"), key="el")
    st.selectbox("Sale Type", options=stype_opts, index=stype_opts.index("WD"), key="stype")
with c2:
    st.selectbox("Paved Drive", options=pd_opts, index=pd_opts.index("Y"), key="pd")
    st.selectbox("Functional", options=func_opts, index=func_opts.index("Typ"), key="func")
    st.selectbox("Sale Condition", options=scond_opts, index=scond_opts.index("Normal"), key="scond")

# ── Predict Button ──
st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔮 Predict Price", type="primary", use_container_width=True, key="predict")

# ── Main Area ──
st.title("🏠 Housing Price Predictor")
st.caption(f"Ames Housing Dataset — {MODEL_NAME} model trained on {TARGET}")

with st.expander("📊 Model Information", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", MODEL_NAME)
    c2.metric("Target", TARGET)
    c3.metric("R²", f"{TEST_R2:.4f}")
    st.caption(f"Test MAE: **${TEST_MAE:,.0f}**  ·  Test RMSE: **${TEST_RMSE:,.0f}**")
    st.caption(
        "Trained on 2,335 houses · Tested on 584 houses. "
        f"Predictions are {'converted from log-space to dollars' if LOG_TARGET else 'in raw dollars'}. "
        "The model does NOT retrain per prediction."
    )
    st.markdown("### How to use")
    st.markdown(
        "- Adjust features in the sidebar\n"
        "- Click **Predict Price**\n"
        "- See the estimated price and feature breakdown\n\n"
        "Unspecified features use sensible defaults. "
        "The pre-trained pipeline handles imputation and scaling automatically."
    )

if predict_btn:
    fp_input = st.sidebar.number_input("Fireplaces", 0, 3, 1, 1, key="fp")

    inputs = {
        "OverallQual": oq, "OverallCond": oc,
        "YearBuilt": yb, "YearRemodAdd": yr,
        "GrLivArea": grliv, "TotalBsmtSF": bsmt,
        "1stFlrSF": fl1, "2ndFlrSF": fl2,
        "LotArea": lot, "GarageCars": gcars, "GarageArea": garea,
        "MasVnrArea": mv,
        "FullBath": fb, "BsmtFullBath": bfb, "BsmtHalfBath": bhb,
        "HalfBath": hb, "BedroomAbvGr": beds, "TotRmsAbvGrd": rooms,
        "Fireplaces": fp_input,
        "LotFrontage": default_feature_dict()["LotFrontage"],
        "GarageYrBlt": gyr,
        "HasGarage": hg, "HasBasement": hbsmt,
        "HasFireplace": hfp, "HasPool": hp2, "HasMasonryVeneer": hmv,
        "ExterQual": st.session_state.get("ord_ExterQual", "Gd"),
        "ExterCond": st.session_state.get("ord_ExterCond", "Gd"),
        "KitchenQual": st.session_state.get("ord_KitchenQual", "Gd"),
        "HeatingQC": st.session_state.get("ord_HeatingQC", "Ex"),
        "BsmtQual": st.session_state.get("ord_BsmtQual", "Gd"),
        "BsmtCond": st.session_state.get("ord_BsmtCond", "Fa"),
        "GarageQual": st.session_state.get("ord_GarageQual", "Gd"),
        "GarageCond": st.session_state.get("ord_GarageCond", "Gd"),
        "MSZoning": st.session_state.get("msz", "RL"),
        "BldgType": st.session_state.get("bldg", "1Fam"),
        "HouseStyle": st.session_state.get("house", "1Story"),
        "Neighborhood": st.session_state.get("nh", "NAmes"),
        "RoofStyle": st.session_state.get("roof", "Gable"),
        "Foundation": st.session_state.get("found", "PConc"),
        "Exterior1st": st.session_state.get("ext1", "VinylSd"),
        "Exterior2nd": st.session_state.get("ext2", "VinylSd"),
        "MasVnrType": st.session_state.get("mvt", "None"),
        "CentralAir": st.session_state.get("ca", "Y"),
        "PavedDrive": st.session_state.get("pd", "Y"),
        "Electrical": st.session_state.get("el", "SBrkr"),
        "Functional": st.session_state.get("func", "Typ"),
        "SaleType": st.session_state.get("stype", "WD"),
        "SaleCondition": st.session_state.get("scond", "Normal"),
    }

    try:
        df = build_input_df(inputs)
        pred_log = float(pipeline.predict(df)[0])
        pred_dollars = float(np.expm1(pred_log)) if LOG_TARGET else pred_log

        col_r, col_i = st.columns([1, 2])

        with col_r:
            st.info(f"### ${pred_dollars:,.0f}")
            st.caption("Estimated Market Price")
            st.markdown("---")
            st.markdown("### Model Details")
            st.markdown(f"- **Model:** {MODEL_NAME}")
            st.markdown(f"- **Target:** {TARGET}")
            st.markdown(f"- **R²:** {TEST_R2:.4f}")
            st.markdown(f"- **MAE:** ${TEST_MAE:,.0f}")
            st.markdown(f"- **RMSE:** ${TEST_RMSE:,.0f}")
            st.markdown(f"- **Features:** {len(ALL_FEATURES)}")

        with col_i:
            st.markdown("### Feature Breakdown")
            st.caption("Values used for this prediction:")

            display_groups = [
                ("Overview", ["OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd"]),
                ("Size", ["GrLivArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LotArea",
                          "GarageCars", "GarageArea", "MasVnrArea"]),
                ("Bathrooms & Rooms", ["FullBath", "BsmtFullBath", "BsmtHalfBath",
                                       "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd", "Fireplaces"]),
                ("Quality Ratings", list(ORDINAL_LABELS.keys())),
                ("Basement, Garage & Outdoor",
                 ["HasGarage", "HasBasement", "HasFireplace", "GarageYrBlt",
                  "PoolArea", "HasPool", "WoodDeckSF", "OpenPorchSF",
                  "EnclosedPorch", "3SsnPorch", "ScreenPorch", "HasMasonryVeneer"]),
                ("Location & Building", ["MSZoning", "BldgType", "HouseStyle",
                                         "Neighborhood", "RoofStyle", "Foundation"]),
                ("Exterior", ["Exterior1st", "Exterior2nd", "MasVnrType"]),
                ("Utilities & Sale", ["CentralAir", "PavedDrive", "Electrical",
                                      "Functional", "SaleType", "SaleCondition"]),
                ("Derived", ["TotalSF", "Age", "RemodAge", "TotalBath", "TotalPorchSF", "IsNew"]),
            ]

            for gname, feats in display_groups:
                rows = []
                for feat in feats:
                    if feat in ORDINAL_LABELS:
                        raw = inputs.get(feat, ORDINAL_LABELS[feat][len(ORDINAL_LABELS[feat]) // 2])
                        labels = ORDINAL_LABELS[feat]
                        idx = ordinal_index(feat, raw) if isinstance(raw, str) else int(raw)
                        disp = labels[idx] if idx < len(labels) else str(raw)
                    elif feat in ("HasGarage", "HasBasement", "HasFireplace", "HasPool", "HasMasonryVeneer", "IsNew"):
                        disp = "Yes" if inputs.get(feat, False) else "No"
                    elif any(feat.startswith(f"{p}_") for p in
                             ["MSZoning", "BldgType", "HouseStyle", "Neighborhood",
                              "RoofStyle", "Foundation", "Exterior1st", "Exterior2nd",
                              "MasVnrType", "CentralAir", "PavedDrive", "Electrical",
                              "Functional", "SaleType", "SaleCondition"]):
                        prefix = feat.split("_")[0] + "_"
                        found = False
                        for c in ALL_FEATURES:
                            if c.startswith(prefix) and df.loc[0, c] == 1.0:
                                disp = c.split("_", 1)[1]
                                found = True
                                break
                        if not found:
                            disp = "—"
                    else:
                        # Numeric or other single-column feature
                        if feat in df.columns:
                            v = df.loc[0, feat]
                            disp = f"{v:,.0f}" if isinstance(v, (int, float)) and abs(v) >= 10 else f"{v:.1f}"
                        else:
                            disp = "—"
                    # Clean feature name for display
                    cn = feat.replace("Bsmt", "Bsmt ").replace("GrLiv", "Living ").replace("1stFlr", "1st Flr").replace("2ndFlr", "2nd Flr")
                    rows.append((cn, disp))
                if rows:
                    st.markdown(f"**{gname}**")
                    for cn, disp in rows:
                        st.caption(f"**{cn}:** {disp}")

        st.markdown("---")
        st.caption(
            f"⚠️ Estimate only. Model's average test error: **${TEST_MAE:,.0f}** (MAE). "
            "Actual prices depend on market conditions and factors not in the model."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)
else:
    st.info("Adjust the features in the sidebar and click **Predict Price** to see an estimate.")
