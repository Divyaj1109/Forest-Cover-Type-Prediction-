import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EcoType — Forest Cover Predictor",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0a1f0e 0%, #0d2b12 40%, #0f1f18 100%);
    min-height: 100vh;
}

/* ── Hero Header ── */
.hero-block {
    background: linear-gradient(120deg, #132a14 0%, #1a3d1e 60%, #0f2d1a 100%);
    border: 1px solid #2e5c32;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-block::before {
    content: "🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲🌲";
    position: absolute;
    top: -10px; left: 0; right: 0;
    font-size: 2rem;
    opacity: 0.06;
    letter-spacing: 8px;
    white-space: nowrap;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 900;
    color: #a8e6a3;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1rem;
    color: #6dbf67;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}
.hero-badge {
    display: inline-block;
    background: #1e4d22;
    border: 1px solid #3a7d3e;
    color: #7ddb78;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Section Cards ── */
.section-card {
    background: rgba(15, 35, 18, 0.85);
    border: 1px solid #2a4d2e;
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(6px);
}
.section-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #8fd48a;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border-bottom: 1px solid #2a4d2e;
    padding-bottom: 0.6rem;
}

/* ── Inputs ── */
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #0e2412 !important;
    border: 1px solid #2e5432 !important;
    color: #c8f0c4 !important;
    border-radius: 8px !important;
}
.stNumberInput label, .stSelectbox label {
    color: #7abf76 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ── Predict Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2d6e32, #1b5e20) !important;
    color: #e8f5e9 !important;
    border: 1px solid #4caf50 !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #388e3c, #2e7d32) !important;
    box-shadow: 0 4px 20px rgba(76, 175, 80, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Result Box ── */
.result-box {
    background: linear-gradient(135deg, #1a3d1e, #1e4d22);
    border: 2px solid #4caf50;
    border-radius: 18px;
    padding: 2.2rem;
    text-align: center;
    margin-top: 1.5rem;
    box-shadow: 0 8px 32px rgba(76,175,80,0.2);
}
.result-label {
    font-size: 0.85rem;
    color: #81c784;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.result-type {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 900;
    color: #a5d6a7;
    margin: 0.3rem 0;
}
.result-confidence {
    font-size: 1rem;
    color: #66bb6a;
    font-weight: 500;
    margin-top: 0.3rem;
}
.result-class {
    font-size: 0.8rem;
    color: #4caf50;
    margin-top: 0.5rem;
    opacity: 0.8;
}

/* ── Probability Bars ── */
.prob-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.55rem;
    gap: 0.8rem;
}
.prob-label {
    color: #a5d6a7;
    font-size: 0.8rem;
    width: 150px;
    flex-shrink: 0;
    font-weight: 500;
}
.prob-bar-bg {
    flex: 1;
    background: #0e2412;
    border-radius: 6px;
    height: 10px;
    border: 1px solid #2a4d2e;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #2e7d32, #66bb6a);
    transition: width 0.6s ease;
}
.prob-val {
    color: #66bb6a;
    font-size: 0.78rem;
    width: 45px;
    text-align: right;
    flex-shrink: 0;
    font-weight: 600;
}

/* ── Info Metric Boxes ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-box {
    flex: 1;
    background: #0e2412;
    border: 1px solid #2a4d2e;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #81c784;
}
.metric-name {
    font-size: 0.72rem;
    color: #4a8c4e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #091a0c !important;
    border-right: 1px solid #1e3d22 !important;
}
[data-testid="stSidebar"] * {
    color: #7abf76 !important;
}

/* ── Divider ── */
hr { border-color: #2a4d2e !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a1f0e; }
::-webkit-scrollbar-thumb { background: #2e5c32; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Load model & artifacts
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load(r"D:\AI&ML\Eco type forest cover\forest_cover_rf_model.pkl")
    features = joblib.load(r"D:\AI&ML\Eco type forest cover\selected_features.pkl")
    encoders = joblib.load(r"D:\AI&ML\Eco type forest cover\all_encoders.pkl")
    le       = encoders['Cover_Type']
    # Build correct label mapping from encoder
    labels   = {i: cls for i, cls in enumerate(le.classes_)}
    return model, features, labels, le


try:
    model, selected_features, label_mapping, le = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error   = str(e)


# ─────────────────────────────────────────────
#  Cover type metadata
# ─────────────────────────────────────────────
cover_info = {
    "Spruce/Fir"        : {"emoji": "🌲", "color": "#2d6e32", "desc": "Dense, high-elevation conifer forests"},
    "Lodgepole Pine"    : {"emoji": "🌳", "color": "#1b5e20", "desc": "Dominant mid-elevation pine forests"},
    "Ponderosa Pine"    : {"emoji": "🌴", "color": "#33691e", "desc": "Lower-elevation open pine forests"},
    "Cottonwood/Willow" : {"emoji": "🍃", "color": "#558b2f", "desc": "Riparian zones near water sources"},
    "Aspen"             : {"emoji": "🍂", "color": "#827717", "desc": "Deciduous forests, fire-adapted areas"},
    "Douglas-fir"       : {"emoji": "🌿", "color": "#2e7d32", "desc": "Mixed conifer mid-elevation zones"},
    "Krummholz"         : {"emoji": "🏔️", "color": "#37474f", "desc": "Stunted trees near treeline zones"},
}

wilderness_options = {
    "Rawah Wilderness"          : 1,
    "Neota Wilderness"          : 2,
    "Comanche Peak Wilderness"  : 3,
    "Cache la Poudre Wilderness": 4,
}

soil_descriptions = {
    1:"Cathedral - Leighcan (rocky)",       2:"Vanet - Ratake (rocky)",
    3:"Haploborolis (rocky/stony)",          4:"Ratake (rocky)",
    5:"Vanet (rocky)",                       6:"Vanet-Wetmore (stony)",
    7:"Gothic (very stony)",                 8:"Supervisor-Limber (very stony)",
    9:"Troutville (very stony)",            10:"Bullwark-Catamount (very stony)",
    11:"Bullwark-Catamount (rubble)",       12:"Legault (rubble)",
    13:"Catamount (very stony)",            14:"Pachic Argiborolis (grassy)",
    15:"Unspecified (extremely stony)",     16:"Cryaquolis (wet/boggy)",
    17:"Gateview (alluvial)",               18:"Rogert (very stony)",
    19:"Typic Cryaquolis (wet)",            20:"Typic Cryaquepts (wet)",
    21:"Typic Cryaquolls (leighcan)",       22:"Leighcan (warm)",
    23:"Leighcan (warm, very stony)",       24:"Leighcan (very stony)",
    25:"Leighcan (extremely stony)",        26:"Leighcan (warm/rubbly)",
    27:"Granile-Catamount (very stony)",    28:"Leighcan-Catamount (very stony)",
    29:"Leighcan-Catamount (extremely stony)",30:"Gothic (extremely stony)",
    31:"Legault (very stony)",              32:"Catamount-Leighcan (very stony)",
    33:"Leighcan-Catamount (sandy)",        34:"Como-Legault (rubbly)",
    35:"Como (rubbly)",                     36:"Leighcan-Catamount (bouldery)",
    37:"Bross (very stony)",               38:"Rock outcrop-Leighcan (complex)",
    39:"Leighcan-Catamount (bouldery)",    40:"Intricate-Rock (complex)",
}


# ─────────────────────────────────────────────
#  Feature Engineering (mirrors training pipeline)
# ─────────────────────────────────────────────
def engineer_features(raw: dict) -> pd.DataFrame:
    d = dict(raw)

    # Log1p Wilderness Area (applied in Step 3)
    d['Wilderness_Area'] = np.log1p(d['Wilderness_Area'])

    # Derived features (Step 4)
    d['Hydro_Distance_Ratio']          = d['Vertical_Distance_To_Hydrology'] / (d['Horizontal_Distance_To_Hydrology'] + 1)
    d['Mean_Hydro_Distance']           = (d['Horizontal_Distance_To_Hydrology'] + d['Vertical_Distance_To_Hydrology']) / 2
    d['Road_Fire_Ratio']               = d['Horizontal_Distance_To_Roadways'] / (d['Horizontal_Distance_To_Fire_Points'] + 1)
    d['Total_Remote_Distance']         = (d['Horizontal_Distance_To_Roadways'] +
                                          d['Horizontal_Distance_To_Fire_Points'] +
                                          d['Horizontal_Distance_To_Hydrology'])
    d['Mean_Hillshade']                = (d['Hillshade_9am'] + d['Hillshade_Noon'] + d['Hillshade_3pm']) / 3
    d['Hillshade_Morning_Afternoon_Diff'] = d['Hillshade_9am'] - d['Hillshade_3pm']
    d['Hillshade_Noon_Morning_Diff']   = d['Hillshade_Noon'] - d['Hillshade_9am']
    d['Elevation_Slope_Interaction']   = d['Elevation'] * d['Slope']
    d['Elevation_Hydro']               = d['Elevation'] - d['Vertical_Distance_To_Hydrology']
    d['Aspect_Sin']                    = np.sin(np.radians(d['Aspect']))
    d['Aspect_Cos']                    = np.cos(np.radians(d['Aspect']))

    df = pd.DataFrame([d])
    return df[selected_features]  # keep only selected 17 features


# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-block">
    <div class="hero-title">🌲 EcoType Forest Cover</div>
    <div class="hero-sub">Machine Learning · Forest Cover Type Prediction · Roosevelt National Forest</div>
    <div class="hero-badge">Random Forest · 96.47% Accuracy · 7 Cover Classes</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Model status + metrics
# ─────────────────────────────────────────────
if model_loaded:
    st.markdown("""
    <div class="metric-row">
        <div class="metric-box"><div class="metric-val">96.47%</div><div class="metric-name">Test Accuracy</div></div>
        <div class="metric-box"><div class="metric-val">300</div><div class="metric-name">Trees</div></div>
        <div class="metric-box"><div class="metric-val">17</div><div class="metric-name">Features</div></div>
        <div class="metric-box"><div class="metric-val">7</div><div class="metric-name">Cover Classes</div></div>
        <div class="metric-box"><div class="metric-val">145K</div><div class="metric-name">Training Rows</div></div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error(f"❌ Model files not found. Please ensure the `models/` folder exists.\n\nError: {load_error}")
    st.stop()


# ─────────────────────────────────────────────
#  Sidebar — About
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌲 EcoType")
    st.markdown("---")
    st.markdown("**Project:** Forest Cover Type Classification")
    st.markdown("**Dataset:** UCI Forest Cover Type")
    st.markdown("**Model:** Random Forest Classifier")
    st.markdown("**Classes:**")
    for name, info in cover_info.items():
        st.markdown(f"{info['emoji']} {name}")
    st.markdown("---")


# ─────────────────────────────────────────────
#  Input Form
# ─────────────────────────────────────────────
st.markdown("## 📋 Enter Cartographic Features")
st.markdown("Fill in the terrain details below to predict the forest cover type.")

col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:

    # ── Terrain Features ──
    st.markdown('<div class="section-card"><div class="section-label">🏔️ Terrain Features</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    elevation = c1.number_input("Elevation (m)",        min_value=1800, max_value=4000, value=2800, step=10,
                                  help="Height above sea level in meters")
    aspect    = c2.number_input("Aspect (°)",            min_value=0,    max_value=360,  value=180, step=1,
                                  help="Slope direction 0–360 degrees")
    slope     = c1.number_input("Slope (°)",             min_value=0,    max_value=90,   value=15,  step=1,
                                  help="Terrain steepness in degrees")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Hydrology Features ──
    st.markdown('<div class="section-card"><div class="section-label">💧 Hydrology Features</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    h_hydro = c1.number_input("Horiz. Dist. to Hydrology (m)", min_value=0,    max_value=1400, value=250,  step=10,
                                help="Horizontal distance to nearest water source")
    v_hydro = c2.number_input("Vert. Dist. to Hydrology (m)",  min_value=-200, max_value=700,  value=30,   step=5,
                                help="Vertical distance to nearest water source")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Distance Features ──
    st.markdown('<div class="section-card"><div class="section-label">🛣️ Distance Features</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    h_road = c1.number_input("Horiz. Dist. to Roadways (m)",    min_value=0, max_value=7000, value=1500, step=50,
                               help="Horizontal distance to nearest road")
    h_fire = c2.number_input("Horiz. Dist. to Fire Points (m)", min_value=0, max_value=7000, value=1500, step=50,
                               help="Horizontal distance to nearest wildfire point")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Hillshade Features ──
    st.markdown('<div class="section-card"><div class="section-label">☀️ Hillshade Features</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    shade_9am  = c1.number_input("Hillshade 9AM",  min_value=0, max_value=255, value=220, step=1,
                                   help="Illumination index at 9:00 AM")
    shade_noon = c2.number_input("Hillshade Noon", min_value=0, max_value=255, value=200, step=1,
                                   help="Illumination index at Noon")
    shade_3pm  = c3.number_input("Hillshade 3PM",  min_value=0, max_value=255, value=140, step=1,
                                   help="Illumination index at 3:00 PM")
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:

    # ── Wilderness Area ──
    st.markdown('<div class="section-card"><div class="section-label">🏞️ Wilderness Area</div>', unsafe_allow_html=True)
    wilderness_name = st.selectbox(
        "Select Wilderness Area",
        options=list(wilderness_options.keys()),
        help="Designated wilderness zone"
    )
    wilderness_val = wilderness_options[wilderness_name]
    st.markdown(f'<div style="font-size:0.78rem;color:#4a8c4e;margin-top:0.3rem;">Encoded value: {wilderness_val} → log1p → {np.log1p(wilderness_val):.4f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Soil Type ──
    st.markdown('<div class="section-card"><div class="section-label">🪨 Soil Type</div>', unsafe_allow_html=True)
    soil_options = [f"Type {k}: {v}" for k, v in soil_descriptions.items()]
    soil_choice  = st.selectbox("Select Soil Type", options=soil_options, index=28,
                                  help="Soil classification at this location")
    soil_val     = int(soil_choice.split(":")[0].replace("Type ", "").strip())
    st.markdown(f'<div style="font-size:0.78rem;color:#4a8c4e;margin-top:0.3rem;">Soil Type Code: {soil_val}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Derived Features Preview ──
    st.markdown('<div class="section-card"><div class="section-label">⚙️ Auto-Computed Features</div>', unsafe_allow_html=True)
    elev_hydro   = elevation - v_hydro
    mean_shade   = (shade_9am + shade_noon + shade_3pm) / 3
    total_remote = h_road + h_fire + h_hydro

    st.markdown(f"""
    <div style="font-size:0.82rem; color:#6dbf67; line-height:2;">
        🔹 <b>Elevation_Hydro</b>: {elev_hydro:.1f}<br>
        🔹 <b>Mean_Hillshade</b>: {mean_shade:.1f}<br>
        🔹 <b>Total_Remote_Dist</b>: {total_remote}<br>
        🔹 <b>Aspect_Sin</b>: {np.sin(np.radians(aspect)):.4f}<br>
        🔹 <b>Aspect_Cos</b>: {np.cos(np.radians(aspect)):.4f}<br>
        🔹 <b>Elev × Slope</b>: {elevation * slope}
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict Button ──
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🌲 PREDICT FOREST COVER TYPE")


# ─────────────────────────────────────────────
#  Prediction
# ─────────────────────────────────────────────
if predict_clicked:
    raw_input = {
        'Elevation'                        : elevation,
        'Aspect'                           : aspect,
        'Slope'                            : slope,
        'Horizontal_Distance_To_Hydrology' : h_hydro,
        'Vertical_Distance_To_Hydrology'   : v_hydro,
        'Horizontal_Distance_To_Roadways'  : h_road,
        'Hillshade_9am'                    : shade_9am,
        'Hillshade_Noon'                   : shade_noon,
        'Hillshade_3pm'                    : shade_3pm,
        'Horizontal_Distance_To_Fire_Points': h_fire,
        'Wilderness_Area'                  : wilderness_val,
        'Soil_Type'                        : soil_val,
    }

    try:
        input_df      = engineer_features(raw_input)
        prediction    = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]


        # Inverse transform using label mapping
        encoders    = joblib.load(r"D:\AI&ML\Eco type forest cover\all_encoders.pkl")
        le          = encoders['Cover_Type']
        cover_name    = le.inverse_transform([prediction])[0]
        confidence    = probabilities[prediction] * 100
        info          = cover_info.get(cover_name, {"emoji":"🌲","desc":"Forest cover type"})

        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")

        res_col, prob_col = st.columns([1, 1.2], gap="large")

        with res_col:
            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Predicted Forest Cover Type</div>
                <div style="font-size:3.5rem;margin:0.5rem 0;">{info['emoji']}</div>
                <div class="result-type">{cover_name}</div>
                <div class="result-confidence">Confidence: {confidence:.2f}%</div>
                <div class="result-class">Class Index: {prediction} &nbsp;|&nbsp; {info['desc']}</div>
                <div class="result-class" style="margin-top:0.5rem;">📍 {wilderness_name}</div>
            </div>
            """, unsafe_allow_html=True)

        with prob_col:
            st.markdown('<div class="section-card"><div class="section-label">📊 Class Probabilities</div>', unsafe_allow_html=True)

            sorted_probs = sorted(
                zip(label_mapping.values(), probabilities),
                key=lambda x: -x[1]
            )
            prob_html = ""
            for cls_name, prob in sorted_probs:
                pct      = prob * 100
                width    = int(pct)
                is_pred  = (cls_name == cover_name)
                bar_color = "#66bb6a" if is_pred else "#2e5c32"
                label_style = "font-weight:700;color:#a5d6a7;" if is_pred else ""
                prob_html += f"""
                <div class="prob-row">
                    <div class="prob-label" style="{label_style}">
                        {cover_info.get(cls_name,{}).get('emoji','🌲')} {cls_name}
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{width}%;background:{bar_color};"></div>
                    </div>
                    <div class="prob-val">{pct:.1f}%</div>
                </div>"""

            st.markdown(prob_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Input Summary Table ──
        st.markdown("### 📋 Input Summary")
        summary_data = {
            "Feature"  : ["Elevation","Aspect","Slope",
                           "Horiz. Dist. Hydrology","Vert. Dist. Hydrology",
                           "Dist. Roadways","Dist. Fire Points",
                           "Hillshade 9AM","Hillshade Noon","Hillshade 3PM",
                           "Wilderness Area","Soil Type"],
            "Value"    : [f"{elevation} m", f"{aspect}°", f"{slope}°",
                          f"{h_hydro} m", f"{v_hydro} m",
                          f"{h_road} m", f"{h_fire} m",
                          shade_9am, shade_noon, shade_3pm,
                          wilderness_name, f"Type {soil_val}"]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.info("💡 Make sure all model files exist in the `models/` folder.")


# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#3a6b3e; font-size:0.78rem; padding:1rem 0;">
    🌲 EcoType Forest Cover Predictor &nbsp;|&nbsp;
    Random Forest · 96.47% Accuracy &nbsp;|&nbsp;
    Roosevelt National Forest, Colorado
</div>
""", unsafe_allow_html=True)