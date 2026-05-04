"""
NBA Player Salary Predictor
Person 3's Task: Streamlit Dashboard
Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Salary Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .metric-card {
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .salary-display {
        background: linear-gradient(135deg, #c84b31, #e25822);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .salary-display h1 { color: white; font-size: 3rem; margin: 0; }
    .salary-display p { color: rgba(255,255,255,0.8); margin: 5px 0 0; }
    .stSlider > div > div { background: #e25822; }
    h1, h2, h3 { color: #f0f6fc !important; }
    .stSelectbox label, .stSlider label { color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/best_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        features = joblib.load("models/features.pkl")
        with open("models/best_model_name.txt") as f:
            model_name = f.read().strip()
        return model, scaler, features, model_name
    except FileNotFoundError:
        return None, None, None, None

model, scaler, features, model_name = load_model()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🏀 NBA Player Salary Predictor")
st.markdown("*Enter a player's stats to predict their market salary using Machine Learning*")
st.divider()

if model is None:
    st.error("⚠️ Model not found! Please run `python models/train_models.py` first.")
    st.stop()

st.success(f"✅ Model loaded: **{model_name}**")

# ─────────────────────────────────────────────
# SIDEBAR — INPUT STATS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎮 Player Stats Input")
    st.markdown("*Adjust the sliders to match a player's season stats*")
    st.divider()

    pos = st.selectbox("Position", ["PG", "SG", "SF", "PF", "C"])
    age = st.slider("Age", 19, 40, 26)
    g = st.slider("Games Played", 1, 82, 65)
    mp = st.slider("Minutes Per Game", 5.0, 40.0, 28.0, step=0.5)

    st.markdown("#### 📊 Scoring")
    pts = st.slider("Points Per Game", 0.0, 40.0, 15.0, step=0.5)
    fg_pct = st.slider("FG%", 0.25, 0.75, 0.46, step=0.01)
    three_pct = st.slider("3P%", 0.20, 0.55, 0.35, step=0.01)
    ft_pct = st.slider("FT%", 0.40, 1.00, 0.78, step=0.01)

    st.markdown("#### 🏃 Other Stats")
    ast = st.slider("Assists Per Game", 0.0, 12.0, 4.0, step=0.1)
    trb = st.slider("Rebounds Per Game", 0.0, 15.0, 5.0, step=0.1)
    stl = st.slider("Steals Per Game", 0.0, 3.5, 1.0, step=0.1)
    blk = st.slider("Blocks Per Game", 0.0, 4.0, 0.5, step=0.1)
    tov = st.slider("Turnovers Per Game", 0.0, 6.0, 2.0, step=0.1)
    gs = st.slider("Games Started", 0, 82, 50)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING (mirror train_models.py)
# ─────────────────────────────────────────────
pos_map = {"PG": 0, "SG": 1, "SF": 2, "PF": 3, "C": 4}
pos_enc = pos_map.get(pos, 2)

per = (pts + trb + ast + stl + blk - tov) / max(mp, 1)
ts = pts / (2 * (pts / max(fg_pct * 2, 0.01) + 0.44 * (pts * 0.3))) if pts > 0 else 0.5
gs_ratio = gs / max(g, 1)
age_prime = abs(age - 27)
pts_per_min = pts / max(mp, 1)

input_data = {
    "Age": age, "G": g, "MP": mp, "PTS": pts, "AST": ast,
    "TRB": trb, "STL": stl, "BLK": blk, "TOV": tov,
    "FG%": fg_pct, "3P%": three_pct, "FT%": ft_pct,
    "PER": per, "TS%": ts, "GS_ratio": gs_ratio,
    "age_prime": age_prime, "pts_per_min": pts_per_min,
    "Pos_enc": pos_enc
}

input_df = pd.DataFrame([input_data])
input_vals = input_df[[f for f in features if f in input_df.columns]].values

# Scale if needed
needs_scale = model_name in ["Ridge Regression", "SVR"]
if needs_scale:
    input_vals = scaler.transform(input_vals)

predicted_salary_m = model.predict(input_vals)[0]
predicted_salary_m = max(0.5, predicted_salary_m)  # floor at $500K

# ─────────────────────────────────────────────
# MAIN DISPLAY
# ─────────────────────────────────────────────
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("## 💰 Predicted Salary")
    st.markdown(f"""
    <div class="salary-display">
        <h1>${predicted_salary_m:.1f}M</h1>
        <p>per year — {model_name}</p>
    </div>
    """, unsafe_allow_html=True)

    # Tier label
    if predicted_salary_m >= 30:
        tier, color = "🌟 SUPERSTAR CONTRACT", "#f1c40f"
    elif predicted_salary_m >= 20:
        tier, color = "⭐ ALL-STAR CONTRACT", "#e67e22"
    elif predicted_salary_m >= 10:
        tier, color = "🔵 STARTER CONTRACT", "#3498db"
    elif predicted_salary_m >= 5:
        tier, color = "🟢 ROTATION PLAYER", "#2ecc71"
    else:
        tier, color = "⚪ MINIMUM CONTRACT", "#95a5a6"

    st.markdown(f"<h3 style='color:{color}; text-align:center'>{tier}</h3>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📊 Player Stats Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Points", f"{pts:.1f}")
    c2.metric("Assists", f"{ast:.1f}")
    c3.metric("Rebounds", f"{trb:.1f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("PER (calc)", f"{per:.2f}")
    c5.metric("Age", age)
    c6.metric("Position", pos)

with col2:
    st.markdown("## 📈 Key Influencing Factors")

    factors = {
        "Points Per Game": pts / 40,
        "Player Efficiency": min(per / 30, 1),
        "Assists": ast / 12,
        "Rebounds": trb / 15,
        "Steals + Blocks": (stl + blk) / 5,
        "Ball Control (low TOV)": max(0, 1 - tov / 6),
        "Shooting FG%": (fg_pct - 0.3) / 0.4,
    }

    for factor, value in factors.items():
        value = max(0, min(1, value))
        bar_fill = int(value * 100)
        color = "#e25822" if value > 0.7 else "#3498db" if value > 0.4 else "#6b7280"
        st.markdown(f"""
        <div style="margin-bottom:12px">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px">
                <span style="color:#c9d1d9; font-size:0.85rem">{factor}</span>
                <span style="color:#8b949e; font-size:0.85rem">{bar_fill}%</span>
            </div>
            <div style="background:#21262d; border-radius:6px; height:8px; width:100%">
                <div style="background:{color}; border-radius:6px; height:8px; width:{bar_fill}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL PERFORMANCE (show saved plots if available)
# ─────────────────────────────────────────────
st.divider()
st.markdown("## 🔬 Model Performance Analysis")

plot_col1, plot_col2 = st.columns(2)

if os.path.exists("plots/model_comparison.png"):
    with plot_col1:
        st.image("plots/model_comparison.png", caption="Model RMSE Comparison")
    with plot_col2:
        st.image("plots/actual_vs_predicted.png", caption="Actual vs Predicted Salary")

    plot_col3, plot_col4 = st.columns(2)
    with plot_col3:
        st.image("plots/feature_importance.png", caption="Feature Importance (Random Forest)")
    with plot_col4:
        st.image("plots/salary_distribution.png", caption="NBA Salary Distribution")
else:
    st.info("📊 Run `python models/train_models.py` to generate performance plots.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#6b7280; font-size:0.85rem'>
    NBA Salary Predictor | CAP 5937 Final Project | Data sourced from Basketball-Reference.com
</div>
""", unsafe_allow_html=True)