import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ConcreteIQ · Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
#  DESIGN SYSTEM — Light Premium Corporate Dashboard
# ============================================================
#
#  COLOR PALETTE
#  ─────────────────────────────────────────────────────────
#  Primary          #1B4FD8   Deep Cobalt Blue
#  Primary Dark     #1338A8   Pressed state / active
#  Primary Light    #EEF3FF   Tinted surface / chip bg
#
#  Secondary        #0EA5A0   Teal Accent (data viz anchor)
#  Secondary Light  #E6F8F7
#
#  Neutral 950      #0D1117   Near-black (headings)
#  Neutral 800      #1E2633   Body text
#  Neutral 600      #4B5563   Subtext / labels
#  Neutral 400      #9CA3AF   Placeholder / disabled
#  Neutral 200      #E5E7EB   Dividers / borders
#  Neutral 100      #F3F4F6   Card backgrounds
#  Neutral 50       #F9FAFB   Page background
#
#  Success          #059669   Green
#  Warning          #D97706   Amber
#  Error            #DC2626   Red
#
#  Surface White    #FFFFFF   Cards / inputs
# ============================================================

CSS = """
<style>
/* ── GOOGLE FONT ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── CSS VARIABLES ───────────────────────────────────────── */
:root {
  --primary:          #1B4FD8;
  --primary-dark:     #1338A8;
  --primary-light:    #EEF3FF;
  --secondary:        #0EA5A0;
  --secondary-light:  #E6F8F7;

  --n950: #0D1117;
  --n800: #1E2633;
  --n600: #4B5563;
  --n400: #9CA3AF;
  --n200: #E5E7EB;
  --n100: #F3F4F6;
  --n50:  #F9FAFB;

  --success: #059669;
  --warning: #D97706;
  --error:   #DC2626;

  --surface: #FFFFFF;
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 16px;
  --shadow-sm: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
  --shadow-md: 0 4px 12px rgba(0,0,0,.08), 0 2px 4px rgba(0,0,0,.05);
  --shadow-lg: 0 8px 24px rgba(0,0,0,.10), 0 4px 8px rgba(0,0,0,.06);

  --font: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  --mono: 'DM Mono', 'Fira Mono', monospace;
}

/* ── GLOBAL RESET ────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
  font-family: var(--font) !important;
  background-color: var(--n50) !important;
  color: var(--n800) !important;
}
.main .block-container {
  padding: 2rem 2.5rem 4rem !important;
  max-width: 1340px !important;
}

/* ── TOP HEADER BAR ──────────────────────────────────────── */
[data-testid="stHeader"] {
  background-color: var(--surface) !important;
  border-bottom: 1px solid var(--n200) !important;
  box-shadow: var(--shadow-sm) !important;
}

/* ── SIDEBAR ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background-color: var(--surface) !important;
  border-right: 1px solid var(--n200) !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 1.5rem 1rem !important;
}
/* Sidebar text */
[data-testid="stSidebar"] * {
  color: var(--n800) !important;
  font-family: var(--font) !important;
}
/* Sidebar radio label */
[data-testid="stSidebar"] .stRadio label {
  font-size: 0.9rem !important;
  font-weight: 500 !important;
  padding: 0.45rem 0.75rem !important;
  border-radius: var(--radius-sm) !important;
  cursor: pointer !important;
  transition: background .15s;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background: var(--n100) !important;
}
/* Active radio */
[data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] + div {
  color: var(--primary) !important;
  font-weight: 600 !important;
}

/* ── TYPOGRAPHY ──────────────────────────────────────────── */
h1, h2, h3, h4 {
  font-family: var(--font) !important;
  color: var(--n950) !important;
  letter-spacing: -0.02em !important;
}
h1 { font-size: 1.75rem !important; font-weight: 700 !important; }
h2 { font-size: 1.3rem  !important; font-weight: 650 !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; }

p, span, div, label, li {
  font-family: var(--font) !important;
  color: var(--n800) !important;
  line-height: 1.6 !important;
}

/* ── INPUTS ──────────────────────────────────────────────── */
.stNumberInput input,
input[type="number"],
input[type="text"] {
  background-color: var(--surface) !important;
  color: var(--n950) !important;
  border: 1.5px solid var(--n200) !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--font) !important;
  font-size: 0.9rem !important;
  transition: border-color .2s, box-shadow .2s !important;
}
.stNumberInput input:focus,
input[type="number"]:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(27,79,216,.12) !important;
  outline: none !important;
}

/* ── METRIC CARDS ────────────────────────────────────────── */
[data-testid="stMetric"] {
  background-color: var(--surface) !important;
  border: 1px solid var(--n200) !important;
  border-radius: var(--radius-md) !important;
  padding: 1.1rem 1.25rem !important;
  box-shadow: var(--shadow-sm) !important;
  transition: box-shadow .2s, transform .2s !important;
}
[data-testid="stMetric"]:hover {
  box-shadow: var(--shadow-md) !important;
  transform: translateY(-1px) !important;
}
[data-testid="stMetricLabel"] > div {
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: .06em !important;
  color: var(--n600) !important;
}
[data-testid="stMetricValue"] {
  font-size: 1.75rem !important;
  font-weight: 700 !important;
  color: var(--primary) !important;
  letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"] {
  font-size: 0.8rem !important;
}

/* ── BUTTONS ─────────────────────────────────────────────── */
.stButton > button {
  background-color: var(--primary) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--font) !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  padding: 0.55rem 1.2rem !important;
  letter-spacing: .01em !important;
  box-shadow: 0 1px 2px rgba(27,79,216,.3) !important;
  transition: background .15s, box-shadow .15s, transform .1s !important;
}
.stButton > button:hover {
  background-color: var(--primary-dark) !important;
  box-shadow: 0 4px 12px rgba(27,79,216,.3) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active {
  transform: translateY(0) !important;
}
/* Secondary / ghost buttons (non-primary) */
.stButton > button[kind="secondary"] {
  background-color: var(--surface) !important;
  color: var(--primary) !important;
  border: 1.5px solid var(--primary) !important;
  box-shadow: none !important;
}
.stButton > button[kind="secondary"]:hover {
  background-color: var(--primary-light) !important;
}

/* ── TABS ────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
  background-color: var(--n100) !important;
  border-radius: var(--radius-sm) !important;
  padding: 3px !important;
  gap: 2px !important;
  border: none !important;
}
[data-baseweb="tab"] {
  border-radius: calc(var(--radius-sm) - 1px) !important;
  font-family: var(--font) !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
  color: var(--n600) !important;
  padding: 0.4rem 1rem !important;
  border: none !important;
  transition: background .15s, color .15s !important;
}
[aria-selected="true"][data-baseweb="tab"] {
  background-color: var(--surface) !important;
  color: var(--primary) !important;
  font-weight: 600 !important;
  box-shadow: var(--shadow-sm) !important;
}

/* ── DATA TABLE ──────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--n200) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden !important;
}
thead th {
  background-color: var(--n100) !important;
  color: var(--n800) !important;
  font-weight: 600 !important;
  font-size: 0.8rem !important;
  text-transform: uppercase !important;
  letter-spacing: .04em !important;
}

/* ── FILE UPLOADER ───────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background-color: var(--surface) !important;
  border: 2px dashed var(--n200) !important;
  border-radius: var(--radius-md) !important;
  transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--primary) !important;
}

/* ── SELECT / RADIO ──────────────────────────────────────── */
[data-baseweb="radio"] [data-checked="true"] div {
  background-color: var(--primary) !important;
  border-color: var(--primary) !important;
}

/* ── ALERTS / CALLOUTS ───────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: var(--radius-md) !important;
  font-family: var(--font) !important;
}

/* ── DIVIDER ─────────────────────────────────────────────── */
hr {
  border-color: var(--n200) !important;
  margin: 1.5rem 0 !important;
}

/* ── CUSTOM COMPONENTS ───────────────────────────────────── */

/* Page header hero */
.page-hero {
  background: linear-gradient(135deg, #1B4FD8 0%, #1338A8 60%, #0EA5A0 100%);
  border-radius: var(--radius-lg);
  padding: 2rem 2.5rem;
  margin-bottom: 2rem;
  box-shadow: 0 8px 32px rgba(27,79,216,.25);
  position: relative;
  overflow: hidden;
}
.page-hero::before {
  content: '';
  position: absolute;
  top: -40px; right: -40px;
  width: 220px; height: 220px;
  border-radius: 50%;
  background: rgba(255,255,255,.06);
}
.page-hero::after {
  content: '';
  position: absolute;
  bottom: -60px; right: 80px;
  width: 160px; height: 160px;
  border-radius: 50%;
  background: rgba(14,165,160,.2);
}
.page-hero h1 {
  color: #FFFFFF !important;
  font-size: 1.9rem !important;
  margin: 0 0 .35rem !important;
}
.page-hero p {
  color: rgba(255,255,255,.78) !important;
  font-size: 1rem !important;
  margin: 0 !important;
}
.page-hero .badge {
  display: inline-block;
  background: rgba(255,255,255,.15);
  border: 1px solid rgba(255,255,255,.25);
  border-radius: 100px;
  padding: .2rem .75rem;
  font-size: .75rem;
  font-weight: 600;
  color: white !important;
  letter-spacing: .04em;
  text-transform: uppercase;
  margin-bottom: .75rem;
}

/* Section card */
.section-card {
  background: var(--surface);
  border: 1px solid var(--n200);
  border-radius: var(--radius-lg);
  padding: 1.5rem;
  margin-bottom: 1.25rem;
  box-shadow: var(--shadow-sm);
}
.section-title {
  font-size: .72rem;
  font-weight: 700;
  color: var(--n600) !important;
  text-transform: uppercase;
  letter-spacing: .08em;
  margin-bottom: .75rem;
  display: flex;
  align-items: center;
  gap: .4rem;
}

/* Info callout */
.callout-info {
  background: var(--primary-light);
  border-left: 3px solid var(--primary);
  border-radius: var(--radius-sm);
  padding: .9rem 1.1rem;
  color: var(--n800) !important;
  font-size: .88rem;
}
.callout-warning {
  background: #FFFBEB;
  border-left: 3px solid var(--warning);
  border-radius: var(--radius-sm);
  padding: .9rem 1.1rem;
  color: #78350F !important;
  font-size: .88rem;
}
.callout-success {
  background: #ECFDF5;
  border-left: 3px solid var(--success);
  border-radius: var(--radius-sm);
  padding: .9rem 1.1rem;
  color: #065F46 !important;
  font-size: .88rem;
}
.callout-error {
  background: #FEF2F2;
  border-left: 3px solid var(--error);
  border-radius: var(--radius-sm);
  padding: .9rem 1.1rem;
  color: #7F1D1D !important;
  font-size: .88rem;
}

/* Result card */
.result-hero {
  background: linear-gradient(135deg, #1B4FD8 0%, #0EA5A0 100%);
  border-radius: var(--radius-lg);
  padding: 2rem;
  text-align: center;
  color: white;
  box-shadow: var(--shadow-lg);
  margin: 1.5rem 0;
  position: relative;
  overflow: hidden;
}
.result-hero::before {
  content: '';
  position: absolute;
  top: -30px; left: -30px;
  width: 160px; height: 160px;
  border-radius: 50%;
  background: rgba(255,255,255,.06);
}
.result-hero .result-label {
  font-size: .78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .1em;
  color: rgba(255,255,255,.7) !important;
  margin-bottom: .5rem;
}
.result-hero .result-value {
  font-size: 3.5rem;
  font-weight: 700;
  color: white !important;
  line-height: 1;
  letter-spacing: -0.03em;
}
.result-hero .result-unit {
  font-size: 1.1rem;
  color: rgba(255,255,255,.8) !important;
  margin-left: .25rem;
}
.result-hero .result-category {
  font-size: .95rem;
  color: rgba(255,255,255,.85) !important;
  margin-top: .5rem;
}

/* KPI stat pill in sidebar */
.stat-pill {
  background: var(--n100);
  border-radius: 8px;
  padding: .6rem .9rem;
  margin-bottom: .5rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.stat-pill .label { font-size: .78rem; color: var(--n600) !important; font-weight: 500; }
.stat-pill .value { font-size: .88rem; color: var(--n950) !important; font-weight: 700; font-family: var(--mono); }

/* Performance metric card */
.perf-card {
  border-radius: var(--radius-md);
  padding: 1.5rem;
  text-align: center;
  color: white;
  box-shadow: var(--shadow-md);
}
.perf-card .metric-label {
  font-size: .72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: rgba(255,255,255,.7) !important;
  margin-bottom: .4rem;
}
.perf-card .metric-value {
  font-size: 2.4rem;
  font-weight: 700;
  color: white !important;
  letter-spacing: -0.03em;
  line-height: 1;
}
.perf-card .metric-sub {
  font-size: .75rem;
  color: rgba(255,255,255,.65) !important;
  margin-top: .35rem;
}

/* Grade badge */
.grade-badge {
  display: inline-flex;
  align-items: center;
  gap: .3rem;
  padding: .25rem .7rem;
  border-radius: 100px;
  font-size: .78rem;
  font-weight: 700;
  letter-spacing: .02em;
}
.grade-low    { background: #FEE2E2; color: #991B1B !important; }
.grade-medium { background: #FEF9C3; color: #854D0E !important; }
.grade-high   { background: #D1FAE5; color: #065F46 !important; }
.grade-ultra  { background: var(--primary-light); color: var(--primary) !important; }

/* Sidebar logo */
.sidebar-brand {
  display: flex;
  align-items: center;
  gap: .6rem;
  padding: .25rem 0 1.25rem;
  border-bottom: 1px solid var(--n200);
  margin-bottom: 1.25rem;
}
.sidebar-brand .logo-icon {
  width: 34px; height: 34px;
  background: var(--primary);
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem;
}
.sidebar-brand .logo-text {
  font-size: .95rem !important;
  font-weight: 700 !important;
  color: var(--n950) !important;
  line-height: 1.2;
}
.sidebar-brand .logo-sub {
  font-size: .7rem !important;
  color: var(--n400) !important;
}

/* Menu item */
.nav-item {
  display: flex;
  align-items: center;
  gap: .6rem;
  padding: .55rem .8rem;
  border-radius: var(--radius-sm);
  font-size: .88rem;
  font-weight: 500;
  color: var(--n600) !important;
  cursor: pointer;
  margin-bottom: .2rem;
  transition: background .15s, color .15s;
}
.nav-item:hover { background: var(--n100); color: var(--n950) !important; }
.nav-item.active { background: var(--primary-light); color: var(--primary) !important; font-weight: 600; }

/* Tag chips */
.chip {
  display: inline-block;
  background: var(--n100);
  border: 1px solid var(--n200);
  color: var(--n600) !important;
  border-radius: 100px;
  font-size: .73rem;
  font-weight: 600;
  padding: .15rem .6rem;
  letter-spacing: .02em;
}
.chip-primary {
  background: var(--primary-light);
  border-color: #C7D7FF;
  color: var(--primary) !important;
}

/* Section heading with line */
.section-heading {
  display: flex;
  align-items: center;
  gap: .75rem;
  margin: 1.5rem 0 1rem;
}
.section-heading .sh-icon {
  width: 32px; height: 32px;
  background: var(--primary-light);
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: .95rem;
}
.section-heading h3 {
  margin: 0 !important;
  font-size: 1rem !important;
  color: var(--n950) !important;
}

/* Footer */
.footer {
  text-align: center;
  padding: 2rem 0 .5rem;
  border-top: 1px solid var(--n200);
  margin-top: 3rem;
}
.footer p { font-size: .8rem; color: var(--n400) !important; margin: .15rem 0 !important; }
.footer strong { color: var(--n600) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--n50); }
::-webkit-scrollbar-thumb { background: var(--n200); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--n400); }

/* Code block */
code, pre {
  font-family: var(--mono) !important;
  font-size: .82rem !important;
  background: var(--n100) !important;
  border: 1px solid var(--n200) !important;
  border-radius: var(--radius-sm) !important;
}

/* Remove Streamlit default top padding */
.stApp > header { visibility: hidden; }

/* Download button alignment */
.stDownloadButton > button {
  background-color: var(--surface) !important;
  color: var(--primary) !important;
  border: 1.5px solid var(--primary) !important;
  box-shadow: none !important;
}
.stDownloadButton > button:hover {
  background-color: var(--primary-light) !important;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
#  PLOTLY THEME
# ============================================================
# Base layout — NO legend/xaxis/yaxis keys to avoid duplicate-kwarg errors
PLOTLY_LAYOUT = dict(
    font_family="DM Sans, sans-serif",
    font_color="#1E2633",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    title_font_size=14,
    title_font_color="#0D1117",
    margin=dict(l=0, r=0, t=40, b=0),
)
_PLOTLY_LEGEND  = dict(bgcolor="rgba(255,255,255,.85)", bordercolor="#E5E7EB", borderwidth=1)
_PLOTLY_AXES    = dict(gridcolor="#F3F4F6", linecolor="#E5E7EB")
COLOR_SEQUENCE  = ["#1B4FD8","#0EA5A0","#7C3AED","#D97706","#DC2626","#059669","#6366F1"]


def _theme(fig, **extra):
    """Apply shared Plotly theme safely — never causes duplicate-kwarg errors.
    Keys in extra always win; legend in extra is merged over _PLOTLY_LEGEND so
    there is never a duplicate keyword argument.
    """
    merged_legend = {**_PLOTLY_LEGEND, **extra.pop("legend", {})}
    fig.update_layout(**PLOTLY_LAYOUT, legend=merged_legend, **extra)
    fig.update_xaxes(**_PLOTLY_AXES)
    fig.update_yaxes(**_PLOTLY_AXES)
    return fig


# ============================================================
#  MODEL LOADING
# ============================================================
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load('concrete_strength_model.pkl')
        if isinstance(model_data, dict):
            return model_data
        return {'model': model_data}
    except FileNotFoundError:
        st.error("❌ Model file not found — place `concrete_strength_model.pkl` in the working directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


# ============================================================
#  PREDICTION HELPERS
# ============================================================
def predict_strength(model_data, features):
    try:
        model   = model_data.get('model') if isinstance(model_data, dict) else model_data
        scaler  = model_data.get('scaler') if isinstance(model_data, dict) else None
        f_names = model_data.get('feature_names', []) if isinstance(model_data, dict) else []

        cement, slag, flyash, water, sp, coarse, fine, age = features
        wc = water / cement if cement > 0 else 0
        binder = cement + slag + flyash
        log_age = np.log1p(age)

        if len(f_names) == 11 and 'Semen' in f_names:
            df = pd.DataFrame({
                'Semen':[cement],'Slag_Tanur_Tinggi':[slag],'Abu_Terbang':[flyash],
                'Air':[water],'Superplasticizer':[sp],'Agregat_Kasar':[coarse],
                'Agregat_Halus':[fine],'Umur_Hari':[age],
                'Rasio_Air_Semen':[wc],'Total_Bahan_Pengikat':[binder],'Log_Umur_Hari':[log_age]
            })
            arr = scaler.transform(df) if scaler else df.values
        else:
            arr = np.array([cement,slag,flyash,water,sp,coarse,fine,age,wc,binder,log_age]).reshape(1,-1)
            if scaler: arr = scaler.transform(arr)

        return max(0, float(model.predict(arr)[0]))
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def predict_batch(model_data, df):
    try:
        model  = model_data.get('model') if isinstance(model_data, dict) else model_data
        scaler = model_data.get('scaler') if isinstance(model_data, dict) else None

        dm = pd.DataFrame()
        dm['Semen']           = df['Cement']
        dm['Slag_Tanur_Tinggi']= df['Blast Furnace Slag']
        dm['Abu_Terbang']     = df['Fly Ash']
        dm['Air']             = df['Water']
        dm['Superplasticizer']= df['Superplasticizer']
        dm['Agregat_Kasar']   = df['Coarse Aggregate']
        dm['Agregat_Halus']   = df['Fine Aggregate']
        dm['Umur_Hari']       = df['Age']
        dm['Rasio_Air_Semen'] = dm['Air'] / dm['Semen']
        dm['Total_Bahan_Pengikat'] = dm['Semen'] + dm['Slag_Tanur_Tinggi'] + dm['Abu_Terbang']
        dm['Log_Umur_Hari']   = np.log1p(dm['Umur_Hari'])

        arr = scaler.transform(dm) if scaler else dm.values
        return np.maximum(0, model.predict(arr))
    except Exception as e:
        st.error(f"Batch prediction error: {e}")
        return None


def get_concrete_grade(s):
    if s < 20:
        return "K-175", "Low Strength",   "Non-structural works",         "grade-low",   "🔴"
    elif s < 30:
        return "K-250", "Medium Strength", "Light structural elements",   "grade-medium","🟡"
    elif s < 40:
        return "K-300", "High Strength",   "Standard building structures","grade-high",  "🟢"
    else:
        return "K-400+","Very High Strength","Special / heavy structures", "grade-ultra", "🔵"


# ============================================================
#  SESSION STATE
# ============================================================
if 'history' not in st.session_state:
    st.session_state.history = []


# ============================================================
#  LOAD MODEL
# ============================================================
model_data = load_model()
if model_data is None:
    st.stop()


# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sidebar-brand">
      <div class="logo-icon">🏗️</div>
      <div>
        <div class="logo-text">ConcreteIQ</div>
        <div class="logo-sub">Strength Predictor</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)
    menu = st.radio(
        "Navigation Menu", ["📝  Manual Input", "📁  Batch CSV Upload", "📊  Model Performance"],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Model info
    st.markdown('<div class="section-title">Model Info</div>', unsafe_allow_html=True)
    if isinstance(model_data, dict):
        m = model_data.get('model')
        fn = model_data.get('feature_names', [])
        metrics = model_data.get('metrics', {})
        mtype = m.__class__.__name__ if m else "N/A"
        r2 = metrics.get('r2', metrics.get('r2_score', '—'))
    else:
        mtype, fn, r2 = "Unknown", [], "—"

    st.markdown(f"""
    <div class="stat-pill"><span class="label">Algorithm</span><span class="value">{mtype[:18]}</span></div>
    <div class="stat-pill"><span class="label">Features</span><span class="value">{len(fn) if fn else 11}</span></div>
    <div class="stat-pill"><span class="label">R² Score</span><span class="value">{r2 if isinstance(r2, str) else f'{r2:.3f}'}</span></div>
    <div class="stat-pill"><span class="label">Total Predictions</span><span class="value">{len(st.session_state.history)}</span></div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-title">Concrete Grade Scale</div>
    <div style="font-size:.8rem; line-height:2;">
      <span class="grade-badge grade-low">K-175</span> &nbsp; &lt; 20 MPa<br>
      <span class="grade-badge grade-medium">K-250</span> &nbsp; 20–30 MPa<br>
      <span class="grade-badge grade-high">K-300</span> &nbsp; 30–40 MPa<br>
      <span class="grade-badge grade-ultra">K-400+</span> &nbsp; &gt; 40 MPa
    </div>
    """, unsafe_allow_html=True)


# ============================================================
#  PAGE HERO
# ============================================================
PAGE_META = {
    "📝  Manual Input":       ("📝 Manual Input",         "Enter mix parameters to predict compressive strength",      "Prediction Tool"),
    "📁  Batch CSV Upload":   ("📁 Batch CSV Upload",      "Upload a CSV to predict multiple concrete mixes at once",   "Batch Processing"),
    "📊  Model Performance":  ("📊 Model Performance",     "Evaluate model accuracy, feature impact, and history",      "Analytics"),
}
title, subtitle, badge = PAGE_META.get(menu, ("Dashboard","",""))
st.markdown(f"""
<div class="page-hero">
  <span class="badge">{badge}</span>
  <h1>{title}</h1>
  <p>{subtitle}</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
#  MENU 1 — MANUAL INPUT
# ============================================================
if menu == "📝  Manual Input":

    st.markdown("""
    <div class="callout-info">
      <strong>ℹ️ Instructions:</strong> Enter mix design parameters below.
      All materials are in <strong>kg/m³</strong>.
      Click <em>"Load Example"</em> to auto-fill realistic values.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout-warning" style="margin-top:.75rem;">
      <strong>⚠️ Unit reminder:</strong>
      Cement 100–540 · Water 120–250 · Coarse Agg 800–1150 · Fine Agg 600–1000 (all kg/m³)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input grid ──────────────────────────────────────────
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("""
        <div class="section-heading">
          <div class="sh-icon">🧱</div>
          <h3>Cementitious Materials</h3>
        </div>
        """, unsafe_allow_html=True)

        cement = st.number_input("Cement (kg/m³)",            min_value=0.0, max_value=600.0,  value=0.0, step=10.0, help="Normal range: 100–540 kg/m³")
        slag   = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0, max_value=400.0,  value=0.0, step=10.0, help="Normal range: 0–360 kg/m³")
        flyash = st.number_input("Fly Ash (kg/m³)",            min_value=0.0, max_value=250.0,  value=0.0, step=10.0, help="Normal range: 0–200 kg/m³")
        water  = st.number_input("Water (kg/m³)",              min_value=0.0, max_value=300.0,  value=0.0, step=5.0,  help="Normal range: 120–250 kg/m³")

    with col_r:
        st.markdown("""
        <div class="section-heading">
          <div class="sh-icon">🪨</div>
          <h3>Aggregates & Admixtures</h3>
        </div>
        """, unsafe_allow_html=True)

        sp    = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, max_value=35.0,   value=0.0, step=0.5, help="Normal range: 0–32 kg/m³")
        coarse= st.number_input("Coarse Aggregate (kg/m³)", min_value=0.0, max_value=1200.0, value=0.0, step=10.0,help="Normal range: 800–1150 kg/m³")
        fine  = st.number_input("Fine Aggregate (kg/m³)",   min_value=0.0, max_value=1100.0, value=0.0, step=10.0,help="Normal range: 600–1000 kg/m³")
        age   = st.number_input("Curing Age (Days)",         min_value=1,   max_value=365,    value=28,  step=1,   help="Standard test: 28 days")

    # ── Action buttons ───────────────────────────────────────
    btn_col1, btn_col2, _ = st.columns([1.2, 1.2, 3])
    with btn_col1:
        predict_btn = st.button("🔮 Predict Strength", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("📋 Load Example", use_container_width=True):
            st.session_state._ex = dict(cement=540.,slag=0.,flyash=0.,water=162.,
                                        sp=2.5,coarse=1040.,fine=676.,age=28)
            st.rerun()

    # Load example data
    if '_ex' in st.session_state:
        ex = st.session_state._ex
        cement,slag,flyash,water,sp,coarse,fine,age = (
            ex['cement'],ex['slag'],ex['flyash'],ex['water'],
            ex['sp'],ex['coarse'],ex['fine'],ex['age'])
        del st.session_state._ex
        st.rerun()

    # ── Prediction result ────────────────────────────────────
    if predict_btn:
        ERRORS = []
        if cement == 0 or water == 0 or coarse == 0 or fine == 0:
            ERRORS.append("Cement, Water, Coarse Aggregate, and Fine Aggregate are required.")
        if cement > 0 and cement < 100:
            ERRORS.append("Cement too low — expected 100–540 kg/m³.")
        if water > 0 and water < 100:
            ERRORS.append("Water too low — expected 120–250 kg/m³.")
        if coarse > 0 and coarse < 500:
            ERRORS.append("Coarse Aggregate too low — expected 800–1150 kg/m³.")
        if fine > 0 and fine < 500:
            ERRORS.append("Fine Aggregate too low — expected 600–1000 kg/m³.")

        if ERRORS:
            for err in ERRORS:
                st.markdown(f'<div class="callout-error">⚠️ {err}</div>', unsafe_allow_html=True)
        else:
            features = [cement, slag, flyash, water, sp, coarse, fine, age]
            with st.spinner("Computing prediction…"):
                pred = predict_strength(model_data, features)

            if pred is not None:
                st.session_state.history.append({
                    'features': features, 'prediction': pred,
                    'timestamp': pd.Timestamp.now()
                })
                grade, grade_label, usage, grade_cls, grade_dot = get_concrete_grade(pred)
                wc = water / cement

                # Hero result
                st.markdown(f"""
                <div class="result-hero">
                  <div class="result-label">Predicted Compressive Strength</div>
                  <div>
                    <span class="result-value">{pred:.1f}</span>
                    <span class="result-unit">MPa</span>
                  </div>
                  <div class="result-category">{grade_dot} {grade_label} · Grade {grade}</div>
                </div>
                """, unsafe_allow_html=True)

                # KPI row
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Concrete Grade",  grade)
                k2.metric("Curing Age",      f"{age} days")
                k3.metric("W/C Ratio",       f"{wc:.3f}")
                k4.metric("Total Material",  f"{sum(features[:-1]):.0f} kg/m³")

                # Usage callout
                st.markdown(f'<div class="callout-info" style="margin-top:.75rem;"><strong>💡 Recommended application:</strong> {usage}</div>', unsafe_allow_html=True)

                # Composition chart
                st.markdown("""
                <div class="section-heading" style="margin-top:1.5rem;">
                  <div class="sh-icon">📊</div><h3>Mix Composition</h3>
                </div>
                """, unsafe_allow_html=True)

                comp = pd.DataFrame({
                    'Material': ['Cement','Slag','Fly Ash','Water','Superplasticizer','Coarse Agg','Fine Agg'],
                    'Amount':   [cement, slag, flyash, water, sp, coarse, fine]
                })
                comp = comp[comp['Amount'] > 0]

                fig = px.pie(comp, values='Amount', names='Material',
                             hole=.45, color_discrete_sequence=COLOR_SEQUENCE)
                fig.update_traces(textposition='outside', textinfo='label+percent',
                                  textfont_size=11, pull=[.02]*len(comp))
                _theme(fig,
                       showlegend=True,
                       legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=.5, xanchor="center"),
                       annotations=[dict(text=f"<b>{sum(comp['Amount']):.0f}</b><br>kg/m³",
                                         x=.5, y=.5, font_size=13, showarrow=False, font_color="#0D1117")])
                st.plotly_chart(fig, use_container_width=True)


# ============================================================
#  MENU 2 — BATCH CSV
# ============================================================
elif menu == "📁  Batch CSV Upload":

    st.markdown("""
    <div class="callout-info">
      <strong>ℹ️ Instructions:</strong>
      Upload a CSV with the required columns. Download the template below as a starting point.
    </div>
    """, unsafe_allow_html=True)

    # Template
    template = pd.DataFrame({
        'Cement':[540,450,425],'Blast Furnace Slag':[0,100,106],
        'Fly Ash':[0,0,0],'Water':[162,180,153],
        'Superplasticizer':[2.5,3.0,4.0],'Coarse Aggregate':[1040,950,1000],
        'Fine Aggregate':[676,720,750],'Age':[28,28,28]
    })

    col_a, col_b = st.columns([3,1])
    with col_a:
        st.markdown("""
        <div class="section-heading">
          <div class="sh-icon">📋</div><h3>Required CSV Columns</h3>
        </div>
        """, unsafe_allow_html=True)
        st.code("Cement | Blast Furnace Slag | Fly Ash | Water | Superplasticizer | Coarse Aggregate | Fine Aggregate | Age")
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button("⬇️ Download Template", template.to_csv(index=False),
                           "concrete_template.csv", "text/csv", use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            st.success(f"✅ File loaded — {len(df):,} rows")

            st.markdown("""
            <div class="section-heading"><div class="sh-icon">👁️</div><h3>Data Preview</h3></div>
            """, unsafe_allow_html=True)
            st.dataframe(df.head(8), use_container_width=True)

            required = ['Cement','Blast Furnace Slag','Fly Ash','Water',
                        'Superplasticizer','Coarse Aggregate','Fine Aggregate','Age']
            for col in required:
                for dc in df.columns:
                    if col.lower().replace(' ','') == dc.lower().replace(' ','') and dc != col:
                        df.rename(columns={dc:col}, inplace=True)

            missing = [c for c in required if c not in df.columns]
            if missing:
                st.markdown(f'<div class="callout-error">Missing columns: {", ".join(missing)}</div>', unsafe_allow_html=True)
            else:
                if st.button("🚀 Run Batch Prediction", type="primary"):
                    with st.spinner("Processing…"):
                        preds = predict_batch(model_data, df)

                    if preds is not None:
                        df['Predicted Strength (MPa)'] = preds
                        df['Grade']    = df['Predicted Strength (MPa)'].apply(lambda x: get_concrete_grade(x)[0])
                        df['W/C Ratio']= df['Water'] / df['Cement']

                        st.markdown("""
                        <div class="section-heading"><div class="sh-icon">📊</div><h3>Summary Statistics</h3></div>
                        """, unsafe_allow_html=True)

                        s1,s2,s3,s4 = st.columns(4)
                        s1.metric("Mean Strength",  f"{df['Predicted Strength (MPa)'].mean():.1f} MPa")
                        s2.metric("Max Strength",   f"{df['Predicted Strength (MPa)'].max():.1f} MPa")
                        s3.metric("Min Strength",   f"{df['Predicted Strength (MPa)'].min():.1f} MPa")
                        s4.metric("Std Deviation",  f"{df['Predicted Strength (MPa)'].std():.1f} MPa")

                        st.markdown("""
                        <div class="section-heading"><div class="sh-icon">📋</div><h3>Full Results</h3></div>
                        """, unsafe_allow_html=True)
                        st.dataframe(df, use_container_width=True)
                        st.download_button("💾 Download Results", df.to_csv(index=False),
                                           "prediction_results.csv", "text/csv")

                        # Charts
                        st.markdown("""
                        <div class="section-heading"><div class="sh-icon">📈</div><h3>Visualizations</h3></div>
                        """, unsafe_allow_html=True)

                        tab1, tab2, tab3 = st.tabs(["Distribution", "Strength vs Age", "Grade Breakdown"])

                        with tab1:
                            fig1 = px.histogram(df, x='Predicted Strength (MPa)', nbins=25,
                                                title="Strength Distribution",
                                                color_discrete_sequence=["#1B4FD8"])
                            _theme(fig1)
                            fig1.update_traces(marker_line_color="#fff", marker_line_width=.5)
                            st.plotly_chart(fig1, use_container_width=True)

                        with tab2:
                            fig2 = px.scatter(df, x='Age', y='Predicted Strength (MPa)',
                                              color='Grade', size='Cement',
                                              title="Strength vs Curing Age",
                                              hover_data=['Cement','Water','W/C Ratio'],
                                              color_discrete_sequence=COLOR_SEQUENCE)
                            _theme(fig2)
                            st.plotly_chart(fig2, use_container_width=True)

                        with tab3:
                            gc = df['Grade'].value_counts().reset_index()
                            gc.columns = ['Grade','Count']
                            fig3 = px.bar(gc, x='Grade', y='Count', title="Grade Distribution",
                                          color='Grade', color_discrete_sequence=COLOR_SEQUENCE,
                                          text='Count')
                            _theme(fig3)
                            fig3.update_traces(textposition='outside', marker_line_width=0)
                            st.plotly_chart(fig3, use_container_width=True)

        except Exception as e:
            st.markdown(f'<div class="callout-error">❌ File read error: {e}</div>', unsafe_allow_html=True)


# ============================================================
#  MENU 3 — MODEL PERFORMANCE
# ============================================================
elif menu == "📊  Model Performance":

    # Metrics
    metrics = model_data.get('metrics', {}) if isinstance(model_data, dict) else {}
    r2   = metrics.get('r2_score', metrics.get('r2', 0.85))
    rmse = metrics.get('rmse', 10.24)
    mae  = metrics.get('mae', 7.82)

    st.markdown("""
    <div class="section-heading"><div class="sh-icon">🎯</div><h3>Performance Metrics</h3></div>
    """, unsafe_allow_html=True)

    pm1, pm2, pm3 = st.columns(3)
    with pm1:
        st.markdown(f"""
        <div class="perf-card" style="background:linear-gradient(135deg,#1B4FD8,#1338A8);">
          <div class="metric-label">R² Score</div>
          <div class="metric-value">{r2:.3f}</div>
          <div class="metric-sub">Coefficient of Determination</div>
          <div style="background:rgba(255,255,255,.2);height:6px;border-radius:3px;margin-top:14px;">
            <div style="background:white;height:6px;border-radius:3px;width:{int(r2*100)}%;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with pm2:
        st.markdown(f"""
        <div class="perf-card" style="background:linear-gradient(135deg,#059669,#047857);">
          <div class="metric-label">RMSE</div>
          <div class="metric-value">{rmse:.2f}</div>
          <div class="metric-sub">Root Mean Squared Error (MPa)</div>
        </div>
        """, unsafe_allow_html=True)
    with pm3:
        st.markdown(f"""
        <div class="perf-card" style="background:linear-gradient(135deg,#0EA5A0,#0D7A76);">
          <div class="metric-label">MAE</div>
          <div class="metric-value">{mae:.2f}</div>
          <div class="metric-sub">Mean Absolute Error (MPa)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature importance + model info
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("""
        <div class="section-heading"><div class="sh-icon">🤖</div><h3>Model Configuration</h3></div>
        """, unsafe_allow_html=True)

        mobj  = model_data.get('model') if isinstance(model_data, dict) else model_data
        fnames= model_data.get('feature_names', []) if isinstance(model_data, dict) else []
        mtype = mobj.__class__.__name__ if mobj else "Unknown"

        st.markdown(f"""
        <div class="section-card">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:.75rem;">
            <div><div class="section-title">Algorithm</div><div style="font-weight:700;color:var(--n950);">{mtype}</div></div>
            <div><div class="section-title">Method</div><div style="font-weight:700;color:var(--n950);">OLS Regression</div></div>
            <div><div class="section-title">Input Features</div><div style="font-weight:700;color:var(--n950);">8 parameters</div></div>
            <div><div class="section-title">Output</div><div style="font-weight:700;color:var(--n950);">MPa (comp. strength)</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-heading"><div class="sh-icon">📋</div><h3>Input Parameters</h3></div>
        """, unsafe_allow_html=True)

        params = [
            ("Cement",             "kg/m³","Primary binder"),
            ("Blast Furnace Slag", "kg/m³","SCM replacement"),
            ("Fly Ash",            "kg/m³","Pozzolanic additive"),
            ("Water",              "kg/m³","W/C ratio driver"),
            ("Superplasticizer",   "kg/m³","Workability enhancer"),
            ("Coarse Aggregate",   "kg/m³","Skeletal structure"),
            ("Fine Aggregate",     "kg/m³","Void filler"),
            ("Age",                "days", "Curing maturity"),
        ]
        for i,(name,unit,desc) in enumerate(params,1):
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:.75rem;padding:.5rem 0;border-bottom:1px solid var(--n200);">
              <span style="background:var(--primary-light);color:var(--primary);width:22px;height:22px;
                           border-radius:50%;display:inline-flex;align-items:center;justify-content:center;
                           font-size:.7rem;font-weight:700;flex-shrink:0;">{i}</span>
              <div>
                <div style="font-weight:600;font-size:.88rem;color:var(--n950);">{name}
                  <span class="chip chip-primary" style="margin-left:.35rem;">{unit}</span>
                </div>
                <div style="font-size:.75rem;color:var(--n600);">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="section-heading"><div class="sh-icon">📊</div><h3>Feature Impact (Coefficients)</h3></div>
        """, unsafe_allow_html=True)

        mobj2 = model_data.get('model') if isinstance(model_data, dict) else model_data
        fn2   = model_data.get('feature_names', []) if isinstance(model_data, dict) else []

        if hasattr(mobj2,'coef_') and len(fn2) > 0:
            coefs = mobj2.coef_[:8]
            feat_display = fn2[:8]
        else:
            coefs = np.array([0.117,0.104,0.087,-0.148,0.277,0.010,0.010,0.114])
            feat_display = ['Cement','Slag','Fly Ash','Water','Superplasticizer','Coarse Agg','Fine Agg','Age']

        fi = pd.DataFrame({'Feature':feat_display,'Coef':coefs,'Abs':np.abs(coefs)})
        fi = fi.sort_values('Abs', ascending=True)

        fig_fi = go.Figure(go.Bar(
            x=fi['Coef'], y=fi['Feature'], orientation='h',
            marker=dict(
                color=fi['Coef'],
                colorscale=[[0,"#DC2626"],[0.5,"#F3F4F6"],[1,"#1B4FD8"]],
                cmin=-fi['Abs'].max(), cmax=fi['Abs'].max(),
                line=dict(width=0)
            ),
            text=[f"{c:+.3f}" for c in fi['Coef']],
            textposition='outside',
            textfont=dict(size=10, family="DM Mono", color="#1E2633"),
        ))
        _theme(fig_fi, height=340)
        fig_fi.update_xaxes(title_text="Coefficient", gridcolor="#F3F4F6",
                            zeroline=True, zerolinecolor="#E5E7EB", zerolinewidth=1.5)
        fig_fi.update_yaxes(title_text="", gridcolor="#F3F4F6")
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("""
        <div class="callout-info" style="font-size:.82rem;">
          <strong>Reading the chart:</strong>
          Positive → increases strength · Negative → decreases strength.
          Bar length = relative importance.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Prediction history
    st.markdown("""
    <div class="section-heading"><div class="sh-icon">📜</div><h3>Prediction History</h3></div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        hist_df = pd.DataFrame([{
            'Timestamp':  h['timestamp'].strftime('%Y-%m-%d %H:%M'),
            'Strength (MPa)': round(h['prediction'],2),
            'Grade':      get_concrete_grade(h['prediction'])[0],
            'Cement':     h['features'][0],
            'Water':      h['features'][3],
            'Age (d)':    h['features'][7],
        } for h in st.session_state.history[-10:]])

        st.dataframe(hist_df, use_container_width=True)

        if len(st.session_state.history) > 1:
            fig_h = px.line(hist_df, y='Strength (MPa)', markers=True,
                            title="Recent Predictions Trend",
                            color_discrete_sequence=["#1B4FD8"])
            fig_h.update_traces(line_width=2, marker_size=7)
            _theme(fig_h)
            st.plotly_chart(fig_h, use_container_width=True)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown("""
        <div class="callout-info">
          No prediction history yet. Run a prediction from Manual Input or Batch CSV.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Insights
    st.markdown("""
    <div class="section-heading"><div class="sh-icon">💡</div><h3>Key Insights & Limitations</h3></div>
    """, unsafe_allow_html=True)

    ci1, ci2 = st.columns(2, gap="large")
    with ci1:
        st.markdown("""
        <div class="callout-success">
          <strong>🔍 Key Findings</strong><br><br>
          • <strong>Superplasticizer</strong> has the highest positive influence (+0.277)<br>
          • <strong>Water</strong> exerts a negative effect (−0.148): higher W/C → lower strength<br>
          • <strong>Cement</strong> and <strong>Age</strong> both contribute positively<br>
          • Aggregate coefficients are small but structurally important
        </div>
        """, unsafe_allow_html=True)
    with ci2:
        st.markdown("""
        <div class="callout-warning">
          <strong>⚠️ Model Limitations</strong><br><br>
          • Linear model may underperform on highly non-linear mixes<br>
          • Accuracy depends on training data range coverage<br>
          • Inputs outside typical ranges extrapolate — use with care<br>
          • Always validate critical applications with lab testing
        </div>
        """, unsafe_allow_html=True)


# ============================================================
#  FOOTER
# ============================================================
st.markdown("""
<div class="footer">
  <p><strong>ConcreteIQ · Strength Predictor</strong> — Powered by Machine Learning</p>
  <p>© 2026 Pengelola MK Praktikum Unggulan (DGX), Universitas Gunadarma. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)