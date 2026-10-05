"""
GWO-Based Network Intrusion Detection System - Streamlit dashboard.

This dashboard is a demonstration layer around the EXISTING implementation in gwo_ids.py
(Binary Grey Wolf Optimization + Random Forest on NSL-KDD). It does not contain any
algorithm logic of its own: every number shown is read from files written by that
implementation (results/*.csv, results/*.json) or from team-recorded development
experiments (experiments/*.csv).

Run with:  streamlit run app.py
"""
import inspect
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import gwo_ids  # the existing IDS engine (unchanged algorithm)

RESULTS = ROOT / "results"
EXPERIMENTS = ROOT / "experiments"
DEFAULT_TRAIN = "data/KDDTrain+.arff"
DEFAULT_TEST = "data/KDDTest+.arff"

# ----------------------------------------------------------------------------- palette
NAVY = "#1f4e79"
TEAL = "#0f8b8d"
SLATE = "#8a9bb0"
AMBER = "#d98324"
INK = "#1b2733"
MUTED = "#5f6b7a"
LINE = "#e1e7ef"

st.set_page_config(
    page_title="GWO-Based Network IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input {{ font-family: 'Inter', 'Segoe UI', Roboto, Arial, sans-serif !important; color: {INK}; }}
.stApp {{ background: linear-gradient(180deg, #eef3fa 0%, #f7f9fc 320px, #f7f9fc 100%); }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1280px; }}
/* Keep Streamlit's native header/sidebar toggle available.
   Hiding stHeader makes the navigation impossible to reopen after
   the sidebar is collapsed. */
#MainMenu, footer {{
    visibility: hidden !important;
    height: 0 !important;
}}

header[data-testid="stHeader"] {{
    visibility: visible !important;
    height: 2.75rem !important;
    background: transparent !important;
    box-shadow: none !important;
    pointer-events: none !important;
}}

header[data-testid="stHeader"] button[data-testid="stSidebarCollapseButton"] {{
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}}

@media (max-width: 768px) {{
  header[data-testid="stHeader"] {{
    height: 2.75rem !important;
  }}

  header[data-testid="stHeader"] button[data-testid="stSidebarCollapseButton"] {{
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
  }}

  section[data-testid="stSidebar"] {{
    display: block;
  }}
}}

/* ---------- hero banner ---------- */
.hero {{ position: relative; overflow: hidden; border-radius: 18px; padding: 30px 36px 26px 36px; margin-bottom: 26px; color: #fff;
  background: linear-gradient(120deg, #12365c 0%, #1f4e79 48%, #0f7c86 100%);
  box-shadow: 0 14px 34px rgba(18,54,92,.28); }}
.hero:before {{ content: ""; position: absolute; right: -80px; top: -110px; width: 340px; height: 340px; border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,.18) 0%, rgba(255,255,255,0) 70%); }}
.hero:after {{ content: ""; position: absolute; right: 150px; bottom: -150px; width: 300px; height: 300px; border-radius: 50%;
  background: radial-gradient(circle, rgba(124,226,214,.22) 0%, rgba(124,226,214,0) 70%); }}
.hero .row {{ position: relative; z-index: 2; display: flex; align-items: center; gap: 22px; }}
.hero .logo {{ flex: 0 0 auto; width: 70px; height: 70px; border-radius: 18px; background: rgba(255,255,255,.14);
  border: 1px solid rgba(255,255,255,.28); display: flex; align-items: center; justify-content: center; font-size: 2.1rem; }}
.hero h1 {{ color: #fff !important; font-size: 2rem; margin: 0 0 4px 0; padding: 0; font-weight: 800; letter-spacing: -.3px; line-height: 1.15; }}
.hero .subtitle {{ color: #d6e6f5; margin: 0; font-size: 1.02rem; font-weight: 500; letter-spacing: .3px; }}
.hero .pills {{ position: relative; z-index: 2; margin-top: 18px; display: flex; flex-wrap: wrap; gap: 8px; }}
.hero .pill {{ font-size: .78rem; font-weight: 600; color: #fff; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.26);
  padding: 5px 13px; border-radius: 20px; backdrop-filter: blur(4px); }}
.hero .pill.accent {{ background: #7ee2d6; color: #0b3b45; border-color: #7ee2d6; }}

/* ---------- headings ---------- */
.page-title {{ font-size: 1.7rem; font-weight: 800; color: {NAVY}; margin: 0; letter-spacing: -.3px; }}
.page-sub {{ color: {MUTED}; margin: 2px 0 20px 0; font-size: 1rem; }}
.sec {{ display: flex; align-items: center; gap: 10px; font-size: 1.1rem; font-weight: 700; color: {INK}; margin: 30px 0 14px 0; }}
.sec:before {{ content: ""; width: 5px; height: 22px; border-radius: 3px; background: linear-gradient(180deg, {NAVY}, {TEAL}); }}

/* ---------- metric cards ---------- */
.card {{ position: relative; background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 16px 18px 14px 18px; height: 100%;
  box-shadow: 0 2px 6px rgba(31,78,121,.06), 0 8px 22px rgba(31,78,121,.06); transition: transform .18s ease, box-shadow .18s ease; overflow: hidden; }}
.card:hover {{ transform: translateY(-3px); box-shadow: 0 6px 14px rgba(31,78,121,.10), 0 16px 34px rgba(31,78,121,.12); }}
.card:before {{ content: ""; position: absolute; left: 0; top: 0; right: 0; height: 4px; background: {NAVY}; }}
.card.teal:before {{ background: linear-gradient(90deg, {TEAL}, #5cc6b8); }}
.card.amber:before {{ background: linear-gradient(90deg, {AMBER}, #f0b36a); }}
.card.slate:before {{ background: linear-gradient(90deg, {SLATE}, #b9c6d6); }}
.card:not(.teal):not(.amber):not(.slate):before {{ background: linear-gradient(90deg, {NAVY}, #4a86c0); }}
.card .ico {{ position: absolute; right: 14px; top: 16px; width: 38px; height: 38px; border-radius: 11px; display: flex; align-items: center;
  justify-content: center; font-size: 1.15rem; background: #eaf1f9; }}
.card.teal .ico {{ background: #e0f4f2; }} .card.amber .ico {{ background: #fdf0e0; }} .card.slate .ico {{ background: #eef1f5; }}
.card .lbl {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .9px; color: {MUTED}; font-weight: 700; padding-right: 46px; margin-top: 2px; }}
.card .val {{ font-size: 1.95rem; font-weight: 800; color: {NAVY}; line-height: 1.2; margin-top: 6px; letter-spacing: -.5px; }}
.card.teal .val {{ color: #0b6e70; }} .card.amber .val {{ color: #b2650f; }}
.card .sub {{ font-size: .77rem; color: {MUTED}; margin-top: 3px; line-height: 1.3; }}

/* ---------- pipeline ---------- */
.flow {{ display: flex; align-items: stretch; flex-wrap: wrap; margin: 6px 0 4px 0; }}
.flow .step {{ position: relative; flex: 1 1 140px; min-width: 140px; background: #fff; border: 1px solid {LINE}; border-radius: 14px;
  padding: 16px 12px 13px 12px; text-align: center; box-shadow: 0 2px 6px rgba(31,78,121,.06), 0 8px 22px rgba(31,78,121,.06); transition: transform .18s ease; }}
.flow .step:hover {{ transform: translateY(-3px); }}
.flow .badge {{ width: 46px; height: 46px; margin: 0 auto 8px auto; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-size: 1.3rem; background: linear-gradient(135deg, #e6eef8, #d3e2f3); border: 2px solid #fff; box-shadow: 0 0 0 2px #c9daee; }}
.flow .step.hl .badge {{ background: linear-gradient(135deg, #dff5f2, #bfe9e4); box-shadow: 0 0 0 2px #a8ddd6; }}
.flow .step.hl {{ border-color: #a8ddd6; }}
.flow .num {{ font-size: .66rem; color: {MUTED}; font-weight: 700; letter-spacing: 1.2px; }}
.flow .name {{ font-weight: 700; color: {NAVY}; font-size: .95rem; margin: 2px 0 4px 0; line-height: 1.2; }}
.flow .desc {{ font-size: .74rem; color: {MUTED}; line-height: 1.35; }}
.flow .arrow {{ align-self: center; color: {TEAL}; font-size: 1.35rem; padding: 0 5px; font-weight: 700; }}

/* ---------- notes, chips ---------- */
.note {{ background: #fff; border: 1px solid {LINE}; border-left: 5px solid {NAVY}; border-radius: 10px; padding: 13px 18px; font-size: .93rem;
  color: {INK}; margin: 12px 0; box-shadow: 0 2px 8px rgba(31,78,121,.05); line-height: 1.5; }}
.note.warn {{ border-left-color: {AMBER}; background: #fffaf3; }}
.note.ok {{ border-left-color: {TEAL}; background: #f4fbfa; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }}
.chip {{ font-size: .8rem; padding: 6px 13px; border-radius: 20px; border: 1px solid {LINE}; background: #fff; color: #8a97a8; }}
.chip.on {{ background: linear-gradient(135deg, {TEAL}, #14a3a0); border-color: {TEAL}; color: #fff; font-weight: 600; box-shadow: 0 3px 8px rgba(15,139,141,.25); }}
.srcline {{ font-size: .75rem; color: {MUTED}; margin-top: 6px; }}
.srcline code {{ background: #eaf1f9; color: {NAVY}; padding: 1px 6px; border-radius: 5px; }}
.kv td {{ padding: 7px 14px 7px 0; vertical-align: top; font-size: .92rem; border-bottom: 1px solid #f0f3f8; }}
.kv td:first-child {{ color: {MUTED}; width: 230px; font-weight: 600; }}
.footer {{ margin-top: 40px; padding-top: 14px; border-top: 1px solid {LINE}; text-align: center; color: {MUTED}; font-size: .78rem; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{ background: #ffffff; border-right: 1px solid {LINE}; box-shadow: 4px 0 22px rgba(31,78,121,.05); }}
.side-brand {{ display: flex; align-items: center; gap: 11px; margin-bottom: 14px; }}
.side-brand .lg {{ width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem;
  background: linear-gradient(135deg, {NAVY}, {TEAL}); }}
.side-brand .t1 {{ font-weight: 800; color: {NAVY}; font-size: 1rem; line-height: 1.1; }}
.side-brand .t2 {{ color: {MUTED}; font-size: .74rem; }}
section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 4px; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{ padding: 9px 12px; border-radius: 10px; width: 100%; transition: background .15s ease; cursor: pointer; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: #eef4fb; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{ background: linear-gradient(90deg, #e3eefa, #eaf7f6); box-shadow: inset 4px 0 0 {NAVY}; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{ display: none !important; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label [data-baseweb="radio"] > div:first-child {{ display: none !important; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {{ font-size: .93rem; font-weight: 500; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{ font-weight: 700; color: {NAVY}; }}

/* ---------- widgets ---------- */
div[data-testid="stButton"] button {{ border-radius: 10px; font-weight: 600; padding: .55rem 1.2rem; border: 1px solid #c9d6e6; transition: all .15s ease; }}
div[data-testid="stButton"] button:hover {{ border-color: {NAVY}; color: {NAVY}; transform: translateY(-1px); }}
div[data-testid="stButton"] button[kind="primary"] {{ background: linear-gradient(120deg, {NAVY}, #2c6aa3); border: none; color: #fff; box-shadow: 0 6px 16px rgba(31,78,121,.3); }}
div[data-testid="stButton"] button[kind="primary"]:hover {{ color: #fff; box-shadow: 0 8px 20px rgba(31,78,121,.4); }}
div[data-testid="stExpander"] {{ background: #fff; border: 1px solid {LINE}; border-radius: 12px; box-shadow: 0 2px 8px rgba(31,78,121,.05); }}
div[data-testid="stPlotlyChart"] {{ background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 8px 10px; box-shadow: 0 2px 6px rgba(31,78,121,.05), 0 8px 22px rgba(31,78,121,.05); }}
div[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(31,78,121,.05); }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers
def stretch(fn, *args, **kwargs):
    """Call a Streamlit element full-width, compatible with old and new Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def show_fig(fig):
    stretch(st.plotly_chart, fig, config={"displayModeBar": False})


def show_df(df, **kwargs):
    stretch(st.dataframe, df, hide_index=True, **kwargs)


@st.cache_data(show_spinner=False)
def _read_csv(path_str, mtime):
    return pd.read_csv(path_str)


@st.cache_data(show_spinner=False)
def _read_json(path_str, mtime):
    with open(path_str) as f:
        return json.load(f)


def load_csv(path: Path):
    return _read_csv(str(path), path.stat().st_mtime) if path.exists() else None


def load_json(path: Path):
    return _read_json(str(path), path.stat().st_mtime) if path.exists() else None


def pct(x, d=2):
    return f"{float(x) * 100:.{d}f}%"


def rel(p: Path):
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


ICONS = [("training", "🗂️"), ("test", "🧪"), ("original", "🧬"), ("selected", "🎯"), ("reduction", "📉"), ("accuracy", "✅"),
         ("precision", "🔍"), ("recall", "📡"), ("f1", "⚖️"), ("runtime", "⏱️"), ("fitness", "📈"), ("iteration", "🔁"),
         ("features", "🧬"), ("negative", "🟦"), ("positive", "🚨"), ("best", "🏁")]


def icon_for(label):
    l = label.lower()
    for k, ic in ICONS:
        if k in l:
            return ic
    return "📊"


def card(label, value, sub="", tone=""):
    return (f'<div class="card {tone}"><div class="ico">{icon_for(label)}</div><div class="lbl">{label}</div>'
            f'<div class="val">{value}</div><div class="sub">{sub}</div></div>')


def cards(items, per_row=4):
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, it in zip(cols, items[i:i + per_row]):
            col.markdown(card(*it), unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


def page_header(title, sub):
    st.markdown(f'<div class="page-title">{title}</div><div class="page-sub">{sub}</div>', unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)


def note(text, kind=""):
    st.markdown(f'<div class="note {kind}">{text}</div>', unsafe_allow_html=True)


def src(*paths):
    st.markdown('<div class="srcline">Source: ' + ", ".join(f"<code>{rel(p) if isinstance(p, Path) else p}</code>" for p in paths)
                + "</div>", unsafe_allow_html=True)


def missing(path: Path, how="Run the IDS from the <b>Run IDS</b> page (or the CLI) to generate it."):
    note(f"Result file <code>{rel(path)}</code> was not found. {how}", "warn")


def style_fig(fig, height=380):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=13, color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=15, color=NAVY), x=0.01),
        hoverlabel=dict(bgcolor="#fff", bordercolor=LINE, font_size=13),
        colorway=[NAVY, TEAL, AMBER, SLATE],
    )
    fig.update_xaxes(showline=True, linecolor=LINE, gridcolor="#eef2f7")
    fig.update_yaxes(showline=True, linecolor=LINE, gridcolor="#eef2f7")
    return fig


def first_row(df):
    return None if df is None or df.empty else df.iloc[0]


# ----------------------------------------------------------------------------- data access
GWO_RES = RESULTS / "gwo_results.csv"
BASE_RES = RESULTS / "baseline_results.csv"
GWO_FEAT = RESULTS / "gwo_selected_features.csv"
BASE_FEAT = RESULTS / "baseline_selected_features.csv"
GWO_CONV = RESULTS / "gwo_convergence.csv"
GWO_PRED = RESULTS / "gwo_predictions.csv"
BASE_PRED = RESULTS / "baseline_predictions.csv"
GWO_IMP = RESULTS / "gwo_feature_importance.csv"
GWO_META = RESULTS / "gwo_run_metadata.json"
BASE_META = RESULTS / "baseline_run_metadata.json"
SEEDS = EXPERIMENTS / "seed_experiments.csv"
ITERS = EXPERIMENTS / "iteration_experiments.csv"
REFS = EXPERIMENTS / "reference_runs.csv"

FILE_STATUS = [GWO_RES, BASE_RES, GWO_FEAT, GWO_CONV, GWO_PRED, GWO_IMP, SEEDS, ITERS]

# ----------------------------------------------------------------------------- sidebar
PAGES = [
    "1 · System Overview",
    "2 · Feature Selection",
    "3 · Performance Comparison",
    "4 · GWO Convergence",
    "5 · Experimental Stability",
    "6 · Confusion Matrix & Importance",
    "7 · Run IDS",
]

with st.sidebar:
    st.markdown('<div class="side-brand"><div class="lg">🛡️</div><div><div class="t1">GWO-Based<br>Network IDS</div>'
                '<div class="t2">Demo of an existing implementation</div></div></div>', unsafe_allow_html=True)
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"<div style='font-size:.78rem;font-weight:700;color:{MUTED};letter-spacing:.8px;'>DATA FILES</div>",
                unsafe_allow_html=True)
    dot_on = f'<span style="color:{TEAL}">●</span>'
    dot_off = '<span style="color:#c0392b">○</span>'
    rows = "".join(
        "<div style='font-size:.8rem;'>" + (dot_on if p.exists() else dot_off) + " " + p.name + "</div>"
        for p in FILE_STATUS)
    st.markdown(rows, unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:.72rem;color:{MUTED};margin-top:6px;'>● found &nbsp; ○ not generated yet</div>",
                unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Existing system: NSL-KDD → Binary GWO → Random Forest. Not the proposed/novel IDS.")

# ----------------------------------------------------------------------------- banner
st.markdown(
    '<div class="hero"><div class="row"><div class="logo">🛡️</div><div>'
    '<h1>GWO-Based Network Intrusion Detection System</h1>'
    '<p class="subtitle">NSL-KDD &nbsp;|&nbsp; Binary Grey Wolf Optimization &nbsp;|&nbsp; Random Forest</p></div></div>'
    '<div class="pills"><span class="pill accent">Existing implementation demo</span><span class="pill">41 input features</span>'
    '<span class="pill">Wrapper-based feature selection</span><span class="pill">Binary classification: Normal / Attack</span></div></div>',
    unsafe_allow_html=True)


# ============================================================================= PAGE 1
def page_overview():
    page_header("System Overview", "The existing pipeline and the headline measurements from the latest result files.")

    section("System architecture")
    steps = [
        ("NSL-KDD Dataset", "KDDTrain+ / KDDTest+ (ARFF), 41 input features", "🗄️"),
        ("Data Preprocessing", "Categorical encoding, imputation, min-max scaling", "🧹"),
        ("Binary GWO", "Wolf positions → binary feature masks", "🐺"),
        ("Feature Selection", "Best mask by validation fitness", "🎯"),
        ("Random Forest", "Trained on the selected features", "🌲"),
        ("Intrusion Classification", "Normal vs Attack on KDDTest+", "🚨"),
    ]
    html = '<div class="flow">'
    for i, (n, d, ic) in enumerate(steps):
        html += (f'<div class="step{" hl" if n in ("Binary GWO", "Feature Selection") else ""}">'
                 f'<div class="badge">{ic}</div><div class="num">STEP {i + 1}</div><div class="name">{n}</div><div class="desc">{d}</div></div>')
        if i < len(steps) - 1:
            html += '<div class="arrow">➜</div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)

    section("Key measurements (latest GWO + Random Forest run)")
    g = first_row(load_csv(GWO_RES))
    feats = load_csv(GWO_FEAT)
    meta = load_json(GWO_META)

    # Do not parse the large NSL-KDD ARFF files during dashboard startup.
    # If run metadata exists, use the recorded values. Otherwise show a
    # placeholder until an IDS run creates metadata.
    if meta:
        n_train = meta.get("training_rows", "—")
        train_sub = "working training set used by the run"
        n_test = meta.get("test_rows", "—")
    else:
        n_train = "—"
        train_sub = "run metadata not available"
        n_test = "—"

    if g is None:
        missing(GWO_RES)
        total = len(feats) if feats is not None else 41
        sel = red = acc = rec = f1 = "—"
    else:
        total = len(feats) if feats is not None else int(round(g["selected_features"] / (1 - g["feature_reduction_percent"] / 100)))
        sel = int(g["selected_features"])
        red = f"{g['feature_reduction_percent']:.2f}%"
        acc, rec, f1 = pct(g["accuracy"]), pct(g["recall"]), pct(g["f1"])

    fmt = lambda v: f"{v:,}" if isinstance(v, (int, np.integer)) else v
    cards([
        ("Training Samples", fmt(n_train), train_sub, "slate"),
        ("Test Samples", fmt(n_test), "KDDTest+ (unique rows)", "slate"),
        ("Original Features", total, "NSL-KDD input features", "slate"),
        ("Selected Features", sel, "chosen by Binary GWO", "teal"),
        ("Feature Reduction", red, "relative to all input features", "teal"),
        ("Accuracy", acc, "KDDTest+", ""),
        ("Recall", rec, "attack class", ""),
        ("F1-score", f1, "attack class", ""),
    ])
    src(GWO_RES, GWO_META if meta else "dataset files")

    if g is not None:
        note(f"The GWO-based configuration selected <b>{sel} of the {total}</b> available features, corresponding to a "
             f"<b>{red}</b> reduction in the feature space. The measured accuracy on KDDTest+ was <b>{acc}</b>.")

    section("Purpose and scope of this demonstration")
    st.markdown(
        "This dashboard demonstrates the **existing** GWO + Random Forest IDS implementation so that its behaviour and "
        "measured results can be inspected without reading the source code. It is a foundation for the next stage of the "
        "project and does not itself propose or evaluate a new method.")
    st.markdown("**Research directions motivating the next stage** (from the literature survey):")
    st.markdown(
        "- ML-based IDS approaches may involve redundant features and high computational requirements.\n"
        "- Metaheuristic-based IDS methods improve feature selection and model optimisation, but efficiency remains a challenge.\n"
        "- Convergence behaviour and computational complexity remain open issues.\n"
        "- Scalability and adaptation to dynamic network environments require further work.")
    note("This demo provides initial observations on feature reduction, optimisation behaviour, convergence, computational "
         "overhead and detection performance. <b>Scalability and dynamic-environment adaptation have not been experimentally "
         "evaluated.</b>", "warn")

    with st.expander("Implementation Details", expanded=False):
        sig = inspect.signature(gwo_ids.BinaryGWO.__init__).parameters
        alpha, beta = sig["alpha"].default, sig["beta"].default
        params = (meta or {}).get("parameters", {})
        rows = [
            ("Dataset", "NSL-KDD (KDDTrain+.arff for training, KDDTest+.arff for testing)"),
            ("Input features", f"{len(gwo_ids.FEATURE_NAMES)}"),
            ("Categorical features", ", ".join(f"<code>{c}</code>" for c in gwo_ids.CATEGORICAL)
             + " (integer-encoded; categories fitted over train + test values)"),
            ("Preprocessing", "Duplicate rows removed; stratified sub-sampling of the training set; labels binarised "
                              "(normal = 0, any attack = 1); numeric coercion and median imputation; min-max scaling fitted on training data"),
            ("Feature-selection algorithm", "Binary Grey Wolf Optimization (alpha / beta / delta leaders; sigmoid transfer, "
                                            "threshold 0.5; at least one feature always kept)"),
            ("Classifier", "Random Forest (<code>class_weight='balanced_subsample'</code>)"),
            ("Fitness function", f"<code>fitness = {alpha} × (1 − validation accuracy) + {beta} × (selected features / total features)</code> "
                                 "(lower is better)"),
            ("Validation for GWO fitness", "80/20 stratified holdout on the working training set"),
            ("Final model", "Random Forest with max(trees, 100) estimators trained on the selected features, evaluated on KDDTest+"),
            ("Test dataset", "KDDTest+"),
            ("Parameters of latest run", ", ".join(f"{k} = {v}" for k, v in params.items()) if params else "not recorded for the current result files"),
        ]
        if meta and "library_versions" in meta:
            rows.append(("Library versions (latest run)", ", ".join(f"{k} {v}" for k, v in meta["library_versions"].items())))
        st.markdown('<table class="kv">' + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows) + "</table>",
                    unsafe_allow_html=True)


# ============================================================================= PAGE 2
def page_features():
    page_header("Feature Selection", "Which of the NSL-KDD input features were retained by Binary GWO.")
    feats, g = load_csv(GWO_FEAT), first_row(load_csv(GWO_RES))
    if feats is None:
        missing(GWO_FEAT)
        return
    total = len(feats)
    sel_n = int(feats["selected"].sum())
    red = 100 * (1 - sel_n / total)
    if g is not None and int(g["selected_features"]) != sel_n:
        note("The selected-feature file and the results file disagree on the feature count; they may come from different runs.", "warn")

    section("Feature-space reduction")
    c1, a1, c2, a2, c3 = st.columns([5, 1, 5, 1, 5])
    c1.markdown(card("Original features", total, "all NSL-KDD inputs", "slate"), unsafe_allow_html=True)
    a1.markdown(f"<div style='text-align:center;font-size:2rem;color:{SLATE};padding-top:18px'>➜</div>", unsafe_allow_html=True)
    c2.markdown(card("Selected features", sel_n, "retained by Binary GWO", "teal"), unsafe_allow_html=True)
    a2.markdown(f"<div style='text-align:center;font-size:2rem;color:{SLATE};padding-top:18px'>➜</div>", unsafe_allow_html=True)
    c3.markdown(card("Feature reduction", f"{red:.2f}%", f"{total - sel_n} features removed", "teal"), unsafe_allow_html=True)
    src(GWO_FEAT)

    section("Selected vs removed")
    d1, d2 = st.columns([2, 3])
    with d1:
        donut = go.Figure(go.Pie(values=[sel_n, total - sel_n], labels=["Selected", "Removed"], hole=0.68, sort=False,
                                 marker=dict(colors=[TEAL, "#dbe3ee"], line=dict(color="#fff", width=3)),
                                 textinfo="none", hovertemplate="%{label}: %{value} features<extra></extra>"))
        donut.add_annotation(text=f"<b>{red:.2f}%</b><br><span style='font-size:12px;color:{MUTED}'>reduction</span>",
                             showarrow=False, font=dict(size=26, color=NAVY))
        donut.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center"), margin=dict(l=20, r=20, t=30, b=40))
        show_fig(style_fig(donut, 320))
    with d2:
        grp = {"Categorical": list(gwo_ids.CATEGORICAL)}
        sel_set = set(feats.loc[feats["selected"] == 1, "feature"])
        kinds = {"Categorical": [f for f in feats["feature"] if f in gwo_ids.CATEGORICAL],
                 "Numeric / rate": [f for f in feats["feature"] if f not in gwo_ids.CATEGORICAL]}
        cat_total = len(kinds["Categorical"]); num_total = len(kinds["Numeric / rate"])
        cat_sel = len([f for f in kinds["Categorical"] if f in sel_set]); num_sel = len([f for f in kinds["Numeric / rate"] if f in sel_set])
        bar = go.Figure()
        bar.add_bar(name="Selected", y=["Categorical", "Numeric / rate"], x=[cat_sel, num_sel], orientation="h", marker_color=TEAL,
                    text=[cat_sel, num_sel], textposition="inside")
        bar.add_bar(name="Not selected", y=["Categorical", "Numeric / rate"], x=[cat_total - cat_sel, num_total - num_sel], orientation="h",
                    marker_color="#dbe3ee", text=[cat_total - cat_sel, num_total - num_sel], textposition="inside")
        bar.update_layout(barmode="stack", title="Selection by feature type", xaxis_title="Number of features", bargap=0.45)
        show_fig(style_fig(bar, 320))

    section("Binary selection map")
    chips = "".join(f'<span class="chip{" on" if s else ""}">{f}</span>' for f, s in zip(feats["feature"], feats["selected"]))
    st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
    st.caption("Highlighted = selected by GWO. Grey = not selected.")

    section("Feature table")
    flt = st.radio("Show", ["All features", "Selected only", "Not selected"], horizontal=True, label_visibility="collapsed")
    t = pd.DataFrame({"#": range(1, total + 1), "Feature": feats["feature"],
                      "Selected": np.where(feats["selected"] == 1, "✓ Selected", "— Not selected")})
    if flt == "Selected only":
        t = t[feats["selected"].to_numpy() == 1]
    elif flt == "Not selected":
        t = t[feats["selected"].to_numpy() == 0]
    show_df(t, height=min(36 * (len(t) + 1) + 4, 560))

    cat_sel = [c for c in gwo_ids.CATEGORICAL if c in set(feats.loc[feats["selected"] == 1, "feature"])]
    note(f"Categorical features retained: <b>{', '.join(cat_sel) if cat_sel else 'none'}</b> "
         f"(of {', '.join(gwo_ids.CATEGORICAL)}).")


# ============================================================================= PAGE 3
def page_performance():
    page_header("Performance Comparison", "Random Forest on all 41 features versus GWO + Random Forest on the selected features.")
    b, g = first_row(load_csv(BASE_RES)), first_row(load_csv(GWO_RES))
    if b is None:
        missing(BASE_RES, "Use <b>Run All-Feature Baseline</b> on the Run IDS page.")
    if g is None:
        missing(GWO_RES)
    if b is None or g is None:
        return

    # Comparability check - only when run metadata is available
    mb, mg = load_json(BASE_META), load_json(GWO_META)
    if mb and mg:
        pb, pg = mb["parameters"], mg["parameters"]
        same = all(pb.get(k) == pg.get(k) for k in ("sample", "trees", "seed"))
        note("Both result files were produced with the same sample size, tree count and seed "
             f"(sample = {pg.get('sample')}, trees = {pg.get('trees')}, seed = {pg.get('seed')})." if same else
             "The two result files were produced with <b>different</b> settings "
             f"(baseline: {pb}; GWO: {pg}). Interpret the comparison accordingly.", "ok" if same else "warn")
    else:
        note("Run metadata (sample size / seed) is not available for the current result files, so it cannot be verified "
             "that both rows were produced with identical settings.", "warn")

    section("Measured values")
    cards([
        ("All Features + RF · Features", f"{int(b['selected_features'])}", "feature count", "slate"),
        ("GWO + RF · Features", f"{int(g['selected_features'])}", f"{g['feature_reduction_percent']:.2f}% reduction", "teal"),
        ("All Features + RF · Runtime", f"{b['runtime_seconds']:.2f} s", "load + preprocess + train + test", "slate"),
        ("GWO + RF · Runtime", f"{g['runtime_seconds']:.2f} s", "load + preprocess + GWO + train + test", "teal"),
    ])

    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    fig = go.Figure()
    fig.add_bar(name=f"All Features + RF ({int(b['selected_features'])})", x=labels, y=[b[m] * 100 for m in metrics],
                marker_color=SLATE, text=[pct(b[m]) for m in metrics], textposition="outside")
    fig.add_bar(name=f"GWO + RF ({int(g['selected_features'])})", x=labels, y=[g[m] * 100 for m in metrics],
                marker_color=NAVY, text=[pct(g[m]) for m in metrics], textposition="outside")
    fig.update_layout(barmode="group", title="Detection metrics on KDDTest+", yaxis_title="Percent",
                      yaxis_range=[0, 112], bargap=0.28)
    show_fig(style_fig(fig, 430))
    src(BASE_RES, GWO_RES)

    section("Side-by-side table")
    tbl = pd.DataFrame({
        "Metric": ["Features used", "Feature reduction", "Accuracy", "Precision", "Recall", "F1-score", "Runtime (s)"],
        "All Features + RF": [f"{int(b['selected_features'])}", f"{b['feature_reduction_percent']:.2f}%", pct(b["accuracy"]),
                              pct(b["precision"]), pct(b["recall"]), pct(b["f1"]), f"{b['runtime_seconds']:.2f}"],
        "GWO + RF": [f"{int(g['selected_features'])}", f"{g['feature_reduction_percent']:.2f}%", pct(g["accuracy"]),
                     pct(g["precision"]), pct(g["recall"]), pct(g["f1"]), f"{g['runtime_seconds']:.2f}"],
    })
    show_df(tbl)

    note(f"The measured accuracy was <b>{pct(g['accuracy'])}</b> for GWO + RF, compared with <b>{pct(b['accuracy'])}</b> "
         f"for the all-feature Random Forest baseline. GWO + RF used {int(g['selected_features'])} features and the baseline used "
         f"{int(b['selected_features'])}. Recall was {pct(g['recall'])} versus {pct(b['recall'])}.")
    st.caption("These are single-run development measurements. The GWO runtime includes the optimisation search; "
               "no scalability or statistical-significance analysis is implied.")


# ============================================================================= PAGE 4
def page_convergence():
    page_header("GWO Convergence", "Best fitness recorded at each GWO iteration (lower is better).")
    c = load_csv(GWO_CONV)
    if c is None:
        missing(GWO_CONV)
        return
    g = first_row(load_csv(GWO_RES))
    best = float(c["fitness"].min())
    best_it = int(c.loc[c["fitness"] <= best + 1e-15, "iteration"].iloc[0])
    sel = int(c.loc[c["iteration"] == best_it, "selected_features"].iloc[0])

    cards([
        ("Best fitness", f"{best:.6f}", "lowest value recorded", "teal"),
        ("Best iteration", best_it, "first iteration at which it was reached", ""),
        ("Selected features", f"{sel}/{len(gwo_ids.FEATURE_NAMES)}", "in the best mask", ""),
        ("Iterations recorded", len(c), (f"{int(g['unique_fitness_evaluations'])} unique fitness evaluations"
                                         if g is not None and "unique_fitness_evaluations" in g else ""), "slate"),
    ])

    lo, hi = float(c["fitness"].min()), float(c["fitness"].max())
    pad = max((hi - lo) * 0.25, abs(hi) * 0.15, 1e-4)
    fig = go.Figure(go.Scatter(x=c["iteration"], y=c["fitness"], mode="lines+markers",
                               line=dict(color=NAVY, width=3.5), marker=dict(size=10, color="#fff", line=dict(color=NAVY, width=3)),
                               hovertemplate="Iteration %{x}<br>Best fitness %{y:.6f}<extra></extra>"))
    fig.update_layout(title="Best fitness per iteration", xaxis_title="Iteration", yaxis_title="Best Fitness",
                      yaxis=dict(range=[max(lo - pad, 0), hi + pad], tickformat=".6f"),
                      xaxis=dict(dtick=1 if len(c) <= 20 else None, range=[0.5, len(c) + 0.5]))
    show_fig(style_fig(fig, 400))
    src(GWO_CONV)

    section("Observation")
    unchanged = bool((c["fitness"] == c["fitness"].iloc[0]).all())
    if best_it == int(c["iteration"].iloc[0]) and unchanged:
        txt = ("The best recorded fitness was reached in the first iteration and remained unchanged in subsequent iterations "
               "for this experimental configuration.")
    else:
        txt = (f"The best recorded fitness ({best:.6f}) was first reached at iteration {best_it} of {len(c)}. "
               f"The fitness values recorded after that iteration are "
               f"{'unchanged' if bool((c.loc[c['iteration'] >= best_it, 'fitness'] <= best + 1e-15).all()) else 'shown in the chart above'}.")
    note(txt)
    st.caption("This is reported as an observation about the recorded run, not as a diagnosis. Whether and why it occurs more "
               "generally is a question for further investigation.")

    section("Iteration-budget experiment (development log, seed 1)")
    it = load_csv(ITERS)
    if it is None:
        missing(ITERS, "This file holds the team-recorded iteration experiments.")
    else:
        t = pd.DataFrame({"Iterations": it["iterations"], "Best fitness": it["fitness"].map(lambda v: f"{v:.6f}"),
                          "Selected": it.apply(lambda r: f"{int(r['selected_features'])}/{int(r['total_features'])}", axis=1),
                          "Accuracy": it["accuracy"].map(pct), "Recall": it["recall"].map(pct), "F1": it["f1"].map(pct)})
        show_df(t)
        src(ITERS)
        if it["fitness"].nunique() == 1:
            note("In the seed-1 iteration experiment, the best recorded fitness was reached in the first iteration and remained "
                 "unchanged through iteration 10; selected-feature count and test metrics were identical across the four budgets.")
        st.caption("Development experiments recorded by the team; not final benchmark results.")


# ============================================================================= PAGE 5
def page_stability():
    page_header("Experimental Stability", "Development experiments conducted using different random seeds.")
    s = load_csv(SEEDS)
    if s is None:
        missing(SEEDS, "This file holds the team-recorded seed experiments (see <code>experiments/README.md</code>).")
        return
    t = pd.DataFrame({
        "Seed": s["seed"],
        "Features": s.apply(lambda r: f"{int(r['selected_features'])}/{int(r['total_features'])}", axis=1),
        "Reduction": s["feature_reduction_percent"].map(lambda v: f"{v:.2f}%"),
        "Accuracy": s["accuracy"].map(pct), "Precision": s["precision"].map(pct),
        "Recall": s["recall"].map(pct), "F1": s["f1"].map(pct),
        "Runtime (s)": s["runtime_seconds"].map(lambda v: f"{v:.2f}"),
    })
    show_df(t)
    src(SEEDS)
    st.caption("Development experiments conducted using different random seeds. Not final benchmark results.")

    fig = go.Figure()
    for m, name, col in [("accuracy", "Accuracy", NAVY), ("recall", "Recall", TEAL), ("f1", "F1-score", AMBER)]:
        fig.add_bar(name=name, x=[f"Seed {x}" for x in s["seed"]], y=s[m] * 100, marker_color=col,
                    text=[pct(v) for v in s[m]], textposition="outside")
    fig.update_layout(barmode="group", title="Metrics by random seed", yaxis_title="Percent", yaxis_range=[0, 105], bargap=0.3)
    show_fig(style_fig(fig, 400))

    note(f"Across the {len(s)} recorded seeds, the number of selected features ranged from "
         f"<b>{int(s['selected_features'].min())}</b> to <b>{int(s['selected_features'].max())}</b> of "
         f"{int(s['total_features'].iloc[0])}, accuracy from <b>{pct(s['accuracy'].min())}</b> to <b>{pct(s['accuracy'].max())}</b>, "
         f"and F1-score from <b>{pct(s['f1'].min())}</b> to <b>{pct(s['f1'].max())}</b>.")
    st.caption("Three seeds are a small sample and no statistical test is applied. The variation shows that individual GWO runs "
               "can select different subsets and yield different test metrics.")


# ============================================================================= PAGE 6
def confusion_fig(pred_df, title):
    y, p = pred_df["actual"].to_numpy(), pred_df["predicted"].to_numpy()
    tn, fp = int(((y == 0) & (p == 0)).sum()), int(((y == 0) & (p == 1)).sum())
    fn, tp = int(((y == 1) & (p == 0)).sum()), int(((y == 1) & (p == 1)).sum())
    z = np.array([[tn, fp], [fn, tp]])
    rowsum = z.sum(axis=1, keepdims=True)
    txt = [[f"<b>{z[i, j]:,}</b><br>{z[i, j] / max(rowsum[i, 0], 1) * 100:.1f}% of actual {['Normal', 'Attack'][i]}"
            for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted Normal", "Predicted Attack"], y=["Actual Normal", "Actual Attack"],
                               text=txt, texttemplate="%{text}", colorscale=[[0, "#eef3f9"], [1, NAVY]],
                               showscale=False, hoverinfo="skip", textfont=dict(size=14)))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title)
    return style_fig(fig, 380), (tn, fp, fn, tp)


def page_diagnostics():
    page_header("Confusion Matrix & Feature Importance",
                "Computed from the actual predictions and trained model saved by the IDS run.")
    pred = load_csv(GWO_PRED)
    section("Confusion matrix — GWO + Random Forest (KDDTest+)")
    if pred is None:
        missing(GWO_PRED, "Predictions are written when the IDS is run. Use <b>Run GWO IDS</b> on the Run IDS page "
                          "(or the CLI) to generate them. No matrix is estimated from the accuracy value.")
    else:
        fig, (tn, fp, fn, tp) = confusion_fig(pred, "GWO + RF")
        left, right = st.columns([3, 2])
        with left:
            show_fig(fig)
            src(GWO_PRED)
        with right:
            cards([("True Negatives", f"{tn:,}", "Normal predicted Normal", "slate"),
                   ("False Positives", f"{fp:,}", "Normal predicted Attack", "amber"),
                   ("False Negatives", f"{fn:,}", "Attack predicted Normal", "amber"),
                   ("True Positives", f"{tp:,}", "Attack predicted Attack", "teal")], per_row=2)
        total = tn + fp + fn + tp
        g = first_row(load_csv(GWO_RES))
        txt = (f"Of {total:,} test records, {fn:,} attack records were predicted as Normal and {fp:,} normal records were "
               f"predicted as Attack.")
        if g is not None:
            same = abs((tn + tp) / total - g["accuracy"]) < 1e-9
            txt += (" The accuracy implied by this matrix matches the results file." if same else
                    " Note: the accuracy implied by this matrix does not match the results file; the files may come from different runs.")
        note(txt)
        bp = load_csv(BASE_PRED)
        if bp is not None:
            with st.expander("Reference: all-feature Random Forest baseline confusion matrix"):
                bfig, _ = confusion_fig(bp, "All Features + RF")
                show_fig(bfig)
                src(BASE_PRED)

    section("Random Forest Feature Importance")
    imp = load_csv(GWO_IMP)
    if imp is None:
        missing(GWO_IMP, "Importances are saved from the trained Random Forest when the IDS is run.")
    else:
        d = imp.sort_values("importance")
        fig = go.Figure(go.Bar(x=d["importance"], y=d["feature"], orientation="h", marker_color=TEAL,
                               hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        fig.update_layout(title="Random Forest Feature Importance (selected features)", xaxis_title="Importance (feature_importances_)")
        show_fig(style_fig(fig, max(340, 24 * len(d) + 120)))
        src(GWO_IMP)
        st.caption("Random Forest importance (impurity-based, from the final trained model) describes how the classifier uses the "
                   "features that were already selected. It is not a measure of how important a feature was to the GWO search.")


# ============================================================================= PAGE 7
def backup_results():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = RESULTS / f"backup_{stamp}"
    files = [p for p in RESULTS.glob("*") if p.is_file()]
    if not files:
        return None
    dest.mkdir(parents=True, exist_ok=True)
    for p in files:
        shutil.copy2(p, dest / p.name)
    return dest


def run_with_progress(mode, kwargs):
    bar = st.progress(0.0)
    status = st.status("Starting…", expanded=True)
    t0 = time.time()

    def progress(stage, message, fraction):
        bar.progress(min(max(float(fraction), 0.0), 1.0))
        status.update(label=message)
        status.write(f"`{time.time() - t0:6.1f}s` &nbsp; {message}")

    try:
        fn = gwo_ids.run_gwo if mode == "gwo" else gwo_ids.run_baseline
        result = fn(progress=progress, results_dir=str(RESULTS), **kwargs)
    except Exception as e:  # shown to the user, not swallowed
        status.update(label="Run failed", state="error")
        st.error(f"The run failed: {type(e).__name__}: {e}")
        return None
    status.update(label=f"Completed in {time.time() - t0:.1f} s — results saved to results/", state="complete", expanded=False)
    return result


def page_run():
    page_header("Run IDS", "Execute the existing implementation from the dashboard. Results are written to the results/ folder.")
    note("Each run calls the same <code>gwo_ids.py</code> functions as the command-line interface. Progress messages "
         "correspond to the stages actually being executed.")

    section("Configuration")
    c1, c2 = st.columns(2)
    train = c1.text_input("Training dataset path", DEFAULT_TRAIN)
    test = c2.text_input("Testing dataset path", DEFAULT_TEST)
    a, b, c, d, e = st.columns(5)
    sample = a.number_input("Training sample size", 500, 125000, 5000, 500)
    pop = b.number_input("GWO population", 3, 50, 5, 1)
    iters = c.number_input("Iterations", 1, 100, 10, 1)
    trees = d.number_input("Random Forest trees", 10, 500, 50, 10)
    seed = e.number_input("Random seed", 0, 10_000, 42, 1)
    backup = st.checkbox("Back up the current result files to results/backup_<timestamp>/ before running", value=True,
                         help="Running overwrites results/gwo_*.csv or results/baseline_*.csv.")

    def resolve(p):
        pp = Path(p)
        return pp if pp.is_absolute() else ROOT / pp

    train_p, test_p = resolve(train), resolve(test)
    ok = True
    for label, p in (("Training", train_p), ("Testing", test_p)):
        if not p.exists():
            st.error(f"{label} dataset not found: {p}. Place the NSL-KDD ARFF files in the data/ folder.")
            ok = False

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    b1, b2, _ = st.columns([1.2, 1.4, 3])
    run_gwo_clicked = b1.button("Run GWO IDS", type="primary", disabled=not ok)
    run_base_clicked = b2.button("Run All-Feature Baseline", disabled=not ok)

    if run_gwo_clicked or run_base_clicked:
        if backup:
            dest = backup_results()
            if dest:
                st.caption(f"Previous results backed up to `{rel(dest)}`.")
        common = dict(train_arff=str(train_p), test_arff=str(test_p), sample=int(sample), trees=int(trees), seed=int(seed))
        if run_gwo_clicked:
            res = run_with_progress("gwo", dict(common, pop=int(pop), iters=int(iters)))
            mode = "GWO + Random Forest"
        else:
            res = run_with_progress("baseline", common)
            mode = "All-feature Random Forest baseline"
        if res is not None:
            st.session_state["last_run"] = (mode, {k: (float(v) if not isinstance(v, (int, np.integer)) else int(v)) for k, v in res.items()},
                                            datetime.now().strftime("%H:%M:%S"))

    last = st.session_state.get("last_run")
    if last:
        mode, r, at = last
        section(f"Latest run from this session — {mode} ({at})")
        items = [("Features", f"{r['selected_features']}/{len(gwo_ids.FEATURE_NAMES)}", f"{r['feature_reduction_percent']:.2f}% reduction", "teal"),
                 ("Accuracy", pct(r["accuracy"]), "", ""), ("Precision", pct(r["precision"]), "", ""),
                 ("Recall", pct(r["recall"]), "", ""), ("F1-score", pct(r["f1"]), "", ""),
                 ("Runtime", f"{r['runtime_seconds']:.2f} s", "", "slate")]
        if "fitness" in r:
            items.append(("Best fitness", f"{r['fitness']:.6f}", f"{r['unique_fitness_evaluations']} unique evaluations", ""))
        cards(items, per_row=4)
        note("Results have been saved. The other dashboard pages now read the updated files.", "ok")

    refs = load_csv(REFS)
    if refs is not None:
        with st.expander("Recorded development reference values (for comparison)"):
            t = pd.DataFrame({"Run": refs["description"], "Features": refs.apply(lambda r: f"{int(r['selected_features'])}/{int(r['total_features'])}", axis=1),
                              "Accuracy": refs["accuracy"].map(pct), "Precision": refs["precision"].map(pct),
                              "Recall": refs["recall"].map(pct), "F1": refs["f1"].map(pct),
                              "Runtime (s)": refs["runtime_seconds"].map(lambda v: f"{v:.2f}")})
            show_df(t)
            src(REFS)
            st.caption("Values recorded by the team during development. Re-running on another machine or with different "
                       "scikit-learn / pandas versions can give slightly different numbers; the library versions are stored in "
                       "the run metadata JSON of every dashboard/CLI run.")


# ----------------------------------------------------------------------------- router
{
    PAGES[0]: page_overview,
    PAGES[1]: page_features,
    PAGES[2]: page_performance,
    PAGES[3]: page_convergence,
    PAGES[4]: page_stability,
    PAGES[5]: page_diagnostics,
    PAGES[6]: page_run,
}[page]()

st.markdown('<div class="footer">Intelligent Network Intrusion Detection Using Metaheuristic Optimization &nbsp;·&nbsp; '
            'Demonstration of an existing GWO + Random Forest implementation &nbsp;·&nbsp; All values are read from result files</div>',
            unsafe_allow_html=True)
