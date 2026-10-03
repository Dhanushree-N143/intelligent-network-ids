"""
GWO-Based Network Intrusion Detection System - Streamlit dashboard.

This dashboard is a demonstration layer around the EXISTING implementation in gwo_ids.py
(Binary Grey Wolf Optimization + Random Forest on NSL-KDD). It does not contain any
algorithm logic of its own: every number shown is read from files written by that
implementation (results/*.csv, results/*.json) or from team-recorded development
experiments (experiments/*.csv).

Run with: streamlit run app.py
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
import gwo_ids

RESULTS = ROOT / "results"
EXPERIMENTS = ROOT / "experiments"

DEFAULT_TRAIN = "data/KDDTrain+.arff"
DEFAULT_TEST = "data/KDDTest+.arff"

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

html, body, [class*="css"], .stMarkdown, button, input {{
    font-family: 'Inter', 'Segoe UI', Roboto, Arial, sans-serif !important;
    color: {INK};
}}

.stApp {{
    background: linear-gradient(
        180deg,
        #eef3fa 0%,
        #f7f9fc 320px,
        #f7f9fc 100%
    );
}}

.block-container {{
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}}

#MainMenu,
footer,
header[data-testid="stHeader"] {{
    visibility: hidden;
    height: 0;
}}

/* ================= HERO ================= */

.hero {{
    position: relative;
    overflow: hidden;
    border-radius: 18px;
    padding: 30px 36px 26px 36px;
    margin-bottom: 26px;
    color: #fff;
    background: linear-gradient(
        120deg,
        #12365c 0%,
        #1f4e79 48%,
        #0f7c86 100%
    );
    box-shadow: 0 14px 34px rgba(18,54,92,.28);
}}

.hero:before {{
    content: "";
    position: absolute;
    right: -80px;
    top: -110px;
    width: 340px;
    height: 340px;
    border-radius: 50%;
    background: radial-gradient(
        circle,
        rgba(255,255,255,.18) 0%,
        rgba(255,255,255,0) 70%
    );
}}

.hero:after {{
    content: "";
    position: absolute;
    right: 150px;
    bottom: -150px;
    width: 300px;
    height: 300px;
    border-radius: 50%;
    background: radial-gradient(
        circle,
        rgba(124,226,214,.22) 0%,
        rgba(124,226,214,0) 70%
    );
}}

.hero .row {{
    position: relative;
    z-index: 2;
    display: flex;
    align-items: center;
    gap: 22px;
}}

.hero .logo {{
    flex: 0 0 auto;
    width: 70px;
    height: 70px;
    border-radius: 18px;
    background: rgba(255,255,255,.14);
    border: 1px solid rgba(255,255,255,.28);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.1rem;
}}

.hero h1 {{
    color: #fff !important;
    font-size: 2rem;
    margin: 0 0 4px 0;
    padding: 0;
    font-weight: 800;
    letter-spacing: -.3px;
    line-height: 1.15;
}}

.hero .subtitle {{
    color: #d6e6f5;
    margin: 0;
    font-size: 1.02rem;
    font-weight: 500;
    letter-spacing: .3px;
}}

.hero .pills {{
    position: relative;
    z-index: 2;
    margin-top: 18px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}}

.hero .pill {{
    font-size: .78rem;
    font-weight: 600;
    color: #fff;
    background: rgba(255,255,255,.14);
    border: 1px solid rgba(255,255,255,.26);
    padding: 5px 13px;
    border-radius: 20px;
}}

.hero .pill.accent {{
    background: #7ee2d6;
    color: #0b3b45;
    border-color: #7ee2d6;
}}

/* ================= HEADINGS ================= */

.page-title {{
    font-size: 1.7rem;
    font-weight: 800;
    color: {NAVY};
    margin: 0;
    letter-spacing: -.3px;
}}

.page-sub {{
    color: {MUTED};
    margin: 2px 0 20px 0;
    font-size: 1rem;
}}

.sec {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.1rem;
    font-weight: 700;
    color: {INK};
    margin: 30px 0 14px 0;
}}

.sec:before {{
    content: "";
    width: 5px;
    height: 22px;
    border-radius: 3px;
    background: linear-gradient(180deg, {NAVY}, {TEAL});
}}

/* ================= METRIC CARDS ================= */

.card {{
    position: relative;
    background: #fff;
    border: 1px solid {LINE};
    border-radius: 14px;
    padding: 16px 18px 14px 18px;
    height: 100%;
    box-shadow:
        0 2px 6px rgba(31,78,121,.06),
        0 8px 22px rgba(31,78,121,.06);
    overflow: hidden;
}}

.card:before {{
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    right: 0;
    height: 4px;
    background: {NAVY};
}}

.card.teal:before {{
    background: linear-gradient(90deg, {TEAL}, #5cc6b8);
}}

.card.amber:before {{
    background: linear-gradient(90deg, {AMBER}, #f0b36a);
}}

.card.slate:before {{
    background: linear-gradient(90deg, {SLATE}, #b9c6d6);
}}

.card:not(.teal):not(.amber):not(.slate):before {{
    background: linear-gradient(90deg, {NAVY}, #4a86c0);
}}

.card .ico {{
    position: absolute;
    right: 14px;
    top: 16px;
    width: 38px;
    height: 38px;
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.15rem;
    background: #eaf1f9;
}}

.card.teal .ico {{
    background: #e0f4f2;
}}

.card.amber .ico {{
    background: #fdf0e0;
}}

.card.slate .ico {{
    background: #eef1f5;
}}

.card .lbl {{
    font-size: .7rem;
    text-transform: uppercase;
    letter-spacing: .9px;
    color: {MUTED};
    font-weight: 700;
    padding-right: 46px;
    margin-top: 2px;
}}

.card .val {{
    font-size: 1.95rem;
    font-weight: 800;
    color: {NAVY};
    line-height: 1.2;
    margin-top: 6px;
    letter-spacing: -.5px;
}}

.card.teal .val {{
    color: #0b6e70;
}}

.card.amber .val {{
    color: #b2650f;
}}

.card .sub {{
    font-size: .77rem;
    color: {MUTED};
    margin-top: 3px;
    line-height: 1.3;
}}

/* ================= PIPELINE ================= */

.flow {{
    display: flex;
    align-items: stretch;
    flex-wrap: wrap;
    margin: 6px 0 4px 0;
}}

.flow .step {{
    position: relative;
    flex: 1 1 140px;
    min-width: 140px;
    background: #fff;
    border: 1px solid {LINE};
    border-radius: 14px;
    padding: 16px 12px 13px 12px;
    text-align: center;
    box-shadow:
        0 2px 6px rgba(31,78,121,.06),
        0 8px 22px rgba(31,78,121,.06);
}}

.flow .badge {{
    width: 46px;
    height: 46px;
    margin: 0 auto 8px auto;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    background: linear-gradient(135deg, #e6eef8, #d3e2f3);
    border: 2px solid #fff;
    box-shadow: 0 0 0 2px #c9daee;
}}

.flow .step.hl .badge {{
    background: linear-gradient(135deg, #dff5f2, #bfe9e4);
    box-shadow: 0 0 0 2px #a8ddd6;
}}

.flow .step.hl {{
    border-color: #a8ddd6;
}}

.flow .num {{
    font-size: .66rem;
    color: {MUTED};
    font-weight: 700;
    letter-spacing: 1.2px;
}}

.flow .name {{
    font-weight: 700;
    color: {NAVY};
    font-size: .95rem;
    margin: 2px 0 4px 0;
    line-height: 1.2;
}}

.flow .desc {{
    font-size: .74rem;
    color: {MUTED};
    line-height: 1.35;
}}

.flow .arrow {{
    align-self: center;
    color: {TEAL};
    font-size: 1.35rem;
    padding: 0 5px;
    font-weight: 700;
}}

/* ================= NOTES ================= */

.note {{
    background: #fff;
    border: 1px solid {LINE};
    border-left: 5px solid {NAVY};
    border-radius: 10px;
    padding: 13px 18px;
    font-size: .93rem;
    color: {INK};
    margin: 12px 0;
    box-shadow: 0 2px 8px rgba(31,78,121,.05);
    line-height: 1.5;
}}

.note.warn {{
    border-left-color: {AMBER};
    background: #fffaf3;
}}

.note.ok {{
    border-left-color: {TEAL};
    background: #f4fbfa;
}}

.chips {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 6px;
}}

.chip {{
    font-size: .8rem;
    padding: 6px 13px;
    border-radius: 20px;
    border: 1px solid {LINE};
    background: #fff;
    color: #8a97a8;
}}

.chip.on {{
    background: linear-gradient(135deg, {TEAL}, #14a3a0);
    border-color: {TEAL};
    color: #fff;
    font-weight: 600;
}}

.srcline {{
    font-size: .75rem;
    color: {MUTED};
    margin-top: 6px;
}}

.srcline code {{
    background: #eaf1f9;
    color: {NAVY};
    padding: 1px 6px;
    border-radius: 5px;
}}

.kv td {{
    padding: 7px 14px 7px 0;
    vertical-align: top;
    font-size: .92rem;
    border-bottom: 1px solid #f0f3f8;
}}

.kv td:first-child {{
    color: {MUTED};
    width: 230px;
    font-weight: 600;
}}

.footer {{
    margin-top: 40px;
    padding-top: 14px;
    border-top: 1px solid {LINE};
    text-align: center;
    color: {MUTED};
    font-size: .78rem;
}}

/* ================= TOP MOBILE NAVIGATION ================= */

.top-nav {{
    background: rgba(255,255,255,.96);
    border: 1px solid {LINE};
    border-radius: 14px;
    padding: 10px 12px;
    margin: 0 0 18px 0;
    box-shadow: 0 4px 16px rgba(31,78,121,.07);
    position: sticky;
    top: 8px;
    z-index: 50;
    backdrop-filter: blur(10px);
}}

.top-nav-label {{
    font-size: .68rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {MUTED};
    font-weight: 800;
    margin: 0 0 4px 2px;
}}

.top-nav-help {{
    font-size: .72rem;
    color: {MUTED};
    margin: 5px 2px 0 2px;
}}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {{
    background: #ffffff;
    border-right: 1px solid {LINE};
    box-shadow: 4px 0 22px rgba(31,78,121,.05);
}}

.side-brand {{
    display: flex;
    align-items: center;
    gap: 11px;
    margin-bottom: 14px;
}}

.side-brand .lg {{
    width: 42px;
    height: 42px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    background: linear-gradient(135deg, {NAVY}, {TEAL});
}}

.side-brand .t1 {{
    font-weight: 800;
    color: {NAVY};
    font-size: 1rem;
    line-height: 1.1;
}}

.side-brand .t2 {{
    color: {MUTED};
    font-size: .74rem;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] {{
    gap: 4px;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    padding: 9px 12px;
    border-radius: 10px;
    width: 100%;
    cursor: pointer;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
    background: #eef4fb;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
    background: linear-gradient(90deg, #e3eefa, #eaf7f6);
    box-shadow: inset 3px 0 0 {TEAL};
}}

/* ================= MOBILE ================= */

@media (max-width: 768px) {{

    .block-container {{
        padding: .65rem .65rem 2rem .65rem;
    }}

    .hero {{
        padding: 20px 18px 18px 18px;
        border-radius: 16px;
        margin-bottom: 16px;
    }}

    .hero .row {{
        gap: 12px;
        align-items: flex-start;
    }}

    .hero .logo {{
        width: 54px;
        height: 54px;
        border-radius: 14px;
        font-size: 1.55rem;
        flex-shrink: 0;
    }}

    .hero h1 {{
        font-size: 1.65rem;
        line-height: 1.08;
    }}

    .hero .subtitle {{
        font-size: .86rem;
        line-height: 1.45;
        margin-top: 6px;
    }}

    .hero .pills {{
        margin-top: 14px;
        gap: 6px;
    }}

    .hero .pill {{
        font-size: .7rem;
        padding: 5px 9px;
    }}

    .page-title {{
        font-size: 1.35rem;
    }}

    .page-sub {{
        font-size: .88rem;
        line-height: 1.4;
    }}

    .sec {{
        margin-top: 23px;
        font-size: 1rem;
    }}

    .flow {{
        display: block;
    }}

    .flow .step {{
        min-width: 0;
        margin-bottom: 9px;
        padding: 13px 10px;
    }}

    .flow .arrow {{
        display: block;
        text-align: center;
        transform: rotate(90deg);
        height: 14px;
        padding: 0;
    }}

    .card {{
        padding: 14px;
        margin-bottom: 10px;
    }}

    .card .val {{
        font-size: 1.55rem;
    }}

    .top-nav {{
        position: sticky;
        top: 5px;
        padding: 9px;
        margin-bottom: 13px;
    }}

    .top-nav-help {{
        display: none;
    }}

    section[data-testid="stSidebar"] {{
        display: none;
    }}

    [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
    }}

    .stPlotlyChart {{
        width: 100% !important;
    }}

    .footer {{
        font-size: .68rem;
        line-height: 1.4;
    }}
}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# =============================================================================
# Paths
# =============================================================================

BASE_RES = RESULTS / "baseline_results.csv"
BASE_SEL = RESULTS / "baseline_selected_features.csv"
BASE_PRED = RESULTS / "baseline_predictions.csv"
BASE_IMP = RESULTS / "baseline_feature_importance.csv"

GWO_RES = RESULTS / "gwo_results.csv"
GWO_SEL = RESULTS / "gwo_selected_features.csv"
GWO_PRED = RESULTS / "gwo_predictions.csv"
GWO_IMP = RESULTS / "gwo_feature_importance.csv"
GWO_CONV = RESULTS / "gwo_convergence.csv"

REFS = EXPERIMENTS / "development_seed_experiments.csv"
ITER_REFS = EXPERIMENTS / "development_iteration_experiment.csv"

PAGES = [
    "System Overview",
    "Feature Selection",
    "Performance Comparison",
    "GWO Convergence",
    "Experimental Stability",
    "Confusion Matrix & Importance",
    "Run IDS",
]

# =============================================================================
# Helper functions
# =============================================================================

def rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def load_csv(path):
    path = Path(path)

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read `{rel(path)}`: {exc}")
        return None


def first_row(df):
    if df is None or df.empty:
        return None
    return df.iloc[0]


def pct(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "—"


def section(title):
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="page-title">{title}</div>
        <div class="page-sub">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def note(text, kind=""):
    cls = "note"
    if kind:
        cls += f" {kind}"

    st.markdown(
        f'<div class="{cls}">{text}</div>',
        unsafe_allow_html=True,
    )


def missing(path, message):
    st.markdown(
        f"""
        <div class="note warn">
            <b>Result not available</b><br>
            {message}<br>
            Expected file: <code>{rel(path)}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )


def src(path):
    st.markdown(
        f'<div class="srcline">Source: <code>{rel(path)}</code></div>',
        unsafe_allow_html=True,
    )


def cards(items, per_row=4):
    for start in range(0, len(items), per_row):
        batch = items[start:start + per_row]
        cols = st.columns(len(batch))

        for col, item in zip(cols, batch):
            label, value, subtitle, variant = item

            with col:
                cls = f"card {variant}".strip()

                st.markdown(
                    f"""
                    <div class="{cls}">
                        <div class="lbl">{label}</div>
                        <div class="val">{value}</div>
                        <div class="sub">{subtitle}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def show_df(df, hide_index=True):
    if df is None or df.empty:
        st.info("No data available.")
        return

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=hide_index,
    )


def style_fig(fig, height=420):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(
            family="Inter, Segoe UI, Arial",
            color=INK,
        ),
        margin=dict(l=30, r=20, t=55, b=35),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return fig


def show_fig(fig):
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )


# =============================================================================
# Top navigation
# =============================================================================

def render_navigation():
    current = st.session_state.get("page", PAGES[0])

    st.markdown(
        """
        <div class="top-nav">
            <div class="top-nav-label">Dashboard Navigation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.selectbox(
        "Navigate",
        PAGES,
        index=PAGES.index(current),
        key="top_navigation",
        label_visibility="collapsed",
    )

    if selected != current:
        st.session_state["page"] = selected
        st.rerun()


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="side-brand">
                <div class="lg">🛡️</div>
                <div>
                    <div class="t1">GWO Network IDS</div>
                    <div class="t2">Academic Demonstration</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            PAGES,
            index=PAGES.index(st.session_state.get("page", PAGES[0])),
            label_visibility="collapsed",
        )

        if page != st.session_state.get("page", PAGES[0]):
            st.session_state["page"] = page
            st.rerun()

        st.markdown("---")

        st.markdown("**Implementation**")
        st.caption("Binary Grey Wolf Optimization")
        st.caption("Random Forest classifier")
        st.caption("NSL-KDD dataset")

        st.markdown("---")

        st.markdown("**Data status**")

        for label, path in [
            ("Training data", ROOT / DEFAULT_TRAIN),
            ("Testing data", ROOT / DEFAULT_TEST),
            ("GWO results", GWO_RES),
            ("Baseline results", BASE_RES),
        ]:
            if path.exists():
                st.success(label)
            else:
                st.warning(label)


# =============================================================================
# Hero
# =============================================================================

def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="row">
                <div class="logo">🛡️</div>
                <div>
                    <h1>GWO-Based Network Intrusion Detection System</h1>
                    <div class="subtitle">
                        NSL-KDD&nbsp; | &nbsp;Binary Grey Wolf Optimization&nbsp; | &nbsp;Random Forest
                    </div>
                </div>
            </div>

            <div class="pills">
                <div class="pill accent">Existing implementation demo</div>
                <div class="pill">41 input features</div>
                <div class="pill">Wrapper-based feature selection</div>
                <div class="pill">Binary classification: Normal / Attack</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE 1 — OVERVIEW
# =============================================================================

def page_overview():
    page_header(
        "System Overview",
        "The existing pipeline and the headline measurements from the latest result files.",
    )

    train_path = ROOT / DEFAULT_TRAIN
    test_path = ROOT / DEFAULT_TEST

    train_rows = None
    test_rows = None

    try:
        train_rows = gwo_ids.count_arff_rows(str(train_path))
    except Exception:
        pass

    try:
        test_rows = gwo_ids.count_arff_rows(str(test_path))
    except Exception:
        pass

    gwo = first_row(load_csv(GWO_RES))

    if gwo is not None:
        selected = int(gwo.get("selected_features", 0))
        total_features = int(gwo.get("total_features", len(gwo_ids.FEATURE_NAMES)))
        reduction = float(gwo.get("feature_reduction_percent", 0))
        accuracy = float(gwo.get("accuracy", 0))
        recall = float(gwo.get("recall", 0))
        f1 = float(gwo.get("f1", 0))
    else:
        selected = None
        total_features = len(gwo_ids.FEATURE_NAMES)
        reduction = None
        accuracy = None
        recall = None
        f1 = None

    cards(
        [
            (
                "Training Samples",
                f"{train_rows:,}" if train_rows is not None else "—",
                "KDDTrain+",
                "",
            ),
            (
                "Test Samples",
                f"{test_rows:,}" if test_rows is not None else "—",
                "KDDTest+",
                "",
            ),
            (
                "Original Features",
                str(total_features),
                "NSL-KDD input features",
                "slate",
            ),
            (
                "Selected Features",
                str(selected) if selected is not None else "—",
                "GWO-selected",
                "teal",
            ),
            (
                "Feature Reduction",
                f"{reduction:.2f}%" if reduction is not None else "—",
                "Feature-space reduction",
                "teal",
            ),
            (
                "Accuracy",
                pct(accuracy) if accuracy is not None else "—",
                "KDDTest+",
                "",
            ),
            (
                "Recall",
                pct(recall) if recall is not None else "—",
                "Attack recall",
                "amber",
            ),
            (
                "F1-score",
                pct(f1) if f1 is not None else "—",
                "Binary classification",
                "",
            ),
        ],
        per_row=4,
    )

    section("System Architecture")

    st.markdown(
        """
        <div class="flow">
            <div class="step">
                <div class="badge">📊</div>
                <div class="num">01</div>
                <div class="name">NSL-KDD Dataset</div>
                <div class="desc">Network traffic records</div>
            </div>

            <div class="arrow">→</div>

            <div class="step">
                <div class="badge">⚙️</div>
                <div class="num">02</div>
                <div class="name">Preprocessing</div>
                <div class="desc">Encoding, scaling and cleaning</div>
            </div>

            <div class="arrow">→</div>

            <div class="step hl">
                <div class="badge">🐺</div>
                <div class="num">03</div>
                <div class="name">Binary GWO</div>
                <div class="desc">Metaheuristic search</div>
            </div>

            <div class="arrow">→</div>

            <div class="step hl">
                <div class="badge">🔎</div>
                <div class="num">04</div>
                <div class="name">Feature Selection</div>
                <div class="desc">Select useful feature subset</div>
            </div>

            <div class="arrow">→</div>

            <div class="step">
                <div class="badge">🌲</div>
                <div class="num">05</div>
                <div class="name">Random Forest</div>
                <div class="desc">Train classifier</div>
            </div>

            <div class="arrow">→</div>

            <div class="step">
                <div class="badge">🚨</div>
                <div class="num">06</div>
                <div class="name">Classification</div>
                <div class="desc">Normal / Attack</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section("Current implementation")

    note(
        "This dashboard demonstrates the existing GWO + Random Forest IDS. "
        "It is not the final proposed or novel system."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Dataset**")
        st.markdown(
            """
            - NSL-KDD
            - 41 input features
            - Binary classification
            - Normal / Attack
            - KDDTrain+ for training
            - KDDTest+ for evaluation
            """
        )

    with c2:
        st.markdown("**Optimization and classification**")
        st.markdown(
            """
            - Binary Grey Wolf Optimization
            - Wrapper-based feature selection
            - 80/20 stratified holdout for GWO fitness
            - Random Forest classifier
            - Evaluation on KDDTest+
            """
        )

    section("Observed research context")

    note(
        "The current implementation provides experimental observations relevant to "
        "feature reduction, optimization behavior, convergence, computational overhead, "
        "and detection performance. Dynamic adaptation and scalability have not been "
        "experimentally evaluated in this demonstration.",
        "warn",
    )


# =============================================================================
# PAGE 2 — FEATURE SELECTION
# =============================================================================

def page_features():
    page_header(
        "Feature Selection",
        "Inspect the feature subset produced by the recorded GWO run.",
    )

    gwo = first_row(load_csv(GWO_RES))
    selected_df = load_csv(GWO_SEL)

    if gwo is None:
        missing(
            GWO_RES,
            "The GWO result file is required to display the recorded feature-selection summary.",
        )
        return

    total = int(gwo.get("total_features", len(gwo_ids.FEATURE_NAMES)))
    selected = int(gwo.get("selected_features", 0))
    reduction = float(gwo.get("feature_reduction_percent", 0))

    cards(
        [
            (
                "Original Features",
                str(total),
                "Available input features",
                "slate",
            ),
            (
                "Selected Features",
                str(selected),
                "Selected by GWO",
                "teal",
            ),
            (
                "Feature Reduction",
                f"{reduction:.2f}%",
                "Reduced feature space",
                "teal",
            ),
        ],
        per_row=3,
    )

    section("Feature-space reduction")

    fig = go.Figure(
        go.Bar(
            x=[total, selected],
            y=["Original", "GWO Selected"],
            orientation="h",
            marker_color=[SLATE, TEAL],
            text=[str(total), str(selected)],
            textposition="auto",
            hovertemplate="%{y}: %{x} features<extra></extra>",
        )
    )

    fig.update_layout(
        title="Original vs Selected Features",
        xaxis_title="Number of features",
        yaxis_title="",
    )

    show_fig(style_fig(fig, 300))

    note(
        f"The GWO-based configuration selected {selected} of the {total} available "
        f"features, corresponding to a {reduction:.2f}% reduction in the feature space."
    )

    section("Selected feature names")

    if selected_df is None:
        missing(
            GWO_SEL,
            "The selected-feature file is not available.",
        )
    else:
        show_df(selected_df)
        src(GWO_SEL)

    section("Feature selection map")

    if selected_df is not None:
        display = selected_df.copy()

        if "selected" in display.columns:
            display["selected"] = display["selected"].astype(int)

        if "feature" in display.columns:
            display = display[["feature"] + [c for c in display.columns if c != "feature"]]

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# PAGE 3 — PERFORMANCE
# =============================================================================

def page_performance():
    page_header(
        "Performance Comparison",
        "Measured values from the all-feature Random Forest baseline and GWO + Random Forest.",
    )

    base = first_row(load_csv(BASE_RES))
    gwo = first_row(load_csv(GWO_RES))

    if base is None:
        missing(
            BASE_RES,
            "The all-feature baseline result file is required for comparison.",
        )

    if gwo is None:
        missing(
            GWO_RES,
            "The GWO result file is required for comparison.",
        )

    if base is None or gwo is None:
        return

    comparison = pd.DataFrame(
        {
            "Configuration": [
                "All Features + RF",
                "GWO + RF",
            ],
            "Accuracy": [
                float(base["accuracy"]),
                float(gwo["accuracy"]),
            ],
            "Precision": [
                float(base["precision"]),
                float(gwo["precision"]),
            ],
            "Recall": [
                float(base["recall"]),
                float(gwo["recall"]),
            ],
            "F1-score": [
                float(base["f1"]),
                float(gwo["f1"]),
            ],
        }
    )

    section("Classification metrics")

    fig = go.Figure()

    metrics = [
        ("Accuracy", NAVY),
        ("Precision", TEAL),
        ("Recall", AMBER),
        ("F1-score", SLATE),
    ]

    for metric, color in metrics:
        fig.add_trace(
            go.Bar(
                name=metric,
                x=comparison["Configuration"],
                y=comparison[metric],
                marker_color=color,
                text=[f"{v * 100:.2f}%" for v in comparison[metric]],
                textposition="outside",
                hovertemplate=f"{metric}: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="group",
        yaxis=dict(
            title="Score",
            tickformat=".0%",
            range=[0, 1.08],
        ),
        xaxis_title="Configuration",
    )

    show_fig(style_fig(fig, 430))

    section("Detailed measurements")

    table = pd.DataFrame(
        {
            "Metric": [
                "Feature count",
                "Feature reduction",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
                "Runtime",
            ],
            "All Features + RF": [
                f"{int(base['selected_features'])}/{int(base['total_features'])}",
                f"{float(base['feature_reduction_percent']):.2f}%",
                pct(base["accuracy"]),
                pct(base["precision"]),
                pct(base["recall"]),
                pct(base["f1"]),
                f"{float(base['runtime_seconds']):.2f} s",
            ],
            "GWO + RF": [
                f"{int(gwo['selected_features'])}/{int(gwo['total_features'])}",
                f"{float(gwo['feature_reduction_percent']):.2f}%",
                pct(gwo["accuracy"]),
                pct(gwo["precision"]),
                pct(gwo["recall"]),
                pct(gwo["f1"]),
                f"{float(gwo['runtime_seconds']):.2f} s",
            ],
        }
    )

    show_df(table)

    src(BASE_RES)
    src(GWO_RES)

    note(
        "The two configurations are displayed as measured values. "
        "The dashboard does not label either configuration as a winner or as universally superior."
    )


# =============================================================================
# PAGE 4 — CONVERGENCE
# =============================================================================

def page_convergence():
    page_header(
        "GWO Convergence",
        "Recorded best-fitness behavior across optimization iterations.",
    )

    conv = load_csv(GWO_CONV)

    if conv is None:
        missing(
            GWO_CONV,
            "The convergence file is generated by the GWO execution.",
        )
        return

    required = {"iteration", "best_fitness"}

    if not required.issubset(conv.columns):
        st.error(
            f"Convergence file is missing required columns: {required - set(conv.columns)}"
        )
        return

    fig = go.Figure(
        go.Scatter(
            x=conv["iteration"],
            y=conv["best_fitness"],
            mode="lines+markers",
            line=dict(color=TEAL, width=3),
            marker=dict(size=7),
            hovertemplate="Iteration %{x}<br>Best fitness %{y:.6f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="GWO Best Fitness by Iteration",
        xaxis_title="Iteration",
        yaxis_title="Best Fitness",
    )

    show_fig(style_fig(fig, 430))

    best_idx = conv["best_fitness"].idxmin()
    best_row = conv.loc[best_idx]

    cards(
        [
            (
                "Best Fitness",
                f"{float(best_row['best_fitness']):.6f}",
                "Lowest recorded fitness",
                "teal",
            ),
            (
                "Best Iteration",
                str(int(best_row["iteration"])),
                "Iteration containing best fitness",
                "",
            ),
        ],
        per_row=2,
    )

    src(GWO_CONV)

    note(
        "The best recorded fitness is identified directly from the convergence CSV. "
        "For the seed-1 development experiment, the best recorded fitness was reached "
        "in the first iteration and remained unchanged in subsequent iterations."
    )


# =============================================================================
# PAGE 5 — STABILITY
# =============================================================================

def page_stability():
    page_header(
        "Experimental Stability",
        "Development experiments conducted using different random seeds.",
    )

    refs = load_csv(REFS)

    if refs is None:
        missing(
            REFS,
            "The development seed experiment summary is not available.",
        )
        return

    section("Different random seeds")

    table = pd.DataFrame(
        {
            "Seed": refs["seed"].astype(int),
            "Features": refs.apply(
                lambda r: f"{int(r['selected_features'])}/{int(r['total_features'])}",
                axis=1,
            ),
            "Reduction": refs["feature_reduction_percent"].map(
                lambda x: f"{float(x):.2f}%"
            ),
            "Accuracy": refs["accuracy"].map(pct),
            "Recall": refs["recall"].map(pct),
            "F1": refs["f1"].map(pct),
            "Runtime": refs["runtime_seconds"].map(
                lambda x: f"{float(x):.2f} s"
            ),
        }
    )

    show_df(table)
    src(REFS)

    note(
        "These are development experiments conducted using different random seeds. "
        "They are shown to illustrate variation in selected subsets and measured "
        "test metrics, not as a final benchmark."
    )

    section("Selected feature count across seeds")

    fig = go.Figure(
        go.Bar(
            x=refs["seed"].astype(str),
            y=refs["selected_features"],
            marker_color=TEAL,
            text=refs["selected_features"],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Number of Selected Features",
        xaxis_title="Random seed",
        yaxis_title="Selected features",
    )

    show_fig(style_fig(fig, 360))

    iteration_refs = load_csv(ITER_REFS)

    if iteration_refs is not None:
        section("Iteration experiment — seed 1")

        iteration_table = pd.DataFrame(
            {
                "Iterations": iteration_refs["iterations"].astype(int),
                "Fitness": iteration_refs["fitness"].map(
                    lambda x: f"{float(x):.6f}"
                ),
                "Selected Features": iteration_refs["selected_features"].astype(int),
                "Accuracy": iteration_refs["accuracy"].map(pct),
                "Recall": iteration_refs["recall"].map(pct),
                "F1": iteration_refs["f1"].map(pct),
            }
        )

        show_df(iteration_table)
        src(ITER_REFS)

        note(
            "The best recorded fitness was reached in the first iteration and remained "
            "unchanged in the subsequent iterations in this experimental configuration."
        )


# =============================================================================
# PAGE 6 — DIAGNOSTICS
# =============================================================================

def confusion_fig(pred_df, title):
    y = pred_df["actual"].to_numpy()
    p = pred_df["predicted"].to_numpy()

    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tp = int(((y == 1) & (p == 1)).sum())

    z = np.array(
        [
            [tn, fp],
            [fn, tp],
        ]
    )

    rowsum = z.sum(axis=1, keepdims=True)

    txt = [
        [
            f"<b>{z[i, j]:,}</b><br>"
            f"{z[i, j] / max(rowsum[i, 0], 1) * 100:.1f}% "
            f"of actual {['Normal', 'Attack'][i]}"
            for j in range(2)
        ]
        for i in range(2)
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Predicted Normal", "Predicted Attack"],
            y=["Actual Normal", "Actual Attack"],
            text=txt,
            texttemplate="%{text}",
            colorscale=[
                [0, "#eef3f9"],
                [1, NAVY],
            ],
            showscale=False,
            hoverinfo="skip",
            textfont=dict(size=14),
        )
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title)

    return style_fig(fig, 380), (tn, fp, fn, tp)


def page_diagnostics():
    page_header(
        "Confusion Matrix & Feature Importance",
        "Computed from actual predictions and the trained Random Forest.",
    )

    pred = load_csv(GWO_PRED)

    section("Confusion matrix — GWO + Random Forest")

    if pred is None:
        missing(
            GWO_PRED,
            "Predictions are written when the IDS is run. "
            "Use Run GWO IDS or the CLI to generate them. "
            "No confusion matrix is estimated from the accuracy value.",
        )
    else:
        fig, (tn, fp, fn, tp) = confusion_fig(pred, "GWO + Random Forest")

        left, right = st.columns([3, 2])

        with left:
            show_fig(fig)
            src(GWO_PRED)

        with right:
            cards(
                [
                    (
                        "True Negatives",
                        f"{tn:,}",
                        "Normal → Normal",
                        "slate",
                    ),
                    (
                        "False Positives",
                        f"{fp:,}",
                        "Normal → Attack",
                        "amber",
                    ),
                    (
                        "False Negatives",
                        f"{fn:,}",
                        "Attack → Normal",
                        "amber",
                    ),
                    (
                        "True Positives",
                        f"{tp:,}",
                        "Attack → Attack",
                        "teal",
                    ),
                ],
                per_row=2,
            )

        total = tn + fp + fn + tp

        g = first_row(load_csv(GWO_RES))

        text = (
            f"Of {total:,} test records, "
            f"{fn:,} attack records were predicted as Normal and "
            f"{fp:,} normal records were predicted as Attack."
        )

        if g is not None and total > 0:
            matrix_accuracy = (tn + tp) / total

            if abs(matrix_accuracy - float(g["accuracy"])) < 1e-9:
                text += " The accuracy implied by this matrix matches the results file."
            else:
                text += (
                    " Note: the accuracy implied by this matrix does not match "
                    "the results file; the files may come from different runs."
                )

        note(text)

    section("Random Forest Feature Importance")

    imp = load_csv(GWO_IMP)

    if imp is None:
        missing(
            GWO_IMP,
            "Feature importances are saved from the trained Random Forest when the IDS is run.",
        )
    else:
        d = imp.sort_values("importance")

        fig = go.Figure(
            go.Bar(
                x=d["importance"],
                y=d["feature"],
                orientation="h",
                marker_color=TEAL,
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Random Forest Feature Importance",
            xaxis_title="Importance",
        )

        show_fig(
            style_fig(
                fig,
                max(340, 24 * len(d) + 120),
            )
        )

        src(GWO_IMP)

        st.caption(
            "Random Forest feature importance describes how the final classifier "
            "uses the selected features. It is not a measure of feature importance "
            "within the GWO search."
        )


# =============================================================================
# PAGE 7 — RUN IDS
# =============================================================================

def backup_results():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dest = RESULTS / f"backup_{stamp}"

    files = [
        p for p in RESULTS.glob("*")
        if p.is_file()
    ]

    if not files:
        return None

    dest.mkdir(
        parents=True,
        exist_ok=True,
    )

    for p in files:
        shutil.copy2(
            p,
            dest / p.name,
        )

    return dest


def run_with_progress(mode, kwargs):
    bar = st.progress(0.0)

    status = st.status(
        "Starting…",
        expanded=True,
    )

    t0 = time.time()

    def progress(stage, message, fraction):
        bar.progress(
            min(
                max(
                    float(fraction),
                    0.0,
                ),
                1.0,
            )
        )

        status.update(
            label=message,
        )

        status.write(
            f"`{time.time() - t0:6.1f}s` &nbsp; {message}"
        )

    try:
        fn = (
            gwo_ids.run_gwo
            if mode == "gwo"
            else gwo_ids.run_baseline
        )

        result = fn(
            progress=progress,
            results_dir=str(RESULTS),
            **kwargs,
        )

    except Exception as e:
        status.update(
            label="Run failed",
            state="error",
        )

        st.error(
            f"The run failed: {type(e).__name__}: {e}"
        )

        return None

    status.update(
        label=(
            f"Completed in "
            f"{time.time() - t0:.1f} s — "
            f"results saved to results/"
        ),
        state="complete",
        expanded=False,
    )

    return result


def page_run():
    page_header(
        "Run IDS",
        "Execute the existing GWO + Random Forest implementation from the dashboard.",
    )

    note(
        "Each run calls the same functions used by the command-line interface. "
        "Progress messages correspond to actual execution stages."
    )

    section("Configuration")

    c1, c2 = st.columns(2)

    train = c1.text_input(
        "Training dataset path",
        DEFAULT_TRAIN,
    )

    test = c2.text_input(
        "Testing dataset path",
        DEFAULT_TEST,
    )

    a, b, c, d, e = st.columns(5)

    sample = a.number_input(
        "Training sample size",
        min_value=500,
        max_value=125000,
        value=5000,
        step=500,
    )

    pop = b.number_input(
        "GWO population",
        min_value=3,
        max_value=50,
        value=5,
        step=1,
    )

    iters = c.number_input(
        "Iterations",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
    )

    trees = d.number_input(
        "Random Forest trees",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
    )

    seed = e.number_input(
        "Random seed",
        min_value=0,
        max_value=10000,
        value=42,
        step=1,
    )

    backup = st.checkbox(
        "Back up current result files before running",
        value=True,
        help="Running overwrites the current GWO or baseline result files.",
    )

    def resolve(p):
        pp = Path(p)

        if pp.is_absolute():
            return pp

        return ROOT / pp

    train_p = resolve(train)
    test_p = resolve(test)

    ok = True

    for label, p in [
        ("Training", train_p),
        ("Testing", test_p),
    ]:
        if not p.exists():
            st.error(
                f"{label} dataset not found: {p}. "
                "Place the NSL-KDD ARFF files in the data/ folder."
            )

            ok = False

    st.markdown(
        "<div style='height:6px'></div>",
        unsafe_allow_html=True,
    )

    b1, b2, _ = st.columns([1.2, 1.4, 3])

    run_gwo_clicked = b1.button(
        "Run GWO IDS",
        type="primary",
        disabled=not ok,
        use_container_width=True,
    )

    run_base_clicked = b2.button(
        "Run All-Feature Baseline",
        disabled=not ok,
        use_container_width=True,
    )

    if run_gwo_clicked or run_base_clicked:

        if backup:
            dest = backup_results()

            if dest:
                st.caption(
                    f"Previous results backed up to `{rel(dest)}`."
                )

        common = dict(
            train_arff=str(train_p),
            test_arff=str(test_p),
            sample=int(sample),
            trees=int(trees),
            seed=int(seed),
        )

        if run_gwo_clicked:
            res = run_with_progress(
                "gwo",
                dict(
                    common,
                    pop=int(pop),
                    iters=int(iters),
                ),
            )

            mode = "GWO + Random Forest"

        else:
            res = run_with_progress(
                "baseline",
                common,
            )

            mode = "All-feature Random Forest baseline"

        if res is not None:
            clean_result = {}

            for key, value in res.items():

                if isinstance(
                    value,
                    (
                        int,
                        np.integer,
                    ),
                ):
                    clean_result[key] = int(value)

                else:
                    try:
                        clean_result[key] = float(value)
                    except Exception:
                        clean_result[key] = value

            st.session_state["last_run"] = (
                mode,
                clean_result,
                datetime.now().strftime("%H:%M:%S"),
            )

    last = st.session_state.get("last_run")

    if last:
        mode, r, at = last

        section(
            f"Latest run from this session — {mode} ({at})"
        )

        items = [
            (
                "Features",
                f"{r['selected_features']}/{len(gwo_ids.FEATURE_NAMES)}",
                f"{r['feature_reduction_percent']:.2f}% reduction",
                "teal",
            ),
            (
                "Accuracy",
                pct(r["accuracy"]),
                "",
                "",
            ),
            (
                "Precision",
                pct(r["precision"]),
                "",
                "",
            ),
            (
                "Recall",
                pct(r["recall"]),
                "",
                "",
            ),
            (
                "F1-score",
                pct(r["f1"]),
                "",
                "",
            ),
            (
                "Runtime",
                f"{r['runtime_seconds']:.2f} s",
                "",
                "slate",
            ),
        ]

        if "fitness" in r:
            items.append(
                (
                    "Best fitness",
                    f"{r['fitness']:.6f}",
                    f"{r['unique_fitness_evaluations']} unique evaluations",
                    "",
                )
            )

        cards(
            items,
            per_row=4,
        )

        note(
            "Results have been saved. The other dashboard pages now read "
            "the updated files.",
            "ok",
        )

    refs = load_csv(REFS)

    if refs is not None:
        with st.expander(
            "Recorded development reference values"
        ):
            table = pd.DataFrame(
                {
                    "Run": refs["description"],
                    "Features": refs.apply(
                        lambda r:
                            f"{int(r['selected_features'])}/"
                            f"{int(r['total_features'])}",
                        axis=1,
                    ),
                    "Accuracy": refs["accuracy"].map(pct),
                    "Precision": refs["precision"].map(pct),
                    "Recall": refs["recall"].map(pct),
                    "F1": refs["f1"].map(pct),
                    "Runtime (s)": refs["runtime_seconds"].map(
                        lambda v:
                            f"{float(v):.2f}"
                    ),
                }
            )

            show_df(table)
            src(REFS)

            st.caption(
                "These values were recorded during development. "
                "Re-running on another machine or with different library "
                "versions may produce different results."
            )


# =============================================================================
# TECHNICAL DETAILS
# =============================================================================

def render_technical_details():
    with st.expander("Implementation Details"):

        st.markdown(
            """
            ### Dataset

            **NSL-KDD**

            ### Input features

            **41**

            ### Categorical features

            - `protocol_type`
            - `service`
            - `flag`

            ### Feature-selection algorithm

            **Binary Grey Wolf Optimization**

            ### Classifier

            **Random Forest**

            ### Validation

            **80/20 stratified holdout** for GWO fitness evaluation.

            ### Test dataset

            **KDDTest+**

            ### Classification

            Binary:

            - Normal
            - Attack

            ### Purpose

            This dashboard demonstrates the existing implementation.
            It is not presented as the final proposed or novel IDS.
            """
        )

        try:
            source = inspect.getsource(
                gwo_ids.BinaryGWO.fitness
            )

            st.markdown("### Fitness function used by the implementation")

            st.code(
                source,
                language="python",
            )

        except Exception:
            st.info(
                "The fitness-function source could not be displayed."
            )


# =============================================================================
# APP HEADER + NAVIGATION
# =============================================================================

if "page" not in st.session_state:
    st.session_state["page"] = PAGES[0]

render_sidebar()
render_hero()
render_navigation()

page = st.session_state["page"]

# =============================================================================
# ROUTER
# =============================================================================

ROUTES = {
    PAGES[0]: page_overview,
    PAGES[1]: page_features,
    PAGES[2]: page_performance,
    PAGES[3]: page_convergence,
    PAGES[4]: page_stability,
    PAGES[5]: page_diagnostics,
    PAGES[6]: page_run,
}

ROUTES[page]()

render_technical_details()

st.markdown(
    """
    <div class="footer">
        Intelligent Network Intrusion Detection Using Metaheuristic Optimization
        &nbsp;·&nbsp;
        Demonstration of an existing GWO + Random Forest implementation
        &nbsp;·&nbsp;
        All displayed results are read from project result files
    </div>
    """,
    unsafe_allow_html=True,
)
