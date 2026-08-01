"""
HILS Anomaly Detection – Mission Control Dashboard
Professional Streamlit interface with RCI branding.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from collections import Counter

# ----------------------------------------------------------------------
# Path setup
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from incremental_hils.synchronization.aligner import align_to_common_rate
from incremental_hils.anomaly_detection.detector import IncrementalAnomalyDetector
from incremental_hils.root_cause.analyzer import RootCauseAnalyzer

# ----------------------------------------------------------------------
# Page Config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="HILS Anomaly Detection | Mission Control",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .status-healthy {
        background: linear-gradient(90deg, #0d9488, #14b8a6);
        color: white;
        padding: 0.7rem 1.2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        font-size: 1.15rem;
        text-align: center;
    }
    .status-anomaly {
        background: linear-gradient(90deg, #b91c1c, #ef4444);
        color: white;
        padding: 0.7rem 1.2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        font-size: 1.15rem;
        text-align: center;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Data Loading
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_align_data():
    sample_dir = ROOT / "data" / "sample"
    data = {
        "imu": pd.read_csv(sample_dir / "imu.csv"),
        "actuator": pd.read_csv(sample_dir / "actuator.csv"),
        "fms": pd.read_csv(sample_dir / "fms.csv"),
        "guidance": pd.read_csv(sample_dir / "guidance.csv"),
        "telemetry": pd.read_csv(sample_dir / "telemetry.csv"),
    }
    aligned = align_to_common_rate(data, target_freq="10ms")
    feature_cols = [c for c in aligned.columns if "feat_" in c]
    return aligned[feature_cols].copy()


# ----------------------------------------------------------------------
# Detection Engine
# ----------------------------------------------------------------------
def run_full_detection(df: pd.DataFrame, warmup: int = 800, percentile: float = 99.0):
    detector = IncrementalAnomalyDetector(n_trees=25, height=8)
    rca = RootCauseAnalyzer()

    scores = []
    thresholds = []
    anomalies = []

    n = len(df)
    progress = st.progress(0, text="Running incremental detection...")

    for i, (_, row) in enumerate(df.iterrows()):
        x = row.to_dict()
        score = detector.learn_one(x)
        scores.append(score)

        if i >= warmup:
            recent = scores[max(warmup, i - 2000):]
            thr = float(np.percentile(recent, percentile))
            thresholds.append(thr)

            if score > thr:
                feat_rc = detector.basic_root_cause(x, top_k=6)
                sub_rc = rca.aggregate_to_subsystem(feat_rc)

                top_sub = sub_rc[0][0] if sub_rc else "unknown"
                top_conf = sub_rc[0][1] if sub_rc else 0.0

                anomalies.append({
                    "index": i,
                    "score": round(score, 4),
                    "threshold": round(thr, 4),
                    "subsystem": top_sub,
                    "confidence": round(top_conf, 2),
                })
        else:
            thresholds.append(np.nan)

        if i % 600 == 0:
            progress.progress(min(i / n, 1.0), text=f"Processing {i:,} / {n:,}")

    progress.progress(1.0, text="Detection complete")
    progress.empty()

    return np.array(scores), np.array(thresholds), anomalies


# ----------------------------------------------------------------------
# Subsystem Health
# ----------------------------------------------------------------------
def compute_subsystem_health(anomalies):
    if not anomalies:
        return {s: 96 for s in ["imu", "actuator", "fms", "guidance", "telemetry"]}

    counts = Counter([a["subsystem"] for a in anomalies])
    total = len(anomalies)

    health = {}
    for sub in ["imu", "actuator", "fms", "guidance", "telemetry"]:
        ratio = counts.get(sub, 0) / total
        score = max(58, 100 - ratio * 85)
        health[sub] = round(score)
    return health


# ----------------------------------------------------------------------
# Main App
# ----------------------------------------------------------------------
def main():
    # ========== HEADER WITH LOGO ==========
    col_logo, col_title = st.columns([1, 7])

    with col_logo:
        try:
            st.image("assets/RCI.webp", width=250)
        except Exception:
            st.markdown("### 🛡️")

    with col_title:
        st.markdown("## HILS Anomaly Detection — Mission Control")
        st.caption("Incremental Learning • Multi-rate Sensor Streams • Subsystem Root-Cause Analysis  |  Research Centre Imarat (RCI)")

    st.markdown("---")

    # ========== SIDEBAR ==========
    with st.sidebar:
        st.header("⚙️ Configuration")
        warmup = st.slider("Warm-up samples", 400, 2000, 800, 100)
        percentile = st.slider("Threshold percentile", 97.0, 99.9, 99.0, 0.1)
        min_conf = st.slider("Min confidence to display", 1.5, 3.0, 1.8, 0.1)

        st.markdown("---")
        st.markdown("**System Info**")
        st.markdown("""
        - Detection: Hybrid (HST + Statistical)  
        - Learning: Fully incremental / online  
        - Alignment: Multi-rate → 10 ms  
        - Root-cause: Feature → Subsystem  
        """)

        run_btn = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

    # ========== IDLE STATE ==========
    if not run_btn:
        st.info("Configure parameters on the left and click **Run Full Analysis** to begin.")
        st.markdown("### Dashboard Features")
        st.markdown("""
        - Overall system status (Healthy / Anomaly Detected)
        - Anomaly score + adaptive threshold graph
        - Root-cause distribution
        - Per-subsystem health indicators
        - Detailed event log with download
        - Confidence distribution
        """)
        return

    # ========== RUN DETECTION ==========
    with st.spinner("Loading and aligning multi-rate data..."):
        df = load_and_align_data()

    scores, thresholds, raw_anomalies = run_full_detection(df, warmup=warmup, percentile=percentile)

    # Filter by confidence
    anomalies = [a for a in raw_anomalies if a["confidence"] >= min_conf]

    # ========== STATUS BANNER ==========
    if anomalies:
        st.markdown(
            f'<div class="status-anomaly">⚠ ANOMALY DETECTED — {len(anomalies)} events flagged</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-healthy">✓ SYSTEM HEALTHY — No significant anomalies</div>',
            unsafe_allow_html=True
        )

    st.markdown("")

    # ========== TOP METRICS ==========
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Samples", f"{len(df):,}")
    m2.metric("Anomalies", len(anomalies))
    m3.metric("Anomaly Rate", f"{len(anomalies)/max(len(df)-warmup,1)*100:.2f}%")
    m4.metric("Warm-up", f"{warmup:,}")
    m5.metric("Avg Confidence", f"{np.mean([a['confidence'] for a in anomalies]):.2f}" if anomalies else "—")

    st.markdown("---")

    # ========== MAIN CHARTS ==========
    left, right = st.columns([1.65, 1])

    with left:
        st.subheader("📈 Anomaly Score & Adaptive Threshold")
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=scores, mode="lines", name="Anomaly Score",
            line=dict(color="#38bdf8", width=1.3), opacity=0.9
        ))
        fig.add_trace(go.Scatter(
            y=thresholds, mode="lines", name="Adaptive Threshold",
            line=dict(color="#f97316", width=1.6, dash="dash")
        ))

        if anomalies:
            idxs = [a["index"] for a in anomalies]
            scs = [a["score"] for a in anomalies]
            fig.add_trace(go.Scatter(
                x=idxs, y=scs, mode="markers", name="Detected Anomaly",
                marker=dict(color="#ef4444", size=7, symbol="diamond")
            ))

        fig.add_vline(x=warmup, line_dash="dot", line_color="#94a3b8",
                      annotation_text="Warm-up End", annotation_position="top left")

        fig.update_layout(
            height=390,
            margin=dict(l=20, r=20, t=40, b=30),
            legend=dict(orientation="h", y=1.12),
            xaxis_title="Sample Index",
            yaxis_title="Score",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("🎯 Root-Cause Distribution")
        if anomalies:
            counts = Counter([a["subsystem"] for a in anomalies])
            labels = [k.upper() for k in counts.keys()]
            values = list(counts.values())

            fig2 = go.Figure(go.Bar(
                x=values, y=labels, orientation="h",
                marker_color=["#38bdf8", "#a78bfa", "#34d399", "#f97316", "#f43f5e"][:len(labels)],
                text=values, textposition="auto"
            ))
            fig2.update_layout(
                height=390,
                margin=dict(l=20, r=20, t=40, b=30),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Count",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No anomalies detected.")

    st.markdown("---")

    # ========== SUBSYSTEM HEALTH ==========
    st.subheader("🩺 Subsystem Health")
    health = compute_subsystem_health(anomalies)

    cols = st.columns(5)
    for col, sub in zip(cols, ["imu", "actuator", "fms", "guidance", "telemetry"]):
        score = health.get(sub, 90)
        if score >= 88:
            color, status = "#22c55e", "Healthy"
        elif score >= 72:
            color, status = "#eab308", "Watch"
        else:
            color, status = "#ef4444", "Attention"

        with col:
            st.markdown(f"**{sub.upper()}**")
            st.progress(score / 100)
            st.markdown(f"<span style='color:{color}; font-weight:600'>{score}% — {status}</span>",
                        unsafe_allow_html=True)

    st.markdown("---")

    # ========== EVENT LOG + CONFIDENCE ==========
    log_col, conf_col = st.columns([1.55, 1])

    with log_col:
        st.subheader("📋 Anomaly Event Log")
        if anomalies:
            event_df = pd.DataFrame(anomalies)
            event_df = event_df.rename(columns={
                "index": "Sample",
                "score": "Score",
                "threshold": "Threshold",
                "subsystem": "Root Cause",
                "confidence": "Confidence"
            })
            st.dataframe(event_df.tail(20), use_container_width=True, height=300)

            csv = event_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Full Event Log (CSV)",
                data=csv,
                file_name=f"hils_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.success("No anomaly events recorded.")

    with conf_col:
        st.subheader("📊 Confidence Distribution")
        if anomalies:
            conf_values = [a["confidence"] for a in anomalies]
            fig3 = go.Figure(go.Histogram(
                x=conf_values, nbinsx=10,
                marker_color="#a78bfa", opacity=0.85
            ))
            fig3.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=20, b=20),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Confidence",
                yaxis_title="Count",
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No confidence data.")

    # Footer
    st.markdown("---")
    st.caption(
        f"HILS Anomaly Detection Dashboard  •  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  •  "
        f"Research Centre Imarat (RCI)  •  Offline / Air-gapped capable"
    )


if __name__ == "__main__":
    main()
