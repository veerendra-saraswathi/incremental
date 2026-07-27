"""
Simple Streamlit GUI for Incremental HILS Anomaly Detection.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from incremental_hils.synchronization.aligner import align_to_common_rate
from incremental_hils.anomaly_detection.detector import IncrementalAnomalyDetector
from incremental_hils.root_cause.analyzer import RootCauseAnalyzer


st.set_page_config(
    page_title="HILS Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("Incremental HILS Anomaly Detection")
st.markdown("Online anomaly detection + root-cause identification for multi-rate HILS data")


@st.cache_data
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
    return aligned[feature_cols]


def run_detection(df: pd.DataFrame, warmup: int = 800, percentile: float = 99.0):
    detector = IncrementalAnomalyDetector(n_trees=25, height=8)
    rca = RootCauseAnalyzer()

    scores = []
    anomalies = []

    progress = st.progress(0)
    status = st.empty()

    n = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        x = row.to_dict()
        score = detector.learn_one(x)
        scores.append(score)

        if i >= warmup:
            threshold = np.percentile(scores[warmup // 2 :], percentile)
            if score > threshold:
                feat_rc = detector.basic_root_cause(x, top_k=6)
                sub_rc = rca.aggregate_to_subsystem(feat_rc, min_score=1.2)
                top_sub = sub_rc[0][0] if sub_rc else "unknown"
                anomalies.append({
                    "index": i,
                    "score": score,
                    "subsystem": top_sub,
                })

        if i % 500 == 0:
            progress.progress(min(i / n, 1.0))
            status.text(f"Processing sample {i}/{n}")

    progress.progress(1.0)
    status.text("Done")
    return np.array(scores), anomalies


def main():
    st.sidebar.header("Settings")
    warmup = st.sidebar.slider("Warmup samples", 200, 2000, 800, 100)
    percentile = st.sidebar.slider("Threshold percentile", 95.0, 99.9, 99.0, 0.1)

    if st.sidebar.button("Run Detection", type="primary"):
        with st.spinner("Loading and aligning multi-rate data..."):
            df = load_and_align_data()

        st.success(f"Aligned data shape: {df.shape}")

        scores, anomalies = run_detection(df, warmup=warmup, percentile=percentile)

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", f"{len(df):,}")
        col2.metric("Anomalies Detected", len(anomalies))
        col3.metric("Anomaly Rate", f"{len(anomalies)/len(df)*100:.2f}%")

        # Score plot
        st.subheader("Anomaly Score Over Time")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(scores, linewidth=0.8, color="steelblue", label="Score")
        if anomalies:
            idxs = [a["index"] for a in anomalies]
            scs = [a["score"] for a in anomalies]
            ax.scatter(idxs, scs, color="red", s=18, zorder=5, label="Anomaly")
        ax.axvline(warmup, color="gray", linestyle="--", label="Warmup end")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Anomaly Score")
        ax.legend()
        st.pyplot(fig)

        # Anomaly table
        if anomalies:
            st.subheader("Detected Anomalies")
            anom_df = pd.DataFrame(anomalies)
            st.dataframe(anom_df, use_container_width=True)

            # Subsystem breakdown
            st.subheader("Root-Cause Distribution")
            counts = anom_df["subsystem"].value_counts()
            st.bar_chart(counts)
        else:
            st.info("No anomalies detected with current settings.")

    else:
        st.info("👈 Adjust settings and click **Run Detection** to start.")


if __name__ == "__main__":
    main()
    