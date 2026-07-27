"""
Improved streaming demo with adaptive threshold.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from incremental_hils.synchronization.aligner import align_to_common_rate
from incremental_hils.anomaly_detection.detector import IncrementalAnomalyDetector
from incremental_hils.root_cause.analyzer import RootCauseAnalyzer

console = Console()


def main():
    console.rule("[bold blue]Incremental HILS – Improved Stream Demo")

    sample_dir = ROOT / "data" / "sample"
    data = {
        "imu": pd.read_csv(sample_dir / "imu.csv"),
        "actuator": pd.read_csv(sample_dir / "actuator.csv"),
        "fms": pd.read_csv(sample_dir / "fms.csv"),
        "guidance": pd.read_csv(sample_dir / "guidance.csv"),
        "telemetry": pd.read_csv(sample_dir / "telemetry.csv"),
    }

    console.print("Aligning multi-rate data...")
    aligned = align_to_common_rate(data, target_freq="10ms")
    feature_cols = [c for c in aligned.columns if "feat_" in c]
    aligned = aligned[feature_cols]
    console.print(f"Aligned shape: {aligned.shape}")

    detector = IncrementalAnomalyDetector(n_trees=25, height=8)
    rca = RootCauseAnalyzer()

    warmup = 800
    scores = []
    anomaly_count = 0

    console.print(f"\nProcessing {len(aligned)} samples (warmup={warmup})...\n")

    for i, (_, row) in enumerate(aligned.iterrows()):
        x = row.to_dict()
        score = detector.learn_one(x)
        scores.append(score)

        if i < warmup:
            continue

        # Adaptive threshold: 99th percentile of scores seen so far
        threshold = np.percentile(scores[warmup//2:], 99.0)

        if score > threshold:
            anomaly_count += 1
            feat_rc = detector.basic_root_cause(x, top_k=4)
            sub_rc = rca.aggregate_to_subsystem(feat_rc)

            console.print(
                f"[red]Anomaly[/] @ {i:5d} | score={score:.3f} | thr={threshold:.3f} | "
                f"subsystems → {sub_rc[:2]}"
            )

    console.print(f"\n[green]Demo finished.[/] Anomalies flagged: {anomaly_count}")


if __name__ == "__main__":
    main()