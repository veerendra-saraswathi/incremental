"""
Quick test of the Online Autoencoder detector.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from incremental_hils.synchronization.aligner import align_to_common_rate
from incremental_hils.self_supervised.autoencoder import OnlineAutoencoderDetector
from incremental_hils.root_cause.analyzer import RootCauseAnalyzer

console = Console()


def main():
    console.rule("[bold blue]Self-Supervised Autoencoder Test")

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
    aligned = aligned[feature_cols]

    input_dim = aligned.shape[1]
    console.print(f"Input dimension: {input_dim}")

    detector = OnlineAutoencoderDetector(
        input_dim=input_dim,
        latent_dim=12,
        hidden_dims=[64, 32],
        buffer_size=400,
        train_every=40,
    )
    rca = RootCauseAnalyzer()

    warmup = 1000
    errors = []
    anomalies = []

    for i, (_, row) in enumerate(aligned.iterrows()):
        x = row.to_dict()
        error = detector.learn_one(x)
        errors.append(error)

        if i < warmup:
            continue

        threshold = np.percentile(errors[warmup // 2 :], 98.5)
        if error > threshold:
            feat_rc = detector.basic_root_cause(top_k=5)
            sub_rc = rca.aggregate_to_subsystem(feat_rc, min_score=0.0)
            anomalies.append((i, error, sub_rc[:2] if sub_rc else []))

    console.print(f"\nAnomalies flagged: {len(anomalies)}")
    for idx, err, subs in anomalies[:8]:
        console.print(f"  idx={idx:5d} | error={err:.5f} | {subs}")

    console.print("\n[green]Autoencoder test finished[/]")


if __name__ == "__main__":
    main()
    