"""
Evaluate anomaly detection + root-cause and save results.
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from incremental_hils.synchronization.aligner import align_to_common_rate
from incremental_hils.anomaly_detection.detector import IncrementalAnomalyDetector
from incremental_hils.root_cause.analyzer import RootCauseAnalyzer
from incremental_hils.evaluation.reporter import ResultReporter

console = Console()


def load_ground_truth(metadata_path: Path, total_samples: int, freq_ms: int = 10):
    with open(metadata_path) as f:
        meta = json.load(f)

    gt_ranges = []
    for sub, info in meta["subsystems"].items():
        for start_s, end_s, atype, _ in info.get("anomalies", []):
            start_idx = int(start_s * 1000 / freq_ms)
            end_idx = int(end_s * 1000 / freq_ms)
            start_idx = max(0, min(start_idx, total_samples - 1))
            end_idx = max(0, min(end_idx, total_samples - 1))
            gt_ranges.append({
                "start": start_idx,
                "end": end_idx,
                "subsystem": sub,
                "type": atype,
            })
    return gt_ranges


def main():
    console.rule("[bold blue]Detection + Root-Cause Evaluation + Reporting")

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
    n_samples = len(aligned)

    gt_ranges = load_ground_truth(sample_dir / "metadata.json", n_samples)
    console.print(f"Aligned samples: {n_samples}")
    console.print(f"Ground-truth anomaly intervals: {len(gt_ranges)}\n")

    detector = IncrementalAnomalyDetector(n_trees=25, height=8)
    rca = RootCauseAnalyzer()
    reporter = ResultReporter(output_dir=ROOT / "outputs" / "reports")

    warmup = 800
    scores = []
    detections = []  # (index, score, top_subsystem)

    for i, (_, row) in enumerate(aligned.iterrows()):
        x = row.to_dict()
        score = detector.learn_one(x)
        scores.append(score)

        if i < warmup:
            continue

        threshold = np.percentile(scores[warmup // 2 :], 99.0)
        if score > threshold:
            feat_rc = detector.basic_root_cause(x, top_k=6)
            sub_rc = rca.aggregate_to_subsystem(feat_rc, min_score=1.2)
            top_sub = sub_rc[0][0] if sub_rc else "unknown"
            detections.append({
                "index": i,
                "score": float(score),
                "subsystem": top_sub,
            })

    detection_map = {d["index"]: d["subsystem"] for d in detections}

    # Evaluation table
    table = Table(title="Detection + Root-Cause Results")
    table.add_column("Subsystem", style="cyan")
    table.add_column("Type")
    table.add_column("GT Range")
    table.add_column("Detected?", justify="center")
    table.add_column("Hits")
    table.add_column("Correct Root-Cause?", justify="center")
    table.add_column("Most common predicted")

    detected_intervals = 0
    correct_rca = 0

    for gt in gt_ranges:
        start, end = gt["start"], gt["end"]
        true_sub = gt["subsystem"]

        hits = [detection_map[i] for i in range(start, end + 1) if i in detection_map]
        found = len(hits) > 0
        if found:
            detected_intervals += 1

        if hits:
            most_common = Counter(hits).most_common(1)[0]
            predicted_sub, count = most_common
            rca_correct = predicted_sub == true_sub
            if rca_correct:
                correct_rca += 1
            rca_text = "[green]YES[/]" if rca_correct else f"[red]NO[/] ({predicted_sub})"
            most_common_text = f"{predicted_sub} ({count})"
        else:
            rca_text = "-"
            most_common_text = "-"

        table.add_row(
            true_sub,
            gt["type"],
            f"{start}-{end}",
            "[green]YES[/]" if found else "[red]NO[/]",
            str(len(hits)),
            rca_text,
            most_common_text,
        )

    console.print(table)

    # Summary
    summary = {
        "total_samples": n_samples,
        "warmup_samples": warmup,
        "ground_truth_intervals": len(gt_ranges),
        "intervals_detected": detected_intervals,
        "detection_rate": f"{detected_intervals / len(gt_ranges):.0%}",
        "root_cause_correct": correct_rca,
        "root_cause_accuracy": f"{correct_rca / max(detected_intervals, 1):.0%}",
        "total_anomaly_points": len(detections),
    }

    console.print("\n[bold]Summary[/]")
    for k, v in summary.items():
        console.print(f"  {k:25s}: {v}")

    # Save results
    csv_path = reporter.save_anomalies(detections, prefix="hils_anomalies")
    summary_path = reporter.save_summary(summary, prefix="hils_summary")

    console.print(f"\n[green]Results saved:[/]")
    console.print(f"  Anomalies → {csv_path}")
    console.print(f"  Summary   → {summary_path}")


if __name__ == "__main__":
    main()
    