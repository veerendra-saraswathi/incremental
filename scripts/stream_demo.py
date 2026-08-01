"""
Refined Streaming Demo – Incremental HILS Anomaly Detection
Cleaner and more professional version for live technical demonstration.
"""

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from incremental_hils.synchronization.aligner import align_to_common_rate
from incremental_hils.anomaly_detection.detector import IncrementalAnomalyDetector
from incremental_hils.root_cause.analyzer import RootCauseAnalyzer

console = Console()


def print_header():
    console.print()
    console.print(Panel.fit(
        "[bold blue]Incremental HILS Anomaly Detection[/bold blue]\n"
        "[white]Online / Streaming Demonstration[/white]\n"
        "[dim]Multi-rate • Adaptive Threshold • Subsystem Root-Cause[/dim]",
        border_style="blue"
    ))
    console.print()


def main():
    print_header()

    # -------------------------------------------------
    # 1. Load & Align
    # -------------------------------------------------
    console.print("[bold]1.[/bold] Loading and aligning multi-rate HILS data...")
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

    console.print(f"   Aligned shape : [cyan]{aligned.shape[0]:,} × {aligned.shape[1]}[/cyan]")
    console.print(f"   Subsystems    : IMU | Actuator | FMS | Guidance | Telemetry")
    console.print()

    # -------------------------------------------------
    # 2. Initialize
    # -------------------------------------------------
    console.print("[bold]2.[/bold] Initializing models...")
    detector = IncrementalAnomalyDetector(n_trees=25, height=8)
    rca = RootCauseAnalyzer()
    console.print("   Detector + Root-Cause Analyzer ready")
    console.print()

    # -------------------------------------------------
    # 3. Streaming parameters
    # -------------------------------------------------
    warmup = 800
    stability_window = 400          # samples after warm-up that are ignored for reporting
    percentile = 99.0
    min_confidence = 1.8            # only report anomalies above this confidence
    total_samples = len(aligned)

    console.print("[bold]3.[/bold] Streaming parameters")
    console.print(f"   Warm-up samples     : {warmup}")
    console.print(f"   Stability window    : {stability_window} (post warm-up)")
    console.print(f"   Threshold           : {percentile}th percentile (adaptive)")
    console.print(f"   Min confidence      : {min_confidence}")
    console.print(f"   Total samples       : {total_samples:,}")
    console.print()

    scores = []
    anomaly_log = []
    anomaly_count = 0
    last_print_idx = -100

    console.rule("[bold yellow]Live Stream[/bold yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:

        task = progress.add_task("Warm-up phase...", total=total_samples)

        for i, (_, row) in enumerate(aligned.iterrows()):
            x = row.to_dict()
            score = detector.learn_one(x)
            scores.append(score)

            # ---- Warm-up ----
            if i < warmup:
                progress.update(task, completed=i + 1,
                                description=f"[cyan]Warm-up[/cyan]  {i+1}/{warmup}")
                continue

            # ---- Stability window (do not report yet) ----
            if i < warmup + stability_window:
                progress.update(task, completed=i + 1,
                                description=f"[yellow]Stabilizing[/yellow]  {i+1}/{warmup + stability_window}")
                continue

            # ---- Detection phase ----
            recent = scores[max(warmup, i - 2000):]
            threshold = np.percentile(recent, percentile)

            if score > threshold:
                feat_rc = detector.basic_root_cause(x, top_k=5)
                sub_rc = rca.aggregate_to_subsystem(feat_rc, min_score=1.2)

                if sub_rc:
                    top_sub = sub_rc[0][0]
                    top_conf = min(sub_rc[0][1], 5.0)  # cap for display
                else:
                    top_sub = "unknown"
                    top_conf = 0.0

                if top_conf >= min_confidence:
                    anomaly_count += 1
                    anomaly_log.append({
                        "index": i,
                        "score": score,
                        "threshold": threshold,
                        "subsystem": top_sub,
                        "confidence": top_conf
                    })

                    # Print only if enough gap from previous print (reduce noise)
                    if i - last_print_idx >= 8:
                        console.print(
                            f"[bold red]⚠ ANOMALY[/bold red]  "
                            f"#{anomaly_count:<3}  "
                            f"sample=[cyan]{i:5d}[/cyan]  "
                            f"score=[yellow]{score:.3f}[/yellow]  "
                            f"→ [bold magenta]{top_sub.upper():<10}[/bold magenta] "
                            f"(conf={top_conf:.2f})"
                        )
                        last_print_idx = i

            progress.update(task, completed=i + 1,
                            description=f"[green]Detecting[/green]  anomalies: {anomaly_count}")

    console.rule("[bold green]Stream Completed[/bold green]")
    console.print()

    # -------------------------------------------------
    # 4. Summary
    # -------------------------------------------------
    console.print(Panel.fit("[bold]Detection Summary[/bold]", border_style="green"))

    summary = Table(show_header=True, header_style="bold cyan")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")

    post_warmup = total_samples - warmup - stability_window
    summary.add_row("Total samples processed", f"{total_samples:,}")
    summary.add_row("Warm-up + Stability", f"{warmup + stability_window:,}")
    summary.add_row("Effective detection window", f"{post_warmup:,}")
    summary.add_row("Anomalies reported", f"[bold red]{anomaly_count}[/bold red]")
    summary.add_row("Anomaly rate", f"{anomaly_count / max(post_warmup, 1) * 100:.2f}%")

    console.print(summary)
    console.print()

    if anomaly_log:
        console.print("[bold]Root-Cause Distribution[/bold]")
        counts = Counter([a["subsystem"] for a in anomaly_log])

        rc_table = Table(show_header=True, header_style="bold")
        rc_table.add_column("Subsystem", style="magenta")
        rc_table.add_column("Count", justify="right")
        rc_table.add_column("Percentage", justify="right")

        for sub, cnt in counts.most_common():
            pct = cnt / anomaly_count * 100
            rc_table.add_row(sub.upper(), str(cnt), f"{pct:.1f}%")

        console.print(rc_table)
        console.print()

        console.print("[bold]Last 6 Reported Anomalies[/bold]")
        last_table = Table(show_header=True, header_style="bold")
        last_table.add_column("Sample", justify="right")
        last_table.add_column("Score", justify="right")
        last_table.add_column("Root-Cause")
        last_table.add_column("Confidence", justify="right")

        for a in anomaly_log[-6:]:
            last_table.add_row(
                str(a["index"]),
                f"{a['score']:.3f}",
                f"[magenta]{a['subsystem'].upper()}[/magenta]",
                f"{a['confidence']:.2f}"
            )
        console.print(last_table)

    console.print()
    console.print("[bold green]✓ Demo completed successfully[/bold green]")
    console.print("[dim]Online incremental learning with adaptive thresholding and subsystem-level root-cause analysis.[/dim]")
    console.print()


if __name__ == "__main__":
    main()
    