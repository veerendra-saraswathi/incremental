"""
Generate synthetic multi-rate HILS-like data for development.

Subsystems simulated:
- imu          : 100 Hz
- actuator     :  50 Hz
- fms          :  20 Hz
- guidance     :  10 Hz
- telemetry    :   5 Hz
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def generate_subsystem(
    name: str,
    freq_hz: float,
    duration_s: float = 180.0,
    n_features: int = 4,
    anomaly_intervals: list | None = None,
) -> pd.DataFrame:
    n_samples = int(duration_s * freq_hz)
    dt = 1.0 / freq_hz
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    timestamps = [t0 + timedelta(seconds=i * dt) for i in range(n_samples)]

    t = np.arange(n_samples) * dt
    data = {}

    for f in range(n_features):
        signal = (
            1.2 * np.sin(2 * np.pi * (0.3 + 0.1 * f) * t)
            + 0.4 * np.sin(2 * np.pi * 1.7 * t + f)
            + 0.05 * np.random.randn(n_samples)
        )
        data[f"feat_{f}"] = signal

    df = pd.DataFrame(data)
    df.insert(0, "timestamp", timestamps)
    df["subsystem"] = name

    # Inject controlled anomalies
    if anomaly_intervals:
        for start_s, end_s, atype, feat_idx in anomaly_intervals:
            mask = (t >= start_s) & (t <= end_s)
            col = f"feat_{feat_idx}"

            if atype == "bias":
                df.loc[mask, col] += 1.8
            elif atype == "spike":
                df.loc[mask, col] += np.random.randn(mask.sum()) * 3.5
            elif atype == "drift":
                drift = np.linspace(0, 2.5, mask.sum())
                df.loc[mask, col] += drift
            elif atype == "noise":
                df.loc[mask, col] += np.random.randn(mask.sum()) * 1.8

    return df


def main():
    duration = 180.0  # 3 minutes

    anomalies_imu = [
        (45.0, 52.0, "spike", 1),
        (110.0, 125.0, "drift", 0),
    ]
    anomalies_actuator = [
        (70.0, 78.0, "bias", 2),
    ]
    anomalies_fms = [
        (95.0, 105.0, "noise", 0),
    ]

    subsystems = {
        "imu": generate_subsystem("imu", 100.0, duration, anomaly_intervals=anomalies_imu),
        "actuator": generate_subsystem("actuator", 50.0, duration, anomaly_intervals=anomalies_actuator),
        "fms": generate_subsystem("fms", 20.0, duration, anomaly_intervals=anomalies_fms),
        "guidance": generate_subsystem("guidance", 10.0, duration),
        "telemetry": generate_subsystem("telemetry", 5.0, duration),
    }

    for name, df in subsystems.items():
        out_path = SAMPLE_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path}  → {len(df)} rows")

    meta = {
        "duration_s": duration,
        "subsystems": {
            "imu": {"freq_hz": 100, "anomalies": anomalies_imu},
            "actuator": {"freq_hz": 50, "anomalies": anomalies_actuator},
            "fms": {"freq_hz": 20, "anomalies": anomalies_fms},
            "guidance": {"freq_hz": 10, "anomalies": []},
            "telemetry": {"freq_hz": 5, "anomalies": []},
        },
    }

    with open(SAMPLE_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSample data generated successfully at: {SAMPLE_DIR}")


if __name__ == "__main__":
    main()