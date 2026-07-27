"""
Save anomaly detection results and generate simple reports.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import json
import pandas as pd


class ResultReporter:
    def __init__(self, output_dir: str | Path = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_anomalies(
        self,
        anomalies: List[Dict[str, Any]],
        prefix: str = "anomalies",
    ) -> Path:
        """
        Save list of detected anomalies to CSV + JSON.
        Each anomaly dict should contain at least:
        - index
        - score
        - subsystem (root-cause)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.output_dir / f"{prefix}_{timestamp}"

        # CSV
        df = pd.DataFrame(anomalies)
        csv_path = base.with_suffix(".csv")
        df.to_csv(csv_path, index=False)

        # JSON
        json_path = base.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(anomalies, f, indent=2)

        return csv_path

    def save_summary(
        self,
        summary: Dict[str, Any],
        prefix: str = "summary",
    ) -> Path:
        """
        Save a high-level summary report as JSON.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"{prefix}_{timestamp}.json"

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        return path

    def print_summary(self, summary: Dict[str, Any]):
        print("\n========== DETECTION SUMMARY ==========")
        for k, v in summary.items():
            print(f"  {k:25s}: {v}")
        print("=======================================\n")
        