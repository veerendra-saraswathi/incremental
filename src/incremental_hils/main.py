"""
Main entry point for Incremental HILS Anomaly Detection.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from loguru import logger
import yaml

from incremental_hils.ingestion.loader import load_multi_rate_data
from incremental_hils.synchronization.aligner import align_to_common_rate
from incremental_hils.anomaly_detection.detector import IncrementalAnomalyDetector
from incremental_hils.root_cause.analyzer import RootCauseAnalyzer


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Incremental HILS Anomaly Detection")
    parser.add_argument("--config", type=str, default="configs/development.yaml")
    parser.add_argument("--mode", type=str, choices=["train", "stream", "evaluate"], default="stream")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    logger.info(f"Running in mode: {args.mode}")

    # Placeholder flow – will be expanded
    logger.info("Pipeline started successfully (skeleton)")


if __name__ == "__main__":
    main()