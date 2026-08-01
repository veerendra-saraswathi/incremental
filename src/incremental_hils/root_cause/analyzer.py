"""
Improved Root Cause Analyzer for HILS Anomaly Detection.

Improvements:
- Better aggregation (considers both strength and number of anomalous features)
- Calibrated confidence scores (more interpretable range)
- Clearer human-readable explanations
- More robust subsystem mapping
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np


class RootCauseAnalyzer:
    def __init__(self, min_feature_score: float = 1.5):
        """
        Parameters
        ----------
        min_feature_score : float
            Minimum z-score for a feature to be considered contributory.
        """
        self.min_feature_score = min_feature_score

    def feature_to_subsystem(self, feature_name: str) -> str:
        """
        Map feature name to subsystem.
        Supports both 'subsystem__feature' and 'subsystem_feature' conventions.
        """
        name = feature_name.lower()

        if "__" in name:
            return name.split("__")[0]

        # Fallback heuristics
        for sub in ["imu", "actuator", "fms", "guidance", "telemetry", "seeker", "radar"]:
            if name.startswith(sub) or f"_{sub}_" in name or name.endswith(f"_{sub}"):
                return sub

        return "unknown"

    def aggregate_to_subsystem(
        self,
        feature_contributions: List[Tuple[str, float]],
        min_score: float = None,
    ) -> List[Tuple[str, float]]:
        """
        Aggregate feature-level contributions into subsystem-level scores.

        Scoring logic:
        - Takes the top features above threshold
        - Combines max strength + how many features from the same subsystem fired
        - Returns calibrated confidence in roughly [1.0 – 5.0] range
        """
        if min_score is None:
            min_score = self.min_feature_score

        subsystem_features = defaultdict(list)

        for feat, score in feature_contributions:
            if score < min_score:
                continue
            sub = self.feature_to_subsystem(feat)
            subsystem_features[sub].append(score)

        results = []

        for sub, scores in subsystem_features.items():
            if not scores:
                continue

            max_score = float(np.max(scores))
            n_features = len(scores)
            mean_score = float(np.mean(scores))

            # Combined score:
            # - Strong single feature is important
            # - Multiple features from same subsystem increase confidence
            raw = (0.65 * max_score) + (0.25 * mean_score) + (0.10 * min(n_features, 4))

            # Calibrate to a nicer range (approx 1.0 – 5.0)
            confidence = min(max(raw, 1.0), 5.0)

            results.append((sub, round(confidence, 2)))

        # Sort by confidence descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def explain(
        self,
        feature_contributions: List[Tuple[str, float]],
        top_k: int = 3,
    ) -> str:
        """
        Generate a human-readable root-cause explanation.
        """
        sub_scores = self.aggregate_to_subsystem(feature_contributions)

        if not sub_scores:
            return "No clear root-cause subsystem identified."

        top = sub_scores[:top_k]

        if len(top) == 1:
            sub, conf = top[0]
            return f"Most likely root cause → {sub.upper()} (confidence: {conf:.2f})"

        parts = [f"{sub.upper()} ({conf:.2f})" for sub, conf in top]
        return "Likely sources → " + " > ".join(parts)

    def detailed_breakdown(
        self,
        feature_contributions: List[Tuple[str, float]],
    ) -> Dict[str, Dict]:
        """
        Return a detailed dictionary useful for logging or GUI.
        """
        sub_scores = self.aggregate_to_subsystem(feature_contributions)
        breakdown = {}

        for sub, conf in sub_scores:
            feats = [
                (f, s) for f, s in feature_contributions
                if self.feature_to_subsystem(f) == sub and s >= self.min_feature_score
            ]
            feats.sort(key=lambda x: x[1], reverse=True)

            breakdown[sub] = {
                "confidence": conf,
                "n_features": len(feats),
                "top_features": feats[:3],
            }

        return breakdown
        