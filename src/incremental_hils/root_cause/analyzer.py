"""
Improved Root Cause Analyzer for HILS anomalies.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np


class RootCauseAnalyzer:
    def __init__(self):
        pass

    def feature_to_subsystem(self, feature_name: str) -> str:
        """Assumes naming convention: subsystem__feature"""
        if "__" in feature_name:
            return feature_name.split("__")[0]
        return "unknown"

    def aggregate_to_subsystem(
        self,
        feature_contributions: List[Tuple[str, float]],
        min_score: float = 1.5,
    ) -> List[Tuple[str, float]]:
        """
        Aggregate feature-level scores into subsystem-level scores.
        Only keeps subsystems that show meaningful contribution.
        """
        subsystem_scores = defaultdict(list)

        for feat, score in feature_contributions:
            if score < min_score:
                continue
            sub = self.feature_to_subsystem(feat)
            subsystem_scores[sub].append(score)

        result = []
        for sub, scores in subsystem_scores.items():
            # Use max score of the subsystem (strongest signal)
            result.append((sub, float(np.max(scores))))

        result.sort(key=lambda t: t[1], reverse=True)
        return result

    def explain(
        self,
        feature_contributions: List[Tuple[str, float]],
        top_k_subsystems: int = 3,
    ) -> str:
        """
        Return a human-readable root-cause explanation.
        """
        sub_scores = self.aggregate_to_subsystem(feature_contributions)

        if not sub_scores:
            return "No clear root cause identified."

        top = sub_scores[:top_k_subsystems]
        parts = [f"{sub} (score={score:.2f})" for sub, score in top]
        return "Likely source → " + ", ".join(parts)
        