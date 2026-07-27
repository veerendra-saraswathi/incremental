"""
Improved Hybrid Incremental Anomaly Detector for HILS.
Combines statistical residuals + Half-Space Trees.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
from river import anomaly, preprocessing, compose, stats


class IncrementalAnomalyDetector:
    def __init__(
        self,
        n_trees: int = 25,
        height: int = 8,
        window_size: int = 250,
    ):
        # Model 1: Half-Space Trees
        self.hst = compose.Pipeline(
            preprocessing.StandardScaler(),
            anomaly.HalfSpaceTrees(
                n_trees=n_trees,
                height=height,
                window_size=window_size,
                seed=42,
            ),
        )

        # Model 2: Per-feature running stats for residual scoring
        self.means: Dict[str, stats.Mean] = {}
        self.vars: Dict[str, stats.Var] = {}

        self.feature_names: List[str] = []
        self.scores: List[float] = []
        self._n_seen = 0

    def _update_stats(self, x: Dict[str, float]):
        for f, v in x.items():
            if f not in self.means:
                self.means[f] = stats.Mean()
                self.vars[f] = stats.Var()
            self.means[f].update(v)
            self.vars[f].update(v)

    def _statistical_score(self, x: Dict[str, float]) -> float:
        """
        Average absolute z-score across features.
        Good at detecting bias and drift.
        """
        z_scores = []
        for f, v in x.items():
            if f not in self.means:
                continue
            mean = self.means[f].get()
            var = self.vars[f].get()
            std = np.sqrt(var) if var > 1e-8 else 1e-8
            z = abs((v - mean) / std)
            z_scores.append(z)

        if not z_scores:
            return 0.0
        return float(np.mean(z_scores))

    def learn_one(self, x: Dict[str, float]) -> float:
        if not self.feature_names:
            self.feature_names = sorted(x.keys())

        # Update both models
        hst_score = self.hst.score_one(x)
        self.hst.learn_one(x)

        self._update_stats(x)
        stat_score = self._statistical_score(x)

        # Combined score (simple weighted sum)
        # Statistical score is scaled down because raw z-scores are larger
        combined = 0.6 * hst_score + 0.4 * min(stat_score / 5.0, 1.0)

        self.scores.append(combined)
        self._n_seen += 1
        return float(combined)

    def basic_root_cause(self, x: Dict[str, float], top_k: int = 5) -> List[Tuple[str, float]]:
        scores = []
        for f, v in x.items():
            if f not in self.means:
                continue
            mean = self.means[f].get()
            var = self.vars[f].get()
            std = np.sqrt(var) if var > 1e-8 else 1e-8
            z = abs((v - mean) / std)
            scores.append((f, float(z)))

        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:top_k]
        