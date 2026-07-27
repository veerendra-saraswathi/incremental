"""
Multi-rate synchronization / alignment for HILS subsystems.
"""

from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np


def align_to_common_rate(
    data: Dict[str, pd.DataFrame],
    time_col: str = "timestamp",
    target_freq: str = "10ms",
) -> pd.DataFrame:
    """
    Align multiple subsystem dataframes (different sampling rates)
    onto a common time grid.
    """
    aligned_frames = []

    for name, df in data.items():
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col).sort_index()

        numeric = df.select_dtypes(include=[np.number])
        resampled = numeric.resample(target_freq).mean().ffill().bfill()
        resampled.columns = [f"{name}__{c}" for c in resampled.columns]
        aligned_frames.append(resampled)

    if not aligned_frames:
        return pd.DataFrame()

    result = pd.concat(aligned_frames, axis=1)
    return result.dropna(how="all")