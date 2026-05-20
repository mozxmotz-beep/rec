"""Temporal feature engineering utilities for PCVR.

Implements prioritized P0/P1 features in a leak-safe way from per-sample
historical event signals.
"""

from typing import Dict
import numpy as np

WINDOW_DAYS = (1, 3, 7, 14, 30, 90)


def build_temporal_features(
    time_buckets: np.ndarray,
    seq_len: np.ndarray,
    half_lifes: tuple = (3.0, 14.0),
    alpha_prior: float = 1.0,
    beta_prior: float = 10.0,
) -> np.ndarray:
    """Builds P0/P1 temporal features from bucketized historical sequence.

    Args:
        time_buckets: [B, L] history age buckets (>0 means valid token).
        seq_len: [B] true sequence lengths.
    """
    B, L = time_buckets.shape
    valid = (time_buckets > 0).astype(np.float32)
    age_days = np.clip(time_buckets.astype(np.float32), 1.0, 365.0)
    out = []

    # P0: multi-window stats + drift (3d vs 30d) + multi-half-life decays.
    for wd in WINDOW_DAYS:
        mask = valid * (age_days <= wd).astype(np.float32)
        cnt = mask.sum(axis=1, keepdims=True)
        smoothed = (cnt + alpha_prior) / (seq_len.reshape(-1, 1) + alpha_prior + beta_prior)
        out.extend([cnt, smoothed])

    cnt3 = (valid * (age_days <= 3).astype(np.float32)).sum(axis=1, keepdims=True)
    cnt30 = (valid * (age_days <= 30).astype(np.float32)).sum(axis=1, keepdims=True)
    drift = (cnt3 + 1.0) / (cnt30 + 1.0)
    out.append(drift)

    for hl in half_lifes:
        w = valid * np.exp(-np.log(2.0) * age_days / hl)
        out.append(w.sum(axis=1, keepdims=True))

    # P1: recency + entropy proxy.
    min_age = np.where(valid > 0, age_days, 1e9).min(axis=1, keepdims=True)
    min_age[min_age > 1e8] = 365.0
    out.append(1.0 / (1.0 + min_age))

    p = valid / np.clip(valid.sum(axis=1, keepdims=True), 1.0, None)
    entropy = -(p * np.log(np.clip(p, 1e-8, 1.0))).sum(axis=1, keepdims=True)
    out.append(entropy)

    return np.concatenate(out, axis=1).astype(np.float32)
