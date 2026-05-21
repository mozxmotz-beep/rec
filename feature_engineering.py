"""Temporal + behavioral feature engineering utilities for PCVR."""

from typing import Dict, Mapping
import numpy as np

WINDOW_DAYS = (1, 3, 7, 14, 30, 90)


def _safe_beta_ratio(num: np.ndarray, den: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return (num + alpha) / (den + alpha + beta)


def get_temporal_feature_dim(num_domains: int, num_half_lifes: int = 2) -> int:
    """Return total appended dense dim introduced by temporal features.

    Per domain dim:
      - each window contributes 2 dims (count + smoothed): 2 * len(WINDOW_DAYS)
      - drift: 1
      - multi half-life decays: num_half_lifes
      - recency: 1
      - entropy: 1
    """
    per_domain = 2 * len(WINDOW_DAYS) + 1 + num_half_lifes + 1 + 1
    return per_domain * max(num_domains, 0)


def build_temporal_features(
    time_buckets: np.ndarray,
    seq_len: np.ndarray,
    half_lifes: tuple = (3.0, 14.0),
    alpha_prior: float = 1.0,
    beta_prior: float = 10.0,
) -> np.ndarray:
    B, L = time_buckets.shape
    valid = (time_buckets > 0).astype(np.float32)
    age_days = np.clip(time_buckets.astype(np.float32), 1.0, 365.0)
    out = []

    for wd in WINDOW_DAYS:
        mask = valid * (age_days <= wd).astype(np.float32)
        cnt = mask.sum(axis=1, keepdims=True)
        smoothed = _safe_beta_ratio(cnt, seq_len.reshape(-1, 1), alpha_prior, beta_prior)
        out.extend([cnt, smoothed])

    cnt3 = (valid * (age_days <= 3).astype(np.float32)).sum(axis=1, keepdims=True)
    cnt30 = (valid * (age_days <= 30).astype(np.float32)).sum(axis=1, keepdims=True)
    out.append((cnt3 + 1.0) / (cnt30 + 1.0))

    for hl in half_lifes:
        w = valid * np.exp(-np.log(2.0) * age_days / hl)
        out.append(w.sum(axis=1, keepdims=True))

    min_age = np.where(valid > 0, age_days, 1e9).min(axis=1, keepdims=True)
    min_age[min_age > 1e8] = 365.0
    out.append(1.0 / (1.0 + min_age))

    p = valid / np.clip(valid.sum(axis=1, keepdims=True), 1.0, None)
    entropy = -(p * np.log(np.clip(p, 1e-8, 1.0))).sum(axis=1, keepdims=True)
    out.append(entropy)

    return np.concatenate(out, axis=1).astype(np.float32)


def build_realtime_item_stats_features(
    stats_1h: Mapping[str, float],
    stats_24h: Mapping[str, float],
    alpha: float = 1.0,
    beta: float = 10.0,
) -> np.ndarray:
    imp_1h = float(stats_1h.get("exp_cnt", 0.0))
    clk_1h = float(stats_1h.get("click_cnt", 0.0))
    imp_24h = float(stats_24h.get("exp_cnt", 0.0))
    clk_24h = float(stats_24h.get("click_cnt", 0.0))

    cvr_1h = (clk_1h + alpha) / (imp_1h + alpha + beta)
    cvr_24h = (clk_24h + alpha) / (imp_24h + alpha + beta)

    trend_imp = np.log1p(imp_1h) - np.log1p(imp_24h / 24.0)
    trend_clk = np.log1p(clk_1h) - np.log1p(clk_24h / 24.0)
    trend_cvr = cvr_1h - cvr_24h
    ratio_imp = (imp_1h + 1.0) / (imp_24h / 24.0 + 1.0)

    return np.asarray([
        imp_1h, clk_1h, cvr_1h,
        imp_24h, clk_24h, cvr_24h,
        trend_imp, trend_clk, trend_cvr, ratio_imp,
    ], dtype=np.float32)


def build_user_item_history_features(
    user_author: Mapping[str, float],
    user_cate: Mapping[str, float],
    user_brand: Mapping[str, float],
    global_cate_cvr: float,
    alpha: float = 2.0,
) -> np.ndarray:
    def _smoothed_cvr(x: Mapping[str, float], prior: float) -> float:
        imp = float(x.get("exp_cnt", 0.0))
        clk = float(x.get("click_cnt", 0.0))
        return (clk + alpha * prior) / (imp + alpha)

    author_cvr = _smoothed_cvr(user_author, global_cate_cvr)
    cate_cvr = _smoothed_cvr(user_cate, global_cate_cvr)
    brand_cvr = _smoothed_cvr(user_brand, global_cate_cvr)

    author_imp = float(user_author.get("exp_cnt", 0.0))
    cate_imp = float(user_cate.get("exp_cnt", 0.0))
    brand_imp = float(user_brand.get("exp_cnt", 0.0))

    return np.asarray([
        author_imp, cate_imp, brand_imp,
        author_cvr, cate_cvr, brand_cvr,
        cate_cvr - float(global_cate_cvr),
    ], dtype=np.float32)
