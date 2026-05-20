"""Point-in-time feature store with online incremental updates."""

from dataclasses import dataclass
from typing import Dict, Tuple
import math


@dataclass
class EntityStats:
    last_ts: int = 0
    exp_cnt: float = 0.0
    click_cnt: float = 0.0
    conv_cnt: float = 0.0


class PointTimeFeatureStore:
    """In-memory point-in-time store supporting online updates.

    Keyed by (namespace, entity_id), e.g. ("user", 123) or ("user_cate", "123#88").
    """

    def __init__(self, half_life_days: float = 14.0) -> None:
        self.state: Dict[Tuple[str, str], EntityStats] = {}
        self.half_life_days = half_life_days

    def _decay(self, st: EntityStats, now_ts: int) -> None:
        if st.last_ts <= 0 or now_ts <= st.last_ts:
            st.last_ts = max(st.last_ts, now_ts)
            return
        dt_days = (now_ts - st.last_ts) / 86400.0
        factor = math.exp(-math.log(2.0) * dt_days / self.half_life_days)
        st.exp_cnt *= factor
        st.click_cnt *= factor
        st.conv_cnt *= factor
        st.last_ts = now_ts

    def update(self, namespace: str, entity_id: str, ts: int, clicked: int, converted: int) -> None:
        k = (namespace, entity_id)
        st = self.state.get(k, EntityStats())
        self._decay(st, ts)
        st.exp_cnt += 1.0
        st.click_cnt += float(clicked)
        st.conv_cnt += float(converted)
        self.state[k] = st

    def snapshot(self, namespace: str, entity_id: str, ts: int) -> Dict[str, float]:
        k = (namespace, entity_id)
        st = self.state.get(k, EntityStats())
        self._decay(st, ts)
        ctr = (st.click_cnt + 1.0) / (st.exp_cnt + 2.0)
        cvr = (st.conv_cnt + 1.0) / (st.click_cnt + 2.0)
        return {
            "exp_cnt": st.exp_cnt,
            "click_cnt": st.click_cnt,
            "conv_cnt": st.conv_cnt,
            "ctr_smooth": ctr,
            "cvr_smooth": cvr,
        }
