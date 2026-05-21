# 模型 AUC 优化方案（三个方向）

本文基于当前项目（`train.py` / `trainer.py` / `model.py` / `feature_engineering.py` / `dataset.py` / `point_time_store.py` / `infer.py`）给出可直接落地的工程化方案，用于提升 CTR/CVR 相关排序任务的 AUC。

---

## 0. 先对当前代码做能力映射

- 训练主流程：`train.py`、`trainer.py`
- 模型结构：`model.py`
- 特征构建：`feature_engineering.py`
- 数据读取与样本组织：`dataset.py`
- 时点特征缓存：`point_time_store.py`
- 推理流程：`infer.py`

建议新增三个模块并注册到现有流程：

- `features/realtime_item_stats.py`（物品实时统计 + 冷热变化）
- `features/user_item_history.py`（User×Item 交互历史）
- `debias/position_bias.py`（位置偏差建模 + 去偏）

---

## 1) 物品实时统计特征（1h/24h 曝光/点击/CVR）+ 冷热度变化

### 1.1 特征定义（严格防泄漏）

以事件时间 `t` 为基准，针对 item `i` 构造：

- `imp_1h(i,t)`：`[t-1h, t)` 曝光数
- `clk_1h(i,t)`：`[t-1h, t)` 点击数
- `cvr_1h(i,t)`：`(clk_1h + α) / (imp_1h + α+β)`（Beta 平滑）
- `imp_24h(i,t)`、`clk_24h(i,t)`、`cvr_24h(i,t)` 同理

冷热变化：

- `trend_imp = log1p(imp_1h) - log1p(imp_24h/24)`
- `trend_clk = log1p(clk_1h) - log1p(clk_24h/24)`
- `trend_cvr = cvr_1h - cvr_24h`
- `ratio_imp = (imp_1h+1)/(imp_24h/24+1)`

> 原则：所有统计窗口都必须是左闭右开并且 `t` 时刻只使用过去数据。

### 1.2 数据结构与计算方式

离线训练：

- 对日志按时间排序。
- 使用 `item_id` 维度维护两个滑动窗口累积器（1h、24h）。
- 建议在 `point_time_store.py` 中扩展 `ItemWindowCounter`：
  - `push(event)`
  - `evict(before_ts)`
  - `query(item_id, now_ts)`

在线推理：

- 用 Redis / 内存 KV 存 `item:{id}` 的滚动计数。
- 每条行为实时更新；推理时读取最近窗口统计。
- 超时 fallback：若缺失则回退到类目/作者级统计。

### 1.3 工程接入步骤

1. `feature_engineering.py` 新增实时统计特征构造函数。
2. `dataset.py` 在样本生成时带上事件时间 `event_ts`。
3. `train.py` 增加开关：`--use_item_realtime_stats`。
4. `infer.py` 加载在线 KV 并补齐默认值。

### 1.4 AUC 提升机理

- 1h 捕捉短时热点，24h 提供稳定基线。
- 趋势特征帮助模型识别“上升中内容”，改善头部竞争样本排序。
- 尤其能提升新内容的 early-stage 识别能力。

---

## 2) User×Item 历史交互特征（同作者/类目/品牌 CVR）

### 2.1 特征定义

设当前候选 item 的属性为 `author=a, cate=c, brand=b`，用户 `u` 的历史交互统计：

- `u_author_imp(u,a,t)` / `u_author_clk(u,a,t)` / `u_author_cvr`
- `u_cate_imp(u,c,t)` / `u_cate_clk(u,c,t)` / `u_cate_cvr`
- `u_brand_imp(u,b,t)` / `u_brand_clk(u,b,t)` / `u_brand_cvr`

增强特征：

- 时间衰减点击率：`decay_clk / decay_imp`（例如 `w=exp(-Δt/τ)`）
- 最近 N 次是否点过同作者/类目/品牌（布尔）
- 最后一次交互距今时长（recency）
- 交叉差值：`u_cate_cvr - global_cate_cvr`

### 2.2 统计层级与回退链路

稀疏问题用分层回退：

1. `user×author`
2. `user×cate`
3. `user×brand`
4. `user` 全局偏好
5. 全局先验

并使用 Bayesian smoothing：

`smoothed_cvr = (clk + α*prior) / (imp + α)`

### 2.3 工程实现

- 在 `feature_engineering.py` 添加 `build_user_item_history_features()`。
- 在 `point_time_store.py` 添加多 key 计数器：key 为 `(user_id, field_value)`。
- 在 `dataset.py` 读取 item 侧元信息并 join 用户历史统计。
- 在 `model.py` 对这些 dense feature 做标准化后输入 MLP。

### 2.4 训练注意事项

- 按时间切分 train/valid/test（不能随机切分）。
- 去重逻辑：同一曝光会话避免重复计数。
- 大 key 控制：按最小曝光阈值保留，长尾走回退。

### 2.5 AUC 提升机理

- 让模型显式感知“用户对该候选语义簇的兴趣强度”，降低仅靠 embedding 泛化的误差。
- 对同质候选排序非常有效，尤其提升中间段样本可分性。

---

## 3) 位置偏差建模 + 推理去偏

### 3.1 问题定义

点击标签受展示位影响：高位天然点击率更高，导致训练集标签偏置，模型学习到“位置”而非“内容质量”。

### 3.2 方案 A：IPS/SNIPS 加权训练（先落地）

1. 训练 propensity 模型 `p(pos | context)`，可先简化成 `p(click|pos)` 或 `p(exposure|pos)`。
2. 主模型训练时样本加权：
   - `w = 1 / max(p, ε)`，并做 clip（如 `[1, 20]`）。
3. 用 SNIPS 归一化减少方差。

在 `trainer.py`：

- BCE loss 改成 `weighted_bce = w * BCE(logit, y)`。
- 增加参数：`--debias_mode ips --ips_clip_min --ips_clip_max`。

### 3.3 方案 B：双塔/多任务解耦（进阶）

- 主任务预测 relevance（期望无位置特征）。
- 辅任务预测 click（含位置）。
- 共享底座 + 两个 head，主排序仅用 relevance head。

在 `model.py`：

- `shared_encoder`
- `relevance_head`
- `click_head`

loss: `L = L_rel + λ * L_click`

### 3.4 推理去偏

- 线上排序分数用 `relevance_head` 输出，不拼接 `position` 特征。
- 若业务仍需位置感知（如瀑布流），单独在 re-rank 层做规则约束，不反哺打分模型。

### 3.5 离线评估

除常规 AUC，再加：

- Position-stratified AUC（每个位置分桶）
- Counterfactual 近似指标（IPS AUC）
- Calibration（分位置 ECE）

---

## 4. 推荐实施顺序（8 周）

- Week 1-2：实时 item 统计 + 冷热趋势
- Week 3-4：User×Item 历史特征
- Week 5：位置 propensity 估计 + IPS 训练
- Week 6：推理去偏联调
- Week 7：A/B 灰度
- Week 8：稳定性优化（延迟、特征缺失、监控）

---

## 5. 实验设计（保证“严谨”）

### 5.1 统一基线

- 固定同一训练窗口、采样策略、模型结构。
- 每次只增加一类特征或一个去偏策略（单变量控制）。

### 5.2 显著性检验

- 对用户维度做 bootstrap，报告 AUC 的 95% CI。
- 关键实验至少跑 3 次不同随机种子。

### 5.3 消融实验

- 仅 1h vs 仅24h vs 1h+24h+trend
- 仅 author vs +cate vs +brand
- 无去偏 vs IPS vs 多任务去偏

---

## 6. 线上工程化细节（高可用）

- 特征服务超时预算：P99 < 20ms，超时立即 fallback。
- 缺失值策略：统一 default + mask 特征。
- 数据一致性：训练/推理共用同一特征定义（建议特征配置化）。
- 监控：
  - 特征覆盖率
  - 实时分布漂移（PSI）
  - 分位置 CTR/AUC

---

## 7. 预期收益（经验区间）

- 实时 item 统计 + trend：AUC +0.003 ~ +0.010
- User×Item 历史：AUC +0.005 ~ +0.015
- 去偏：离线 AUC 可能小涨或持平，但线上长期 CTR/CVR 更稳，泛化更好

组合落地后，常见总增益在 +0.01 ~ +0.03（视数据规模与基线成熟度）。

