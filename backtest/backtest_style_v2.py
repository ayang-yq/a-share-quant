"""
风格轮动 v2 — 修正版回测
==========================
修正点:
  1. 窗口: 60/120日 (风格是低频信号)
  2. 信号: 离散分位 (只在极端区域切换, 非极端=均衡50/50)
  3. 交易成本: 双边万三 + 滑点万二 = 单边0.05%
  4. 调仓频率: 降低, split变化<10pp不调仓
  5. 极端阈值: 分位5%/95% (少数时间才触发切换)

风格指标:
  - 小盘/大盘 (801813/801811): 最直接对应300 vs 2000
  - 高PB/低PB (801821/801823): 与小/大相关性0.64, 有信息增量
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, akshare as ak

print("=" * 70)
print("风格轮动 v2 — 修正版回测")
print("=" * 70)

# ══════════════════════════════════════════════
# 1. 数据获取
# ══════════════════════════════════════════════
print("  数据获取...", flush=True)

idx_map = {"沪深300": "sh000300", "国证2000": "sz399303"}
all_price = {}
for name, code in idx_map.items():
    df = ak.stock_zh_index_daily(symbol=code)[["date","close"]].rename(columns={"close": name})
    df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date").reset_index(drop=True)
    all_price[name] = df

sw_codes = {"大盘": "801811", "小盘": "801813", "高PB": "801821", "低PB": "801823"}
sw_data = {}
for name, code in sw_codes.items():
    df = ak.index_hist_sw(symbol=code, period="day")[["日期","收盘"]].rename(columns={"日期":"date","收盘":name})
    df["date"] = pd.to_datetime(df["date"])
    sw_data[name] = df

df_msh = ak.macro_china_market_margin_sh()[["日期","融资买入额"]].rename(columns={"日期":"date","融资买入额":"rzye"})
df_msz = ak.macro_china_market_margin_sz()[["日期","融资买入额"]].rename(columns={"日期":"date","融资买入额":"rzye"})
df_msh["date"] = pd.to_datetime(df_msh["date"])
df_msz["date"] = pd.to_datetime(df_msz["date"])
df_margin = df_msh.merge(df_msz, on="date", how="outer", suffixes=("_sh","_sz"))
df_margin = df_margin.sort_values("date").reset_index(drop=True)
df_margin["rzye_total"] = pd.to_numeric(df_margin["rzye_sh"], errors="coerce") + pd.to_numeric(df_margin["rzye_sz"], errors="coerce")

# 合并
base = all_price["沪深300"].copy()
base = base.merge(all_price["国证2000"][["date","国证2000"]], on="date", how="left")
for name in sw_data:
    base = base.merge(sw_data[name][["date", name]], on="date", how="left")
base = base.merge(df_margin[["date","rzye_total"]], on="date", how="left")
for c in list(sw_data.keys()) + ["国证2000","rzye_total"]:
    base[c] = base[c].ffill()
base = base[(base.date >= "2015-01-01") & (base.date <= "2026-04-30")].reset_index(drop=True)
df = base.copy()
print(f"  合并完成: {len(df)}天")

# ══════════════════════════════════════════════
# 2. 指标计算
# ══════════════════════════════════════════════
print("  计算指标...", flush=True)
W_TIMING = 20  # 择时仍用20日(已验证最优)
COST = 0.0005  # 单边交易成本0.05%

# v3.0择时
df["bb_z"] = (df["沪深300"] - df["沪深300"].rolling(W_TIMING).mean()) / df["沪深300"].rolling(W_TIMING).std()
df["rz_z"] = (df["rzye_total"] - df["rzye_total"].rolling(W_TIMING).mean()) / df["rzye_total"].rolling(W_TIMING).std()

def z2p(z, lo=0.20, hi=0.80):
    z = float(np.clip(z, -3, 3))
    return 0.50 + z / 3 * (hi - 0.50)

df["v30_pos"] = df["bb_z"].apply(lambda v: z2p(v)*0.70 if not pd.isna(v) else 0.35) + \
                df["rz_z"].apply(lambda v: z2p(v)*0.30 if not pd.isna(v) else 0.15)

# 风格比值
df["ratio_xd"] = df["小盘"] / df["大盘"]
df["ratio_pb"] = df["高PB"] / df["低PB"]

# 多窗口分位数
for win in [60, 120, 250]:
    for ratio_name in ["ratio_xd", "ratio_pb"]:
        col = f"{ratio_name}_p{win}"
        df[col] = df[ratio_name].rolling(win).rank(pct=True)

df = df.iloc[260:].reset_index(drop=True)
print(f"  预热后: {len(df)}天")

# ══════════════════════════════════════════════
# 3. 风格信号生成(离散 + 极端触发)
# ══════════════════════════════════════════════
def make_discrete_signal(pct_col, lo_thresh=0.10, hi_thresh=0.90, 
                         extreme_split=0.80, neutral_split=0.50, direction=1):
    """
    离散风格信号:
    - pct > hi_thresh → 极端区域 → 偏移split
    - pct < lo_thresh → 极端区域 → 反向偏移split
    - 中间 → 均衡 neutral_split
    
    direction=1: pct高(小盘强) → split高(加仓300) [反转逻辑]
    direction=-1: pct高(小盘强) → split低(加仓2000) [趋势逻辑]
    """
    splits = np.full(len(df), neutral_split)
    pcts = df[pct_col].values
    
    if direction == 1:  # 反转: 小盘极端强 → 加仓300
        splits[pcts > hi_thresh] = extreme_split
        splits[pcts < lo_thresh] = 1.0 - extreme_split
    else:  # 趋势: 小盘强 → 加仓2000
        splits[pcts > hi_thresh] = 1.0 - extreme_split
        splits[pcts < lo_thresh] = extreme_split
    
    return splits

# ══════════════════════════════════════════════
# 4. 回测引擎(含交易成本)
# ══════════════════════════════════════════════
def bt_with_cost(price_300, price_2000, pos_arr, split_arr, label="", 
                 min_split_change=0.10):
    """
    回测含交易成本:
    - 总仓位变化触发成本
    - split变化超过min_split_change才调仓
    """
    ret_300 = price_300.pct_change().fillna(0).values
    ret_2000 = price_2000.pct_change().fillna(0).values
    pos = np.nan_to_num(pos_arr, nan=0.50)
    split = np.nan_to_num(split_arr, nan=0.50)
    
    # 实际持仓
    w_300 = pos * split
    w_2000 = pos * (1 - split)
    w_cash = 1 - pos
    
    # 交易成本: 当天持仓变化量 × 成本
    cost = np.zeros(len(df))
    for i in range(1, len(df)):
        # split变化导致的调仓
        old_w300 = w_300[i-1]
        old_w2000 = w_2000[i-1]
        old_cash = w_cash[i-1]
        
        new_w300 = w_300[i]
        new_w2000 = w_2000[i]
        new_cash = w_cash[i]
        
        # split变化阈值过滤
        split_delta = abs(split[i] - split[i-1])
        if split_delta < min_split_change:
            # 不调split, 用昨天的split
            new_w300 = pos[i] * split[i-1]
            new_w2000 = pos[i] * (1 - split[i-1])
            new_cash = 1 - pos[i]
            w_300[i] = new_w300
            w_2000[i] = new_w2000
            w_cash[i] = new_cash
        
        turnover = abs(new_w300 - old_w300) + abs(new_w2000 - old_w2000)
        cost[i] = turnover * COST
    
    # 收益
    strat_ret = w_300 * ret_300 + w_2000 * ret_2000 + w_cash * 0 - cost
    
    # 基准: v3.0纯300
    bench_ret = pos * ret_300 + (1 - pos) * 0
    
    # 满仓基准
    rf_d = 1.02 ** (1/252) - 1
    full300_ret = ret_300
    full5050_ret = 0.5 * ret_300 + 0.5 * ret_2000
    
    def stats(ret, name):
        nav = np.cumprod(1 + ret)
        yrs = len(nav) / 252
        r = pd.Series(ret)
        dd = (pd.Series(nav) / pd.Series(nav).cummax() - 1).min()
        ann = (nav[-1] / nav[0]) ** (1/yrs) - 1 if yrs > 0 else 0
        sharpe = np.sqrt(252) * (r - rf_d).mean() / r.std() if r.std() > 0 else 0
        # 年化换手
        turnover_proxy = np.mean(np.abs(np.diff(ret))) * 252
        return {"ann": ann, "sharpe": sharpe, "max_dd": dd}
    
    s = stats(strat_ret, "s")
    b = stats(bench_ret, "b")
    f300 = stats(full300_ret, "300")
    f50 = stats(full5050_ret, "5050")
    
    # 调仓次数统计
    if min_split_change > 0:
        adj_count = np.sum(np.abs(np.diff(split)) >= min_split_change)
    else:
        adj_count = len(df)
    
    # split分布
    split_mean = np.mean(split)
    split_std = np.std(split)
    
    return {
        "label": label,
        "ann": s["ann"], "sharpe": s["sharpe"], "max_dd": s["max_dd"],
        "bench_ann": b["ann"], "bench_sharpe": b["sharpe"], "bench_dd": b["max_dd"],
        "full300_ann": f300["ann"], "full300_sharpe": f300["sharpe"],
        "full5050_ann": f50["ann"], "full5050_sharpe": f50["sharpe"],
        "ann_lift": s["ann"] - b["ann"], "sharpe_lift": s["sharpe"] - b["sharpe"],
        "adj_count": adj_count, "split_mean": split_mean, "split_std": split_std,
    }

# ══════════════════════════════════════════════
# 5. 参数扫描
# ══════════════════════════════════════════════
print("\n" + "=" * 70)
print("参数扫描")
print("=" * 70)

results = []

# 基准
r0 = bt_with_cost(df["沪深300"], df["国证2000"], df["v30_pos"].values,
                   np.full(len(df), 0.50), "v3.0纯300(无成本)", min_split_change=0)
results.append(r0)

r0c = bt_with_cost(df["沪深300"], df["国证2000"], df["v30_pos"].values,
                    np.full(len(df), 0.50), "v3.0纯300(含成本)", min_split_change=0.10)
results.append(r0c)

# 参数组合
configs = []
for win in [60, 120, 250]:
    for lo_t, hi_t in [(0.05, 0.95), (0.10, 0.90), (0.15, 0.85), (0.20, 0.80)]:
        for ext_split in [0.70, 0.80, 0.90]:
            for direction in [1, -1]:  # 1=反转, -1=趋势
                for ratio_name in ["ratio_xd", "ratio_pb"]:
                    pct_col = f"{ratio_name}_p{win}"
                    label_base = "小/大" if "xd" in ratio_name else "PB"
                    dir_str = "反转" if direction == 1 else "趋势"
                    configs.append({
                        "pct_col": pct_col,
                        "lo": lo_t, "hi": hi_t,
                        "ext": ext_split,
                        "dir": direction,
                        "label": f"{label_base} W{win} [{lo_t:.0%},{hi_t:.0%}] ext{ext_split:.0%} {dir_str}"
                    })

print(f"  共{len(configs)}组参数...", flush=True)

for cfg in configs:
    split = make_discrete_signal(cfg["pct_col"], cfg["lo"], cfg["hi"], cfg["ext"], 0.50, cfg["dir"])
    r = bt_with_cost(df["沪深300"], df["国证2000"], df["v30_pos"].values, split,
                     cfg["label"], min_split_change=0.10)
    results.append(r)

print(f"\n{'策略':<45} {'年化':>7} {'夏普':>7} {'回撤':>7} {'夏普Δ':>7} {'年化Δ':>7} {'调仓次':>6} {'splitμ':>6}")
print("-" * 100)

# 按夏普提升排序
ranked = sorted(results, key=lambda x: x["sharpe_lift"], reverse=True)
for r in ranked[:30]:
    print(f"{r['label']:<45} {r['ann']*100:>6.1f}% {r['sharpe']:>7.2f} {r['max_dd']*100:>6.1f}% "
          f"{r['sharpe_lift']:>+7.2f} {r['ann_lift']*100:>+6.1f}pp {r['adj_count']:>6} {r['split_mean']:>6.3f}")

# 满仓基准
print(f"\n满仓沪深300:  年化 {results[0]['full300_ann']*100:.1f}%  夏普 {results[0]['full300_sharpe']:.2f}")
print(f"满仓50/50:    年化 {results[0]['full5050_ann']*100:.1f}%  夏普 {results[0]['full5050_sharpe']:.2f}")

# ══════════════════════════════════════════════
# 6. Top3分年度表现
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("Top3方案 分年度对比")
print("=" * 70)

top3 = ranked[2:5]  # 跳过两个基准
rf_d = 1.02 ** (1/252) - 1

df["year"] = df["date"].dt.year
ret_300 = df["沪深300"].pct_change().fillna(0)
ret_2000 = df["国证2000"].pct_change().fillna(0)
pos = df["v30_pos"].values

# 预先缓存所有split结果
split_cache = {}
for cfg in configs:
    split_cache[cfg["label"]] = make_discrete_signal(cfg["pct_col"], cfg["lo"], cfg["hi"], cfg["ext"], 0.50, cfg["dir"])

for tr in top3:
    print(f"\n--- {tr['label']} ---")
    split = split_cache[tr["label"]]
    
    # 成本过滤
    w_300 = np.copy(pos * split)
    for i in range(1, len(df)):
        if abs(split[i] - split[i-1]) < 0.10:
            w_300[i] = pos[i] * split[i-1]
        else:
            w_300[i] = pos[i] * split[i]
    w_2000 = pos - w_300
    
    strat_ret = w_300 * ret_300 + w_2000 * ret_2000
    bench_ret = pos * ret_300
    
    print(f"  {'年份':>6} {'风格策略':>10} {'纯300基准':>10} {'Δ':>8}")
    for y in sorted(df["year"].unique()):
        m = df["year"] == y
        if m.sum() < 200: continue
        n = m.sum()
        sa = (np.prod(1+strat_ret[m]))**(252/n) - 1
        ba = (np.prod(1+bench_ret[m]))**(252/n) - 1
        print(f"  {y:>6} {sa*100:>9.1f}% {ba*100:>9.1f}% {(sa-ba)*100:>+7.1f}pp")

print("\n完成.")
