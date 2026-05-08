"""
均线动态止损 vs 固定止损 vs 买入持有
========================================
急涨股(20日涨幅>X%) → 跌破5/10日均线止损
缓涨股(20日涨幅<X%) → 跌破60日均线止损
对比固定止损和买入持有
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, akshare as ak

print("=" * 65)
print("均线动态止损回测")
print("=" * 65)

# ── 数据 ──
print("  沪深300日K...", end=" ", flush=True)
df = ak.stock_zh_index_daily(symbol="sh000300")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df = df[(df.date >= "2015-01-01") & (df.date <= "2026-04-30")].reset_index(drop=True)
N = len(df)
print(f"OK ({N}天)")

# 均线
for w in [5, 10, 20, 60]:
    df[f"ma{w}"] = df["close"].rolling(w).mean()

# 20日涨幅(用于判断急涨/缓涨)
df["ret_20d"] = df["close"].pct_change(20)

# ── 回测引擎 ──
def backtest_strategy(entry_interval=20, hold_max=250, stop_fn=None, label=""):
    """
    每entry_interval天建仓一笔，按stop_fn判断退出。
    stop_fn(row, hold_days) -> True表示退出
    """
    trades = []
    i = 65  # 跳过60MA预热
    while i < N - hold_max:
        entry_price = df.loc[i, "close"]
        entry_date = df.loc[i, "date"]
        entry_ret20 = df.loc[i, "ret_20d"]
        exited = False

        for h in range(1, hold_max + 1):
            j = i + h
            if j >= N:
                break
            cur = df.loc[j]
            cur_ret = (cur["close"] - entry_price) / entry_price

            # 检查退出信号
            reason = stop_fn(cur, h, entry_ret20) if stop_fn else None
            if reason:
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": cur["date"],
                    "entry_price": entry_price,
                    "exit_price": cur["close"],
                    "return": cur_ret,
                    "hold_days": h,
                    "exit_reason": reason,
                    "entry_ret20": entry_ret20,
                })
                exited = True
                break

        if not exited:
            j = min(i + hold_max, N - 1)
            trades.append({
                "entry_date": entry_date,
                "exit_date": df.loc[j, "date"],
                "entry_price": entry_price,
                "exit_price": df.loc[j, "close"],
                "return": (df.loc[j, "close"] - entry_price) / entry_price,
                "hold_days": hold_max,
                "exit_reason": "timeout",
                "entry_ret20": entry_ret20,
            })
        i += entry_interval

    return pd.DataFrame(trades)


def calc_stats(tdf, label):
    if len(tdf) == 0:
        return None
    n = len(tdf)
    wr = (tdf["return"] > 0).mean()
    avg_r = tdf["return"].mean()
    med_r = tdf["return"].median()
    max_loss = tdf["return"].min()
    avg_h = tdf["hold_days"].mean()

    # 止损后反弹分析
    stop_trades = tdf[~tdf["exit_reason"].isin(["timeout"])]
    regret = 0
    total_stop = len(stop_trades)
    if total_stop > 0:
        for _, row in stop_trades.iterrows():
            exit_idx = df.index[df["date"] == row["exit_date"]]
            if len(exit_idx) > 0:
                fi = min(exit_idx[0] + 60, N - 1)
                fp = df.loc[fi, "close"]
                if fp >= row["entry_price"]:
                    regret += 1
        regret_rate = regret / total_stop
    else:
        regret_rate = 0

    # 分急涨/缓涨统计
    fast = tdf[tdf["entry_ret20"] > 0.10] if "entry_ret20" in tdf.columns else pd.DataFrame()
    slow = tdf[tdf["entry_ret20"] <= 0.10] if "entry_ret20" in tdf.columns else pd.DataFrame()

    fast_wr = (fast["return"] > 0).mean() if len(fast) > 3 else None
    fast_avg = fast["return"].mean() if len(fast) > 3 else None
    slow_wr = (slow["return"] > 0).mean() if len(slow) > 3 else None
    slow_avg = slow["return"].mean() if len(slow) > 3 else None

    # 各退出原因
    reason_detail = {}
    for reason, grp in tdf.groupby("exit_reason"):
        reason_detail[reason] = {
            "count": len(grp),
            "avg_ret": grp["return"].mean(),
            "win_rate": (grp["return"] > 0).mean(),
            "avg_hold": grp["hold_days"].mean(),
        }

    return {
        "label": label,
        "n": n, "wr": wr, "avg_r": avg_r, "med_r": med_r,
        "max_loss": max_loss, "avg_h": avg_h,
        "regret_rate": regret_rate,
        "fast_wr": fast_wr, "fast_avg": fast_avg, "fast_n": len(fast),
        "slow_wr": slow_wr, "slow_avg": slow_avg, "slow_n": len(slow),
        "reason_detail": reason_detail,
    }


# ════════════════════════════════════════════════════
# 策略定义
# ════════════════════════════════════════════════════

# --- A组: 均线动态止损（急缓分离）---
def make_ma_stop(fast_ma, slow_ma, fast_threshold=0.10):
    """急涨(entry_ret20>threshold)用fast_ma，缓涨用slow_ma"""
    def stop_fn(row, h, entry_ret20):
        price = row["close"]
        is_fast = entry_ret20 > fast_threshold if not pd.isna(entry_ret20) else False
        ma_col = f"ma{fast_ma}" if is_fast else f"ma{slow_ma}"
        ma_val = row[ma_col]
        if pd.isna(ma_val):
            return None
        # 收盘价跌破均线 → 次日开盘卖出(这里简化为当天收盘)
        if price < ma_val:
            tag = f"{'急' if is_fast else '缓'}_破MA{fast_ma if is_fast else slow_ma}"
            return tag
        return None
    return stop_fn

# --- B组: 统一均线止损 ---
def make_unified_ma_stop(ma_period):
    def stop_fn(row, h, entry_ret20):
        ma_val = row[f"ma{ma_period}"]
        if pd.isna(ma_val):
            return None
        if row["close"] < ma_val:
            return f"破MA{ma_period}"
        return None
    return stop_fn

# --- C组: 固定止损 ---
def make_fixed_stop(pct):
    def stop_fn(row, h, entry_ret20):
        if h >= 2:  # 至少持有一天
            entry_price = ... # 需要从外部获取，改用不同方式
        return None
    return None  # 不用这个，单独处理

# 固定止损用简单逻辑
def backtest_fixed_stop(stop_pct, entry_interval=20, hold_max=250):
    trades = []
    i = 65
    while i < N - hold_max:
        entry_price = df.loc[i, "close"]
        entry_date = df.loc[i, "date"]
        entry_ret20 = df.loc[i, "ret_20d"]
        exited = False
        for h in range(1, hold_max + 1):
            j = i + h
            if j >= N: break
            cur_ret = (df.loc[j, "close"] - entry_price) / entry_price
            if cur_ret <= -stop_pct:
                trades.append({
                    "entry_date": entry_date, "exit_date": df.loc[j, "date"],
                    "entry_price": entry_price, "exit_price": df.loc[j, "close"],
                    "return": cur_ret, "hold_days": h,
                    "exit_reason": f"止损{stop_pct:.0%}", "entry_ret20": entry_ret20,
                })
                exited = True; break
        if not exited:
            j = min(i + hold_max, N - 1)
            trades.append({
                "entry_date": entry_date, "exit_date": df.loc[j, "date"],
                "entry_price": entry_price, "exit_price": df.loc[j, "close"],
                "return": (df.loc[j, "close"] - entry_price) / entry_price,
                "hold_days": hold_max, "exit_reason": "timeout", "entry_ret20": entry_ret20,
            })
        i += entry_interval
    return pd.DataFrame(trades)


# ════════════════════════════════════════════════════
# 运行所有策略
# ════════════════════════════════════════════════════
results = []

print("  A组: 急缓分离均线止损...", flush=True)
for fast_ma in [5, 10]:
    for slow_ma in [20, 60]:
        for fast_thresh in [0.05, 0.10, 0.15]:
            fn = make_ma_stop(fast_ma, slow_ma, fast_thresh)
            tdf = backtest_strategy(20, 250, fn, f"急MA{fast_ma}/缓MA{slow_ma}(阈值{fast_thresh:.0%})")
            r = calc_stats(tdf, tdf.columns[0])
            if r:
                r["label"] = f"急MA{fast_ma}/缓MA{slow_ma}(>{fast_thresh:.0%})"
                results.append(r)

print("  B组: 统一均线止损...", flush=True)
for ma in [5, 10, 20, 60]:
    fn = make_unified_ma_stop(ma)
    tdf = backtest_strategy(20, 250, fn, f"统一MA{ma}")
    r = calc_stats(tdf, "")
    if r:
        r["label"] = f"统一MA{ma}"
        results.append(r)

print("  C组: 固定止损...", flush=True)
for sp in [0.10, 0.15, 0.20, 0.25]:
    tdf = backtest_fixed_stop(sp)
    r = calc_stats(tdf, "")
    if r:
        r["label"] = f"固定{sp:.0%}"
        results.append(r)

print("  对照组: 买入持有...", flush=True)
tdf_bh = backtest_strategy(20, 250, None, "买入持有")
r = calc_stats(tdf_bh, "")
if r:
    r["label"] = "买入持有"
    results.append(r)


# ════════════════════════════════════════════════════
# 输出
# ════════════════════════════════════════════════════
print(f"\n{'='*100}")
print(f"{'策略':<30} {'交易':>4} {'胜率':>6} {'平均收益':>8} {'中位收益':>8} {'最大亏损':>8} {'平均天数':>7} {'冤杀率':>6}")
print("-" * 100)

# 按类型分组
groups = [
    ("── 急缓分离均线止损 ──", [r for r in results if "/" in r["label"]]),
    ("── 统一均线止损 ──", [r for r in results if r["label"].startswith("统一")]),
    ("── 固定止损 ──", [r for r in results if r["label"].startswith("固定")]),
    ("── 对照组 ──", [r for r in results if r["label"] == "买入持有"]),
]

for header, group in groups:
    if not group:
        continue
    print(f"\n{header}")
    for r in group:
        regret_str = f"{r['regret_rate']:.0%}" if r['regret_rate'] > 0 else "-"
        print(f"  {r['label']:<28} {r['n']:>4} {r['wr']:>5.1%} {r['avg_r']:>7.1%} "
              f"{r['med_r']:>7.1%} {r['max_loss']:>7.1%} {r['avg_h']:>6.0f}天 {regret_str:>6}")

# ── 急涨 vs 缓涨分别表现 ──
print(f"\n{'='*100}")
print("急涨 vs 缓涨 分组表现 (最佳急缓分离策略)")
print("=" * 100)

# 选急缓分离中平均收益最高的
hybrid = [r for r in results if "/" in r["label"]]
if hybrid:
    best = max(hybrid, key=lambda x: x["avg_r"])
    print(f"\n策略: {best['label']}")
    print(f"{'类型':<8} {'笔数':>6} {'胜率':>8} {'平均收益':>10}")
    print("-" * 35)
    print(f"{'急涨':<8} {best['fast_n']:>6} {best['fast_wr']:>7.1%} {best['fast_avg']:>9.1%}" if best['fast_wr'] else f"{'急涨':<8} {best['fast_n']:>6} {'N/A':>8} {'N/A':>10}")
    print(f"{'缓涨':<8} {best['slow_n']:>6} {best['slow_wr']:>7.1%} {best['slow_avg']:>9.1%}" if best['slow_wr'] else f"{'缓涨':<8} {best['slow_n']:>6} {'N/A':>8} {'N/A':>10}")

    # 各退出原因
    print(f"\n退出原因明细:")
    print(f"{'退出原因':<25} {'次数':>6} {'平均收益':>10} {'胜率':>8} {'平均天数':>8}")
    print("-" * 60)
    for reason, stats in best["reason_detail"].items():
        print(f"  {reason:<23} {stats['count']:>6} {stats['avg_ret']:>9.1%} {stats['win_rate']:>7.1%} {stats['avg_hold']:>7.0f}天")

# ── 关键对比：均线止损 vs 固定止损 冤杀率 ──
print(f"\n{'='*100}")
print("核心对比：冤杀率 (止损后60天反弹回成本)")
print("=" * 100)
print(f"\n{'策略':<30} {'止损笔数':>8} {'冤杀率':>8} {'冤杀含义':<40}")
print("-" * 90)
for r in results:
    if r['regret_rate'] > 0:
        total_stop = sum(1 for _, row in pd.DataFrame() for _ in [])  # placeholder
    stop_count = sum(s["count"] for s in r["reason_detail"].values() if s["count"] and "timeout" not in s[0])
    if r['regret_rate'] > 0:
        print(f"  {r['label']:<28} {stop_count:>8} {r['regret_rate']:>7.0%}")

# ── 最终排名 ──
print(f"\n{'='*100}")
print("按平均收益排名 (全部策略)")
print("=" * 100)
ranked = sorted(results, key=lambda x: x["avg_r"], reverse=True)
print(f"\n{'排名':>3} {'策略':<30} {'胜率':>6} {'平均收益':>8} {'最大亏损':>8} {'平均天数':>7} {'冤杀率':>6}")
print("-" * 75)
for i, r in enumerate(ranked, 1):
    regret = f"{r['regret_rate']:.0%}" if r['regret_rate'] > 0 else "-"
    print(f"{i:>3} {r['label']:<30} {r['wr']:>5.1%} {r['avg_r']:>7.1%} {r['max_loss']:>7.1%} {r['avg_h']:>6.0f}天 {regret:>6}")
