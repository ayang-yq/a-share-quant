"""
A股：大盘择时止损 vs 固定百分比止损 — 回测对比
==================================================
核心问题：利弗莫尔的固定%止损在A股被正常波动反复触发，
能否用BB大盘择时替代，减少无效止损同时保留风险保护？

回测设计：
  - 每20天滚动建仓（模拟分散买入）
  - 三种退出策略对比
  - 标的：沪深300指数（个股结果只更差）
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, akshare as ak

print("=" * 60)
print("大盘择时止损 vs 固定百分比止损")
print("=" * 60)

# ── 数据 ──
print("  沪深300日K...", end=" ", flush=True)
df = ak.stock_zh_index_daily(symbol="sh000300")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df = df[(df.date >= "2015-01-01") & (df.date <= "2026-04-30")].reset_index(drop=True)
print(f"OK ({len(df)}天)")

# BB指标
W = 20
df["bb_ma"] = df["close"].rolling(W).mean()
df["bb_std"] = df["close"].rolling(W).std()
df["bb_z"] = (df["close"] - df["bb_ma"]) / df["bb_std"]

# 日收益率
df["ret"] = df["close"].pct_change()

# ── 回测函数 ──
def simulate_trades(entry_interval=20, hold_max=250, stop_pct=None, bb_exit_z=None):
    """
    滚动建仓模拟：
    每entry_interval天买入，持有到：
      - 止损触发(stop_pct)
      - BB信号触发(bb_exit_z，BB z < 该值时卖出)
      - 最长持有hold_max天
      - 三者取最先触发
    """
    trades = []
    i = W + 5  # 跳过BB预热期
    while i < len(df) - hold_max:
        entry_date = df.loc[i, "date"]
        entry_price = df.loc[i, "close"]
        pos = 1.0  # 持仓状态
        entry_bb_z = df.loc[i, "bb_z"]

        for h in range(1, hold_max + 1):
            j = i + h
            if j >= len(df):
                break
            cur_price = df.loc[j, "close"]
            cur_ret = (cur_price - entry_price) / entry_price
            cur_bb_z = df.loc[j, "bb_z"]

            # 止损检查
            if stop_pct and cur_ret <= -stop_pct:
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": df.loc[j, "date"],
                    "entry_price": entry_price,
                    "exit_price": cur_price,
                    "return": cur_ret,
                    "hold_days": h,
                    "exit_reason": f"stop_{stop_pct:.0%}",
                    "entry_bb_z": entry_bb_z,
                    "exit_bb_z": cur_bb_z,
                })
                pos = 0
                break

            # BB择时退出检查
            if bb_exit_z is not None and pd.notna(cur_bb_z) and cur_bb_z < bb_exit_z:
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": df.loc[j, "date"],
                    "entry_price": entry_price,
                    "exit_price": cur_price,
                    "return": cur_ret,
                    "hold_days": h,
                    "exit_reason": f"bb_exit_{bb_exit_z}",
                    "entry_bb_z": entry_bb_z,
                    "exit_bb_z": cur_bb_z,
                })
                pos = 0
                break

        # 超时退出
        if pos == 1:
            j = min(i + hold_max, len(df) - 1)
            cur_price = df.loc[j, "close"]
            cur_ret = (cur_price - entry_price) / entry_price
            trades.append({
                "entry_date": entry_date,
                "exit_date": df.loc[j, "date"],
                "entry_price": entry_price,
                "exit_price": cur_price,
                "return": cur_ret,
                "hold_days": hold_max,
                "exit_reason": "timeout",
                "entry_bb_z": entry_bb_z,
                "exit_bb_z": df.loc[j, "bb_z"],
            })

        i += entry_interval

    return pd.DataFrame(trades)


def summarize(tdf, label):
    """汇总交易统计"""
    if len(tdf) == 0:
        return None
    total_trades = len(tdf)
    win_rate = (tdf["return"] > 0).mean()
    avg_return = tdf["return"].mean()
    median_return = tdf["return"].median()
    avg_hold = tdf["hold_days"].mean()
    max_loss = tdf["return"].min()

    # 累计净值（每笔交易等权重）
    nav = np.cumprod(1 + tdf["return"].values)
    total_return = nav[-1] - 1
    # 假设每笔交易平均持仓期，算年化
    avg_days = tdf["hold_days"].mean()
    n_years = len(tdf) * avg_days / 252
    annual = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # 止损交易占比
    stop_trades = tdf[tdf["exit_reason"].str.startswith("stop")].shape[0]
    bb_trades = tdf[tdf["exit_reason"].str.startswith("bb")].shape[0]
    timeout_trades = tdf[tdf["exit_reason"] == "timeout"].shape[0]

    # 止损后反弹（被止损的笔，如果持有到250天会怎样）
    stop_df = tdf[tdf["exit_reason"].str.startswith("stop")]
    regret_count = 0
    if len(stop_df) > 0:
        for _, row in stop_df.iterrows():
            # 看止损后60天内是否反弹回成本线以上
            entry_idx = df[df["date"] == row["entry_date"]].index
            exit_idx = df[df["date"] == row["exit_date"]].index
            if len(entry_idx) > 0 and len(exit_idx) > 0:
                future_idx = min(exit_idx[0] + 60, len(df) - 1)
                future_price = df.loc[future_idx, "close"]
                if future_price >= row["entry_price"]:
                    regret_count += 1
        regret_rate = regret_count / len(stop_df) if len(stop_df) > 0 else 0
    else:
        regret_rate = 0

    # 各退出原因的平均收益
    reason_stats = {}
    for reason, grp in tdf.groupby("exit_reason"):
        reason_stats[reason] = {
            "count": len(grp),
            "avg_ret": grp["return"].mean(),
            "win_rate": (grp["return"] > 0).mean(),
        }

    return {
        "label": label,
        "交易次数": total_trades,
        "胜率": win_rate,
        "平均收益": avg_return,
        "中位收益": median_return,
        "最大单笔亏损": max_loss,
        "平均持仓天数": avg_hold,
        "年化收益": annual,
        "止损退出": stop_trades,
        "BB退出": bb_trades,
        "超时退出": timeout_trades,
        "止损后反弹率": regret_rate,
        "reason_stats": reason_stats,
        "total_return": total_return,
    }

# ── 运行回测 ──
results = []

# 1. 固定止损策略
print("\n  回测固定止损策略...", flush=True)
for sp in [0.08, 0.10, 0.15, 0.20, 0.25, 0.30]:
    tdf = simulate_trades(entry_interval=20, hold_max=250, stop_pct=sp)
    r = summarize(tdf, f"止损{sp:.0%}")
    if r: results.append(r)

# 2. BB择时退出策略
print("  回测BB择时策略...", flush=True)
for bb_z in [-2.0, -1.5, -1.0, -0.5, 0.0]:
    tdf = simulate_trades(entry_interval=20, hold_max=250, bb_exit_z=bb_z)
    r = summarize(tdf, f"BB退出(z<{bb_z})")
    if r: results.append(r)

# 3. 混合策略：BB择时 + 极端止损(只在大跌时止损)
print("  回测混合策略...", flush=True)
for bb_z in [-1.5, -1.0, -0.5]:
    for sp in [0.20, 0.25, 0.30]:
        # 先BB退出，BB不触发才看止损
        trades = []
        i = W + 5
        while i < len(df) - 250:
            entry_date = df.loc[i, "date"]
            entry_price = df.loc[i, "close"]
            entry_bb_z = df.loc[i, "bb_z"]
            exited = False

            for h in range(1, 251):
                j = i + h
                if j >= len(df):
                    break
                cur_price = df.loc[j, "close"]
                cur_ret = (cur_price - entry_price) / entry_price
                cur_bb_z = df.loc[j, "bb_z"]

                # 极端止损优先（跌破30%无条件走）
                if cur_ret <= -sp:
                    trades.append({
                        "entry_date": entry_date, "exit_date": df.loc[j, "date"],
                        "entry_price": entry_price, "exit_price": cur_price,
                        "return": cur_ret, "hold_days": h,
                        "exit_reason": f"extreme_stop_{sp:.0%}",
                        "entry_bb_z": entry_bb_z, "exit_bb_z": cur_bb_z,
                    })
                    exited = True
                    break

                # BB择时退出
                if pd.notna(cur_bb_z) and cur_bb_z < bb_z:
                    trades.append({
                        "entry_date": entry_date, "exit_date": df.loc[j, "date"],
                        "entry_price": entry_price, "exit_price": cur_price,
                        "return": cur_ret, "hold_days": h,
                        "exit_reason": f"bb_{bb_z}",
                        "entry_bb_z": entry_bb_z, "exit_bb_z": cur_bb_z,
                    })
                    exited = True
                    break

            if not exited:
                j = min(i + 250, len(df) - 1)
                cur_ret = (df.loc[j, "close"] - entry_price) / entry_price
                trades.append({
                    "entry_date": entry_date, "exit_date": df.loc[j, "date"],
                    "entry_price": entry_price, "exit_price": df.loc[j, "close"],
                    "return": cur_ret, "hold_days": 250,
                    "exit_reason": "timeout",
                    "entry_bb_z": entry_bb_z, "exit_bb_z": df.loc[j, "bb_z"],
                })

            i += 20

        r = summarize(pd.DataFrame(trades), f"BB{bb_z}+止损{sp:.0%}")
        if r: results.append(r)

# 4. 对照组：买入持有250天不退出
print("  回测买入持有...", flush=True)
tdf_bh = simulate_trades(entry_interval=20, hold_max=250, stop_pct=None, bb_exit_z=None)
r = summarize(tdf_bh, "买入持有250天")
if r: results.append(r)

# ── 输出结果 ──
print(f"\n{'='*90}")
print(f"{'策略':<22} {'交易':>4} {'胜率':>6} {'平均收益':>8} {'最大亏损':>8} {'平均天数':>8} {'止损后反弹':>9}")
print(f"{'':22} {'次数':>4} {'':>6} {'':>8} {'':>8} {'':>8} {'':>9}")
print("-" * 90)

# 分类排序
pure_stop = [r for r in results if r["label"].startswith("止损")]
pure_bb = [r for r in results if r["label"].startswith("BB退出")]
hybrid = [r for r in results if "+" in r["label"] and "止损" in r["label"]]
bh = [r for r in results if r["label"] == "买入持有250天"]

for group, header in [(pure_stop, "── 固定止损 ──"), (bh, "── 对照组 ──"), (pure_bb, "── BB择时退出 ──"), (hybrid, "── 混合策略(BB+极端止损) ──")]:
    if group:
        print(f"\n{header}")
        for r in group:
            regret = f"{r['止损后反弹率']:.0%}" if r['止损后反弹率'] > 0 else "-"
            print(f"{r['label']:<22} {r['交易次数']:>4} {r['胜率']:>5.1%} {r['平均收益']:>7.1%} {r['最大单笔亏损']:>7.1%} {r['平均持仓天数']:>7.0f}天 {regret:>9}")

# ── 深度分析：止损后到底发生了什么 ──
print(f"\n{'='*90}")
print("深度分析：止损后60天内价格走势")
print("=" * 90)

for sp in [0.10, 0.15, 0.20]:
    tdf = simulate_trades(entry_interval=20, hold_max=250, stop_pct=sp)
    stop_df = tdf[tdf["exit_reason"].str.startswith("stop")]
    if len(stop_df) == 0:
        continue

    outcomes = {"反弹回本": 0, "涨10%+": 0, "续跌10%+": 0, "续跌20%+": 0}
    for _, row in stop_df.iterrows():
        exit_idx = df[df["date"] == row["exit_date"]].index
        if len(exit_idx) == 0:
            continue
        idx = exit_idx[0]
        for future_d in [20, 40, 60]:
            fi = min(idx + future_d, len(df) - 1)
            fp = df.loc[fi, "close"]
            ret_from_exit = (fp - row["exit_price"]) / row["exit_price"]
            ret_from_entry = (fp - row["entry_price"]) / row["entry_price"]

        if ret_from_entry > 0:
            outcomes["反弹回本"] += 1
        if ret_from_exit > 0.10:
            outcomes["涨10%+"] += 1
        if ret_from_exit < -0.10:
            outcomes["续跌10%+"] += 1
        if ret_from_exit < -0.20:
            outcomes["续跌20%+"] += 1

    n = len(stop_df)
    print(f"\n止损{sp:.0%} (共{n}笔止损):")
    print(f"  止损后60天内反弹回成本线: {outcomes['反弹回本']} ({outcomes['反弹回本']/n:.0%})")
    print(f"  止损后60天内从止损价涨10%+: {outcomes['涨10%+']} ({outcomes['涨10%+']/n:.0%})")
    print(f"  止损后60天内从止损价续跌10%+: {outcomes['续跌10%+']} ({outcomes['续跌10%+']/n:.0%})")
    print(f"  止损后60天内从止损价续跌20%+: {outcomes['续跌20%+']} ({outcomes['续跌20%+']/n:.0%})")

# ── 各退出原因明细 ──
print(f"\n{'='*90}")
print("最佳混合策略：各退出原因效果")
print("=" * 90)

# 找夏普最高的混合策略
if hybrid:
    # 选交易次数合理且平均收益最好的
    best_hybrid = max([r for r in hybrid if r["交易次数"] >= 20], key=lambda x: x["平均收益"])
    print(f"\n最佳混合: {best_hybrid['label']}")
    print(f"{'退出原因':<20} {'次数':>6} {'平均收益':>10} {'胜率':>8}")
    print("-" * 48)
    for reason, stats in best_hybrid["reason_stats"].items():
        print(f"{reason:<20} {stats['count']:>6} {stats['avg_ret']:>9.1%} {stats['win_rate']:>7.1%}")

# ── 最终推荐 ──
print(f"\n{'='*90}")
print("结论")
print("=" * 90)

# 对比：止损10% vs 最佳BB vs 混合
stop10 = next((r for r in results if r["label"] == "止损10%"), None)
bb_best = next((r for r in pure_bb), None) if pure_bb else None
if hybrid:
    best_mix = max([r for r in hybrid if r["交易次数"] >= 20], key=lambda x: x["平均收益"])
else:
    best_mix = None
bh_r = bh[0] if bh else None

print(f"\n{'策略':<22} {'胜率':>6} {'平均收益':>8} {'平均天数':>8}")
print("-" * 48)
for r in [stop10, bb_best, best_mix, bh_r]:
    if r:
        print(f"{r['label']:<22} {r['胜率']:>5.1%} {r['平均收益']:>7.1%} {r['平均持仓天数']:>7.0f}天")
