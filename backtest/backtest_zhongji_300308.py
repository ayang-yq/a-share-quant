"""中际旭创(300308) — 个股止损回测明细"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, akshare as ak

print("=" * 60)
print("中际旭创(300308) 止损回测")
print("=" * 60)

# ── 沪深300 BB ──
df_300 = ak.stock_zh_index_daily(symbol="sh000300")
df_300["date"] = pd.to_datetime(df_300["date"])
df_300 = df_300.sort_values("date").reset_index(drop=True)
df_300 = df_300[(df_300.date >= "2018-01-01") & (df_300.date <= "2026-04-30")].reset_index(drop=True)
df_300["bb_z"] = (df_300["close"] - df_300["close"].rolling(20).mean()) / df_300["close"].rolling(20).std()
bb_map = dict(zip(df_300["date"], df_300["bb_z"]))

# ── 个股数据 ──
print("  获取中际旭创数据...", flush=True)
df = ak.stock_zh_a_hist(symbol="300308", period="daily", start_date="20180101", end_date="20260430")
df["date"] = pd.to_datetime(df["日期"])
df["close"] = df["收盘"]
df["high"] = df["最高"]
df["low"] = df["最低"]
df["vol"] = df["成交量"]
df = df.sort_values("date").reset_index(drop=True)
for w in [5, 10, 20, 60]:
    df[f"ma{w}"] = df["close"].rolling(w).mean()
df["ret20"] = df["close"].pct_change(20)
print(f"  OK ({len(df)}天, {df['date'].min().date()} ~ {df['date'].max().date()})")
print(f"  股价区间: {df['close'].min():.2f} ~ {df['close'].max():.2f}")

ENTRY_BB_MAX = 0.5
HOLD_MAX = 120
REENTRY_WAIT = 10

# ── 回测函数 ──
def run_backtest(df, strategy, **kw):
    trades = []
    i = 65
    last_exit = -999

    while i < len(df) - HOLD_MAX:
        z = bb_map.get(df.loc[i, "date"], np.nan)
        if pd.isna(z) or z >= ENTRY_BB_MAX or i - last_exit < REENTRY_WAIT:
            i += 1; continue

        ep = df.loc[i, "close"]
        ed = df.loc[i, "date"]
        er20 = df.loc[i, "ret20"]
        entry_bb = z
        exited = False

        for h in range(1, HOLD_MAX + 1):
            j = i + h
            if j >= len(df): break
            cp = df.loc[j, "close"]
            ret = (cp - ep) / ep
            reason = None

            if strategy == "no_stop":
                pass
            elif strategy == "fixed" and ret <= -kw["stop_pct"]:
                reason = f"止损{kw['stop_pct']:.0%}"
            elif strategy == "ma_dyn":
                is_fast = er20 > kw.get("fast_thresh", 0.10) if not pd.isna(er20) else False
                mw = kw["ma_fast"] if is_fast else kw["ma_slow"]
                mv = df.loc[j, f"ma{mw}"]
                if not pd.isna(mv) and cp < mv:
                    reason = f"{'急' if is_fast else '缓'}破MA{mw}"
            elif strategy == "ma_uni":
                mv = df.loc[j, f"ma{kw['ma']}"]
                if not pd.isna(mv) and cp < mv:
                    reason = f"破MA{kw['ma']}"
            elif strategy == "hybrid":
                # 均线止损 + 极端固定止损
                ma_mv = df.loc[j, f"ma{kw['ma']}"]
                ma_hit = not pd.isna(ma_mv) and cp < ma_mv
                extreme_hit = ret <= -kw.get("extreme_stop", 0.25)
                if extreme_hit:
                    reason = f"极端止损{kw.get('extreme_stop', 0.25):.0%}"
                elif ma_hit:
                    reason = f"破MA{kw['ma']}"

            if reason:
                trades.append({
                    "entry_date": ed, "exit_date": df.loc[j, "date"],
                    "entry_price": ep, "exit_price": cp,
                    "return": ret, "hold_days": h,
                    "exit_reason": reason, "entry_bb": entry_bb,
                    "exit_bb": bb_map.get(df.loc[j, "date"], np.nan),
                    "exit_ma5": df.loc[j, "ma5"], "exit_ma10": df.loc[j, "ma10"],
                    "exit_ma20": df.loc[j, "ma20"], "exit_ma60": df.loc[j, "ma60"],
                })
                last_exit = j; exited = True; break

        if not exited:
            j = min(i + HOLD_MAX, len(df) - 1)
            trades.append({
                "entry_date": ed, "exit_date": df.loc[j, "date"],
                "entry_price": ep, "exit_price": df.loc[j, "close"],
                "return": (df.loc[j, "close"] - ep) / ep,
                "hold_days": HOLD_MAX, "exit_reason": "持有到期",
                "entry_bb": entry_bb,
                "exit_bb": bb_map.get(df.loc[j, "date"], np.nan),
                "exit_ma5": df.loc[j, "ma5"], "exit_ma10": df.loc[j, "ma10"],
                "exit_ma20": df.loc[j, "ma20"], "exit_ma60": df.loc[j, "ma60"],
            })
            last_exit = j
        i += 1
    return pd.DataFrame(trades) if trades else pd.DataFrame()

# ── 运行 ──
strategies = [
    ("买入持有", "no_stop", {}),
    ("固定10%止损", "fixed", {"stop_pct": 0.10}),
    ("固定15%止损", "fixed", {"stop_pct": 0.15}),
    ("固定20%止损", "fixed", {"stop_pct": 0.20}),
    ("固定25%止损", "fixed", {"stop_pct": 0.25}),
    ("固定30%止损", "fixed", {"stop_pct": 0.30}),
    ("急MA5/缓MA60(>10%)", "ma_dyn", {"ma_fast": 5, "ma_slow": 60, "fast_thresh": 0.10}),
    ("急MA10/缓MA60(>10%)", "ma_dyn", {"ma_fast": 10, "ma_slow": 60, "fast_thresh": 0.10}),
    ("统一MA10", "ma_uni", {"ma": 10}),
    ("统一MA20", "ma_uni", {"ma": 20}),
    ("统一MA60", "ma_uni", {"ma": 60}),
    ("MA20+极端止损25%", "hybrid", {"ma": 20, "extreme_stop": 0.25}),
    ("MA60+极端止损30%", "hybrid", {"ma": 60, "extreme_stop": 0.30}),
]

all_results = {}
for name, stype, params in strategies:
    tdf = run_backtest(df, stype, **params)
    if len(tdf) > 0:
        n = len(tdf)
        wr = (tdf["return"] > 0).mean()
        avg_r = tdf["return"].mean()
        total_r = (np.prod(1 + tdf["return"].values) - 1)
        max_loss = tdf["return"].min()
        avg_h = tdf["hold_days"].mean()
        all_results[name] = {
            "df": tdf, "n": n, "wr": wr, "avg_r": avg_r,
            "total": total_r, "max_loss": max_loss, "avg_h": avg_h,
        }

# ── 汇总 ──
print(f"\n{'='*100}")
print(f"{'策略':<25} {'交易':>4} {'胜率':>6} {'均收益':>8} {'总收益':>8} {'最亏':>8} {'均天数':>6}")
print("-" * 75)
for name, r in all_results.items():
    print(f"  {name:<23} {r['n']:>4} {r['wr']:>5.1%} {r['avg_r']:>7.1%} {r['total']:>7.1%} {r['max_loss']:>7.1%} {r['avg_h']:>5.0f}天")

# ── 详细交易记录: 5个代表策略 ──
detail_strategies = ["买入持有", "固定15%止损", "急MA5/缓MA60(>10%)", "统一MA20", "MA20+极端止损25%"]

for strat_name in detail_strategies:
    if strat_name not in all_results:
        continue
    r = all_results[strat_name]
    tdf = r["df"]
    print(f"\n{'='*100}")
    print(f"  {strat_name} — 逐笔明细 ({r['n']}笔)")
    print(f"{'='*100}")
    print(f"{'#':>3} {'买入日期':<12} {'买入价':>8} {'卖出日期':<12} {'卖出价':>8} {'收益':>8} {'天数':>5} {'退出原因':<15} {'入场BB':>6} {'卖出BB':>6}")
    print("-" * 100)
    for idx, (_, row) in enumerate(tdf.iterrows(), 1):
        bb_entry = f"{row['entry_bb']:.2f}" if not pd.isna(row['entry_bb']) else "N/A"
        bb_exit = f"{row['exit_bb']:.2f}" if not pd.isna(row['exit_bb']) else "N/A"
        ret_str = f"{row['return']:>7.1%}"
        if row['return'] > 0.20:
            ret_str += " 🚀"
        elif row['return'] > 0.10:
            ret_str += " ✓"
        elif row['return'] < -0.15:
            ret_str += " 💀"
        print(f"{idx:>3} {row['entry_date'].strftime('%Y-%m-%d'):<12} {row['entry_price']:>8.2f} "
              f"{row['exit_date'].strftime('%Y-%m-%d'):<12} {row['exit_price']:>8.2f} "
              f"{ret_str:<12} {row['hold_days']:>4}天 {row['exit_reason']:<15} {bb_entry:>6} {bb_exit:>6}")

# ── 冤杀分析 ──
print(f"\n{'='*100}")
print("冤杀分析: 止损后60天内价格走势")
print("=" * 100)

for strat_name in ["固定15%止损", "急MA5/缓MA60(>10%)", "统一MA20"]:
    if strat_name not in all_results:
        continue
    tdf = all_results[strat_name]["df"]
    stop_df = tdf[tdf["exit_reason"] != "持有到期"]
    if len(stop_df) == 0:
        continue

    print(f"\n{strat_name} ({len(stop_df)}笔止损):")
    print(f"  {'#':>3} {'止损日期':<12} {'止损价':>8} {'成本价':>8} {'止损收益':>8} {'60天后价':>8} {'60天涨幅':>8} {'是否冤杀'}")
    print("  " + "-" * 80)

    regret_count = 0
    for idx, (_, row) in enumerate(stop_df.iterrows(), 1):
        exit_ts = pd.Timestamp(row["exit_date"])
        future_ts = exit_ts + pd.Timedelta(days=60)
        # 找该股票60天后的价格
        future_rows = df[(df["date"] > exit_ts) & (df["date"] <= future_ts)]
        if len(future_rows) == 0:
            future_price = row["exit_price"]
        else:
            future_price = future_rows["close"].iloc[-1]

        ret_from_stop = (future_price - row["exit_price"]) / row["exit_price"]
        ret_from_entry = (future_price - row["entry_price"]) / row["entry_price"]
        is_regret = ret_from_entry > 0
        if is_regret:
            regret_count += 1

        regret_mark = "⚠️冤杀" if is_regret else "✓正确"
        print(f"  {idx:>3} {row['exit_date'].strftime('%Y-%m-%d'):<12} {row['exit_price']:>8.2f} "
              f"{row['entry_price']:>8.2f} {row['return']:>7.1%} {future_price:>8.2f} "
              f"{ret_from_stop:>+7.1%} {retreat_mark}")

    print(f"\n  冤杀率: {regret_count}/{len(stop_df)} = {regret_count/len(stop_df):.0%}")
