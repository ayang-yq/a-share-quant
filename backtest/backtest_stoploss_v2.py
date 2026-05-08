"""
A股个股止损回测 v2.1 — 简化但正确的组合指标
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, akshare as ak, time

print("=" * 70)
print("A股个股止损回测 v2.1")
print("=" * 70)

# ── 数据 ──
print("\n  获取成分股...", end=" ", flush=True)
try:
    cons = ak.index_stock_cons_csindex(symbol="000300")
    tickers = cons["成分券代码"].tolist()[:50]
except:
    cons = ak.index_stock_cons(symbol="000300")
    tickers = cons["品种代码"].tolist()[:50]

df_300 = ak.stock_zh_index_daily(symbol="sh000300")
df_300["date"] = pd.to_datetime(df_300["date"])
df_300 = df_300.sort_values("date").reset_index(drop=True)
df_300 = df_300[(df_300.date >= "2018-01-01") & (df_300.date <= "2026-04-30")].reset_index(drop=True)
W_BB = 20
df_300["bb_z"] = (df_300["close"] - df_300["close"].rolling(W_BB).mean()) / df_300["close"].rolling(W_BB).std()
bb_map = dict(zip(df_300["date"], df_300["bb_z"]))
print(f"OK ({len(tickers)}只股票)")

ENTRY_BB_MAX = 0.5
HOLD_MAX = 120
REENTRY_WAIT = 10

# ── 单股回测 ──
def run_stock(ticker, strategy, **kw):
    try:
        df = ak.stock_zh_a_hist(symbol=ticker, period="daily", start_date="20180101", end_date="20260430")
    except:
        return None
    if len(df) < 300:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["日期"])
    df["close"] = df["收盘"]
    df = df.sort_values("date").reset_index(drop=True)
    for w in [5, 10, 20, 60]:
        df[f"ma{w}"] = df["close"].rolling(w).mean()
    df["ret20"] = df["close"].pct_change(20)

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
        exited = False

        for h in range(1, HOLD_MAX + 1):
            j = i + h
            if j >= len(df): break
            cp = df.loc[j, "close"]
            ret = (cp - ep) / ep
            reason = None

            if strategy == "fixed" and ret <= -kw["stop_pct"]:
                reason = "stop"
            elif strategy == "ma_dyn":
                is_fast = er20 > kw.get("fast_thresh", 0.10) if not pd.isna(er20) else False
                mw = kw["ma_fast"] if is_fast else kw["ma_slow"]
                mv = df.loc[j, f"ma{mw}"]
                if not pd.isna(mv) and cp < mv:
                    reason = f"ma_{mw}"
            elif strategy == "ma_uni":
                mv = df.loc[j, f"ma{kw['ma']}"]
                if not pd.isna(mv) and cp < mv:
                    reason = f"ma_{kw['ma']}"

            if reason:
                trades.append({"entry_date": ed, "exit_date": df.loc[j, "date"],
                               "entry_price": ep, "exit_price": cp,
                               "return": ret, "hold_days": h,
                               "exit_reason": reason, "entry_ret20": er20})
                last_exit = j; exited = True; break

        if not exited:
            j = min(i + HOLD_MAX, len(df) - 1)
            trades.append({"entry_date": ed, "exit_date": df.loc[j, "date"],
                           "entry_price": ep, "exit_price": df.loc[j, "close"],
                           "return": (df.loc[j, "close"] - ep) / ep,
                           "hold_days": HOLD_MAX, "exit_reason": "timeout",
                           "entry_ret20": er20})
            last_exit = j
        i += 1
    return pd.DataFrame(trades) if trades else None


# ── 组合指标: 简单直接 ──
def portfolio_stats(all_trades_list):
    """
    all_trades_list: [(ticker, trades_df), ...]
    计算组合层面指标。假设同时只持有1笔交易(串行按日期)。
    """
    # 展开所有交易, 按日期排序
    all_t = []
    for ticker, tdf in all_trades_list:
        for _, row in tdf.iterrows():
            all_t.append({
                "entry_date": pd.Timestamp(row["entry_date"]),
                "exit_date": pd.Timestamp(row["exit_date"]),
                "entry_price": row["entry_price"],
                "exit_price": row["exit_price"],
                "return": row["return"],
                "hold_days": row["hold_days"],
                "exit_reason": row["exit_reason"],
            })
    if not all_t:
        return None
    adf = pd.DataFrame(all_t).sort_values("entry_date").reset_index(drop=True)

    # 构建日度PnL序列: 每天计算当前持仓的浮动盈亏
    start_date = adf["entry_date"].min()
    end_date = adf["exit_date"].max()
    all_dates = pd.bdate_range(start_date, end_date)

    # 每天的持仓状态
    daily_nav = []
    active_trade = None
    event_idx = 0
    sorted_events = sorted(
        [(t["entry_date"], "entry", i) for i, t in adf.iterrows()] +
        [(t["exit_date"], "exit", i) for i, t in adf.iterrows()],
        key=lambda x: (x[0], 0 if x[1] == "exit" else 1)
    )

    nav = 1.0
    daily_returns = []

    for dt in all_dates:
        # 处理当天的事件(先exit再entry)
        while event_idx < len(sorted_events) and sorted_events[event_idx][0] <= dt:
            evt_dt, evt_type, evt_i = sorted_events[event_idx]
            if evt_type == "exit" and active_trade is not None:
                ret = active_trade["return"]
                nav *= (1 + ret)
                daily_returns.append(ret / max(active_trade["hold_days"], 1))
                active_trade = None
            elif evt_type == "entry" and active_trade is None:
                active_trade = adf.iloc[evt_i]
            event_idx += 1

        daily_nav.append({"date": dt, "nav": nav})

    if not daily_nav:
        return None

    nav_series = pd.Series([d["nav"] for d in daily_nav],
                           index=[d["date"] for d in daily_nav])

    # 实际日收益率
    dr = nav_series.pct_change().dropna()
    # 去掉0收益日(空仓期)
    dr = dr[dr != 0]

    if len(dr) < 50:
        return None

    rf_d = 1.02 ** (1/252) - 1
    excess = dr - rf_d
    sharpe = np.sqrt(252) * excess.mean() / dr.std() if dr.std() > 0 else 0
    downside = dr[dr < 0]
    sortino = np.sqrt(252) * excess.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else 0
    cummax = nav_series.cummax()
    max_dd = ((nav_series - cummax) / cummax).min()

    total_ret = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    n_years = (nav_series.index[-1] - nav_series.index[0]).days / 365.25
    annual = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0

    # 交易统计
    n_trades = len(adf)
    win_rate = (adf["return"] > 0).mean()
    avg_return = adf["return"].mean()
    max_loss = adf["return"].min()
    avg_hold = adf["hold_days"].mean()

    # 冤杀率: 止损后120天内价格是否涨回成本
    stop_df = adf[adf["exit_reason"] != "timeout"]
    regret = 0
    total_stop = len(stop_df)
    if total_stop > 0:
        for _, row in stop_df.iterrows():
            exit_ts = pd.Timestamp(row["exit_date"])
            future_ts = exit_ts + pd.Timedelta(days=120)
            # 用沪深300近似
            mask = (df_300["date"] >= exit_ts) & (df_300["date"] <= future_ts)
            if mask.sum() > 0:
                future_300 = df_300.loc[mask, "close"].max()
                entry_300_approx = df_300.loc[df_300["date"] <= exit_ts, "close"].iloc[-1] if (df_300["date"] <= exit_ts).sum() > 0 else row["exit_price"]
                # 如果指数涨回了成本线以上
                ret_from_stop = (future_300 - entry_300_approx) / entry_300_approx
                if ret_from_stop > 0.05:  # 指数涨了5%以上, 视为冤杀
                    regret += 1
        regret_rate = regret / total_stop
    else:
        regret_rate = 0

    return {
        "n_stocks": len(all_trades_list), "n_trades": n_trades,
        "win_rate": win_rate, "avg_return": avg_return, "max_loss": max_loss,
        "avg_hold": avg_hold, "total_return": total_ret, "annual": annual,
        "sharpe": sharpe, "sortino": sortino, "max_dd": max_dd,
        "calmar": annual / abs(max_dd) if max_dd != 0 else 0,
        "regret_rate": regret_rate,
    }


# ── 策略列表 ──
strategies = [
    ("买入持有", "no_stop", {}),
    ("固定10%", "fixed", {"stop_pct": 0.10}),
    ("固定15%", "fixed", {"stop_pct": 0.15}),
    ("固定20%", "fixed", {"stop_pct": 0.20}),
    ("固定25%", "fixed", {"stop_pct": 0.25}),
    ("固定30%", "fixed", {"stop_pct": 0.30}),
    ("急MA5/缓MA60(>10%)", "ma_dyn", {"ma_fast": 5, "ma_slow": 60, "fast_thresh": 0.10}),
    ("急MA10/缓MA60(>10%)", "ma_dyn", {"ma_fast": 10, "ma_slow": 60, "fast_thresh": 0.10}),
    ("急MA5/缓MA60(>5%)", "ma_dyn", {"ma_fast": 5, "ma_slow": 60, "fast_thresh": 0.05}),
    ("统一MA10", "ma_uni", {"ma": 10}),
    ("统一MA20", "ma_uni", {"ma": 20}),
    ("统一MA60", "ma_uni", {"ma": 60}),
]

# ── 执行 ──
print(f"  回测区间: 2018-07 ~ 2026-04")
print(f"  入场: BB z<{ENTRY_BB_MAX}, 持仓≤{HOLD_MAX}天, 冷却{REENTRY_WAIT}天\n")

all_results = {}
for name, stype, params in strategies:
    all_trades = []
    t0 = time.time()
    for ticker in tickers:
        tdf = run_stock(ticker, stype, **params)
        if tdf is not None and len(tdf) > 0:
            all_trades.append((ticker, tdf))
    metrics = portfolio_stats(all_trades)
    if metrics:
        metrics["label"] = name
        all_results[name] = metrics
    elapsed = time.time() - t0
    n_trades = metrics["n_trades"] if metrics else 0
    print(f"  {name:<25} {len(all_trades):>2}只 {n_trades:>4}笔 {elapsed:>3.0f}s", flush=True)

# ── 输出 ──
print(f"\n{'='*115}")
print(f"{'策略':<25} {'股票':>4} {'交易':>4} {'胜率':>6} {'均收益':>8} {'最亏损':>8} {'均天数':>6} "
      f"{'年化':>7} {'夏普':>6} {'Sortino':>7} {'最大回撤':>8} {'Calmar':>6} {'冤杀率':>6}")
print("-" * 115)

ranked = sorted(all_results.values(), key=lambda x: x["sortino"], reverse=True)
for r in ranked:
    reg = f"{r['regret_rate']:.0%}" if r['regret_rate'] > 0 else "-"
    print(f"  {r['label']:<23} {r['n_stocks']:>4} {r['n_trades']:>4} {r['win_rate']:>5.1%} "
          f"{r['avg_return']:>7.1%} {r['max_loss']:>7.1%} {r['avg_hold']:>5.0f}d "
          f"{r['annual']:>6.1%} {r['sharpe']:>6.2f} {r['sortino']:>7.2f} "
          f"{r['max_dd']:>7.1%} {r['calmar']:>6.2f} {reg:>6}")

# vs 基准
bh = all_results.get("买入持有", {})
print(f"\n{'='*115}")
print("各策略 vs 买入持有")
print("=" * 115)
print(f"\n基准: 年化{bh.get('annual',0):.1%}, Sortino{bh.get('sortino',0):.2f}, 回撤{bh.get('max_dd',0):.1%}\n")
print(f"{'策略':<25} {'ΔSortino':>9} {'Δ回撤':>9} {'Δ年化':>9} {'结论'}")
print("-" * 70)
for r in ranked:
    ds = r["sortino"] - bh.get("sortino", 0)
    dd = abs(bh.get("max_dd", 0)) - abs(r["max_dd"])
    da = r["annual"] - bh.get("annual", 0)
    if ds > 0.05 and dd > 0.02:
        v = "✅ 全面优于基准"
    elif dd > 0.05 and da > -0.02:
        v = "⚠️ 回撤大幅改善, 收益略降"
    elif ds < -0.1:
        v = "❌ 显著劣于基准"
    elif abs(ds) < 0.05 and abs(dd) < 0.02:
        v = "≈ 持平"
    else:
        v = "—"
    print(f"  {r['label']:<23} {ds:>+8.2f} {dd:>+8.1pp} {da:>+8.1%} {v}")

print(f"\n{'='*115}")
print(f"最佳Sortino: {ranked[0]['label']} ({ranked[0]['sortino']:.2f})")
best_dd = max(ranked, key=lambda x: abs(x['max_dd']) - abs(bh.get('max_dd', 0)))
print(f"最大回撤改善: {best_dd['label']} ({bh.get('max_dd',0):.1%}→{best_dd['max_dd']:.1%})")
best_calmar = max(ranked, key=lambda x: x["calmar"])
print(f"最佳Calmar: {best_calmar['label']} ({best_calmar['calmar']:.2f})")
