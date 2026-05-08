"""A股个股正常回撤幅度统计 — 回答止损线该设多少"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, akshare as ak

print("获取沪深300成分股...", flush=True)
stocks = ak.index_stock_cons_csindex(symbol="000300")
tickers = stocks["成分券代码"].tolist()[:50]

results = []
for i, code in enumerate(tickers):
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20200101", end_date="20260430")
        if len(df) < 500: continue
        close = df["收盘"].values
        high = df["最高"].values
        low = df["最低"].values

        # 单日日内回撤
        intraday_dd = (low - high) / high
        max_intraday = np.min(intraday_dd)

        # N日滚动最大回撤
        for window in [5, 10, 20, 60]:
            cummax = pd.Series(close).rolling(window).max()
            dd = (pd.Series(close) - cummax) / cummax
            results.append({
                "code": code, "window": window,
                "max_dd": dd.min(),
            })
        # 全样本最大回撤
        cummax_all = pd.Series(close).cummax()
        dd_all = (pd.Series(close) - cummax_all) / cummax_all
        results.append({
            "code": code, "window": 9999,
            "max_dd": dd_all.min(),
            "max_intraday": max_intraday,
        })
    except Exception as e:
        continue
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(tickers)}", flush=True)

rdf = pd.DataFrame(results)
valid_codes = rdf.drop_duplicates("code")
print(f"\n共{len(valid_codes)}只股票, {len(rdf)}条记录\n")

# 1. 滚动窗口回撤分布
for window in [5, 10, 20, 60]:
    subset = rdf[rdf.window == window]
    print(f"=== {window}日滚动最大回撤 ===")
    for q in [0.25, 0.50, 0.75, 0.90, 0.95]:
        print(f"  {int(q*100):>3}分位: {subset['max_dd'].quantile(q):.1%}")
    for thresh in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        pct = (subset["max_dd"] < -thresh).mean()
        print(f"  回撤 > {thresh:.0%}: 占比{pct:.1%}")
    print()

# 2. 止损触发概率
print("=" * 50)
print("止损触发概率 (随机买入后持有N天)")
print("=" * 50)
for window in [20, 60]:
    subset = rdf[rdf.window == window]
    print(f"\n持有{window}天:")
    for sl in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
        hit = (subset["max_dd"] < -sl).mean()
        survive = 1 - hit
        print(f"  止损线{sl:>5.0%}: {hit:>5.1%}被止损, {survive:>5.1%}活下来")

# 3. 全样本最大回撤
print(f"\n{'=' * 50}")
print("全样本(2020-2026)最大回撤")
print("=" * 50)
full = rdf[rdf.window == 9999]
for q in [0.25, 0.50, 0.75, 0.90, 0.95]:
    print(f"  {int(q*100):>3}分位: {full['max_dd'].quantile(q):.1%}")
print(f"  均值:     {full['max_dd'].mean():.1%}")
print(f"  中位数:   {full['max_dd'].median():.1%}")

# 4. 单日极端
print(f"\n单日最大日内回撤: 均值{valid_codes['max_intraday'].mean():.1%}, 极值{valid_codes['max_intraday'].min():.1%}")
