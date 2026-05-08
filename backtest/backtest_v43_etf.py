"""
投资体系v4.3 回测 — ETF实战版(长历史)
=======================================
4只同类最早ETF:
  红利     → 510880 红利ETF华泰柏瑞 (2007-01-18)
  沪深300  → 510300 沪深300ETF华泰柏瑞 (2012-05-28)
  国证2000 → 159907 国证2000ETF广发 (2011-08-10)
  进攻资产 → 159908 创业板ETF博时 (2011-07-13)
  类现金   → 货币基金年化2%

回测区间: 2012-05-28 ~ 2026-04-30 (14年)
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, akshare as ak

print("=" * 80)
print("投资体系v4.3 回测 — ETF实战版(长历史)")
print("=" * 80)

# ══════════════════════════════════════════════════════════
# 1. 数据获取
# ══════════════════════════════════════════════════════════
ETF_MAP = {
    "红利": "510880",
    "沪深300": "510300",
    "国证2000": "159907",
    "进攻": "159908",
}

print("  获取ETF日K...", flush=True)
all_price = {}
for name, code in ETF_MAP.items():
    print(f"    {name}({code})...", end=" ", flush=True)
    df = ak.fund_etf_hist_em(symbol=code, period="daily",
                              start_date="20050101", end_date="20260505", adjust="qfq")
    df = df[["日期", "收盘"]].rename(columns={"日期": "date", "收盘": name})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    all_price[name] = df
    print(f"OK ({df.date.min().date()} ~ {df.date.max().date()}, {len(df)}天)")

# 融资数据
print("  融资数据...", end=" ", flush=True)
df_msh = ak.macro_china_market_margin_sh()[["日期", "融资买入额"]].rename(
    columns={"日期": "date", "融资买入额": "rzye"})
df_msz = ak.macro_china_market_margin_sz()[["日期", "融资买入额"]].rename(
    columns={"日期": "date", "融资买入额": "rzye"})
df_msh["date"] = pd.to_datetime(df_msh["date"])
df_msz["date"] = pd.to_datetime(df_msz["date"])
df_margin = df_msh.merge(df_msz, on="date", how="outer", suffixes=("_sh", "_sz"))
df_margin = df_margin.sort_values("date").reset_index(drop=True)
df_margin["rzye_total"] = pd.to_numeric(df_margin["rzye_sh"], errors="coerce") + \
                           pd.to_numeric(df_margin["rzye_sz"], errors="coerce")
print("OK")

# 申万风格指数
print("  申万风格...", end=" ", flush=True)
sw_dp = ak.index_hist_sw(symbol="801811", period="day")[["日期", "收盘"]].rename(
    columns={"日期": "date", "收盘": "sw_large"})
sw_xp = ak.index_hist_sw(symbol="801813", period="day")[["日期", "收盘"]].rename(
    columns={"日期": "date", "收盘": "sw_small"})
sw_dp["date"] = pd.to_datetime(sw_dp["date"])
sw_xp["date"] = pd.to_datetime(sw_xp["date"])
print("OK")

# 沪深300指数(择时用)
print("  沪深300指数...", end=" ", flush=True)
df_hs300 = ak.stock_zh_index_daily(symbol="sh000300")[["date", "close"]].rename(
    columns={"close": "hs300"})
df_hs300["date"] = pd.to_datetime(df_hs300["date"])
df_hs300 = df_hs300.sort_values("date").reset_index(drop=True)
print("OK")

# ══════════════════════════════════════════════════════════
# 2. 合并
# ══════════════════════════════════════════════════════════
print("\n  合并数据...", flush=True)
base = all_price["沪深300"].copy()  # 510300起始最晚(2012-05-28)
for name in ["红利", "国证2000", "进攻"]:
    base = base.merge(all_price[name][["date", name]], on="date", how="left")
base = base.merge(df_hs300[["date", "hs300"]], on="date", how="left")
base = base.merge(df_margin[["date", "rzye_total"]], on="date", how="left")
base = base.merge(sw_dp[["date", "sw_large"]], on="date", how="left")
base = base.merge(sw_xp[["date", "sw_small"]], on="date", how="left")

for c in ["红利", "国证2000", "进攻", "hs300", "rzye_total", "sw_large", "sw_small"]:
    base[c] = base[c].ffill()

df_full = base[(base.date >= "2007-01-01") & (base.date <= "2026-04-30")].reset_index(drop=True)
print(f"  全量数据: {len(df_full)}天")

# ══════════════════════════════════════════════════════════
# 3. 指标计算(全量)
# ══════════════════════════════════════════════════════════
print("  计算指标...", flush=True)
W = 20
STYLE_W = 60
COST = 0.0005

# v3.0择时
df_full["bb_z"] = (df_full["hs300"] - df_full["hs300"].rolling(W).mean()) / df_full["hs300"].rolling(W).std()
df_full["rz_z"] = (df_full["rzye_total"] - df_full["rzye_total"].rolling(W).mean()) / df_full["rzye_total"].rolling(W).std()

def z2p(z, lo=0.20, hi=0.80):
    z = float(np.clip(z, -3, 3))
    return 0.50 + z / 3 * (hi - 0.50)

df_full["pos_price"] = df_full["bb_z"].apply(lambda v: z2p(v) * 0.70 if not pd.isna(v) else 0.35)
df_full["pos_margin"] = df_full["rz_z"].apply(lambda v: z2p(v) * 0.30 if not pd.isna(v) else 0.15)
df_full["pos_total"] = df_full["pos_price"] + df_full["pos_margin"]

# v2风格
df_full["ratio_xd"] = df_full["sw_small"] / df_full["sw_large"]
df_full["pct_xd"] = df_full["ratio_xd"].rolling(STYLE_W).rank(pct=True)

def calc_split(pct):
    if pd.isna(pct): return 0.50
    if pct > 0.80: return 0.10
    if pct < 0.20: return 0.90
    return 0.50

df_full["split"] = df_full["pct_xd"].apply(calc_split)

# 三层映射
CFG = {
    "红利":    {"防御": 0.20, "均衡": 0.15, "进攻": 0.10, "lo": 0.05, "hi": 0.25},
    "宽基合计": {"防御": 0.10, "均衡": 0.15, "进攻": 0.20, "lo": 0.02, "hi": 0.25},
    "进攻":    {"防御": 0.05, "均衡": 0.25, "进攻": 0.45, "lo": 0.05, "hi": 0.45},
}

def calc_weights(pos_total, split):
    t = float(np.clip((pos_total - 0.40) / 0.25, 0, 1))
    w_div_base = CFG["红利"]["防御"] + t * (CFG["红利"]["进攻"] - CFG["红利"]["防御"])
    w_brd_base = CFG["宽基合计"]["防御"] + t * (CFG["宽基合计"]["进攻"] - CFG["宽基合计"]["防御"])
    w_atk_base = CFG["进攻"]["防御"] + t * (CFG["进攻"]["进攻"] - CFG["进攻"]["防御"])
    style_factor = (split - 0.50) / 0.40
    w_div = w_div_base + 0.03 * style_factor
    w_atk = w_atk_base - 0.03 * style_factor
    w_300 = w_brd_base * split
    w_2000 = w_brd_base * (1 - split)
    w_cash = 1 - w_300 - w_2000 - w_div - w_atk
    w_div = float(np.clip(w_div, CFG["红利"]["lo"], CFG["红利"]["hi"]))
    w_atk = float(np.clip(w_atk, CFG["进攻"]["lo"], CFG["进攻"]["hi"]))
    w_cash = 1 - w_300 - w_2000 - w_div - w_atk
    if w_cash < 0.20:
        deficit = 0.20 - w_cash
        w_atk = max(w_atk - deficit, CFG["进攻"]["lo"])
        w_cash = 1 - w_300 - w_2000 - w_div - w_atk
    return w_div, w_300, w_2000, w_atk, w_cash

weights = df_full.apply(lambda r: calc_weights(r["pos_total"], r["split"]), axis=1)
df_full["w_div"] = [w[0] for w in weights]
df_full["w_300"] = [w[1] for w in weights]
df_full["w_2000"] = [w[2] for w in weights]
df_full["w_atk"] = [w[3] for w in weights]
df_full["w_cash"] = [w[4] for w in weights]

# ══════════════════════════════════════════════════════════
# 4. 截取回测区间(需要260天预热后)
# ══════════════════════════════════════════════════════════
# 融资数据2010年起，申万风格2000年起，预热260天后安全
START = "2012-05-28"
END = "2026-04-30"
df = df_full[df_full.date >= START].reset_index(drop=True)
print(f"  回测区间: {df.date.min().date()} ~ {df.date.max().date()}, {len(df)}天\n")

# ══════════════════════════════════════════════════════════
# 5. 回测引擎
# ══════════════════════════════════════════════════════════
RF_DAILY = 1.02 ** (1 / 252) - 1

for name in ["红利", "沪深300", "国证2000", "进攻"]:
    df[f"ret_{name}"] = df[name].pct_change().fillna(0)

df["strat_ret"] = (df["w_div"] * df["ret_红利"] +
                   df["w_300"] * df["ret_沪深300"] +
                   df["w_2000"] * df["ret_国证2000"] +
                   df["w_atk"] * df["ret_进攻"] +
                   df["w_cash"] * RF_DAILY)

# 交易成本
for col in ["w_div", "w_300", "w_2000", "w_atk", "w_cash"]:
    df[f"{col}_prev"] = df[col].shift(1).fillna(df[col].iloc[0])
df["turnover"] = (abs(df["w_div"] - df["w_div_prev"]) +
                  abs(df["w_300"] - df["w_300_prev"]) +
                  abs(df["w_2000"] - df["w_2000_prev"]) +
                  abs(df["w_atk"] - df["w_atk_prev"]))
df["cost"] = df["turnover"] * COST
df["strat_ret_net"] = df["strat_ret"] - df["cost"]

# NAV
df["strat_ret_nc"] = df["strat_ret"]  # 无成本版本
df["nav_strat"] = np.cumprod(1 + df["strat_ret_net"])

# 基准
df["ret_4eq"] = 0.25 * (df["ret_红利"] + df["ret_沪深300"] + df["ret_国证2000"] + df["ret_进攻"])
df["nav_4eq"] = np.cumprod(1 + df["ret_4eq"])
df["nav_300"] = np.cumprod(1 + df["ret_沪深300"])
df["nav_cyb"] = np.cumprod(1 + df["ret_进攻"])
df["nav_5050"] = np.cumprod(1 + 0.5 * df["ret_沪深300"] + 0.5 * RF_DAILY)
df["nav_timing"] = np.cumprod(1 + df["pos_total"] * df["ret_沪深300"] + (1 - df["pos_total"]) * RF_DAILY)

# ══════════════════════════════════════════════════════════
# 6. 整体统计
# ══════════════════════════════════════════════════════════
def stats(ret_s, name):
    nav = np.cumprod(1 + ret_s.values)
    yrs = len(nav) / 252
    ann = (nav[-1] / nav[0]) ** (1 / yrs) - 1 if yrs > 0 and nav[-1] > 0 else -1.0
    sharpe = np.sqrt(252) * np.mean(ret_s - RF_DAILY) / np.std(ret_s) if np.std(ret_s) > 0 else 0
    dd = (pd.Series(nav) / pd.Series(nav).cummax() - 1).min()
    calmar = ann / abs(dd) if dd != 0 else 0
    return {"name": name, "ann": ann, "sharpe": sharpe, "max_dd": dd, "calmar": calmar, "nav": nav[-1]}

print("=" * 80)
print(f"整体表现 ({df.date.min().date()} ~ {df.date.max().date()}, {len(df)/252:.1f}年)")
print("=" * 80)

benchmarks = [
    stats(df["strat_ret_net"], "v4.3策略(含成本)"),
    stats(df["strat_ret_nc"], "v4.3策略(无成本)"),
    stats(df["pos_total"] * df["ret_沪深300"] + (1 - df["pos_total"]) * RF_DAILY, "纯择时(全300)"),
    stats(df["ret_4eq"], "等权4ETF"),
    stats(df["ret_沪深300"], "满仓沪深300"),
    stats(df["ret_进攻"], "满仓创业板"),
    stats(0.5 * df["ret_沪深300"] + 0.5 * RF_DAILY, "50%300+50%现金"),
]

print(f"\n{'策略':<20} {'年化':>7} {'夏普':>7} {'回撤':>8} {'Calmar':>7} {'终值':>8}")
print("-" * 72)
for s in benchmarks:
    print(f"{s['name']:<20} {s['ann']*100:>6.1f}% {s['sharpe']:>7.2f} {s['max_dd']*100:>7.1f}% "
          f"{s['calmar']:>7.2f} {s['nav']:>8.3f}")

# ══════════════════════════════════════════════════════════
# 7. 分年度
# ══════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("分年度表现")
print("=" * 80)

df["year"] = df["date"].dt.year

for strat_name, strat_ret_col in [("v4.3策略", "strat_ret_net"), ("等权4ETF", "ret_4eq"), ("满仓沪深300", "ret_沪深300"), ("满仓创业板", "ret_进攻")]:
    print(f"\n--- {strat_name} ---")
    print(f"  {'年份':>6} {'收益':>10} {'沪深300':>10} {'超额':>8}")
    for y in sorted(df["year"].unique()):
        m = df["year"] == y
        n = m.sum()
        if n < 100: continue
        sa = (np.prod(1 + df.loc[m, strat_ret_col])) ** (252 / n) - 1
        ba = (np.prod(1 + df.loc[m, "ret_沪深300"])) ** (252 / n) - 1
        marker = "✓" if sa > ba else "✗"
        print(f"  {y:>6} {sa * 100:>9.1f}% {ba * 100:>9.1f}% {(sa - ba) * 100:>+7.1f}pp {marker}")

# ══════════════════════════════════════════════════════════
# 8. 持仓分析
# ══════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("持仓分析")
print("=" * 80)

print(f"\n  平均权重:")
for col, name in [("w_div", "红利"), ("w_300", "沪深300"), ("w_2000", "国证2000"), ("w_atk", "进攻"), ("w_cash", "类现金")]:
    print(f"    {name:<8} {df[col].mean() * 100:>6.1f}%  (范围: {df[col].min() * 100:.1f}% ~ {df[col].max() * 100:.1f}%)")

print(f"\n  市场状态分布:")
df["market_state"] = pd.cut(df["pos_total"], bins=[-0.1, 0.40, 0.65, 1.0], labels=["防御", "均衡", "进攻"])
for state in ["防御", "均衡", "进攻"]:
    cnt = (df["market_state"] == state).sum()
    print(f"    {state:<6} {cnt / len(df) * 100:>5.1f}% ({cnt}天)")

print(f"\n  交易成本:")
print(f"    年化换手率: {df['turnover'].mean() * 252 * 100:.1f}%")
print(f"    年化成本:   {df['cost'].mean() * 252 * 100:.3f}%")

# ══════════════════════════════════════════════════════════
# 9. 危机期
# ══════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("危机期表现")
print("=" * 80)

crises = [
    ("2015牛熊", "2015-06-12", "2015-09-30"),
    ("2016熔断", "2016-01-04", "2016-02-29"),
    ("2018贸易战", "2018-01-26", "2018-10-19"),
    ("2020疫情", "2020-01-14", "2020-03-23"),
    ("2022熊市(1-4月)", "2022-01-04", "2022-04-29"),
    ("2022.10回调", "2022-07-01", "2022-10-31"),
    ("2023回调", "2023-05-08", "2023-12-29"),
    ("2024初暴跌", "2024-01-02", "2024-02-05"),
    ("2024.9政策牛", "2024-09-18", "2024-10-08"),
]

for name, s, e in crises:
    m = (df.date >= s) & (df.date <= e)
    if m.sum() < 10: continue
    sr = np.prod(1 + df.loc[m, "strat_ret_net"]) - 1
    br = np.prod(1 + df.loc[m, "ret_沪深300"]) - 1
    marker = "✓" if sr > br else "✗"
    print(f"  {name:<20} 策略:{sr * 100:>+6.1f}%  300:{br * 100:>+6.1f}%  Δ:{(sr - br) * 100:>+5.1f}pp {marker}")

# ══════════════════════════════════════════════════════════
# 10. 滚动夏普(3年窗)
# ══════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("滚动3年夏普比率(策略 vs 沪深300)")
print("=" * 80)

WINDOW = 3 * 252
print(f"\n  {'起':>12}  {'止':>12}  {'策略夏普':>8}  {'300夏普':>8}  {'Δ':>6}")
for i in range(0, len(df) - WINDOW, 252):
    m = df.iloc[i:i + WINDOW]
    r_s = m["strat_ret_net"]
    r_b = m["ret_沪深300"]
    sh_s = np.sqrt(252) * (r_s - RF_DAILY).mean() / r_s.std() if r_s.std() > 0 else 0
    sh_b = np.sqrt(252) * (r_b - RF_DAILY).mean() / r_b.std() if r_b.std() > 0 else 0
    print(f"  {m.date.iloc[0].date()}  {m.date.iloc[-1].date()}  {sh_s:>8.2f}  {sh_b:>8.2f}  {sh_s - sh_b:>+6.2f}")

# ══════════════════════════════════════════════════════════
# 11. Walk-forward验证
# ══════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("Walk-forward验证(4年训练 / 2年测试)")
print("=" * 80)

wf_results = []
total_days = len(df)
train_len = 4 * 252
test_len = 2 * 252

for train_end_idx in range(train_len + 260, total_days - test_len, test_len):
    test_start = df.iloc[train_end_idx]["date"]
    test_end_idx = min(train_end_idx + test_len, total_days)
    test_end = df.iloc[test_end_idx - 1]["date"]

    m = (df.date >= test_start) & (df.date < test_end)
    if m.sum() < 200: continue

    sr = np.prod(1 + df.loc[m, "strat_ret_net"]) - 1
    br = np.prod(1 + df.loc[m, "ret_沪深300"]) - 1
    ann_sr = (1 + sr) ** (252 / m.sum()) - 1
    ann_br = (1 + br) ** (252 / m.sum()) - 1
    sh_s = np.sqrt(252) * (df.loc[m, "strat_ret_net"] - RF_DAILY).mean() / df.loc[m, "strat_ret_net"].std()

    wf_results.append({
        "period": f"{test_start.date()}~{test_end.date()}",
        "sr": sr, "br": br, "ann_sr": ann_sr, "ann_br": ann_br, "sharpe": sh_s
    })

print(f"\n  {'测试期':<25} {'策略':>8} {'300':>8} {'超额':>8} {'夏普':>6}")
print("  " + "-" * 60)
win_count = 0
for r in wf_results:
    marker = "✓" if r["sr"] > r["br"] else "✗"
    if r["sr"] > r["br"]: win_count += 1
    print(f"  {r['period']:<25} {r['sr']*100:>+7.1f}% {r['br']*100:>+7.1f}% {(r['sr']-r['br'])*100:>+7.1f}pp {r['sharpe']:>6.2f} {marker}")

print(f"\n  胜率: {win_count}/{len(wf_results)} ({win_count/len(wf_results)*100:.0f}%)")

# ══════════════════════════════════════════════════════════
# 12. 200万实盘模拟
# ══════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("200万实盘模拟(期末资产)")
print("=" * 80)

nav_final = df["nav_strat"].iloc[-1]
nav_300_final = df["nav_300"].iloc[-1]
nav_cyb_final = df["nav_cyb"].iloc[-1]
initial = 2_000_000
print(f"\n  初始资金: {initial:,.0f}")
print(f"  v4.3策略:   {initial * nav_final:>12,.0f}  (年化{stats(df['strat_ret_net'],'')['ann']*100:.1f}%)")
print(f"  满仓沪深300: {initial * nav_300_final:>12,.0f}  (年化{stats(df['ret_沪深300'],'')['ann']*100:.1f}%)")
print(f"  满仓创业板:  {initial * nav_cyb_final:>12,.0f}  (年化{stats(df['ret_进攻'],'')['ann']*100:.1f}%)")
print(f"  策略多赚(vs 300): {initial * nav_final - initial * nav_300_final:>12,.0f}")
print(f"  策略多赚(vs 创业板): {initial * nav_final - initial * nav_cyb_final:>12,.0f}")

print("\n完成.")
