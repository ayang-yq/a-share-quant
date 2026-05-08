"""
邓普顿极度悲观信号 — 5指标回测
================================
指标:
  1. BB指标(沪深300价格z-score) — 已有右侧主力
  2. 融资买入额BB — 已有右侧辅助
  3. 沪深300 PE百分位 — 估值(乐咕乐股)
  4. 成交额萎缩(20日均量比) — 流动性
  5. 北向资金连续净流出 — ❌数据2024-08断更，跳过

测试内容:
  - 各指标独立回测(沪深300)
  - 4指标等权/优化权重合成
  - 与v3.0基线对比
  - 分年度+危机期表现
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, akshare as ak
from itertools import product

# ══════════════════════════════════════════════════════════
# 1. 数据获取
# ══════════════════════════════════════════════════════════
print("=" * 70)
print("邓普顿极度悲观信号 — 回测")
print("=" * 70)

def safe_ak(fn, *args, retries=3, **kwargs):
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if i < retries - 1:
                import time; time.sleep(5)
            else:
                raise

# 1a. 沪深300日K (含成交额)
print("  沪深300日K...", end=" ", flush=True)
df_300 = safe_ak(ak.stock_zh_index_daily, symbol="sh000300")
df_300["date"] = pd.to_datetime(df_300["date"])
df_300 = df_300.sort_values("date").reset_index(drop=True)
print(f"OK ({len(df_300)}天)")

# 1b. 融资买入额(沪+深)
print("  融资数据...", end=" ", flush=True)
df_msh = safe_ak(ak.macro_china_market_margin_sh)[["日期","融资买入额"]].rename(columns={"日期":"date","融资买入额":"rzye_sh"})
df_msh["date"] = pd.to_datetime(df_msh["date"]); df_msh = df_msh.sort_values("date").reset_index(drop=True)
df_msz = safe_ak(ak.macro_china_market_margin_sz)[["日期","融资买入额"]].rename(columns={"日期":"date","融资买入额":"rzye_sz"})
df_msz["date"] = pd.to_datetime(df_msz["date"]); df_msz = df_msz.sort_values("date").reset_index(drop=True)
print("OK")

# 1c. 沪深300 PE (乐咕乐股)
print("  沪深300 PE...", end=" ", flush=True)
df_pe = safe_ak(ak.stock_index_pe_lg, symbol="沪深300")
# columns: 日期, 指数, 等权静态PE, 静态PE, 静态PE中位数, 等权滚动PE, 滚动PE, 滚动PE中位数
# 用滚动PE(TTM)
df_pe = df_pe[["日期","滚动市盈率"]].rename(columns={"日期":"date","滚动市盈率":"pe_ttm"})
df_pe["date"] = pd.to_datetime(df_pe["date"])
df_pe = df_pe.sort_values("date").reset_index(drop=True)
df_pe = df_pe[df_pe["pe_ttm"] > 0].reset_index(drop=True)  # 去负PE
print(f"OK ({len(df_pe)}天)")

# ══════════════════════════════════════════════════════════
# 2. 合并数据
# ══════════════════════════════════════════════════════════
print("\n  合并数据...", flush=True)
df = df_300[["date","close","volume"]].copy()
df = df.merge(df_msh, on="date", how="left")
df = df.merge(df_msz, on="date", how="left")
df = df.merge(df_pe, on="date", how="left")

df["rzye_total"] = df["rzye_sh"] + df["rzye_sz"]
for c in ["rzye_sh","rzye_sz","rzye_total","pe_ttm"]:
    df[c] = df[c].ffill()

# 回测区间: 2015-01 ~ 2026-04 (与v3.0一致)
df = df[(df.date >= "2015-01-01") & (df.date <= "2026-04-30")].reset_index(drop=True)
print(f"  合并完成: {len(df)}天, {df.date.min().date()} ~ {df.date.max().date()}")

# ══════════════════════════════════════════════════════════
# 3. 指标计算
# ══════════════════════════════════════════════════════════
print("  计算指标...", flush=True)

# ── 指标1: 价格BB (z-score) ──
# 方向: 趋势跟随, z>0→加仓
W_BB = 20
df["bb_ma"] = df["close"].rolling(W_BB).mean()
df["bb_std"] = df["close"].rolling(W_BB).std()
df["bb_z"] = (df["close"] - df["bb_ma"]) / df["bb_std"]

# ── 指标2: 融资买入额BB (z-score) ──
# 方向: 趋势跟随, z>0→加仓
W_RZ = 20
df["rz_ma"] = df["rzye_total"].rolling(W_RZ).mean()
df["rz_std"] = df["rzye_total"].rolling(W_RZ).std()
df["rz_z"] = (df["rzye_total"] - df["rz_ma"]) / df["rz_std"]

# ── 指标3: PE百分位 ──
# 方向: 估值反向(便宜多买), pct低→加仓
# 多窗口测试
for W_PE in [500, 750, 1000]:
    col = f"pe_pct_{W_PE}"
    df[col] = df["pe_ttm"].rolling(W_PE, min_periods=200).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

# ── 指标4: 成交额萎缩 ──
# 方向: 成交萎缩→悲观→加仓(邓普顿逆向)
# vol_ratio = 当前20日均量 / 60日均量, <0.7为地量
W_VOL_SHORT, W_VOL_LONG = 20, 60
df["vol_ma_s"] = df["volume"].rolling(W_VOL_SHORT).mean()
df["vol_ma_l"] = df["volume"].rolling(W_VOL_LONG).mean()
df["vol_ratio"] = df["vol_ma_s"] / df["vol_ma_l"]

# 去预热期
df = df.iloc[max(W_BB,W_RZ,W_VOL_LONG)+50:].reset_index(drop=True)
print(f"  预热后: {len(df)}天")

# ══════════════════════════════════════════════════════════
# 4. 回测引擎
# ══════════════════════════════════════════════════════════
rf_d = 1.02 ** (1/252) - 1
COST = 0.001  # 单边0.1%

def z2pos(z, lo=0.20, hi=0.80):
    """z-score → 仓位, z∈[-3,3] → [lo,hi]"""
    z = np.clip(z, -3, 3)
    return 0.50 + z / 3 * (hi - 0.50)

def pct2pos(pct, lo=0.20, hi=0.80):
    """百分位 → 仓位(反向: pct低→加仓)"""
    # pct=0(最便宜) → hi仓位, pct=1(最贵) → lo仓位
    return hi - pct * (hi - lo)

def vol2pos(ratio, lo=0.20, hi=0.80, threshold_low=0.6, threshold_high=1.0):
    """成交额比 → 仓位(反向: 萎缩→加仓)"""
    # ratio<threshold_low → 极度萎缩→加仓
    # ratio>threshold_high → 放量→正常
    normalized = np.clip((ratio - threshold_high) / (threshold_low - threshold_high), 0, 1)
    return lo + normalized * (hi - lo)

def backtest(price_col, pos_series, label=""):
    """基础回测, 零成本"""
    mkt_ret = df[price_col].pct_change().fillna(0).values
    pos = np.nan_to_num(pos_series, nan=0.50)
    strat_ret = pos * mkt_ret + (1-pos) * rf_d
    strat_nav = np.cumprod(1+strat_ret)
    bench_nav = np.cumprod(1+mkt_ret)
    yrs = len(strat_nav) / 252
    if yrs < 1: return None
    r = pd.Series(strat_ret)
    b = pd.Series(mkt_ret)
    strat_dd = (pd.Series(strat_nav) / pd.Series(strat_nav).cummax() - 1).min()
    bench_dd = (pd.Series(bench_nav) / pd.Series(bench_nav).cummax() - 1).min()
    m_nav = pd.Series(strat_nav, index=df["date"].values).resample("ME").last()
    m_ret = m_nav.pct_change().dropna()
    return {
        "label": label,
        "年化": (strat_nav[-1]/strat_nav[0])**(1/yrs)-1,
        "基准年化": (bench_nav[-1]/bench_nav[0])**(1/yrs)-1,
        "超额": ((strat_nav[-1]/strat_nav[0])**(1/yrs)-1) - ((bench_nav[-1]/bench_nav[0])**(1/yrs)-1),
        "夏普": np.sqrt(252)*(r-rf_d).mean()/r.std() if r.std()>0 else 0,
        "基准夏普": np.sqrt(252)*(b-rf_d).mean()/b.std() if b.std()>0 else 0,
        "最大回撤": strat_dd,
        "基准回撤": bench_dd,
        "回撤改善": abs(bench_dd) - abs(strat_dd),
        "月胜率": (m_ret > 0).mean(),
        "Calmar": ((strat_nav[-1]/strat_nav[0])**(1/yrs)-1) / abs(strat_dd) if strat_dd != 0 else 0,
    }

def backtest_yearly(price_col, pos_series):
    """分年度回测"""
    mkt_ret = df[price_col].pct_change().fillna(0).values
    pos = np.nan_to_num(pos_series, nan=0.50)
    strat_ret = pos * mkt_ret + (1-pos) * rf_d
    bench_ret = mkt_ret
    dates = df["date"].values
    yearly = {}
    for year in range(2015, 2027):
        mask = (pd.Series(dates).dt.year == year)
        if mask.sum() < 100: continue
        sr = strat_ret[mask]
        br = bench_ret[mask]
        s_nav = np.cumprod(1+sr)
        b_nav = np.cumprod(1+br)
        yearly[year] = {
            "策略": (s_nav[-1]/s_nav[0]-1)*100,
            "基准": (b_nav[-1]/b_nav[0]-1)*100,
        }
    return yearly

def backtest_crisis(price_col, pos_series):
    """危机期回测"""
    crises = [
        ("2015-06-12","2015-09-30","2015股灾"),
        ("2016-01-01","2016-02-29","2016熔断"),
        ("2018-01-26","2018-12-31","2018贸易战"),
        ("2020-01-20","2020-03-23","2020疫情"),
        ("2022-01-01","2022-12-31","2022熊市"),
        ("2024-01-01","2024-09-23","2024阴跌"),
    ]
    mkt_ret = df[price_col].pct_change().fillna(0).values
    pos = np.nan_to_num(pos_series, nan=0.50)
    strat_ret = pos * mkt_ret + (1-pos) * rf_d
    bench_ret = mkt_ret
    dates = df["date"].values
    results = []
    for s,e,name in crises:
        mask = (pd.Series(dates) >= pd.Timestamp(s)) & (pd.Series(dates) <= pd.Timestamp(e))
        if mask.sum() < 10: continue
        sr = strat_ret[mask]
        br = bench_ret[mask]
        s_nav = np.cumprod(1+sr)
        b_nav = np.cumprod(1+br)
        results.append({
            "危机": name,
            "策略": (s_nav[-1]/s_nav[0]-1)*100,
            "基准": (b_nav[-1]/b_nav[0]-1)*100,
            "少亏": (b_nav[-1]/b_nav[0]-1)*100 - (s_nav[-1]/s_nav[0]-1)*100,
        })
    return results

# ══════════════════════════════════════════════════════════
# 5. 独立回测: 各指标单独表现
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Part 1: 各指标独立回测 (沪深300, 零成本)")
print("=" * 70)

results = []

# 1. 价格BB (趋势跟随)
pos_bb = df["bb_z"].apply(lambda v: z2pos(v) if not pd.isna(v) else 0.50)
r = backtest("close", pos_bb, "价格BB")
results.append(r)

# 2. 融资BB (趋势跟随)
pos_rz = df["rz_z"].apply(lambda v: z2pos(v) if not pd.isna(v) else 0.50)
r = backtest("close", pos_rz, "融资BB")
results.append(r)

# 3. v3.0基线 (BB70% + 融资30%)
pos_v30 = df["bb_z"].apply(lambda v: z2pos(v)*0.70 if not pd.isna(v) else 0.35) + \
          df["rz_z"].apply(lambda v: z2pos(v)*0.30 if not pd.isna(v) else 0.15)
r = backtest("close", pos_v30, "v3.0基线(BB70+融资30)")
results.append(r)

# 4. PE百分位 (反向: 便宜多买)
for W_PE in [500, 750, 1000]:
    col = f"pe_pct_{W_PE}"
    pos_pe = df[col].apply(lambda v: pct2pos(v) if not pd.isna(v) else 0.50)
    r = backtest("close", pos_pe, f"PE百分位反向(W{W_PE})")
    results.append(r)

# 5. 成交额萎缩 (反向: 萎缩→加仓)
for tlo in [0.55, 0.60, 0.65, 0.70]:
    pos_vol = df["vol_ratio"].apply(lambda v: vol2pos(v, threshold_low=tlo) if not pd.isna(v) else 0.50)
    r = backtest("close", pos_vol, f"成交额萎缩(阈值{tlo})")
    results.append(r)

# 6. PE百分位正向 (贵多买 = 趋势跟随)
for W_PE in [500, 750, 1000]:
    col = f"pe_pct_{W_PE}"
    # pct高→贵→加仓(趋势跟随逻辑)
    pos_pe_fwd = df[col].apply(lambda v: z2pos(2*v-1) if not pd.isna(v) else 0.50)
    r = backtest("close", pos_pe_fwd, f"PE百分位正向(W{W_PE})")
    results.append(r)

# 7. 成交额正向 (放量→加仓 = 趋势跟随)
pos_vol_fwd = df["vol_ratio"].apply(lambda v: z2pos((v-0.8)/0.3) if not pd.isna(v) else 0.50)
r = backtest("close", pos_vol_fwd, "成交额正向(趋势)")
results.append(r)

# 打印结果
print(f"\n{'指标':<30} {'年化':>7} {'夏普':>7} {'回撤':>8} {'回撤改善':>8} {'月胜率':>7}")
print("-" * 75)
for r in results:
    print(f"{r['label']:<30} {r['年化']:>6.1%} {r['夏普']:>7.2f} {r['最大回撤']:>8.1%} {r['回撤改善']:>+7.1f}pp {r['月胜率']:>6.1%}")

# ══════════════════════════════════════════════════════════
# 6. 合成回测: v3.0 + 新指标
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Part 2: 合成回测 (v3.0基线 + 新指标)")
print("=" * 70)

# 将新指标也标准化为z-score，方便加权
def to_zscore(series, W=60):
    ma = series.rolling(W).mean()
    std = series.rolling(W).std()
    return (series - ma) / std

# PE百分位 → z-score (反向: pct低→z高→加仓)
# 先反向化: 1-pct, 这样便宜→高值→趋势跟随逻辑
for W_PE in [500, 750, 1000]:
    col = f"pe_pct_{W_PE}"
    df[f"pe_rev_{W_PE}"] = 1 - df[col]  # 反向
    df[f"pe_z_{W_PE}"] = to_zscore(df[f"pe_rev_{W_PE}"], 60)

# 成交额萎缩 → z-score (反向: ratio低→z高→加仓)
df["vol_rev"] = -df["vol_ratio"]  # 反向: 成交萎缩→高值
df["vol_z"] = to_zscore(df["vol_rev"], 60)

composite_results = []

# --- 方案A: v3.0 + PE百分位 ---
print("\n--- A: v3.0 + PE百分位 ---")
for W_PE in [500, 750, 1000]:
    for w_pe in [0.10, 0.15, 0.20]:
        w_bb = 0.70 * (1 - w_pe)
        w_rz = 0.30 * (1 - w_pe)
        pos = df["bb_z"].apply(lambda v: z2pos(v)*w_bb if not pd.isna(v) else 0.50*w_bb) + \
              df["rz_z"].apply(lambda v: z2pos(v)*w_rz if not pd.isna(v) else 0.50*w_rz) + \
              df[f"pe_z_{W_PE}"].apply(lambda v: z2pos(v)*w_pe if not pd.isna(v) else 0.50*w_pe)
        r = backtest("close", pos, f"BB+融资+PE{W_PE}(w{w_pe:.0%})")
        composite_results.append(r)

# --- 方案B: v3.0 + 成交额萎缩 ---
print("--- B: v3.0 + 成交额萎缩 ---")
for tlo in [0.60, 0.65]:
    for w_vol in [0.10, 0.15, 0.20]:
        w_bb = 0.70 * (1 - w_vol)
        w_rz = 0.30 * (1 - w_vol)
        pos = df["bb_z"].apply(lambda v: z2pos(v)*w_bb if not pd.isna(v) else 0.50*w_bb) + \
              df["rz_z"].apply(lambda v: z2pos(v)*w_rz if not pd.isna(v) else 0.50*w_rz) + \
              df["vol_z"].apply(lambda v: z2pos(v)*w_vol if not pd.isna(v) else 0.50*w_vol)
        r = backtest("close", pos, f"BB+融资+成交萎缩(t{tlo},w{w_vol:.0%})")
        composite_results.append(r)

# --- 方案C: v3.0 + PE + 成交额(三因子) ---
print("--- C: v3.0 + PE + 成交额 ---")
for W_PE in [750]:
    for w_pe, w_vol in [(0.10, 0.10), (0.10, 0.15), (0.15, 0.10)]:
        w_bb = (1 - w_pe - w_vol) * 0.70
        w_rz = (1 - w_pe - w_vol) * 0.30
        pos = df["bb_z"].apply(lambda v: z2pos(v)*w_bb if not pd.isna(v) else 0.50*w_bb) + \
              df["rz_z"].apply(lambda v: z2pos(v)*w_rz if not pd.isna(v) else 0.50*w_rz) + \
              df[f"pe_z_{W_PE}"].apply(lambda v: z2pos(v)*w_pe if not pd.isna(v) else 0.50*w_pe) + \
              df["vol_z"].apply(lambda v: z2pos(v)*w_vol if not pd.isna(v) else 0.50*w_vol)
        r = backtest("close", pos, f"BB+融资+PE+成交(w_pe{w_pe:.0%},w_vol{w_vol:.0%})")
        composite_results.append(r)

# 添加v3.0基线便于对比
composite_results.insert(0, results[2])  # v3.0基线

# 按夏普排序
composite_results.sort(key=lambda x: x["夏普"], reverse=True)

print(f"\n{'方案':<42} {'年化':>7} {'夏普':>7} {'回撤':>8} {'vs基线夏普':>10}")
print("-" * 82)
baseline_sharpe = results[2]["夏普"]
for r in composite_results:
    delta = r["夏普"] - baseline_sharpe
    print(f"{r['label']:<42} {r['年化']:>6.1%} {r['夏普']:>7.2f} {r['最大回撤']:>8.1%} {delta:>+10.2f}")

# ══════════════════════════════════════════════════════════
# 7. 最佳方案详细分析
# ══════════════════════════════════════════════════════════
best = composite_results[0]
print(f"\n{'='*70}")
print(f"最佳方案: {best['label']}")
print(f"{'='*70}")

# 重建最佳方案的仓位
if "PE" in best["label"] and "成交" in best["label"]:
    # 三因子
    label = best["label"]
    # 解析权重
    import re
    nums = re.findall(r'w_pe([\d.]+)%.*w_vol([\d.]+)%', label)
    if nums:
        w_pe, w_vol = float(nums[0][0])/100, float(nums[0][1])/100
        w_bb = (1-w_pe-w_vol)*0.70
        w_rz = (1-w_pe-w_vol)*0.30
    else:
        w_pe, w_vol = 0.10, 0.10
        w_bb, w_rz = 0.56, 0.24
    pos_best = df["bb_z"].apply(lambda v: z2pos(v)*w_bb if not pd.isna(v) else 0.50*w_bb) + \
               df["rz_z"].apply(lambda v: z2pos(v)*w_rz if not pd.isna(v) else 0.50*w_rz) + \
               df[f"pe_z_750"].apply(lambda v: z2pos(v)*w_pe if not pd.isna(v) else 0.50*w_pe) + \
               df["vol_z"].apply(lambda v: z2pos(v)*w_vol if not pd.isna(v) else 0.50*w_vol)
elif "PE" in best["label"]:
    nums = re.findall(r'w(\d+)%', best["label"])
    if nums:
        w_pe = float(nums[-1])/100
    else:
        w_pe = 0.15
    w_bb = 0.70*(1-w_pe); w_rz = 0.30*(1-w_pe)
    W_PE_match = re.findall(r'PE(\d+)', best["label"])
    W_PE = int(W_PE_match[0]) if W_PE_match else 750
    pos_best = df["bb_z"].apply(lambda v: z2pos(v)*w_bb if not pd.isna(v) else 0.50*w_bb) + \
               df["rz_z"].apply(lambda v: z2pos(v)*w_rz if not pd.isna(v) else 0.50*w_rz) + \
               df[f"pe_z_{W_PE}"].apply(lambda v: z2pos(v)*w_pe if not pd.isna(v) else 0.50*w_pe)
elif "成交" in best["label"]:
    pos_best = pos_v30  # fallback
else:
    pos_best = pos_v30

# 分年度
yearly = backtest_yearly("close", pos_best)
print(f"\n分年度表现:")
print(f"{'年份':>6}  {'策略':>8}  {'基准':>8}  {'超额':>8}  {'最优?':>5}")
print("-" * 42)
for y, v in yearly.items():
    excess = v["策略"] - v["基准"]
    best_mark = " ★" if v["策略"] >= v["基准"] else ""
    print(f"{y:>6}  {v['策略']:>7.1f}%  {v['基准']:>7.1f}%  {excess:>+7.1f}%{best_mark}")

# 危机期
crises = backtest_crisis("close", pos_best)
print(f"\n危机期表现:")
print(f"{'危机':<12}  {'策略':>8}  {'基准':>8}  {'少亏':>8}")
print("-" * 42)
for c in crises:
    print(f"{c['危机']:<12}  {c['策略']:>7.1f}%  {c['基准']:>7.1f}%  {c['少亏']:>+7.1f}%")

# ══════════════════════════════════════════════════════════
# 8. 极端区域专项分析
# ══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 3: 极端区域触发分析")
print("=" * 70)

# 看各指标在历史大底部的值
bottoms = {
    "2016-01-28": "熔断底",
    "2018-10-18": "贸易战底",
    "2020-03-23": "疫情底",
    "2022-10-31": "熊市底",
    "2024-02-05": "雪球底",
    "2024-09-13": "政策底前",
}
print(f"\n{'日期':<14} {'事件':<10} {'BB_z':>7} {'融资_z':>7} {'PE百分位':>8} {'成交比':>7}")
print("-" * 58)
for d, name in bottoms.items():
    row = df[df.date == d]
    if len(row) == 0:
        # 找最近日期
        nearest = df[df.date <= d].tail(1)
        if len(nearest) == 0: continue
        row = nearest
    r = row.iloc[0]
    bb_z = f"{r['bb_z']:.2f}" if not pd.isna(r['bb_z']) else "N/A"
    rz_z = f"{r['rz_z']:.2f}" if not pd.isna(r['rz_z']) else "N/A"
    pe_pct = f"{r['pe_pct_750']:.1%}" if not pd.isna(r['pe_pct_750']) else "N/A"
    vol_r = f"{r['vol_ratio']:.2f}" if not pd.isna(r['vol_ratio']) else "N/A"
    print(f"{r['date'].strftime('%Y-%m-%d'):<14} {name:<10} {bb_z:>7} {rz_z:>7} {pe_pct:>8} {vol_r:>7}")

# 多指标同时极端的天数统计
print(f"\n多指标同时触发极端的天数:")
for bb_thresh in [-1.5, -2.0]:
    for pe_thresh in [0.10, 0.20]:
        mask_bb = df["bb_z"] < bb_thresh
        mask_pe = df["pe_pct_750"] < pe_thresh
        mask_vol = df["vol_ratio"] < 0.65
        both = mask_bb & mask_pe
        three = mask_bb & mask_pe & mask_vol
        print(f"  BB<{bb_thresh} + PE<{pe_thresh:.0%}: {both.sum()}天")
        print(f"  BB<{bb_thresh} + PE<{pe_thresh:.0%} + 成交<0.65: {three.sum()}天")
        if three.sum() > 0:
            dates = df[three]["date"].dt.strftime("%Y-%m-%d").tolist()
            print(f"    日期: {dates}")

# ══════════════════════════════════════════════════════════
# 9. 样本外验证 (2020-2026)
# ══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 4: 样本外验证 (2020-01 ~ 2026-04)")
print("=" * 70)

oos_results = []
# v3.0基线样本外
mask_oos = df.date >= "2020-01-01"
pos_v30_arr = pos_v30.values
r_v30_oos = backtest("close", np.where(mask_oos, pos_v30_arr, 0.50), "v3.0基线")
oos_results.append(r_v30_oos)

# 各新指标样本外
for W_PE in [500, 750, 1000]:
    pos_pe = df[f"pe_z_{W_PE}"].apply(lambda v: z2pos(v) if not pd.isna(v) else 0.50)
    r = backtest("close", pos_pe, f"PE反向z(W{W_PE})")
    oos_results.append(r)

pos_vol = df["vol_z"].apply(lambda v: z2pos(v) if not pd.isna(v) else 0.50)
r = backtest("close", pos_vol, "成交萎缩z")
oos_results.append(r)

print(f"\n{'指标':<25} {'样本内夏普':>10} {'样本外夏普':>10}")
print("-" * 50)
# 重新跑全样本
full_results = {
    "v3.0基线": backtest("close", pos_v30, ""),
    "PE反向z(W500)": backtest("close", df["pe_z_500"].apply(lambda v: z2pos(v) if not pd.isna(v) else 0.50), ""),
    "PE反向z(W750)": backtest("close", df["pe_z_750"].apply(lambda v: z2pos(v) if not pd.isna(v) else 0.50), ""),
    "PE反向z(W1000)": backtest("close", df["pe_z_1000"].apply(lambda v: z2pos(v) if not pd.isna(v) else 0.50), ""),
    "成交萎缩z": backtest("close", df["vol_z"].apply(lambda v: z2pos(v) if not pd.isna(v) else 0.50), ""),
}
for name, fr in full_results.items():
    oos_r = [r for r in oos_results if r["label"] == name]
    oos_s = oos_r[0]["夏普"] if oos_r else 0
    print(f"{name:<25} {fr['夏普']:>10.2f} {oos_s:>10.2f}")

print(f"\n{'='*70}")
print("回测完成")
print("=" * 70)
