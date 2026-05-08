"""
风格轮动 × v3.0择时 回测
========================
架构:
  总仓位 = v3.0 (价格BB 70% + 融资BB 30%) ∈ [20%, 80%]
  风格分配 = 风格指标决定 沪深300 vs 国证2000 比例
  实际: 沪深300权重 = 总仓位 × split, 国证2000权重 = 总仓位 × (1-split)

风格指标(申万, 5组):
  1. 小盘/大盘 (801813/801811)
  2. 高PB/低PB (801821/801823)
  3. 高价/低价 (801831/801833)
  4. 高PE/低PE (801841/801843)
  5. 绩优股 单独看 (801231, 亏损股不可用)

风格信号生成方式(与中信建投一致):
  - ratio = 风格A指数 / 风格B指数
  - 20日平滑 → z-score → 趋势跟随(ratio↑ → 做多对应风格)
  - z-score映射到 [0, 1] 的 split
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, akshare as ak

print("=" * 70)
print("风格轮动 × v3.0择时 回测")
print("=" * 70)

# ══════════════════════════════════════════════
# 1. 数据获取
# ══════════════════════════════════════════════

# 市场指数
print("  市场指数...", end=" ", flush=True)
idx_map = {"沪深300": "sh000300", "中证800": "sh000906", "国证2000": "sz399303"}
all_price = {}
for name, code in idx_map.items():
    df = ak.stock_zh_index_daily(symbol=code)[["date","close"]].rename(columns={"close": name})
    df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date").reset_index(drop=True)
    all_price[name] = df
print("OK")

# 申万风格指数
print("  申万风格指数...", end=" ", flush=True)
sw_codes = {
    "绩优": "801231", "大盘": "801811", "小盘": "801813",
    "高PB": "801821", "低PB": "801823",
    "高价": "801831", "低价": "801833",
    "高PE": "801841", "低PE": "801843",
}
sw_data = {}
for name, code in sw_codes.items():
    df = ak.index_hist_sw(symbol=code, period="day")[["日期","收盘"]].rename(columns={"日期":"date","收盘":name})
    df["date"] = pd.to_datetime(df["date"])
    sw_data[name] = df
print("OK")

# 融资数据
print("  融资数据...", end=" ", flush=True)
df_msh = ak.macro_china_market_margin_sh()[["日期","融资买入额"]].rename(columns={"日期":"date","融资买入额":"rzye"})
df_msz = ak.macro_china_market_margin_sz()[["日期","融资买入额"]].rename(columns={"日期":"date","融资买入额":"rzye"})
df_msh["date"] = pd.to_datetime(df_msh["date"])
df_msz["date"] = pd.to_datetime(df_msz["date"])
df_margin = df_msh.merge(df_msz, on="date", how="outer", suffixes=("_sh","_sz"))
df_margin = df_margin.sort_values("date").reset_index(drop=True)
df_margin["rzye_total"] = pd.to_numeric(df_margin["rzye_sh"], errors="coerce") + pd.to_numeric(df_margin["rzye_sz"], errors="coerce")
print("OK")

# ══════════════════════════════════════════════
# 2. 合并
# ══════════════════════════════════════════════
print("  合并数据...", end=" ", flush=True)
base = all_price["沪深300"].copy()
for name in ["中证800", "国证2000"]:
    base = base.merge(all_price[name][["date", name]], on="date", how="left")
for name in sw_data:
    base = base.merge(sw_data[name][["date", name]], on="date", how="left")
base = base.merge(df_margin[["date","rzye_total"]], on="date", how="left")

# ffill风格缺失值
for c in list(sw_data.keys()) + ["中证800","国证2000","rzye_total"]:
    base[c] = base[c].ffill()

base = base[(base.date >= "2015-01-01") & (base.date <= "2026-04-30")].reset_index(drop=True)
df = base.copy()
print(f"OK, {len(df)}天")

# ══════════════════════════════════════════════
# 3. 指标计算
# ══════════════════════════════════════════════
print("  计算指标...", flush=True)
W = 20

# v3.0择时指标
df["bb_ma"] = df["沪深300"].rolling(W).mean()
df["bb_std"] = df["沪深300"].rolling(W).std()
df["bb_z"] = (df["沪深300"] - df["bb_ma"]) / df["bb_std"]

df["rzye_ma"] = df["rzye_total"].rolling(W).mean()
df["rzye_std"] = df["rzye_total"].rolling(W).std()
df["rzye_z"] = (df["rzye_total"] - df["rzye_ma"]) / df["rzye_std"]

# 风格比值(归一化: 第一天=100)
# ratio↑ = 风格A相对走强
style_pairs = [
    ("小盘/大盘", "小盘", "大盘"),   # ratio↑ → 小盘强 → 做多国证2000
    ("高PB/低PB", "高PB", "低PB"),  # ratio↑ → 成长风格 → 偏小盘
    ("高价/低价", "高价", "低价"),  # ratio↑ → 高价股强 → 偏大盘(但信号弱)
    ("高PE/低PE", "高PE", "低PE"),  # ratio↑ → 投机偏好 → 偏小盘
]

for label, col_a, col_b in style_pairs:
    df[f"{label}_raw"] = df[col_a] / df[col_b]

# 风格比值 z-score (趋势跟随: z↑ → split↑ → 做多国证2000)
for label, _, _ in style_pairs:
    ratio = df[f"{label}_raw"]
    ma = ratio.rolling(W).mean()
    std = ratio.rolling(W).std()
    df[f"{label}_z"] = (ratio - ma) / std

# 预热期
df = df.iloc[50:].reset_index(drop=True)
print(f"  预热后: {len(df)}天")

# ══════════════════════════════════════════════
# 4. 仓位生成
# ══════════════════════════════════════════════
rf_d = 1.02 ** (1/252) - 1

def z2p(z, lo=0.20, hi=0.80):
    """z-score → 仓位"""
    z = np.clip(z, -3, 3)
    return 0.50 + z / 3 * (hi - 0.50)

def z2split(z):
    """z-score → 风格split: z∈[-3,3] → [0.10, 0.90], 0.5=均衡"""
    z = np.clip(z, -3, 3)
    return 0.50 + z / 3 * 0.40  # range [0.10, 0.90]

# v3.0总仓位
df["v30_pos"] = df["bb_z"].apply(lambda v: z2p(v)*0.70 if not pd.isna(v) else 0.50*0.70) + \
                df["rzye_z"].apply(lambda v: z2p(v)*0.30 if not pd.isna(v) else 0.50*0.30)

# 五个风格split
for label, _, _ in style_pairs:
    df[f"split_{label}"] = df[f"{label}_z"].apply(lambda v: z2split(v) if not pd.isna(v) else 0.50)

# ══════════════════════════════════════════════
# 5. 回测引擎
# ══════════════════════════════════════════════
def bt_style(price_col_300, price_col_2000, pos_arr, split_arr, label=""):
    """风格轮动回测: 总仓位 × split分配到两个指数"""
    ret_300 = df[price_col_300].pct_change().fillna(0).values
    ret_2000 = df[price_col_2000].pct_change().fillna(0).values
    pos = np.nan_to_num(pos_arr, nan=0.50)
    split = np.nan_to_num(split_arr, nan=0.50)
    
    # 组合收益 = split * pos * ret_300 + (1-split) * pos * ret_2000 + (1-pos) * rf
    strat_ret = split * pos * ret_300 + (1-split) * pos * ret_2000 + (1-pos) * rf_d
    
    # 基准: 纯v3.0全配沪深300
    bench_ret = pos * ret_300 + (1-pos) * rf_d
    
    # 满仓基准
    full_300 = ret_300
    full_2000 = ret_2000
    full_5050 = 0.5 * ret_300 + 0.5 * ret_2000
    
    def nav_stats(ret_arr, name):
        nav = np.cumprod(1+ret_arr)
        yrs = len(nav)/252
        r = pd.Series(ret_arr)
        dd = (pd.Series(nav)/pd.Series(nav).cummax()-1).min()
        ann = (nav[-1]/nav[0])**(1/yrs)-1 if yrs > 0 else 0
        sharpe = np.sqrt(252)*(r - rf_d).mean()/r.std() if r.std() > 0 else 0
        return {"nav": nav, "ann": ann, "sharpe": sharpe, "max_dd": dd}
    
    s = nav_stats(strat_ret, "strat")
    b = nav_stats(bench_ret, "bench")
    f300 = nav_stats(full_300, "full300")
    f2000 = nav_stats(full_2000, "full2000")
    f5050 = nav_stats(full_5050, "full5050")
    
    return {
        "label": label,
        "strat_ann": s["ann"], "strat_sharpe": s["sharpe"], "strat_dd": s["max_dd"],
        "bench_ann": b["ann"], "bench_sharpe": b["sharpe"], "bench_dd": b["max_dd"],
        "full300_ann": f300["ann"], "full300_sharpe": f300["sharpe"],
        "full2000_ann": f2000["ann"], "full2000_sharpe": f2000["sharpe"],
        "full5050_ann": f5050["ann"], "full5050_sharpe": f5050["sharpe"],
        "ann_lift": s["ann"] - b["ann"],
        "sharpe_lift": s["sharpe"] - b["sharpe"],
    }

# ══════════════════════════════════════════════
# 6. 运行回测
# ══════════════════════════════════════════════
print("\n" + "=" * 70)
print("回测结果 (2015-2026)")
print("=" * 70)

results = []

# 基准: v3.0纯沪深300 (split固定0.5)
r0 = bt_style("沪深300", "国证2000", df["v30_pos"].values, np.full(len(df), 0.50), "v3.0 纯300")
results.append(r0)

# 基准: v3.0 固定50/50
r0b = bt_style("沪深300", "国证2000", df["v30_pos"].values, np.full(len(df), 0.50), "v3.0 固定5050")
results.append(r0b)

# 五个风格信号
for label, _, _ in style_pairs:
    r = bt_style("沪深300", "国证2000", df["v30_pos"].values, df[f"split_{label}"].values, f"v3.0+{label}")
    results.append(r)

# 五个风格信号的反向 (测试方向)
for label, _, _ in style_pairs:
    r = bt_style("沪深300", "国证2000", df["v30_pos"].values, 1.0 - df[f"split_{label}"].values, f"v3.0+{label}(反向)")
    results.append(r)

# 风格z-score等权合成
style_z_cols = [f"{label}_z" for label, _, _ in style_pairs]
df["style_z_avg"] = df[style_z_cols].mean(axis=1)
df["split_ensemble"] = df["style_z_avg"].apply(lambda v: z2split(v) if not pd.isna(v) else 0.50)
r_ens = bt_style("沪深300", "国证2000", df["v30_pos"].values, df["split_ensemble"].values, "v3.0+风格合成")
results.append(r_ens)
r_ens_inv = bt_style("沪深300", "国证2000", df["v30_pos"].values, 1.0 - df["split_ensemble"].values, "v3.0+风格合成(反向)")
results.append(r_ens_inv)

# 打印
print(f"\n{'策略':<22} {'年化':>7} {'夏普':>7} {'回撤':>7} {'基准年化':>9} {'基准夏普':>9} {'年化Δ':>7} {'夏普Δ':>7}")
print("-" * 90)
for r in results:
    print(f"{r['label']:<22} {r['strat_ann']*100:>6.1f}% {r['strat_sharpe']:>7.2f} {r['strat_dd']*100:>6.1f}% "
          f"{r['bench_ann']*100:>8.1f}% {r['bench_sharpe']:>9.2f} {r['ann_lift']*100:>+6.1f}pp {r['sharpe_lift']:>+7.2f}")

# 满仓基准对照
print(f"\n{'─'*50}")
print(f"满仓沪深300:  年化 {results[0]['full300_ann']*100:.1f}%  夏普 {results[0]['full300_sharpe']:.2f}")
print(f"满仓国证2000: 年化 {results[0]['full2000_ann']*100:.1f}%  夏普 {results[0]['full2000_sharpe']:.2f}")
print(f"满仓50/50:    年化 {results[0]['full5050_ann']*100:.1f}%  夏普 {results[0]['full5050_sharpe']:.2f}")

# ══════════════════════════════════════════════
# 7. 风格指标间相关性分析
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("风格z-score相关性矩阵")
print("=" * 70)
corr = df[style_z_cols].corr()
labels_short = ["小/大", "PB", "价", "PE"]
print(f"{'':>10}", end="")
for lb in labels_short:
    print(f"{lb:>10}", end="")
print()
for i, lb in enumerate(labels_short):
    print(f"{lb:>10}", end="")
    for j in range(len(labels_short)):
        print(f"{corr.iloc[i,j]:>10.3f}", end="")
    print()

# ══════════════════════════════════════════════
# 8. 分年度表现(最优风格 vs 纯300)
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("分年度表现对比 (v3.0纯300 vs v3.0+最优风格)")
print("=" * 70)

# 找最优风格
best = max([r for r in results if "+" in r["label"] and "(反向)" not in r["label"]], key=lambda x: x["sharpe_lift"])
print(f"\n最优风格信号: {best['label']} (夏普提升 {best['sharpe_lift']:+.2f})")

df["year"] = df["date"].dt.year
ret_300 = df["沪深300"].pct_change().fillna(0)
ret_2000 = df["国证2000"].pct_change().fillna(0)
pos = df["v30_pos"].values
if "小盘/大盘" in best["label"]:
    split = df["split_小盘/大盘"].values
elif "高PB" in best["label"]:
    split = df["split_高PB/低PB"].values
elif "高价" in best["label"]:
    split = df["split_高价/低价"].values
elif "高PE" in best["label"]:
    split = df["split_高PE/低PE"].values
elif "合成" in best["label"]:
    split = df["split_ensemble"].values
else:
    split = np.full(len(df), 0.50)

strat_ret_all = split * pos * ret_300 + (1-split) * pos * ret_2000 + (1-pos) * rf_d
bench_ret_all = pos * ret_300 + (1-pos) * rf_d

print(f"\n{'年份':>6} {'策略年化':>10} {'基准年化':>10} {'Δ':>8}")
print("-" * 40)
for year in sorted(df["year"].unique()):
    mask = df["year"] == year
    n_days = mask.sum()
    if n_days < 200:
        continue
    s_ret = strat_ret_all[mask]
    b_ret = bench_ret_all[mask]
    s_ann = (np.prod(1+s_ret))**(252/n_days) - 1
    b_ann = (np.prod(1+b_ret))**(252/n_days) - 1
    print(f"{year:>6} {s_ann*100:>9.1f}% {b_ann*100:>9.1f}% {(s_ann-b_ann)*100:>+7.1f}pp")

print("\n完成.")
