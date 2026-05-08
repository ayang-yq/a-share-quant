"""
风格轮动v2 Walk-Forward 验证
=============================
1. 固定分割: 2015-2020训练, 2021-2026样本外
2. 滚动窗口: 每3年训练 → 2年测试, 逐步前移
3. 对比: 全样本最优参数 vs 样本外表现
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, akshare as ak
from itertools import product
import time

def safe_ak_call(fn, retries=3, delay=5, **kwargs):
    for attempt in range(retries):
        try:
            return fn(**kwargs)
        except Exception as e:
            if attempt < retries - 1:
                print(f"    API重试 {attempt+1}/{retries}: {str(e)[:60]}...", flush=True)
                time.sleep(delay)
            else:
                raise

print("=" * 70)
print("风格轮动v2 — Walk-Forward 验证")
print("=" * 70)

# ══════════════════════════════════════════════
# 1. 数据获取(与v2相同)
# ══════════════════════════════════════════════
print("  数据获取...", flush=True)

idx_map = {"沪深300": "sh000300", "国证2000": "sz399303"}
all_price = {}
for name, code in idx_map.items():
    df = safe_ak_call(ak.stock_zh_index_daily, symbol=code)[["date","close"]].rename(columns={"close": name})
    df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date").reset_index(drop=True)
    all_price[name] = df

sw_codes = {"大盘": "801811", "小盘": "801813"}
sw_data = {}
for name, code in sw_codes.items():
    df = safe_ak_call(ak.index_hist_sw, symbol=code, period="day")[["日期","收盘"]].rename(columns={"日期":"date","收盘":name})
    df["date"] = pd.to_datetime(df["date"])
    sw_data[name] = df

df_msh = safe_ak_call(ak.macro_china_market_margin_sh)[["日期","融资买入额"]].rename(columns={"日期":"date","融资买入额":"rzye"})
df_msz = safe_ak_call(ak.macro_china_market_margin_sz)[["日期","融资买入额"]].rename(columns={"日期":"date","融资买入额":"rzye"})
df_msh["date"] = pd.to_datetime(df_msh["date"])
df_msz["date"] = pd.to_datetime(df_msz["date"])
df_margin = df_msh.merge(df_msz, on="date", how="outer", suffixes=("_sh","_sz"))
df_margin = df_margin.sort_values("date").reset_index(drop=True)
df_margin["rzye_total"] = pd.to_numeric(df_margin["rzye_sh"], errors="coerce") + pd.to_numeric(df_margin["rzye_sz"], errors="coerce")

base = all_price["沪深300"].copy()
base = base.merge(all_price["国证2000"][["date","国证2000"]], on="date", how="left")
for name in sw_data:
    base = base.merge(sw_data[name][["date", name]], on="date", how="left")
base = base.merge(df_margin[["date","rzye_total"]], on="date", how="left")
for c in list(sw_data.keys()) + ["国证2000","rzye_total"]:
    base[c] = base[c].ffill()
# 扩大范围到2013年起，覆盖2014-2015完整周期
base = base[(base.date >= "2013-01-01") & (base.date <= "2026-04-30")].reset_index(drop=True)
df_full = base.copy()
print(f"  合并完成: {len(df_full)}天, {df_full.date.dt.date.min()} ~ {df_full.date.dt.date.max()}")

# ══════════════════════════════════════════════
# 2. 指标计算
# ══════════════════════════════════════════════
print("  计算指标...", flush=True)
W_TIMING = 20
COST = 0.0005
rf_d = 1.02 ** (1/252) - 1

df_full["bb_z"] = (df_full["沪深300"] - df_full["沪深300"].rolling(W_TIMING).mean()) / df_full["沪深300"].rolling(W_TIMING).std()
df_full["rz_z"] = (df_full["rzye_total"] - df_full["rzye_total"].rolling(W_TIMING).mean()) / df_full["rzye_total"].rolling(W_TIMING).std()

def z2p(z, lo=0.20, hi=0.80):
    z = float(np.clip(z, -3, 3))
    return 0.50 + z / 3 * (hi - 0.50)

df_full["v30_pos"] = df_full["bb_z"].apply(lambda v: z2p(v)*0.70 if not pd.isna(v) else 0.35) + \
                     df_full["rz_z"].apply(lambda v: z2p(v)*0.30 if not pd.isna(v) else 0.15)
df_full["ratio_xd"] = df_full["小盘"] / df_full["大盘"]
for win in [60, 120, 250]:
    df_full[f"ratio_xd_p{win}"] = df_full["ratio_xd"].rolling(win).rank(pct=True)

# 预热(260天)
df_full = df_full.iloc[260:].reset_index(drop=True)
print(f"  预热后: {len(df_full)}天")

# ══════════════════════════════════════════════
# 3. 回测引擎
# ══════════════════════════════════════════════
def make_signal(pct_col, lo_thresh=0.10, hi_thresh=0.90, ext_split=0.80, direction=-1):
    """direction=-1: 趋势(小盘强→加仓2000)"""
    splits = np.full(len(df_full), 0.50)
    pcts = df_full[pct_col].values
    if direction == -1:  # 趋势
        splits[pcts > hi_thresh] = 1.0 - ext_split
        splits[pcts < lo_thresh] = ext_split
    return splits

def backtest_slice(df_slice, pos_arr, split_arr, min_split_change=0.10):
    """对某个时间切片做回测，返回统计"""
    n = len(df_slice)
    if n < 50:
        return None
    
    ret_300 = df_slice["沪深300"].pct_change().fillna(0).values
    ret_2000 = df_slice["国证2000"].pct_change().fillna(0).values
    pos = np.nan_to_num(pos_arr[:n], nan=0.50)
    split = np.nan_to_num(split_arr[:n], nan=0.50)
    
    w_300 = pos * split
    w_2000 = pos * (1 - split)
    
    cost = np.zeros(n)
    for i in range(1, n):
        if abs(split[i] - split[i-1]) < min_split_change:
            w_300[i] = pos[i] * split[i-1]
            w_2000[i] = pos[i] * (1 - split[i-1])
        turnover = abs(w_300[i] - w_300[i-1]) + abs(w_2000[i] - w_2000[i-1])
        cost[i] = turnover * COST
    
    strat_ret = w_300 * ret_300 + w_2000 * ret_2000 - cost
    bench_ret = pos * ret_300  # v3.0纯300基准
    
    nav = np.cumprod(1 + strat_ret)
    bench_nav = np.cumprod(1 + bench_ret)
    yrs = n / 252
    
    r = pd.Series(strat_ret)
    dd = (pd.Series(nav) / pd.Series(nav).cummax() - 1).min()
    ann = (nav[-1] / nav[0]) ** (1/yrs) - 1 if yrs > 0 else 0
    sharpe = np.sqrt(252) * (r - rf_d).mean() / r.std() if r.std() > 0 else 0
    
    b = pd.Series(bench_ret)
    bench_ann = (bench_nav[-1] / bench_nav[0]) ** (1/yrs) - 1 if yrs > 0 else 0
    bench_sharpe = np.sqrt(252) * (b - rf_d).mean() / b.std() if b.std() > 0 else 0
    
    # 满仓300
    full300_nav = np.cumprod(1 + ret_300)
    full300_ann = (full300_nav[-1] / full300_nav[0]) ** (1/yrs) - 1 if yrs > 0 else 0
    
    return {
        "ann": ann, "sharpe": sharpe, "max_dd": dd,
        "bench_ann": bench_ann, "bench_sharpe": bench_sharpe,
        "ann_lift": ann - bench_ann, "sharpe_lift": sharpe - bench_sharpe,
        "full300_ann": full300_ann,
    }

# ══════════════════════════════════════════════
# 4. 参数网格
# ══════════════════════════════════════════════
param_grid = []
for win in [60, 120]:
    for lo_t, hi_t in [(0.05, 0.95), (0.10, 0.90), (0.15, 0.85), (0.20, 0.80)]:
        for ext_split in [0.70, 0.80, 0.90]:
            param_grid.append({
                "win": win, "lo": lo_t, "hi": hi_t, "ext": ext_split,
                "pct_col": f"ratio_xd_p{win}",
                "label": f"W{win} [{lo_t:.0%},{hi_t:.0%}] ext{ext_split:.0%}"
            })

print(f"  参数空间: {len(param_grid)}组")

# ══════════════════════════════════════════════
# 5. 固定分割验证
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 1: 固定分割 (训练2015-2020, 样本外2021-2026)")
print("=" * 70)

train_mask = (df_full.date >= "2015-01-01") & (df_full.date <= "2020-12-31")
test_mask = (df_full.date >= "2021-01-01") & (df_full.date <= "2026-04-30")
df_train = df_full[train_mask].reset_index(drop=True)
df_test = df_full[test_mask].reset_index(drop=True)
pos_train = df_full[train_mask]["v30_pos"].values
pos_test = df_full[test_mask]["v30_pos"].values

print(f"  训练集: {len(df_train)}天 ({df_train.date.dt.date.min()} ~ {df_train.date.dt.date.max()})")
print(f"  测试集: {len(df_test)}天 ({df_test.date.dt.date.min()} ~ {df_test.date.dt.date.max()})")

# 训练集选最优
train_results = []
for p in param_grid:
    split = make_signal(p["pct_col"], p["lo"], p["hi"], p["ext"], direction=-1)
    split_train = split[train_mask.values]
    r = backtest_slice(df_train, pos_train, split_train)
    if r:
        train_results.append({**r, **p})

train_ranked = sorted(train_results, key=lambda x: x["sharpe_lift"], reverse=True)

print(f"\n  训练集 Top5:")
print(f"  {'参数':<40} {'夏普':>6} {'夏普Δ':>7} {'年化Δ':>7}")
for r in train_ranked[:5]:
    print(f"  {r['label']:<40} {r['sharpe']:>6.2f} {r['sharpe_lift']:>+7.2f} {r['ann_lift']*100:>+6.1f}pp")

# 用训练集Top1参数在测试集上验证
best = train_ranked[0]
print(f"\n  ★ 训练集最优: {best['label']}")
split_best = make_signal(best["pct_col"], best["lo"], best["hi"], best["ext"], direction=-1)
split_best_test = split_best[test_mask.values]
r_oos = backtest_slice(df_test, pos_test, split_best_test)

# 纯300测试集基准
r_300_test = backtest_slice(df_test, pos_test, np.full(len(df_test), 0.50))

print(f"\n  {'指标':<20} {'风格v2(样本外)':>15} {'v3.0纯300':>12} {'差值':>10}")
print(f"  {'-'*57}")
print(f"  {'年化收益':<18} {r_oos['ann']*100:>14.1f}% {r_300_test['ann']*100:>11.1f}% {(r_oos['ann']-r_300_test['ann'])*100:>+9.1f}pp")
print(f"  {'夏普比率':<18} {r_oos['sharpe']:>15.2f} {r_300_test['sharpe']:>12.2f} {r_oos['sharpe']-r_300_test['sharpe']:>+10.2f}")
print(f"  {'最大回撤':<18} {r_oos['max_dd']*100:>14.1f}% {r_300_test['max_dd']*100:>11.1f}% {(r_oos['max_dd']-r_300_test['max_dd'])*100:>+9.1f}pp")

# 也测全样本最优参数在测试集的表现(作为对照)
print(f"\n  对照: 全样本最优参数(W60 [20%,80%] ext90%)在测试集:")
split_fullbest = make_signal("ratio_xd_p60", 0.20, 0.80, 0.90, direction=-1)
split_fullbest_test = split_fullbest[test_mask.values]
r_fullbest_oos = backtest_slice(df_test, pos_test, split_fullbest_test)
print(f"    年化 {r_fullbest_oos['ann']*100:.1f}%, 夏普 {r_fullbest_oos['sharpe']:.2f}, 回撤 {r_fullbest_oos['max_dd']*100:.1f}%")

# ══════════════════════════════════════════════
# 6. 滚动窗口验证
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 2: 滚动窗口 (3年训练 → 2年测试)")
print("=" * 70)

# 窗口: [2015-2017]→[2018-2019], [2017-2019]→[2020-2021], [2019-2021]→[2022-2023], [2021-2023]→[2024-2026]
windows = [
    ("2015-01-01", "2017-12-31", "2018-01-01", "2019-12-31"),
    ("2017-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
    ("2019-01-01", "2021-12-31", "2022-01-01", "2023-12-31"),
    ("2021-01-01", "2023-12-31", "2024-01-01", "2026-04-30"),
]

rolling_results = []

for tr_s, tr_e, te_s, te_e in windows:
    tr_m = (df_full.date >= tr_s) & (df_full.date <= tr_e)
    te_m = (df_full.date >= te_s) & (df_full.date <= te_e)
    df_tr = df_full[tr_m].reset_index(drop=True)
    df_te = df_full[te_m].reset_index(drop=True)
    pos_tr = df_full[tr_m]["v30_pos"].values
    pos_te = df_full[te_m]["v30_pos"].values
    
    # 训练集选最优
    best_sharpe = -999
    best_param = None
    for p in param_grid:
        split = make_signal(p["pct_col"], p["lo"], p["hi"], p["ext"], direction=-1)
        r = backtest_slice(df_tr, pos_tr, split[tr_m.values])
        if r and r["sharpe_lift"] > best_sharpe:
            best_sharpe = r["sharpe_lift"]
            best_param = p
    
    # 用最优参数在测试集上验证
    split_test = make_signal(best_param["pct_col"], best_param["lo"], best_param["hi"], best_param["ext"], direction=-1)
    r_test = backtest_slice(df_te, pos_te, split_test[te_m.values])
    r_300 = backtest_slice(df_te, pos_te, np.full(len(df_te), 0.50))
    
    rolling_results.append({
        "train": f"{tr_s[:4]}-{tr_e[:4]}", "test": f"{te_s[:4]}-{te_e[:4]}",
        "best_param": best_param["label"],
        "ann": r_test["ann"], "sharpe": r_test["sharpe"], "max_dd": r_test["max_dd"],
        "bench_ann": r_300["ann"], "bench_sharpe": r_300["sharpe"],
        "ann_lift": r_test["ann"] - r_300["ann"],
        "sharpe_lift": r_test["sharpe"] - r_300["sharpe"],
    })

print(f"\n  {'训练期':<16} {'测试期':<16} {'最优参数':<35} {'夏普':>6} {'夏普Δ':>7} {'年化Δ':>7}")
print(f"  {'-'*95}")
for r in rolling_results:
    print(f"  {r['train']:<16} {r['test']:<16} {r['best_param']:<35} {r['sharpe']:>6.2f} {r['sharpe_lift']:>+7.2f} {r['ann_lift']*100:>+6.1f}pp")

# 汇总
avg_sharpe = np.mean([r["sharpe"] for r in rolling_results])
avg_lift = np.mean([r["sharpe_lift"] for r in rolling_results])
avg_ann_lift = np.mean([r["ann_lift"] for r in rolling_results])
win_rate = sum(1 for r in rolling_results if r["sharpe_lift"] > 0) / len(rolling_results) * 100

print(f"\n  滚动窗口汇总:")
print(f"    平均夏普: {avg_sharpe:.2f}")
print(f"    平均夏普提升: {avg_lift:+.2f}")
print(f"    平均年化提升: {avg_ann_lift*100:+.1f}pp")
print(f"    胜率(夏普正提升): {win_rate:.0f}% ({sum(1 for r in rolling_results if r['sharpe_lift'] > 0)}/{len(rolling_results)})")

# ══════════════════════════════════════════════
# 7. 扩展样本: 2014年起(覆盖2014-2015极端行情)
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 3: 扩展样本 (2014-01 ~ 2026-04, 含2014券商暴动+2015股灾)")
print("=" * 70)

ext_mask = (df_full.date >= "2014-01-01") & (df_full.date <= "2026-04-30")
df_ext = df_full[ext_mask].reset_index(drop=True)
pos_ext = df_full[ext_mask]["v30_pos"].values

# 全样本最优: W60 [20%,80%] ext90%
split_ext = make_signal("ratio_xd_p60", 0.20, 0.80, 0.90, direction=-1)
split_ext_vals = split_ext[ext_mask.values]

r_style_ext = backtest_slice(df_ext, pos_ext, split_ext_vals)
r_300_ext = backtest_slice(df_ext, pos_ext, np.full(len(df_ext), 0.50))

print(f"\n  {'指标':<20} {'v3.0+风格v2':>15} {'v3.0纯300':>12}")
print(f"  {'-'*47}")
print(f"  {'年化收益':<18} {r_style_ext['ann']*100:>14.1f}% {r_300_ext['ann']*100:>11.1f}%")
print(f"  {'夏普比率':<18} {r_style_ext['sharpe']:>15.2f} {r_300_ext['sharpe']:>12.2f}")
print(f"  {'最大回撤':<18} {r_style_ext['max_dd']*100:>14.1f}% {r_300_ext['max_dd']*100:>11.1f}%")
print(f"  {'年化超额':<18} {(r_style_ext['ann']-r_300_ext['ann'])*100:>14.1f}pp")

# 2014-2015分年度
print(f"\n  分年度(含2014-2015):")
df_ext["year"] = df_ext["date"].dt.year
ret_300_ext = df_ext["沪深300"].pct_change().fillna(0)
ret_2000_ext = df_ext["国证2000"].pct_change().fillna(0)
w_300_ext = pos_ext * split_ext_vals
for i in range(1, len(df_ext)):
    if abs(split_ext_vals[i] - split_ext_vals[i-1]) < 0.10:
        w_300_ext[i] = pos_ext[i] * split_ext_vals[i-1]
w_2000_ext = pos_ext - w_300_ext
strat_ret_ext = w_300_ext * ret_300_ext.values + w_2000_ext * ret_2000_ext.values
bench_ret_ext = pos_ext * ret_300_ext.values

print(f"  {'年份':>6} {'风格v2':>10} {'纯300':>10} {'Δ':>8}")
for y in sorted(df_ext["year"].unique()):
    m = df_ext["year"] == y
    if m.sum() < 200: continue
    n = m.sum()
    sa = (np.prod(1+strat_ret_ext[m]))**(252/n) - 1
    ba = (np.prod(1+bench_ret_ext[m]))**(252/n) - 1
    print(f"  {y:>6} {sa*100:>9.1f}% {ba*100:>9.1f}% {(sa-ba)*100:>+7.1f}pp")

# ══════════════════════════════════════════════
# 8. 总结
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print("总结")
print("=" * 70)

# 判断是否通过
oos_pass = r_oos["sharpe_lift"] > 0.3 and r_oos["ann_lift"] > 0.03
rolling_pass = avg_lift > 0.2 and win_rate >= 75
ext_2014_pass = True  # 检查2014-2015是否有负超额

print(f"\n  1. 固定分割样本外:  夏普提升 {r_oos['sharpe_lift']:+.2f}, 年化提升 {r_oos['ann_lift']*100:+.1f}pp {'✓ 通过' if oos_pass else '✗ 未通过(阈值: 夏普Δ>0.3, 年化Δ>3pp)'}")
print(f"  2. 滚动窗口:        平均夏普提升 {avg_lift:+.2f}, 胜率 {win_rate:.0f}% {'✓ 通过' if rolling_pass else '✗ 未通过(阈值: 夏普Δ>0.2, 胜率>75%)'}")

# 检查2014-2015
yr14_m = df_ext["year"] == 2014
yr15_m = df_ext["year"] == 2015
if yr14_m.sum() >= 200:
    n14 = yr14_m.sum()
    s14 = (np.prod(1+strat_ret_ext[yr14_m]))**(252/n14) - 1
    b14 = (np.prod(1+bench_ret_ext[yr14_m]))**(252/n14) - 1
    print(f"  3. 2014券商暴动:    风格v2 {s14*100:+.1f}% vs 纯300 {b14*100:+.1f}% (Δ{(s14-b14)*100:+.1f}pp)")
if yr15_m.sum() >= 200:
    n15 = yr15_m.sum()
    s15 = (np.prod(1+strat_ret_ext[yr15_m]))**(252/n15) - 1
    b15 = (np.prod(1+bench_ret_ext[yr15_m]))**(252/n15) - 1
    ext_2014_pass = (s15 - b15) * 100 > -5  # 允许小幅跑输
    print(f"  4. 2015股灾:       风格v2 {s15*100:+.1f}% vs 纯300 {b15*100:+.1f}% (Δ{(s15-b15)*100:+.1f}pp)")

all_pass = oos_pass and rolling_pass and ext_2014_pass
print(f"\n  {'★ 全部通过，可以纳入v4.0 ★' if all_pass else '⚠ 存在未通过项，需谨慎评估'}")

print("\n完成.")
