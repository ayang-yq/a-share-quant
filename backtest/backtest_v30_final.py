"""
A股多因子择时策略 v3.0 — 最终验证
=====================================
前序测试总结:
  - ERP: 所有"便宜多买"映射均负超额，砍掉
  - VIX: 全线失败，砍掉
  - 融资BB与持仓占比: 频率不匹配(3% vs 97%)，合成有害，但连续加权可解
  - 价格BB: 最强指标，夏普1.21~1.31
  - 融资BB: 低频高精度，连续z-score加权后有效补充
  - 大小盘风格BB: 仅国证2000有效(夏普1.14)，大盘股无效
  - 左右合成(离散信号): 稀释右侧，夏普从1.47降到0.74~1.18
  - 连续加权60/40(价格BB+融资BB): 三指数夏普1.46~1.51，最优右侧方案

本脚本验证:
  1. 核心引擎: 连续加权右侧 (多权重组合)
  2. 附加值测试: 融资持仓占比是否值得保留
  3. 大小盘风格BB: 国证2000专项测试
  4. 样本内外稳定性
  5. 危机期表现
  6. 与满仓基准/50%半仓的全面对比
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, akshare as ak

# ══════════════════════════════════════════════════════════
# 1. 数据获取
# ══════════════════════════════════════════════════════════
print("=" * 80)
print("A股多因子择时策略 v3.0 — 最终验证")
print("=" * 80)

indices = {"沪深300": "sh000300", "中证800": "sh000906", "国证2000": "sz399303"}
all_price = {}
for name, code in indices.items():
    print(f"  {name}...", end=" ", flush=True)
    df = ak.stock_zh_index_daily(symbol=code)[["date","close"]].rename(columns={"close": name})
    df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date").reset_index(drop=True)
    all_price[name] = df
    print("OK")

print("  融资数据...", end=" ", flush=True)
df_mg = ak.stock_margin_account_info()
df_mg["date"] = pd.to_datetime(df_mg["日期"]); df_mg = df_mg.sort_values("date").reset_index(drop=True)
df_msh = ak.macro_china_market_margin_sh()[["日期","融资买入额"]].rename(columns={"日期":"date","融资买入额":"rzye_sh"})
df_msh["date"] = pd.to_datetime(df_msh["date"]); df_msh = df_msh.sort_values("date").reset_index(drop=True)
df_msz = ak.macro_china_market_margin_sz()[["日期","融资买入额"]].rename(columns={"日期":"date","融资买入额":"rzye_sz"})
df_msz["date"] = pd.to_datetime(df_msz["date"]); df_msz = df_msz.sort_values("date").reset_index(drop=True)
print("OK")

print("  申万风格...", end=" ", flush=True)
df_dpz = ak.index_hist_sw(symbol="801811", period="day")[["日期","收盘"]].rename(columns={"日期":"date","收盘":"dpz"})
df_dpz["date"] = pd.to_datetime(df_dpz["date"])
df_xpz = ak.index_hist_sw(symbol="801813", period="day")[["日期","收盘"]].rename(columns={"日期":"date","收盘":"xpz"})
df_xpz["date"] = pd.to_datetime(df_xpz["date"])
print("OK")

# 合并
base = all_price["沪深300"].copy()
for name in ["中证800", "国证2000"]:
    base = base.merge(all_price[name][["date", name]], on="date", how="left")
base = base.merge(df_mg[["date","有融资融券负债的投资者数量","个人投资者数量"]], on="date", how="left")
base = base.merge(df_msh[["date","rzye_sh"]], on="date", how="left")
base = base.merge(df_msz[["date","rzye_sz"]], on="date", how="left")
base = base.merge(df_dpz[["date","dpz"]], on="date", how="left")
base = base.merge(df_xpz[["date","xpz"]], on="date", how="left")
for c in ["中证800","国证2000","有融资融券负债的投资者数量","个人投资者数量","rzye_sh","rzye_sz","dpz","xpz"]:
    base[c] = base[c].ffill()
base = base[(base.date >= "2015-01-01") & (base.date <= "2026-04-30")].reset_index(drop=True)
df = base.copy()
print(f"\n  合并完成: {len(df)}天")

# ══════════════════════════════════════════════════════════
# 2. 指标计算
# ══════════════════════════════════════════════════════════
print("计算指标...", flush=True)

# ── A. 价格BB (沪深300) ──
df["bb_ma20"] = df["沪深300"].rolling(20).mean()
df["bb_std20"] = df["沪深300"].rolling(20).std()
df["bb_z"] = (df["沪深300"] - df["bb_ma20"]) / df["bb_std20"]

# ── B. 融资买入额BB ──
df["rzye_total"] = df["rzye_sh"] + df["rzye_sz"]
df["rzye_ma20"] = df["rzye_total"].rolling(20).mean()
df["rzye_std20"] = df["rzye_total"].rolling(20).std()
df["rzye_z"] = (df["rzye_total"] - df["rzye_ma20"]) / df["rzye_std20"]

# ── C. 融资持仓占比(反向) ──
df["margin_ratio"] = df["有融资融券负债的投资者数量"] / df["个人投资者数量"]
df["margin_ma10"] = df["margin_ratio"].rolling(10).mean()
df["margin_diff"] = df["margin_diff_raw"] = df["margin_ma10"] - df["margin_ma10"].shift(5)

# ── D. 大小盘风格BB (趋势跟随) ──
df["size_ratio"] = df["xpz"] / df["dpz"]
df["size_ma20"] = df["size_ratio"].rolling(20).mean()
df["size_std20"] = df["size_ratio"].rolling(20).std()
df["size_z"] = (df["size_ratio"] - df["size_ma20"]) / df["size_std20"]

# 预热期
df = df.iloc[30:].reset_index(drop=True)
print(f"  预热后: {len(df)}天")

# ══════════════════════════════════════════════════════════
# 3. 仓位生成器
# ══════════════════════════════════════════════════════════
rf_d = 1.02 ** (1/252) - 1

def z2p(z, lo=0.20, hi=0.80):
    """z-score → 仓位: z∈[-3,3] → [lo, hi], 中性50%"""
    z = np.clip(z, -3, 3)
    return 0.50 + z / 3 * (hi - 0.50)

def make_right(df, w_bb=0.60, w_rzye=0.40):
    """右侧: 价格BB和融资BB的连续加权"""
    return df["bb_z"].apply(lambda v: z2p(v) * w_bb if not pd.isna(v) else 0.50 * w_bb) + \
           df["rzye_z"].apply(lambda v: z2p(v) * w_rzye if not pd.isna(v) else 0.50 * w_rzye)

def make_right_with_size(df, w_bb=0.50, w_rzye=0.30, w_size=0.20):
    """右侧+大小盘风格: 三因子连续加权(仅国证2000)"""
    return df["bb_z"].apply(lambda v: z2p(v) * w_bb if not pd.isna(v) else 0.50 * w_bb) + \
           df["rzye_z"].apply(lambda v: z2p(v) * w_rzye if not pd.isna(v) else 0.50 * w_rzye) + \
           df["size_z"].apply(lambda v: z2p(v) * w_size if not pd.isna(v) else 0.50 * w_size)

def make_left(df):
    """左侧: 融资持仓占比(反向), 连续化"""
    # diff5 → z-score化 → 仓位
    df_margin = df["margin_diff"]
    ma = df_margin.rolling(60).mean()
    std = df_margin.rolling(60).std()
    z = (df_margin - ma) / std
    # z<0(下降=底部) → 多, z>0(上升=顶部) → 空
    # 反转: 所以乘以-1
    return z.apply(lambda v: z2p(-v) if not pd.isna(v) else 0.50)

# ══════════════════════════════════════════════════════════
# 4. 回测引擎
# ══════════════════════════════════════════════════════════
def bt(price_col, pos_arr, label=""):
    mkt_ret = df[price_col].pct_change().fillna(0).values
    pos = np.nan_to_num(pos_arr, nan=0.50)
    strat_ret = pos * mkt_ret + (1-pos) * rf_d
    strat_nav = np.cumprod(1+strat_ret)
    bench_nav = np.cumprod(1+mkt_ret)
    yrs = len(strat_nav)/252
    r = pd.Series(strat_ret)
    b = pd.Series(mkt_ret)
    stats = {
        "label": label,
        "bench_ann": (bench_nav[-1]/bench_nav[0])**(1/yrs)-1,
        "strat_ann": (strat_nav[-1]/strat_nav[0])**(1/yrs)-1,
        "excess": ((strat_nav[-1]/strat_nav[0])**(1/yrs)-1) - ((bench_nav[-1]/bench_nav[0])**(1/yrs)-1),
        "bench_sharpe": np.sqrt(252)*(b-rf_d).mean()/b.std() if b.std()>0 else 0,
        "sharpe": np.sqrt(252)*(r-rf_d).mean()/r.std() if r.std()>0 else 0,
        "bench_dd": (pd.Series(bench_nav)/pd.Series(bench_nav).cummax()-1).min(),
        "max_dd": (pd.Series(strat_nav)/pd.Series(strat_nav).cummax()-1).min(),
        "calmar": ((strat_nav[-1]/strat_nav[0])**(1/yrs)-1) / abs((pd.Series(strat_nav)/pd.Series(strat_nav).cummax()-1).min()) if (pd.Series(strat_nav)/pd.Series(strat_nav).cummax()-1).min()!=0 else 0,
    }
    # 月胜率
    m_nav = pd.Series(strat_nav, index=df["date"].values).resample("ME").last()
    m_ret = m_nav.pct_change().dropna()
    stats["win_rate"] = (m_ret > 0).mean()
    stats["dd_improve"] = abs(stats["bench_dd"]) - abs(stats["max_dd"])
    return stats

def bt_section(price_col, pos_arr, start, end, label=""):
    mask = (df.date >= start) & (df.date <= end)
    idx = np.where(mask)[0]
    if len(idx) < 50: return None
    sub = df.iloc[idx]
    sub_pos = pos_arr[idx]
    mkt_ret = sub[price_col].pct_change().fillna(0).values
    bench_nav = np.cumprod(1+mkt_ret)
    strat_ret = sub_pos * mkt_ret + (1-sub_pos) * rf_d
    strat_nav = np.cumprod(1+strat_ret)
    yrs = len(strat_nav)/252
    b_nav_end = (bench_nav[-1]/bench_nav[0])-1
    s_nav_end = (strat_nav[-1]/strat_nav[0])-1
    s_dd = (pd.Series(strat_nav)/pd.Series(strat_nav).cummax()-1).min()
    r = pd.Series(strat_ret)
    sh = np.sqrt(252)*(r-rf_d).mean()/r.std() if r.std()>0 else 0
    return {"label": label, "bench_ret": b_nav_end, "strat_ret": s_nav_end,
            "excess": s_nav_end - b_nav_end, "dd": s_dd, "sharpe": sh}

# ══════════════════════════════════════════════════════════
# 5. 指标全面审计
# ══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("第一部分: 指标全面审计 (7个候选指标 × 3个指数)")
print("="*80)

def s2p(s): return 0.80 if s==1 else (0.20 if s==-1 else 0.50)

# 离散信号版
df["sig_bb"] = df["bb_z"].apply(lambda v: 1 if not pd.isna(v) and v>2 else (-1 if not pd.isna(v) and v<-2 else 0))
df["sig_rzye"] = df["rzye_z"].apply(lambda v: 1 if not pd.isna(v) and v>2 else (-1 if not pd.isna(v) and v<-2 else 0))
df["sig_margin"] = df["margin_diff"].apply(lambda v: 1 if not pd.isna(v) and v<0 else (-1 if not pd.isna(v) and v>0 else 0))
df["sig_size"] = df["size_z"].apply(lambda v: 1 if not pd.isna(v) and v>2 else (-1 if not pd.isna(v) and v<-2 else 0))

indicators = [
    ("① 价格BB(趋势)", "sig_bb"),
    ("② 融资买入额BB(趋势)", "sig_rzye"),
    ("③ 融资持仓占比(反转)", "sig_margin"),
    ("④ 大小盘风格BB(趋势)", "sig_size"),
    ("⑤ 价格BB连续", "bb_z", True),
    ("⑥ 融资BB连续", "rzye_z", True),
    ("⑦ 大小盘风格连续", "size_z", True),
]

for idx in ["沪深300", "中证800", "国证2000"]:
    b = bt(idx, np.ones(len(df)))
    print(f"\n  {idx}  基准: 年化{b['bench_ann']:.1%} 夏普{b['bench_sharpe']:.2f} 回撤{b['bench_dd']:.1%}")
    print(f"  {'指标':<30s} {'年化':>7s} {'超额':>7s} {'夏普':>6s} {'回撤':>7s} {'改善':>7s}")
    print(f"  {'─'*70}")

    results = []
    # 直接遍历 indicators
    results = []
    for item in indicators:
        name, col = item[0], item[1]
        is_cont = len(item) > 2
        if is_cont:
            pos = df[col].apply(lambda v: z2p(v) if not pd.isna(v) else 0.50).values
        else:
            pos = df[col].apply(s2p).values
        r = bt(idx, pos, name)
        results.append(r)

    for r in sorted(results, key=lambda x: x["excess"], reverse=True):
        print(f"  {r['label']:<30s} {r['strat_ann']:>6.1%} {r['excess']:>+6.1%} {r['sharpe']:>6.2f} {r['max_dd']:>7.1%} {r['dd_improve']:>+6.1%}pp")

# ══════════════════════════════════════════════════════════
# 6. 核心引擎: 右侧连续加权 (多权重对比)
# ══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("第二部分: 核心引擎 — 右侧连续加权 (价格BB + 融资BB)")
print("="*80)

weights = [
    ("纯价格BB", 1.0, 0.0),
    ("纯融资BB", 0.0, 1.0),
    ("70/30", 0.70, 0.30),
    ("65/35", 0.65, 0.35),
    ("60/40", 0.60, 0.40),
    ("55/45", 0.55, 0.45),
    ("50/50", 0.50, 0.50),
]

for idx in ["沪深300", "中证800", "国证2000"]:
    print(f"\n  {idx}:")
    print(f"  {'权重(价格/融资)':<22s} {'年化':>7s} {'超额':>7s} {'夏普':>6s} {'回撤':>7s} {'改善':>7s} {'Calmar':>6s}")
    print(f"  {'─'*75}")
    b = bt(idx, np.ones(len(df)))
    for name, w1, w2 in weights:
        pos = make_right(df, w1, w2).values
        r = bt(idx, pos, name)
        print(f"  {name:<22s} {r['strat_ann']:>6.1%} {r['excess']:>+6.1%} {r['sharpe']:>6.2f} {r['max_dd']:>7.1%} {r['dd_improve']:>+6.1%}pp {r['calmar']:>6.2f}")

# ══════════════════════════════════════════════════════════
# 7. 国证2000专项: 加入大小盘风格
# ══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("第三部分: 国证2000专项 — 三因子 (价格BB + 融资BB + 大小盘风格)")
print("="*80)

idx = "国证2000"
b = bt(idx, np.ones(len(df)))
print(f"\n  基准: 年化{b['bench_ann']:.1%} 夏普{b['bench_sharpe']:.2f} 回撤{b['bench_dd']:.1%}")
print(f"  {'权重(价/融/风)':<22s} {'年化':>7s} {'超额':>7s} {'夏普':>6s} {'回撤':>7s} {'改善':>7s}")
print(f"  {'─'*70}")

size_weights = [
    ("二因子 60/40/0", 0.60, 0.40, 0.00),
    ("二因子 65/35/0", 0.65, 0.35, 0.00),
    ("三因子 50/30/20", 0.50, 0.30, 0.20),
    ("三因子 45/30/25", 0.45, 0.30, 0.25),
    ("三因子 40/30/30", 0.40, 0.30, 0.30),
    ("三因子 50/25/25", 0.50, 0.25, 0.25),
]

for name, w1, w2, w3 in size_weights:
    if w3 == 0:
        pos = make_right(df, w1, w2).values
    else:
        pos = make_right_with_size(df, w1, w2, w3).values
    r = bt(idx, pos, name)
    print(f"  {name:<22s} {r['strat_ann']:>6.1%} {r['excess']:>+6.1%} {r['sharpe']:>6.2f} {r['max_dd']:>7.1%} {r['dd_improve']:>+6.1%}pp")

# ══════════════════════════════════════════════════════════
# 8. 最终框架对比
# ══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("第四部分: 最终框架方案对比")
print("="*80)

# 方案定义
configs = {
    "满仓基准": lambda idx: np.ones(len(df)),
    "50%半仓": lambda idx: np.full(len(df), 0.50),
    "仅价格BB离散": lambda idx: df["sig_bb"].apply(s2p).values,
    "仅融资BB离散": lambda idx: df["sig_rzye"].apply(s2p).values,
    "v2.0框架(ERP+持仓+宽松投票BB)": lambda idx: None,  # placeholder
    "★ v3.0核心(60/40连续)": lambda idx: make_right(df, 0.60, 0.40).values,
    "★ v3.0激进(65/35连续)": lambda idx: make_right(df, 0.65, 0.35).values,
    "★ v3.0保守(70/30连续)": lambda idx: make_right(df, 0.70, 0.30).values,
}

# v2.0框架 (ERP分位+持仓占比+宽松投票) for reference
def make_v20(df):
    df_pe = ak.stock_index_pe_lg(symbol="沪深300")[["日期","滚动市盈率"]].rename(columns={"日期":"date","滚动市盈率":"pe"})
    df_pe["date"] = pd.to_datetime(df_pe["date"]); df_pe = df_pe.sort_values("date").reset_index(drop=True)
    df_bond = ak.bond_zh_us_rate()[["日期","中国国债收益率10年"]].rename(columns={"日期":"date","中国国债收益率10年":"bond10y"})
    df_bond["date"] = pd.to_datetime(df_bond["date"]); df_bond = df_bond.sort_values("date").reset_index(drop=True)
    tmp = df[["date"]].copy().merge(df_pe[["date","pe"]], on="date", how="left").merge(df_bond[["date","bond10y"]], on="date", how="left")
    for c in ["pe","bond10y"]: tmp[c] = tmp[c].ffill()
    tmp["erp"] = 1.0/tmp["pe"] - tmp["bond10y"]/100
    def rolling_pct(s, w=10):
        n=len(s); r=np.full(n,np.nan); win=int(w*252); v=s.values
        for i in range(n):
            st=max(0,i-win); p=v[st:i]
            if len(p)>=252 and not np.isnan(v[i]): r[i]=(p<v[i]).sum()/len(p)
        return r
    tmp["erp_pct"] = rolling_pct(tmp["erp"])
    def erp_pos(p):
        if pd.isna(p): return 0.65
        if p<0.10: return 0.50
        if p<0.30: return 0.85
        if p<0.50: return 0.70
        if p<0.70: return 0.55
        if p<0.90: return 0.40
        return 0.30
    tmp["pos_erp"] = tmp["erp_pct"].apply(erp_pos)
    tmp["pos_margin"] = df["margin_diff"].apply(lambda v: 0.80 if (not pd.isna(v) and v<0) else (0.20 if (not pd.isna(v) and v>0) else 0.50))
    tmp["left_avg"] = (tmp["pos_erp"] + tmp["pos_margin"]) / 2
    # 右侧宽松投票
    def vote(r):
        s1 = r["sig_bb"]; s2 = r["sig_rzye"]
        if s1==1 or s2==1: return 0.80
        if s1==-1 and s2==-1: return 0.20
        return 0.50
    tmp["right"] = df.apply(lambda r: 0.80 if (r["sig_bb"]==1 or r["sig_rzye"]==1) else (0.20 if (r["sig_bb"]==-1 and r["sig_rzye"]==-1) else 0.50), axis=1)
    # 左右合成
    left_sig = tmp["left_avg"].apply(lambda v: 1 if v > 0.60 else (-1 if v < 0.40 else 0))
    right_sig = tmp["right"].apply(lambda v: 1 if v > 0.60 else (-1 if v < 0.40 else 0))
    combo = []
    for l, r in zip(left_sig.values, right_sig.values):
        if l==1 and r==1: combo.append(0.80)
        elif l==-1 and r==-1: combo.append(0.20)
        else: combo.append(0.50)
    return np.array(combo)

print("  生成v2.0基准...", flush=True)
v20_pos = make_v20(df)

print(f"\n  {'方案':<38s} {'沪深300':>14s} {'中证800':>14s} {'国证2000':>14s}")
print(f"  {'':>38s} {'夏普':>6s} {'超额':>6s} {'夏普':>6s} {'超额':>6s} {'夏普':>6s} {'超额':>6s}")
print(f"  {'─'*95}")

all_configs = [
    ("满仓基准", lambda idx: np.ones(len(df))),
    ("50%半仓", lambda idx: np.full(len(df), 0.50)),
    ("仅价格BB(离散)", lambda idx: df["sig_bb"].apply(s2p).values),
    ("仅融资BB(离散)", lambda idx: df["sig_rzye"].apply(s2p).values),
    ("v2.0(ERP+持仓+宽松投票)", lambda idx: v20_pos),
    ("v3.0 核心版 (60/40连续)", lambda idx: make_right(df, 0.60, 0.40).values),
    ("v3.0 激进版 (65/35连续)", lambda idx: make_right(df, 0.65, 0.35).values),
    ("v3.0 保守版 (70/30连续)", lambda idx: make_right(df, 0.70, 0.30).values),
]

for name, gen in all_configs:
    row = f"  {name:<38s}"
    for idx in ["沪深300", "中证800", "国证2000"]:
        pos = gen(idx)
        r = bt(idx, pos)
        row += f" {r['sharpe']:>5.2f} {r['excess']:>+5.1%}"
    print(row)

# ══════════════════════════════════════════════════════════
# 9. 样本内外稳定性
# ══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("第五部分: 样本内外稳定性 (v3.0核心 60/40)")
print("="*80)

splits = [("2015-2019(样本内)", "2015-01-01", "2019-12-31"),
          ("2020-2022(样本外前期)", "2020-01-01", "2022-12-31"),
          ("2023-2026(样本外近期)", "2023-01-01", "2026-04-30"),
          ("2020-2026(全样本外)", "2020-01-01", "2026-04-30")]

pos_6040 = make_right(df, 0.60, 0.40).values

for idx in ["沪深300", "中证800", "国证2000"]:
    print(f"\n  {idx}:")
    print(f"  {'区间':<30s} {'基准年化':>8s} {'策略年化':>8s} {'超额':>7s} {'回撤':>7s} {'夏普':>6s}")
    print(f"  {'─'*75}")
    for name, ss, se in splits:
        r = bt_section(idx, pos_6040, ss, se)
        if r:
            ok = "✓" if r["excess"] > 0 else "✗"
            print(f"  {name:<30s} {r['bench_ret']:>+7.1%} {r['strat_ret']:>+7.1%} {r['excess']:>+6.1%} {r['dd']:>7.1%} {r['sharpe']:>6.2f} {ok}")

# ══════════════════════════════════════════════════════════
# 10. 危机期表现
# ══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("第六部分: 危机期表现 (v3.0核心 60/40 vs 满仓)")
print("="*80)

crises = [
    ("2015股灾", "2015-06-12", "2015-09-30"),
    ("2016熔断", "2016-01-01", "2016-02-29"),
    ("2018贸易战", "2018-01-26", "2018-10-19"),
    ("2020疫情", "2020-01-14", "2020-03-23"),
    ("2022熊市", "2021-12-13", "2022-10-31"),
    ("2024阴跌", "2023-08-01", "2024-09-13"),
]

for idx in ["沪深300"]:
    print(f"\n  {idx}:")
    print(f"  {'危机':<18s} {'满仓收益':>8s} {'策略收益':>8s} {'超额':>7s} {'少亏':>7s}")
    print(f"  {'─'*60}")
    for name, ss, se in crises:
        r = bt_section(idx, pos_6040, ss, se)
        if r:
            saved = r["bench_ret"] - r["strat_ret"]
            print(f"  {name:<18s} {r['bench_ret']:>+7.1%} {r['strat_ret']:>+7.1%} {r['excess']:>+6.1%} {saved:>+6.1%}pp")

# ══════════════════════════════════════════════════════════
# 11. 仓位分布与换手分析
# ══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("第七部分: 仓位分布")
print("="*80)

pos_arr = pos_6040
pos_series = pd.Series(pos_arr)
print(f"\n  v3.0核心(60/40) 仓位分布:")
print(f"    均值: {pos_series.mean():.1%}")
print(f"    中位: {pos_series.median():.1%}")
print(f"    标准差: {pos_series.std():.1%}")
print(f"    范围: [{pos_series.min():.1%}, {pos_series.max():.1%}]")
print(f"    <30% (低仓位): {(pos_series < 0.30).sum()}天 ({(pos_series < 0.30).mean():.0%})")
print(f"    30~40%: {((pos_series >= 0.30) & (pos_series < 0.40)).sum()}天")
print(f"    40~50%: {((pos_series >= 0.40) & (pos_series < 0.50)).sum()}天")
print(f"    50~60%: {((pos_series >= 0.50) & (pos_series < 0.60)).sum()}天")
print(f"    60~70%: {((pos_series >= 0.60) & (pos_series < 0.70)).sum()}天")
print(f"    >70% (高仓位): {(pos_series > 0.70).sum()}天 ({(pos_series > 0.70).mean():.0%})")

# 日均换手
turnover = np.abs(np.diff(pos_arr)).mean()
annual_turnover = turnover * 252
print(f"\n  日均换手: {turnover:.2%}")
print(f"  年化换手: {annual_turnover:.0%}倍")

# ══════════════════════════════════════════════════════════
# 12. 国证2000最终推荐
# ══════════════════════════════════════════════════════════
print("\n" + "="*80)
print("第八部分: 国证2000 — 二因子 vs 三因子 (含大小盘风格)")
print("="*80)

idx = "国证2000"
pos_2f = make_right(df, 0.60, 0.40).values
pos_3f = make_right_with_size(df, 0.45, 0.30, 0.25).values

for label, pos in [("二因子(60/40/0)", pos_2f), ("三因子(45/30/25)", pos_3f)]:
    print(f"\n  {label}:")
    for sn, ss, se in splits:
        r = bt_section(idx, pos, ss, se)
        if r:
            ok = "✓" if r["excess"] > 0 else "✗"
            print(f"    {sn:<30s} 超额{r['excess']:>+6.1%}  回撤{r['dd']:>7.1%}  夏普{r['sharpe']:>5.2f} {ok}")

print("\n" + "="*80)
print("验证完成。")
print("="*80)
