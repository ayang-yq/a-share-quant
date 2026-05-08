"""
投资体系v4.2 第一层回测：三因子仓位信号回测
=============================================
因子：PE分位 / 社融 / PMI → 仓位状态（进攻/均衡/防御）
基准：沪深300
回测期：2015-01 至 2026-04

评分规则：
  PE: +1(30-56分位) / 0(56-70或<30) / -1(>70)
  社融: +1(连续2月加速) / 0(波动<0.5pp) / -1(连续2月减速)
  PMI: +1(连续3月>50且上升) / 0(49-51) / -1(连续3月<50且下降)
  合计≥+2→进攻(90%) / ±1→均衡(72%) / ≤-2→防御(42%)
  封顶95% 下限20%
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime
import sys

# ── 1. 数据获取 ──────────────────────────────────────────────

print("正在拉取数据...")

# 1a. 沪深300 PE (日度)
print("  [1/4] 沪深300 PE...")
df_pe = ak.stock_index_pe_lg(symbol="沪深300")
df_pe = df_pe[["日期", "指数", "滚动市盈率"]].rename(columns={
    "日期": "date", "指数": "close", "滚动市盈率": "pe_ttm"
})
df_pe["date"] = pd.to_datetime(df_pe["date"])
df_pe = df_pe.sort_values("date").reset_index(drop=True)
# 过滤回测区间
df_pe = df_pe[df_pe["date"] >= "2014-06-01"].reset_index(drop=True)
print(f"    PE数据: {df_pe['date'].min().date()} ~ {df_pe['date'].max().date()}, {len(df_pe)}行")

# 1b. PMI (月度)
print("  [2/4] PMI...")
df_pmi_raw = ak.macro_china_pmi()
df_pmi = df_pmi_raw[["月份", "制造业-指数"]].rename(columns={
    "月份": "date_str", "制造业-指数": "pmi"
})
# 解析日期: "2026年03月份" -> "2026-03"
df_pmi["date_str"] = df_pmi["date_str"].str.replace("年", "-").str.replace("月份", "")
df_pmi["date"] = pd.to_datetime(df_pmi["date_str"] + "-01")
df_pmi = df_pmi.sort_values("date").reset_index(drop=True)
df_pmi = df_pmi[["date", "pmi"]]
print(f"    PMI数据: {df_pmi['date'].min().date()} ~ {df_pmi['date'].max().date()}, {len(df_pmi)}行")

# 1c. 社融 (月度)
print("  [3/4] 社融...")
df_sf_raw = ak.macro_china_new_financial_credit()
df_sf = df_sf_raw[["月份", "当月"]].rename(columns={
    "月份": "date_str", "当月": "sf_value"
})
df_sf["date_str"] = df_sf["date_str"].str.replace("年", "-").str.replace("月份", "")
df_sf["date"] = pd.to_datetime(df_sf["date_str"] + "-01")
df_sf = df_sf.sort_values("date").reset_index(drop=True)
df_sf = df_sf[["date", "sf_value"]]
print(f"    社融数据: {df_sf['date'].min().date()} ~ {df_sf['date'].max().date()}, {len(df_sf)}行")

# 1d. 沪深300 日行情 (用于计算收益)
print("  [4/4] 沪深300日行情...")
df_idx = ak.stock_zh_index_daily(symbol="sh000300")
df_idx["date"] = pd.to_datetime(df_idx["date"])
df_idx = df_idx.sort_values("date").reset_index(drop=True)
df_idx = df_idx[["date", "close"]].rename(columns={"close": "idx_close"})
print(f"    行情数据: {df_idx['date'].min().date()} ~ {df_idx['date'].max().date()}, {len(df_idx)}行")

# ── 2. PE分位计算 (滚动10年窗口) ────────────────────────────

print("\n计算PE分位...")

def calc_pe_percentile(row, history):
    """计算当前PE在历史窗口中的分位"""
    cutoff = row["date"] - pd.DateOffset(years=10)
    past = history[(history["date"] < row["date"]) & (history["date"] >= cutoff)]["pe_ttm"]
    if len(past) < 60:  # 至少60个交易日
        return np.nan
    return (past < row["pe_ttm"]).sum() / len(past)

df_pe["pe_pct"] = df_pe.apply(lambda r: calc_pe_percentile(r, df_pe), axis=1)

# 取每月15号的PE分位作为月度快照
df_pe["month"] = df_pe["date"].dt.to_period("M")
df_pe_monthly = df_pe.groupby("month").apply(
    lambda g: g.loc[g.index[(g["date"] - pd.Timestamp(g["date"].min().year, g["date"].min().month, 15)).abs().argsort()[:1]]]
).reset_index(drop=True)
df_pe_monthly = df_pe_monthly[["date", "pe_ttm", "pe_pct", "close"]]
df_pe_monthly = df_pe_monthly.rename(columns={"close": "pe_close"})

print(f"  PE分位: {df_pe_monthly['date'].min().date()} ~ {df_pe_monthly['date'].max().date()}, {len(df_pe_monthly)}月")

# ── 3. 社融YoY变化 (用于判断加速/减速) ─────────────────────

print("计算社融同比变化...")
df_sf = df_sf.sort_values("date").reset_index(drop=True)
# 当月同比增长率已在原数据，但"环比"不太稳定，用同比增速的变化更合理
# 改用：本月同比 - 上月同比 = 加速/减速
df_sf_raw2 = ak.macro_china_new_financial_credit()[["月份", "当月-同比增长"]].rename(columns={
    "月份": "date_str", "当月-同比增长": "sf_yoy"
})
df_sf_raw2["date_str"] = df_sf_raw2["date_str"].str.replace("年", "-").str.replace("月份", "")
df_sf_raw2["date"] = pd.to_datetime(df_sf_raw2["date_str"] + "-01")
df_sf_raw2 = df_sf_raw2.sort_values("date").reset_index(drop=True)
df_sf_raw2["sf_yoy_diff"] = df_sf_raw2["sf_yoy"].diff()
df_sf = df_sf.merge(df_sf_raw2[["date", "sf_yoy", "sf_yoy_diff"]], on="date", how="left")
df_sf = df_sf.sort_values("date").reset_index(drop=True)

# ── 4. 三因子评分 ────────────────────────────────────────────

print("计算三因子评分...")

# 找到三个月度数据的公共日期范围
start_date = max(df_pe_monthly["date"].min(), df_pmi["date"].min(), df_sf["date"].min())
end_date = min(df_pe_monthly["date"].max(), df_pmi["date"].max(), df_sf["date"].max())

# 合并月度数据
df_monthly = df_pe_monthly[["date", "pe_ttm", "pe_pct"]].copy()
df_monthly = df_monthly.merge(df_pmi[["date", "pmi"]], on="date", how="left")
df_monthly = df_monthly.merge(df_sf[["date", "sf_yoy", "sf_yoy_diff"]], on="date", how="left")
df_monthly = df_monthly.sort_values("date").reset_index(drop=True)
df_monthly = df_monthly[(df_monthly["date"] >= start_date) & (df_monthly["date"] <= end_date)]

# 评分函数
def score_pe(pct):
    """PE分位评分"""
    if pd.isna(pct):
        return 0
    if 0.30 <= pct <= 0.56:
        return 1
    elif pct > 0.70:
        return -1
    else:
        return 0

def score_sf(sf_yoy_diff_series, idx):
    """社融评分：连续2月加速/减速"""
    if idx < 2 or pd.isna(sf_yoy_diff_series.iloc[idx]) or pd.isna(sf_yoy_diff_series.iloc[idx-1]):
        return 0
    curr = sf_yoy_diff_series.iloc[idx]
    prev = sf_yoy_diff_series.iloc[idx - 1]
    # 加速=同比增速在提升
    if curr > 0.5 and prev > 0.5:
        return 1
    elif curr < -0.5 and prev < -0.5:
        return -1
    elif abs(curr) < 0.5 and abs(prev) < 0.5:
        return 0
    else:
        # mixed signal -> 0
        return 0

def score_pmi(pmi_series, idx):
    """PMI评分：连续3月>50且上升 / 连续3月<50且下降"""
    if idx < 2:
        return 0
    p0, p1, p2 = pmi_series.iloc[idx], pmi_series.iloc[idx-1], pmi_series.iloc[idx-2]
    if any(pd.isna(x) for x in [p0, p1, p2]):
        return 0
    # 连续3月>50且上升
    if p0 > 50 and p1 > 50 and p2 > 50 and p0 > p1 > p2:
        return 1
    # 连续3月<50且下降
    if p0 < 50 and p1 < 50 and p2 < 50 and p0 < p1 < p2:
        return -1
    # 49-51震荡区间
    if all(49 <= x <= 51 for x in [p0, p1, p2]):
        return 0
    return 0

# 计算每月评分
scores_pe = df_monthly["pe_pct"].apply(score_pe)
scores_sf = [score_sf(df_monthly["sf_yoy_diff"].reset_index(drop=True), i) for i in range(len(df_monthly))]
scores_pmi = [score_pmi(df_monthly["pmi"].reset_index(drop=True), i) for i in range(len(df_monthly))]

df_monthly["score_pe"] = scores_pe
df_monthly["score_sf"] = scores_sf
df_monthly["score_pmi"] = scores_pmi
df_monthly["total_score"] = df_monthly["score_pe"] + df_monthly["score_sf"] + df_monthly["score_pmi"]

# 映射仓位
def map_position(score):
    if score >= 2:
        return 0.90  # 进攻 90% (85-95中值)
    elif score <= -2:
        return 0.42  # 防御 42% (30-55中值)
    else:
        return 0.72  # 均衡 72% (65-80中值)

df_monthly["position"] = df_monthly["total_score"].apply(map_position)
df_monthly["state"] = df_monthly["total_score"].apply(
    lambda s: "进攻" if s >= 2 else ("防御" if s <= -2 else "均衡")
)

print(f"\n评分分布:")
print(df_monthly["state"].value_counts())

# ── 5. 回测引擎 ─────────────────────────────────────────────

print("\n运行回测...")

# 将月度仓位映射到日度
df_idx_test = df_idx[df_idx["date"] >= "2015-01-01"].copy().reset_index(drop=True)

# 为每个交易日找到当月仓位
df_idx_test["month"] = df_idx_test["date"].dt.to_period("M")
df_monthly["month"] = df_monthly["date"].dt.to_period("M")
pos_map = df_monthly.set_index("month")["position"].to_dict()

df_idx_test["position"] = df_idx_test["month"].map(pos_map)
df_idx_test = df_idx_test.dropna(subset=["position"]).reset_index(drop=True)

# 计算日收益
df_idx_test["daily_ret"] = df_idx_test["idx_close"].pct_change()

# 策略收益 = position * market_return (剩余资金假设0收益/货币基金收益)
# 更真实：假设空仓部分获得年化2%的货币基金收益
risk_free_daily = (1.02) ** (1/252) - 1

df_idx_test["strategy_ret"] = (
    df_idx_test["position"] * df_idx_test["daily_ret"] +
    (1 - df_idx_test["position"]) * risk_free_daily
)
df_idx_test["benchmark_ret"] = df_idx_test["daily_ret"]

# 累计净值
df_idx_test["strategy_nav"] = (1 + df_idx_test["strategy_ret"]).cumprod()
df_idx_test["benchmark_nav"] = (1 + df_idx_test["benchmark_ret"]).cumprod()

# ── 6. 统计指标 ─────────────────────────────────────────────

def calc_stats(nav_series, name):
    """计算核心统计指标"""
    ret = nav_series.pct_change().dropna()
    
    # 年化收益
    total_ret = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    n_years = (nav_series.index[-1] - nav_series.index[0]) / 252
    annual_ret = (1 + total_ret) ** (1 / n_years) - 1
    
    # 最大回撤
    cummax = nav_series.cummax()
    drawdown = (nav_series - cummax) / cummax
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    # 夏普比 (年化, 假设无风险2%)
    excess_ret = ret - risk_free_daily
    sharpe = np.sqrt(252) * excess_ret.mean() / excess_ret.std() if excess_ret.std() > 0 else 0
    
    # Calmar比率
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0
    
    # 胜率 (日频)
    win_rate = (ret > 0).sum() / len(ret)
    
    # 月度收益统计
    monthly_nav = nav_series.resample("ME").last()
    monthly_ret = monthly_nav.pct_change().dropna()
    
    # 最大连续亏损月数
    losing_months = (monthly_ret < 0).astype(int)
    max_consec_loss = 0
    curr = 0
    for v in losing_months:
        if v == 1:
            curr += 1
            max_consec_loss = max(max_consec_loss, curr)
        else:
            curr = 0
    
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  回测区间:      {nav_series.index[0].date()} ~ {nav_series.index[-1].date()}")
    print(f"  总收益:        {total_ret*100:+.1f}%")
    print(f"  年化收益:      {annual_ret*100:+.2f}%")
    print(f"  最大回撤:      {max_dd*100:.1f}% ({max_dd_date.date()})")
    print(f"  夏普比率:      {sharpe:.2f}")
    print(f"  Calmar比率:    {calmar:.2f}")
    print(f"  日胜率:        {win_rate*100:.1f}%")
    print(f"  月均收益:      {monthly_ret.mean()*100:+.2f}%")
    print(f"  月收益标准差:  {monthly_ret.std()*100:.2f}%")
    print(f"  最大连亏月数:  {max_consec_loss}")
    print(f"  正收益月份占比: {(monthly_ret > 0).sum() / len(monthly_ret) * 100:.1f}%")
    
    return {
        "name": name,
        "total_ret": total_ret,
        "annual_ret": annual_ret,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "calmar": calmar,
    }

stats_strat = calc_stats(df_idx_test.set_index("date")["strategy_nav"], "三因子策略")
stats_bench = calc_stats(df_idx_test.set_index("date")["benchmark_nav"], "沪深300满仓")

# ── 7. 分年度统计 ───────────────────────────────────────────

print(f"\n{'='*50}")
print("  分年度收益对比")
print(f"{'='*50}")
print(f"  {'年份':>6}  {'策略':>8}  {'基准':>8}  {'超额':>8}  {'状态占比(进/均/防)':>20}")
print(f"  {'-'*56}")

df_yr = df_idx_test.set_index("date")
for year in range(2015, 2027):
    mask = df_yr.index.year == year
    if mask.sum() < 20:
        continue
    
    # 合并月度状态
    yr_months = df_monthly[df_monthly["date"].dt.year == year]
    
    strat_nav = df_yr["strategy_nav"][mask]
    bench_nav = df_yr["benchmark_nav"][mask]
    
    s_ret = strat_nav.iloc[-1] / strat_nav.iloc[0] - 1
    b_ret = bench_nav.iloc[-1] / bench_nav.iloc[0] - 1
    
    n_total = len(yr_months)
    n_agg = (yr_months["state"] == "进攻").sum()
    n_bal = (yr_months["state"] == "均衡").sum()
    n_def = (yr_months["state"] == "防御").sum()
    
    print(f"  {year:>6}  {s_ret*100:>+7.1f}%  {b_ret*100:>+7.1f}%  {(s_ret-b_ret)*100:>+7.1f}%  "
          f"进{n_agg} 均{n_bal} 防{n_def}")

# ── 8. 状态切换统计 ─────────────────────────────────────────

print(f"\n{'='*50}")
print("  仓位状态切换统计")
print(f"{'='*50}")

# 各状态下的市场实际表现
for state in ["进攻", "均衡", "防御"]:
    mask = df_monthly["state"] == state
    if mask.sum() == 0:
        continue
    
    state_dates = df_monthly[mask]["date"]
    
    # 计算该状态下后续3个月的市场平均收益
    fwd_rets = []
    for d in state_dates:
        try:
            future_idx = df_idx_test[df_idx_test["date"] >= d]
            if len(future_idx) >= 60:  # ~3个月
                fwd_ret = future_idx["idx_close"].iloc[60] / future_idx["idx_close"].iloc[0] - 1
                fwd_rets.append(fwd_ret)
        except:
            pass
    
    if fwd_rets:
        avg_fwd = np.mean(fwd_rets) * 100
        avg_pos = df_monthly[mask]["position"].mean() * 100
        print(f"  {state}: {mask.sum()}个月, 平均仓位{avg_pos:.0f}%, "
              f"后续3月市场均收益{avg_fwd:+.1f}%, "
              f"正收益占比{(np.array(fwd_rets)>0).mean()*100:.0f}%")

# ── 9. 输出最终数据 ─────────────────────────────────────────

# 保存结果
output_path = "/home/andy/backtest/backtest_result.csv"
df_result = df_idx_test[["date", "idx_close", "position", "daily_ret", "benchmark_ret", "strategy_ret", "strategy_nav", "benchmark_nav"]].copy()
df_result["drawdown"] = (df_result["strategy_nav"] / df_result["strategy_nav"].cummax() - 1)
df_result.to_csv(output_path, index=False)
print(f"\n结果已保存: {output_path}")

# 保存月度评分
monthly_path = "/home/andy/backtest/monthly_scores.csv"
df_monthly.to_csv(monthly_path, index=False)
print(f"月度评分已保存: {monthly_path}")

# ── 10. 总结 ─────────────────────────────────────────────────

print(f"\n{'='*50}")
print("  核心结论")
print(f"{'='*50}")

excess_annual = (stats_strat["annual_ret"] - stats_bench["annual_ret"]) * 100
dd_improvement = (abs(stats_bench["max_dd"]) - abs(stats_strat["max_dd"])) * 100

print(f"  年化超额收益: {excess_annual:+.2f}%")
print(f"  最大回撤改善: {dd_improvement:+.1f}pp")
print(f"  夏普比提升:   {stats_strat['sharpe'] - stats_bench['sharpe']:+.2f}")

if stats_strat["max_dd"] > stats_bench["max_dd"]:
    print(f"  ⚠️ 策略最大回撤反而更大，需检查防御信号有效性")
if excess_annual > 0 and stats_strat["sharpe"] > stats_bench["sharpe"]:
    print(f"  ✅ 策略在降低风险的同时获得了正超额收益")
elif excess_annual < 0:
    print(f"  ⚠️ 策略跑输基准，仓位管理可能过度保守")
