#!/usr/bin/env python3
"""
豆瓜教程策略回测 #2: ETF动量轮动
策略: 4只ETF(黄金/纳指/创业板/沪深300), 哪个动量最强全仓买哪个, 全跌则空仓
动量: 20日涨幅, 周频调仓(周一)
成本: 测试多个滑点级别(0/0.1%/0.5%/1%)
数据源: 新浪(fund_etf_hist_sina + stock_zh_index_daily)
"""

import akshare as ak
import pandas as pd
import numpy as np
import os

OUTPUT = "/home/andy/backtest/dougua_results"
os.makedirs(OUTPUT, exist_ok=True)

MOM_WINDOW = 20
COSTS = [0.0, 0.001, 0.005, 0.01]

# ETF代码(新浪格式: sz=深市, sh=沪市)
ETF_POOL = {
    'sh518880': '黄金ETF',
    'sz159915': '创业板ETF',
    'sh510300': '沪深300ETF',
}
# 纳指ETF(QDII)新浪可能没有, 用纳指100指数代替
# 如果获取不到则跳过


def fetch_etf_data():
    """获取ETF历史日K线(新浪)"""
    print("[ETF轮动] 获取ETF数据(新浪)...")
    all_data = {}

    for code, name in ETF_POOL.items():
        print(f"  获取 {name}({code})...")
        try:
            df = ak.fund_etf_hist_sina(symbol=code)
            df = df.rename(columns={'date': 'date', 'close': 'close'})
            df['date'] = pd.to_datetime(df['date'])
            df['close'] = df['close'].astype(float)
            df = df.sort_values('date').reset_index(drop=True)
            all_data[code] = df[['date', 'close']].copy()
            print(f"    {len(df)} 条, {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
        except Exception as e:
            print(f"    获取失败: {e}")

    # 纳指替代：用sh513100(纳指ETF)
    print("  获取 纳指ETF(sh513100)...")
    try:
        df = ak.fund_etf_hist_sina(symbol='sh513100')
        df = df.rename(columns={'date': 'date', 'close': 'close'})
        df['date'] = pd.to_datetime(df['date'])
        df['close'] = df['close'].astype(float)
        df = df.sort_values('date').reset_index(drop=True)
        all_data['sh513100'] = df[['date', 'close']].copy()
        print(f"    {len(df)} 条, {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
    except Exception as e:
        print(f"    纳指ETF获取失败: {e}")

    # 合并到统一日期轴
    dfs = []
    for code, df in all_data.items():
        temp = df.rename(columns={'close': code})
        dfs.append(temp.set_index('date'))
    merged = pd.concat(dfs, axis=1)

    # 保留至少有2只ETF数据的日期
    merged = merged.dropna(thresh=2).sort_index().reset_index()
    print(f"  合并后 {len(merged)} 个交易日, 列: {list(merged.columns)}")
    return merged


def calc_momentum(df, window=20):
    """计算每只ETF的动量"""
    etf_cols = [c for c in df.columns if c != 'date']
    for code in etf_cols:
        df[f'mom_{code}'] = df[code].pct_change(window)
    return df


def backtest_single(df, cost=0.001):
    """单次回测"""
    etf_cols = [c for c in df.columns if c != 'date' and not c.startswith('mom_')]
    position = None
    nav = 1.0
    navs = []
    trades = 0

    df['weekday'] = df['date'].dt.dayofweek
    df['is_rebalance'] = df['weekday'] == 0

    for i in range(len(df)):
        if df.iloc[i]['is_rebalance'] and i > MOM_WINDOW:
            moms = {}
            for c in etf_cols:
                mcol = f'mom_{c}'
                if mcol in df.columns:
                    m = df.iloc[i][mcol]
                    if not np.isnan(m):
                        moms[c] = m

            best = None
            best_mom = -999
            for c, m in moms.items():
                if m > best_mom:
                    best_mom = m
                    best = c

            if best_mom <= 0:
                best = None

            if best != position:
                if position is not None:
                    nav *= (1 - cost)
                    trades += 1
                if best is not None:
                    nav *= (1 - cost)
                    trades += 1
                position = best

        if position is not None and i > 0:
            if position in df.columns:
                ret = df.iloc[i][position] / df.iloc[i - 1][position] - 1
                nav *= (1 + ret)

        navs.append(nav)

    return navs, trades


def calc_metrics(dates, navs):
    """计算回测指标"""
    s = pd.Series(navs, index=pd.to_datetime(dates))
    daily_ret = s.pct_change().dropna()

    total = s.iloc[-1] / s.iloc[0] - 1
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    annual = (1 + total) ** (1 / yrs) - 1

    dd = (s - s.cummax()) / s.cummax()
    max_dd = dd.min()

    sharpe = (daily_ret.mean() - 0.02 / 252) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0

    monthly = s.resample('ME').last().dropna()
    m_ret = monthly.pct_change().dropna()
    win_rate = (m_ret > 0).mean()

    yearly = {}
    for yr, grp in s.groupby(s.index.year):
        yearly[yr] = f"{grp.iloc[-1] / grp.iloc[0] - 1:.2%}"

    return {
        'total_return': f"{total:.2%}",
        'annual_return': f"{annual:.2%}",
        'max_drawdown': f"{max_dd:.2%}",
        'sharpe': round(sharpe, 2),
        'monthly_win_rate': f"{win_rate:.2%}",
        'yearly_returns': yearly,
    }


def main():
    print("=" * 60)
    print("豆瓜教程策略 #2: ETF动量轮动")
    print("=" * 60)

    df = fetch_etf_data()
    df = calc_momentum(df, MOM_WINDOW)

    all_results = {}

    for cost in COSTS:
        print(f"\n[ETF轮动] 滑点={cost:.1%} 回测...")
        navs, trades = backtest_single(df, cost)
        dates = df['date'].values
        metrics = calc_metrics(dates, navs)
        metrics['trades'] = trades
        metrics['cost'] = f"{cost:.1%}"
        all_results[f"cost_{cost}"] = metrics

        print(f"  年化: {metrics['annual_return']}, 回撤: {metrics['max_drawdown']}, "
              f"夏普: {metrics['sharpe']}, 交易: {trades}")

    valid_start = df.iloc[MOM_WINDOW]['date']

    print("\n" + "=" * 60)
    print(f"回测期间: {valid_start.strftime('%Y-%m-%d')} ~ {df.iloc[-1]['date'].strftime('%Y-%m-%d')}")
    print("=" * 60)
    for cost_label, m in all_results.items():
        print(f"\n--- 滑点: {m['cost']} ---")
        print(f"  总收益: {m['total_return']}")
        print(f"  年化: {m['annual_return']}")
        print(f"  最大回撤: {m['max_drawdown']}")
        print(f"  夏普: {m['sharpe']}")
        print(f"  月胜率: {m['monthly_win_rate']}")
        print(f"  交易次数: {m['trades']}")
        print("  年度:")
        for yr, r in m['yearly_returns'].items():
            print(f"    {yr}: {r}")

    with open(os.path.join(OUTPUT, "etf_momentum_result.txt"), 'w') as f:
        f.write(f"回测期间: {valid_start.strftime('%Y-%m-%d')} ~ {df.iloc[-1]['date'].strftime('%Y-%m-%d')}\n")
        f.write(f"ETF池: {ETF_POOL}\n")
        f.write(f"动量窗口: {MOM_WINDOW}日, 调仓: 周频(周一)\n\n")
        for cost_label, m in all_results.items():
            f.write(f"--- 滑点: {m['cost']} ---\n")
            for k, v in m.items():
                if k == 'yearly_returns':
                    f.write("  年度:\n")
                    for yr, r in v.items():
                        f.write(f"    {yr}: {r}\n")
                elif k not in ('cost',):
                    f.write(f"  {k}: {v}\n")
            f.write("\n")

    navs_01, _ = backtest_single(df, 0.001)
    nav_df = pd.DataFrame({'date': df['date'], 'nav': navs_01})
    nav_df.to_csv(os.path.join(OUTPUT, "etf_momentum_nav.csv"), index=False)

    print(f"\n结果已保存到 {OUTPUT}/etf_momentum_*")
    return all_results


if __name__ == '__main__':
    main()
