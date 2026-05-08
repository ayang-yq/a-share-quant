#!/usr/bin/env python3
"""
申万一级行业景气度轮动策略回测 v1.0
独立框架，不依赖v4.3.1择时体系

策略逻辑：
  景气度 = 市场认可的基本面边际改善
  代理指标：价格动量 + 量能 + 相对强度

回测方案：
  A: 纯60日动量轮动(基线)
  B: 景气度综合评分(动量+加速度+量能+相对强度)
  C: 景气度 + PE趋势过滤(排除"增收不增利"假景气)
  D: 景气度 + 滞后确认(信号出现后延迟1个月再入场)
  基准: 沪深300
"""

import akshare as ak
import pandas as pd
import numpy as np
import time, os, pickle, warnings
warnings.filterwarnings('ignore')

CACHE = '/tmp/prosperity_bt'
os.makedirs(CACHE, exist_ok=True)

# =========================================================
# Step 1: 获取申万一级31行业数据
# =========================================================
print("=" * 60)
print("Step 1: 获取申万一级行业代码和历史行情")

# 获取行业代码
code_cache = f'{CACHE}/sw_codes.pkl'
if os.path.exists(code_cache):
    with open(code_cache, 'rb') as f:
        sw_info = pickle.load(f)
else:
    sw_info = ak.sw_index_first_info()
    with open(code_cache, 'wb') as f:
        pickle.dump(sw_info, f)

industries = {}
for _, row in sw_info.iterrows():
    code = row['行业代码'].replace('.SI', '')
    industries[row['行业名称']] = code

print(f"  31行业: {list(industries.keys())[:5]}... (共{len(industries)}个)")

# =========================================================
# Step 2: 下载历史行情数据
# =========================================================
print("\nStep 2: 下载行业历史行情(可能需要2-3分钟)...")

price_cache = f'{CACHE}/price_data.pkl'
if os.path.exists(price_cache):
    with open(price_cache, 'rb') as f:
        price_data = pickle.load(f)
    print(f"  从缓存加载({len(price_data)}个行业)")
else:
    price_data = {}
    for i, (name, code) in enumerate(industries.items()):
        try:
            df = ak.index_hist_sw(symbol=code, period="day")
            df = df.rename(columns={'日期': 'date', '收盘': 'close', 
                                     '成交量': 'volume', '成交额': 'amount',
                                     '开盘': 'open', '最高': 'high', '最低': 'low'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            price_data[name] = df
            if (i + 1) % 10 == 0 or i == len(industries) - 1:
                print(f"  进度: {i+1}/{len(industries)}")
        except Exception as e:
            print(f"  {name}: 失败 - {e}")
        time.sleep(0.3)
    
    with open(price_cache, 'wb') as f:
        pickle.dump(price_data, f)
    print(f"  数据已缓存到 {price_cache}")

# =========================================================
# Step 3: 下载基准(沪深300)
# =========================================================
print("\nStep 3: 下载沪深300基准...")

bench_cache = f'{CACHE}/hs300.pkl'
if os.path.exists(bench_cache):
    with open(bench_cache, 'rb') as f:
        bench_df = pickle.load(f)
    print("  从缓存加载")
else:
    # 尝试多个数据源
    bench_df = None
    for attempt in range(3):
        try:
            # 新浪源
            bench_df = ak.stock_zh_index_daily(symbol="sh000300")
            bench_df = bench_df.rename(columns={'date': 'date', 'close': 'close'})
            print(f"  新浪源成功")
            break
        except:
            try:
                time.sleep(2)
                bench_df = ak.stock_zh_index_daily_em(symbol="sh000300")
                bench_df = bench_df.rename(columns={'close': 'close'})
                print(f"  东财源成功")
                break
            except:
                time.sleep(3)
    
    if bench_df is None:
        raise RuntimeError("无法获取沪深300数据，请稍后重试")
    
    bench_df['date'] = pd.to_datetime(bench_df['date'])
    bench_df = bench_df.set_index('date').sort_index()
    bench_df = bench_df[['close']]  # 只保留close列
    with open(bench_cache, 'wb') as f:
        pickle.dump(bench_df, f)
    print(f"  沪深300: {len(bench_df)}天, 范围 {bench_df.index[0]} ~ {bench_df.index[-1]}")

# =========================================================
# Step 4: 构建月度面板数据
# =========================================================
print("\nStep 4: 构建月度数据面板...")

# 行业月度收盘价面板
all_closes = pd.DataFrame()
for name, df in price_data.items():
    m = df['close'].resample('ME').last()
    all_closes[name] = m

# 行业月度成交量面板
all_volumes = pd.DataFrame()
for name, df in price_data.items():
    m = df['volume'].resample('ME').sum()
    all_volumes[name] = m

# 月度收益率
monthly_returns = all_closes.pct_change()

# 沪深300月度收益率
bench_monthly = bench_df['close'].resample('ME').last().pct_change()

# 对齐日期范围
start_date = pd.Timestamp('2015-06-01')  # 需要前6个月初始化
end_date = pd.Timestamp('2026-04-30')
valid_months = monthly_returns.index[
    (monthly_returns.index >= start_date) & 
    (monthly_returns.index <= end_date)
]

print(f"  回测区间: {valid_months[0].strftime('%Y-%m')} ~ {valid_months[-1].strftime('%Y-%m')}")
print(f"  共{len(valid_months)}个月")

# =========================================================
# Step 5: 景气度评分函数
# =========================================================
def calc_prosperity_scores(date_idx, monthly_ret, all_vol, bench_ret, 
                           use_pe_filter=False, all_pe=None):
    """
    计算某月月末所有行业的景气度评分
    
    返回: Series(行业名 -> 综合评分)
    """
    # 截止到当前的数据
    ret_hist = monthly_ret.iloc[:date_idx + 1]
    vol_hist = all_vol.iloc[:date_idx + 1]
    
    scores = {}
    for name in ret_hist.columns:
        r = ret_hist[name].dropna()
        v = vol_hist[name].dropna() if name in vol_hist.columns else None
        
        if len(r) < 6:
            continue
        
        # 指标1: 3个月动量(市场认可景气)
        mom_3m = r.iloc[-1] / r.iloc[-4] - 1 if len(r) >= 4 else 0
        
        # 指标2: 6个月动量(中期趋势)
        mom_6m = r.iloc[-1] / r.iloc[-7] - 1 if len(r) >= 7 else 0
        
        # 指标3: 动量加速度(3m - 6m/2，景气在加速还是减速)
        accel = mom_3m - mom_6m * 0.5
        
        # 指标4: 量能比率(近3月均量/前3月均量)
        vol_score = 1.0
        if v is not None and len(v) >= 6:
            recent_vol = v.iloc[-3:].mean()
            prev_vol = v.iloc[-6:-3].mean()
            vol_score = recent_vol / prev_vol if prev_vol > 0 else 1.0
        
        # 指标5: 相对沪深300超额(3月)
        br = bench_ret.iloc[:date_idx + 1].dropna()
        rel_str = 0
        if len(r) >= 4 and len(br) >= 4:
            rel_str = mom_3m - (br.iloc[-1] / br.iloc[-4] - 1)
        
        # PE趋势过滤(可选)
        pe_ok = True
        if use_pe_filter and all_pe is not None and name in all_pe.columns:
            pe_hist = all_pe[name].dropna()
            if len(pe_hist) >= 3:
                # PE在快速上升 + 价格没涨 = 利润在下滑，假景气
                pe_change = pe_hist.iloc[-1] / pe_hist.iloc[-3] - 1
                if pe_change > 0.15:  # PE 3个月涨超15%
                    pe_ok = False
        
        scores[name] = {
            'mom_3m': mom_3m,
            'mom_6m': mom_6m,
            'accel': accel,
            'vol_ratio': vol_score,
            'rel_str': rel_str,
            'pe_ok': pe_ok
        }
    
    if len(scores) < 10:
        return pd.Series(dtype=float)
    
    df = pd.DataFrame(scores).T
    
    if use_pe_filter:
        df = df[df['pe_ok']]
    
    if len(df) < 5:
        return pd.Series(dtype=float)
    
    # 百分位排名
    for col in ['mom_3m', 'mom_6m', 'accel', 'vol_ratio', 'rel_str']:
        df[f'rank_{col}'] = df[col].rank(pct=True)
    
    # 综合评分
    df['score'] = (
        0.30 * df['rank_mom_3m'] +
        0.25 * df['rank_mom_6m'] +
        0.15 * df['rank_accel'] +
        0.10 * df['rank_vol_ratio'] +
        0.20 * df['rank_rel_str']
    )
    
    return df['score']


def calc_momentum_scores(date_idx, monthly_ret, all_vol=None, bench_ret=None, **kwargs):
    """纯动量策略: 只用6个月动量排名"""
    ret_hist = monthly_ret.iloc[:date_idx + 1]
    scores = {}
    for name in ret_hist.columns:
        r = ret_hist[name].dropna()
        if len(r) < 7:
            continue
        scores[name] = r.iloc[-1] / r.iloc[-7] - 1
    
    if len(scores) < 10:
        return pd.Series(dtype=float)
    
    return pd.Series(scores).rank(pct=True)


# =========================================================
# Step 6: 回测引擎
# =========================================================
def run_backtest(strategy_name, score_func, valid_months, monthly_returns, 
                 bench_monthly, top_n=5, cost=0.002, delay=0, **kwargs):
    """
    通用回测引擎
    
    delay: 信号延迟月数(0=当月信号当月执行, 1=延迟1个月)
    """
    returns_list = []
    holdings_list = []
    
    for i in range(6, len(valid_months) - delay):  # 跳过前6个月初始化
        signal_date = valid_months[i]
        exec_date = valid_months[i + delay]
        
        if exec_date not in monthly_returns.index:
            continue
        
        # 计算评分
        scores = score_func(i, monthly_returns, all_volumes, bench_monthly, **kwargs)
        
        if len(scores) == 0:
            continue
        
        # 选Top N
        selected = scores.nlargest(top_n).index.tolist()
        
        # 下一月收益
        next_idx = list(valid_months).index(exec_date) + 1
        if next_idx >= len(valid_months):
            break
        
        next_date = valid_months[next_idx]
        if next_date not in monthly_returns.index:
            continue
        
        next_rets = monthly_returns.loc[next_date, selected]
        valid_rets = next_rets.dropna()
        
        if len(valid_rets) == 0:
            continue
        
        # 等权组合收益 - 扣交易成本
        port_ret = valid_rets.mean() - cost * (len(valid_rets) / top_n)
        
        returns_list.append({
            'date': next_date,
            'return': port_ret,
            'selected': selected,
            'n_selected': len(valid_rets)
        })
        
        holdings_list.append({
            'date': exec_date,
            'holdings': selected
        })
    
    if not returns_list:
        return None
    
    # 构建收益序列
    ret_df = pd.DataFrame(returns_list).set_index('date')
    
    # 累计净值
    ret_df['cumulative'] = (1 + ret_df['return']).cumprod()
    
    # 基准对齐
    bench_aligned = bench_monthly.reindex(ret_df.index)
    bench_aligned = bench_aligned.fillna(0)
    ret_df['bench_cum'] = (1 + bench_aligned).cumprod()
    
    # 计算指标
    total_months = len(ret_df)
    total_return = ret_df['cumulative'].iloc[-1] - 1
    bench_total = ret_df['bench_cum'].iloc[-1] - 1
    years = total_months / 12
    
    annual_ret = (1 + total_return) ** (1 / years) - 1
    bench_annual = (1 + bench_total) ** (1 / years) - 1
    
    # 最大回撤
    running_max = ret_df['cumulative'].cummax()
    drawdown = (ret_df['cumulative'] - running_max) / running_max
    max_dd = drawdown.min()
    
    # 波动率 & 夏普
    monthly_std = ret_df['return'].std()
    annual_std = monthly_std * np.sqrt(12)
    sharpe = (annual_ret - 0.03) / annual_std if annual_std > 0 else 0
    bench_sharpe = (bench_annual - 0.03) / (bench_aligned.std() * np.sqrt(12)) if bench_aligned.std() > 0 else 0
    
    # Calmar ratio
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0
    
    # 胜率
    win_rate = (ret_df['return'] > 0).mean()
    
    # 月均换手(新增/移除的行业数)
    turnovers = []
    holdings = holdings_list
    for j in range(1, len(holdings)):
        prev = set(holdings[j-1]['holdings'])
        curr = set(holdings[j]['holdings'])
        turnover = len(prev.symmetric_difference(curr)) / 2
        turnovers.append(turnover)
    avg_turnover = np.mean(turnovers) if turnovers else 0
    
    # 最大单月亏损/盈利
    max_month_loss = ret_df['return'].min()
    max_month_gain = ret_df['return'].max()
    
    return {
        'strategy': strategy_name,
        'total_return': total_return,
        'annual_return': annual_ret,
        'bench_total': bench_total,
        'bench_annual': bench_annual,
        'excess_return': annual_ret - bench_annual,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'bench_sharpe': bench_sharpe,
        'calmar': calmar,
        'annual_std': annual_std,
        'win_rate': win_rate,
        'total_months': total_months,
        'avg_turnover': avg_turnover,
        'max_month_loss': max_month_loss,
        'max_month_gain': max_month_gain,
        'ret_df': ret_df
    }


# =========================================================
# Step 7: 运行所有策略
# =========================================================
print("\nStep 5: 运行回测...")
print("-" * 60)

strategies = []

# A: 纯6个月动量轮动
print("  A: 纯6月动量轮动...")
r = run_backtest("A: 纯6月动量轮动", calc_momentum_scores, 
                 valid_months, monthly_returns, bench_monthly, top_n=5)
if r: strategies.append(r)

# B: 景气度综合评分
print("  B: 景气度综合评分...")
r = run_backtest("B: 景气度综合评分", calc_prosperity_scores,
                 valid_months, monthly_returns, bench_monthly, top_n=5)
if r: strategies.append(r)

# C: 景气度 + Top3(更集中)
print("  C: 景气度综合评分(Top3)...")
r = run_backtest("C: 景气度Top3(集中)", calc_prosperity_scores,
                 valid_months, monthly_returns, bench_monthly, top_n=3, cost=0.002)
if r: strategies.append(r)

# D: 景气度 + 延迟1月确认(更保守)
print("  D: 景气度 + 延迟1月确认...")
r = run_backtest("D: 景气度+延迟确认", calc_prosperity_scores,
                 valid_months, monthly_returns, bench_monthly, top_n=5, delay=1)
if r: strategies.append(r)

# E: 景气度 + 零交易成本(理想值)
print("  E: 景气度(零成本理想值)...")
r = run_backtest("E: 景气度(零成本)", calc_prosperity_scores,
                 valid_months, monthly_returns, bench_monthly, top_n=5, cost=0)
if r: strategies.append(r)

# F: 景气度 + 高换手成本(0.3%单边)
print("  F: 景气度(高成本0.3%)...")
r = run_backtest("F: 景气度(高成本)", calc_prosperity_scores,
                 valid_months, monthly_returns, bench_monthly, top_n=5, cost=0.006)
if r: strategies.append(r)

# G: 沪深300等权基准(额外对比)
print("  基准: 沪深300...")

# =========================================================
# Step 8: 输出结果
# =========================================================
print("\n" + "=" * 60)
print("回测结果")
print("=" * 60)

# 结果表格
header = f"{'策略':<22} {'年化':>7} {'夏普':>6} {'最大回撤':>8} {'Calmar':>7} {'超额':>7} {'胜率':>6} {'换手':>5}"
print(header)
print("-" * len(header))

for s in strategies:
    line = (f"{s['strategy']:<22} "
            f"{s['annual_return']*100:>6.1f}% "
            f"{s['sharpe']:>6.2f} "
            f"{s['max_drawdown']*100:>7.1f}% "
            f"{s['calmar']:>7.2f} "
            f"{s['excess_return']*100:>6.1f}% "
            f"{s['win_rate']*100:>5.0f}% "
            f"{s['avg_turnover']:>5.1f}")
    print(line)

# 基准行
bench_s = strategies[0]  # use first strategy's benchmark
print("-" * len(header))
bline = (f"{'基准:沪深300':<22} "
         f"{bench_s['bench_annual']*100:>6.1f}% "
         f"{bench_s['bench_sharpe']:>6.2f} "
         f"{'--':>8} "
         f"{'--':>7} "
         f"{'--':>7} "
         f"{'--':>6} "
         f"{'--':>5}")
print(bline)

print(f"\n回测区间: {strategies[0]['ret_df'].index[0].strftime('%Y-%m')} ~ "
      f"{strategies[0]['ret_df'].index[-1].strftime('%Y-%m')}")
print(f"共 {strategies[0]['total_months']} 个月 ({strategies[0]['total_months']/12:.1f}年)")

# =========================================================
# Step 9: 分年度收益
# =========================================================
print("\n" + "=" * 60)
print("策略B(景气度综合) 分年度收益")
print("=" * 60)

# 找策略B
b_data = None
for s in strategies:
    if 'B:' in s['strategy']:
        b_data = s['ret_df']
        break

if b_data is not None:
    b_data['year'] = b_data.index.year
    bench_aligned = b_data['bench_cum']
    
    yearly = b_data.groupby('year').agg(
        port_ret=('return', lambda x: (1 + x).prod() - 1),
        bench_ret=('return', lambda x: 0),  # placeholder
        months=('return', 'count')
    )
    
    # 基准年度收益
    bench_rets_by_year = {}
    for year in yearly.index:
        mask = bench_monthly.index.year == year
        mask = mask & (bench_monthly.index >= b_data.index[0]) & (bench_monthly.index <= b_data.index[-1])
        if mask.sum() > 0:
            bench_rets_by_year[year] = (1 + bench_monthly[mask]).prod() - 1
        else:
            bench_rets_by_year[year] = 0.0
    
    print(f"{'年份':>6} {'策略收益':>10} {'基准收益':>10} {'超额':>8} {'月数':>4}")
    print("-" * 42)
    for year, row in yearly.iterrows():
        bench_r = bench_rets_by_year.get(year, 0)
        excess = row['port_ret'] - bench_r
        print(f"{year:>6} {row['port_ret']*100:>9.1f}% {bench_r*100:>9.1f}% {excess*100:>7.1f}% {int(row['months']):>4}")

# =========================================================
# Step 10: 行业入选频次统计
# =========================================================
print("\n" + "=" * 60)
print("行业入选Top5频次(策略B)")
print("=" * 60)

if b_data is not None:
    all_selected = []
    for sel in b_data['selected']:
        all_selected.extend(sel)
    
    from collections import Counter
    freq = Counter(all_selected)
    total = len(b_data)
    
    # 按频次排序
    for name, count in freq.most_common():
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {name:<8} {count:>3}次 ({pct:>5.1f}%) {bar}")

# =========================================================
# Step 11: 最近一期持仓
# =========================================================
print("\n" + "=" * 60)
print("最近一期信号(策略B)")
print("=" * 60)

if b_data is not None:
    last = b_data.iloc[-1]
    print(f"  信号日期: {b_data.index[-1].strftime('%Y-%m')}")
    print(f"  入选行业: {last['selected']}")
    print(f"  当月收益: {last['return']*100:+.2f}%")

print("\n完成。")
