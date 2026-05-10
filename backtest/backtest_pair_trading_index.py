#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股指数配对交易回测
基于Engle-Granger两步法协整检验 + Z-score触发
数据源：AKShare stock_zh_index_daily
"""

import akshare as ak
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def EG_cointegration_test(X, Y, significance_level=0.05):
    """Engle-Granger两步法检验协整关系"""
    X_const = sm.add_constant(X)
    model = sm.OLS(Y, X_const).fit()
    epsilon = model.resid
    
    adf_result = adfuller(epsilon)
    p_value = adf_result[1]
    is_cointegrated = p_value < significance_level
    hedge_ratio = model.params[1] if len(model.params) > 1 else 1.0
    
    return is_cointegrated, hedge_ratio, model, p_value

def pair_trading_backtest_index(symbol1, symbol2, name1, name2,
                                start_date='20200101', end_date=None,
                                window=250, z_upper=1.0, z_lower=-1.0,
                                transaction_cost=0.001):
    """指数配对交易回测"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    print(f"\n{'='*60}")
    print(f"配对交易回测: {name1}({symbol1}) vs {name2}({symbol2})")
    print(f"回测区间: {start_date} ~ {end_date}")
    print(f"窗口: {window}天, Z阈值: [{z_lower}, {z_upper}]")
    print(f"{'='*60}\n")
    
    # 获取指数数据
    print("获取指数历史数据...")
    try:
        df1 = ak.stock_zh_index_daily(symbol=symbol1)
        df2 = ak.stock_zh_index_daily(symbol=symbol2)
    except Exception as e:
        print(f"数据获取失败: {e}")
        return None
    
    # 处理数据
    df1['date'] = pd.to_datetime(df1['date'])
    df2['date'] = pd.to_datetime(df2['date'])
    df1 = df1.set_index('date')[['close']]
    df2 = df2.set_index('date')[['close']]
    df1.columns = ['price1']
    df2.columns = ['price2']
    
    # 过滤日期
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df1 = df1[(df1.index >= start_dt) & (df1.index <= end_dt)]
    df2 = df2[(df2.index >= start_dt) & (df2.index <= end_dt)]
    
    # 合并数据
    data = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
    
    if len(data) < window:
        print(f"数据不足: {len(data)}天 < {window}天")
        return None
    
    print(f"有效数据: {len(data)}天\n")
    
    # 初始化信号
    signals = pd.DataFrame(index=data.index)
    signals['price1'] = data['price1']
    signals['price2'] = data['price2']
    signals['position1'] = 0
    signals['position2'] = 0
    signals['z_score'] = np.nan
    signals['hedge_ratio'] = np.nan
    signals['spread'] = np.nan
    
    current_position1 = 0
    current_position2 = 0
    
    # 滚动窗口回测
    print("执行协整检验与回测...")
    cointegration_count = 0
    total_days = len(signals) - window
    
    for i in range(window, len(signals)):
        window_data1 = signals['price1'].iloc[i-window:i]
        window_data2 = signals['price2'].iloc[i-window:i]
        
        # 协整检验
        is_coint, hedge_ratio, model, p_val = EG_cointegration_test(
            window_data1.values, window_data2.values, significance_level=0.05
        )
        
        if is_coint:
            cointegration_count += 1
            signals.at[signals.index[i], 'hedge_ratio'] = hedge_ratio
            
            spread = (signals.at[signals.index[i], 'price2'] -
                     hedge_ratio * signals.at[signals.index[i], 'price1'])
            signals.at[signals.index[i], 'spread'] = spread
            
            spread_mean = model.resid.mean()
            spread_std = model.resid.std()
            
            if spread_std > 0:
                z_score = (spread - spread_mean) / spread_std
                signals.at[signals.index[i], 'z_score'] = z_score
                
                # 交易信号
                if z_score > z_upper and current_position1 != 1:
                    signals.at[signals.index[i], 'position1'] = 1
                    signals.at[signals.index[i], 'position2'] = -1
                    current_position1 = 1
                    current_position2 = -1
                elif z_score < z_lower and current_position1 != -1:
                    signals.at[signals.index[i], 'position1'] = -1
                    signals.at[signals.index[i], 'position2'] = 1
                    current_position1 = -1
                    current_position2 = 1
                elif abs(z_score) < 0.5 and current_position1 != 0:
                    signals.at[signals.index[i], 'position1'] = 0
                    signals.at[signals.index[i], 'position2'] = 0
                    current_position1 = 0
                    current_position2 = 0
                else:
                    signals.at[signals.index[i], 'position1'] = current_position1
                    signals.at[signals.index[i], 'position2'] = current_position2
        else:
            # 协整破裂，平仓
            if current_position1 != 0:
                signals.at[signals.index[i], 'position1'] = 0
                signals.at[signals.index[i], 'position2'] = 0
                current_position1 = 0
                current_position2 = 0
    
    # 填充持仓
    signals['position1'] = signals['position1'].replace(0, np.nan).ffill().fillna(0)
    signals['position2'] = signals['position2'].replace(0, np.nan).ffill().fillna(0)
    
    # 计算收益
    signals['return1'] = signals['price1'].pct_change()
    signals['return2'] = signals['price2'].pct_change()
    signals['pair_return'] = (signals['position1'].shift(1) * signals['return1'] +
                             signals['position2'].shift(1) * signals['return2'])
    
    # 交易成本
    signals['trade_count'] = (signals['position1'].diff().abs()).fillna(0)
    signals['cost'] = signals['trade_count'] * transaction_cost
    signals['net_return'] = signals['pair_return'] - signals['cost']
    signals['cumulative_return'] = (1 + signals['net_return']).cumprod()
    
    # 统计指标
    total_trades = int(signals['trade_count'].sum() / 2)
    final_return = signals['cumulative_return'].iloc[-1] - 1
    annual_return = (1 + final_return) ** (252 / len(signals)) - 1
    sharpe = np.sqrt(252) * signals['net_return'].mean() / signals['net_return'].std()
    max_drawdown = (signals['cumulative_return'] / signals['cumulative_return'].cummax() - 1).min()
    cointegration_rate = cointegration_count / total_days * 100 if total_days > 0 else 0
    
    print(f"\n回测结果:")
    print(f"  总收益率: {final_return*100:.2f}%")
    print(f"  年化收益: {annual_return*100:.2f}%")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  最大回撤: {max_drawdown*100:.2f}%")
    print(f"  交易次数: {total_trades}")
    print(f"  协整关系存在天数: {cointegration_count} ({cointegration_rate:.1f}%)")
    
    return {
        'signals': signals,
        'final_return': final_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'cointegration_rate': cointegration_rate
    }

if __name__ == '__main__':
    print("\n" + "="*80)
    print("A股指数配对交易回测")
    print("="*80)
    
    # 指数配对（宽基 vs 风格）
    pairs = [
        ('sh000300', 'sz399303', '沪深300', '国证2000'),  # 大盘 vs 小盘
        ('sh000300', 'sh000905', '沪深300', '中证500'),  # 大盘 vs 中盘
        ('sz399006', 'sz399303', '创业板指', '国证2000'),  # 成长 vs 小盘
    ]
    
    results = {}
    for s1, s2, n1, n2 in pairs:
        try:
            result = pair_trading_backtest_index(s1, s2, n1, n2, start_date='20200101', window=250)
            if result is not None:
                results[f"{n1} vs {n2}"] = result
        except Exception as e:
            print(f"\n{n1} vs {n2} 回测失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("配对交易回测总结")
    print("="*80)
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  年化收益: {res['annual_return']*100:.2f}%")
        print(f"  夏普: {res['sharpe']:.2f}")
        print(f"  回撤: {res['max_drawdown']*100:.2f}%")
        print(f"  交易: {res['total_trades']}次")
        print(f"  协整率: {res['cointegration_rate']:.1f}%")
