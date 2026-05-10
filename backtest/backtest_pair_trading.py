#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股配对交易回测
基于Engle-Granger两步法协整检验 + Z-score触发
"""

import akshare as ak
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 安全调用AKShare
def safe_ak_call(func, *args, max_retries=3, delay=5, **kwargs):
    """安全调用AKShare API，带重试"""
    import time
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  API调用失败 (尝试 {attempt+1}/{max_retries}): {str(e)[:100]}")
                time.sleep(delay)
            else:
                raise

# Engle-Granger两步法协整检验
def EG_cointegration_test(X, Y, significance_level=0.05):
    """
    Engle-Granger两步法检验协整关系
    
    Returns:
        (is_cointegrated, hedge_ratio, model, p_value)
    """
    # Step 1: 长期均衡回归 Y = alpha + beta*X + epsilon
    X_const = sm.add_constant(X)
    model = sm.OLS(Y, X_const).fit()
    epsilon = model.resid
    
    # 检验残差平稳性 (ADF检验)
    adf_result = adfuller(epsilon)
    p_value = adf_result[1]
    
    # 如果p值小于显著性水平，残差平稳→存在协整关系
    is_cointegrated = p_value < significance_level
    
    # hedge_ratio就是beta系数
    hedge_ratio = model.params[1] if len(model.params) > 1 else 1.0
    
    return is_cointegrated, hedge_ratio, model, p_value

# 配对交易回测
def pair_trading_backtest(symbol1, symbol2, start_date='20200101', end_date=None,
                          window=120, z_upper=1.0, z_lower=-1.0,
                          transaction_cost=0.001):
    """
    配对交易回测
    
    Args:
        symbol1, symbol2: 股票代码 (如 '600519', '000858')
        window: 滚动窗口大小 (天)
        z_upper, z_lower: Z-score上下界触发阈值
        transaction_cost: 单边交易成本
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    print(f"\n{'='*60}")
    print(f"配对交易回测: {symbol1} vs {symbol2}")
    print(f"回测区间: {start_date} ~ {end_date}")
    print(f"窗口: {window}天, Z阈值: [{z_lower}, {z_upper}]")
    print(f"{'='*60}\n")
    
    # 获取数据
    print("获取历史数据...")
    try:
        df1 = safe_ak_call(ak.stock_zh_a_hist, symbol=symbol1, period="daily",
                          start_date=start_date, end_date=end_date, adjust="qfq")
        df2 = safe_ak_call(ak.stock_zh_a_hist, symbol=symbol2, period="daily",
                          start_date=start_date, end_date=end_date, adjust="qfq")
    except Exception as e:
        print(f"数据获取失败: {e}")
        return None
    
    # 统一日期索引
    df1 = df1.set_index('日期')[['收盘']]
    df2 = df2.set_index('日期')[['收盘']]
    df1.columns = ['price1']
    df2.columns = ['price2']
    
    # 合并数据
    data = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
    
    if len(data) < window:
        print(f"数据不足: {len(data)}天 < {window}天")
        return None
    
    print(f"有效数据: {len(data)}天\n")
    
    # 初始化
    signals = pd.DataFrame(index=data.index)
    signals['price1'] = data['price1']
    signals['price2'] = data['price2']
    signals['position1'] = 0  # 1=做多, -1=做空, 0=空仓
    signals['position2'] = 0
    signals['z_score'] = np.nan
    signals['hedge_ratio'] = np.nan
    signals['spread'] = np.nan
    signals['cointegrated'] = False
    
    current_position1 = 0
    current_position2 = 0
    
    # 信号生成 (滚动窗口)
    print("执行回测...")
    cointegration_count = 0
    total_days = len(signals) - window
    
    for i in range(window, len(signals)):
        # 滚动窗口数据
        window_data1 = signals['price1'].iloc[i-window:i]
        window_data2 = signals['price2'].iloc[i-window:i]
        
        # 协整检验
        is_coint, hedge_ratio, model, p_val = EG_cointegration_test(
            window_data1.values, window_data2.values, significance_level=0.05
        )
        
        if is_coint:
            cointegration_count += 1
            signals.at[signals.index[i], 'cointegrated'] = True
            signals.at[signals.index[i], 'hedge_ratio'] = hedge_ratio
            
            # 计算价差 (spread = price2 - hedge_ratio * price1)
            spread = (signals.at[signals.index[i], 'price2'] -
                     hedge_ratio * signals.at[signals.index[i], 'price1'])
            signals.at[signals.index[i], 'spread'] = spread
            
            # Z-score标准化 (用窗口内spread的均值和标准差)
            spread_mean = model.resid.mean()
            spread_std = model.resid.std()
            
            if spread_std > 0:
                z_score = (spread - spread_mean) / spread_std
                signals.at[signals.index[i], 'z_score'] = z_score
                
                # 交易逻辑
                # Z-score > 上界 → 做空价差 (做空asset2, 做多asset1)
                # Z-score < 下界 → 做多价差 (做多asset2, 做空asset1)
                # |Z-score| < 0.5 → 平仓
                
                if z_score > z_upper and current_position1 != 1:
                    # 做空价差：做多asset1, 做空asset2
                    signals.at[signals.index[i], 'position1'] = 1
                    signals.at[signals.index[i], 'position2'] = -1
                    current_position1 = 1
                    current_position2 = -1
                    
                elif z_score < z_lower and current_position1 != -1:
                    # 做多价差：做空asset1, 做多asset2
                    signals.at[signals.index[i], 'position1'] = -1
                    signals.at[signals.index[i], 'position2'] = 1
                    current_position1 = -1
                    current_position2 = 1
                    
                elif abs(z_score) < 0.5 and current_position1 != 0:
                    # 平仓
                    signals.at[signals.index[i], 'position1'] = 0
                    signals.at[signals.index[i], 'position2'] = 0
                    current_position1 = 0
                    current_position2 = 0
                else:
                    # 保持持仓
                    signals.at[signals.index[i], 'position1'] = current_position1
                    signals.at[signals.index[i], 'position2'] = current_position2
        else:
            # 协整破裂，平仓
            if current_position1 != 0:
                signals.at[signals.index[i], 'position1'] = 0
                signals.at[signals.index[i], 'position2'] = 0
                current_position1 = 0
                current_position2 = 0
    
    # 填充持仓向前
    signals['position1'] = signals['position1'].replace(0, np.nan).ffill().fillna(0)
    signals['position2'] = signals['position2'].replace(0, np.nan).ffill().fillna(0)
    
    # 计算收益
    signals['return1'] = signals['price1'].pct_change()
    signals['return2'] = signals['price2'].pct_change()
    
    # 配对收益
    signals['pair_return'] = (signals['position1'].shift(1) * signals['return1'] +
                             signals['position2'].shift(1) * signals['return2'])
    
    # 扣除交易成本
    signals['trade_count'] = (signals['position1'].diff().abs()).fillna(0)
    signals['cost'] = signals['trade_count'] * transaction_cost
    signals['net_return'] = signals['pair_return'] - signals['cost']
    
    # 累计收益
    signals['cumulative_return'] = (1 + signals['net_return']).cumprod()
    
    # 统计
    total_trades = int(signals['trade_count'].sum() / 2)
    final_return = signals['cumulative_return'].iloc[-1] - 1
    annual_return = (1 + final_return) ** (252 / len(signals)) - 1
    sharpe = np.sqrt(252) * signals['net_return'].mean() / signals['net_return'].std()
    max_drawdown = (signals['cumulative_return'] / signals['cumulative_return'].cummax() - 1).min()
    
    cointegration_rate = cointegration_count / total_days * 100 if total_days > 0 else 0
    
    # 输出结果
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

# 主程序
if __name__ == '__main__':
    print("\n" + "="*80)
    print("A股配对交易回测")
    print("="*80)
    
    pairs = [
        ('600519', '000858', '茅台 vs 五粮液'),
        ('000333', '600690', '美的 vs 海尔'),
        ('600036', '000001', '招行 vs 平安银行'),
    ]
    
    results = {}
    for s1, s2, name in pairs:
        try:
            result = pair_trading_backtest(s1, s2, start_date='20200101', window=120)
            if result is not None:
                results[name] = result
        except Exception as e:
            print(f"\n{name} 回测失败: {e}")
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
