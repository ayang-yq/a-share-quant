"""
纯高股息基本面策略回测 v4
优化：先做纯财务筛选大幅减少需要查价格的股票数

数据源：
- akshare: 全市场分红 + 季报(2012-2020)
- baostock: 价格 + 财务(2021-2025补丁)

策略：3年股息率前10% → PEG 0.1-2 → 营收增>3%/净利增>8%/ROE>4.5% → 股息率排序取5 → 季调仓
"""

import baostock as bs
import pandas as pd
import numpy as np
import time
import warnings
from datetime import datetime, timedelta
import akshare as ak

warnings.filterwarnings('ignore')

# 全局socket timeout（防止baostock/requests hang）
import socket
socket.setdefaulttimeout(10)

# 全局requests timeout
import requests
old_send = requests.Session.send
def _timeout_send(self, *args, **kwargs):
    kwargs.setdefault('timeout', 15)
    return old_send(self, *args, **kwargs)
requests.Session.send = _timeout_send

COST_RATE = 0.001
HOLD_NUM = 5
START_DATE = '2013-01-01'
END_DATE = '2025-05-09'

# 价格缓存（避免重复查同一股票同一日期）
_price_cache = {}

def get_stock_price_bs(code, date, days_back=5):
    """baostock获取某日收盘价（带缓存）"""
    key = (code, date)
    if key in _price_cache:
        return _price_cache[key]
    
    rs = bs.query_history_k_data_plus(
        code, "date,close",
        start_date=date, end_date=date,
        frequency="d", adjustflag="2"
    )
    data = []
    while rs.next():
        data.append(rs.get_row_data())
    if data and data[0][1]:
        price = float(data[0][1])
        _price_cache[key] = price
        return price
    # 往前找最多5天
    dt = datetime.strptime(date, '%Y-%m-%d')
    for i in range(1, days_back):
        dt2 = (dt - timedelta(days=i)).strftime('%Y-%m-%d')
        k2 = (code, dt2)
        if k2 in _price_cache:
            price = _price_cache[k2]
            _price_cache[key] = price
            return price
        rs = bs.query_history_k_data_plus(
            code, "date,close",
            start_date=dt2, end_date=dt2,
            frequency="d", adjustflag="2"
        )
        data = []
        while rs.next():
            data.append(rs.get_row_data())
        if data and data[0][1]:
            price = float(data[0][1])
            _price_cache[key] = price
            _price_cache[k2] = price
            return price
    _price_cache[key] = None
    return None

def get_stock_range_return(code, start_date, end_date):
    """计算区间收益率（一次查询）"""
    rs = bs.query_history_k_data_plus(
        code, "date,close",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="2"
    )
    data = []
    while rs.next():
        data.append(rs.get_row_data())
    valid = [float(d[1]) for d in data if d[1] and float(d[1]) > 0]
    if len(valid) < 2:
        return 0.0
    return (valid[-1] - valid[0]) / valid[0]

def get_index_nav(index_code, start_date, end_date):
    rs = bs.query_history_k_data_plus(
        index_code, "date,close",
        start_date=start_date, end_date=end_date,
        frequency="d"
    )
    data = []
    while rs.next():
        data.append(rs.get_row_data())
    df = pd.DataFrame(data, columns=['date', 'close'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    return df.dropna()

def run_backtest():
    print("=" * 60)
    print("纯高股息基本面策略回测 v4")
    print(f"期间: {START_DATE} 至 {END_DATE}")
    print(f"持仓: {HOLD_NUM}, 成本: {COST_RATE*100}%")
    print("=" * 60)
    
    lg = bs.login()
    print(f"baostock login: {lg.error_msg}")
    
    # [1] 批量获取全市场分红数据
    print("\n[1] 批量获取年度分红数据(akshare)...")
    div_cache = {}
    for year in range(2010, 2026):
        try:
            df = ak.stock_fhps_em(date=f"{year}1231")
            df = df[df['方案进度'] == '实施分配']
            df['现金分红-现金分红比例'] = pd.to_numeric(df['现金分红-现金分红比例'], errors='coerce')
            div_agg = df.groupby('代码').agg({
                '现金分红-现金分红比例': 'sum',
                '名称': 'first'
            }).reset_index()
            div_cache[year] = div_agg
            print(f"  {year}: {len(div_agg)} 只分红股", flush=True)
        except Exception as e:
            print(f"  {year}: 失败 - {str(e)[:60]}", flush=True)
            div_cache[year] = pd.DataFrame()
        time.sleep(0.3)
    
    # [2] 批量获取季报数据(akshare, 2012-2020)
    print("\n[2] 批量获取季报数据(akshare, 2012-2020)...")
    fin_cache = {}
    report_dates = []
    for y in range(2012, 2026):
        report_dates.append(f"{y}1231")
        report_dates.append(f"{y}0930")
    
    for rd in report_dates:
        if rd > '20201231':
            fin_cache[rd] = pd.DataFrame()
            continue
        try:
            df = ak.stock_yjbb_em(date=rd)
            df = df[['股票代码', '股票简称', '营业总收入-同比增长', 
                    '净利润-同比增长', '净资产收益率', '每股收益']].copy()
            for col in ['营业总收入-同比增长', '净利润-同比增长', '净资产收益率', '每股收益']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            fin_cache[rd] = df
            print(f"  {rd}: {len(df)} 条", flush=True)
        except Exception as e:
            print(f"  {rd}: 失败 - {str(e)[:40]}", flush=True)
            fin_cache[rd] = pd.DataFrame()
        time.sleep(0.3)
    
    # [3] 调仓模拟
    print("\n[3] 开始调仓模拟...")
    
    rebalance_schedule = []
    for y in range(2013, 2026):
        rebalance_schedule.append({
            'date': f'{y}-05-06',
            'report_key': f'{y-1}1231',
            'ref_year': y - 1,
            'quarter': 4,
            'label': f'{y}05(年报)'
        })
        rebalance_schedule.append({
            'date': f'{y}-11-04',
            'report_key': f'{y}0930',
            'ref_year': y,
            'quarter': 3,
            'label': f'{y}11(三季报)'
        })
    
    nav = 1.0
    nav_history = [{'date': START_DATE, 'nav': nav}]
    current_holdings = []
    all_holdings_log = []
    
    for i, rebal in enumerate(rebalance_schedule):
        rdate = rebal['date']
        if rdate > END_DATE:
            break
        
        t0 = time.time()
        rkey = rebal['report_key']
        ref_year = rebal['ref_year']
        quarter = rebal['quarter']
        label = rebal['label']
        
        # --- 计算当前持仓收益 ---
        if current_holdings:
            rets = []
            for h in current_holdings:
                r = get_stock_range_return(h['code'], h['buy_date'], rdate)
                rets.append(r)
            if rets:
                nav *= (1 + np.mean(rets))
            nav *= (1 - COST_RATE)  # 换仓成本
        
        # --- 获取该期财务数据 ---
        fin_df = fin_cache.get(rkey, pd.DataFrame())
        has_akshare_fin = not fin_df.empty
        
        # --- 构建候选池 ---
        # Step A: 3年分红筛选
        div_years = [ref_year - 2, ref_year - 1, ref_year]
        stock_div = {}
        for dy in div_years:
            if dy not in div_cache or div_cache[dy].empty:
                continue
            dydf = div_cache[dy]
            for _, row in dydf.iterrows():
                code = row['代码']
                cash = row['现金分红-现金分红比例']
                if not np.isnan(cash) and cash > 0:
                    if code not in stock_div:
                        stock_div[code] = {'total': 0, 'years': 0, 'name': row.get('名称', '')}
                    stock_div[code]['total'] += cash
                    stock_div[code]['years'] += 1
        
        # 至少2年分红
        div_stocks = {k: v for k, v in stock_div.items() if v['years'] >= 2}
        if not div_stocks:
            print(f"  {label}: 无3年分红股", flush=True)
            continue
        
        # 按总分红排序取前10%
        div_sorted = sorted(div_stocks.items(), key=lambda x: x[1]['total'], reverse=True)
        top_n = max(1, len(div_sorted) // 10)
        div_top = div_sorted[:top_n]
        
        # Step B: 财务筛选（先不查价格）
        pre_screened = []  # (code, bs_code, dinfo, financials)
        
        if has_akshare_fin:
            for code, dinfo in div_top:
                if code.startswith('6'):
                    bs_code = f'sh.{code}'
                else:
                    bs_code = f'sz.{code}'
                
                fin_match = fin_df[fin_df['股票代码'] == code]
                if fin_match.empty:
                    continue
                
                row = fin_match.iloc[0]
                rev_g = row['营业总收入-同比增长']
                profit_g = row['净利润-同比增长']
                roe = row['净资产收益率']
                eps = row['每股收益']
                
                if pd.isna(profit_g) or profit_g <= 8: continue
                if pd.isna(roe) or roe <= 4.5: continue
                if not pd.isna(rev_g) and rev_g <= 3: continue
                if pd.isna(eps) or eps <= 0: continue
                
                pre_screened.append((code, bs_code, dinfo, {
                    'roe': roe, 'eps': eps, 'profit_g': profit_g, 'rev_g': rev_g
                }))
        else:
            # baostock逐股查财务
            for code, dinfo in div_top:
                if code.startswith('6'):
                    bs_code = f'sh.{code}'
                else:
                    bs_code = f'sz.{code}'
                
                rs1 = bs.query_profit_data(code=bs_code, year=ref_year, quarter=quarter)
                d1 = []
                while rs1.next(): d1.append(rs1.get_row_data())
                if not d1: continue
                try:
                    roe = float(d1[0][3]) * 100 if d1[0][3] else None
                    eps = float(d1[0][7]) if d1[0][7] else None
                except: continue
                if roe is None or roe <= 4.5: continue
                if eps is None or eps <= 0: continue
                
                rs2 = bs.query_growth_data(code=bs_code, year=ref_year, quarter=quarter)
                d2 = []
                while rs2.next(): d2.append(rs2.get_row_data())
                if not d2: continue
                try:
                    profit_g = float(d2[0][5]) * 100 if d2[0][5] else None
                except: continue
                if profit_g is None or profit_g <= 8: continue
                
                pre_screened.append((code, bs_code, dinfo, {
                    'roe': roe, 'eps': eps, 'profit_g': profit_g, 'rev_g': None
                }))
        
        # Step C: 只对财务通过的股票查价格（大幅减少API调用）
        candidates = []
        for code, bs_code, dinfo, fin in pre_screened:
            price = get_stock_price_bs(bs_code, rdate)
            if not price:
                continue
            
            pe = price / fin['eps']
            if quarter == 3:
                pe = pe * 3 / 4
            
            peg = pe / fin['profit_g'] if fin['profit_g'] > 0 else None
            if peg is None or peg < 0.1 or peg > 2:
                continue
            
            annual_dps = dinfo['total'] / 3 / 10
            div_yield = annual_dps / price
            
            candidates.append({
                'code': bs_code,
                'num_code': code,
                'name': dinfo['name'],
                'div_yield': div_yield,
                'pe': pe,
                'peg': peg,
                'roe': fin['roe'],
                'profit_growth': fin['profit_g'],
                'rev_growth': fin['rev_g'],
                'price': price,
            })
        
        if not candidates:
            print(f"  {label}: 无候选 (分红top{len(div_top)}, 财务通过{len(pre_screened)}, ak={'Y' if has_akshare_fin else 'N'}) [{time.time()-t0:.0f}s]", flush=True)
            continue
        
        # 按股息率排序取前5
        cand_df = pd.DataFrame(candidates)
        cand_df = cand_df.sort_values('div_yield', ascending=False)
        selected = cand_df.head(HOLD_NUM)
        
        current_holdings = []
        for _, row in selected.iterrows():
            current_holdings.append({
                'code': row['code'],
                'buy_price': row['price'],
                'buy_date': rdate,
            })
        
        nav_history.append({'date': rdate, 'nav': nav})
        
        names = ', '.join([f"{r['name']}(P{r['peg']:.1f},Y{r['div_yield']*100:.1f}%)" 
                          for _, r in selected.iterrows()])
        data_src = 'AK' if has_akshare_fin else 'BS'
        elapsed = time.time() - t0
        print(f"  {label} [{data_src}]: NAV={nav:.3f} | 分红{len(div_top)}→财务{len(pre_screened)}→候选{len(candidates)}→选{len(selected)} | {names} [{elapsed:.0f}s]", flush=True)
        
        all_holdings_log.append({
            'date': rdate, 'nav': nav, 'data_src': data_src,
            'stocks': names,
        })
    
    # 最后一期收益
    if current_holdings:
        rets = []
        for h in current_holdings:
            r = get_stock_range_return(h['code'], h['buy_date'], END_DATE)
            rets.append(r)
        if rets:
            nav *= (1 + np.mean(rets))
        nav_history.append({'date': END_DATE, 'nav': nav})
    
    # [4] 统计
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    
    total_days = (datetime.strptime(END_DATE, '%Y-%m-%d') - 
                  datetime.strptime(START_DATE, '%Y-%m-%d')).days
    total_years = total_days / 365.25
    total_return = nav - 1
    annual_return = (nav ** (1/total_years) - 1) * 100
    
    print(f"期间: {START_DATE} 至 {END_DATE} ({total_years:.1f}年)")
    print(f"最终净值: {nav:.4f}")
    print(f"总收益: {total_return*100:.1f}%")
    print(f"年化收益: {annual_return:.1f}%")
    
    # 最大回撤
    navs = [n['nav'] for n in nav_history]
    peak = 1.0
    max_dd = 0
    dd_peak_date = ''
    dd_trough_date = ''
    for n_rec in nav_history:
        n = n_rec['nav']
        if n > peak: 
            peak = n
            dd_peak_date = n_rec['date']
        dd = (n - peak) / peak
        if dd < max_dd: 
            max_dd = dd
            dd_trough_date = n_rec['date']
    print(f"最大回撤: {max_dd*100:.1f}% ({dd_peak_date}→{dd_trough_date})")
    
    # 基准
    hs300 = get_index_nav("sh.000300", START_DATE, END_DATE)
    if not hs300.empty:
        hs300_ret = hs300['close'].iloc[-1] / hs300['close'].iloc[0]
        hs300_annual = (hs300_ret ** (1/total_years) - 1) * 100
        print(f"\n沪深300: 年化{hs300_annual:.1f}%")
        print(f"超额: 年化{annual_return - hs300_annual:.1f}%")
    
    if max_dd < 0 and annual_return > 0:
        calmar = annual_return / abs(max_dd * 100)
        print(f"Calmar: {calmar:.2f}")
    
    # 持仓统计
    print(f"\n调仓次数: {len(all_holdings_log)}")
    ak_count = sum(1 for h in all_holdings_log if h['data_src'] == 'AK')
    bs_count = sum(1 for h in all_holdings_log if h['data_src'] == 'BS')
    print(f"数据源: AK={ak_count}期, BS={bs_count}期")
    
    # 年度收益
    print("\n年度收益:")
    for j in range(len(nav_history)-1):
        n0 = nav_history[j]
        n1 = nav_history[j+1]
        yr_ret = (n1['nav'] / n0['nav'] - 1) * 100
        print(f"  {n0['date']}→{n1['date']}: {yr_ret:+.1f}%")
    
    # 保存
    pd.DataFrame(nav_history).to_csv('/home/andy/backtest/dougua_results/pure_dividend_v4_nav.csv', index=False)
    
    with open('/home/andy/backtest/dougua_results/pure_dividend_v4_result.txt', 'w') as f:
        f.write(f"纯高股息基本面策略回测 v4\n")
        f.write(f"期间: {START_DATE} 至 {END_DATE} ({total_years:.1f}年)\n")
        f.write(f"持仓: {HOLD_NUM}, 成本: {COST_RATE*100}%\n")
        f.write(f"年化收益: {annual_return:.1f}%\n")
        f.write(f"总收益: {total_return*100:.1f}%\n")
        f.write(f"最大回撤: {max_dd*100:.1f}%\n")
        if not hs300.empty:
            f.write(f"沪深300年化: {hs300_annual:.1f}%\n")
            f.write(f"超额: 年化{annual_return - hs300_annual:.1f}%\n")
        if max_dd < 0 and annual_return > 0:
            f.write(f"Calmar: {calmar:.2f}\n")
        f.write(f"\n调仓记录:\n")
        for h in all_holdings_log:
            f.write(f"  {h['date']} [{h['data_src']}] NAV={h['nav']:.3f}: {h['stocks']}\n")
    
    print("\n结果已保存到 pure_dividend_v4_*")
    bs.logout()
    return nav_history

if __name__ == '__main__':
    run_backtest()
