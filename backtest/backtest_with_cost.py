"""v4.3.2回测：精确A股交易成本(Zipline启发) + 迟滞带 + 调仓频率分层
成本模型改进：
  - v1: 固定0.1%单边(粗糙)
  - v2: A股三费精确(佣金+印花税+过户费) + 二次滑点模型
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, akshare as ak

# ============ 精确A股交易成本模型(Zipline启发) ============
class AShareCostModel:
    """A股ETF交易成本精确模型
    
    灵感来自Zipline的VolumeShareSlippage + Commission分层架构：
    - Commission层：固定费用(佣金+印花税+过户费)
    - Slippage层：二次冲击成本 = price_impact * (volume_share^2)
    
    ETF交易费用明细(2023年后)：
      佣金: 双边万2.5，不足5元按5元收
      印花税: 卖出万0.5 (2023.8.28降为万0.5，之前万1)
      过户费: 双边万0.1 (2022.4.29降为万0.1)
    """
    def __init__(self, commission_rate=0.00025, min_commission=5.0,
                 stamp_tax_rate=0.00005, transfer_fee_rate=0.00001,
                 price_impact=0.1, volume_limit=0.025):
        self.commission_rate = commission_rate      # 券商佣金率(双边)
        self.min_commission = min_commission        # 最低佣金(元)
        self.stamp_tax_rate = stamp_tax_rate        # 印花税率(仅卖出)
        self.transfer_fee_rate = transfer_fee_rate  # 过户费率(双边)
        self.price_impact = price_impact            # 滑点冲击系数
        self.volume_limit = volume_limit            # 单笔成交量占比上限
    
    def calculate(self, trade_value, is_buy=True, volume_share=0.0):
        """计算单笔交易总成本
        
        Args:
            trade_value: 交易金额(元)
            is_buy: 买入/卖出
            volume_share: 成交量占当日总成交量比例(0~1)，用于滑点
        
        Returns:
            (commission, stamp_tax, transfer_fee, slippage) 四项成本
        """
        if trade_value <= 0:
            return 0.0, 0.0, 0.0, 0.0
        
        # 佣金(双边，最低5元)
        commission = max(trade_value * self.commission_rate, self.min_commission)
        
        # 印花税(仅卖出)
        stamp_tax = 0.0 if is_buy else trade_value * self.stamp_tax_rate
        
        # 过户费(双边)
        transfer_fee = trade_value * self.transfer_fee_rate
        
        # 二次滑点(Zipline VolumeShareSlippage)
        # 冲击成本 = price_impact * volume_share^2
        vs = min(volume_share, self.volume_limit)
        slippage = trade_value * self.price_impact * (vs ** 2)
        
        return commission, stamp_tax, transfer_fee, slippage
    
    def total_rate(self, is_buy=True, volume_share=0.0):
        """简化接口：返回总费率(用于快速估算)"""
        rate = self.commission_rate + self.transfer_fee_rate
        if not is_buy:
            rate += self.stamp_tax_rate
        vs = min(volume_share, self.volume_limit)
        rate += self.price_impact * (vs ** 2)
        return rate
    
    def blended_rate(self, volume_share=0.0):
        """买卖混合单边费率(假设50%买+50%卖)"""
        buy_rate = self.total_rate(True, volume_share)
        sell_rate = self.total_rate(False, volume_share)
        return (buy_rate + sell_rate) / 2

# 全局成本模型实例
COST_MODEL = AShareCostModel()

# ============ 数据获取 ============
ETF_MAP = {'红利':'510880','沪深300':'510300','国证2000':'159907','进攻':'159908'}
all_price = {}
for name, code in ETF_MAP.items():
    df = ak.fund_etf_hist_em(symbol=code, period='daily', start_date='20050101', end_date='20260505', adjust='qfq')
    df = df[['日期','收盘']].rename(columns={'日期':'date','收盘':name})
    df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date').reset_index(drop=True)
    all_price[name] = df

df_msh = ak.macro_china_market_margin_sh()[['日期','融资买入额']].rename(columns={'日期':'date','融资买入额':'rzye'})
df_msz = ak.macro_china_market_margin_sz()[['日期','融资买入额']].rename(columns={'日期':'date','融资买入额':'rzye'})
df_msh['date'] = pd.to_datetime(df_msh['date']); df_msz['date'] = pd.to_datetime(df_msz['date'])
df_margin = df_msh.merge(df_msz, on='date', how='outer', suffixes=('_sh','_sz'))
df_margin = df_margin.sort_values('date').reset_index(drop=True)
df_margin['rzye_total'] = pd.to_numeric(df_margin['rzye_sh'], errors='coerce') + pd.to_numeric(df_margin['rzye_sz'], errors='coerce')

sw_dp = ak.index_hist_sw(symbol='801811', period='day')[['日期','收盘']].rename(columns={'日期':'date','收盘':'sw_large'})
sw_xp = ak.index_hist_sw(symbol='801813', period='day')[['日期','收盘']].rename(columns={'日期':'date','收盘':'sw_small'})
sw_dp['date'] = pd.to_datetime(sw_dp['date']); sw_xp['date'] = pd.to_datetime(sw_xp['date'])

df_hs300 = ak.stock_zh_index_daily(symbol='sh000300')[['date','close']].rename(columns={'close':'hs300'})
df_hs300['date'] = pd.to_datetime(df_hs300['date']); df_hs300 = df_hs300.sort_values('date').reset_index(drop=True)

base = all_price['沪深300'].copy()
for name in ['红利','国证2000','进攻']:
    base = base.merge(all_price[name][['date',name]], on='date', how='left')
base = base.merge(df_hs300[['date','hs300']], on='date', how='left')
base = base.merge(df_margin[['date','rzye_total']], on='date', how='left')
base = base.merge(sw_dp[['date','sw_large']], on='date', how='left')
base = base.merge(sw_xp[['date','sw_small']], on='date', how='left')
for c in ['红利','国证2000','进攻','hs300','rzye_total','sw_large','sw_small']:
    base[c] = base[c].ffill()
df_full = base[(base.date>='2007-01-01')&(base.date<='2026-04-30')].reset_index(drop=True)

W=20; STYLE_W=60
df_full['bb_z'] = (df_full['hs300']-df_full['hs300'].rolling(W).mean())/df_full['hs300'].rolling(W).std()
df_full['rz_z'] = (df_full['rzye_total']-df_full['rzye_total'].rolling(W).mean())/df_full['rzye_total'].rolling(W).std()

def z2p(z, lo=0.20, hi=0.80):
    z = float(np.clip(z,-3,3)); return 0.50+z/3*(hi-0.50)
df_full['pos_total'] = df_full['bb_z'].apply(lambda v: z2p(v)*0.70 if not pd.isna(v) else 0.35) + \
                        df_full['rz_z'].apply(lambda v: z2p(v)*0.30 if not pd.isna(v) else 0.15)
df_full['ratio_xd'] = df_full['sw_small']/df_full['sw_large']
df_full['pct_xd'] = df_full['ratio_xd'].rolling(STYLE_W).rank(pct=True)

# ============ 迟滞带三态 ============
def calc_split(pct):
    if pd.isna(pct): return 0.50
    if pct>0.80: return 0.10
    if pct<0.20: return 0.90
    return 0.50

def state_with_hysteresis(pos_series, low_down=0.35, low_up=0.45, high_down=0.60, high_up=0.70):
    """三态迟滞带：上升用高阈值(up)，下降用低阈值(down)"""
    states = []
    current = 1  # 0=防御, 1=均衡, 2=进攻
    for pt in pos_series:
        if pd.isna(pt):
            states.append(current)
            continue
        if current == 0:  # 防御
            if pt >= low_up:
                current = 1
            elif pt >= high_up:
                current = 2
        elif current == 1:  # 均衡
            if pt <= low_down:
                current = 0
            elif pt >= high_up:
                current = 2
        elif current == 2:  # 进攻
            if pt <= high_down:
                current = 1
            elif pt <= low_down:
                current = 0
        states.append(current)
    return states

def calc_weights_with_state(pt, sp, state):
    """用离散三态计算权重，消除阈值附近振荡"""
    CFG = {
        0: {'红利':0.20,'宽基':0.10,'进攻':0.05,'cash':0.65},  # 防御
        1: {'红利':0.15,'宽基':0.15,'进攻':0.25,'cash':0.45},  # 均衡
        2: {'红利':0.10,'宽基':0.20,'进攻':0.45,'cash':0.25},  # 进攻
    }
    c = CFG[state]
    wd = c['红利']; wb = c['宽基']; wa = c['进攻']; wc = c['cash']
    # 风格微调 ±3pp
    sf = (sp - 0.50) / 0.40
    wd += 0.03 * sf
    wa -= 0.03 * sf
    # split拆分宽基
    w3 = wb * sp; w2 = wb * (1 - sp)
    # 约束
    wd = float(np.clip(wd, 0.05, 0.25))
    wa = float(np.clip(wa, 0.05, 0.45))
    wc = 1 - w3 - w2 - wd - wa
    if wc < 0.20:
        wa = max(wa - (0.20 - wc), 0.05)
        wc = 1 - w3 - w2 - wd - wa
    return wd, w3, w2, wa, wc

# ============ 回测引擎 ============
ETF_COLS = ['红利','沪深300','国证2000','进攻']
WEIGHT_COLS = ['w_div','w_300','w_2000','w_atk']
PRICE_COLS = ['红利','沪深300','国证2000','进攻']

def backtest(df, weights_fn, rebalance_freq='daily', drift_threshold=0.0, 
             cost_rate=0.001, cost_model=None, portfolio_value=1e6):
    """
    回测引擎 v4.3.2
    
    cost_rate: 简单单边费率(向后兼容)，仅当cost_model=None时使用
    cost_model: AShareCostModel实例，启用精确三费+滑点
    portfolio_value: 初始资金(元)，用于精确计算佣金最低5元门槛
    """
    dates = df['date'].values
    prices = df[PRICE_COLS].values  # (N, 4)
    n = len(df)
    
    # 预计算目标权重
    target_weights = np.zeros((n, 4))
    for i in range(n):
        target_weights[i] = weights_fn(i)
    
    # 模拟
    nav = 1.0
    holdings = np.array([0.25, 0.25, 0.25, 0.25])
    total_cost = 0.0
    total_commission = 0.0
    total_stamp_tax = 0.0
    total_transfer = 0.0
    total_slippage = 0.0
    rebalance_count = 0
    nav_series = [nav]
    
    last_rebalance_day = -999
    
    for i in range(n):
        if i == 0:
            # 第一天：调到目标
            tw = target_weights[0]
            trade_amounts = np.abs(holdings - tw) * nav * portfolio_value
            cost, cost_detail = _calc_trade_cost(trade_amounts, holdings, tw, 
                                                   cost_rate, cost_model)
            nav -= cost / portfolio_value
            total_cost += cost
            if cost_detail:
                total_commission += cost_detail['commission']
                total_stamp_tax += cost_detail['stamp_tax']
                total_transfer += cost_detail['transfer_fee']
                total_slippage += cost_detail['slippage']
            holdings = tw.copy()
            rebalance_count += 1
            last_rebalance_day = i
            nav_series.append(nav)
            continue
        
        # 计算当日净值(收盘价变化)
        if i < n - 1:
            ret = prices[i] / prices[i-1] - 1
        else:
            ret = np.zeros(4)
        
        for j in range(4):
            nav += holdings[j] * nav * ret[j]
        
        nav_series.append(nav)
        
        # 判断是否调仓
        if rebalance_freq == 'daily':
            check = True
        elif rebalance_freq == 'weekly':
            check = (i - last_rebalance_day) >= 5
        elif rebalance_freq == 'biweekly':
            check = (i - last_rebalance_day) >= 10
        elif rebalance_freq == 'monthly':
            check = (i - last_rebalance_day) >= 20
        else:
            check = True
        
        if not check:
            continue
        
        # 漂移检查
        if drift_threshold > 0:
            drift = np.sum(np.abs(holdings - target_weights[i]))
            if drift < drift_threshold:
                continue
        
        # 执行调仓
        tw = target_weights[i]
        trade_amounts = np.abs(holdings - tw) * nav * portfolio_value
        
        if cost_model is not None:
            cost, cost_detail = _calc_trade_cost(trade_amounts, holdings, tw,
                                                   cost_rate, cost_model)
        else:
            turnover = np.sum(np.abs(holdings - tw))
            cost = turnover * nav * portfolio_value * cost_rate
            cost_detail = None
        
        nav -= cost / portfolio_value
        total_cost += cost
        if cost_detail:
            total_commission += cost_detail['commission']
            total_stamp_tax += cost_detail['stamp_tax']
            total_transfer += cost_detail['transfer_fee']
            total_slippage += cost_detail['slippage']
        
        holdings = tw.copy()
        rebalance_count += 1
        last_rebalance_day = i
    
    nav_series = np.array(nav_series)
    returns = np.diff(nav_series) / nav_series[:-1]
    ann_ret = (nav / 1.0) ** (252 / n) - 1
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cummax = np.maximum.accumulate(nav_series)
    dd = (nav_series - cummax) / cummax
    max_dd = np.min(dd)
    
    result = {
        'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe,
        'max_dd': max_dd, 'final_nav': nav, 'rebalance_count': rebalance_count,
        'total_cost': total_cost, 'total_cost_pct': total_cost / (nav * portfolio_value) * 100,
        'nav_series': nav_series
    }
    
    if cost_model is not None:
        result.update({
            'cost_commission': total_commission,
            'cost_stamp_tax': total_stamp_tax,
            'cost_transfer': total_transfer,
            'cost_slippage': total_slippage,
        })
    
    return result


def _calc_trade_cost(trade_amounts, old_weights, new_weights, simple_rate, cost_model):
    """计算调仓交易成本
    
    Returns: (total_cost, cost_detail_dict)
    """
    total_commission = 0.0
    total_stamp_tax = 0.0
    total_transfer = 0.0
    total_slippage = 0.0
    
    if cost_model is None:
        # 简单模式(向后兼容)
        total = np.sum(trade_amounts) * simple_rate
        return total, None
    
    for j in range(4):
        amt = trade_amounts[j]
        if amt < 1e-6:
            continue
        is_buy = new_weights[j] > old_weights[j]
        comm, stamp, xfer, slip = cost_model.calculate(
            trade_value=amt, is_buy=is_buy, volume_share=0.0  # ETF日频回测volume_share≈0
        )
        total_commission += comm
        total_stamp_tax += stamp
        total_transfer += xfer
        total_slippage += slip
    
    total = total_commission + total_stamp_tax + total_transfer + total_slippage
    return total, {
        'commission': total_commission,
        'stamp_tax': total_stamp_tax,
        'transfer_fee': total_transfer,
        'slippage': total_slippage,
    }



# ============ 运行回测 ============
configs = [
    ('A: 连续三态+日调+0成本',    make_weights_fn(False, 'continuous'), 'daily',    0.0,  0.0,   None),
    ('B: 连续三态+日调+0.1%成本', make_weights_fn(False, 'continuous'), 'daily',    0.0,  COST,  None),
    ('C: 迟滞离散+日调+0.1%成本', make_weights_fn(True,  'discrete'),   'daily',    0.0,  COST,  None),
    ('D: 迟滞离散+周调+0.1%成本', make_weights_fn(True,  'discrete'),   'weekly',   0.0,  COST,  None),
    ('E: 迟滞离散+双周调+0.1%成本',make_weights_fn(True,  'discrete'),   'biweekly', 0.0,  COST,  None),
    ('F: 迟滞离散+月调+0.1%成本',  make_weights_fn(True,  'discrete'),   'monthly',  0.0,  COST,  None),
    ('G: 迟滞离散+周调+3%阈值',   make_weights_fn(True,  'discrete'),   'weekly',   0.03, COST,  None),
    ('H: 迟滞离散+月调+3%阈值',   make_weights_fn(True,  'discrete'),   'monthly',  0.03, COST,  None),
    ('I: 迟滞离散+月调+5%阈值',   make_weights_fn(True,  'discrete'),   'monthly',  0.05, COST,  None),
    ('J: 迟滞离散+月调+8%阈值',   make_weights_fn(True,  'discrete'),   'monthly',  0.08, COST,  None),
    ('K: 迟滞月调3%+精确成本',    make_weights_fn(True,  'discrete'),   'monthly',  0.03, 0.0,   COST_MODEL),
    ('L: 迟滞日调+精确成本',      make_weights_fn(True,  'discrete'),   'daily',    0.0,  0.0,   COST_MODEL),
    ('基准: 满仓沪深300',           buy_hold_300_fn,                       'daily',    0.0,  0.0,   None),
]

print(f'回测资金: ¥2,000,000 (200万)')
print(f'成本模型: v1=固定0.1% | v2=佣金万2.5+印花税万0.5(卖出)+过户费万0.1+二次滑点')
print()
hdr = f'{"策略":<30} {"年化":>7} {"波动":>7} {"夏普":>6} {"最大回撤":>8} {"调仓次数":>8} {"成本占比":>8}'
print(hdr)
print('-' * 90)

results = {}
for label, wfn, freq, drift, cost, cmodel in configs:
    r = backtest(df_bt, wfn, rebalance_freq=freq, drift_threshold=drift,
                 cost_rate=cost, cost_model=cmodel, portfolio_value=2_000_000)
    results[label] = r
    print(f'{label:<30} {r["ann_ret"]*100:>6.1f}% {r["ann_vol"]*100:>6.1f}% {r["sharpe"]:>6.2f} {r["max_dd"]*100:>7.1f}% {r["rebalance_count"]:>7}次 {r["total_cost_pct"]:>7.2f}%')

# 精确成本分解
print()
print('=== 精确成本模型(v2)费用分解 ===')
for label in ['K: 迟滞月调3%+精确成本', 'L: 迟滞日调+精确成本']:
    r = results[label]
    tc = r['total_cost']
    print(f'\n  {label}:')
    print(f'    总成本: ¥{tc:,.0f} (占终值{r["total_cost_pct"]:.2f}%)')
    print(f'      佣金:   ¥{r["cost_commission"]:>10,.0f} ({r["cost_commission"]/tc*100:.1f}%)')
    print(f'      印花税: ¥{r["cost_stamp_tax"]:>10,.0f} ({r["cost_stamp_tax"]/tc*100:.1f}%)')
    print(f'      过户费: ¥{r["cost_transfer"]:>10,.0f} ({r["cost_transfer"]/tc*100:.1f}%)')
    print(f'      滑点:   ¥{r["cost_slippage"]:>10,.0f} ({r["cost_slippage"]/tc*100:.1f}%)')
    # v1 vs v2对比
    if '月调' in label:
        v1_label = 'H: 迟滞离散+月调+3%阈值'
    else:
        v1_label = 'C: 迟滞离散+日调+0.1%成本'
    v1_cost = results[v1_label]['total_cost']
    diff_pct = (tc - v1_cost) / v1_cost * 100
    print(f'    vs v1固定0.1%: ¥{v1_cost:,.0f} -> ¥{tc:,.0f} ({diff_pct:+.1f}%)')

print()

# ============ 迟滞带 vs 原始 三态切换对比 ============
print('=== 迟滞带效果 ===')
orig_switches = np.sum(states_orig[idx_start:idx_end] != np.roll(states_orig[idx_start:idx_end], 1))
hyst_switches = np.sum(np.array(states_hyst[idx_start:idx_end]) != np.roll(np.array(states_hyst[idx_start:idx_end]), 1))
print(f'  原始三态切换: {orig_switches}次')
print(f'  迟滞三态切换: {hyst_switches}次 (减少{(1-hyst_switches/orig_switches)*100:.0f}%)')
print()

# 迟滞态分布
hyst_arr = np.array(states_hyst[idx_start:idx_end])
for s, sn in [(0,'防御'),(1,'均衡'),(2,'进攻')]:
    cnt = (hyst_arr==s).sum()
    print(f'  {sn}: {cnt}天 ({cnt/n_bt*100:.0f}%)')

print()

# ============ 分年度对比(选最实用的几个策略) ============
print('=== 分年度对比(核心策略) ===')
key_strategies = ['A: 连续三态+日调+0成本', 'K: 迟滞月调3%+精确成本', 'H: 迟滞离散+月调+3%阈值']
key_labels = ['日调0成本(理想)', '月调精确成本(v2)', '月调0.1%(v1)']

df_years = pd.DataFrame()
for label, klabel in zip(key_strategies, key_labels):
    nav = results[label]['nav_series']
    yr_ret = {}
    years = sorted(set(d.year for d in df_bt['date']))
    for y in years:
        mask_y = [d.year == y for d in df_bt['date']]
        idx_y = np.where(mask_y)[0]
        if len(idx_y) > 0:
            start_nav = nav[idx_y[0]]
            end_nav = nav[min(idx_y[-1]+1, len(nav)-1)]
            yr_ret[y] = (end_nav / start_nav - 1) * 100
    df_years[klabel] = pd.Series(yr_ret)

print(f'{"年份":<6}', end='')
for l in key_labels:
    print(f'{l:>18}', end='')
print()
print('-' * 60)
for y in df_years.index:
    print(f'{y:<6}', end='')
    for l in key_labels:
        v = df_years[l].get(y, 0)
        print(f'{v:>17.1f}%', end='')
    print()

# 年化差异
r0 = results[key_strategies[0]]
rK = results[key_strategies[1]]
print(f'\n精确成本(v2) vs 理想值:')
print(f'  夏普损失: {r0["sharpe"] - rK["sharpe"]:.2f}')
print(f'  年化损失: {(r0["ann_ret"] - rK["ann_ret"])*100:.1f}%')
print(f'  调仓次数: {rK["rebalance_count"]}')

print()

# ============ 200万模拟(精确成本) ============
print('=== 200万模拟(迟滞+月调+3%+精确成本) ===')
r = results['K: 迟滞月调3%+精确成本']
nav = r['nav_series']
start_val = 2_000_000
end_val = start_val * nav[-1]
max_loss = start_val * r['max_dd']
print(f'  初始: ¥{start_val:,.0f}')
print(f'  终值: ¥{end_val:,.0f} ({(end_val/start_val-1)*100:.0f}%)')
print(f'  最大亏损: ¥{max_loss:,.0f}')
print(f'  年化收益: ¥{(end_val/start_val)**(1/(n_bt/252))*start_val - start_val:,.0f}/年')
print(f'  总交易成本: ¥{r["total_cost"]:,.0f} ({r["total_cost_pct"]:.2f}%)')

print('\n完成.')
