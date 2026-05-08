"""v4.3回测：带交易成本 + 迟滞带 + 调仓频率分层"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, akshare as ak

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

def backtest(df, weights_fn, rebalance_freq='daily', drift_threshold=0.0, cost_rate=0.001):
    """
    rebalance_freq: 'daily','weekly','biweekly','monthly'
    drift_threshold: 品类权重累计漂移超此值才触发调仓(0=每次freq都调)
    cost_rate: 单边交易成本(ETF约0.03-0.05%,这里用0.1%含滑点)
    """
    dates = df['date'].values
    prices = df[PRICE_COLS].values  # (N, 4)
    n = len(df)
    
    # 预计算目标权重
    target_weights = np.zeros((n, 4))  # [红利, 300, 2000, 进攻]
    for i in range(n):
        w = weights_fn(i)
        target_weights[i] = w
    
    # 模拟
    cash_weight = np.ones(n)
    nav = 1.0
    holdings = np.array([0.25, 0.25, 0.25, 0.25])  # 初始等权
    cash = 0.0  # 无现金
    total_cost = 0.0
    rebalance_count = 0
    nav_series = [nav]
    
    last_rebalance_day = -999
    
    for i in range(n):
        if i == 0:
            # 第一天：调到目标
            tw = target_weights[0]
            turnover = np.sum(np.abs(holdings - tw))
            cost = turnover * nav * cost_rate
            nav -= cost
            total_cost += cost
            holdings = tw.copy()
            cash = nav * cash_weight[0] if i < len(cash_weight) else 0
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
        # 先算含现金的目标权重
        cw = 1.0 - np.sum(tw)  # cash weight
        turnover = np.sum(np.abs(holdings - tw))
        cost = turnover * nav * cost_rate
        nav -= cost
        total_cost += cost
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
    
    return {
        'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe,
        'max_dd': max_dd, 'final_nav': nav, 'rebalance_count': rebalance_count,
        'total_cost': total_cost, 'total_cost_pct': total_cost / nav * 100,
        'nav_series': nav_series
    }

# ============ 准备回测 ============
df_full['split'] = df_full['pct_xd'].apply(calc_split)

# 迟滞带状态序列
states_hyst = state_with_hysteresis(df_full['pos_total'].values, 
                                      low_down=0.35, low_up=0.45, 
                                      high_down=0.60, high_up=0.70)

# 原始连续三态(无迟滞)
states_orig = pd.cut(df_full['pos_total'], bins=[-0.1,0.40,0.65,1.0], labels=[0,1,2]).cat.codes.values

# 截取回测区间
mask = df_full['date'] >= '2012-05-28'
idx_start = np.where(mask)[0][0]
idx_end = len(df_full)
df_bt = df_full[idx_start:idx_end].reset_index(drop=True)
n_bt = len(df_bt)

print(f'回测区间: {df_bt["date"].iloc[0].strftime("%Y-%m-%d")} ~ {df_bt["date"].iloc[-1].strftime("%Y-%m-%d")}')
print(f'交易日: {n_bt} ({n_bt/252:.1f}年)')
print()

# ============ 定义多组权重函数 ============
# 策略A: 原始连续三态(无迟滞)，每日调仓，无成本
# 策略B: 原始连续三态，每日调仓，0.1%成本
# 策略C: 迟滞带离散三态，每日调仓，0.1%成本
# 策略D: 迟滞带离散三态，周度调仓，0.1%成本
# 策略E: 迟滞带离散三态，双周调仓，0.1%成本
# 策略F: 迟滞带离散三态，月度调仓，0.1%成本
# 策略G: 迟滞带离散三态，周度+3%漂移阈值，0.1%成本
# 策略H: 迟滞带离散三态，月度+3%漂移阈值，0.1%成本
# 策略I: 迟滞带离散三态，月度+5%漂移阈值，0.1%成本

COST = 0.001  # 0.1%单边 (ETF佣金+滑点)

def make_weights_fn(use_hyst, mode='continuous'):
    """返回一个 weights_fn(i_offset) -> [wd, w3, w2, wa]"""
    def fn(i):
        gi = i + idx_start  # global index
        pt = df_full['pos_total'].iloc[gi]
        sp = df_full['split'].iloc[gi]
        
        if mode == 'continuous':
            # 原始连续映射
            t = float(np.clip((pt - 0.40) / 0.25, 0, 1))
            CFG = {'红利':{'防御':0.20,'进攻':0.10},'宽基':{'防御':0.10,'进攻':0.20},'进攻':{'防御':0.05,'进攻':0.45}}
            wd = CFG['红利']['防御'] + t * (CFG['红利']['进攻'] - CFG['红利']['防御'])
            wb = CFG['宽基']['防御'] + t * (CFG['宽基']['进攻'] - CFG['宽基']['防御'])
            wa = CFG['进攻']['防御'] + t * (CFG['进攻']['进攻'] - CFG['进攻']['防御'])
            sf = (sp - 0.50) / 0.40; wd += 0.03 * sf; wa -= 0.03 * sf
            w3 = wb * sp; w2 = wb * (1 - sp)
            wc = 1 - w3 - w2 - wd - wa
            wd = float(np.clip(wd, 0.05, 0.25))
            wa = float(np.clip(wa, 0.05, 0.45))
            wc = 1 - w3 - w2 - wd - wa
            if wc < 0.20:
                wa = max(wa - (0.20 - wc), 0.05)
                wc = 1 - w3 - w2 - wd - wa
            return [wd, w3, w2, wa]
        else:
            state = states_hyst[gi] if use_hyst else states_orig[gi]
            w = calc_weights_with_state(pt, sp, state)
            return [w[0], w[1], w[2], w[3]]  # 4个权益权重
    return fn

# 基准：满仓300
def buy_hold_300_fn(i):
    return [0.0, 1.0, 0.0, 0.0]

# ============ 运行回测 ============
configs = [
    ('A: 连续三态+日调+0成本',    make_weights_fn(False, 'continuous'), 'daily',    0.0,  0.0),
    ('B: 连续三态+日调+0.1%成本', make_weights_fn(False, 'continuous'), 'daily',    0.0,  COST),
    ('C: 迟滞离散+日调+0.1%成本', make_weights_fn(True,  'discrete'),   'daily',    0.0,  COST),
    ('D: 迟滞离散+周调+0.1%成本', make_weights_fn(True,  'discrete'),   'weekly',   0.0,  COST),
    ('E: 迟滞离散+双周调+0.1%成本',make_weights_fn(True,  'discrete'),   'biweekly', 0.0,  COST),
    ('F: 迟滞离散+月调+0.1%成本',  make_weights_fn(True,  'discrete'),   'monthly',  0.0,  COST),
    ('G: 迟滞离散+周调+3%阈值',   make_weights_fn(True,  'discrete'),   'weekly',   0.03, COST),
    ('H: 迟滞离散+月调+3%阈值',   make_weights_fn(True,  'discrete'),   'monthly',  0.03, COST),
    ('I: 迟滞离散+月调+5%阈值',   make_weights_fn(True,  'discrete'),   'monthly',  0.05, COST),
    ('J: 迟滞离散+月调+8%阈值',   make_weights_fn(True,  'discrete'),   'monthly',  0.08, COST),
    ('基准: 满仓沪深300',           buy_hold_300_fn,                       'daily',    0.0,  0.0),
]

print(f'{"策略":<30} {"年化":>7} {"波动":>7} {"夏普":>6} {"最大回撤":>8} {"调仓次数":>8} {"成本占比":>8}')
print('-' * 90)

results = {}
for label, wfn, freq, drift, cost in configs:
    r = backtest(df_bt, wfn, rebalance_freq=freq, drift_threshold=drift, cost_rate=cost)
    results[label] = r
    yrs = n_bt / 252
    print(f'{label:<30} {r["ann_ret"]*100:>6.1f}% {r["ann_vol"]*100:>6.1f}% {r["sharpe"]:>6.2f} {r["max_dd"]*100:>7.1f}% {r["rebalance_count"]:>7}次 {r["total_cost_pct"]:>7.2f}%')

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
print('=== 分年度对比(迟滞+月调+3%阈值 vs 原始+日调) ===')
key_strategies = ['A: 连续三态+日调+0成本', 'C: 迟滞离散+日调+0.1%成本', 'H: 迟滞离散+月调+3%阈值']
key_labels = ['日调0成本', '迟滞日调', '迟滞月调+3%']

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
    print(f'{l:>12}', end='')
print()
print('-' * 42)
for y in df_years.index:
    print(f'{y:<6}', end='')
    for l in key_labels:
        v = df_years[l].get(y, 0)
        print(f'{v:>11.1f}%', end='')
    print()

# 年化差异
r0 = results[key_strategies[0]]
rH = results[key_strategies[2]]
print(f'\n夏普损失: {r0["sharpe"] - rH["sharpe"]:.2f}')
print(f'年化损失: {(r0["ann_ret"] - rH["ann_ret"])*100:.1f}%')
print(f'成本节省: 调仓{rH["rebalance_count"]}次 vs {r0["rebalance_count"]}次 (减少{(1-rH["rebalance_count"]/r0["rebalance_count"])*100:.0f}%)')

print()

# ============ 200万模拟 ============
print('=== 200万模拟(迟滞+月调+3%阈值) ===')
r = results['H: 迟滞离散+月调+3%阈值']
nav = r['nav_series']
start_val = 2000000
end_val = start_val * nav[-1]
max_loss = start_val * r['max_dd']
cost_total = r['total_cost_pct'] / 100 * end_val  # 近似
print(f'  初始: ¥2,000,000')
print(f'  终值: ¥{end_val:,.0f} ({(end_val/start_val-1)*100:.0f}%)')
print(f'  最大亏损: ¥{max_loss:,.0f}')
print(f'  年化收益: ¥{(end_val/start_val)**(1/(n_bt/252))*start_val - start_val:,.0f}/年')

print('\n完成.')
