# 配对交易回测（Pair Trading Backtest）

基于quant-trading仓库B类策略的A股适配回测。

## 策略来源

- **原始仓库**: je-suis-tm/quant-trading
- **策略文件**: `Pair trading backtest.py`
- **核心逻辑**: Engle-Granger两步法协整检验 + Z-score触发

## A股适配

### 数据源问题
- 原策略使用Yahoo Finance获取美股/外汇数据
- A股个股数据：东财push2his接口被代理阻断 ❌
- **解决方案**: 改用AKShare指数数据（stock_zh_index_daily）

### 回测脚本

| 脚本 | 说明 | 数据源 | 状态 |
|------|------|--------|------|
| `backtest_pair_trading.py` | 个股配对（原始版本） | ak.stock_zh_a_hist | ❌ 数据不可用 |
| `backtest_pair_trading_etf.py` | ETF配对 | ak.fund_etf_hist_sina | ❌ 接口返回空 |
| `backtest_pair_trading_index.py` | 指数配对 | ak.stock_zh_index_daily | ✅ 可运行 |
| `backtest_pair_trading_quantstats.py` | 指数配对（完整版） | ak.stock_zh_index_daily | ✅ **推荐** |

## 核心发现

### 回测结果（2020-2026）

| 配对 | 年化收益 | 夏普 | Sortino | 最大回撤 | 交易次数 | 协整率 |
|------|---------|------|---------|---------|---------|--------|
| 沪深300 vs 国证2000 | -3.90% | -0.18 | -0.27 | -41.76% | 1次 | 12.6% |
| 沪深300 vs 中证500 | 0.51% | 0.10 | 0.14 | -23.46% | 1次 | 23.0% |
| 创业板指 vs 国证2000 | -4.86% | -0.23 | -0.35 | -48.15% | 0次 | 5.2% |

### 结论

**❌ 配对交易在A股失效**：
1. **协整关系极弱**（5-23%），A股指数间不存在长期稳定均衡
2. **交易次数几乎为0**（0-1次/6年），Z-score难以触发±1σ边界
3. **收益全部为负**，夏普<-0.2，远不如买入持有

**失败原因**：
- A股政策市特征明显，行业/风格受政策驱动而结构性断裂
- 协整检验假设长期均衡，但A股市场结构变化太快（注册制、行业监管）
- 原策略针对美股/外汇设计，A股市场特性不符

## 使用方法

### 运行回测

```bash
cd ~/a-share-quant/backtest
python3 backtest_pair_trading_quantstats.py
```

### 输出

- **终端输出**: 基础统计 + QuantStats完整指标
- **HTML报告**: `reports/pair_trading_*.html`（含19种图表）

### 查看报告

```bash
# 在浏览器中打开
firefox reports/pair_trading_沪深300_vs_中证500.html
```

## 与v4.3.1对比

| 维度 | v4.3.1（右侧趋势） | Pair Trading（统计套利） |
|------|-------------------|----------------------|
| 年化收益 | 11.8% | -3.90% ~ 0.51% |
| 夏普比率 | 0.73 | -0.23 ~ 0.10 |
| 交易频率 | 160次/14年 | 0-1次/6年 |
| A股适配性 | ✅ 专为A股优化 | ❌ 美股设计，不适配 |

**结论**: v4.3.1的右侧趋势框架远优于统计套利策略。

## Oil Money策略

**原始逻辑**: 原油价格 vs 产油国货币（NOK/USD、CAD/USD）
**A股不适配**:
- 无外汇数据（RMB/USD受管制）
- 无商品期货（需特殊账户）
- A股是内生市场，不受外部商品直接驱动

**A股替代方案**（需商品数据）:
- 商品股票 vs 商品指数（中国石油 vs 原油期货）
- 黄金股票 vs 黄金ETF（山东黄金 vs 518880）
- 铜业股 vs 铜期货（江西铜业 vs LME铜）

但这些都依赖期货/商品数据，且A股上市公司受多重因素影响，单商品属性弱。

## 依赖

```bash
python3 -m pip install akshare pandas numpy statsmodels quantstats
```

## 参考文献

- quant-trading仓库: https://github.com/je-suis-tm/quant-trading
- Engle-Granger两步法: https://en.wikipedia.org/wiki/Engle–Granger_two-step_method
- A股择时体系v4.3.1: 见`backtest_v43_etf.py`
