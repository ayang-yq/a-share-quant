# 投资财经数据源清单（稳定可用）

> 最后更新: 2026-05-04 | 基于 AKShare 全量审计 + 压力测试实测  
> 原则：每个数据维度标注唯一最优源，避免重复调用，降低被限流风险
> 
> ⚠️ **CSIndex (csindex.com.cn) 标记为次选数据源** — 实测有IP级限流（30次零间隔→403，冷却>10分钟），仅提供PE无PB，仅20日快照无历史序列。仅用于中证指数成分股查询和行情数据。

---

## 一、数据源总览

| 数据维度 | 最优源 | 次选源 | 调用方式 | 响应速度 | 稳定性 | 限流风险 |
|----------|--------|--------|----------|----------|--------|----------|
| A股指数日K | 东财 | - | `stock_zh_index_daily_em` | 0.4s | ✅ 极稳 | 无 |
| 国证指数日K | 国证 | - | `index_hist_cni` | 0.1s | ✅ 极稳 | 无 |
| 申万行业日K | 申万 | - | `index_hist_sw` | 0.7s | ✅ 稳 | 低 |
| 中证指数PE/股息率(快照) | - | **CSIndex⚠️** | `stock_zh_index_value_csindex` | 0.3s | ⚠️ 次选 | **高** |
| 申万行业PE/PB/股息率(截面) | 申万 | - | `sw_index_first_info` | 1.4s | ✅ 稳 | 低(≤10次) |
| 申万行业PE/PB历史(时序) | 申万 | - | `index_analysis_daily_sw` | 3s | ✅ 稳 | 低(≤5次) |
| 指数成分股+权重 | 东财 | CSIndex⚠️ | `index_stock_cons_weight_csindex` | 0.3s | ⚠️ 次选 | **高(共享CSIndex配额)** |
| 指数成分股(东财) | 东财 | - | `index_stock_cons` | 1-11s | ✅ 稳 | 无 |
| 港股指数日K | 东财 | - | `stock_hk_index_daily_em` | 0.3-5s | ✅ 稳 | 无 |
| 港股指数实时 | 东财 | - | `stock_hk_index_spot_em` | 3s | ✅ 稳 | 无 |
| 美股指数日K | 新浪 | - | `index_us_stock_sina` | 0.3s | ✅ 极稳 | 无 |
| 全球指数实时 | 东财 | - | `index_global_spot_em` | 0.1s | ✅ 极稳 | 无 |
| 全球指数日K | 东财 | - | `index_global_hist_em` | 0.5s | ✅ 稳 | 无 |
| A股指数实时 | 东财 | - | `stock_zh_index_spot_em` | 2-3s | ✅ 稳 | 无 |
| A股指数实时(全量) | 新浪 | - | `stock_zh_index_spot_sina` | 1.5s | ✅ 稳 | 无 |

---

## 二、CSIndex 限流实测结论

**2026-05-04 压力测试实锤：CSIndex 存在 IP 级别限流。**

| 测试场景 | 结果 |
|----------|------|
| 单次调用 | 0.3s 正常 |
| 10次(100ms间隔) | 0.25s × 10，正常 |
| 30次(零间隔轮换) | 第8-9次出现16秒延迟，后续恢复 |
| 30次(零间隔) + 后续重试 | **403 Forbidden，超过8分钟未恢复** |

**结论**: CSIndex 对高频请求有 IP 封禁机制。零间隔轰炸30次后触发 403，冷却时间可能 > 10分钟。

**使用规则**:
- 每次调用间隔 ≥ 0.5秒
- 单次批量拉取不超过 10 个指数
- 日累计不超过 50 次调用
- 被 403 后等待 10-15 分钟再重试
- `stock_zh_index_value_csindex` 和 `index_stock_cons_weight_csindex` **共享同一域名配额**

---

## 三、按数据维度查找（使用指南）

### 3.1 指数历史K线（价格）

```
A股宽基(上证/深证/沪深300/中证500/中证1000等):
  ak.stock_zh_index_daily_em(symbol="sh000300")
  → 0.4s, 全历史(5178行, 2005至今), 含成交额

国证系列(创业板指/国证1000/国证2000等):
  ak.index_hist_cni(symbol="399006", start_date="20200101", end_date="20260504")
  → 0.1s, 需指定日期区间, ⚠️ 399305(国证2000)有bug

申万行业(31个一级行业):
  ak.index_hist_sw(symbol="801010", period="day")
  → 0.7s, 日/周/月K, 1999年至今
  注意: period参数替代了start_date/end_date

港股(恒生指数/国企指数/恒生科技):
  ak.stock_hk_index_daily_em(symbol="HSI")
  symbol: "HSI"(恒生), "HSCEI"(国企), "HSTECH"(恒生科技)

美股(道琼斯/纳斯达克/标普500):
  ak.index_us_stock_sina(symbol=".DJI")
  symbol: ".DJI"(道琼斯), ".IXIC"(纳指), ".INX"(标普500)

全球(日经/富时/DAX等):
  ak.index_global_hist_em(symbol="日经225指数")
  注意: symbol用中文名, 不支持代码
```

### 3.2 指数估值（PE/PB/股息率）

```
首选 — 申万行业估值(最全最稳):
  截面(当前值):
    ak.sw_index_first_info()           → 31个一级行业
    ak.sw_index_second_info()          → 131个二级行业
    ak.sw_index_third_info()           → 336个三级行业
    列: 静态市盈率, TTM(滚动)市盈率, 市净率, 静态股息率
    → 1.4s, 稳定

  时序(历史序列, 可算百分位):
    ak.index_analysis_daily_sw(symbol="一级行业", start_date="20221103", end_date="20260504")
    → 31行业 × 日度PE/PB/换手率/股息率 (2022-11至今)
    → 3s, 每次间隔≥1秒, 批量≤5次

次选 — CSIndex中证指数估值(有PE无PB, 有限流):
  ak.stock_zh_index_value_csindex(symbol="000300")
  → 仅中证系列(000xxx/Hxxxxx), 近20个交易日
  → 市盈率1(静态), 市盈率2(TTM), 股息率1, 股息率2
  → 无PB! 无历史百分位!
  → ⚠️ 间隔≥0.5s, 批量≤10, 日累计≤50
  → 注意: 与`index_stock_cons_weight_csindex`共享配额, 零间隔30次→403

备选 — 蛋卷基金 browser提取(Wind口径, 含百分位):
  URL: danjuanfunds.com/djmodule/value-center
  → JS提取63个指数PE/PB/ROE/股息率/10年百分位
  → 需browser 3-5分钟, 作为百分位的权威交叉验证源
```

### 3.3 指数成分股

```
中证系列(含权重, 但有CSIndex限流):
  ak.index_stock_cons_weight_csindex(symbol="000300")
  → 0.3s, 含权重百分比

东财(不限流, 但无权重, 大指数慢):
  ak.index_stock_cons(symbol="000300")
  → 300条=1.5s, 500条=3s, 1000条=5.7s, 2000条=11s

建议: 需要权重时用CSIndex(限流), 只需成分股列表时用东财
```

### 3.4 实时行情

```
A股全指数:    ak.stock_zh_index_spot_em()     → 268个, 2-3s
A股全指数:    ak.stock_zh_index_spot_sina()   → 562个, 1.5s
港股:         ak.stock_hk_index_spot_em()      → 356个, 3s
全球56指数:   ak.index_global_spot_em()        → 56个, 0.1s
```

---

## 四、PE/PB/股息率各数据源口径对比

| 来源 | 定位 | PE口径 | PB口径 | 股息率 | 百分位 | 覆盖 |
|------|------|--------|--------|--------|--------|------|
| 申万 | **首选** | 静态+TTM | MRQ | 静态股息率 | ❌无(需本地算) | 31一级行业+131二级行业 |
| CSIndex | **次选⚠️** | 静态+TTM | ❌无 | 过去12月+年度 | ❌无 | 中证系列指数 |
| 蛋卷(Wind) | 备选 | TTM(总量法) | MRQ | - | 10年百分位 | 63个主要指数 |
| 乐咕乐股 | 补充 | 多种 | 多种 | - | 5年/10年/全部历史 | 主要宽基+行业 |

**投资决策用哪个**:
- **PB → 申万/蛋卷/乐咕乐股 均可，跨源偏差 < 5%**
- **PE → 蛋卷(Wind总量法)最权威，CSIndex对亏损公司多的指数偏差可达86%**
- **百分位 → 蛋卷(10年) 或 本地积累申万数据自算**
- **行业估值 → 申万是唯一免费时序数据源，必须本地积累**
- **CSIndex → 仅用于中证指数成分股查询和行情数据，避免用于估值分析**

---

## 五、不可用 / 已废弃的接口 / 次选说明

| 接口 | 状态 | 替代/说明 |
|------|------|------|
| `index_value_hist_fund_sw` | 函数已不存在 | `index_analysis_daily_sw` |
| `index_hist_fund_sw` | 内部KeyError | `index_analysis_daily_sw` |
| `index_hist_cni("399305")` | Length mismatch | 无替代(国证2000代码有bug) |
| `stock_zh_index_value_csindex("399006")` | 404(国证代码不支持) | 申万 `index_analysis_daily_sw` |
| `stock_zh_index_value_csindex("HSHKI")` | 404(港股权重不支持) | 蛋卷基金 browser |
| `stock_zh_index_daily_tx` | 可用但极慢(14s) | `stock_zh_index_daily_em`(0.4s) |
| `index_realtime_sw` | 可用但极慢(23s) | `sw_index_first_info`(1.4s) |
| `index_global_hist_sina` | symbol必须用中文名, 易错 | `index_global_hist_em` |
| `index_zh_a_hist(symbol="000001")` | 首次22s冷启动 | `stock_zh_index_daily_em` 更快 |

### CSIndex 次选数据源说明

| 接口 | 定位 | 为什么是次选 |
|------|------|----------|
| `stock_zh_index_value_csindex` | 次选(估值) | 仅PE无PB，仅20日快照，零间隔30次→403 |
| `index_stock_cons_weight_csindex` | 次选(成分) | 与估值接口共享CSIndex配额，触发同一限流 |
| `stock_zh_index_hist_csindex` | 次选(行情) | 可用但非最优，东财`stock_zh_index_daily_em`更快更稳定 |

**使用规则**:
- 每次调用间隔 ≥ 0.5秒
- 单次批量拉取不超过 10 个指数
- 日累计不超过 50 次调用
- 被 403 后等待 10-15 分钟再重试
- 估值分析优先用申万，CSIndex仅作中证系列补充

---

## 六、API签名速查（与文档不一致的已标注⚠️）

```python
# === A股指数日K ===
ak.stock_zh_index_daily_em(symbol="sh000300")          # 全历史，最快0.4s
ak.index_zh_a_hist(symbol="000300", period="daily",    # 可指定区间，上证首次慢
                    start_date="20250101", end_date="20260504")

# === 国证指数 ===
ak.index_all_cni()                                      # 1406个指数列表，含PE
ak.index_hist_cni(symbol="399006", start_date="20200101", end_date="20260504")

# === CSIndex 次选数据源(⚠️有限流，零间隔30次→403) ===
ak.stock_zh_index_value_csindex(symbol="000300")        # ⚠️废弃了data_type参数，仅PE无PB
ak.index_stock_cons_weight_csindex(symbol="000300")     # ⚠️与估值接口共享配额

# === 申万行业(首选) ===
ak.sw_index_first_info()                                # 一级31行业PE/PB
ak.sw_index_second_info()                               # 二级131行业PE/PB
ak.index_analysis_daily_sw(symbol="一级行业",           # ⚠️symbol是枚举值非代码
                           start_date="20221103", end_date="20260504")
ak.index_hist_sw(symbol="801010", period="day")         # ⚠️period替代了日期参数

# === 港股 ===
ak.stock_hk_index_daily_em(symbol="HSI")                # ⚠️废弃了adjust参数
ak.stock_hk_index_spot_em()                             # ⚠️无参数

# === 美股 ===
ak.index_us_stock_sina(symbol=".DJI")

# === 全球 ===
ak.index_global_spot_em()                               # 56个指数实时
ak.index_global_hist_em(symbol="恒生指数")              # ⚠️中文名，无日期参数
```

---

## 七、推荐的数据获取流水线

### 日常估值检查（秒级）

```python
# 1. 申万行业当前PE/PB/股息率（首选）
df = ak.sw_index_first_info()                    # 1.4s → 31行业截面

# 2. A股指数实时行情
df = ak.stock_zh_index_spot_sina()               # 1.5s → 562指数最新价/涨跌幅
```

### 深度分析（分钟级）

```python
# 1. 申万行业历史PE/PB（首选，本地积累后可算百分位）
df = ak.index_analysis_daily_sw(symbol="一级行业", start_date="20221103", end_date="20260504")  # 3s

# 2. 指数日K线
df = ak.stock_zh_index_daily_em(symbol="sh000300")  # 0.4s
df = ak.index_hist_cni(symbol="399006", start_date="20200101", end_date="20260504")  # 0.1s

# 3. CSIndex 次选数据源（有限流，谨慎使用）
# ⚠️ 仅用于中证指数成分股查询，避免用于估值分析
for sym in ["000300", "000905", "000852", "000016", "000922", "H30269"]:
    df = ak.stock_zh_index_value_csindex(symbol=sym)  # 仅PE无PB，20日快照
    time.sleep(0.5)  # 必须！防403
```

### 估值百分位（需本地数据库）

```
无免费API直接返回历史百分位。方案：

方案A — 申万本地积累(首选，自动化):
  每日cron调用 index_analysis_daily_sw 拉取31行业PE/PB
  本地计算5年/10年百分位
  ⚠️ 仅覆盖申万一级行业, 不含宽基指数

方案B — 蛋卷基金 browser提取(备选, 3-5分钟):
  danjuanfunds.com/djmodule/value-center JS提取
  63个指数, Wind口径, 10年百分位, 权威

方案C — CSIndex PE每日快照(次选, 有限流):
  每日拉取一次, 本地积累, 自算百分位
  ⚠️ 仅中证系列, 仅PE(无PB), 有限流风险
  ⚠️ 零间隔30次→403, 冷却>10分钟
```
