"""
基本面选股框架 — 完整三层漏斗
================================
Phase 1: 行业筛选（5个代表行业，市值>200亿预过滤）
Phase 2: 龙头筛选（杜邦拆解：ROE/净利率/周转率/杠杆）
Phase 3: 催化剂筛选（营收增速加速 + 估值安全边际）
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, akshare as ak
import time, sys

print("=" * 65)
print("基本面选股框架 — 三层漏斗实战")
print("=" * 65)

# ════════════════════════════════════════════════════
# Phase 0: 获取全A实时数据（市值+PE）
# ════════════════════════════════════════════════════
print("\n── Phase 0: 全A基础数据 ──")
print("  拉取全A股行情...", end=" ", flush=True)
all_a = ak.stock_zh_a_spot_em()
print(f"OK ({len(all_a)}只)")

# 标准化代码为6位
all_a["代码"] = all_a["代码"].astype(str).str.zfill(6)
all_a["总市值"] = pd.to_numeric(all_a["总市值"], errors="coerce")
all_a["流通市值"] = pd.to_numeric(all_a["流通市值"], errors="coerce")
all_a["市盈率-动态"] = pd.to_numeric(all_a["市盈率-动态"], errors="coerce")
all_a["市净率"] = pd.to_numeric(all_a["市净率"], errors="coerce")

# ════════════════════════════════════════════════════
# Phase 1: 行业筛选
# ════════════════════════════════════════════════════
print("\n── Phase 1: 行业筛选 ──")
target_sectors = ["半导体", "消费电子", "白色家电", "汽车零部件", "中药"]
# 备选（中药可能失败）：医药
fallback_sectors = ["医药"]

sector_stocks = {}
for sec in target_sectors + fallback_sectors:
    if sec in sector_stocks:
        continue
    print(f"  [{sec}]", end=" ", flush=True)
    try:
        df = ak.stock_board_industry_cons_em(symbol=sec)
        if len(df) == 0:
            print("空")
            continue
        df["代码"] = df["代码"].astype(str).str.zfill(6)
        df["行业"] = sec
        # 合并市值
        df = df.merge(all_a[["代码","总市值","流通市值","市盈率-动态","市净率"]], on="代码", how="left")
        sector_stocks[sec] = df
        print(f"{len(df)}只, >200亿: {len(df[df['总市值']>2e11])}只")
    except Exception as e:
        print(f"FAIL: {e}")
    time.sleep(0.5)

# 合并所有行业
stocks = pd.concat(sector_stocks.values(), ignore_index=True)
# 过滤
stocks = stocks[~stocks["名称"].str.contains("ST", na=False)]
stocks = stocks[stocks["总市值"] > 2e11]  # >200亿

print(f"\n  预筛选: {len(stocks)}只 (>200亿, 非ST)")
for sec, df in sector_stocks.items():
    n = len(df[(df["总市值"]>2e11) & (~df["名称"].str.contains("ST",na=False))])
    if n > 0:
        top3 = df[(df["总市值"]>2e11)].nlargest(3, "总市值")["名称"].tolist()
        print(f"    {sec}: {n}只 — {', '.join(top3)}")

# ════════════════════════════════════════════════════
# Phase 2: 龙头筛选 — 获取财务指标
# 用 stock_financial_analysis_indicator 获取ROE、净利率等
# ════════════════════════════════════════════════════
print("\n── Phase 2: 财务指标获取(ROE/净利率/周转率) ──")
print("  逐只拉取，预计2-3分钟...\n")

codes = stocks["代码"].unique().tolist()
print(f"  待处理: {len(codes)} 只股票")

financial_data = []
for i, code in enumerate(codes):
    name = stocks[stocks["代码"]==code]["名称"].iloc[0]
    industry = stocks[stocks["代码"]==code]["行业"].iloc[0]
    mktcap = stocks[stocks["代码"]==code]["总市值"].iloc[0]
    
    bar = f"  [{i+1}/{len(codes)}] {code} {name}"
    print(bar, end=" ", flush=True)
    
    try:
        df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2022")
        if df is None or len(df) == 0:
            print("无数据")
            continue
        
        # 取最近一年(最新年报)
        latest = df.iloc[0]
        
        row = {
            "代码": code,
            "名称": name,
            "行业": industry,
            "总市值": mktcap,
            "ROE": pd.to_numeric(latest.get("净资产收益率(%)", np.nan), errors="coerce"),
            "净利率": pd.to_numeric(latest.get("销售净利率(%)", np.nan), errors="coerce"),
            "毛利率": pd.to_numeric(latest.get("销售毛利率(%)", np.nan), errors="coerce"),
            "资产周转率": pd.to_numeric(latest.get("总资产周转率(次)", np.nan), errors="coerce"),
            "资产负债率": pd.to_numeric(latest.get("资产负债率(%)", np.nan), errors="coerce"),
            "营收增速": pd.to_numeric(latest.get("营业总收入同比增长率(%)", np.nan), errors="coerce"),
            "净利增速": pd.to_numeric(latest.get("净利润同比增长率(%)", np.nan), errors="coerce"),
        }
        financial_data.append(row)
        
        # 简要输出
        roe_val = row["ROE"]
        rev_g = row["营收增速"]
        if pd.notna(roe_val):
            print(f"ROE={roe_val:.1f}%, 营收增速={rev_g:.1f}%" if pd.notna(rev_g) else f"ROE={roe_val:.1f}%")
        else:
            print("OK(无ROE)")
            
    except Exception as e:
        print(f"ERR: {str(e)[:40]}")
    
    time.sleep(0.3)

fin_df = pd.DataFrame(financial_data)
print(f"\n  成功获取: {len(fin_df)} 只股票的财务数据")

# ════════════════════════════════════════════════════
# Phase 2.5: 杜邦拆解评分
# ════════════════════════════════════════════════════
print("\n── Phase 2.5: 杜邦拆解评分 ──")

def score_duPont(row):
    """杜邦拆解评分(0-5分)"""
    score = 0
    details = []
    
    # ROE: >15%=5, >12%=4, >10%=3, >8%=2, >5%=1
    roe = row.get("ROE", np.nan)
    if pd.notna(roe):
        if roe >= 15: score += 5; details.append(f"ROE={roe:.1f}%[5]")
        elif roe >= 12: score += 4; details.append(f"ROE={roe:.1f}%[4]")
        elif roe >= 10: score += 3; details.append(f"ROE={roe:.1f}%[3]")
        elif roe >= 8: score += 2; details.append(f"ROE={roe:.1f}%[2]")
        elif roe >= 5: score += 1; details.append(f"ROE={roe:.1f}%[1]")
    else:
        details.append("ROE=NA[0]")
    
    # 净利率: >20%=5, >15%=4, >10%=3, >5%=2, >0%=1
    npm = row.get("净利率", np.nan)
    if pd.notna(npm):
        if npm >= 20: score += 5; details.append(f"净利率={npm:.1f}%[5]")
        elif npm >= 15: score += 4; details.append(f"净利率={npm:.1f}%[4]")
        elif npm >= 10: score += 3; details.append(f"净利率={npm:.1f}%[3]")
        elif npm >= 5: score += 2; details.append(f"净利率={npm:.1f}%[2]")
        elif npm >= 0: score += 1; details.append(f"净利率={npm:.1f}%[1]")
    
    # 资产周转率: >1.0=5, >0.7=4, >0.5=3, >0.3=2, >0=1
    ato = row.get("资产周转率", np.nan)
    if pd.notna(ato):
        if ato >= 1.0: score += 5; details.append(f"周转={ato:.2f}[5]")
        elif ato >= 0.7: score += 4; details.append(f"周转={ato:.2f}[4]")
        elif ato >= 0.5: score += 3; details.append(f"周转={ato:.2f}[3]")
        elif ato >= 0.3: score += 2; details.append(f"周转={ato:.2f}[2]")
        elif ato > 0: score += 1; details.append(f"周转={ato:.2f}[1]")
    
    # 资产负债率: 30-60%=5, 20-70%=4, 10-80%=3, 其他=1
    de = row.get("资产负债率", np.nan)
    if pd.notna(de):
        if 30 <= de <= 60: score += 5; details.append(f"负债率={de:.1f}%[5]")
        elif 20 <= de <= 70: score += 4; details.append(f"负债率={de:.1f}%[4]")
        elif 10 <= de <= 80: score += 3; details.append(f"负债率={de:.1f}%[3]")
        else: score += 1; details.append(f"负债率={de:.1f}%[1]")
    
    # 毛利率: >50%=5, >40%=4, >30%=3, >20%=2, >0%=1 (品牌/壁垒指标)
    gpm = row.get("毛利率", np.nan)
    if pd.notna(gpm):
        if gpm >= 50: score += 5; details.append(f"毛利率={gpm:.1f}%[5]")
        elif gpm >= 40: score += 4; details.append(f"毛利率={gpm:.1f}%[4]")
        elif gpm >= 30: score += 3; details.append(f"毛利率={gpm:.1f}%[3]")
        elif gpm >= 20: score += 2; details.append(f"毛利率={gpm:.1f}%[2]")
        elif gpm >= 0: score += 1; details.append(f"毛利率={gpm:.1f}%[1]")
    
    return score, "; ".join(details)

fin_df["杜邦分"], fin_df["杜邦详情"] = zip(*fin_df.apply(score_duPont, axis=1))
fin_df = fin_df.sort_values("杜邦分", ascending=False)

print("\n  杜邦拆解评分 TOP 20:")
print("-" * 80)
fmt = "{:>6} {:<8} {:<12} {:>6} {:>8} {:>8} {:>8} {:>5}  {}"
print(fmt.format("代码","行业","名称","总市值","ROE%","净利率%","毛利率%","杜邦分","详情"))
for _, r in fin_df.head(20).iterrows():
    mktcap_str = f"{r['总市值']/1e8:.0f}亿"
    roe_s = f"{r['ROE']:.1f}" if pd.notna(r['ROE']) else "NA"
    npm_s = f"{r['净利率']:.1f}" if pd.notna(r['净利率']) else "NA"
    gpm_s = f"{r['毛利率']:.1f}" if pd.notna(r['毛利率']) else "NA"
    print(fmt.format(r['代码'], r['行业'], r['名称'], mktcap_str, roe_s, npm_s, gpm_s, r['杜邦分'], r['杜邦详情'][:50]))

# ════════════════════════════════════════════════════
# Phase 3: 催化剂筛选
# 条件: 营收增速>10% 且 净利润增速>营收增速(经营杠杆)
# 估值: PE在合理区间
# ════════════════════════════════════════════════════
print("\n\n── Phase 3: 催化剂筛选 ──")

# 合并PE数据
pe_data = all_a[["代码","市盈率-动态","市净率","60日涨跌幅","年初至今涨跌幅"]].copy()
pe_data["代码"] = pe_data["代码"].astype(str).str.zfill(6)
fin_df = fin_df.merge(pe_data, on="代码", how="left")

# 催化剂评分
def score_catalyst(row):
    """催化剂评分(0-5分)"""
    score = 0
    details = []
    
    # 营收增速: >30%=5, >20%=4, >15%=3, >10%=2, >0%=1
    rg = row.get("营收增速", np.nan)
    if pd.notna(rg):
        if rg >= 30: score += 5; details.append(f"营收+{rg:.0f}%[5]")
        elif rg >= 20: score += 4; details.append(f"营收+{rg:.0f}%[4]")
        elif rg >= 15: score += 3; details.append(f"营收+{rg:.0f}%[3]")
        elif rg >= 10: score += 2; details.append(f"营收+{rg:.0f}%[2]")
        elif rg >= 0: score += 1; details.append(f"营收+{rg:.0f}%[1]")
        else: score += 0; details.append(f"营收{rg:.0f}%[0]")
    
    # 净利增速 vs 营收增速: 净利>营收(经营杠杆)=加2分
    pg = row.get("净利增速", np.nan)
    if pd.notna(rg) and pd.notna(pg):
        if pg > rg * 1.2: score += 3; details.append(f"净利>营收[3]")
        elif pg > rg: score += 2; details.append(f"净利>营收[2]")
        elif pg > 0: score += 1; details.append(f"净利+{pg:.0f}%[1]")
        else: details.append(f"净利{pg:.0f}%[0]")
    
    # PE合理: 10-25=5, 8-30=4, 5-40=3, 其他=1
    pe = row.get("市盈率-动态", np.nan)
    if pd.notna(pe) and pe > 0:
        if 10 <= pe <= 25: score += 5; details.append(f"PE={pe:.1f}[5]")
        elif 8 <= pe <= 30: score += 4; details.append(f"PE={pe:.1f}[4]")
        elif 5 <= pe <= 40: score += 3; details.append(f"PE={pe:.1f}[3]")
        elif pe > 0: score += 1; details.append(f"PE={pe:.1f}[1]")
    
    return score, "; ".join(details)

fin_df["催化分"], fin_df["催化详情"] = zip(*fin_df.apply(score_catalyst, axis=1))

# ════════════════════════════════════════════════════
# 综合评分
# ════════════════════════════════════════════════════
fin_df["总分"] = fin_df["杜邦分"] + fin_df["催化分"]
fin_df = fin_df.sort_values("总分", ascending=False)

print("\n" + "=" * 85)
print("综合评分 TOP 25 (杜邦分 + 催化分)")
print("=" * 85)
fmt2 = "{:<3} {:>6} {:<8} {:<10} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5} {:>4}  {}"
print(fmt2.format("#","代码","行业","名称","市值","ROE%","净利%","营收+","PE","杜邦","催化","催化详情"))
for rank, (_, r) in enumerate(fin_df.head(25).iterrows(), 1):
    mktcap_str = f"{r['总市值']/1e8:.0f}亿"
    roe_s = f"{r['ROE']:.1f}" if pd.notna(r['ROE']) else "-"
    npm_s = f"{r['净利率']:.1f}" if pd.notna(r['净利率']) else "-"
    rg_s = f"{r['营收增速']:.0f}" if pd.notna(r['营收增速']) else "-"
    pe_s = f"{r['市盈率-动态']:.1f}" if pd.notna(r['市盈率-动态']) and r['市盈率-动态']>0 else "-"
    print(fmt2.format(rank, r['代码'], r['行业'], r['名称'], mktcap_str, 
                      roe_s, npm_s, rg_s, pe_s, r['杜邦分'], r['催化分'], 
                      r['催化详情'][:35]))

# 分行业统计
print("\n\n── 分行业入选统计(总分>=15) ──")
qualified = fin_df[fin_df["总分"] >= 15]
for sec in qualified["行业"].unique():
    sub = qualified[qualified["行业"] == sec].sort_values("总分", ascending=False)
    names = "、".join(sub["名称"].tolist())
    print(f"\n  【{sec}】{len(sub)}只入选:")
    for _, r in sub.iterrows():
        pe_s = f"PE={r['市盈率-动态']:.1f}" if pd.notna(r['市盈率-动态']) else "PE=N/A"
        rg_s = f"营收+{r['营收增速']:.0f}%" if pd.notna(r['营收增速']) else ""
        pg_s = f"净利+{r['净利增速']:.0f}%" if pd.notna(r['净利增速']) else ""
        print(f"    {r['名称']}({r['代码']}) 总分{r['总分']} 杜邦{r['杜邦分']} 催化{r['催化分']} | {pe_s} {rg_s} {pg_s}")

# 保存结果
fin_df.to_csv("/home/andy/backtest/screening_result.csv", index=False, encoding="utf-8-sig")
print(f"\n  结果已保存: /home/andy/backtest/screening_result.csv")

# 最终推荐池
print("\n" + "=" * 65)
print("最终推荐池: 总分>=15 且 杜邦分>=12 且 催化分>=4")
print("=" * 65)
final = fin_df[(fin_df["总分"]>=15) & (fin_df["杜邦分"]>=12) & (fin_df["催化分"]>=4)]
final = final.sort_values("总分", ascending=False)
if len(final) > 0:
    for _, r in final.iterrows():
        pe_s = f"PE={r['市盈率-动态']:.1f}" if pd.notna(r['市盈率-动态']) else "PE=N/A"
        print(f"  ★ {r['名称']}({r['代码']}) [{r['行业']}] 市值{r['总市值']/1e8:.0f}亿 总分{r['总分']} | {pe_s}")
else:
    print("  无股票满足条件，放宽到 总分>=14 且 杜邦分>=10 且 催化分>=3")
    final = fin_df[(fin_df["总分"]>=14) & (fin_df["杜邦分"]>=10) & (fin_df["催化分"]>=3)]
    final = final.sort_values("总分", ascending=False)
    for _, r in final.iterrows():
        pe_s = f"PE={r['市盈率-动态']:.1f}" if pd.notna(r['市盈率-动态']) else "PE=N/A"
        print(f"  ★ {r['名称']}({r['代码']}) [{r['行业']}] 市值{r['总市值']/1e8:.0f}亿 总分{r['总分']} | {pe_s}")
