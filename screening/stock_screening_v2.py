"""
基本面选股框架 v2 — 完整三层漏斗(修正版)
=========================================
修正: 取年报数据(12-31), 正确列名
扩大: 8个行业覆盖
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, akshare as ak
import time, sys

print("=" * 70)
print("  基本面选股框架 v2 — 三层漏斗实战(修正版)")
print("=" * 70)

# ════════════════════════════════════════════════════
# Phase 0: 全A基础数据
# ════════════════════════════════════════════════════
print("\n[Phase 0] 全A基础数据...", end=" ", flush=True)
all_a = ak.stock_zh_a_spot_em()
all_a["代码"] = all_a["代码"].astype(str).str.zfill(6)
all_a["总市值"] = pd.to_numeric(all_a["总市值"], errors="coerce")
all_a["市盈率-动态"] = pd.to_numeric(all_a["市盈率-动态"], errors="coerce")
all_a["市净率"] = pd.to_numeric(all_a["市净率"], errors="coerce")
print(f"OK ({len(all_a)}只)")

# ════════════════════════════════════════════════════
# Phase 1: 行业筛选 — 8个行业
# ════════════════════════════════════════════════════
print("\n[Phase 1] 行业筛选")
sectors = {
    "半导体": "科技成长(芯片国产化)",
    "消费电子": "科技成长(AI硬件)",
    "白色家电": "消费龙头(出海+高ROE)",
    "白酒": "消费升级(品牌壁垒)",
    "汽车零部件": "制造升级(新能源车)",
    "化学制药": "医疗健康(创新药)",
    "医疗器械": "医疗健康(国产替代)",
    "光伏": "新能源(全球需求)",
}

all_stocks = []
for sec, desc in sectors.items():
    print(f"  [{sec}] {desc}", end=" ... ", flush=True)
    try:
        df = ak.stock_board_industry_cons_em(symbol=sec)
        if len(df) == 0:
            print("空"); continue
        df["代码"] = df["代码"].astype(str).str.zfill(6)
        df["行业"] = sec
        all_stocks.append(df)
        n_large = 0
        if "总市值" in df.columns:
            n_large = len(df[pd.to_numeric(df["总市值"], errors="coerce") > 2e11])
        else:
            merged = df.merge(all_a[["代码","总市值"]], on="代码", how="left")
            n_large = len(merged[merged["总市值"] > 2e11])
        print(f"{len(df)}只 (>200亿: {n_large})")
    except Exception as e:
        print(f"FAIL: {str(e)[:50]}")
    time.sleep(0.3)

stocks = pd.concat(all_stocks, ignore_index=True)
stocks = stocks.merge(all_a[["代码","总市值","流通市值","市盈率-动态","市净率","60日涨跌幅","年初至今涨跌幅"]], 
                      on="代码", how="left")

# 过滤
stocks = stocks[~stocks["名称"].str.contains("ST|退", na=False)]
stocks = stocks[stocks["总市值"] > 2e11]  # >200亿
stocks = stocks.sort_values(["行业","总市值"], ascending=[True, False])

print(f"\n  预筛选: {len(stocks)}只 (>200亿, 非ST)")
for sec in sectors:
    sub = stocks[stocks["行业"]==sec]
    if len(sub) > 0:
        names = sub.nlargest(3, "总市值")["名称"].tolist()
        print(f"    {sec}: {len(sub)}只 — {', '.join(names)}")

# ════════════════════════════════════════════════════
# Phase 2: 财务指标获取 — 只取年报(12-31)
# ════════════════════════════════════════════════════
print(f"\n[Phase 2] 财务数据(年报) — {len(stocks)}只, 预计2分钟...")

codes = stocks["代码"].unique().tolist()
financial_data = []

for i, code in enumerate(codes):
    name = stocks[stocks["代码"]==code]["名称"].iloc[0]
    industry = stocks[stocks["代码"]==code]["行业"].iloc[0]
    mktcap = stocks[stocks["代码"]==code]["总市值"].iloc[0]
    
    sys.stdout.write(f"\r  [{i+1}/{len(codes)}] {code} {name:<6} ")
    sys.stdout.flush()
    
    try:
        df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2022")
        if df is None or len(df) == 0:
            print("无数据"); continue
        
        # 筛选年报
        annual = df[df["日期"].astype(str).str.contains("12-31")]
        if len(annual) == 0:
            # 如果没有年报数据，取最新一行
            latest = df.iloc[0]
        else:
            latest = annual.iloc[0]  # 最新年报
        
        def g(col):
            v = latest.get(col, np.nan)
            return pd.to_numeric(v, errors="coerce") if pd.notna(v) else np.nan
        
        # 获取两年年报做趋势
        if len(annual) >= 2:
            prev = annual.iloc[1]
            def g_prev(col):
                v = prev.get(col, np.nan)
                return pd.to_numeric(v, errors="coerce") if pd.notna(v) else np.nan
            rev_growth = g("主营业务收入增长率(%)")
            rev_growth_prev = g_prev("主营业务收入增长率(%)")
            profit_growth = g("净利润增长率(%)")
            profit_growth_prev = g_prev("净利润增长率(%)")
        else:
            rev_growth = g("主营业务收入增长率(%)")
            rev_growth_prev = np.nan
            profit_growth = g("净利润增长率(%)")
            profit_growth_prev = np.nan
        
        row = {
            "代码": code, "名称": name, "行业": industry, "总市值": mktcap,
            "ROE": g("净资产收益率(%)"),
            "净利率": g("销售净利率(%)"),
            "毛利率": g("销售毛利率(%)"),
            "资产周转率": g("总资产周转率(次)"),
            "资产负债率": g("资产负债率(%)"),
            "营收增速": rev_growth,
            "营收增速_前年": rev_growth_prev,
            "净利增速": profit_growth,
            "净利增速_前年": profit_growth_prev,
        }
        financial_data.append(row)
        
        roe_v = row["ROE"]
        npm_v = row["净利率"]
        s = f"ROE={roe_v:.1f}%" if pd.notna(roe_v) else "ROE=NA"
        s += f" 净利率={npm_v:.1f}%" if pd.notna(npm_v) else ""
        s += f" 营收+{row['营收增速']:.0f}%" if pd.notna(row["营收增速"]) else ""
        print(s)
        
    except Exception as e:
        print(f"ERR:{str(e)[:30]}")
    
    time.sleep(0.25)

fin_df = pd.DataFrame(financial_data)
print(f"\n  成功: {len(fin_df)}只")

# 重新合并PE等实时数据
fin_df = fin_df.merge(all_a[["代码","市盈率-动态","市净率","60日涨跌幅","年初至今涨跌幅"]], on="代码", how="left")

# ════════════════════════════════════════════════════
# Phase 2.5: 杜邦评分(满分25)
# ════════════════════════════════════════════════════
print("\n[Phase 2.5] 杜邦拆解评分(满分25)")

def score_duPont(r):
    s = 0
    # ROE(5分): >15=5, >12=4, >10=3, >8=2, >5=1
    if pd.notna(r["ROE"]):
        if r["ROE"]>=15: s+=5
        elif r["ROE"]>=12: s+=4
        elif r["ROE"]>=10: s+=3
        elif r["ROE"]>=8: s+=2
        elif r["ROE"]>=5: s+=1
    # 净利率(5分): >20=5, >15=4, >10=3, >5=2, >0=1
    if pd.notna(r["净利率"]):
        if r["净利率"]>=20: s+=5
        elif r["净利率"]>=15: s+=4
        elif r["净利率"]>=10: s+=3
        elif r["净利率"]>=5: s+=2
        elif r["净利率"]>=0: s+=1
    # 毛利率(5分): >50=5, >40=4, >30=3, >20=2, >0=1
    if pd.notna(r["毛利率"]):
        if r["毛利率"]>=50: s+=5
        elif r["毛利率"]>=40: s+=4
        elif r["毛利率"]>=30: s+=3
        elif r["毛利率"]>=20: s+=2
        elif r["毛利率"]>=0: s+=1
    # 周转率(5分): >1=5, >0.7=4, >0.5=3, >0.3=2, >0=1
    if pd.notna(r["资产周转率"]):
        if r["资产周转率"]>=1: s+=5
        elif r["资产周转率"]>=0.7: s+=4
        elif r["资产周转率"]>=0.5: s+=3
        elif r["资产周转率"]>=0.3: s+=2
        elif r["资产周转率"]>0: s+=1
    # 负债率(5分): 30-60=5, 20-70=4, 10-80=3, else=1
    if pd.notna(r["资产负债率"]):
        if 30<=r["资产负债率"]<=60: s+=5
        elif 20<=r["资产负债率"]<=70: s+=4
        elif 10<=r["资产负债率"]<=80: s+=3
        else: s+=1
    return s

fin_df["杜邦分"] = fin_df.apply(score_duPont, axis=1)

# ════════════════════════════════════════════════════
# Phase 3: 催化剂评分(满分15)
# ════════════════════════════════════════════════════
print("[Phase 3] 催化剂评分(满分15)")

def score_catalyst(r):
    s = 0
    # 营收增速(5分): >30=5, >20=4, >15=3, >10=2, >0=1
    if pd.notna(r["营收增速"]):
        if r["营收增速"]>=30: s+=5
        elif r["营收增速"]>=20: s+=4
        elif r["营收增速"]>=15: s+=3
        elif r["营收增速"]>=10: s+=2
        elif r["营收增速"]>=0: s+=1
    # 净利增速 vs 营收增速(经营杠杆, 5分)
    rg = r.get("营收增速", np.nan)
    pg = r.get("净利增速", np.nan)
    if pd.notna(rg) and pd.notna(pg):
        if pg > rg*1.2: s+=5
        elif pg > rg: s+=4
        elif pg > 0: s+=2
    elif pd.notna(pg) and pg > 0: s+=1
    # PE合理(5分): 10-25=5, 8-30=4, 5-40=3, else=1
    pe = r.get("市盈率-动态", np.nan)
    if pd.notna(pe) and pe > 0:
        if 10<=pe<=25: s+=5
        elif 8<=pe<=30: s+=4
        elif 5<=pe<=40: s+=3
        else: s+=1
    return s

fin_df["催化分"] = fin_df.apply(score_catalyst, axis=1)
fin_df["总分"] = fin_df["杜邦分"] + fin_df["催化分"]
fin_df = fin_df.sort_values("总分", ascending=False)

# ════════════════════════════════════════════════════
# 输出结果
# ════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(f"  综合排名 TOP 30 (满分40 = 杜邦25 + 催化15)")
print("=" * 80)

hdr = "{:<3} {:>6} {:<6} {:<10} {:>5} {:>5} {:>5} {:>5} {:>4} {:>5} {:>3} {:>3} {:<30}"
print(hdr.format("#","代码","行业","名称","市值亿","ROE%","净利率","营收+","PE","杜邦","催化","总分","亮点"))
print("-" * 100)

for rank, (_, r) in enumerate(fin_df.head(30).iterrows(), 1):
    m = f"{r['总市值']/1e8:.0f}"
    roe = f"{r['ROE']:.1f}" if pd.notna(r["ROE"]) else "-"
    npm = f"{r['净利率']:.1f}" if pd.notna(r["净利率"]) else "-"
    rg = f"{r['营收增速']:.0f}" if pd.notna(r["营收增速"]) else "-"
    pe = f"{r['市盈率-动态']:.1f}" if pd.notna(r["市盈率-动态"]) and r["市盈率-动态"]>0 else "-"
    
    # 亮点
    hl = []
    if pd.notna(r["ROE"]) and r["ROE"]>=15: hl.append(f"高ROE{r['ROE']:.0f}%")
    if pd.notna(r["营收增速"]) and r["营收增速"]>=15: hl.append(f"营收+{r['营收增速']:.0f}%")
    if pd.notna(r["净利增速"]) and r["净利增速"]>r["营收增速"] and pd.notna(r["营收增速"]): hl.append("经营杠杆")
    if pd.notna(r["市盈率-动态"]) and 10<=r["市盈率-动态"]<=25: hl.append("PE合理")
    if pd.notna(r["毛利率"]) and r["毛利率"]>=40: hl.append(f"高毛利{r['毛利率']:.0f}%")
    
    print(hdr.format(rank, r["代码"], r["行业"], r["名称"], m, roe, npm, rg, pe,
                     r["杜邦分"], r["催化分"], r["总分"], ", ".join(hl)))

# 分行业统计
print("\n\n── 各行业TOP3 ──")
for sec in sectors:
    sub = fin_df[fin_df["行业"]==sec].head(3)
    if len(sub) == 0: continue
    print(f"\n  【{sec}】")
    for _, r in sub.iterrows():
        pe_s = f"PE={r['市盈率-动态']:.1f}" if pd.notna(r["市盈率-动态"]) else "PE=NA"
        rg_s = f"营收+{r['营收增速']:.0f}%" if pd.notna(r["营收增速"]) else ""
        pg_s = f"净利+{r['净利增速']:.0f}%" if pd.notna(r["净利增速"]) else ""
        print(f"    {r['名称']}({r['代码']}) 总分{r['总分']} 杜邦{r['杜邦分']} 催化{r['催化分']} | {pe_s} {rg_s} {pg_s}")

# 最终推荐
print("\n" + "=" * 70)
print("  ★ 最终推荐池: 总分>=25 且 ROE>=12%")
print("=" * 70)
rec = fin_df[(fin_df["总分"]>=25)]
if len(rec) == 0:
    print("  放宽: 总分>=22 且 ROE>=10%")
    rec = fin_df[(fin_df["总分"]>=22)]
if len(rec) > 0:
    for _, r in rec.iterrows():
        pe_s = f"PE={r['市盈率-动态']:.1f}" if pd.notna(r["市盈率-动态"]) else "PE=NA"
        rg_s = f"营收+{r['营收增速']:.0f}%" if pd.notna(r["营收增速"]) else ""
        print(f"  ★ {r['名称']}({r['代码']}) [{r['行业']}] 市值{r['总市值']/1e8:.0f}亿 总分{r['总分']} | ROE={r['ROE']:.1f}% {pe_s} {rg_s}")

# 保存
fin_df.to_csv("/home/andy/backtest/screening_result_v2.csv", index=False, encoding="utf-8-sig")
print(f"\n  完整数据: /home/andy/backtest/screening_result_v2.csv")
