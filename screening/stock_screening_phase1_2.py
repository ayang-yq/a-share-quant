"""
基本面选股框架 Phase 1-2: 行业筛选 + 龙头筛选
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, akshare as ak
import time, sys, json

print("=" * 65)
print("基本面选股框架 — 实战选股")
print("=" * 65)

# ════════════════════════════════════════════════════
# Phase 1: 行业筛选
# 思路：选取5个代表不同风格维度的行业
#   成长：电子（半导体/AI硬件）
#   成长：计算机（AI软件）
#   消费稳定：家用电器
#   医疗健康：医药生物
#   制造升级：汽车
# 排除：银行/非银金融（纯红利，已在红利ETF覆盖）
#       基建/地产（周期下行）
#       传媒/教育（政策风险）
# ════════════════════════════════════════════════════
print("\n── Phase 1: 行业筛选 ──")
target_sectors = ["半导体", "消费电子", "白色家电", "汽车零部件", "中药"]

# 获取每个行业的成分股
all_stocks = []
for sec in target_sectors:
    print(f"  获取 [{sec}] 成分股...", end=" ", flush=True)
    try:
        df = ak.stock_board_industry_cons_em(symbol=sec)
        df["行业"] = sec
        all_stocks.append(df)
        print(f"OK ({len(df)}只)")
    except Exception as e:
        print(f"FAIL: {e}")
    time.sleep(0.5)

stocks = pd.concat(all_stocks, ignore_index=True)
print(f"\n  合计 {len(stocks)} 只股票")

# 看列名
print(f"  列名: {stocks.columns.tolist()}")
print(stocks[["行业","名称","代码","总市值"]].head(10).to_string())

# ════════════════════════════════════════════════════
# Phase 2: 预筛选 — 市值>200亿 + 非ST
# ════════════════════════════════════════════════════
print("\n── Phase 2: 预筛选(市值>200亿, 非ST) ──")

# 总市值列，转数值
stocks["总市值"] = pd.to_numeric(stocks["总市值"], errors="coerce")

# 过滤ST
stocks = stocks[~stocks["名称"].str.contains("ST", na=False)]
# 市值>200亿
stocks = stocks[stocks["总市值"] > 2e10]
stocks = stocks.sort_values(["行业", "总市值"], ascending=[True, False])

print(f"  过滤后 {len(stocks)} 只")
for sec in target_sectors:
    sub = stocks[stocks["行业"] == sec]
    print(f"    {sec}: {len(sub)}只 — {', '.join(sub['名称'].head(5).tolist())}")

# 保存中间结果
stocks[["代码","名称","行业","总市值"]].to_csv("/home/andy/backtest/phase2_filtered.csv", index=False)
print(f"\n  中间结果保存到 phase2_filtered.csv")
