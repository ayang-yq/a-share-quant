"""
基本面选股框架 — 三层漏斗实战
================================
第一层：行业筛选（选赛道）
第二层：龙头筛选（杜邦拆解）
第三层：催化剂筛选（择时）
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, akshare as ak
import time, sys

print("=" * 65)
print("基本面选股框架 — 实战选股")
print("=" * 65)

# ════════════════════════════════════════════════════
# Phase 1: 行业筛选 — 从申万一级行业中选赛道
# ════════════════════════════════════════════════════
print("\n── Phase 1: 行业筛选 ──\n")

# 获取申万一级行业涨跌幅(近3年)
print("  获取申万一级行业数据...", end=" ", flush=True)
try:
    sectors = ak.stock_board_industry_name_em()
    print(f"OK ({len(sectors)}个行业)")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# 显示行业列表前10个看格式
print(f"  列名: {sectors.columns.tolist()}")
print(sectors.head(3).to_string())
