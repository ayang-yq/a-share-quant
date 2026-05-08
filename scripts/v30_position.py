"""
v4.3.1 仓位计算脚本 — cron每日调用
====================================
v3.0择时: Pos = 0.7 * Price_BB_z + 0.3 * Margin_BB_z
v2风格:   申万小/大比 60日分位 → split (极端10%/90%, 中间50%)
映射:     三态插值 + 风格偏好调整 → 5大类权重(红利/沪深300/国证2000/进攻/类现金)
调仓建议: 迟滞带三态 + 漂移阈值3% → 是否需要调仓

输出: JSON到stdout, cron job读取后判断是否推送
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, os
from pathlib import Path
import pandas as pd, numpy as np
import akshare as ak

STATE_FILE = Path(__file__).parent / "rebalance_state.json"
DRIFT_THRESHOLD = 0.03  # 3%漂移阈值
TODAY = pd.Timestamp.now().normalize()

# ── 判断今天是否可能是交易日(排除周末) ──
weekday = TODAY.weekday()
if weekday >= 5:
    print(json.dumps({"status": "skip", "reason": f"周末({TODAY.strftime('%Y-%m-%d')})"}))
    sys.exit(0)

# ── 1. 获取沪深300日K ──
try:
    df_price = ak.stock_zh_index_daily(symbol="sh000300")[["date","close"]].rename(columns={"close":"hs300"})
    df_price["date"] = pd.to_datetime(df_price["date"])
    df_price = df_price.sort_values("date").reset_index(drop=True)
except Exception as e:
    print(json.dumps({"status": "error", "reason": f"沪深300数据失败: {e}"}))
    sys.exit(1)

# ── 2. 获取融资买入额(沪+深合计) ──
try:
    df_msh = ak.macro_china_market_margin_sh()
    df_msz = ak.macro_china_market_margin_sz()
    df_msh["date"] = pd.to_datetime(df_msh["日期"])
    df_msz["date"] = pd.to_datetime(df_msz["日期"])
    df_msh2 = df_msh[["date","融资买入额"]].rename(columns={"融资买入额":"rzye"})
    df_msz2 = df_msz[["date","融资买入额"]].rename(columns={"融资买入额":"rzye"})
    df_margin = df_msh2.merge(df_msz2, on="date", how="outer", suffixes=("_sh","_sz"))
    df_margin = df_margin.sort_values("date").reset_index(drop=True)
    df_margin["rzye_total"] = pd.to_numeric(df_margin["rzye_sh"], errors="coerce") + pd.to_numeric(df_margin["rzye_sz"], errors="coerce")
except Exception as e:
    print(json.dumps({"status": "error", "reason": f"融资数据失败: {e}"}))
    sys.exit(1)

# ── 3. 获取申万风格指数(大盘801811 + 小盘801813) ──
try:
    sw_dp = ak.index_hist_sw(symbol="801811", period="day")[["日期","收盘"]].rename(columns={"日期":"date","收盘":"sw_large"})
    sw_xp = ak.index_hist_sw(symbol="801813", period="day")[["日期","收盘"]].rename(columns={"日期":"date","收盘":"sw_small"})
    sw_dp["date"] = pd.to_datetime(sw_dp["date"])
    sw_xp["date"] = pd.to_datetime(sw_xp["date"])
    sw_style = sw_dp.merge(sw_xp, on="date", how="outer")
    sw_style = sw_style.sort_values("date").reset_index(drop=True)
except Exception as e:
    print(json.dumps({"status": "error", "reason": f"申万风格指数失败: {e}"}))
    sys.exit(1)

# ── 4. 合并 ──
df = df_price.merge(df_margin[["date","rzye_total"]], on="date", how="left")
df = df.merge(sw_style, on="date", how="left")
df["rzye_total"] = df["rzye_total"].ffill()
df["sw_large"] = df["sw_large"].ffill()
df["sw_small"] = df["sw_small"].ffill()

# ── 检查数据是否已更新到今天 ──
latest_date = df["date"].max()
if latest_date < TODAY:
    print(json.dumps({"status": "skip", "reason": f"数据未更新, 最新{latest_date.strftime('%Y-%m-%d')}, 今天{TODAY.strftime('%Y-%m-%d')}"}))
    sys.exit(0)

# ── 5. v3.0择时层 ──
W = 20
df["bb_ma"] = df["hs300"].rolling(W).mean()
df["bb_std"] = df["hs300"].rolling(W).std()
df["bb_z"] = (df["hs300"] - df["bb_ma"]) / df["bb_std"]

df["rzye_ma"] = df["rzye_total"].rolling(W).mean()
df["rzye_std"] = df["rzye_total"].rolling(W).std()
df["rzye_z"] = (df["rzye_total"] - df["rzye_ma"]) / df["rzye_std"]

def z2p(z, lo=0.20, hi=0.80):
    z = float(np.clip(z, -3, 3))
    return 0.50 + z / 3 * (hi - 0.50)

latest = df.iloc[-1]
date_str = latest["date"].strftime("%Y-%m-%d")
close = float(latest["hs300"])
bb_z = float(latest["bb_z"])
rzye_z = float(latest["rzye_z"])

pos_price = z2p(bb_z) * 0.70
pos_margin = z2p(rzye_z) * 0.30
pos_total = pos_price + pos_margin

# ── 6. v2风格层 ──
STYLE_W = 60
LO_THRESH, HI_THRESH = 0.20, 0.80  # 极端阈值

df["ratio_xd"] = df["sw_small"] / df["sw_large"]
df["pct_xd"] = df["ratio_xd"].rolling(STYLE_W).rank(pct=True)

latest = df.iloc[-1]  # re-fetch after style columns added
pct = float(latest["pct_xd"])
ratio = float(latest["ratio_xd"])

# 趋势方向: 小盘强(pct高) → split低(加仓2000)
if pd.isna(pct):
    split = 0.50
    style_level = "数据不足"
elif pct > HI_THRESH:
    split = 0.10  # 小盘极端强, 重仓国证2000
    style_level = "小盘极端强"
elif pct < LO_THRESH:
    split = 0.90  # 大盘极端强, 重仓沪深300
else:
    split = 0.50
    style_level = "均衡"

# ── 7. 三层映射 → 5大类权重 ──

# 三态配置表(含宽基合计用于插值，拆分后单独输出300/2000)
CFG = {
    "红利":    {"防御": 0.20, "均衡": 0.15, "进攻": 0.10, "lo": 0.05, "hi": 0.25},
    "宽基合计": {"防御": 0.10, "均衡": 0.15, "进攻": 0.20, "lo": 0.02, "hi": 0.25},
    "进攻":    {"防御": 0.05, "均衡": 0.25, "进攻": 0.45, "lo": 0.05, "hi": 0.45},
}

# ── 迟滞带三态判定 ──
# 上升/下降用不同阈值，消除阈值附近振荡
HYSTERESIS = {
    "up":   {"防御→均衡": 0.45, "均衡→进攻": 0.70},  # 上升阈值(更高)
    "down": {"进攻→均衡": 0.60, "均衡→防御": 0.35},  # 下降阈值(更低)
}
STATE_ORDER = ["防御", "均衡", "进攻"]  # 0, 1, 2

def load_state():
    """读取上次状态"""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return None

def save_state(state):
    """保存当前状态"""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def hysteresis_state(pos, prev_state_name):
    """迟滞带状态判定"""
    prev_idx = STATE_ORDER.index(prev_state_name) if prev_state_name in STATE_ORDER else 1
    # 检查上升
    if prev_idx < 2:
        next_up = STATE_ORDER[prev_idx + 1]
        key = f"{prev_state_name}→{next_up}"
        if pos >= HYSTERESIS["up"][key]:
            return next_up
    # 检查下降
    if prev_idx > 0:
        next_down = STATE_ORDER[prev_idx - 1]
        key = f"{prev_state_name}→{next_down}"
        if pos <= HYSTERESIS["down"][key]:
            return next_down
    return prev_state_name

# 简单阈值判定(用于首次/无状态时)
def simple_state(pos):
    if pos > 0.65:
        return "进攻"
    elif pos >= 0.40:
        return "均衡"
    else:
        return "防御"

prev = load_state()
if prev and prev.get("market_state"):
    market_state = hysteresis_state(pos_total, prev["market_state"])
else:
    market_state = simple_state(pos_total)

# 迟滞前的简单状态(用于输出对比)
simple_st = simple_state(pos_total)
state_hysteresis_active = (market_state != simple_st)  # 迟滞带是否在起作用

# Step 1: 三态线性插值
t = float(np.clip((pos_total - 0.40) / 0.25, 0, 1))
# t=0 → 防御, t=1 → 进攻, 中间线性

def interp(cat):
    """线性插值: 防御 → 进攻"""
    return CFG[cat]["防御"] + t * (CFG[cat]["进攻"] - CFG[cat]["防御"])

w_dividend_base = interp("红利")
w_broad_base = interp("宽基合计")
w_attack_base = interp("进攻")

# Step 2: 风格偏好调整
style_factor = (split - 0.50) / 0.40  # split=90%→+1, 10%→-1, 50%→0

w_dividend = w_dividend_base + 0.03 * style_factor   # 大盘强→红利加
w_attack = w_attack_base - 0.03 * style_factor       # 大盘强→进攻减

# 宽基按split拆分
w_300 = w_broad_base * split
w_2000 = w_broad_base * (1 - split)

# 类现金 = 归一化: 确保五大类合计=100%
w_cash = 1 - w_300 - w_2000 - w_dividend - w_attack

# Step 3: 硬性区间裁剪(先裁权益类，再重算类现金归一化)
def clamp_cat(val, cat):
    if cat in CFG:
        return float(np.clip(val, CFG[cat]["lo"], CFG[cat]["hi"]))
    return val

w_dividend = clamp_cat(w_dividend, "红利")
w_attack = clamp_cat(w_attack, "进攻")

# 裁剪后重算类现金确保归一化
w_cash = 1 - w_300 - w_2000 - w_dividend - w_attack
if w_cash < 0.20:
    # 类现金不够20%，从最大的权益类缩减
    deficit = 0.20 - w_cash
    w_attack = max(w_attack - deficit, CFG["进攻"]["lo"])
    w_cash = 1 - w_300 - w_2000 - w_dividend - w_attack

result = {
    "status": "ok",
    "date": date_str,
    "version": "v4.3.1",
    # 信息1：市场风格 + 建议总仓位
    "market_state": market_state,
    "style_level": style_level,
    "split": f"{split*100:.0f}%",
    "position": f"{pos_total*100:.1f}%",
    "position_raw": pos_total,
    # 择时信号
    "signals": {
        "bb_z_price": round(bb_z, 4),
        "bb_z_margin": round(rzye_z, 4),
        "pos_price": round(pos_price, 4),
        "pos_margin": round(pos_margin, 4),
    },
    # 信息2：5大类建议比例(合计=100%)
    "allocation": {
        "红利": f"{w_dividend*100:.0f}%",
        "沪深300": f"{w_300*100:.0f}%",
        "国证2000": f"{w_2000*100:.0f}%",
        "进攻": f"{w_attack*100:.0f}%",
        "类现金": f"{w_cash*100:.0f}%",
    },
    # 调仓建议相关
    "allocation_raw": {
        "红利": round(w_dividend, 4),
        "沪深300": round(w_300, 4),
        "国证2000": round(w_2000, 4),
        "进攻": round(w_attack, 4),
        "类现金": round(w_cash, 4),
    },
}

# ── 8. 调仓建议 ──
current_weights = {
    "红利": round(w_dividend, 4),
    "沪深300": round(w_300, 4),
    "国证2000": round(w_2000, 4),
    "进攻": round(w_attack, 4),
    "类现金": round(w_cash, 4),
}

rebalance_advice = {
    "should_rebalance": False,
    "drift_pct": 0.0,
    "reason": "无需调仓",
    "prev_rebalance_date": None,
    "days_since_rebalance": None,
    "state_changed": False,
    "prev_state": None,
}

# 计算与上次调仓权重的漂移
if prev and prev.get("last_rebalance_weights"):
    last_w = prev["last_rebalance_weights"]
    drift = sum(abs(current_weights.get(k, 0) - last_w.get(k, 0)) for k in current_weights)
    drift_pct = drift * 100
    last_date = prev.get("last_rebalance_date", "未知")
    try:
        days_since = (TODAY - pd.Timestamp(last_date)).days
    except Exception:
        days_since = None

    rebalance_advice["drift_pct"] = round(drift_pct, 1)
    rebalance_advice["prev_rebalance_date"] = last_date
    rebalance_advice["days_since_rebalance"] = days_since

    # 三态是否变化
    prev_st = prev.get("market_state")
    rebalance_advice["prev_state"] = prev_st
    major_state_change = False
    if prev_st and prev_st != market_state:
        rebalance_advice["state_changed"] = True
        # 大级别切换: 防御↔进攻
        if set([prev_st, market_state]) == set(["防御", "进攻"]):
            major_state_change = True
            rebalance_advice["should_rebalance"] = True
            rebalance_advice["reason"] = f"三态大级别切换: {prev_st}→{market_state}, 建议立即调仓"
        elif set([prev_st, market_state]) == set(["防御", "均衡"]):
            rebalance_advice["should_rebalance"] = True
            rebalance_advice["reason"] = f"三态切换: {prev_st}→{market_state}, 漂移{drift_pct:.1f}%"
        elif set([prev_st, market_state]) == set(["均衡", "进攻"]):
            rebalance_advice["should_rebalance"] = True
            rebalance_advice["reason"] = f"三态切换: {prev_st}→{market_state}, 漂移{drift_pct:.1f}%"

    # 漂移超阈值(仅当三态未大级别切换时)
    if not major_state_change and drift >= DRIFT_THRESHOLD:
        rebalance_advice["should_rebalance"] = True
        rebalance_advice["reason"] = f"权重漂移{drift_pct:.1f}%超阈值{DRIFT_THRESHOLD*100:.0f}%"
    elif not rebalance_advice["should_rebalance"]:
        rebalance_advice["reason"] = f"漂移{drift_pct:.1f}%<阈值{DRIFT_THRESHOLD*100:.0f}%, 无需调仓"
else:
    # 首次运行，标记为需要调仓(初始化)
    rebalance_advice["should_rebalance"] = True
    rebalance_advice["reason"] = "首次运行，建议按当前权重初始化持仓"

result["rebalance"] = rebalance_advice

# 如果建议调仓，更新"上次调仓权重"
# (实际调仓后cron agent应触发标记，但这里先按"建议即调仓"逻辑)
# 用漂移超阈值作为"已调仓"标记
if rebalance_advice["should_rebalance"]:
    save_state({
        "market_state": market_state,
        "last_rebalance_weights": current_weights,
        "last_rebalance_date": date_str,
        "last_position": pos_total,
        "last_split": split,
    })
else:
    # 只更新状态(三态可能变了但没触发调仓)
    save_state({
        "market_state": market_state,
        "last_rebalance_weights": prev["last_rebalance_weights"],
        "last_rebalance_date": prev["last_rebalance_date"],
        "last_position": pos_total,
        "last_split": split,
    })

print(json.dumps(result, ensure_ascii=False))
