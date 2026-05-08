#!/usr/bin/env python3
"""
卫斯理红黄绿趋势模型 - 逆向工程实现
========================================
基于文章分析推测的双均线趋势跟踪系统：
- 中期趋势：50日均线（灵敏）
- 长期趋势：200日均线（稳健）
- 状态判定：价格位置 + 均线斜率 + 时间滤波(3日确认)
- 三态：红(向上)/黄(震荡)/绿(向下)

用法:
  python3 wesley_trend_model.py              # 默认输出最新信号表
  python3 wesley_trend_model.py --backtest 510300  # 回测指定ETF
  python3 wesley_trend_model.py --backtest-all     # 回测所有核心宽基
  python3 wesley_trend_model.py --params              # 参数敏感性分析
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd
import numpy as np

# ============================================================
# 1. 标的定义
# ============================================================

ETF_LIST = {
    # 宽基
    "沪深300": "510300",
    "中证500": "510500",
    "中证1000": "512100",
    "上证50": "510050",
    "创业板50": "159949",
    "科创50": "588000",
    "科创100": "588190",
    # 港股/海外
    "恒生科技": "513180",
    "恒生医疗": "513120",
    "纳指ETF": "513100",
    "黄金ETF": "518880",
    # 行业/红利
    "医疗ETF": "512170",
    "芯片ETF": "159995",
    "红利低波": "515100",
}

# 分类
CATEGORY = {
    "宽基": ["沪深300", "中证500", "中证1000", "上证50", "创业板50", "科创50", "科创100"],
    "港股/海外": ["恒生科技", "恒生医疗", "纳指ETF", "黄金ETF"],
    "行业/红利": ["医疗ETF", "芯片ETF", "红利低波"],
}


# ============================================================
# 2. 数据获取
# ============================================================

def fetch_etf_data(code: str, start_date: str = "20100101") -> pd.DataFrame:
    """通过AKShare获取ETF日线数据"""
    try:
        df = ak.fund_etf_hist_em(
            symbol=code,
            period="daily",
            adjust="qfq",
            start_date=start_date,
        )
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.sort_values("日期").reset_index(drop=True)
        df = df.rename(columns={"收盘": "close", "日期": "date"})
        return df[["date", "close"]].copy()
    except Exception as e:
        print(f"  [WARN] 获取 {code} 失败: {e}", file=sys.stderr)
        return pd.DataFrame()


# ============================================================
# 3. 模型核心
# ============================================================

class TrafficLightModel:
    """
    红黄绿趋势模型

    推测参数:
      - 中期均线: short_ma (默认50日)
      - 长期均线: long_ma (默认200日)
      - 斜率计算窗口: slope_window (默认20日)
      - 时间滤波确认天数: confirm_days (默认3日)
      - 斜率阈值: slope_pct (默认0，即MA方向为正/负即可)
    """

    def __init__(
        self,
        short_ma: int = 50,
        long_ma: int = 200,
        slope_window: int = 20,
        confirm_days: int = 3,
    ):
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.slope_window = slope_window
        self.confirm_days = confirm_days

    def _compute_raw_signal(self, series: pd.Series, ma_window: int) -> pd.Series:
        """
        计算单维度的原始信号
        RED=1, YELLOW=0, GREEN=-1
        """
        ma = series.rolling(ma_window, min_periods=ma_window).mean()
        # MA斜率: (MA_today - MA_{N天前}) / MA_{N天前}
        ma_slope = (ma - ma.shift(self.slope_window)) / ma.shift(self.slope_window)

        raw = pd.Series(0, index=series.index)  # 默认YELLOW

        # RED: 价格在均线上方 AND 均线斜率为正
        red_mask = (series > ma) & (ma_slope > 0)
        # GREEN: 价格在均线下方 AND 均线斜率为负
        green_mask = (series < ma) & (ma_slope < 0)
        # 其他保持YELLOW

        raw[red_mask] = 1
        raw[green_mask] = -1

        return raw

    def _apply_time_filter(self, raw: pd.Series) -> pd.Series:
        """
        时间滤波: 新状态需要连续N日确认才生效，否则保持旧状态
        """
        filtered = raw.copy()
        # 第一个有效信号直接采用
        first_valid = raw.first_valid_index()
        if first_valid is None:
            return filtered

        current_state = raw.loc[first_valid]
        for i in range(len(raw)):
            idx = raw.index[i]
            proposed = raw.iloc[i]
            if i >= self.confirm_days:
                # 检查最近confirm_days天是否都是同一个新状态
                recent = raw.iloc[i - self.confirm_days + 1 : i + 1]
                if (recent == proposed).all() and proposed != current_state:
                    current_state = proposed
            elif i == 0:
                current_state = proposed
            filtered.iloc[i] = current_state

        return filtered

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算完整的红黄绿信号

        Returns: DataFrame with columns [date, close, mid_raw, mid_signal, long_raw, long_signal]
        """
        result = df[["date", "close"]].copy()

        # 中期信号
        result["mid_raw"] = self._compute_raw_signal(result["close"], self.short_ma)
        result["mid_signal"] = self._apply_time_filter(result["mid_raw"])

        # 长期信号
        result["long_raw"] = self._compute_raw_signal(result["close"], self.long_ma)
        result["long_signal"] = self._apply_time_filter(result["long_raw"])

        return result

    def get_latest_signals(self, df: pd.DataFrame) -> dict:
        """获取最新信号状态"""
        signals = self.compute_signals(df)
        if len(signals) == 0:
            return {"mid": "N/A", "long": "N/A", "action": "无数据"}

        last = signals.iloc[-1]
        prev = signals.iloc[-2] if len(signals) > 1 else None

        def to_color(val):
            if pd.isna(val):
                return "—"
            return {1: "🔴红", 0: "🟡黄", -1: "🟢绿"}.get(int(val), "—")

        mid = to_color(last["mid_signal"])
        long = to_color(last["long_signal"])
        action = self._get_action(int(last["mid_signal"]), int(last["long_signal"]))

        # 检测今日变化
        change = ""
        if prev is not None:
            mid_chg = last["mid_signal"] != prev["mid_signal"]
            long_chg = last["long_signal"] != prev["long_signal"]
            if mid_chg or long_chg:
                parts = []
                if mid_chg:
                    parts.append(f"中期:{to_color(prev['mid_signal'])}→{mid}")
                if long_chg:
                    parts.append(f"长期:{to_color(prev['long_signal'])}→{long}")
                change = " | ".join(parts)

        return {
            "mid": mid,
            "long": long,
            "action": action,
            "change": change,
            "date": last["date"].strftime("%Y-%m-%d"),
            "close": last["close"],
        }

    @staticmethod
    def _get_action(mid: int, long: int) -> str:
        action_map = {
            (1, 1): "✅ 持有为主",
            (1, 0): "⚠️ 谨慎持有",
            (1, -1): "⚠️ 超跌反弹范畴",
            (0, 1): "👀 关注，等变红",
            (0, 0): "👀 观望",
            (0, -1): "🚫 不做多",
            (-1, 1): "⚠️ 谨慎",
            (-1, 0): "🚫 日内为主",
            (-1, -1): "🚫 控仓/空仓",
        }
        return action_map.get((mid, long), "—")


# ============================================================
# 4. 回测引擎
# ============================================================

def _signal_to_position(mid: int, long: int) -> float:
    """
    信号→目标仓位映射 (分级仓位制)
    核心思想: 只有两个维度都看多才重仓，任何一维看空就大幅减仓

    中期\长期   红(1)     黄(0)     绿(-1)
    红(1)     100%      60%        20%
    黄(0)      60%      30%         0%
    绿(-1)     20%       0%         0%
    """
    pos_map = {
        (1, 1): 1.0,
        (1, 0): 0.6,
        (1, -1): 0.2,
        (0, 1): 0.6,
        (0, 0): 0.3,
        (0, -1): 0.0,
        (-1, 1): 0.2,
        (-1, 0): 0.0,
        (-1, -1): 0.0,
    }
    return pos_map.get((mid, long), 0.0)


def backtest(df: pd.DataFrame, model: TrafficLightModel, mode: str = "graded") -> dict:
    """
    回测引擎

    mode="graded": 分级仓位制 (推荐)
    mode="binary": 二元全仓/空仓 (原始版)
    """
    signals = model.compute_signals(df)

    # 从有长期信号的那一天开始
    valid = signals.dropna(subset=["long_signal"])
    if len(valid) < 10:
        return None

    # 计算目标仓位
    target_pos = pd.Series(0.0, index=valid.index)
    for i, (idx, row) in enumerate(valid.iterrows()):
        mid = int(row["mid_signal"])
        long = int(row["long_signal"])
        if mode == "binary":
            target_pos.loc[idx] = 1.0 if (mid == 1 and long == 1) else (0.0 if (mid == -1 and long == -1) else target_pos.iloc[max(0, i - 1)])
        else:
            target_pos.loc[idx] = _signal_to_position(mid, long)

    # 计算收益
    returns = valid["close"].pct_change().fillna(0)
    strategy_returns = target_pos.shift(1).fillna(0) * returns  # T日仓位 * T日收益

    # 累计收益
    strategy_cumret = (1 + strategy_returns).cumprod().fillna(1.0)
    benchmark_cumret = (1 + returns).cumprod().fillna(1.0)

    total_return = strategy_cumret.iloc[-1] - 1
    benchmark_return = benchmark_cumret.iloc[-1] - 1

    # 最大回撤
    rolling_max = strategy_cumret.cummax()
    drawdown = (strategy_cumret - rolling_max) / rolling_max

    # 基准最大回撤
    bench_max = benchmark_cumret.cummax()
    bench_drawdown = (benchmark_cumret - bench_max) / bench_max

    # 统计信号变化次数
    mid_changes = (valid["mid_signal"].diff() != 0).sum()
    long_changes = (valid["long_signal"].diff() != 0).sum()

    # 年化收益 (按交易日252天)
    n_years = len(valid) / 252
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_benchmark = (1 + benchmark_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    return {
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "ann_return": ann_return,
        "ann_benchmark": ann_benchmark,
        "max_drawdown": drawdown.min(),
        "bench_max_drawdown": bench_drawdown.min(),
        "avg_position": target_pos.shift(1).mean(),
        "mid_changes": mid_changes,
        "long_changes": long_changes,
        "trading_days": len(valid),
        "start_date": valid.iloc[0]["date"].strftime("%Y-%m-%d"),
        "end_date": valid.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "n_years": n_years,
    }


# ============================================================
# 5. 输出
# ============================================================

def print_signal_table(model: TrafficLightModel):
    """打印最新信号表"""
    print(f"\n{'='*70}")
    print(f"  卫斯理红黄绿趋势模型 — 最新信号 ({datetime.now().strftime('%Y-%m-%d')})")
    print(f"  参数: MA{model.short_ma}(中期) / MA{model.long_ma}(长期) | 确认{model.confirm_days}日")
    print(f"{'='*70}")

    all_results = []

    for cat_name, names in CATEGORY.items():
        print(f"\n  【{cat_name}】")
        print(f"  {'品种':<10} {'代码':<8} {'中期':<8} {'长期':<8} {'操作建议':<16} {'收盘价':>8} {'变化'}")
        print(f"  {'-'*66}")

        for name in names:
            code = ETF_LIST[name]
            df = fetch_etf_data(code)
            if df.empty or len(df) < 220:
                print(f"  {name:<10} {code:<8} 数据不足或获取失败")
                continue

            info = model.get_latest_signals(df)
            change_str = f"⚡{info['change']}" if info["change"] else ""
            print(
                f"  {name:<10} {code:<8} {info['mid']:<8} {info['long']:<8} "
                f"{info['action']:<16} {info['close']:>8.3f} {change_str}"
            )
            all_results.append({
                "name": name,
                "code": code,
                "mid": info["mid"],
                "long": info["long"],
                "action": info["action"],
                "close": info["close"],
                "change": info["change"],
            })

    # 汇总
    red_count = sum(1 for r in all_results if "红" in r["mid"] and "红" in r["long"])
    green_count = sum(1 for r in all_results if "绿" in r["mid"] and "绿" in r["long"])
    print(f"\n  📊 汇总: {len(all_results)}个标的 | 双红{red_count}个 | 双绿{green_count}个 | 其他{len(all_results)-red_count-green_count}个")
    print()


def print_backtest(code: str, name: str = None, model: TrafficLightModel = None):
    """打印单个ETF回测结果"""
    if model is None:
        model = TrafficLightModel()

    name = name or code
    print(f"\n  回测: {name} ({code})", file=sys.stderr)
    print(f"  拉取数据中...", file=sys.stderr)

    df = fetch_etf_data(code)
    if df.empty:
        print(f"  [ERROR] 数据获取失败", file=sys.stderr)
        return

    print(f"  数据: {df.iloc[0]['date'].strftime('%Y-%m-%d')} ~ {df.iloc[-1]['date'].strftime('%Y-%m-%d')} ({len(df)}天)", file=sys.stderr)
    print(f"  计算信号中...", file=sys.stderr)

    result = backtest(df, model)
    if result is None:
        print(f"  [ERROR] 数据不足以计算200日均线", file=sys.stderr)
        return

    print(f"\n{'='*60}")
    print(f"  回测结果: {name} ({code})")
    print(f"  参数: MA{model.short_ma} / MA{model.long_ma} / 确认{model.confirm_days}日")
    print(f"  区间: {result['start_date']} ~ {result['end_date']} ({result['n_years']:.1f}年)")
    print(f"{'='*60}")
    print(f"  {'指标':<20} {'策略':>12} {'基准(持有)':>12}")
    print(f"  {'-'*46}")
    print(f"  {'累计收益':<20} {result['total_return']:>11.1%} {result['benchmark_return']:>11.1%}")
    print(f"  {'年化收益':<20} {result['ann_return']:>11.1%} {result['ann_benchmark']:>11.1%}")
    print(f"  {'最大回撤':<20} {result['max_drawdown']:>11.1%} {result['bench_max_drawdown']:>11.1%}")
    print(f"  {'平均仓位':<20} {result['avg_position']:>11.0%} {'':>12}")
    print(f"  {'中期信号变化次数':<20} {result['mid_changes']:>12} {'':>12}")
    print(f"  {'长期信号变化次数':<20} {result['long_changes']:>12} {'':>12}")

    # 收益/回撤比
    if result["max_drawdown"] != 0:
        calmar = result["ann_return"] / abs(result["max_drawdown"])
        print(f"  {'年化/最大回撤比':<20} {calmar:>11.2f} {'':>12}")

    # 收益倍数
    if result["total_return"] > 0:
        print(f"\n  💡 策略收益倍数: {result['total_return']:.1%} → 总资产变为 {1+result['total_return']:.2f}倍")
    else:
        print(f"\n  💡 策略亏损: {result['total_return']:.1%}")
    print()


def print_backtest_all(model: TrafficLightModel = None):
    """回测所有宽基"""
    if model is None:
        model = TrafficLightModel()

    print(f"\n{'='*80}")
    print(f"  宽基ETF回测汇总")
    print(f"  参数: MA{model.short_ma} / MA{model.long_ma} / 确认{model.confirm_days}日")
    print(f"{'='*80}")
    print(f"  {'品种':<12} {'代码':<8} {'区间(年)':<8} {'策略累计':>10} {'基准累计':>10} {'策略年化':>10} {'基准年化':>10} {'最大回撤':>10} {'中期变':>6} {'长期变':>6}")
    print(f"  {'-'*100}")

    core_etfs = ["沪深300", "中证500", "中证1000", "上证50", "创业板50", "科创50", "科创100"]

    for name in core_etfs:
        code = ETF_LIST[name]
        df = fetch_etf_data(code)
        if df.empty or len(df) < 220:
            print(f"  {name:<12} {code:<8} 数据不足")
            continue

        result = backtest(df, model)
        if result is None:
            print(f"  {name:<12} {code:<8} 回测失败")
            continue

        print(
            f"  {name:<12} {code:<8} {result['n_years']:>7.1f} "
            f"{result['total_return']:>9.1%} {result['benchmark_return']:>9.1%} "
            f"{result['ann_return']:>9.1%} {result['ann_benchmark']:>9.1%} "
            f"{result['max_drawdown']:>9.1%} "
            f"{int(result['mid_changes']):>6} {int(result['long_changes']):>6}"
        )

    print()


def print_param_sensitivity():
    """参数敏感性分析 (仅沪深300)"""
    code = "510300"
    df = fetch_etf_data(code)
    if df.empty:
        print("[ERROR] 数据获取失败")
        return

    print(f"\n{'='*80}")
    print(f"  参数敏感性分析: 沪深300 ({code})")
    print(f"{'='*80}")
    print(f"  {'短期MA':>6} {'长期MA':>6} {'确认天':>6} {'策略累计':>10} {'基准累计':>10} {'策略年化':>10} {'最大回撤':>10} {'中期变':>6} {'长期变':>6}")
    print(f"  {'-'*80}")

    params = [
        (20, 120, 2),
        (20, 120, 3),
        (20, 200, 3),
        (50, 120, 3),
        (50, 200, 2),
        (50, 200, 3),
        (50, 200, 5),
        (60, 200, 3),
        (60, 250, 3),
    ]

    for short, long, confirm in params:
        m = TrafficLightModel(short_ma=short, long_ma=long, confirm_days=confirm)
        result = backtest(df, m)
        if result is None:
            continue
        print(
            f"  {short:>6} {long:>6} {confirm:>6} "
            f"{result['total_return']:>9.1%} {result['benchmark_return']:>9.1%} "
            f"{result['ann_return']:>9.1%} {result['max_drawdown']:>9.1%} "
            f"{int(result['mid_changes']):>6} {int(result['long_changes']):>6}"
        )

    print(f"\n  💡 选择标准: 年化收益高 + 回撤小 + 信号变化次数适中(太频繁=摩擦大)")
    print(f"  💡 卫斯理原话回测: 沪深300 2005至今约45.8倍(未含分红)")
    print()


# ============================================================
# 6. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="卫斯理红黄绿趋势模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
预设模式:
  --sensitive   灵敏模式 (MA20/60/2日) - 信号多、反应快、Calmar最优
  --stable      稳健模式 (MA50/200/3日) - 信号少、抗噪音、更接近原版推测
  默认使用灵敏模式

示例:
  python3 wesley_trend_model.py                        # 最新信号表(灵敏模式)
  python3 wesley_trend_model.py --stable               # 最新信号表(稳健模式)
  python3 wesley_trend_model.py --backtest 510300      # 回测沪深300
  python3 wesley_trend_model.py --backtest-all         # 回测所有宽基
  python3 wesley_trend_model.py --params               # 参数敏感性分析
  python3 wesley_trend_model.py --short-ma 30 --long-ma 120  # 自定义参数
        """,
    )
    parser.add_argument("--backtest", type=str, default=None, help="回测指定ETF代码")
    parser.add_argument("--backtest-all", action="store_true", help="回测所有核心宽基")
    parser.add_argument("--params", action="store_true", help="参数敏感性分析")
    parser.add_argument("--sensitive", action="store_true", help="灵敏模式 (MA20/60/2)")
    parser.add_argument("--stable", action="store_true", help="稳健模式 (MA50/200/3)")
    parser.add_argument("--short-ma", type=int, default=None, help="短期均线")
    parser.add_argument("--long-ma", type=int, default=None, help="长期均线")
    parser.add_argument("--confirm", type=int, default=None, help="确认天数")

    args = parser.parse_args()

    # 参数优先级: 显式指定 > 预设模式 > 默认(sensitive)
    if args.short_ma is not None:
        short, long, confirm = args.short_ma, args.long_ma or 60, args.confirm or 2
        preset_name = "自定义"
    elif args.stable:
        short, long, confirm = 50, 200, 3
        preset_name = "稳健"
    else:
        short, long, confirm = 20, 60, 2
        preset_name = "灵敏"

    model = TrafficLightModel(short_ma=short, long_ma=long, confirm_days=confirm)

    if args.params:
        print_param_sensitivity()
    elif args.backtest:
        name = None
        for n, c in ETF_LIST.items():
            if c == args.backtest:
                name = n
                break
        print_backtest(args.backtest, name, model)
    elif args.backtest_all:
        print_backtest_all(model)
    else:
        print_signal_table(model)


if __name__ == "__main__":
    main()
