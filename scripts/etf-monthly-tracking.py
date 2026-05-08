#!/usr/bin/env python3
"""红利ETF月度跟踪脚本 - 6只ETF净值、回报、追踪差异检查"""

import os
import sys
import json
import urllib.request
from datetime import datetime, timedelta

# Proxy bypass for Chinese financial sites
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(k, None)

ETFS = [
    {"code": "515080", "name": "中证红利ETF招商", "index": "中证红利(000922)", "fee": 0.30, "role": "A股底仓"},
    {"code": "563020", "name": "红利低波ETF易方达", "index": "红利低波(H30269)", "fee": 0.20, "role": "A股防御"},
    {"code": "159307", "name": "红利低波100ETF博时", "index": "红利低波100(930955)", "fee": 0.20, "role": "A股分散"},
    {"code": "159691", "name": "港股红利ETF工银", "index": "中证港股通高股息(931028)", "fee": 0.52, "role": "港股精选"},
    {"code": "513530", "name": "港股通红利ETF华泰柏瑞", "index": "恒生港股通高股息", "fee": 0.60, "role": "港股现金流"},
    {"code": "563510", "name": "A500红利低波ETF易方达", "index": "A500红利低波(932422)", "fee": 0.20, "role": "观察仓"},
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://fund.eastmoney.com",
}


def fetch_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    resp = urllib.request.urlopen(req, timeout=15)
    return json.loads(resp.read().decode("utf-8"))


def get_realtime(code):
    """Get real-time NAV estimate from fundgz API"""
    url = f"https://fundgz.1234567.com.cn/js/{code}.js"
    req = urllib.request.Request(url, headers=HEADERS)
    resp = urllib.request.urlopen(req, timeout=15)
    text = resp.read().decode("utf-8")
    json_str = text[text.index("(") + 1 : text.rindex(")")]
    return json.loads(json_str)


def get_nav_history(code, days=500):
    """Get NAV history from Eastmoney API, multi-page (20/page limit)"""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    all_items = []
    for page in range(1, 30):  # 30 pages * 20 = 600 records max
        url = (
            f"https://api.fund.eastmoney.com/f10/lsjz"
            f"?fundCode={code}&pageIndex={page}&pageSize=20"
            f"&startDate={start_date}&endDate={end_date}"
        )
        try:
            data = fetch_json(url)
            items = data.get("Data", {}).get("LSJZList", [])
            if not items:
                break
            all_items.extend(items)
            if len(items) < 20:
                break
        except Exception:
            break
    return all_items


def calc_returns(items, periods_days=None):
    """Calculate NAV returns for various periods"""
    if periods_days is None:
        periods_days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365}

    if not items:
        return {}

    # items are in descending date order (newest first)
    nav_map = {}
    for item in items:
        nav_map[item["FSRQ"]] = float(item["DWJZ"])

    results = {}
    for label, days in periods_days.items():
        target_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        # Find closest date
        closest = min(nav_map.keys(), key=lambda d: abs((datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(target_date, "%Y-%m-%d")).days))
        latest_nav = nav_map[list(nav_map.keys())[0]]
        target_nav = nav_map[closest]
        ret = (latest_nav - target_nav) / target_nav * 100
        results[label] = {"date": closest, "nav": target_nav, "return": round(ret, 2)}

    return results


def get_fund_scale(code):
    """Try to get fund scale from Eastmoney F10 page"""
    url = f"https://fundf10.eastmoney.com/FundArchivesDatas.aspx?type=jjcc&per=1&page=1&code={code}"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        resp = urllib.request.urlopen(req, timeout=15)
        text = resp.read().decode("utf-8")
        import re
        scale_match = re.search(r'"FCZE":"([\d.]+)"', text)
        if scale_match:
            return f"{float(scale_match.group(1)):.2f}亿"
    except Exception:
        pass
    return "N/A"


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    lines = []
    lines.append(f"📊 红利ETF月度跟踪报告 ({today})")
    lines.append("=" * 60)

    summary_rows = []

    for etf in ETFS:
        code = etf["code"]
        name = etf["name"]
        lines.append(f"\n{'─' * 40}")
        lines.append(f"【{code}】{name} ({etf['role']})")
        lines.append(f"跟踪指数: {etf['index']} | 费率: {etf['fee']}%")

        try:
            # Real-time NAV
            try:
                rt = get_realtime(code)
                nav_display = rt["dwjz"]
                nav_date = rt["jzrq"]
                gz_display = f"估值: {rt['gsz']} ({rt['gszzl']}%)"
            except Exception:
                # Fallback: get latest from history
                items_temp = get_nav_history(code, days=10)
                if items_temp:
                    nav_display = items_temp[0]["DWJZ"]
                    nav_date = items_temp[0]["FSRQ"]
                    gz_display = "估值: N/A"
                else:
                    raise

            lines.append(f"最新净值: {nav_display} ({nav_date}) | {gz_display}")

            # Historical NAV & returns
            items = get_nav_history(code, days=500)
            if items:
                returns = calc_returns(items)
                lines.append(f"历史数据: {len(items)}条")
                ret_str = " | ".join(f"{k}:{v['return']:+.2f}%({v['date'][:7]})" for k, v in returns.items())
                lines.append(f"净值涨幅: {ret_str}")

                # Collect summary data
                row = {
                    "code": code,
                    "name": name,
                    "nav": nav_display,
                    "1M": returns.get("1M", {}).get("return", "N/A"),
                    "3M": returns.get("3M", {}).get("return", "N/A"),
                    "6M": returns.get("6M", {}).get("return", "N/A"),
                    "1Y": returns.get("1Y", {}).get("return", "N/A"),
                }
                summary_rows.append(row)
            else:
                lines.append("⚠️ 无法获取历史净值数据")
                summary_rows.append({"code": code, "name": name, "nav": "N/A", "1M": "N/A", "3M": "N/A", "6M": "N/A", "1Y": "N/A"})

        except Exception as e:
            lines.append(f"❌ 数据获取失败: {e}")
            summary_rows.append({"code": code, "name": name, "nav": "ERR", "1M": "ERR", "3M": "ERR", "6M": "ERR", "1Y": "ERR"})

    # Summary table
    lines.append(f"\n{'=' * 60}")
    lines.append("📋 汇总对比")
    lines.append(f"{'代码':<8} {'名称':<20} {'净值':>7} {'1M':>7} {'3M':>7} {'6M':>7} {'1Y':>7}")
    lines.append("─" * 68)
    for row in summary_rows:
        def fmt(v):
            if isinstance(v, str):
                return f"{v:>7}"
            return f"{v:>+7.2f}%"
        lines.append(
            f"{row['code']:<8} {row['name']:<18} {row['nav']:>7} "
            f"{fmt(row['1M'])} {fmt(row['3M'])} {fmt(row['6M'])} {fmt(row['1Y'])}"
        )

    # Alerts section
    lines.append(f"\n{'=' * 60}")
    lines.append("⚠️ 关注事项")
    alerts = []

    for row in summary_rows:
        code = row["code"]
        if code == "563510":
            # Check if 563510 has enough data for tracking error evaluation
            if row["1Y"] == "N/A" or (isinstance(row["1Y"], str) and not row["1Y"].startswith("+") and not row["1Y"].startswith("-")):
                alerts.append(f"563510 A500红利低波：数据仍在积累中，暂无法评估追踪差异")
            continue
        # Check for significant drawdown (any period < -10%)
        for period in ["1M", "3M", "6M", "1Y"]:
            val = row.get(period)
            if isinstance(val, (int, float)) and val < -10:
                alerts.append(f"{code} {row['name']}：{period}跌幅{val:.2f}%，需关注是否触发估值加仓区间")

    if alerts:
        for a in alerts:
            lines.append(f"• {a}")
    else:
        lines.append("• 本月无特别预警事项")

    # Next steps
    lines.append(f"\n📌 下月关注")
    lines.append("• 563510(A500红利低波)追踪差异评估")
    lines.append("• 各指数PE/PB百分位更新")
    lines.append("• 同指数是否有更低费率新ETF上市")

    output = "\n".join(lines)
    print(output)
    return output


if __name__ == "__main__":
    main()
