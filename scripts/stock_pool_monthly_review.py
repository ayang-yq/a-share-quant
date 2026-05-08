#!/usr/bin/env python3
"""
股票池月度审视脚本
- 采集44只股票的最新财务数据(东方财富API)
- 输出Markdown格式报告到stdout
- 由cron job调用，结果推送到飞书
"""
import os, sys, json, urllib.request, datetime, time

os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

# ── 股票池定义 ──
POOL = {
    "核心-消费": ["600519", "000858", "600809", "600887", "605499", "000333"],
    "核心-医药": ["300760", "600276", "300015", "603259"],
    "核心-半导体": ["002371", "688012", "688008", "002049", "688981", "603986"],
    "核心-科技制造": ["300750", "002594", "002475", "300308", "300124", "002415"],
    "核心-互联网软件": ["688041", "688111"],
    "核心-金融": ["600036", "601318"],
    "核心-公用事业": ["600900"],
    "核心-交运": ["002352", "601021"],
    "港股": ["00700", "09988", "09618", "03690", "01810"],
    "周期-资源": ["601899", "601857", "601088"],
    "周期-化工制造": ["600309", "300274"],
    "观察池": ["601012", "600438", "002460", "002714", "601919"],
}

# 名称映射
NAMES = {
    "600519": "贵州茅台", "000858": "五粮液", "600809": "山西汾酒",
    "600887": "伊利股份", "605499": "东鹏饮料", "000333": "美的集团",
    "300760": "迈瑞医疗", "600276": "恒瑞医药", "300015": "爱尔眼科",
    "603259": "药明康德",
    "002371": "北方华创", "688012": "中微公司", "688008": "澜起科技",
    "002049": "紫光国微", "688981": "中芯国际", "603986": "兆易创新",
    "300750": "宁德时代", "002594": "比亚迪", "002475": "立讯精密",
    "300308": "中际旭创", "300124": "汇川技术", "002415": "海康威视",
    "688041": "海光信息", "688111": "金山办公",
    "600036": "招商银行", "601318": "中国平安",
    "600900": "长江电力",
    "002352": "顺丰控股", "601021": "春秋航空",
    "00700": "腾讯控股", "09988": "阿里巴巴", "09618": "京东集团",
    "03690": "美团", "01810": "小米集团",
    "601899": "紫金矿业", "601857": "中国石油", "601088": "中国神华",
    "600309": "万华化学", "300274": "阳光电源",
    "601012": "隆基绿能", "600438": "通威股份", "002460": "赣锋锂业",
    "002714": "牧原股份", "601919": "中远海控",
}

def fetch_a_stock_financial(code):
    """用东方财富API拉取A股最新两期年报数据"""
    url = (
        f"https://datacenter.eastmoney.com/securities/api/data/v1/get?"
        f"reportName=RPT_F10_FINANCE_MAINFINADATA"
        f"&columns=SECURITY_CODE,REPORT_DATE_NAME,TOTALOPERATEREVE,PARENTNETPROFIT,"
        f"KCFJCXSYJLR,TOTALOPERATEREVETZ,PARENTNETPROFITTZ,KCFJCXSYJLRTZ,"
        f"XSMLL,XSJLL,ROEJQ,EPSJB,RDEXPEND"
        f"&filter=(SECURITY_CODE=%22{code}%22)(REPORT_TYPE=%22%E5%B9%B4%E6%8A%A5%22)"
        f"&pageNumber=1&pageSize=5&sortTypes=-1&sortColumns=REPORT_DATE"
    )
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://emweb.securities.eastmoney.com/"
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        if data.get("success") and data.get("result"):
            return data["result"]["data"][:2]  # 最新两年
    except Exception as e:
        return []
    return []


def fetch_a_stock_latest(code):
    """拉取最新季度数据（含TTM）"""
    url = (
        f"https://datacenter.eastmoney.com/securities/api/data/v1/get?"
        f"reportName=RPT_LICO_FN_CPD"
        f"&columns=SECURITY_CODE,REPORTDATE,DATATYPE,TOTAL_OPERATE_INCOME,"
        f"PARENT_NETPROFIT,WEIGHTAVG_ROE,XSMLL,YSTZ,SJLTZ"
        f"&filter=(SECURITY_CODE=%22{code}%22)"
        f"&pageNumber=1&pageSize=6&sortTypes=-1&sortColumns=REPORTDATE"
    )
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://emweb.securities.eastmoney.com/"
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        if data.get("success") and data.get("result"):
            return data["result"]["data"][:4]
    except:
        return []
    return []


def fetch_a_stock_spot(code):
    """拉取实时行情（PE/PB/市值）用东财datacenter"""
    url = (
        f"https://datacenter.eastmoney.com/securities/api/data/v1/get?"
        f"reportName=RPT_LICO_FN_CPD"
        f"&columns=SECURITY_CODE,SECURITY_NAME_ABBR,BASIC_EPS,TOTAL_OPERATE_INCOME,"
        f"PARENT_NETPROFIT,WEIGHTAVG_ROE,XSMLL,YSTZ,SJLTZ,DATATYPE,ZXGXL"
        f"&filter=(SECURITY_CODE=%22{code}%22)"
        f"&pageNumber=1&pageSize=2&sortTypes=-1&sortColumns=REPORTDATE"
    )
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://emweb.securities.eastmoney.com/"
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        if data.get("success") and data.get("result"):
            rows = data["result"]["data"]
            if rows:
                return rows[0]
    except:
        pass
    return {}


def format_change(val):
    """格式化变化率"""
    if val is None:
        return "N/A"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.1f}%"


def analyze_a_stock(code):
    """分析单只A股，返回结构化数据"""
    result = {"code": code, "name": NAMES.get(code, code), "type": "A股"}

    # 年报数据
    annual = fetch_a_stock_financial(code)
    def safe_fmt(val, divisor=1, suffix="", fmt=".1f"):
        """安全格式化，None -> '—'"""
        if val is None:
            return "—"
        return f"{val/divisor:{fmt}}{suffix}"

    if len(annual) >= 2:
        latest = annual[0]
        prev = annual[1]
        result["latest_year"] = latest.get("REPORT_DATE_NAME", "")
        result["revenue"] = safe_fmt(latest.get('TOTALOPERATEREVE'), 1e8, "亿")
        result["net_profit"] = safe_fmt(latest.get('PARENTNETPROFIT'), 1e8, "亿")
        result["rev_yoy"] = format_change(latest.get("TOTALOPERATEREVETZ"))
        result["profit_yoy"] = format_change(latest.get("PARENTNETPROFITTZ"))
        result["deduct_yoy"] = format_change(latest.get("KCFJCXSYJLRTZ"))
        result["gross_margin"] = safe_fmt(latest.get('XSMLL'), suffix="%")
        result["net_margin"] = safe_fmt(latest.get('XSJLL'), suffix="%")
        result["roe"] = safe_fmt(latest.get('ROEJQ'), suffix="%")

        # 对比前一年，标注重大变化
        alerts = []
        roe_new = latest.get("ROEJQ") or 0
        roe_old = prev.get("ROEJQ") or 0
        roe_delta = roe_new - roe_old
        if abs(roe_delta) > 5:
            sign = "下降" if roe_delta < 0 else "上升"
            alerts.append(f"ROE{sign}{abs(roe_delta):.1f}pp")

        gm_new = latest.get("XSMLL") or 0
        gm_old = prev.get("XSMLL") or 0
        if gm_old and (gm_new - gm_old) < -5:
            alerts.append(f"毛利率下降{abs(gm_new - gm_old):.1f}pp")

        profit_chg = latest.get("PARENTNETPROFITTZ") or 0
        if profit_chg < -30:
            alerts.append(f"净利润同比{profit_chg:.0f}%")

        result["alerts"] = alerts
    elif len(annual) == 1:
        latest = annual[0]
        result["latest_year"] = latest.get("REPORT_DATE_NAME", "")
        result["revenue"] = safe_fmt(latest.get('TOTALOPERATEREVE'), 1e8, "亿")
        result["net_profit"] = safe_fmt(latest.get('PARENTNETPROFIT'), 1e8, "亿")
        result["rev_yoy"] = format_change(latest.get("TOTALOPERATEREVETZ"))
        result["profit_yoy"] = format_change(latest.get("PARENTNETPROFITTZ"))
        result["gross_margin"] = safe_fmt(latest.get('XSMLL'), suffix="%")
        result["roe"] = safe_fmt(latest.get('ROEJQ'), suffix="%")
        result["alerts"] = []

    # 安全处理None值
    for key in ["revenue", "net_profit", "gross_margin", "net_margin", "roe", "rev_yoy", "profit_yoy", "deduct_yoy"]:
        if key not in result:
            result[key] = "—"

    time.sleep(0.15)  # 限速
    return result


def generate_markdown(results_by_category, report_date):
    """生成Markdown报告"""
    lines = []
    lines.append(f"# 📊 股票池月度审视报告")
    lines.append(f"")
    lines.append(f"**报告日期**: {report_date}")
    lines.append(f"**股票池版本**: v2.3 (44只)")
    lines.append(f"**数据来源**: 东方财富API")
    lines.append(f"")
    lines.append("---")
    lines.append("")

    # 汇总：有重大变化的标的
    all_alerts = []
    for cat, stocks in results_by_category.items():
        for s in stocks:
            if s.get("alerts"):
                all_alerts.append((s["name"], s["code"], cat, s["alerts"]))

    if all_alerts:
        lines.append("## ⚠️ 重大变化提醒")
        lines.append("")
        for name, code, cat, alerts in all_alerts:
            alert_text = " | ".join(alerts)
            lines.append(f"- **{name}** ({code}) [{cat}] — {alert_text}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # 逐分类输出
    for cat, stocks in results_by_category.items():
        lines.append(f"## {cat} ({len(stocks)}只)")
        lines.append("")
        lines.append("| 代码 | 名称 | ROE | 净利润 | 营收YoY | 利润YoY | 毛利率 | 变化提醒 |")
        lines.append("|------|------|-----|--------|---------|---------|--------|---------|")
        for s in stocks:
            alert = "⚠️" + ",".join(s.get("alerts", [])) if s.get("alerts") else "—"
            lines.append(
                f"| {s['code']} | {s['name']} | {s.get('roe','—')} "
                f"| {s.get('net_profit','—')} | {s.get('rev_yoy','—')} "
                f"| {s.get('profit_yoy','—')} | {s.get('gross_margin','—')} "
                f"| {alert} |"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    # 底部说明
    lines.append("## 📝 使用说明")
    lines.append("- 本报告自动生成，仅供研究参考，不构成投资建议")
    lines.append("- ROE为年报加权值，PE/PB为实时值")
    lines.append("- ⚠️ 标记表示该标的同比出现重大变化(ROE变动>5pp/毛利率下降>5pp/利润下降>30%)")
    lines.append("- 建议重点关注⚠️标的，核实变化原因是否影响核心逻辑")

    return "\n".join(lines)


def main():
    now = datetime.datetime.now()
    report_date = now.strftime("%Y年%m月%d日")
    results_by_category = {}

    all_a_codes = []
    for cat, codes in POOL.items():
        if cat == "港股":
            continue
        all_a_codes.extend([(cat, c) for c in codes])

    print(f"开始采集 {len(all_a_codes)} 只A股数据...", file=sys.stderr)

    for cat, code in all_a_codes:
        if cat not in results_by_category:
            results_by_category[cat] = []
        result = analyze_a_stock(code)
        results_by_category[cat].append(result)
        print(f"  已完成: {result['name']} ({code})", file=sys.stderr)

    # 港股简单处理（API不同，仅标记）
    hk_results = []
    for code in POOL.get("港股", []):
        hk_results.append({
            "code": code,
            "name": NAMES.get(code, code),
            "type": "港股",
            "price": "—",
            "market_cap": "—",
            "pe": "—",
            "roe": "—",
            "net_profit": "—",
            "rev_yoy": "—",
            "profit_yoy": "—",
            "gross_margin": "—",
            "alerts": ["港股数据需单独采集"],
        })
    results_by_category["港股"] = hk_results

    md = generate_markdown(results_by_category, report_date)

    # 输出到stdout
    print(md)


if __name__ == "__main__":
    main()
