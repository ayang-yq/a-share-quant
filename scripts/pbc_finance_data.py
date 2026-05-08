#!/usr/bin/env python3
"""
央行金融统计数据抓取与解析工具
从央行官网获取每月"金融统计数据报告"，解析社融、信贷等关键指标。
仅使用 Python stdlib（urllib, re）
"""

import urllib.request
import re
import json
import sys
import time

BASE = "https://www.pbc.gov.cn"
INDEX_URL = BASE + "/diaochatongjisi/116219/116225/index.html"
PAGE_URL = BASE + "/diaochatongjisi/116219/116225/11871-{}.html"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "zh-CN,zh;q=0.9",
}


def fetch_url(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                for enc in ['utf-8', 'gbk', 'gb2312']:
                    try:
                        return raw.decode(enc)
                    except (UnicodeDecodeError, LookupError):
                        continue
                return raw.decode('utf-8', errors='replace')
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"  [ERROR] Failed to fetch {url}: {e}", file=sys.stderr)
                return None


def strip_html(html):
    text = re.sub(r'<br\s*/?>', '\n', html, flags=re.I)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def to_yi(val_str, unit_str):
    """Convert value to 亿元 based on unit."""
    val = float(val_str)
    if '万亿' in unit_str:
        return val * 10000
    return val


def get_report_links(num_months=15):
    reports = []
    seen_titles = set()

    html = fetch_url(INDEX_URL)
    if not html:
        return reports

    page_nums = re.findall(r"11871-(\d+)\.html", html)
    max_page = max(int(p) for p in page_nums) if page_nums else 1

    for page_num in range(1, max_page + 1):
        if len(reports) >= num_months + 10:
            break

        if page_num == 1:
            page_html = html
        else:
            time.sleep(0.3)
            page_html = fetch_url(PAGE_URL.format(page_num))

        if not page_html:
            continue

        links = re.findall(
            r'href="(/diaochatongjisi/116219/116225/[^"]+)"[^>]*>([^<]+)',
            page_html,
        )

        for url, title in links:
            title = title.strip()
            if not re.search(r'\d{4}年.*金融统计数据报告', title):
                continue
            if any(x in title for x in ['小额贷款', '地区', '社会融资规模增量统计', '社会融资规模存量统计']):
                continue
            if title in seen_titles:
                continue
            seen_titles.add(title)
            reports.append((title, BASE + url))

    return reports


def parse_report(url):
    html = fetch_url(url)
    if not html:
        return None

    text = strip_html(html)
    data = {}

    period_match = re.search(r'(\d{4})年(\d{1,2})月', text)
    if period_match:
        data['year'] = int(period_match.group(1))
        data['month'] = int(period_match.group(2))
    else:
        return None

    if re.search(r'一季度', text[:800]):
        data['report_type'] = 'Q1'
    elif re.search(r'上半年', text[:800]):
        data['report_type'] = 'H1'
    elif re.search(r'前三季度', text[:800]):
        data['report_type'] = 'Q3'
    elif re.search(r'前两个月', text[:800]):
        data['report_type'] = '2M'
    else:
        data['report_type'] = 'monthly'

    # --- 1. 社会融资规模存量 ---
    m = re.search(r'社会融资规模存量为([\d.]+)万亿元[，,]同比增长([\d.]+)%', text)
    if m:
        data['shrz_stock'] = float(m.group(1))
        data['shrz_stock_yoy'] = float(m.group(2))

    # --- 2. 社会融资规模增量累计 ---
    m = re.search(r'社会融资规模增量[^。]*?为([\d.]+)万亿', text)
    if m:
        data['shrz_flow'] = float(m.group(1))

    # --- 3. M2 / M1 ---
    m = re.search(r'广义货币[^)]*\(M2\)[^，]*?余额([\d.]+)万亿元[，,]同比增长([\d.]+)%', text)
    if not m:
        m = re.search(r'M2[^，]*?余额([\d.]+)万亿元[，,]同比增长([\d.]+)%', text)
    if m:
        data['m2_stock'] = float(m.group(1))
        data['m2_yoy'] = float(m.group(2))

    m = re.search(r'狭义货币[^)]*\(M1\)[^，]*?余额([\d.]+)万亿元[，,]同比增长([\d.]+)%', text)
    if not m:
        m = re.search(r'M1[^，]*?余额([\d.]+)万亿元[，,]同比增长([\d.]+)%', text)
    if m:
        data['m1_stock'] = float(m.group(1))
        data['m1_yoy'] = float(m.group(2))

    # --- 4. 人民币贷款增量 ---
    m = re.search(r'人民币贷款增加([\d.]+)(万亿元|亿元)', text)
    if m:
        data['rmb_loans_add'] = to_yi(m.group(1), m.group(2))

    # --- 5. 住户贷款 ---
    m = re.search(r'住户贷款(增加|减少)([\d.]+)(万亿元|亿元)', text)
    if m:
        val = to_yi(m.group(2), m.group(3))
        if '减少' in m.group(1):
            val = -val
        data['hh_loans_add'] = val

    # 住户中长期贷款 - use captured unit group
    m = re.search(
        r'住户贷款[^。]*?短期贷款(增加|减少)([\d.]+)(万亿元|亿元)'
        r'[，；]中长期贷款(增加|减少)([\d.]+)(万亿元|亿元)',
        text,
    )
    if m:
        val = to_yi(m.group(5), m.group(6))
        if '减少' in m.group(4):
            val = -val
        data['hh_midlong_loans_add'] = val

    # --- 6. 企事业单位贷款 ---
    m = re.search(r'企[（(]事[）)]业单位贷款(增加|减少)([\d.]+)(万亿元|亿元)', text)
    if not m:
        m = re.search(r'企事业单位贷款(增加|减少)([\d.]+)(万亿元|亿元)', text)
    if m:
        val = to_yi(m.group(2), m.group(3))
        if '减少' in m.group(1):
            val = -val
        data['ent_loans_add'] = val

    # 企业中长期贷款 - use captured unit group
    m = re.search(
        r'企[（(]事[）)]业单位贷款[^。]*?中长期贷款(增加|减少)([\d.]+)(万亿元|亿元)',
        text,
    )
    if not m:
        m = re.search(
            r'企事业单位贷款[^。]*?中长期贷款(增加|减少)([\d.]+)(万亿元|亿元)',
            text,
        )
    if m:
        val = to_yi(m.group(2), m.group(3))
        if '减少' in m.group(1):
            val = -val
        data['ent_midlong_loans_add'] = val

    return data


def fetch_all_reports(num_months=12):
    print(f"Fetching report list from PBC website...")
    reports = get_report_links(num_months)

    monthly_reports = {}
    for title, url in reports:
        m = re.search(r'(\d{4})年(\d{1,2})月', title)
        if not m:
            continue
        year, month = int(m.group(1)), int(m.group(2))
        key = (year, month)
        is_quarter = any(q in title for q in ['一季度', '上半年', '前三季度', '全年'])
        if key not in monthly_reports:
            monthly_reports[key] = (title, url, is_quarter)
        elif not is_quarter and monthly_reports[key][2]:
            monthly_reports[key] = (title, url, is_quarter)

    sorted_keys = sorted(monthly_reports.keys(), reverse=True)[:num_months]

    all_data = []
    for i, key in enumerate(sorted_keys):
        title, url, is_quarter = monthly_reports[key]
        print(f"  [{i + 1}/{len(sorted_keys)}] Parsing: {title}")
        time.sleep(0.5)

        data = parse_report(url)
        if data:
            data['title'] = title
            data['is_quarter'] = is_quarter
            all_data.append(data)
        else:
            print(f"    [WARN] Failed to parse: {title}")

    all_data.sort(key=lambda x: (x['year'], x['month']))
    return all_data


def fmt(val, suffix='', dec=2):
    if val is None:
        return 'N/A'
    if isinstance(val, float):
        if dec == 0:
            return f'{int(val)}{suffix}'
        return f'{val:.{dec}f}{suffix}'
    return f'{val}{suffix}'


def print_table(data_list):
    if not data_list:
        print("No data to display.")
        return

    cols = [
        ('月份', 8, lambda d: f"{d['year']}.{d['month']:02d}"),
        ('社融存量\n(万亿)', 10, lambda d: fmt(d.get('shrz_stock'))),
        ('社融存量\n同比%', 8, lambda d: fmt(d.get('shrz_stock_yoy'), '%')),
        ('社融增量\n累计(万亿)', 10, lambda d: fmt(d.get('shrz_flow'))),
        ('M2同比%', 7, lambda d: fmt(d.get('m2_yoy'), '%')),
        ('M1同比%', 7, lambda d: fmt(d.get('m1_yoy'), '%')),
        ('人民币贷款\n增量(亿)', 11, lambda d: fmt(d.get('rmb_loans_add'), dec=0)),
        ('住户贷款\n(亿)', 10, lambda d: fmt(d.get('hh_loans_add'), dec=0)),
        ('住户中长期\n(亿)', 11, lambda d: fmt(d.get('hh_midlong_loans_add'), dec=0)),
        ('企业贷款\n(亿)', 10, lambda d: fmt(d.get('ent_loans_add'), dec=0)),
        ('企业中长期\n(亿)', 11, lambda d: fmt(d.get('ent_midlong_loans_add'), dec=0)),
        ('类型', 4, lambda d: d.get('report_type', '')),
    ]

    headers = [c[0] for c in cols]
    widths = [c[1] for c in cols]

    sep = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'

    print(sep)
    row1 = []
    row2 = []
    for h, w in zip(headers, widths):
        parts = h.split('\n')
        row1.append(parts[0].center(w + 2))
        row2.append(parts[1].center(w + 2) if len(parts) > 1 else ' ' * (w + 2))
    print('|' + '|'.join(row1) + '|')
    print('|' + '|'.join(row2) + '|')
    print(sep)

    for d in data_list:
        cells = []
        for _, w, fn in cols:
            val = str(fn(d))
            cells.append((' ' + val).ljust(w + 2))
        print('|' + '|'.join(cells) + '|')

    print(sep)
    print("\n注: 贷款增量为报告期内累计增量(单位:亿元)，社融存量单位为万亿元")
    print("    类型: Q1=一季度, H1=上半年, Q3=前三季度, 2M=前两月, monthly=月度/累计")
    print("    数据来源: 中国人民银行调查统计司")


if __name__ == '__main__':
    num_months = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(f"=== 央行金融统计数据（最近 {num_months} 个月）===\n")

    data = fetch_all_reports(num_months)
    print(f"\n成功获取 {len(data)} 份报告数据\n")

    print_table(data)

    json_path = '/home/andy/pbc_finance_data.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nJSON data saved to: {json_path}")
