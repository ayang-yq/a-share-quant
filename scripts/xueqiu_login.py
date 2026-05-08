#!/usr/bin/env python3
"""
雪球手动登录 + Cookie持久化
用法: python3 xueqiu_login.py
登录成功后自动保存cookie到 ~/.hermes/secrets/xueqiu_cookies.json
"""

import json
import os
import sys
import time
from pathlib import Path

SECRETS_DIR = Path.home() / ".hermes" / "secrets"
SECRETS_DIR.mkdir(parents=True, exist_ok=True)
COOKIE_FILE = SECRETS_DIR / "xueqiu_cookies.json"
USER_DATA_DIR = Path.home() / ".hermes" / "browser_data" / "xueqiu"

XUEQIU_URL = "https://xueqiu.com"


def save_cookies(cookies: list[dict]):
    """Save cookies to JSON file, also generate a cookie string."""
    # Save full cookies as JSON
    with open(COOKIE_FILE, "w") as f:
        json.dump(cookies, f, indent=2)
    
    # Also save as cookie string for easy HTTP header use
    cookie_str = "; ".join(f"{c['name']}={c['value']}" for c in cookies)
    cookie_str_file = SECRETS_DIR / "xueqiu_cookies.txt"
    with open(cookie_str_file, "w") as f:
        f.write(cookie_str)
    
    print(f"\n✓ Cookies saved to: {COOKIE_FILE}")
    print(f"✓ Cookie string saved to: {cookie_str_file}")
    return COOKIE_FILE


def check_login(cookies: list[dict]) -> bool:
    """Check if login cookies are present."""
    names = {c["name"] for c in cookies}
    return "xq_a_token" in names and "xq_is_login" in names


def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Installing playwright...")
        os.system(f"{sys.executable} -m pip install playwright")
        os.system(f"{sys.executable} -m playwright install chromium")
        from playwright.sync_api import sync_playwright

    print("=" * 50)
    print("  雪球手动登录工具")
    print("=" * 50)
    print()
    print(f"浏览器数据目录: {USER_DATA_DIR}")
    print(f"Cookie保存位置: {COOKIE_FILE}")
    print()

    with sync_playwright() as p:
        # Launch with persistent context (cookies survive across sessions)
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(USER_DATA_DIR),
            headless=False,
            viewport={"width": 1280, "height": 800},
            locale="zh-CN",
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )

        page = context.pages[0] if context.pages else context.new_page()

        # Navigate to xueqiu
        print("正在打开雪球...")
        page.goto(XUEQIU_URL, wait_until="domcontentloaded")

        # Check if already logged in from previous session
        cookies = context.cookies()
        if check_login(cookies):
            print("✓ 检测到已有登录态！Cookie已更新。")
            save_cookies(cookies)
            context.close()
            return

        # Wait for manual login
        print()
        print("请在浏览器中手动登录雪球：")
        print("  1. 输入手机号")
        print("  2. 获取并输入验证码")
        print("  3. 完成登录")
        print()
        print("等待登录中... (检测到登录后自动保存cookie)")

        # Poll for login cookie every 2 seconds, timeout after 5 minutes
        start = time.time()
        logged_in = False
        while time.time() - start < 300:
            time.sleep(3)
            cookies = context.cookies()
            if check_login(cookies):
                logged_in = True
                break

        if logged_in:
            print()
            print("✓ 登录成功！")
            # Wait a bit for all cookies to settle
            time.sleep(2)
            cookies = context.cookies()
            
            # Extract user info
            u_id = next((c["value"] for c in cookies if c["name"] == "u"), "unknown")
            print(f"  用户ID: {u_id}")
            
            save_cookies(cookies)
            print()
            print("✓ 下次运行将自动使用已保存的登录态")
            print("  (如cookie过期，重新运行此脚本登录即可)")
        else:
            print("\n✗ 登录超时(5分钟)，请重新运行")

        context.close()


if __name__ == "__main__":
    main()
