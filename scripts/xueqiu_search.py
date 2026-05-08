#!/usr/bin/env python3
"""
雪球搜索文章 - 使用已登录的Playwright持久化上下文
输出JSON结果到stdout
"""
import json, time, re, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

USER_DATA_DIR = str(Path.home() / '.hermes' / 'browser_data' / 'xueqiu')
OUTPUT_FILE = str(Path.home() / '.hermes' / 'data' / 'xueqiu_search_results.json')

def search_xueqiu(queries):
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR,
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox'],
        )
        page = context.pages[0] if context.pages else context.new_page()
        
        all_articles = []
        
        for query in queries:
            encoded = query  # already URL-encoded or simple
            url = f'https://xueqiu.com/k?q={encoded}&type=1'
            page.goto(url, wait_until='networkidle', timeout=30000)
            time.sleep(3)
            
            # Extract text content
            text = page.inner_text('body')
            
            # Extract article links from HTML
            articles = page.evaluate("""() => {
                const results = [];
                // Look for discussion items
                const items = document.querySelectorAll('[class*="container"], [class*="item"], [class*="status"], [class*="timeline"]');
                document.querySelectorAll('a[href]').forEach(a => {
                    const href = a.getAttribute('href') || '';
                    const text = a.textContent.trim();
                    // Match article URLs
                    if ((href.includes('/today/') || href.includes('/column/')) && text.length > 8) {
                        results.push({
                            url: href.startsWith('http') ? href : 'https://xueqiu.com' + href,
                            title: text.substring(0, 100),
                            type: 'column' if '/column/' in href else 'today'
                        });
                    }
                });
                return results;
            }""")
            
            # Also extract the page text for articles shown inline
            # Parse the structured text to find articles
            lines = text.split('\n')
            current_article = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Look for article titles with "专栏" prefix or dates
                if '专栏' in line and len(line) > 10:
                    title = line.replace('专栏', '').strip()
                    if title:
                        all_articles.append({
                            'title': title,
                            'query': query,
                            'source': 'inline'
                        })
            
            for a in articles:
                a['query'] = query
                a['source'] = 'link'
                all_articles.append(a)
            
            time.sleep(2)
        
        context.close()
    
    # Deduplicate
    seen_titles = set()
    unique = []
    for a in all_articles:
        t = a.get('title', '')[:50]
        if t and t not in seen_titles:
            seen_titles.add(t)
            unique.append(a)
    
    # Save
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    
    return unique

if __name__ == '__main__':
    queries = sys.argv[1:] if len(sys.argv) > 1 else [
        '%E4%BA%A4%E6%98%93%E4%BD%93%E7%B3%BB%20%E6%9E%84%E5%BB%BA',
        '%E6%8A%95%E8%B5%84%E4%BA%A4%E6%98%93%E4%BD%93%E7%B3%BB',
        '%E4%B8%AA%E4%BA%BA%E6%8A%95%E8%B5%84%E8%80%85%20%E4%BD%93%E7%B3%BB',
    ]
    
    results = search_xueqiu(queries)
    print(json.dumps(results, ensure_ascii=False, indent=2))
