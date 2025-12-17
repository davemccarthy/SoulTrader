#!/usr/bin/env python3
"""Test script to inspect StockStory HTML structure"""

import requests
from bs4 import BeautifulSoup
import re

url = "https://www.barchart.com/news/authors/285/stockstory"

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
}

response = requests.get(url, headers=headers, timeout=10)
response.raise_for_status()

soup = BeautifulSoup(response.content, 'html.parser')

# Find all article links
all_links = soup.find_all('a', href=True)
print(f"Total links found: {len(all_links)}\n")

article_links = []
for link in all_links:
    href = link.get('href', '')
    text = link.get_text(strip=True)
    
    if '/story/news/' in href and len(text) > 20:
        if any(skip in href.lower() for skip in ['/authors/', '/exclusives', '/chart-of', '/categories']):
            continue
        
        article_links.append((link, href, text))

print(f"Article links found: {len(article_links)}\n")

# Show first 3 article links and their surrounding HTML
for i, (link, href, text) in enumerate(article_links[:3]):
    print(f"\n=== Article {i+1} ===")
    print(f"Text: {text[:100]}")
    print(f"Href: {href}")
    print(f"\nLink element HTML:")
    print(str(link)[:500])
    print(f"\nParent element:")
    parent = link.parent
    if parent:
        print(f"Parent tag: {parent.name}")
        print(f"Parent classes: {parent.get('class', [])}")
        print(f"Parent text: {parent.get_text(strip=True)[:300]}")
        print(f"\nParent HTML (first 800 chars):")
        print(str(parent)[:800])
    
    # Check siblings
    print(f"\nNext siblings:")
    for j, sibling in enumerate(link.find_next_siblings()[:3]):
        print(f"  Sibling {j+1}: {sibling.name} - {sibling.get_text(strip=True)[:100]}")
    
    # Check parent siblings
    if parent:
        print(f"\nParent next siblings:")
        for j, parent_sib in enumerate(parent.find_next_siblings()[:3]):
            print(f"  Parent sibling {j+1}: {parent_sib.name} - {parent_sib.get_text(strip=True)[:100]}")
    
    print("\n" + "="*80)





























































