"""debug_html.py - 排查 spread_matrix.html 的 JS 问题"""
import re

html_path = r'c:\Users\wang\WorkBuddy\20260425111208\directional_calendar\reports\spread_matrix.html'
with open(html_path, 'r', encoding='utf-8') as f:
    c = f.read()

scripts = re.findall(r'<script>(.*?)</script>', c, re.DOTALL)
print(f"Found {len(scripts)} <script> blocks\n")

for idx, s in enumerate(scripts):
    if len(s) < 100:
        print(f"Script {idx}: {len(s)} chars (small)")
        continue
    
    print(f"=== Script {idx}: {len(s)} chars ===")
    opens = s.count('{')
    closes = s.count('}')
    paren_o = s.count('(')
    paren_c = s.count(')')
    bracket_o = s.count('[')
    bracket_c = s.count(']')
    print(f"  Braces: {{ {opens} / }} {closes}  diff={opens-closes}")
    print(f"  Parens: ( {paren_o} / ) {paren_c}  diff={paren_o-paren_c}")
    print(f"  Brackets: [ {bracket_o} / ] {bracket_c}  diff={bracket_o-bracket_c}")
    
    if 'PRODUCT_INFO' in s:
        print(f"  PRODUCT_INFO ref: YES")
        if 'const PRODUCT_INFO' in s or 'var PRODUCT_INFO' in s:
            print("  -> DEFINED in this script")
        else:
            print("  *** NOT DEFINED HERE ***")
    
    # Check for common JS errors
    lines = s.split('\n')
    for li, line in enumerate(lines):
        stripped = line.strip()
        # Look for template literal issues
        if '${' in stripped and '`' not in stripped:
            print(f"  L{li}: ${{ outside backtick? -> {stripped[:100]}")

# Also check HTML onclicks
print("\n=== ONCLICK handlers ===")
for m in re.findall(r'onclick="([^"]*)"', c):
    print(f"  {m}")

print("\n=== ONCHANGE handlers ===")
for m in re.findall(r'onchange="([^"]*)"', c):
    print(f"  {m}")

# Look for any obvious syntax issues
print("\n=== Potential issues ===")
# Check for bare ${...} that might be unescaped
bad_patterns = re.findall(r'\$\{[a-zA-Z_]', c)
if bad_patterns:
    for p in set(bad_patterns):
        print(f"  Bare ${{var}} found: {p}")
else:
    print("  No bare ${var} found - OK")
