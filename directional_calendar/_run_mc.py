import subprocess, sys, os

sys.path.insert(0, os.path.dirname(__file__))
# 直接导入并运行，捕获stdout
result = subprocess.run(
    [r'D:\veighna_studio\python.exe', os.path.join(os.path.dirname(__file__), '_multi_contract_bt.py')],
    capture_output=True, text=True, encoding='utf-8', errors='replace',
    cwd=os.path.dirname(__file__)
)

with open(os.path.join(os.path.dirname(__file__), '_mc_output.txt'), 'w', encoding='utf-8') as f:
    f.write("=== STDOUT ===\n")
    f.write(result.stdout)
    f.write("\n\n=== STDERR (last 5KB) ===\n")
    f.write(result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr)
    f.write(f"\n\n=== returncode: {result.returncode} ===")

print(f"Done. stdout={len(result.stdout)} chars, stderr={len(result.stderr)} chars")
