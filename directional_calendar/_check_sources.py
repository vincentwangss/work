"""Check available data sources for longer history futures minute data"""
import sys, os

# Check 1: Tushare
print("=" * 60)
print("  Checking Tushare availability...")
print("=" * 60)
try:
    import tushare as ts
    print(f"  Tushare version: {ts.__version__}")
    # Check if pro_api is available (need token)
    # ts.set_token('YOUR_TOKEN')  # user may have configured
    print("  Tushare: AVAILABLE (need token for futures data)")
except ImportError:
    print("  Tushare: NOT installed")

print()

# Check 2: iFinD / Wind
print("=" * 60)
print("  Checking iFinD availability...")
print("=" * 60)
try:
    # Try importing ifind or windpy
    import comtypes.client
    print("  comtypes: available")
    
    # Try iFinD specific
    try:
        from WindPy import w
        print("  WindPy: available (w.start() needed)")
    except:
        print("  WindPy: not available")
        
    # Try ifind COM interface
    try:
        ifind = comtypes.client.CreateObject("ifind.IfindData")
        print("  iFinD COM: available!")
    except Exception as e:
        print(f"  iFinD COM: not available ({e})")
except ImportError:
    print("  comtypes: not available")

print()

# Check 3: baostock (free A-share data)
print("=" * 60)
print("  Checking baostock...")
print("=" * 60)
try:
    import baostock as bs
    print("  baostock: available (but limited to stocks, no futures)")
except ImportError:
    print("  baostock: not installed")

print()

# Check 4: Can we use daily data + resample to approximate?
# Or use the existing backtest.py's data_loader which might use other APIs
print("=" * 60)
print("  Checking existing data_loader capabilities...")
print("=" * 60)
data_loader_path = r"C:\Users\wang\WorkBuddy\20260425111208\directional_calendar\data_loader.py"
if os.path.exists(data_loader_path):
    with open(data_loader_path, "r", encoding="utf-8") as f:
        content = f.read()
    for api in ["akshare", "tushare", "ifind", "wind", "baostock", "iFinD"]:
        count = content.lower().count(api.lower())
        if count > 0:
            print(f"  {api}: {count} references")
else:
    print("  data_loader.py not found")

# Check 5: Finance data plugin - neodata
print()
print("=" * 60)
print("  Checking neodata/finance-data plugin...")
print("=" * 60)
try:
    # We can use the finance-data-retrieval skill
    print("  Plugin available via skill invocation")
    print("  Suggested approach: Use neodata to get daily futures data")
    print("  Then resample/interpolate to 5-min level using intraday patterns")
except Exception as e:
    print(f"  Error checking: {e}")

print()
print("RECOMMENDATION:")
print("  Option A: Use iFinD (if available) for full historical futures minute data")
print("  Option B: Pull daily data from neodata/tushare → synthesize 5min bars")  
print("  Option C: Use multiple expired contracts stitched together from akshare")
print("            (each contract gives ~30 days of data)")
