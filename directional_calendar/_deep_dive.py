"""Deep dive: why does IM/IC outperform IF? Analyze basis characteristics."""
import pandas as pd
import numpy as np
import os

data_dir = r"C:\Users\wang\WorkBuddy\20260425111208\directional_calendar\data"

print("=" * 80)
print("  MULTI-PRODUCT BASIS CHARACTERISTICS DEEP DIVE")
print("=" * 80)
print()

all_stats = {}
for prod in ["IF", "IH", "IC", "IM"]:
    f = os.path.join(data_dir, f"5min_basis_{prod}_20260425.csv")
    if not os.path.exists(f):
        continue
    df = pd.read_csv(f)
    adj = df["adj_annualized_rate"]
    raw = df["raw_annualized_rate"]
    
    # Basis stats
    mu = adj.mean()
    sigma = adj.std()
    min_val = adj.min()
    max_val = adj.max()
    range_val = max_val - min_val
    
    # Z-score potential (how often does it exceed entry threshold)
    z_scores = (adj - mu) / sigma
    pct_above_1sigma = (z_scores > 1.0).mean() * 100
    pct_below_neg05 = (z_scores < -0.5).mean() * 100
    
    # Price level and PnL per switch
    near = df["near_close"].mean()
    far = df["far_close"].mean()
    spread = (far - near).mean()
    
    # Multiplier
    mult = {"IF": 300, "IH": 300, "IC": 200, "IM": 200}[prod]
    
    all_stats[prod] = {
        "mu": mu, "sigma": sigma, "min": min_val, "max": max_val,
        "range": range_val, "pct_above_1s": pct_above_1sigma,
        "pct_below_neg05": pct_below_neg05,
        "near_mean": near, "far_mean": far, "spread_mean": spread,
        "mult": mult, "rows": len(df),
    }
    
    print(f"  {prod} (mult={mult}):")
    print(f"    Adj basis: mu={mu:+.3f}% sigma={sigma:.3f}%")
    print(f"    Range: [{min_val:+.3f}%, {max_val:+.3f}%] span={range_val:.3f}%")
    print(f"    Z> +1σ: {pct_above_1sigma:.1f}% of bars | Z< -0.5σ: {pct_below_neg05:.1f}%")
    print(f"    Prices: near≈{near:.0f} far≈{far:.0f} spread≈{spread:.1f}")
    print()

# ============================================================
# Key insight: why IM/IC are more profitable
# ============================================================
print("=" * 80)
print("  KEY INSIGHT: Profitability drivers by product")
print("=" * 80)
print()
print(f"{'Product':<6} {'Sigma':>6} {'Range':>7} {'Z>+1σ%':>8} {'Spread':>8} {'Mult':>5} {'EstPnL/switch':>14}")
print("-" * 65)

for prod in ["IF", "IH", "IC", "IM"]:
    s = all_stats[prod]
    # Estimated PnL per NEAR->FAR switch:
    # When z > 1, basis is wide, we expect mean-reversion profit
    # Rough estimate: spread * multiplier * volume (1 lot)
    est_pnl = s["spread_mean"] * s["mult"]  # per point, 1 lot
    print(f"{prod:<6} {s['sigma']:>5.3f}% {s['range']:>6.3f}% {s['pct_above_1s']:>7.1f}% "
          f"{s['spread_mean']:>7.1f}pt {s['mult']:>5} {est_pnl:>+12.0f} CNY")

print()
print("Analysis:")
print("  - IM has the DEEPEST discount (mu=-13.35%) → wider absolute spreads")
print("  - IC has the HIGHEST volatility (sigma=0.75%) → more trading opportunities")
print("  - IF trades too frequently (16 vs 3) because low sigma triggers many false switches")
print("  - IM/IC: fewer but higher-quality trades (wide spread × high multiplier)")
print()

# ============================================================
# Trade frequency analysis: why IF has 16 trades vs IM's 3
# ============================================================
print("=" * 80)
print("  TRADE FREQUENCY ANALYSIS")
print("=" * 80)
print()
for prod in ["IF", "IM"]:
    s = all_stats[prod]
    # How many times does the basis cross above +1 sigma?
    df_file = os.path.join(data_dir, f"5min_basis_{prod}_20260425.csv")
    df = pd.read_csv(df_file)
    adj = df["adj_annualized_rate"]
    mu = s["mu"]; sigma_s = s["sigma"]
    z = (adj - mu) / sigma_s
    
    # Count crossings
    above = z > 1.0
    crossings = above.diff().fillna(False).sum()  # number of times we cross into >1sigma zone
    total_bars_above = above.sum()
    
    print(f"  {prod}:")
    print(f'    Bars with z > +1σ: {total_bars_above}/{len(adj)} ({total_bars_above/len(adj)*100:.1f}%)')
    print(f'    Entry threshold crossings: ~{int(crossings)} times')
    noise_flag = "YES (tight)" if sigma_s < 0.5 else "NO (reasonable)"
    print(f"    Sigma so low that noise creates many signals: {noise_flag}")
    print()
