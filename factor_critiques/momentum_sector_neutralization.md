# Factor Critique: Momentum Without Sector Neutralization

## Factor Definition Under Review

**Name:** Simple Price Momentum  
**Formula:** `Momentum_i = (P_t / P_{t-12m}) - 1`  
**Universe:** All stocks above median market cap  
**Implementation:** Long top decile, rebalance monthly

---

## The Problem: Hidden Sector Bets

### What's Missing

This momentum factor definition does NOT neutralize for sector exposure. This is a critical omission that transforms what appears to be a "momentum" strategy into a **sector rotation strategy with momentum characteristics**.

### Why It Matters

When you rank all stocks by price momentum without sector adjustment:

1. **Sector clustering:** Top momentum stocks tend to cluster in 1-2 sectors
2. **You're not testing momentum:** You're testing "which sectors are hot"
3. **Different risk profile:** Sector risk ≠ stock selection risk
4. **Capacity constraints:** Sector-driven strategies have lower capacity

### Concrete Example

Consider a period where Technology is up 40% and Utilities are down 5%:

| Stock | Sector | 12m Return | Within-Sector Rank |
|-------|--------|------------|-------------------|
| NVDA | Tech | +150% | Top 10% of Tech |
| AAPL | Tech | +35% | Bottom 30% of Tech |
| AMD | Tech | +45% | Middle of Tech |
| NEE | Utilities | +2% | Top 5% of Utilities |
| DUK | Utilities | -8% | Bottom 20% of Utilities |

**Without sector neutralization:**
- Portfolio is: NVDA, AMD, AAPL (all Tech)
- You're long Technology, not "winners"
- AAPL is in top decile despite being a Tech laggard

**With sector neutralization:**
- NVDA selected (top within Tech)
- NEE selected (top within Utilities)
- Portfolio is diversified across sectors
- You're buying within-sector winners

---

## Impact Analysis

### Historical Bias

During 1990-2020, Tech and Growth outperformed:
- Non-neutralized momentum: ~60% Tech exposure
- Returns inflated by sector tailwind
- When Tech crashed (2000-2002), momentum "failed"
- Actually, it was a sector bet that failed

### Risk Decomposition

For non-neutralized momentum:
```
Total Variance = Sector Variance + Stock Selection Variance + Residual
              ≈    60%          +         25%              +   15%
```

For sector-neutralized momentum:
```
Total Variance = Sector Variance + Stock Selection Variance + Residual
              ≈    10%          +         65%              +   25%
```

**Conclusion:** Non-neutralized momentum is primarily a sector bet.

### Performance Attribution

If non-neutralized momentum returns 8% annually:
- Sector timing contribution: ~5%
- Stock selection contribution: ~3%

If sector-neutralized momentum returns 5% annually:
- Sector contribution: ~0% (by design)
- Stock selection contribution: ~5%

**The "true" momentum premium is ~5%, not 8%.**

---

## The Fix: Sector-Neutral Momentum

### Corrected Definition

```python
def calculate_sector_neutral_momentum(prices, sectors, date):
    """
    Calculate momentum as deviation from sector average.
    
    This isolates stock selection skill from sector rotation.
    """
    # Calculate raw momentum
    raw_momentum = calculate_raw_momentum(prices, date)
    
    # Calculate sector average momentum
    sector_avg = raw_momentum.groupby(sectors).transform('mean')
    
    # Sector-neutral momentum = raw - sector average
    neutral_momentum = raw_momentum - sector_avg
    
    return neutral_momentum
```

### Alternative: Cross-Sectional Z-Score Within Sector

```python
def calculate_momentum_zscore_within_sector(prices, sectors, date):
    """
    Calculate momentum z-score within each sector.
    
    This ensures equal representation from each sector.
    """
    raw_momentum = calculate_raw_momentum(prices, date)
    
    # Z-score within sector
    def zscore(x):
        return (x - x.mean()) / x.std()
    
    momentum_z = raw_momentum.groupby(sectors).transform(zscore)
    
    return momentum_z
```

---

## When to Use Each Approach

### Use Raw (Non-Neutralized) Momentum When:
- You explicitly WANT sector rotation
- Sector timing is part of your thesis
- You have capacity constraints (fewer positions)
- You understand and accept the sector risk

### Use Sector-Neutral Momentum When:
- You want to test "pure" momentum
- Sector exposure should be controlled separately
- You have sector exposure limits
- You want diversified risk sources

### Use Industry-Neutral Momentum When:
- Even finer control needed
- Industry dynamics differ within sectors
- Reducing factor crowding

---

## Implications for Reported Results

### If You See Non-Neutralized Momentum Results

**Sharpe 0.6, Annual Return 8%**

Questions to ask:
1. What % of return came from sector bets?
2. What was the Tech/Growth exposure over time?
3. How did it perform when Tech underperformed?
4. Is this skill or sector luck?

### Proper Benchmark

Non-neutralized momentum should be compared to:
- Sector rotation strategies
- Growth/Value factor
- NOT the market (unfair comparison)

Sector-neutralized momentum should be compared to:
- Other stock selection factors
- Sector-neutral value, quality
- Pure alpha strategies

---

## Red Flags in Factor Implementations

| Red Flag | What It Means |
|----------|---------------|
| No mention of sector neutralization | Likely has sector exposure |
| Top holdings all in one sector | Sector bet, not stock picking |
| Performance correlates with sector ETFs | Not pure factor |
| Strategy "works" 1995-2000, fails 2000-2002 | It was a Tech bet |

---

## Summary

**The Critique:**  
Momentum without sector neutralization conflates stock selection with sector rotation. The reported "momentum premium" is substantially attributable to sector bets rather than identifying winning stocks.

**The Fix:**  
Subtract sector average momentum from each stock's momentum before ranking. This isolates the stock selection component.

**The Impact:**  
- Returns likely 2-3% lower annually
- But risk is also lower and more diversified
- The remaining return is "pure" momentum, more likely to persist

---

## References

- Fama, E. F., & French, K. R. (2012). Size, value, and momentum in international stock returns.
- Asness, C. S., et al. (2013). Value and Momentum Everywhere.
- Israel, R., & Moskowitz, T. J. (2013). The role of shorting, firm size, and time on market anomalies.
