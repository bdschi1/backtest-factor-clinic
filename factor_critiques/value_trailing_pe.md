# Factor Critique: Value Factor Using Trailing P/E

## Factor Definition Under Review

**Name:** Price-to-Earnings Value Factor  
**Formula:** `Value = 1 / (Price / Trailing_12M_EPS)`  
**Implementation:** Long lowest P/E decile, short highest P/E decile  
**Rebalance:** Quarterly

---

## Critical Issues

### Issue 1: Negative Earnings Not Handled

**The Problem:**  
Trailing P/E is undefined or negative when earnings are negative. Many implementations simply exclude these stocks, but this introduces a systematic bias.

**Why It Matters:**
- Cyclical stocks have negative earnings at cycle bottoms
- These are often the best "value" opportunities
- Excluding them means missing turnaround plays
- Strategy becomes "profitable companies value" not "value"

**Example - GFC Recovery:**

| Stock | Dec 2008 | Trailing EPS | P/E | Treatment |
|-------|----------|--------------|-----|-----------|
| C | $3.77 | -$15.30 | Negative | Excluded |
| BAC | $14.08 | -$2.81 | Negative | Excluded |
| JPM | $25.60 | +$0.81 | 31.6x | "Expensive" |

Citigroup and Bank of America were the best value opportunities, but excluded from "value" portfolio because of negative trailing earnings. A forward-looking measure would have captured them.

**The Fix:**
```python
# Option 1: Use forward P/E (analyst estimates)
value = 1 / (price / forward_eps_estimate)

# Option 2: Use normalized earnings (10-year average)
normalized_eps = eps.rolling(40).mean()  # 40 quarters = 10 years
value = 1 / (price / normalized_eps)

# Option 3: Use enterprise value / EBITDA (harder to go negative)
value = ebitda / enterprise_value
```

---

### Issue 2: Reporting Lag Not Accounted

**The Problem:**  
Trailing 12-month earnings aren't known until fiscal year-end filings, which can be 60-90 days after the period ends.

**Timeline:**
```
Dec 31, 2023: Fiscal year ends
Feb 15, 2024: Earliest 10-K filing (45 days for accelerated filers)
Mar 1, 2024: Most large caps have filed
Mar 31, 2024: Deadline for large accelerated filers

If your backtest uses Dec 31 earnings on Jan 1:
  → Look-ahead bias of 45-90 days
  → Sharpe inflation of ~0.1-0.2
```

**Example:**

You're making a decision on January 15, 2024. The "trailing 12-month" earnings through December 2023 aren't available yet. The most recent complete data is Q3 2023 (filed by November).

**The Fix:**
```python
def get_available_earnings(date, earnings_data):
    """Get earnings that were actually available at decision date."""
    
    # Assume 60-day reporting lag
    REPORTING_LAG = pd.Timedelta(days=60)
    
    # Only use earnings from periods ending before (date - lag)
    available_date = date - REPORTING_LAG
    
    # Get most recent 4 quarters ending before available_date
    available_quarters = earnings_data[
        earnings_data['period_end'] <= available_date
    ].tail(4)
    
    return available_quarters['eps'].sum()
```

---

### Issue 3: Earnings Quality Not Considered

**The Problem:**  
Trailing earnings include one-time items, making P/E misleading.

**Examples of Misleading P/E:**

| Situation | Impact on P/E | Reality |
|-----------|---------------|---------|
| Asset sale gain | Artificially low P/E | Not recurring |
| Restructuring charge | Artificially high P/E | May improve future |
| Goodwill impairment | High or negative P/E | Non-cash |
| Tax benefit | Low P/E | One-time |

**Real Example - GE (2018):**
- Reported EPS: -$2.64 (including $22B goodwill impairment)
- Adjusted EPS: +$0.65
- Using reported: Excluded from value (negative)
- Using adjusted: P/E = 18x (moderate value)

**The Fix:**
```python
# Use adjusted/operating earnings when available
def get_quality_adjusted_eps(stock, date):
    """
    Use adjusted EPS if available, with quality checks.
    """
    reported_eps = get_reported_eps(stock, date)
    adjusted_eps = get_adjusted_eps(stock, date)  # From I/B/E/S
    
    # If adjusted is very different from reported, investigate
    if abs(adjusted_eps - reported_eps) > 0.5 * abs(reported_eps):
        # Large adjustment - might be legitimate or manipulation
        # Use more conservative (lower) of the two
        return min(adjusted_eps, reported_eps)
    
    return adjusted_eps
```

---

### Issue 4: Industry Effects Not Neutralized

**The Problem:**  
Different industries have structurally different P/E ratios. A "low P/E" strategy without industry adjustment becomes an **industry rotation strategy**.

**Structural P/E Differences:**

| Industry | Typical P/E | Why |
|----------|-------------|-----|
| Utilities | 15-18x | Stable, low growth |
| Banks | 10-14x | Regulated, leverage |
| Tech | 25-40x | High growth |
| Pharma | 12-20x | Patent risk |
| Industrials | 15-20x | Cyclical |

**Consequence:**  
A low P/E portfolio is structurally:
- Overweight: Banks, Utilities, Industrials
- Underweight: Tech, Healthcare, Consumer

When Tech outperforms (1990s, 2010s), "value" underperforms. This is mostly a sector bet, not a stock selection signal.

**The Fix:**
```python
def calculate_industry_neutral_value(pe_ratios, industries):
    """
    Calculate value as deviation from industry median.
    
    A stock with P/E of 15 in Banking (median 12) is expensive.
    A stock with P/E of 15 in Tech (median 30) is cheap.
    """
    industry_median = pe_ratios.groupby(industries).transform('median')
    
    # Value = how cheap vs. industry peers
    # Negative = cheaper than peers (good for value)
    relative_value = (pe_ratios - industry_median) / industry_median
    
    return -relative_value  # Flip sign so higher = more value
```

---

### Issue 5: P/E Level vs. Change

**The Problem:**  
Static P/E doesn't capture whether a stock is getting cheaper or more expensive.

**Example:**

| Stock | P/E Today | P/E 1Y Ago | Signal |
|-------|-----------|------------|--------|
| A | 10x | 15x | P/E compressing - bullish |
| B | 10x | 8x | P/E expanding - bearish |
| C | 20x | 30x | Was expensive, now cheaper |

Stocks A and B have the same P/E but very different trajectories. Stock C has a higher P/E than A/B but is on a better trajectory.

**The Fix:**
```python
def calculate_value_with_momentum(pe_current, pe_past):
    """
    Combine value level with value momentum.
    
    Low P/E that's falling = best value signal
    High P/E that's rising = worst value signal
    """
    # Value level (low P/E = high value)
    value_level = 1 / pe_current
    
    # Value momentum (P/E compression = good)
    value_momentum = (pe_past - pe_current) / pe_past
    
    # Combined signal
    combined = 0.5 * zscore(value_level) + 0.5 * zscore(value_momentum)
    
    return combined
```

---

## Impact on Reported Results

### Typical Claim:
"Value factor has Sharpe 0.4, annual return 4% over benchmark"

### After Adjustments:

| Issue | Sharpe Impact | Return Impact |
|-------|---------------|---------------|
| Negative earnings exclusion | -0.05 | Variable |
| Reporting lag | -0.05 to -0.10 | -0.5% |
| No quality adjustment | -0.05 | Variable |
| Industry exposure | -0.10 to -0.20 | -1% to -2% |

**Adjusted expectations:**  
- Sharpe: 0.15-0.25 (not 0.4)
- Alpha: 1-2% (not 4%)

---

## Recommended Value Factor Construction

```python
def construct_robust_value_factor(data, date):
    """
    Build a robust value factor addressing all critiques.
    """
    # 1. Use multiple value metrics (not just P/E)
    metrics = {
        'ep': data['earnings'] / data['price'],           # E/P
        'bp': data['book_value'] / data['price'],         # B/P
        'sp': data['sales'] / data['price'],              # S/P
        'cfp': data['cashflow'] / data['price'],          # CF/P
        'ebitda_ev': data['ebitda'] / data['ev']          # EBITDA/EV
    }
    
    # 2. Winsorize extremes (handle negative earnings)
    for key in metrics:
        metrics[key] = winsorize(metrics[key], limits=[0.01, 0.01])
    
    # 3. Industry neutralize each metric
    for key in metrics:
        metrics[key] = metrics[key].groupby(data['industry']).transform(
            lambda x: (x - x.mean()) / x.std()
        )
    
    # 4. Combine into composite
    value_composite = sum(metrics.values()) / len(metrics)
    
    # 5. Apply reporting lag
    # (Assume data already lagged appropriately)
    
    return value_composite
```

---

## Summary

**The Critique:**  
Simple trailing P/E as a value factor has multiple issues: negative earnings handling, reporting lag, earnings quality, industry effects, and ignoring valuation trends.

**The Fix:**  
Use a composite of multiple value metrics, industry-neutralize, apply proper reporting lags, and consider valuation momentum.

**The Impact:**  
- Raw P/E "value" factor returns are ~50% overstated
- Industry neutralization reveals the "true" value premium is smaller but more robust
- Proper implementation still shows positive alpha, just not as dramatic

---

## References

- Asness, C. S., & Frazzini, A. (2013). The Devil in HML's Details.
- Fama, E. F., & French, K. R. (1992). The Cross-Section of Expected Stock Returns.
- Lakonishok, J., Shleifer, A., & Vishny, R. W. (1994). Contrarian Investment, Extrapolation, and Risk.
- Novy-Marx, R. (2013). The Other Side of Value.
