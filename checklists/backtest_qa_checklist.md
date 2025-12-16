# Backtest Quality Assurance Checklist

A systematic checklist for evaluating the validity of quantitative backtests.
Use this to identify common issues before trusting any backtest results.

---

## Quick Summary Checklist

| Category | Key Question | Pass? |
|----------|--------------|-------|
| **Look-Ahead** | Is ALL data point-in-time? | ☐ |
| **Survivorship** | Does data include delisted stocks? | ☐ |
| **Costs** | Are transaction costs realistic? | ☐ |
| **Execution** | Can these trades actually be made? | ☐ |
| **Overfitting** | Is out-of-sample testing done properly? | ☐ |
| **Data Quality** | Are corporate actions handled? | ☐ |
| **Capacity** | Can this strategy manage real money? | ☐ |

---

## Detailed Checklist

### 1. LOOK-AHEAD BIAS

**Definition:** Using information that wasn't available at the time of the trading decision.

#### 1.1 Price Data
- [ ] Entry prices use next-day open/VWAP (not same-day close)
- [ ] Exit prices are after decision, not before
- [ ] No use of future highs/lows in position sizing
- [ ] Index/ETF prices don't use intraday when stock uses close

#### 1.2 Fundamental Data
- [ ] Earnings data includes reporting lag (60-90 days for annual, 45 days for quarterly)
- [ ] Restatements handled (using as-reported, not restated)
- [ ] Point-in-time database used (not current values applied historically)
- [ ] Fiscal year-end differences accounted for

#### 1.3 Index/Universe Membership
- [ ] Universe determined using point-in-time constituents
- [ ] Additions/deletions dated correctly
- [ ] IPO date restrictions respected

#### 1.4 Technical Indicators
- [ ] Moving averages exclude current day if trading on close
- [ ] Signals generated BEFORE prices used for execution
- [ ] No circular references (using tomorrow's price in today's signal)

#### Common Red Flags:
- Sharpe > 2.0 for simple strategies
- Perfect timing of entries/exits
- No losing months
- Strategy works immediately from day 1

---

### 2. SURVIVORSHIP BIAS

**Definition:** Only including securities that survived to the present, excluding bankruptcies, delistings, and acquisitions.

#### 2.1 Data Coverage
- [ ] Delisted stocks included in universe
- [ ] Bankrupt companies have price history until delisting
- [ ] Acquired companies included until acquisition date
- [ ] Companies dropped from index included for historical periods

#### 2.2 Delisting Treatment
- [ ] Delisting returns calculated correctly (usually negative)
- [ ] Delisting date known and used
- [ ] Acquisition premiums not assumed for distressed cases
- [ ] Return on final trading day included

#### 2.3 Universe Construction
- [ ] Historical index constituents used (not current)
- [ ] Point-in-time constituent data sourced
- [ ] Approximately 40% turnover over 20 years for S&P 500

#### Estimated Impact:
| Strategy Type | Survivorship Bias Impact |
|---------------|-------------------------|
| Long-only equity | +1-2% annual return |
| Small cap | +2-4% annual return |
| Value (low P/E) | +1-3% annual return |
| Momentum | +0.5-1% annual return |

---

### 3. TRANSACTION COSTS & SLIPPAGE

**Definition:** The real-world costs of implementing trades.

#### 3.1 Commission Costs
- [ ] Broker commissions included
- [ ] Exchange fees included
- [ ] SEC fees included (for US equities)

#### 3.2 Market Impact
- [ ] Bid-ask spread accounted for
- [ ] Price impact for large orders modeled
- [ ] Impact scales with position size
- [ ] Different costs for different market caps

#### 3.3 Typical Cost Assumptions
| Market Cap | One-Way Cost (bps) |
|------------|-------------------|
| Mega cap (>$100B) | 5-10 |
| Large cap ($10-100B) | 10-20 |
| Mid cap ($2-10B) | 20-40 |
| Small cap (<$2B) | 40-100 |

#### 3.4 Slippage
- [ ] Market orders don't execute at quoted price
- [ ] Delay between signal and execution modeled
- [ ] Partial fills considered for illiquid names

---

### 4. EXECUTION REALISM

**Definition:** Can the backtest trades actually be executed in the real world?

#### 4.1 Liquidity Constraints
- [ ] Position size < daily volume (typically <10% ADV)
- [ ] Can enter/exit within reasonable time
- [ ] Illiquid positions flagged
- [ ] Short-selling availability checked

#### 4.2 Timing
- [ ] Execution time specified (open, close, VWAP)
- [ ] Time zone differences accounted for
- [ ] Market hours respected
- [ ] Holiday calendars used

#### 4.3 Short Selling
- [ ] Borrow availability checked
- [ ] Borrow costs included (can be 0.3% to 50%+ annually)
- [ ] Recall risk considered
- [ ] Uptick rule (if applicable)

#### 4.4 Leverage
- [ ] Margin requirements checked
- [ ] Funding costs included
- [ ] Margin calls modeled
- [ ] Maximum leverage is realistic

---

### 5. OVERFITTING / DATA SNOOPING

**Definition:** Fitting parameters to noise rather than signal, resulting in strategies that don't work out-of-sample.

#### 5.1 Sample Design
- [ ] True out-of-sample period (never touched during development)
- [ ] Walk-forward validation used
- [ ] Training/validation/test split is temporal (not random)
- [ ] Test set is substantial (>20% of data)

#### 5.2 Parameter Selection
- [ ] Parameters have economic rationale
- [ ] Parameters are round numbers (not 247 days, but 252)
- [ ] Number of parameters is small relative to observations
- [ ] Similar parameters give similar results (robustness)

#### 5.3 Multiple Testing
- [ ] Number of strategies/parameters tested disclosed
- [ ] Deflated Sharpe Ratio applied
- [ ] Bonferroni or similar correction used
- [ ] Publication bias considered

#### 5.4 Red Flags for Overfitting
| Red Flag | Interpretation |
|----------|---------------|
| Parameters to 3+ decimals | Likely optimized to noise |
| Works only on specific subperiod | Not robust |
| >10 free parameters | Too many degrees of freedom |
| Strategy "discovered" by optimization | Not theory-driven |
| Works only in one market | May be spurious |

#### Multiple Testing Penalty:
```
Expected Spurious Sharpe ≈ √(2 × ln(N))
Where N = number of strategies tested

N=10  → Expected best Sharpe = 0.7 (under null)
N=100 → Expected best Sharpe = 1.0 (under null)
N=1000 → Expected best Sharpe = 1.2 (under null)
```

---

### 6. CORPORATE ACTIONS

**Definition:** Events that affect stock prices requiring adjustment.

#### 6.1 Splits and Dividends
- [ ] Stock splits adjusted correctly
- [ ] Dividend adjustments applied consistently
- [ ] Rights issues handled
- [ ] Spin-offs tracked

#### 6.2 Mergers and Acquisitions
- [ ] M&A dates correct
- [ ] Merger ratio applied properly
- [ ] Cash vs stock deals distinguished
- [ ] Delisting on correct date

#### 6.3 Data Vendor Issues
- [ ] Adjusted prices used consistently
- [ ] Adjustment method documented
- [ ] Point-in-time vs restated understood
- [ ] Multiple data sources cross-checked

---

### 7. CAPACITY AND SCALABILITY

**Definition:** How much capital can the strategy manage before returns degrade?

#### 7.1 Market Impact Analysis
- [ ] Capacity estimated explicitly
- [ ] Impact modeled as function of AUM
- [ ] Degradation curve estimated
- [ ] Realistic AUM assumption used

#### 7.2 Capacity Estimation
```
Simple Capacity Estimate:
Capacity = Avg Daily Turnover × Participation Rate × Days to Trade

Example:
- Universe average ADV: $10M per stock
- 30 stocks in portfolio
- Target 5% participation rate
- 5 days to trade

Capacity = $10M × 30 × 5% × 5 = $75M
```

#### 7.3 Crowding Risk
- [ ] Strategy uniqueness assessed
- [ ] Similar strategies in market identified
- [ ] Crowding indicators tracked
- [ ] Capacity shared among similar strategies

---

### 8. BENCHMARK AND RISK-ADJUSTMENT

**Definition:** Proper comparison and risk measurement.

#### 8.1 Benchmark Selection
- [ ] Benchmark is investable
- [ ] Benchmark has same survivorship treatment
- [ ] Appropriate for strategy style
- [ ] Currency-consistent

#### 8.2 Risk-Free Rate
- [ ] Correct rate for period
- [ ] Same compounding convention
- [ ] Currency-matched
- [ ] Term-appropriate

#### 8.3 Risk Metrics
- [ ] Volatility calculated correctly (annualized properly)
- [ ] Drawdowns include all peaks and troughs
- [ ] Beta calculated vs appropriate benchmark
- [ ] Risk metrics are out-of-sample

---

## Checklist Application Example

### Evaluating a Claimed Sharpe 1.5 Momentum Strategy:

1. **Look-Ahead Check:**
   - Does it use T+1 prices for execution? → If using same-day close, FAIL
   - Is the universe from today's data? → If yes, FAIL

2. **Survivorship Check:**
   - Are delisted stocks included? → If no, subtract ~1% from returns

3. **Cost Check:**
   - Transaction costs included? → If no, subtract (turnover × 30bps)
   - With 200% annual turnover: subtract 1.2% annually

4. **Overfitting Check:**
   - How many parameters? → More than 5 is concerning
   - Out-of-sample test done? → If no, divide Sharpe by ~1.5

**Adjusted Sharpe:**
```
Claimed: 1.5
After survivorship: 1.35 (subtract ~0.15)
After costs: 1.15 (subtract ~0.20)
After overfitting haircut: 0.75 (divide by ~1.5)
Realistic estimate: 0.75
```

---

## References

- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio
- Harvey, C. R., et al. (2016). ...and the Cross-Section of Expected Returns
- McLean, R. D., & Pontiff, J. (2016). Does Academic Research Destroy Stock Return Predictability?

---

*Use this checklist before allocating to any systematic strategy.*
