# Prompts for LLM Backtest Diagnosis

**Prompts designed to get an LLM to identify issues in backtests.**

These prompts are structured to elicit specific, actionable feedback when reviewing backtest code or results.

---

## 1. General Backtest Review Prompt

```markdown
You are a senior quantitative researcher reviewing a backtest for data integrity, 
methodology, and execution realism. Please analyze the following backtest and 
identify any issues.

For each issue found:
1. Classify it (Look-Ahead Bias, Survivorship Bias, Overfitting, Execution, Other)
2. Identify the specific line(s) of code or assumption
3. Explain the impact on results (estimated Sharpe inflation, return bias)
4. Suggest a fix

[PASTE BACKTEST CODE HERE]

Focus especially on:
- How dates are handled (is future data ever accessible?)
- How the tradeable universe is defined
- How fundamental data availability is modeled
- Transaction costs and execution assumptions
- Train/test methodology
```

---

## 2. Look-Ahead Bias Detection Prompt

```markdown
Review this backtest code specifically for look-ahead (look-forward) bias.

Look-ahead bias occurs when information that wasn't available at the decision 
time is used to make that decision. Common forms include:

1. Using same-day close prices for trading decisions
2. Using earnings/fundamentals before their release dates
3. Market cap calculated from future prices
4. Moving averages that include the current observation
5. Forward-filling data from future observations
6. Using point-in-time data that has been restated

For each potential issue:
- Quote the relevant code
- Explain why it's look-ahead bias
- Estimate the performance impact
- Show the correct implementation

[PASTE CODE HERE]

Be especially vigilant about date indexing and `.loc` operations.
```

---

## 3. Survivorship Bias Detection Prompt

```markdown
Analyze this backtest for survivorship bias.

Survivorship bias inflates returns by excluding securities that:
- Went bankrupt
- Were delisted
- Were acquired at distressed prices
- Dropped out of indices

Check for:
1. Is the universe point-in-time or current constituents?
2. Are delisted stocks included in the price data?
3. How are stocks that "disappear" handled?
4. Does the benchmark have the same bias?

For each issue:
- Explain the specific bias
- Estimate annual return inflation (typically 1-3%)
- Describe how to construct a survivorship-free dataset

[PASTE CODE OR DESCRIBE METHODOLOGY]
```

---

## 4. Overfitting/Data Snooping Prompt

```markdown
Review this strategy for signs of overfitting and data snooping.

Overfitting indicators include:
1. Suspiciously specific parameters (247 days, not 252)
2. Weights to many decimal places (0.314, not 0.30)  
3. Large number of tested parameter combinations
4. In-sample Sharpe >> out-of-sample Sharpe
5. No economic rationale for parameter choices
6. Cherry-picked time periods

For this backtest:
1. List all parameters that appear optimized
2. Estimate how many combinations might have been tested
3. Calculate the expected Sharpe inflation from multiple testing
4. Suggest parameters with economic justification

[PASTE CODE OR STRATEGY DESCRIPTION]

Use the formula: Expected max Sharpe under null ≈ √(2 × ln(n_trials))
```

---

## 5. ML Leakage Detection Prompt

```markdown
Review this ML-based financial model for data leakage.

Leakage in ML finance typically occurs through:

1. FEATURE LEAKAGE
   - Features calculated using future data
   - Same-day information included
   - Target variable appears in features (perhaps transformed)

2. TEMPORAL LEAKAGE
   - Random train/test split instead of temporal
   - No embargo between train and test
   - Validation set overlaps with test

3. PREPROCESSING LEAKAGE
   - Scaler fit on all data (including test)
   - Missing value imputation using future data
   - Feature selection using full dataset

For each potential leak:
- Identify the specific code/step
- Explain the mechanism of leakage
- Estimate accuracy/performance inflation
- Show correct implementation

[PASTE CODE HERE]

Pay special attention to:
- How StandardScaler is applied
- train_test_split parameters
- Feature engineering date handling
```

---

## 6. Transaction Cost Reality Check Prompt

```markdown
Evaluate whether this backtest has realistic transaction costs and execution 
assumptions.

Check for:
1. Are transaction costs included? What rate?
2. Is bid-ask spread modeled?
3. Is market impact estimated?
4. Is slippage included?
5. Are borrow costs included for shorts?
6. What prices are used for execution (close, next open, VWAP)?

Provide:
1. Assessment of current cost assumptions
2. Suggested realistic cost model for this strategy
3. Estimated impact on returns if costs are understated
4. Break-even analysis: what costs would make this unprofitable?

[PASTE BACKTEST CODE/RESULTS]

Reference costs:
- Large-cap equity: 5-15 bps each way
- Small-cap: 15-50 bps
- EM: 30-100 bps
```

---

## 7. Sharpe Ratio Reality Check Prompt

```markdown
The following backtest reports a Sharpe ratio of [X]. Evaluate whether this 
is plausible.

Context needed:
1. What is the strategy type? (factor, stat arb, momentum, etc.)
2. What is the turnover?
3. What is the universe size?
4. How were parameters selected?
5. What is the sample period length?

Provide:
1. Expected Sharpe range for this strategy type
2. Potential sources of Sharpe inflation
3. "Haircut" estimate for likely biases
4. Adjusted Sharpe expectation

[PASTE STRATEGY DESCRIPTION AND REPORTED METRICS]

Reference Sharpe expectations:
- Simple factors (value, momentum): 0.2-0.5
- Combined multi-factor: 0.4-0.7
- Stat arb (high turnover): 0.5-1.0
- If claimed > 1.0: extraordinary proof required
```

---

## 8. Corporate Actions Prompt

```markdown
Review how this backtest handles corporate actions.

Corporate actions that must be handled:
1. Stock splits and reverse splits
2. Cash dividends
3. Spin-offs
4. Mergers and acquisitions  
5. Rights offerings
6. Name/ticker changes

For each:
- Does the backtest handle it?
- Is the handling correct?
- What's the impact of incorrect handling?

[PASTE CODE OR DESCRIBE DATA SOURCE]

Common mistakes:
- Using unadjusted prices
- Dividends not reinvested (or reinvested at wrong time)
- Spin-offs treated as zero return
- Merged companies disappear without return
```

---

## 9. Capacity and Scalability Prompt

```markdown
Estimate the realistic capacity of this strategy.

Analyze:
1. What are the typical position sizes?
2. What is the average daily volume of positions?
3. What % of ADV would this strategy require?
4. How would market impact scale with AUM?

Provide:
1. Estimated capacity (AUM) at current return expectations
2. At what AUM does market impact reduce Sharpe by 50%?
3. What modifications would increase capacity?

[PASTE STRATEGY DETAILS]

Rules of thumb:
- Position should be < 1% of ADV for minimal impact
- 5% of ADV: significant impact
- 10% of ADV: strategy is capacity constrained
```

---

## 10. Full Diagnostic Prompt

```markdown
Perform a comprehensive review of this backtest. Structure your analysis as:

## EXECUTIVE SUMMARY
- Overall assessment (Trust / Cautious / Do Not Trust)
- Estimated true Sharpe after adjustments
- Critical issues found

## DATA INTEGRITY
- Survivorship bias: [FOUND/NOT FOUND] - details
- Look-ahead bias: [FOUND/NOT FOUND] - details
- Data quality issues: [FOUND/NOT FOUND] - details

## METHODOLOGY
- Train/test split: [PROPER/IMPROPER] - details
- Parameter selection: [JUSTIFIED/SUSPICIOUS] - details
- Multiple testing: [ACCOUNTED/NOT ACCOUNTED] - details

## EXECUTION REALISM
- Transaction costs: [REALISTIC/UNDERSTATED] - details
- Execution assumptions: [REALISTIC/OPTIMISTIC] - details
- Capacity: [ADEQUATE/CONSTRAINED] - details

## STATISTICAL VALIDITY
- Sample size: [ADEQUATE/INSUFFICIENT] - details
- Significance: [SIGNIFICANT/NOT SIGNIFICANT] - details
- Robustness: [DEMONSTRATED/NOT TESTED] - details

## ADJUSTMENTS
| Issue | Estimated Sharpe Impact |
|-------|------------------------|
| [Issue 1] | -X.XX |
| [Issue 2] | -X.XX |
| **Total Haircut** | **-X.XX** |

## RECOMMENDATIONS
1. [First priority fix]
2. [Second priority fix]
3. [Additional improvements]

[PASTE FULL BACKTEST CODE AND RESULTS]
```

---

## Usage Tips

1. **Be specific about what you want checked**
   - The more focused the prompt, the better the diagnosis

2. **Provide context**
   - Include the full code, not just snippets
   - Include reported results for sanity checking

3. **Ask for code fixes**
   - Request corrected code, not just problem identification

4. **Verify the verification**
   - LLMs can miss subtle issues
   - Use these prompts as a first pass, not final word

5. **Iterate**
   - Run multiple prompts on the same backtest
   - Cross-check findings
