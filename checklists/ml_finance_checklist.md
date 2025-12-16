# Machine Learning in Finance Checklist

**A systematic checklist for ML models applied to financial prediction.**

ML in finance is notoriously prone to overfitting due to low signal-to-noise ratios. Use this checklist to validate any ML-based financial model.

---

## 1. DATA PREPARATION

### 1.1 Feature Engineering
- [ ] Are features calculated using ONLY past data at each point?
- [ ] Is there any information from the future in features?
- [ ] Are features normalized using ONLY training data statistics?
- [ ] Are missing values handled without using future information?

**Common Leakage Points:**
| Feature Type | Leakage Risk |
|--------------|--------------|
| Moving averages | Including current observation |
| Z-scores | Normalizing with full-sample mean/std |
| Forward returns | Accidentally using as feature |
| Technical indicators | Wrong window alignment |

### 1.2 Label Creation
- [ ] Is the prediction target clearly defined?
- [ ] Is there a gap between feature date and label date?
- [ ] Can the label actually be observed at prediction time?
- [ ] Are labels created without look-ahead?

**Example of label leakage:**
```python
# WRONG - leaks future return into same-day features
df['label'] = df['return'].shift(-1)  # Tomorrow's return
df['feature'] = df['return'].rolling(5).mean()  # Includes today

# CORRECT - predict T+2 return using T data
df['label'] = df['return'].shift(-2)  # Day after tomorrow
df['feature'] = df['return'].shift(1).rolling(5).mean()  # Through yesterday
```

### 1.3 Data Splitting
- [ ] Is the split TEMPORAL (not random)?
- [ ] Is there a gap between train and test periods?
- [ ] Is validation set also temporally separated?
- [ ] Is test set truly held out (never touched during development)?

**Proper split structure:**
```
|-------- Train --------|-- Gap --|--- Validation ---|-- Gap --|--- Test ---|
     2010-2017           1 month      2018-2019        1 month    2020-2023
```

---

## 2. LEAKAGE DETECTION

### 2.1 Feature-Label Correlation Check
- [ ] Are any features suspiciously highly correlated with labels?
- [ ] Does feature importance show unexpected variables?
- [ ] Do simple features outperform complex ones suspiciously?

**Red flag thresholds:**
| Correlation | Interpretation |
|-------------|----------------|
| > 0.30 | Highly suspicious - likely leakage |
| 0.15-0.30 | Investigate carefully |
| 0.05-0.15 | Possible signal, verify |
| < 0.05 | Typical for legitimate features |

### 2.2 Train vs Test Performance Gap
- [ ] Is the gap between train and test accuracy reasonable?
- [ ] Does performance degrade gracefully over time?
- [ ] Is cross-validation performance consistent?

**Acceptable gaps:**
| Model Complexity | Max Train-Test Gap |
|------------------|-------------------|
| Linear regression | 5% |
| Random Forest | 15% |
| Deep Learning | 20% |
| If larger → Overfitting likely |

### 2.3 Time Series Cross-Validation
- [ ] Is walk-forward validation used?
- [ ] Is there an embargo period to prevent leakage?
- [ ] Is purging applied for overlapping labels?

**Walk-forward structure:**
```
Fold 1: Train [2010-2015] → Gap → Test [2016]
Fold 2: Train [2010-2016] → Gap → Test [2017]
Fold 3: Train [2010-2017] → Gap → Test [2018]
...
```

---

## 3. MODEL VALIDATION

### 3.1 Baseline Comparison
- [ ] Is performance compared to a naive baseline?
- [ ] Does the model beat buy-and-hold?
- [ ] Does the model beat simple rules (momentum, value)?

**Minimum baselines:**
1. Random guessing (50% for binary)
2. Always predict majority class
3. Simple momentum/mean-reversion
4. Buy-and-hold benchmark

### 3.2 Economic Significance
- [ ] Are predictions actionable after costs?
- [ ] Is the information ratio meaningful?
- [ ] Does the strategy have realistic capacity?

**Reality check:**
```
If accuracy = 52% (above 50%)
   Average gain on correct = 0.5%
   Average loss on incorrect = 0.5%
   Gross expected return = 0.52×0.5% - 0.48×0.5% = 0.02%
   After costs (0.03%) = -0.01%  ← UNPROFITABLE
```

### 3.3 Stability Analysis
- [ ] Is performance stable across different time periods?
- [ ] Do predictions make sense in different market regimes?
- [ ] Is the model robust to small input perturbations?

---

## 4. FEATURE IMPORTANCE

### 4.1 Interpretability
- [ ] Can you explain why top features should predict returns?
- [ ] Is feature importance stable across time?
- [ ] Do important features have economic rationale?

### 4.2 Leakage Through Features
- [ ] Review the top 5 most important features
- [ ] Verify each doesn't contain future information
- [ ] Check if removing top feature destroys performance (suspicious)

**If removing one feature dramatically hurts performance:**
- That feature likely contains the "answer"
- Investigate for leakage
- If no leakage, strategy depends on single variable (risky)

---

## 5. HYPERPARAMETER SELECTION

### 5.1 Overfitting Through Tuning
- [ ] How many hyperparameter combinations were tested?
- [ ] Was tuning done on validation set (not test)?
- [ ] Is the search space documented?

**Multiple testing penalty:**
```
Expected Sharpe inflation = √(2 × ln(n_trials))

n_trials = 10   → inflation ≈ 0.27
n_trials = 100  → inflation ≈ 0.43
n_trials = 1000 → inflation ≈ 0.52
```

### 5.2 Regularization
- [ ] Is regularization used (L1, L2, dropout)?
- [ ] Are ensemble methods considered?
- [ ] Is early stopping implemented?

---

## 6. PRACTICAL DEPLOYMENT

### 6.1 Inference Pipeline
- [ ] Can features be calculated in real-time?
- [ ] Is the model stable for incremental updates?
- [ ] What is the retraining frequency?

### 6.2 Monitoring
- [ ] Are prediction distributions monitored?
- [ ] Is feature drift detected?
- [ ] Are confidence intervals tracked?

### 6.3 Fail-Safes
- [ ] What happens if the model fails?
- [ ] Are there position limits based on confidence?
- [ ] Is there a human override capability?

---

## 7. COMMON ML FINANCE MISTAKES

### 7.1 Label Leakage Patterns

| Mistake | Example |
|---------|---------|
| Same-day features | Using close price to predict close return |
| Overlapping labels | 5-day return with daily rebalancing |
| Target in features | "Market sentiment" = forward return |
| Corporate action leak | Using adjusted prices before adjustment |

### 7.2 Validation Mistakes

| Mistake | Problem |
|---------|---------|
| Random split | Future data in training set |
| No embargo | Autocorrelated features leak |
| Test set reuse | Not truly out-of-sample |
| Validation on full data | Hyperparameters fit to test |

### 7.3 Overfitting Indicators

| Symptom | Likely Cause |
|---------|--------------|
| Train acc 95%, Test acc 52% | Model memorized training data |
| Perfect in-sample, fails live | Backtest leakage |
| Performance degrades over time | Strategy is overfit to past |
| Single feature dominates | Potential leakage |

---

## 8. SANITY CHECK QUESTIONS

Before deploying any ML model in finance, ask:

1. **Why would this pattern exist?**
   - Is there economic theory supporting it?
   - Who is losing money for you to make money?

2. **Why hasn't this been arbitraged away?**
   - Is the capacity too small for institutions?
   - Are there barriers to implementation?

3. **What could go wrong?**
   - How does it perform in tail events?
   - What regime change would kill it?

4. **Would you bet your own money?**
   - With realistic position sizes?
   - With your full net worth?

---

## 9. QUICK VALIDATION SEQUENCE

```
1. Check feature-label correlation
   If any feature corr > 0.3 → STOP, investigate leakage

2. Run temporal train/test split
   If test accuracy >> 55% for daily prediction → suspicious

3. Compare train vs test performance
   If gap > 20% → likely overfitting

4. Calculate after-cost returns
   If unprofitable after costs → not viable

5. Test on different time period
   If doesn't work → likely overfit to specific period

6. Explain to colleague
   If can't explain why it works → probably doesn't
```

---

## References

- de Prado, M. L. (2018). *Advances in Financial Machine Learning*
- de Prado, M. L. (2020). *Machine Learning for Asset Managers*
- Bailey, D. H., & de Prado, M. L. (2014). "The Deflated Sharpe Ratio"
