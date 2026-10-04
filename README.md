# Nifty 50 Autonomous Quant Predictor

An end-to-end quantitative trading application that forecasts next-day opening price levels and directional gaps for Nifty 50 (`^NSEI`). Built with FastAPI, Scikit-Learn (Calibrated Gradient Boosting), and Tailwind CSS.

## 🕒 Trading Workflow & Schedule

1. **09:30 AM IST (Automated Ground-Truth Verification):**
   - Ingests realized Nifty 50 open price at market open.
   - Compares predicted vs actual gap direction.
   - Automatically triggers adaptive model retraining on updated data.

2. **11:30 AM IST (Automated Forecast Signal):**
   - Evaluates mid-session momentum, India VIX, Put-Call Ratio (PCR), and global benchmark cues.
   - Generates tomorrow's predicted open price, directional probabilities, and 1-sigma volatility bounds.

3. **12:00 PM IST (Execution Window):**
   - Trader reviews 11:30 AM signal probability ($P > 55\%$) on dashboard to place trades.

---

## 🛠️ Step-by-Step Git & Local Setup

### 1. Initialize Repository
```bash
git init nifty-gap-predictor
cd nifty-gap-predictor