# 🔍 EarningsLens

**Earnings Move Intelligence** — Predicts post-earnings stock price direction and magnitude using FinBERT NLP + Ensemble ML + Quantile Regression.

## What It Does

- Fetches pre-earnings news articles automatically
- Runs **FinBERT** (finance-specific BERT) on headlines to extract sentiment
- Computes **25+ features** including guidance language, RSI, momentum, EPS surprise
- Predicts **direction** (UP/DOWN) with confidence %
- Predicts **magnitude** — the actual % move, not just direction
- Gives a **confidence interval** (80% range) so you know how uncertain the prediction is

## Tech Stack

- **NLP**: FinBERT (ProsusAI/finbert) + TextBlob
- **ML**: XGBoost + LightGBM + Random Forest + Gradient Boosting (Voting Ensemble)
- **Regression**: XGBoost Regressor + Quantile Regression (Q10/Q50/Q90)
- **Data**: 232 earnings events across 30 S&P 500 companies (2023–2026)
- **Tracking**: MLflow experiment tracking
- **Frontend**: Streamlit + Plotly

## Supported Tickers

AAPL, MSFT, GOOGL, AMZN, META, NVDA, AMD, TSLA, JPM, NFLX, BAC, GS, V, MA, DIS, SBUX, NKE, JNJ, PFE, QCOM, AVGO, PYPL, ADBE, SHOP, UBER, ABNB, CRM, ORCL, INTC, SQ

## Dataset

- **232** real earnings events
- **30** S&P 500 companies across 6 sectors
- **3 years** of historical data (2023–2026)
- Features: FinBERT sentiment + EPS surprise + technical indicators

## Disclaimer

⚠️ For research purposes only. Not financial advice. Do not make investment decisions based on this tool.

## Author

Anirudh Koganti — M.S. Computer Science, Missouri S&T  
Research Engineer, PHLAI Lab
