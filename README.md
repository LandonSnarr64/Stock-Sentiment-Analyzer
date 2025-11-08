
# Stock Sentiment Analyzer

Explores how machine learning can integrate stock price data and sentiment indicators to study short-term market trends.  
Built as an independent learning project — not for trading or financial advice.

---

##  Overview
This model compares the sentiment and performance of a target stock against the overall market, its sector, and a close competitor.  
It uses **seven technical indicators** of high applicability and effectiveness to capture momentum, trend, and volatility patterns.  
The model produces a **Buy / Hold / No-Action decision** with a calibrated **confidence percentage**, an **auto-tuned decision threshold**, and a **recommended holding period**.  
Results include a narrative summary explaining methodology, drivers, and limitations.

---

## Purpose
To test whether combining quantitative market data with sentiment-based features can modestly improve predictive power compared to purely technical models.

---

##  Method
- **Libraries:** `pandas`, `numpy`, `yfinance`, `scikit-learn`, `XGBoost`, `SHAP`, `matplotlib`, `streamlit`
- **Data Sources:** Yahoo Finance (e.g., NVDA, SPY, SOXX, AMD) and curated sentiment scores from financial headlines.
- **Process:**
  1. Collects historical price data for the target, benchmark, sector, and peer tickers.
  2. Generates seven features capturing momentum, volatility, and relative strength.
  3. Trains a calibrated ensemble model (XGBoost or Random Forest) with automated threshold tuning.
  4. Evaluates performance on unseen data using accuracy, precision, recall, and F1 metrics.
  5. Presents interpretable results with confidence scores, SHAP feature explanations, and back-tested curves.

---

##  Results
- Sentiment-augmented models showed **small but consistent improvement** over purely technical baselines.
- The calibrated threshold improved decision balance between false positives and missed opportunities.
- Outputs include an interactive Streamlit dashboard displaying confidence, rationale, and risk commentary.

*(Exploratory only — not production-ready.)*

---

##  Limitations
- Short dataset window; limited sentiment sample.
- Simplified NLP approach (basic polarity, not transformer-based).
- No macroeconomic, fundamental, or news-timing adjustments.

---

## What I Learned
- How to structure an end-to-end ML pipeline from raw data to deployed web apps.  
- The importance of model calibration, interpretability, and reproducibility.  
- How to evaluate predictions honestly and quantify uncertainty.  

## Ongoing Work and Improvement
- Integrate OpenBB data pipelines to access a broader range of financial datasets and expand historical coverage for both target and peer tickers.
-Implement advanced technical indicators available through OpenBB to enhance model robustness and feature diversity.
- Incorporate dynamic sentiment and news feeds using OpenBB’s aggregation tools to provide richer, real-time contextual inputs.
-Develop portfolio and risk analytics modules for evaluating risk-adjusted returns, optimal holding periods, and diversification effects.
- Enhance visualization and interpretability through OpenBB’s plotting and reporting utilities to improve user experience and transparency.
-Collaborate with the OpenBB open-source research community to adopt best practices in calibration, ethical AI design, and continuous model improvement.
-Draw on academic research from UC Berkeley’s Haas School of Business and the Department of Economics, such as “AI and Perception Biases in Investments: An Experimental Study,” to inform improvement of my model’s interpretability and bias reduction. Their insights on how AI agents emulate diverse investor behaviors will guide how sentiment and demographic features are applied to enhance fairness and reliability in automated investment reasoning.


---

## 🗂 Files
- `stock_signal.py` — Streamlit application (interactive dashboard)
- `model.ipynb` — Jupyter notebook (training and evaluation)
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation
