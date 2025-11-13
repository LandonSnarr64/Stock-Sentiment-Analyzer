Stock Sentiment & Technical Signal Research Model
Academic Abstract
This independent research project evaluates whether machine-learning models based on technical market indicators can support systematic stock‑signal generation. The model compares an individual equity against market, sector, and peer benchmarks using seven momentum, volatility, and trend indicators. Outputs include directional sentiment classifications, calibrated confidence levels, and back‑tested performance. Research emphasis is placed on methodology, interpretability, uncertainty, and academic grounding rather than real‑world trading use.

Motivation

Modern financial markets increasingly use automated decision systems. This project explores whether technical indicators, paired with machine‑learning techniques, can provide useful signal direction insights and probability estimates for equity‑performance forecasting.
Built as a self‑directed learning and research project — not for financial advice.

Research Question

Can a calibrated machine‑learning model using selected technical indicators generate stock‑direction sentiment signals and confidence scores that outperform individual indicator heuristics?

Methodology
Data Collection
Historical price data collected from: Target Stock, Market Benchmark (S&P 500), Sector Benchmark, Peer Company

Feature Engineering

Seven technical indicators capturing momentum, volatility, and trend: Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Moving Average windows, Bollinger Bands, Rate of Change (ROC), Volume‑based signals, Sector/peer relative performance factors

Model Architecture

Ensemble models (XGBoost or Random Forest),Automated decision‑threshold tuning, Hyperparameter exploration


Evaluation

Train/test split with unseen data;Metrics: Accuracy, Precision, Recall, F1;Back‑tested on familiar equities (e.g., NVDA, AMD, META, GOOG);Benchmark: S&P 500 index (SPY), sector ETF (XLK/SOXX)

Interpretability

SHAP values for feature contribution; Confidence probabilities; Narrative rationale


Deployment

Interactive Streamlit dashboard for results exploration.

Results
- ~70% accuracy in back‑testing, outperforming single‑indicator baselines
- Calibrated thresholds improved balance between false signals and missed opportunities
- Dashboard presents signal, rationale, and risk commentary

Exploratory research — not production trading.

Limitations

Limited data window; Simplified sentiment pipeline; No macroeconomic or fundamental inputs; Past performance not indicative of future results


Ethical & Practical Notes

Educational and experimental purposes only; No financial advice or trading recommendation; Focus on transparency and responsible AI experimentation


What I Learned

End‑to‑end ML research workflow; Back‑testing and model evaluation strategies; Importance of interpretability and uncertainty estimation; Streamlit deployment and reproducibility techniques



Future Work

Integrate OpenBB for expanded data sources and advanced indicators; Implement dynamic sentiment feeds and refined NLP methods; Add portfolio analytics and risk‑adjusted return evaluation; Expand back‑testing universe and extend time windows; Collaborate with open‑source communities for peer review; Apply insights from academic behavioral‑finance research


References

1. Cilingiroglu, E. (2023). Artificial Intelligence in the Stock Market: Quantitative Technical Analysis, Model Weight Optimization, and Financial Sentiment Evaluation to Predict Stock Prices. Intersect: The Stanford Journal of Science, Technology, and Society, Vol 17, No 1. Stanford Open Journal Systems. Published February 7, 2024. Produced from Inspirit AI program. 

2.Fedyk, A., Kakhbod, A., Li, P., & Malmendier, U. (2024). AI and Perception Biases in Investments: An Experimental Study. Haas School of Business, University of California Berkeley; National Bureau of Economic Research (NBER); Centre for Economic Policy Research (CEPR). 

3. Ghose, S. (2025). Dr. Jekyll and Mr. Hyde Revisited: Agentic AI, Chain-of-Thought & Emergent Misalignment. UC Berkeley Sutardja Center for Entrepreneurship & Technology (SCET), Innovation Collider. Published August 8, 
rYahoo Finance & OpenBB documentation



File Guide
- model.ipynb — Research, modeling, evaluation
- stock_signal.py — Streamlit dashboard
- requirements.txt — Dependencies
- README.md — Documentation



How to Run
pip install -r requirements.txt
streamlit run stock_signal.py


Acknowledgment
This project was independently designed and executed as part of a self‑directed research portfolio in machine‑learning and financial data science.
