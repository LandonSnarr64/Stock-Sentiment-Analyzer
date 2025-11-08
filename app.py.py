# app.py — NVDA predictor app
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

try:
    import xgboost as xgb
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False



# -------------------------
# Data + features
# -------------------------
@st.cache_data(show_spinner=False)
def download_prices(tickers, start="2015-01-01", end=None):
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)
    # robust "Adj Close" extraction no matter the column shape
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(-1):
            raw = data.xs('Adj Close', axis=1, level=-1).dropna()
        elif 'Adj Close' in data.columns.get_level_values(0):
            raw = data.xs('Adj Close', axis=1, level=0).dropna()
        else:
            raise KeyError("Could not find 'Adj Close'")
    else:
        if 'Adj Close' not in data.columns:
            raise KeyError("Could not find 'Adj Close'")
        raw = data[['Adj Close']].rename(columns={'Adj Close': tickers[0]}).dropna()
    return raw

def build_features(raw):
    """Return df with features + label y, and the list of feature names."""
    rets = raw.pct_change()

    nvda = pd.DataFrame(index=raw.index)
    nvda["price"] = raw["NVDA"]

    # momentum (approx 1m, 3m)
    nvda["ret1m"] = raw["NVDA"].pct_change(21)
    nvda["ret3m"] = raw["NVDA"].pct_change(63)

    # trend
    nvda["ma50"]  = raw["NVDA"].rolling(50).mean()
    nvda["ma200"] = raw["NVDA"].rolling(200).mean()
    nvda["trend"] = (nvda["ma50"] - nvda["ma200"]) / nvda["ma200"]

    # volatility
    nvda["vol20"] = rets["NVDA"].rolling(20).std()

    # relative strength
    if {"SPY", "SOXX", "AMD"}.issubset(raw.columns):
        nvda["rel_mkt_20"]  = rets["NVDA"].rolling(20).sum() - rets["SPY"].rolling(20).sum()
        nvda["rel_semi_20"] = rets["NVDA"].rolling(20).sum() - rets["SOXX"].rolling(20).sum()
        nvda["rel_amd_20"]  = rets["NVDA"].rolling(20).sum() - rets["AMD"].rolling(20).sum()
    else:
        nvda["rel_mkt_20"]  = 0.0
        nvda["rel_semi_20"] = 0.0
        nvda["rel_amd_20"]  = 0.0

    # label: will next 20 trading days be positive?
    future_ret20 = raw["NVDA"].pct_change(20).shift(-20)
    nvda["y"] = (future_ret20 > 0).astype(int)

    feat_cols = ["ret1m","ret3m","trend","vol20","rel_mkt_20","rel_semi_20","rel_amd_20"]
    df = nvda[feat_cols + ["y"]].dropna()
    return df, feat_cols

# -------------------------
# Modeling utils
# -------------------------
def make_model(model_name, random_state=42):
    if model_name == "XGBoost (recommended)" and XGB_OK:
        # small, sensible defaults for tabular data
        model = xgb.XGBClassifier(
            n_estimators=600,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            learning_rate=0.03,
            reg_lambda=1.0,
            random_state=random_state,
            tree_method="hist",
            eval_metric="logloss",
        )
    else:
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1
        )
    # probability calibration gives better thresholds
    return CalibratedClassifierCV(model, method="isotonic", cv=3)

def tune_threshold(y_true, p_train, metric="f1"):
    best_t, best_val = 0.5, -1
    for t in np.linspace(0.2, 0.8, 61):
        pred = (p_train >= t).astype(int)
        if metric == "f1":
            val = f1_score(y_true, pred)
        else:
            val = accuracy_score(y_true, pred)
        if val > best_val:
            best_val, best_t = val, t
    return best_t, best_val

def backtest_on_test(raw, test_dates, proba, threshold=0.5, hold_days=5, cost=0.0005):
    """Return equity curve Series for strategy vs buy/hold using only test period."""
    signal = pd.Series((proba >= threshold).astype(int), index=test_dates)

    daily_nvda_ret = raw["NVDA"].pct_change().reindex(test_dates).fillna(0.0)
    log_ret = np.log1p(daily_nvda_ret)
    hold_log = log_ret.rolling(hold_days).sum().shift(-(hold_days-1))
    hold_forward = np.expm1(hold_log).fillna(0.0)

    trades = signal.diff().clip(lower=0).fillna(0)
    strat_ret = signal * hold_forward - trades * cost

    eq = (1 + strat_ret).cumprod()
    bh = (1 + daily_nvda_ret).cumprod()
    return eq, bh

def explain_with_shap(model, X_row, feature_names):
    """Return a DataFrame of feature contributions for one row."""
    try:
        explainer = shap.TreeExplainer(model.base_estimator if isinstance(model, CalibratedClassifierCV) else model)
        sv = explainer.shap_values(X_row.reshape(1, -1))
        # shap returns list for binary classifiers in some versions
        if isinstance(sv, list):
            sv = sv[1]  # contribution to class 1 (Up)
        contrib = pd.Series(sv.flatten(), index=feature_names).sort_values(key=np.abs, ascending=False)
        return contrib
    except Exception:
        return None

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="NVDA Predictor", page_icon="📈", layout="wide")
st.title("📈 NVDA Predictor — signal, confidence, and explanations")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", ["XGBoost (recommended)" if XGB_OK else "RandomForest", "RandomForest"])
    hold_days = st.slider("Hold period (trading days)", 1, 20, 5, 1)
    train_frac = st.slider("Train fraction", 0.5, 0.95, 0.80, 0.01)
    auto_tune = st.checkbox("Auto-tune threshold on train (F1)", True)
    user_threshold = st.slider("Decision threshold", 0.20, 0.80, 0.50, 0.01)
    cost = st.number_input("Entry cost (one-way)", value=0.0005, step=0.0001, format="%.4f")
    st.caption("Cost 0.0005 ≈ 0.05% per entry")

    if st.button("🔄 Refresh data now"):
        download_prices.clear()  # clear cache

# 1) Data
tickers = ["NVDA","SPY","SOXX","AMD"]
raw = download_prices(tickers)
st.write("Latest data:", raw.tail())

# 2) Features
df, feat_cols = build_features(raw)
st.write("Feature sample:", df.tail())

# 3) Chronological split
n = len(df)
split = int(n * train_frac)
X_train, X_test = df[feat_cols].iloc[:split].values, df[feat_cols].iloc[split:].values
y_train, y_test = df["y"].iloc[:split].values, df["y"].iloc[split:].values
dates_test = df.index[split:]

# 4) Train
clf = make_model(model_name)
clf.fit(X_train, y_train)

# 5) Probabilities and threshold
p_train = clf.predict_proba(X_train)[:, 1]
p_test  = clf.predict_proba(X_test)[:, 1]

if auto_tune:
    best_t, best_val = tune_threshold(y_train, p_train, metric="f1")
    threshold = best_t
else:
    threshold = user_threshold

st.subheader("Performance on holdout")
y_pred = (p_test >= threshold).astype(int)
col1, col2 = st.columns([1,1])

with col1:
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    st.metric("Precision (Up)", f"{precision_score(y_test, y_pred):.3f}")
    st.metric("Recall (Up)", f"{recall_score(y_test, y_pred):.3f}")
    st.metric("F1 (Up)", f"{f1_score(y_test, y_pred):.3f}")
    st.write(f"Threshold used: **{threshold:.2f}** {'(auto)' if auto_tune else '(manual)'}")
with col2:
    fig, ax = plt.subplots(figsize=(5,4))
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Down/Flat (0)","Up (1)"])
    disp.plot(values_format='d', ax=ax)
    ax.set_title("Confusion Matrix (Holdout)")
    st.pyplot(fig)

st.text("Classification report:\n" + classification_report(y_test, y_pred, digits=3))

# 6) Backtest on test period
eq, bh = backtest_on_test(raw, dates_test, p_test, threshold=threshold, hold_days=hold_days, cost=cost)
st.subheader("Out-of-sample backtest (test period only)")
fig, ax = plt.subplots(figsize=(9,4))
eq.plot(ax=ax, label=f"Strategy (hold {hold_days}d, t={threshold:.2f})")
bh.plot(ax=ax, label="NVDA Buy&Hold")
ax.set_title("Equity curves")
ax.legend()
st.pyplot(fig)

# 7) "Predict now" — latest row
latest_X = df[feat_cols].iloc[[-1]].values
latest_date = df.index[-1]
latest_prob = clf.predict_proba(latest_X)[:,1][0]
latest_pred = int(latest_prob >= threshold)
st.subheader("Today's signal")
st.write(f"**As of {latest_date.date()}** → Prediction: **{'UP' if latest_pred else 'DOWN/FLAT'}**  |  Confidence: **{latest_prob:.2%}**  (threshold {threshold:.2f})")
st.write("Latest feature values:", df[feat_cols].tail(1).T.rename(columns={df.index[-1]:"value"}))

# 8) Explain the decision
st.subheader("Why? (feature contributions)")
contrib = None
if SHAP_OK:
    contrib = explain_with_shap(clf, latest_X[0], feat_cols)

if contrib is not None:
    top = contrib.head(10)[::-1]  # show top absolute contributions
    fig, ax = plt.subplots(figsize=(6,4))
    top.plot(kind="barh", ax=ax)
    ax.set_title("SHAP contributions toward 'Up'")
    ax.set_xlabel("Impact on model output")
    st.pyplot(fig)
else:
    # fallback: permutation importance on the test slice
    pi = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)
    imp = pd.Series(pi.importances_mean, index=feat_cols).sort_values()
    fig, ax = plt.subplots(figsize=(6,4))
    imp.plot(kind="barh", ax=ax)
    ax.set_title("Permutation importance (holdout)")
    st.pyplot(fig)

st.caption("Educational tool only. Not investment advice.")
