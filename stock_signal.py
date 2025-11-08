# stock_signal.py — General stock signal (business view)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, streamlit as st, yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.inspection import permutation_importance

# Optional models/explanations
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

# ======================== Small helpers ========================
def kpi(label, value, sub=""):
    st.markdown(
        f"""
        <div style="border:1px solid #e5e7eb;border-radius:12px;padding:12px 14px;margin:4px 0;">
          <div style="font-size:13px;color:#6b7280;">{label}</div>
          <div style="font-size:22px;font-weight:700;margin-top:2px;">{value}</div>
          <div style="font-size:12px;color:#6b7280;margin-top:2px;">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def conf_word(p):
    return ("Very High" if p>=0.80 else
            "High"      if p>=0.65 else
            "Moderate"  if p>=0.55 else
            "Balanced"  if p>=0.45 else
            "Cautious"  if p>=0.35 else
            "Low")

def pct(x, nd=1):
    try: return f"{100*x:.{nd}f}%"
    except: return "—"

# ======================== Data & features ========================
@st.cache_data(show_spinner=False)
def download_prices(tickers, start="2015-01-01", end=None):
    """Download a list of tickers with robust Adj Close extraction."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    if data.empty:
        raise RuntimeError("No data returned. Check ticker symbols.")
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(-1):
            raw = data.xs("Adj Close", axis=1, level=-1).dropna(how="all")
        elif "Adj Close" in data.columns.get_level_values(0):
            raw = data.xs("Adj Close", axis=1, level=0).dropna(how="all")
        else:
            raise KeyError("Adj Close not found in downloaded data.")
    else:
        if "Adj Close" not in data.columns:
            raise KeyError("Adj Close not found in downloaded data.")
        # single ticker only
        t0 = tickers[0]
        raw = data[["Adj Close"]].rename(columns={"Adj Close": t0}).dropna()
    # clean empty cols if a bad ticker
    raw = raw[[c for c in raw.columns if raw[c].notna().sum()>10]]
    if raw.empty:
        raise RuntimeError("All requested tickers were empty. Check symbols.")
    return raw

def build_features(raw, target):
    """Features for any target ticker using market/sector/peer context if provided."""
    rets = raw.pct_change()

    df = pd.DataFrame(index=raw.index)
    df["ret1m"] = raw[target].pct_change(21)
    df["ret3m"] = raw[target].pct_change(63)

    ma50  = raw[target].rolling(50).mean()
    ma200 = raw[target].rolling(200).mean()
    df["trend"] = (ma50 - ma200) / ma200

    df["vol20"] = rets[target].rolling(20).std()

    # Relative strength (only if exists)
    if "SPY" in raw.columns:
        df["rel_mkt_20"] = rets[target].rolling(20).sum() - rets["SPY"].rolling(20).sum()
    else:
        df["rel_mkt_20"] = 0.0
    if "SECTOR" in raw.columns:
        df["rel_semi_20"] = rets[target].rolling(20).sum() - rets["SECTOR"].rolling(20).sum()
    else:
        df["rel_semi_20"] = 0.0
    if "PEER" in raw.columns:
        df["rel_peer_20"] = rets[target].rolling(20).sum() - rets["PEER"].rolling(20).sum()
    else:
        df["rel_peer_20"] = 0.0

    # Label: will next 20 trading days be positive?
    future_ret20 = raw[target].pct_change(20).shift(-20)
    df["y"] = (future_ret20 > 0).astype(int)

    feat_cols = ["ret1m","ret3m","trend","vol20","rel_mkt_20","rel_semi_20","rel_peer_20"]
    df = df[feat_cols + ["y"]].dropna()
    return df, feat_cols

# ======================== Modeling ========================
def make_model(kind="XGB"):
    if kind=="XGB" and XGB_OK:
        base = xgb.XGBClassifier(
            n_estimators=600, max_depth=4, subsample=0.9, colsample_bytree=0.9,
            learning_rate=0.03, reg_lambda=1.0, tree_method="hist",
            random_state=42, eval_metric="logloss"
        )
    else:
        base = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=42)
    return CalibratedClassifierCV(base, method="isotonic", cv=3)

def tune_threshold(y_true, p_train):
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.2, 0.8, 61):
        f1 = f1_score(y_true, (p_train >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

def explain_with_shap(model, X_row, names):
    try:
        base = model.base_estimator if isinstance(model, CalibratedClassifierCV) else model
        explainer = shap.TreeExplainer(base)
        sv = explainer.shap_values(X_row.reshape(1,-1))
        if isinstance(sv, list): sv = sv[1]
        return pd.Series(sv.flatten(), index=names)
    except Exception:
        return None

def backtest_test(prices, dates_test, proba, threshold=0.5, hold_days=5, cost=0.0005, target="TGT"):
    signal = pd.Series((proba>=threshold).astype(int), index=dates_test)
    daily = prices[target].pct_change().reindex(dates_test).fillna(0.0)
    log_ret = np.log1p(daily)
    hold_log = log_ret.rolling(hold_days).sum().shift(-(hold_days-1))
    fwd = np.expm1(hold_log).fillna(0.0)
    trades = signal.diff().clip(lower=0).fillna(0)
    strat = signal*fwd - trades*cost
    return (1+strat).cumprod(), (1+daily).cumprod()

# ======================== App ========================
st.set_page_config(page_title="Stock Signal", page_icon="📈", layout="wide")
st.title("📈 Stock Signal (choose any ticker)")
st.caption("Business-style decision, confidence, rationale, and risks. Educational—Not investment advice.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Target ticker", value="NVDA").upper().strip()
    bench  = st.text_input("Benchmark (e.g., SPY)", value="SPY").upper().strip()
    sector = st.text_input("Sector ETF (optional, e.g., SOXX, XLK)", value="SOXX").upper().strip()
    peer   = st.text_input("Peer ticker (optional)", value="AMD").upper().strip()

    model_choice = st.selectbox("Model", ["XGBoost (recommended)" if XGB_OK else "RandomForest","RandomForest"])
    hold_days = st.slider("Hold period (days)", 1, 20, 5)
    train_frac = st.slider("Train fraction", 0.50, 0.95, 0.80, 0.01)
    cost = st.number_input("Entry cost (one-way)", value=0.0005, step=0.0001, format="%.4f")
    if st.button("🔄 Refresh prices"):
        download_prices.clear()

# Build ticker list with canonical column names
tickers = [ticker]
rename_map = {}
if bench:
    tickers.append(bench); rename_map[bench] = "SPY"
if sector:
    tickers.append(sector); rename_map[sector] = "SECTOR"
if peer:
    tickers.append(peer);   rename_map[peer]   = "PEER"

# 1) Data
try:
    raw_dl = download_prices(list(dict.fromkeys(tickers)))  # dedupe order
except Exception as e:
    st.error(f"Data download error: {e}")
    st.stop()

# rename optional cols to canonical names
raw = raw_dl.rename(columns=rename_map, errors="ignore")
if ticker not in raw.columns:
    st.error(f"Could not find data for target ticker '{ticker}'.")
    st.stop()

st.write("Latest prices:", raw.tail())

# 2) Features
df, feats = build_features(raw, target=ticker)
st.write("Feature sample:", df.tail())

# 3) Split chronologically
n = len(df); cut = int(n*train_frac)
if cut < 200 or (n - cut) < 100:
    st.warning("Very short history for training/testing. Consider a lower train fraction or older start date.")
Xtr, Xte = df[feats].iloc[:cut].values, df[feats].iloc[cut:].values
ytr, yte = df["y"].iloc[:cut].values, df["y"].iloc[cut:].values
dates_test = df.index[cut:]

# 4) Train + tune
clf = make_model("XGB" if "XGBoost" in model_choice else "RF")
clf.fit(Xtr, ytr)
ptr, pte = clf.predict_proba(Xtr)[:,1], clf.predict_proba(Xte)[:,1]
threshold = tune_threshold(ytr, ptr)

# 5) Eval on test (hidden by default)
yp = (pte>=threshold).astype(int)
acc, prec, rec, f1 = accuracy_score(yte, yp), precision_score(yte, yp), recall_score(yte, yp), f1_score(yte, yp)

# 6) Today's decision
X_now = df[feats].iloc[[-1]].values
p_now = clf.predict_proba(X_now)[:,1][0]
pred_now = int(p_now>=threshold)
latest_date = df.index[-1]
latest_row = df[feats].iloc[-1].to_dict()

# ---- Business header ----
col1, col2, col3 = st.columns([2,2,3])
with col1: kpi("Decision", "BUY ✅" if pred_now else "NO ACTION ⏸️", f"{ticker} — as of {latest_date.date()}")
with col2: kpi("Confidence", f"{p_now:.1%}", conf_word(p_now))
with col3: kpi("Decision threshold", f"{threshold:.2f}", "Auto-tuned on history")

# ---- Narrative why ----
st.markdown("### Why this decision")
contrib = explain_with_shap(clf, X_now[0], feats) if SHAP_OK else None
bullets = []
r1, r3, tr, v20 = latest_row.get("ret1m",0), latest_row.get("ret3m",0), latest_row.get("trend",0), latest_row.get("vol20",0)

if contrib is not None:
    top = contrib.abs().sort_values(ascending=False).head(6).index.tolist()
else:
    top = ["ret1m","ret3m","trend","vol20","rel_mkt_20","rel_semi_20","rel_peer_20"]

# Momentum
if "ret1m" in top or "ret3m" in top:
    if r1>0 and r3>0: bullets.append(f"Momentum positive: {pct(r1)} (1M) and {pct(r3)} (3M).")
    elif r3>0 and r1<=0: bullets.append(f"Medium-term momentum positive {pct(r3)} (3M) despite softer {pct(r1)} (1M).")
    else: bullets.append(f"Momentum mixed/negative: {pct(r1)} (1M), {pct(r3)} (3M).")
# Trend
if "trend" in top: bullets.append("Trend positive (above long-term average)." if tr>0 else "Trend negative (below long-term average).")
# Volatility
if "vol20" in top: bullets.append("Volatility contained (more reliable signals)." if v20<=0.03 else "Volatility elevated (choppier follow-through).")
# Relative strength context
if "rel_mkt_20" in top and "SPY" in raw.columns: bullets.append(f"Relative to SPY, {ticker}’s recent strength/weakness is a key driver.")
if "rel_semi_20" in top and "SECTOR" in raw.columns: bullets.append(f"Sector ETF ({sector}) relative strength is influencing the outlook.")
if "rel_peer_20" in top and "PEER" in raw.columns: bullets.append(f"Peer comparison ({peer}) matters here—pairwise moves can foreshadow follow-through.")

if not bullets: bullets = ["Drivers are balanced; no single dominant factor today."]
for b in bullets: st.write(f"- {b}")

# Risks
risks = []
if v20>0.04: risks.append("High volatility → whipsaw risk.")
if r1>0.25: risks.append("Large recent gains → mean-reversion risk.")
if tr<0: risks.append("Longer-term trend negative → rallies can fade.")
if risks:
    st.markdown("**Risks to watch:**")
    for r in risks: st.write(f"- {r}")

# Details (hidden)
with st.expander("Details • Contribution chart / factors"):
    if contrib is not None:
        show = contrib.abs().sort_values(ascending=False).head(8).index
        fig, ax = plt.subplots(figsize=(6,4))
        contrib[show].sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Feature contributions toward 'Up'")
        st.pyplot(fig)
    st.dataframe(pd.DataFrame([latest_row], index=[latest_date]).T.rename(columns={latest_date:"value"}))

with st.expander("Details • Backtest & metrics (unseen test window)"):
    st.caption("Metrics use the most recent window the model did **not** train on.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Accuracy", f"{acc:.3f}")
    with c2: kpi("Precision (Up)", f"{prec:.3f}")
    with c3: kpi("Recall (Up)", f"{rec:.3f}")
    with c4: kpi("F1 (Up)", f"{f1:.3f}", f"Hold {hold_days}d | Cost {cost:.2%}")
    eq, bh = backtest_test(raw.rename(columns={ticker:"TGT"}), dates_test, pte, threshold, hold_days, cost, target="TGT")
    fig, ax = plt.subplots(figsize=(9,4))
    eq.plot(ax=ax, label=f"Strategy (hold {hold_days}d)")
    bh.plot(ax=ax, label=f"{ticker} Buy&Hold")
    ax.legend(); ax.set_title("Equity curves (test window)")
    st.pyplot(fig)
    with st.expander("Confusion matrix & classification report"):
        fig2, ax2 = plt.subplots(figsize=(5,4))
        cm = confusion_matrix(yte, yp, labels=[0,1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Down/Flat (0)","Up (1)"])
        disp.plot(values_format='d', ax=ax2)
        st.pyplot(fig2)
        st.text("Classification report:\n"+classification_report(yte, yp, digits=3))
