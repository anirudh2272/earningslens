
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
import torch
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from textblob import TextBlob
from transformers import (BertTokenizer,
                           BertForSequenceClassification)
from curl_cffi import requests as curl_requests
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──
st.set_page_config(
    page_title  = "EarningsLens",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
.main { background-color: #0e1117; }
.prediction-card {
    background: linear-gradient(135deg, #1e2130, #252a3a);
    border-radius: 12px;
    padding: 24px;
    border: 1px solid #2d3250;
    margin: 10px 0;
}
.metric-card {
    background: #1e2130;
    border-radius: 8px;
    padding: 16px;
    border: 1px solid #2d3250;
    text-align: center;
}
.signal-tag-positive {
    background: rgba(0,200,100,0.15);
    border: 1px solid rgba(0,200,100,0.4);
    border-radius: 4px;
    padding: 4px 10px;
    color: #00c864;
    font-size: 13px;
    display: inline-block;
    margin: 3px;
}
.signal-tag-negative {
    background: rgba(255,80,80,0.15);
    border: 1px solid rgba(255,80,80,0.4);
    border-radius: 4px;
    padding: 4px 10px;
    color: #ff5050;
    font-size: 13px;
    display: inline-block;
    margin: 3px;
}
.signal-tag-neutral {
    background: rgba(150,150,150,0.15);
    border: 1px solid rgba(150,150,150,0.3);
    border-radius: 4px;
    padding: 4px 10px;
    color: #aaaaaa;
    font-size: 13px;
    display: inline-block;
    margin: 3px;
}
.disclaimer {
    background: rgba(255,200,0,0.08);
    border: 1px solid rgba(255,200,0,0.3);
    border-radius: 8px;
    padding: 12px;
    color: #ffc800;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──
BASE         = "."
NEWS_API_KEY = "22143ec738db477b8f4783502bdba4e6"

COMPANIES = {
    "AAPL":"Apple",     "MSFT":"Microsoft",
    "GOOGL":"Alphabet", "AMZN":"Amazon",
    "META":"Meta",      "NVDA":"NVIDIA",
    "AMD":"AMD",        "TSLA":"Tesla",
    "JPM":"JPMorgan",   "NFLX":"Netflix",
    "BAC":"BankAmerica","GS":"Goldman",
    "V":"Visa",         "MA":"Mastercard",
    "DIS":"Disney",     "SBUX":"Starbucks",
    "NKE":"Nike",       "JNJ":"Johnson",
    "PFE":"Pfizer",     "QCOM":"Qualcomm",
    "AVGO":"Broadcom",  "PYPL":"PayPal",
    "ADBE":"Adobe",     "SHOP":"Shopify",
    "UBER":"Uber",      "ABNB":"Airbnb",
    "CRM":"Salesforce", "ORCL":"Oracle",
    "INTC":"Intel",     "SQ":"Block",
}

GUIDANCE_RAISED  = [
    "raised guidance","increased outlook","raised forecast",
    "above expectations","beat estimates","record revenue",
    "record earnings","strong demand","raised annual",
]
GUIDANCE_LOWERED = [
    "lowered guidance","reduced outlook","below expectations",
    "missed estimates","challenging environment","headwinds",
    "uncertain demand","softness","cut guidance",
]
CONFIDENCE_HIGH  = [
    "confident","strong momentum","well positioned",
    "significant opportunity","robust pipeline",
]
UNCERTAINTY_HIGH = [
    "uncertain","challenging","difficult to predict",
    "cautious","monitor closely","volatile",
]

# ── Load models ──
@st.cache_resource
def load_models():
    with open(f"{BASE}/models/classifier_v3.pkl",   "rb") as f:
        clf = pickle.load(f)
    with open(f"{BASE}/models/label_encoder_v3.pkl","rb") as f:
        le  = pickle.load(f)
    with open(f"{BASE}/models/scaler_v3.pkl",       "rb") as f:
        sc  = pickle.load(f)
    with open(f"{BASE}/models/feature_cols_v3.pkl", "rb") as f:
        fc  = pickle.load(f)
    with open(f"{BASE}/models/regressor.pkl",       "rb") as f:
        reg = pickle.load(f)
    with open(f"{BASE}/models/quantile_models.pkl", "rb") as f:
        qm  = pickle.load(f)
    return clf, le, sc, fc, reg, qm

@st.cache_resource
def load_finbert():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    tok = BertTokenizer.from_pretrained("ProsusAI/finbert")
    mdl = BertForSequenceClassification.from_pretrained(
        "ProsusAI/finbert")
    mdl = mdl.to(device)
    mdl.eval()
    return tok, mdl, device

# ── Helper functions ──
def get_news(ticker, earnings_date, days_before=4):
    earn_dt   = pd.Timestamp(earnings_date)
    date_from = (earn_dt - timedelta(
        days=days_before)).strftime("%Y-%m-%d")
    date_to   = earn_dt.strftime("%Y-%m-%d")
    company   = COMPANIES.get(ticker, ticker)
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q"       : f"{company} earnings results",
                "from"    : date_from,
                "to"      : date_to,
                "language": "en",
                "sortBy"  : "relevancy",
                "pageSize": 10,
                "apiKey"  : NEWS_API_KEY
            },
            timeout=10
        )
        data = resp.json()
        if data.get("status") != "ok":
            return []
        return [
            f"{a.get(chr(116)+(chr(105))+(chr(116))+(chr(108))+(chr(101)),chr(32))}. "
            f"{a.get(chr(100)+(chr(101))+(chr(115))+(chr(99))+(chr(114))+(chr(105))+(chr(112))+(chr(116))+(chr(105))+(chr(111))+(chr(110)),chr(32))}".strip()
            for a in data.get("articles", [])
            if len(f"{a.get(chr(116)+(chr(105))+(chr(116))+(chr(108))+(chr(101)),chr(32))}") > 10
        ]
    except:
        return []

def get_yahoo_news(ticker, earnings_date):
    try:
        curl_sess = curl_requests.Session(impersonate="chrome")
        stock     = yf.Ticker(ticker, session=curl_sess)
        news      = stock.news or []
        earn_dt   = pd.Timestamp(earnings_date)
        articles  = []
        for a in news[:10]:
            title   = a.get("title",   "") or ""
            summary = a.get("summary", "") or ""
            pub_ts  = a.get("providerPublishTime", 0)
            if pub_ts:
                pub_dt    = pd.Timestamp(pub_ts, unit="s")
                days_diff = abs((pub_dt - earn_dt).days)
                if days_diff <= 5 and len(title) > 10:
                    articles.append(f"{title}. {summary}")
        return articles
    except:
        return []

def get_all_text(ticker, earnings_date):
    texts = get_news(ticker, earnings_date)
    if len(texts) < 3:
        texts += get_yahoo_news(ticker, earnings_date)
    return list(set(texts))

def finbert_score(texts, tokenizer, model, device):
    if not texts:
        return 0.0, 0.0, 0.0, 0.0
    pos_list, neg_list, neu_list = [], [], []
    for text in texts[:5]:
        try:
            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=512,
                padding=True)
            inputs = {k: v.to(device)
                      for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            p = torch.nn.functional.softmax(
                out.logits, dim=-1).cpu().numpy()[0]
            pos_list.append(float(p[0]))
            neg_list.append(float(p[1]))
            neu_list.append(float(p[2]))
        except:
            pass
    if not pos_list:
        return 0.0, 0.0, 0.0, 0.0
    avg_p = float(np.mean(pos_list))
    avg_n = float(np.mean(neg_list))
    avg_u = float(np.mean(neu_list))
    return avg_p, avg_n, avg_u, max(avg_p, avg_n, avg_u)

def get_technical_features(ticker, earnings_date):
    try:
        earn_dt   = pd.Timestamp(earnings_date)
        csv_path  = f"{BASE}/prices/{ticker}.csv"
        if os.path.exists(csv_path):
            prices = pd.read_csv(csv_path)
        else:
            curl_sess = curl_requests.Session(
                impersonate="chrome")
            stock = yf.Ticker(ticker, session=curl_sess)
            df    = stock.history(period="3mo", interval="1d")
            if df.empty:
                raise Exception("No data")
            df.reset_index(inplace=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df["Date"] = pd.to_datetime(df["Date"])
            prices = df
        prices["Date"] = pd.to_datetime(
            prices["Date"]).dt.tz_localize(None)
        pre    = prices[prices["Date"] < earn_dt].tail(15)
        if len(pre) < 5:
            return None
        close  = pre["Close"].values
        volume = pre["Volume"].values
        returns= np.diff(close) / close[:-1]
        mom_5d = float((close[-1]-close[-5])/close[-5]*100)                  if len(close) >= 5 else 0.0
        mom_10d= float((close[-1]-close[0])/close[0]*100)
        vol    = float(np.std(returns)*100)
        avg_v  = float(np.mean(volume))
        rec_v  = float(np.mean(volume[-3:]))
        vsurge = rec_v/avg_v if avg_v > 0 else 1.0
        gains  = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]
        ag     = np.mean(gains)  if gains  else 0.001
        al     = np.mean(losses) if losses else 0.001
        rsi    = 100 - (100/(1 + ag/al))
        all_p  = prices[prices["Date"] < earn_dt].tail(252)
        hi52   = float(all_p["Close"].max())                  if len(all_p) > 0 else close[-1]
        d52hi  = (close[-1]-hi52)/hi52*100
        return {
            "momentum_5d"  : round(mom_5d,  4),
            "momentum_10d" : round(mom_10d, 4),
            "volatility"   : round(vol,     4),
            "volume_surge" : round(vsurge,  4),
            "rsi"          : round(rsi,     2),
            "dist_52w_high": round(d52hi,   4),
        }
    except:
        return None

def build_features(ticker, earnings_date,
                   tokenizer, model, device,
                   feature_cols):
    texts      = get_all_text(ticker, earnings_date)
    text_all   = " ".join(texts)
    text_lower = text_all.lower()

    fp, fn, fnu, fc = finbert_score(
        texts, tokenizer, model, device)

    guid_r = sum(1 for p in GUIDANCE_RAISED
                 if p in text_lower)
    guid_l = sum(1 for p in GUIDANCE_LOWERED
                 if p in text_lower)
    conf_h = sum(1 for p in CONFIDENCE_HIGH
                 if p in text_lower)
    unc_h  = sum(1 for p in UNCERTAINTY_HIGH
                 if p in text_lower)
    guid_d = (1.0 if guid_r > guid_l else
              -1.0 if guid_l > guid_r else 0.0)

    import re
    sents  = [s.strip() for s in
              re.split(r"[.!?]", text_all)
              if len(s.strip()) > 20]
    pols   = [TextBlob(s).sentiment.polarity
              for s in sents[:50]]
    pos_s  = sum(1 for p in pols if p >  0.05)
    neg_s  = sum(1 for p in pols if p < -0.05)
    n_s    = max(len(pols), 1)
    blob   = TextBlob(text_all[:3000])
    tb_pol = float(blob.sentiment.polarity)
    avg_ss = float(np.mean(pols)) if pols else 0.0
    pos_sr = pos_s / n_s
    neg_sr = neg_s / n_s
    fwd_w  = ["expect","anticipate","outlook","guidance",
               "forecast","project","next quarter"]
    fwd_r  = sum(text_lower.count(w)
                 for w in fwd_w) / max(len(
                     text_lower.split()), 1)

    tech = get_technical_features(ticker, earnings_date)
    if tech is None:
        tech = {
            "momentum_5d"  : 0.0, "momentum_10d": 0.0,
            "volatility"   : 0.0, "volume_surge": 1.0,
            "rsi"          : 50.0,"dist_52w_high": 0.0,
        }

    SECTORS = {
        "AAPL":0,"MSFT":0,"GOOGL":0,"AMZN":0,
        "META":0,"CRM":0,"ADBE":0,"SHOP":0,
        "ORCL":0,"INTC":0,
        "NVDA":1,"AMD":1,"QCOM":1,"AVGO":1,
        "JPM":2,"BAC":2,"GS":2,"V":2,"MA":2,
        "TSLA":3,"NFLX":3,"DIS":3,"SBUX":3,
        "NKE":3,"UBER":3,"ABNB":3,
        "JNJ":4,"PFE":4,
        "PYPL":5,"SQ":5,
    }

    raw = {
        "finbert_positive"       : fp,
        "finbert_negative"       : fn,
        "finbert_neutral"        : fnu,
        "finbert_confidence"     : fc,
        "guidance_raised_count"  : guid_r,
        "guidance_lowered_count" : guid_l,
        "confidence_score"       : conf_h,
        "uncertainty_score"      : unc_h,
        "guidance_direction"     : guid_d,
        "avg_sentence_sentiment" : avg_ss,
        "sentiment_volatility"   : float(
            np.std(pols)) if pols else 0.0,
        "positive_sentence_ratio": pos_sr,
        "negative_sentence_ratio": neg_sr,
        "forward_looking_ratio"  : fwd_r,
        "tb_polarity"            : tb_pol,
        "tb_subjectivity"        : float(
            blob.sentiment.subjectivity),
        "unique_positive_signals": guid_r + conf_h,
        "unique_negative_signals": guid_l + unc_h,
        "momentum_5d"            : tech["momentum_5d"],
        "momentum_10d"           : tech["momentum_10d"],
        "volatility"             : tech["volatility"],
        "volume_surge"           : tech["volume_surge"],
        "rsi"                    : tech["rsi"],
        "dist_52w_high"          : tech["dist_52w_high"],
        "eps_surprise_pct"       : 0.0,
        "beat_miss"              : 0,
        "sector_code"            : SECTORS.get(ticker, 0),
    }

    row = pd.DataFrame([raw])
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0.0
    return row[feature_cols], raw, texts

# ── Main App ──
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔍 EarningsLens")
        st.markdown(
            "Earnings move intelligence using "
            "FinBERT NLP + ML"
        )
        st.markdown("---")
        st.markdown("### Supported Tickers")
        tickers_list = list(COMPANIES.keys())
        half         = len(tickers_list) // 2
        col1, col2   = st.columns(2)
        with col1:
            for t in tickers_list[:half]:
                st.markdown(f"• {t}")
        with col2:
            for t in tickers_list[half:]:
                st.markdown(f"• {t}")
        st.markdown("---")
        st.markdown(
            '<div class="disclaimer">'
            "⚠️ For research purposes only.<br>"
            "Not financial advice."
            "</div>",
            unsafe_allow_html=True
        )

    # Header
    st.markdown(
        "# 🔍 EarningsLens"
    )
    st.markdown(
        "**Earnings Move Intelligence** — "
        "FinBERT NLP + Ensemble ML + "
        "Quantile Regression"
    )
    st.markdown("---")

    # Input form
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        ticker = st.selectbox(
            "Select Ticker",
            options=list(COMPANIES.keys()),
            format_func=lambda x:
                f"{x} — {COMPANIES[x]}"
        )
    with col2:
        earnings_date = st.date_input(
            "Earnings Date",
            value=datetime(2026, 1, 29)
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button(
            "🔍 Analyze", use_container_width=True)

    if analyze:
        earnings_date_str = str(earnings_date)

        with st.spinner(
            "Loading FinBERT and fetching data..."
        ):
            try:
                clf, le, sc, fc, reg, qm = load_models()
                tok, mdl, dev = load_finbert()
            except Exception as e:
                st.error(f"Error loading models: {e}")
                return

        with st.spinner(
            "Fetching news and computing features..."
        ):
            X_row, raw_feats, articles = build_features(
                ticker, earnings_date_str,
                tok, mdl, dev, fc
            )

        with st.spinner("Running predictions..."):
            # Direction
            dir_proba  = clf.predict_proba(X_row)[0]
            dir_idx    = np.argmax(dir_proba)
            dir_pred   = le.inverse_transform([dir_idx])[0]
            dir_conf   = round(float(np.max(dir_proba))*100,1)

            # Magnitude
            X_sc       = pd.DataFrame(
                sc.transform(X_row), columns=fc)
            mag_pred   = float(reg.predict(X_sc)[0])
            q10        = float(
                qm["q10"].predict(X_sc)[0])
            q50        = float(
                qm["q50"].predict(X_sc)[0])
            q90        = float(
                qm["q90"].predict(X_sc)[0])

        # ── Results ──
        st.markdown("---")
        st.markdown(
            f"## {ticker} — "
            f"{COMPANIES.get(ticker,ticker)} "
            f"| {earnings_date_str}"
        )

        # Top metrics
        m1, m2, m3, m4 = st.columns(4)

        dir_color = (
            "🟢" if dir_pred == "UP" else "🔴")
        with m1:
            st.metric(
                "Direction",
                f"{dir_color} {dir_pred}",
                f"{dir_conf}% confidence"
            )
        with m2:
            st.metric(
                "Predicted Move",
                f"{mag_pred:+.2f}%",
                "point estimate"
            )
        with m3:
            st.metric(
                "Range (80%)",
                f"{q10:+.1f}% to {q90:+.1f}%",
                f"Width: {q90-q10:.1f}%"
            )
        with m4:
            st.metric(
                "Articles Found",
                len(articles),
                "news sources"
            )

        # Confidence interval chart
        st.markdown("### 📊 Predicted Move Distribution")
        fig = go.Figure()

        x_range = np.linspace(
            min(q10-2, -15), max(q90+2, 15), 200)
        mid      = (q10 + q90) / 2
        spread   = max((q90 - q10) / 3, 0.5)
        y_gauss  = np.exp(
            -0.5 * ((x_range - mid) / spread) ** 2)
        y_gauss  = y_gauss / y_gauss.max()

        fig.add_trace(go.Scatter(
            x=x_range, y=y_gauss,
            fill="tozeroy",
            fillcolor="rgba(30,100,200,0.15)",
            line=dict(color="rgba(30,100,200,0.0)"),
            name="Probability distribution"
        ))
        fig.add_vline(
            x=q10, line_dash="dash",
            line_color="#ff6b6b",
            annotation_text=f"Q10: {q10:+.1f}%"
        )
        fig.add_vline(
            x=q90, line_dash="dash",
            line_color="#ff6b6b",
            annotation_text=f"Q90: {q90:+.1f}%"
        )
        fig.add_vline(
            x=mag_pred, line_dash="solid",
            line_color="#00c864", line_width=2,
            annotation_text=f"Predicted: {mag_pred:+.1f}%"
        )
        fig.add_vline(
            x=0, line_dash="dot",
            line_color="rgba(255,255,255,0.3)"
        )
        fig.update_layout(
            plot_bgcolor  = "rgba(0,0,0,0)",
            paper_bgcolor = "rgba(0,0,0,0)",
            font_color    = "white",
            showlegend    = False,
            height        = 250,
            margin        = dict(l=20,r=20,t=30,b=20),
            xaxis_title   = "Expected Move (%)",
            yaxis_visible = False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Signals and features
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### 🎯 Key Signals")
            signals = []

            if raw_feats["guidance_direction"] == 1.0:
                signals.append((
                    "Guidance raised language", "positive"))
            elif raw_feats["guidance_direction"] == -1.0:
                signals.append((
                    "Guidance lowered language", "negative"))

            if raw_feats["finbert_positive"] > 0.6:
                signals.append((
                    f"Strong FinBERT positive "
                    f"({raw_feats['finbert_positive']:.2f})",
                    "positive"
                ))
            elif raw_feats["finbert_negative"] > 0.5:
                signals.append((
                    f"Strong FinBERT negative "
                    f"({raw_feats['finbert_negative']:.2f})",
                    "negative"
                ))

            rsi_val = raw_feats.get("rsi", 50)
            if rsi_val > 70:
                signals.append((
                    f"Overbought RSI ({rsi_val:.0f})"
                    f" — sell-the-news risk", "negative"))
            elif rsi_val < 30:
                signals.append((
                    f"Oversold RSI ({rsi_val:.0f})"
                    f" — relief rally possible", "positive"))

            mom = raw_feats.get("momentum_5d", 0)
            if mom > 5:
                signals.append((
                    f"Strong pre-earnings momentum "
                    f"(+{mom:.1f}%)", "positive"))
            elif mom < -5:
                signals.append((
                    f"Weak pre-earnings momentum "
                    f"({mom:.1f}%)", "negative"))

            if raw_feats["unique_negative_signals"] >= 3:
                signals.append((
                    f"Multiple bearish language signals "
                    f"({raw_feats['unique_negative_signals']})",
                    "negative"
                ))
            if raw_feats["unique_positive_signals"] >= 3:
                signals.append((
                    f"Multiple bullish language signals "
                    f"({raw_feats['unique_positive_signals']})",
                    "positive"
                ))

            if not signals:
                signals.append((
                    "No strong signals — low conviction",
                    "neutral"
                ))

            signal_html = ""
            for sig, typ in signals:
                css_class = f"signal-tag-{typ}"
                icon      = ("✅" if typ=="positive" else
                             "❌" if typ=="negative" else "⚪")
                signal_html += (
                    f"<div class='{css_class}'>"
                    f"{icon} {sig}</div>"
                )
            st.markdown(signal_html, unsafe_allow_html=True)

        with col_b:
            st.markdown("### 📈 Feature Values")
            feat_data = {
                "Feature"  : [
                    "FinBERT Positive",
                    "FinBERT Negative",
                    "FinBERT Confidence",
                    "Guidance Direction",
                    "Momentum 5d",
                    "RSI",
                    "Volatility",
                    "Dist 52w High",
                ],
                "Value": [
                    f"{raw_feats['finbert_positive']:.3f}",
                    f"{raw_feats['finbert_negative']:.3f}",
                    f"{raw_feats['finbert_confidence']:.3f}",
                    ("+1 (Raised)" if
                     raw_feats["guidance_direction"]==1 else
                     "-1 (Lowered)" if
                     raw_feats["guidance_direction"]==-1
                     else "0 (Neutral)"),
                    f"{raw_feats['momentum_5d']:+.2f}%",
                    f"{raw_feats['rsi']:.1f}",
                    f"{raw_feats['volatility']:.2f}%",
                    f"{raw_feats['dist_52w_high']:+.1f}%",
                ]
            }
            st.dataframe(
                pd.DataFrame(feat_data),
                hide_index=True,
                use_container_width=True
            )

        # News preview
        if articles:
            st.markdown("### 📰 News Articles Used")
            for i, article in enumerate(articles[:5], 1):
                clean = article[:200].replace(
                    "<", "&lt;").replace(">", "&gt;")
                st.markdown(
                    f"**{i}.** {clean}..."
                    if len(article) > 200
                    else f"**{i}.** {clean}"
                )

        # Disclaimer
        st.markdown("---")
        st.markdown(
            '<div class="disclaimer">'
            "⚠️ <strong>Disclaimer:</strong> "
            "EarningsLens is a research tool only. "
            "Predictions are based on historical patterns "
            "and public information. "
            "This is NOT financial advice. "
            "Do not make investment decisions based on "
            "this tool."
            "</div>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
