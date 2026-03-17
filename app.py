import os
import time
import requests
import yfinance as yf
import streamlit as st
from openai import OpenAI
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import feedparser
# ----------------------------
# ページ設定
# ----------------------------
st.set_page_config(
    page_title="Platinum Vision AI",
    page_icon="🪙",
    layout="centered"
)
st.markdown("""
<style>
/* 上のメニュー（Forkとか）消す */
header {visibility: hidden;}

/* 下の「Hosted with Streamlit」消す */
footer {visibility: hidden;}

/* 右下のロゴ消す */
.stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)
# ----------------------------
# データ取得
# ----------------------------
def get_pt_usd_per_oz(max_retries: int = 3, sleep_sec: float = 1.0):
    for attempt in range(max_retries):
        try:
            df = yf.Ticker("PL=F").history(period="2d", interval="1h")

            if df is None or df.empty or df["Close"].dropna().empty:
                raise RuntimeError("yfinance returned empty data")

            return float(df["Close"].dropna().iloc[-1])

        except Exception:
            if attempt == max_retries - 1:
                return None
            time.sleep(sleep_sec)

    return None


def get_usdjpy_from_alphavantage(timeout_sec: int = 20):
    key = os.getenv("ALPHAVANTAGE_KEY")
    if not key:
        return None

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": "USD",
        "to_currency": "JPY",
        "apikey": key,
    }

    try:
        r = requests.get(url, params=params, timeout=timeout_sec)
        data = r.json()
        rate = data.get("Realtime Currency Exchange Rate", {}).get("5. Exchange Rate")
        return float(rate) if rate else None
    except Exception:
        return None
def get_market_news():

    urls = [
        "https://news.google.com/rss/search?q=gold+price",
        "https://news.google.com/rss/search?q=platinum+market",
        "https://news.google.com/rss/search?q=oil+price",
        "https://news.google.com/rss/search?q=fed+inflation"
    ]

    news_list = []

    for url in urls:

        feed = feedparser.parse(url)

        for entry in feed.entries[:2]:
            news_list.append(entry.title)

    return news_list

# ----------------------------
# ロジック
# ----------------------------
def judge_cycle(period: str, price_view: str, inflation: str) -> str:
    if period == "長期" and price_view == "安い" and inflation == "続く":
        return "静穏蓄積期"
    elif price_view == "高い":
        return "過熱警戒期"
    elif inflation == "落ち着く":
        return "冷却調整期"
    else:
        return "圧力上昇期"


def get_cycle_message(cycle: str) -> tuple[str, str]:
    if cycle == "静穏蓄積期":
        return (
            "金が先行して動いた後、資金が流入する可能性があります。",
            "まだ一般参加者が少ない段階かもしれません。"
        )
    elif cycle == "過熱警戒期":
        return (
            "短期的な過熱リスクに注意が必要です。",
            "急激な調整が起きる可能性もあります。"
        )
    elif cycle == "冷却調整期":
        return (
            "一時的な調整局面の可能性があります。",
            "次のサイクル準備段階かもしれません。"
        )
    else:
        return (
            "圧力が徐々に高まっている段階です。",
            "明確なトリガー待ちの状態かもしれません。"
        )


def get_ai_analysis(pt_price, usdjpy, pt_jpy_per_oz, cycle, news):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY が未設定のため、AI分析を表示できません。"

    try:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "あなたは冷静で簡潔なプロ投資分析AIです。"
                        "日本語で回答してください。"
                        "断定しすぎず、現在の数値から読める構造だけを述べてください。"
                        "過去時点の一般論や古い情報に基づく説明は禁止します。"
                        "見出しを使って、短くわかりやすく整理してください。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
以下は最新取得データです。

・Platinum (USD/oz): {pt_price:.2f}
・USDJPY: {usdjpy:.3f}
・Platinum (JPY/oz): {pt_jpy_per_oz:.2f}
・現在のサイクル判定: {cycle}

この数値のみを根拠に、現在の短期構造を分析してください。

出力ルール：
1. 「現在の構造」
2. 「注目点」
3. 「投資視点」
の3項目で整理してください。

過去の一般論や、2023年時点などの表現は禁止します。
""".strip(),
                },
            ],
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI分析の取得中にエラーが発生しました: {e}"


# ----------------------------
# UI
# ----------------------------
st.title("Platinum Vision AI")
st.caption("感情ではなく『構造』を見る、プラチナ分析AI")

st.markdown("### 入力項目")

period = st.selectbox(
    "投資期間",
    ["短期", "中期", "長期"]
)

price_view = st.selectbox(
    "現在の価格感覚",
    ["安い", "適正", "高い", "不明"]
)

inflation = st.selectbox(
    "インフレ見通し",
    ["続く", "落ち着く", "不明"]
)

analyze_button = st.button("AI分析を開始", use_container_width=True)

if analyze_button:
    with st.spinner("市場データを取得しています..."):
        pt_price = get_pt_usd_per_oz()
        usdjpy = get_usdjpy_from_alphavantage()
  # ニュース取得
        news = get_market_news()
        news_text = "\n".join(news)      
    if pt_price is None:
        st.error("プラチナ価格の取得に失敗しました。しばらくして再度お試しください。")
        st.stop()

    if usdjpy is None:
        st.error("USDJPYの取得に失敗しました。ALPHAVANTAGE_KEY を確認してください。")
        st.stop()

    pt_jpy_per_oz = pt_price * usdjpy
    pt_jpy_per_g = pt_jpy_per_oz / 31.1034768
    cycle = judge_cycle(period, price_view, inflation)
    msg1, msg2 = get_cycle_message(cycle)

    st.markdown("---")
    st.markdown("## 現在の市場データ")

    col1, col2 = st.columns(2)
    col1.metric("Platinum (USD/oz)", f"{pt_price:,.2f}")
    col2.metric("USDJPY", f"{usdjpy:,.3f}")

    col3, col4 = st.columns(2)
    col3.metric("Platinum (JPY/oz)", f"{pt_jpy_per_oz:,.2f}")
    col4.metric("Platinum (JPY/g)", f"{pt_jpy_per_g:,.2f}")

    st.markdown("---")
    st.markdown("## サイクル判定")
    st.success(f"現在のサイクル判定: {cycle}")
    st.write(msg1)
    st.write(msg2)


    st.markdown("---")
    st.markdown("## AI分析")

    with st.spinner("AIが構造を分析しています..."):
        ai_result = get_ai_analysis(
            pt_price=pt_price,
            usdjpy=usdjpy,
            pt_jpy_per_oz=pt_jpy_per_oz,
            cycle=cycle,
            news=news_text
        )

    st.write(ai_result)

# =========================
# プラチナチャート
# =========================

st.markdown("---")
st.header("プラチナ価格チャート (USD/oz)")


# -------------------------
# チャート期間選択
# -------------------------

period_option = st.selectbox(
    "チャート期間",
    ["1ヶ月", "3ヶ月", "6ヶ月", "1年"]
)

period_map = {
    "1ヶ月": "1mo",
    "3ヶ月": "3mo",
    "6ヶ月": "6mo",
    "1年": "1y"
}

selected_period = period_map[period_option]


# -------------------------
# データ取得
# -------------------------

pt_chart = yf.download(
    "PL=F",
    period=selected_period,
    interval="1d",
    progress=False
)

if pt_chart is None or pt_chart.empty:
    st.warning("チャートデータ取得失敗")

else:

    # MultiIndex対策
    if isinstance(pt_chart.columns, pd.MultiIndex):
        pt_chart.columns = pt_chart.columns.get_level_values(0)

    # -------------------------
    # チャート作成
    # -------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pt_chart.index,
            y=pt_chart["Close"],
            mode="lines",
            name="Platinum",
            line=dict(
                color="#00FF88",
                width=3
            )
        )
    )

    # -------------------------
    # レイアウト
    # -------------------------

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Date",
        yaxis_title="USD / oz",
        hovermode="x unified",
        yaxis=dict(
            tickmode="linear",
            dtick=100
        )
    )

    st.plotly_chart(fig, use_container_width=True)
# =========================
# 市場データ取得
# =========================

gold = yf.download("GC=F", period="1mo", interval="1d", progress=False)
silver = yf.download("SI=F", period="1mo", interval="1d", progress=False)
platinum = yf.download("PL=F", period="1mo", interval="1d", progress=False)
oil = yf.download("CL=F", period="1mo", interval="1d", progress=False)
sp500 = yf.download("^GSPC", period="1mo", interval="1d", progress=False)
dxy = yf.download("DX-Y.NYB", period="1mo", interval="1d", progress=False)

def safe_get_price(df, name):
    try:
        if df is not None and not df.empty and "Close" in df.columns:
            return float(df["Close"].dropna().iloc[-1])
        else:
            st.warning(f"{name}データ取得失敗")
            return None
    except:
        st.warning(f"{name}データエラー")
        return None


gold_price = safe_get_price(gold, "Gold")
silver_price = safe_get_price(silver, "Silver")
platinum_price = safe_get_price(platinum, "Platinum")
oil_price = safe_get_price(oil, "Oil")
sp500_price = safe_get_price(sp500, "SP500")
dxy_price = safe_get_price(dxy, "DXY")

# =========================
# 市場データ表示
# =========================

st.markdown("---")
st.header("市場データ")

col1, col2, col3 = st.columns(3)

col1.metric("Gold", f"${gold_price:.2f}")
col2.metric("Silver", f"${silver_price:.2f}")
col3.metric("Platinum", f"${platinum_price:.2f}")

col1, col2, col3 = st.columns(3)

col1.metric("Oil", f"${oil_price:.2f}")
col2.metric("S&P500", f"{sp500_price:.0f}")
col3.metric("Dollar Index", f"{dxy_price:.2f}")

# =========================
# レシオ計算
# =========================

gold_platinum_ratio = gold_price / platinum_price
gold_silver_ratio = gold_price / silver_price

st.markdown("---")
st.header("レシオ分析")

col1, col2 = st.columns(2)

col1.metric("Gold / Platinum", f"{gold_platinum_ratio:.2f}")
col2.metric("Gold / Silver", f"{gold_silver_ratio:.2f}")

# =========================
# レシオチャート
# =========================

st.markdown("---")
st.header("Gold / Platinum Ratio Chart")

ratio_df = pd.DataFrame()
ratio_df["Gold"] = gold["Close"]
ratio_df["Platinum"] = platinum["Close"]

ratio_df["Ratio"] = ratio_df["Gold"] / ratio_df["Platinum"]

fig_ratio = go.Figure()

fig_ratio.add_trace(
    go.Scatter(
        x=ratio_df.index,
        y=ratio_df["Ratio"],
        mode="lines",
        name="Gold / Platinum",
        line=dict(
            color="#FFD700",
            width=3
        )
    )
)

fig_ratio.update_layout(
    template="plotly_dark",
    height=500,
    xaxis_title="Date",
    yaxis_title="Ratio",
    hovermode="x unified"
)

st.plotly_chart(fig_ratio, use_container_width=True)