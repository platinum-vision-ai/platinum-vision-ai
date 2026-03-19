import os
import time
import requests
import yfinance as yf
import streamlit as st
from openai import OpenAI
import pandas as pd
import plotly.graph_objects as go

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

/* 上のヘッダー消す */
header {visibility: hidden;}

/* フッター消す */
footer {visibility: hidden;}

/* Streamlitのメニュー系 */
[data-testid="stToolbar"] {display: none;}
[data-testid="stDecoration"] {display: none;}
[data-testid="stStatusWidget"] {display: none;}
[data-testid="stDeployButton"] {display: none;}

 /* 🔥 アンカーリンク（🔗）消す */
a[href^="#"] {
    display: none !important;
}
                       
/* 右下の「Hosted with Streamlit」完全消去 */
iframe {display: none !important;}

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


def get_ai_analysis(pt_price, usdjpy, pt_jpy_per_oz, cycle):
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


def load_pt_chart_1mo():
    try:
        pt_chart = yf.download(
            "PL=F",
            period="1mo",
            interval="1d",
            progress=False,
            auto_adjust=False
        )

        if pt_chart is None or pt_chart.empty:
            return None

        if isinstance(pt_chart.columns, pd.MultiIndex):
            pt_chart.columns = pt_chart.columns.get_level_values(0)

        if "Close" not in pt_chart.columns:
            return None

        pt_chart = pt_chart.dropna(subset=["Close"]).copy()
        return pt_chart

    except Exception:
        return None


def render_pt_chart(pt_chart: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pt_chart.index,
            y=pt_chart["Close"],
            mode="lines+markers",
            name="Platinum",
            line=dict(color="#00FF88", width=3),
            marker=dict(size=5),
            hovertemplate="日付: %{x|%Y-%m-%d}<br>価格: $%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode="x unified",
        showlegend=False,
        dragmode=False,
        xaxis=dict(
            title="",
            fixedrange=True,
            showgrid=False
        ),
        yaxis=dict(
            title="USD / oz",
            tickformat=".0f",
            fixedrange=True,
            range=[1500, 3000]
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False
        }
    )


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
# 🔥 回数制限（軽い制限）
if "count" not in st.session_state:
    st.session_state.count = 0

limit = 3

# 🔥 ボタン制御
analyze_button = st.button(
    "AI分析を開始",
    use_container_width=True,
    disabled=st.session_state.count >= limit
)

# 🔥 残り回数表示
st.caption(f"本日あと {limit - st.session_state.count} 回使えます")

# 🔥 制限表示
if st.session_state.count >= limit:
    st.warning(f"本日は{limit}回までです🙏")

if analyze_button:
    st.session_state.count += 1
    with st.spinner("市場データを取得しています..."):
        pt_price = get_pt_usd_per_oz()
        usdjpy = get_usdjpy_from_alphavantage()

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
            cycle=cycle
        )

    st.write(ai_result)

# ----------------------------
# プラチナチャート（1ヶ月固定）
# ----------------------------
st.markdown("---")
st.header("プラチナ価格チャート (USD/oz)")

pt_chart = load_pt_chart_1mo()

if pt_chart is None or pt_chart.empty:
    st.warning("チャートデータ取得失敗")
else:
    render_pt_chart(pt_chart)