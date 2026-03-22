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

/* アンカーリンク（🔗）消す */
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


def get_ticker_snapshot(ticker: str, label: str):
    """
    直近終値と前日比(%)を取得
    """
    try:
        df = yf.download(
            ticker,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=False
        )

        if df is None or df.empty:
            return {
                "label": label,
                "ticker": ticker,
                "price": None,
                "change_pct": None
            }

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Close" not in df.columns:
            return {
                "label": label,
                "ticker": ticker,
                "price": None,
                "change_pct": None
            }

        close_series = df["Close"].dropna()
        if close_series.empty:
            return {
                "label": label,
                "ticker": ticker,
                "price": None,
                "change_pct": None
            }

        latest = float(close_series.iloc[-1])

        if len(close_series) >= 2:
            prev = float(close_series.iloc[-2])
            change_pct = ((latest - prev) / prev) * 100 if prev != 0 else None
        else:
            change_pct = None

        return {
            "label": label,
            "ticker": ticker,
            "price": latest,
            "change_pct": round(change_pct, 2) if change_pct is not None else None
        }

    except Exception:
        return {
            "label": label,
            "ticker": ticker,
            "price": None,
            "change_pct": None
        }


def get_macro_market_context():
    """
    AI分析の厚みを出すための補助データ
    """
    context = {
        "gold": get_ticker_snapshot("GC=F", "Gold"),
        "silver": get_ticker_snapshot("SI=F", "Silver"),
        "crude": get_ticker_snapshot("CL=F", "Crude Oil"),
        "sp500": get_ticker_snapshot("^GSPC", "S&P 500"),
        "vix": get_ticker_snapshot("^VIX", "VIX")
    }
    return context


def get_news_headlines(max_items_per_ticker: int = 3):
    """
    yfinanceのニュースを複数銘柄から集めて重複を除去
    """
    tickers = ["PL=F", "GC=F", "CL=F", "^GSPC"]
    headlines = []
    seen = set()

    for ticker in tickers:
        try:
            items = yf.Ticker(ticker).news or []
            for item in items[:max_items_per_ticker]:
                title = item.get("title")
                if title and title not in seen:
                    seen.add(title)
                    headlines.append(title)
        except Exception:
            continue

    return headlines[:8]


def build_news_summary_text(headlines):
    if not headlines:
        return "関連ニュース見出しは取得できませんでした。"

    return "\n".join([f"- {h}" for h in headlines])


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


def format_context_value(item: dict, price_digits: int = 2):
    price = item.get("price")
    change_pct = item.get("change_pct")

    if price is None and change_pct is None:
        return "取得失敗"

    price_text = f"{price:,.{price_digits}f}" if price is not None else "不明"
    change_text = f"{change_pct:+.2f}%" if change_pct is not None else "不明"

    return f"{price_text} / 前日比 {change_text}"


def get_ai_analysis(
    pt_price,
    usdjpy,
    pt_jpy_per_oz,
    cycle,
    period,
    price_view,
    inflation,
    news_text,
    macro_context
):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY が未設定のため、AI分析を表示できません。"

    gold_text = format_context_value(macro_context["gold"])
    silver_text = format_context_value(macro_context["silver"])
    crude_text = format_context_value(macro_context["crude"])
    sp500_text = format_context_value(macro_context["sp500"])
    vix_text = format_context_value(macro_context["vix"])

    try:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "あなたは冷静で実務的なマクロ投資アナリストです。"
                        "必ず日本語で回答してください。"
                        "ニュース見出しと現在の市場データから、相場の構造を読み解いてください。"
                        "地政学リスク、株安、原油、為替、ボラティリティ、貴金属の関係を整理してください。"
                        "『地政学リスクだから必ず貴金属が上がる』のような単純化は禁止です。"
                        "安全資産買いと、流動性確保のための売りが同時に起こりうることを踏まえてください。"
                        "分からないことは分からないと書いてください。"
                        "一般論だけで終わらず、今回のデータとニュースに結びつけて分析してください。"
                        "見出しを使って、簡潔だが中身のある文章にしてください。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
以下は最新取得データです。

【ユーザー入力】
・投資期間: {period}
・現在の価格感覚: {price_view}
・インフレ見通し: {inflation}
・現在のサイクル判定: {cycle}

【プラチナ現在値】
・Platinum (USD/oz): {pt_price:.2f}
・USDJPY: {usdjpy:.3f}
・Platinum (JPY/oz): {pt_jpy_per_oz:.2f}

【周辺市場】
・Gold: {gold_text}
・Silver: {silver_text}
・Crude Oil: {crude_text}
・S&P 500: {sp500_text}
・VIX: {vix_text}

【関連ニュース見出し】
{news_text}

以下のルールで分析してください。

1. 「現在の構造」
- プラチナの位置づけを説明
- 株式、為替、原油、ボラティリティ、他の貴金属との関係を整理
- 地政学リスクがあってもプラチナが下がるケースなら、その理由も説明

2. 「ニュースの影響」
- 今回のニュースがプラチナにどう効いているか
- イラン・中東情勢、株安、リスクオフ、流動性確保売り、インフレ懸念などの可能性を切り分ける
- 見出しから読める範囲だけを使う

3. 「注目点」
- 次に見るべき市場指標や条件を2〜4個に絞る

4. 「投資視点」
- 短期・中期のスタンスを簡潔に
- 断定ではなく、条件付きで述べる

5. 「総合コメント」
- 最後に2〜4文で、今回の相場の核心をまとめる

注意:
- 古い一般論は禁止
- 必ず今回のニュースと今回の数値に結びつける
- 情報が足りない時は足りないと明記する
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

# 回数制限（軽い制限）
if "count" not in st.session_state:
    st.session_state.count = 0

limit = 3

analyze_button = st.button(
    "AI分析を開始",
    use_container_width=True,
    disabled=st.session_state.count >= limit
)

st.caption(f"本日あと {limit - st.session_state.count} 回使えます")

if st.session_state.count >= limit:
    st.warning(f"本日は{limit}回までです🙏")

if analyze_button:
    st.session_state.count += 1

    with st.spinner("市場データとニュースを取得しています..."):
        pt_price = get_pt_usd_per_oz()
        usdjpy = get_usdjpy_from_alphavantage()
        headlines = get_news_headlines()
        news_text = build_news_summary_text(headlines)
        macro_context = get_macro_market_context()

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
    if headlines:
        st.markdown("## 取得ニュース")
        for h in headlines:
            st.markdown(f"- {h}")

    st.markdown("---")
    st.markdown("## AI分析")

    with st.spinner("AIがニュースと市場構造を分析しています..."):
        ai_result = get_ai_analysis(
            pt_price=pt_price,
            usdjpy=usdjpy,
            pt_jpy_per_oz=pt_jpy_per_oz,
            cycle=cycle,
            period=period,
            price_view=price_view,
            inflation=inflation,
            news_text=news_text,
            macro_context=macro_context
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