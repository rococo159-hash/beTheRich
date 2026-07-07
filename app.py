import os
import time
import tempfile
import streamlit as st
import yfinance as yf
import google.generativeai as genai
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# yfinance 캐시 경로 설정 (Streamlit Cloud rate-limit 완화)
# ============================================================
try:
    _cache_dir = os.path.join(tempfile.gettempdir(), "py-yfinance")
    os.makedirs(_cache_dir, exist_ok=True)
    yf.set_tz_cache_location(_cache_dir)
except Exception:
    pass

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(page_title="관심종목분석", page_icon="📊", layout="wide")

# ============================================================
# 국내/미국 종목명 → 티커 매핑
# ============================================================
KR_NAME_TO_TICKER = {
    "삼성전자": "005930.KS", "SK하이닉스": "000660.KS", "LG에너지솔루션": "373220.KS",
    "삼성바이오로직스": "207940.KS", "현대차": "005380.KS", "기아": "000270.KS",
    "셀트리온": "068270.KS", "POSCO홀딩스": "005490.KS", "NAVER": "035420.KS",
    "카카오": "035720.KS", "삼성SDI": "006400.KS", "LG화학": "051910.KS",
    "현대모비스": "012330.KS", "삼성물산": "028260.KS", "KB금융": "105560.KS",
    "신한지주": "055550.KS", "하나금융지주": "086790.KS", "삼성생명": "032830.KS",
    "LG전자": "066570.KS", "SK이노베이션": "096770.KS", "한국전력": "015760.KS",
    "포스코퓨처엠": "003670.KS", "KT&G": "033780.KS", "고려아연": "010130.KS",
    "HMM": "011200.KS", "두산에너빌리티": "034020.KS", "삼성전기": "009150.KS",
    "SK텔레콤": "017670.KS", "기업은행": "024110.KS", "우리금융지주": "316140.KS",
    "S-Oil": "010950.KS", "대한항공": "003490.KS", "한화에어로스페이스": "012450.KS",
    "현대제철": "004020.KS", "LG": "003550.KS", "메리츠금융지주": "138040.KS",
    "삼성화재": "000810.KS", "한미반도체": "042700.KS", "두산": "000150.KS",
    "크래프톤": "259960.KS", "엔씨소프트": "036570.KS", "넷마블": "251270.KS",
    "아모레퍼시픽": "090430.KS", "한온시스템": "018880.KS", "현대글로비스": "086280.KS",
    "CJ제일제당": "097950.KS", "오리온": "271560.KS", "한국조선해양": "009540.KS",
    "에코프로비엠": "247540.KQ", "에코프로": "086520.KQ", "알테오젠": "196170.KQ",
    "엔켐": "348370.KQ", "HLB": "028300.KQ", "리노공업": "058470.KQ",
    "셀트리온제약": "068760.KQ", "JYP Ent.": "035900.KQ", "펄어비스": "263750.KQ",
    "위메이드": "112040.KQ", "카카오게임즈": "293490.KQ", "클래시스": "214150.KQ",
    "레인보우로보틱스": "277810.KQ", "루닛": "328130.KQ", "에스엠": "041510.KQ",
    "셀바스AI": "108860.KQ", "이오테크닉스": "039030.KQ", "솔브레인": "357780.KQ",
    "테스나": "131970.KQ", "동진쎄미켐": "005290.KQ", "주성엔지니어링": "036930.KQ",
}

US_NAME_TO_TICKER = {
    "NVIDIA": "NVDA", "Apple": "AAPL", "Microsoft": "MSFT", "Amazon": "AMZN",
    "Alphabet (Google)": "GOOGL", "Meta (Facebook)": "META", "Tesla": "TSLA",
    "Broadcom": "AVGO", "Netflix": "NFLX", "AMD": "AMD", "Intel": "INTC",
    "Palantir": "PLTR", "Tempus AI": "TEM", "Eli Lilly": "LLY", "JP Morgan": "JPM",
    "Visa": "V", "Mastercard": "MA", "Coca-Cola": "KO", "Walmart": "WMT",
    "Berkshire Hathaway": "BRK-B", "Costco": "COST", "Qualcomm": "QCOM",
    "Micron": "MU", "ASML": "ASML", "TSMC": "TSM", "Super Micro": "SMCI",
    "Arm Holdings": "ARM", "Uber": "UBER", "Disney": "DIS", "Boeing": "BA",
}

# ============================================================
# 티커 검색 헬퍼 (야후 심볼 검색 메인 + Gemini 보조)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def yahoo_symbol_search(query, region_hint):
    """야후 파이낸스 심볼 검색. 후보 리스트(dict) 반환."""
    results = []
    try:
        from yfinance import Search
        res = Search(query, max_results=10)
        quotes = getattr(res, "quotes", []) or []
        for q in quotes:
            sym = q.get("symbol", "")
            name = q.get("shortname") or q.get("longname") or ""
            exch = q.get("exchange", "")
            qtype = q.get("quoteType", "")
            if not sym or qtype not in ("EQUITY", "ETF"):
                continue
            if region_hint == "KR" and not (sym.endswith(".KS") or sym.endswith(".KQ")):
                continue
            if region_hint == "US" and ("." in sym):
                continue
            results.append({"symbol": sym, "name": name, "exchange": exch})
    except Exception:
        pass
    return results


def gemini_ticker_lookup(query, region_hint, api_key):
    """야후가 못 찾을 때 Gemini에게 티커를 물어보는 fallback."""
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        suffix_hint = (
            "한국 주식이면 코스피는 .KS, 코스닥은 .KQ 접미사를 붙여줘."
            if region_hint == "KR" else
            "미국 주식이면 접미사 없이 티커만."
        )
        prompt = (
            f"'{query}'라는 회사의 야후 파이낸스(Yahoo Finance) 티커 심볼만 알려줘. "
            f"{suffix_hint} 설명 없이 티커 심볼 하나만 정확히 출력해. "
            f"확실하지 않으면 'UNKNOWN'이라고만 답해."
        )
        for model_name in ["gemini-2.5-flash", "gemini-2.0-flash"]:
            try:
                try:
                    model = genai.GenerativeModel(model_name, tools=[{"google_search": {}}])
                except Exception:
                    model = genai.GenerativeModel(model_name)
                resp = model.generate_content(prompt)
                text = (resp.text or "").strip().upper().replace("`", "")
                text = text.split()[0] if text else ""
                if text and text != "UNKNOWN" and len(text) <= 12:
                    return text
            except Exception:
                continue
    except Exception:
        pass
    return None


# ============================================================
# 사이드바 — Gemini API Key
# ============================================================
st.sidebar.title("🔐 시스템 보안 통제")
secret_key = st.secrets.get("GEMINI_API_KEY", "")
if secret_key:
    api_key = secret_key
    st.sidebar.success("🔒 Secrets 금고에서 API Key를 자동으로 불러왔습니다.")
    with st.sidebar.expander("🔑 다른 API Key로 대체하기 (선택)", expanded=False):
        user_override = st.text_input("대체 API Key 입력:", type="password")
        if user_override:
            api_key = user_override
else:
    with st.sidebar.expander("🔑 구글 Gemini API Key 입력 (수동)", expanded=True):
        api_key = st.text_input("API Key:", type="password")
    st.sidebar.caption("※ 클라우드 Secrets 설정 전이므로 수동 입력이 필요합니다.")

# ============================================================
# 메인 헤더
# ============================================================
st.title("📊 관심종목분석")
st.write("시장 컨센서스, 기술적 수급, 실시간 성장 촉매를 융합하여 균형 잡힌 투자 결정을 지원합니다.")
st.write("---")

# ============================================================
# 입력부 — 시장 선택
# ============================================================
market_choice = st.radio("📊 분석 대상 시장 선택", ["미국 주식 (US)", "국내 주식 (KR)"], horizontal=True)
region_hint = "US" if market_choice == "미국 주식 (US)" else "KR"

# ---- 세션 상태 초기화 ----
if "resolved_ticker" not in st.session_state:
    st.session_state.resolved_ticker = None

# ============================================================
# 🔎 기업 이름으로 티커 찾기 (야후 검색 + Gemini 보조)
# ============================================================
with st.expander("🔎 기업 이름으로 티커 찾기 (이름만 알아도 OK)", expanded=False):
    st.caption("회사 이름을 입력하면 야후에서 티커 후보를 찾아드려요. 안 나오면 AI가 한 번 더 찾아봐요.")
    sc1, sc2 = st.columns([3, 1])
    with sc1:
        search_query = st.text_input(
            "회사 이름 입력",
            placeholder="예: 삼성전자 / Samsung / Apple / 애플",
            key="ticker_search_query",
        )
    with sc2:
        st.write("")
        st.write("")
        do_search = st.button("🔍 티커 검색", use_container_width=True)

    if do_search and search_query.strip():
        with st.spinner("티커를 찾는 중..."):
            candidates = yahoo_symbol_search(search_query.strip(), region_hint)

        if candidates:
            st.success(f"✅ {len(candidates)}개 후보를 찾았어요. 아래에서 골라주세요.")
            for i, c in enumerate(candidates):
                bcol1, bcol2 = st.columns([4, 1])
                with bcol1:
                    st.markdown(f"**{c['symbol']}** — {c['name']}  `{c['exchange']}`")
                with bcol2:
                    if st.button("이걸로", key=f"pick_{i}"):
                        st.session_state.resolved_ticker = c["symbol"]
                        st.rerun()
        else:
            # 야후 실패 → Gemini fallback
            with st.spinner("야후에서 못 찾아서 AI에게 물어보는 중..."):
                g_ticker = gemini_ticker_lookup(search_query.strip(), region_hint, api_key)
            if g_ticker:
                st.info(f"🤖 AI가 찾은 티커: **{g_ticker}**  (정확한지 아래 회사명으로 꼭 확인하세요)")
                if st.button(f"'{g_ticker}' 로 분석하기"):
                    st.session_state.resolved_ticker = g_ticker
                    st.rerun()
            else:
                st.warning("티커를 찾지 못했어요. 철자를 바꿔보거나 아래에서 직접 입력해주세요.")

    if st.session_state.resolved_ticker:
        st.markdown(f"👉 현재 선택된 티커: **`{st.session_state.resolved_ticker}`**")
        if st.button("❌ 선택 해제"):
            st.session_state.resolved_ticker = None
            st.rerun()

# ============================================================
# 입력부 — 종목명 드롭다운 / 코드 직접입력
# ============================================================
in_col1, in_col2 = st.columns(2)

if market_choice == "미국 주식 (US)":
    name_map = US_NAME_TO_TICKER
    with in_col1:
        chosen_name = st.selectbox("🔍 회사 이름으로 검색 (자동완성)", ["(직접 입력)"] + list(name_map.keys()))
    with in_col2:
        typed = st.text_input("⌨️ 또는 티커 직접 입력", "NVDA" if chosen_name == "(직접 입력)" else "")
    default_ticker = (name_map.get(chosen_name) if chosen_name != "(직접 입력)" else typed.upper().strip()) or "NVDA"
    currency_symbol, cap_unit, cap_divider, small_cap_threshold = "$", "B", 1_000_000_000, 10.0
else:
    name_map = KR_NAME_TO_TICKER
    with in_col1:
        chosen_name = st.selectbox("🔍 회사 이름으로 검색 (자동완성)", ["(직접 입력)"] + list(name_map.keys()))
    with in_col2:
        typed = st.text_input("⌨️ 또는 종목코드 6자리 직접 입력", "" if chosen_name != "(직접 입력)" else "005930")
    if chosen_name != "(직접 입력)":
        default_ticker = name_map[chosen_name]
    else:
        raw_digits = "".join(ch for ch in typed.strip() if ch.isdigit())
        default_ticker = f"{raw_digits.zfill(6)}.KS" if raw_digits else "005930.KS"
    currency_symbol, cap_unit, cap_divider, small_cap_threshold = "₩", "조 원", 1_000_000_000_000, 1.5

# 티커 검색으로 고른 값이 있으면 그것을 우선 사용
if st.session_state.resolved_ticker:
    ticker = st.session_state.resolved_ticker
    st.info(f"🔎 티커 검색으로 선택한 **{ticker}** 를 분석합니다. (위 '선택 해제'로 취소 가능)")
else:
    ticker = default_ticker

# ============================================================
# 데이터 로드 (rate-limit 대비 재시도 포함)
# ============================================================
@st.cache_data(ttl=600, show_spinner="📡 시세 데이터를 불러오는 중...")
def load_history(tk, retries=3):
    last_err = None
    for attempt in range(retries):
        try:
            df = yf.Ticker(tk).history(period="1y")
            if len(df) > 0:
                return df, None
        except Exception as e:
            last_err = e
        time.sleep(1.5 * (attempt + 1))
    return pd.DataFrame(), last_err

stock_data = yf.Ticker(ticker)
hist, load_err = load_history(ticker)

# 코스피 실패 시 코스닥 재시도
if len(hist) == 0 and region_hint == "KR" and ticker.endswith(".KS"):
    ticker = ticker.replace(".KS", ".KQ")
    stock_data = yf.Ticker(ticker)
    hist, load_err = load_history(ticker)

if len(hist) == 0 and load_err is not None and "RateLimit" in type(load_err).__name__:
    st.error("⏳ 야후 파이낸스가 일시적으로 요청을 제한하고 있습니다 (Rate Limit).")
    st.info(
        "이 현상은 보통 **일시적**이며, 클라우드의 공유 IP 때문에 자주 발생합니다.\n\n"
        "- **5~10분 정도 기다린 뒤** 새로고침(F5)해 보세요.\n"
        "- 같은 종목을 연속으로 여러 번 조회하지 말고 잠시 간격을 두세요.\n"
        "- 대부분 한 시간 안에 자동으로 풀립니다."
    )
    st.stop()

if len(hist) > 0:
    try:
        info = stock_data.info
    except Exception:
        info = {}

    company_name = info.get('longName', info.get('shortName', ticker))

    # ---- 종목코드 ↔ 실제 데이터 매칭 검증 ----
    mismatch_warning = None
    if region_hint == "KR":
        returned_symbol = (info.get('symbol') or "").upper()
        expected_code = ticker.split(".")[0]
        if returned_symbol and expected_code not in returned_symbol:
            mismatch_warning = (
                f"⚠️ 요청한 종목코드({ticker})와 야후가 반환한 정보(symbol: {returned_symbol})가 "
                f"일치하지 않습니다. 아래 회사명이 원하시는 종목이 맞는지 꼭 확인해주세요."
            )
        elif not info.get('longName') and not info.get('shortName'):
            mismatch_warning = (
                f"⚠️ {ticker}에 대한 회사명 정보를 가져오지 못했습니다. "
                f"종목코드를 다시 확인하거나 잠시 후 다시 시도해주세요."
            )

    valid_close = hist['Close'].dropna()
    current_price = valid_close.iloc[-1] if not valid_close.empty else info.get('regularMarketPrice', info.get('previousClose', 0.0))
    if pd.isna(current_price) or current_price is None:
        current_price = 0.0

    # ---- 보조지표 ----
    hist['MA20'] = hist['Close'].rolling(window=20, min_periods=1).mean()
    hist['MA60'] = hist['Close'].rolling(window=60, min_periods=1).mean()
    hist['MA120'] = hist['Close'].rolling(window=120, min_periods=1).mean()

    low_14 = hist['Low'].rolling(window=14, min_periods=1).min()
    high_14 = hist['High'].rolling(window=14, min_periods=1).max()
    denom = (high_14 - low_14).replace(0, 1)
    hist['%K'] = 100 * ((hist['Close'] - low_14) / denom)
    hist['%D'] = hist['%K'].rolling(window=3, min_periods=1).mean()

    exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
    exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
    hist['MACD'] = exp1 - exp2
    hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
    hist['Histogram'] = hist['MACD'] - hist['Signal']

    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss.replace(0, 1)
    hist['RSI'] = 100 - (100 / (1 + rs))

    diff_price = hist['MA20'] - hist['MA60']
    hist['GoldenCross'] = (diff_price.shift(1) < 0) & (diff_price > 0)
    hist['DeadCross'] = (diff_price.shift(1) > 0) & (diff_price < 0)
    diff_macd = hist['MACD'] - hist['Signal']
    hist['MacdGolden'] = (diff_macd.shift(1) < 0) & (diff_macd > 0)
    hist['MacdDead'] = (diff_macd.shift(1) > 0) & (diff_macd < 0)

    def last_valid(col, fallback):
        s = hist[col].dropna()
        return s.iloc[-1] if not s.empty else fallback

    ma20_val = last_valid('MA20', current_price)
    ma120_val = last_valid('MA120', current_price)
    stoch_k = last_valid('%K', 50)
    stoch_d = last_valid('%D', 50)
    macd_val = last_valid('MACD', 0.0)
    macd_signal = last_valid('Signal', 0.0)
    rsi_val = last_valid('RSI', 50.0)
    volume_ratio = (hist['Volume'].iloc[-1] / hist['Volume'].mean()) * 100 if hist['Volume'].mean() else 0

    target_mean = info.get('targetMeanPrice')
    target_high = info.get('targetHighPrice')
    target_low = info.get('targetLowPrice')
    recommendation_key = (info.get('recommendationKey') or 'N/A').upper()
    num_analysts = info.get('numberOfAnalystOpinions', 'N/A')
    recommendation_mapping = {
        'STRONG_BUY': '🔥 강력 매수', 'BUY': '🟢 매수', 'HOLD': '🟡 보유',
        'UNDERPERFORM': '🟠 비중축소', 'SELL': '🔴 매도', 'N/A': '데이터 공백'
    }
    recommendation_kor = recommendation_mapping.get(recommendation_key, recommendation_key)

    mc = info.get('marketCap') or 0
    market_cap = mc / cap_divider if mc and cap_divider else 0

    per = info.get('trailingPE') or info.get('forwardPE')
    if not isinstance(per, (int, float)) or pd.isna(per):
        eps = info.get('trailingEps') or info.get('forwardEps')
        if isinstance(eps, (int, float)) and eps > 0 and current_price > 0:
            per = current_price / eps
        else:
            per = None
    per_display = f"{per:.2f}" if isinstance(per, (int, float)) and per > 0 else "N/A (적자)"

    roe_raw = info.get('returnOnEquity')
    roe = roe_raw * 100 if isinstance(roe_raw, (int, float)) else 0

    # ========================================================
    # 헤더 + 종합 상태 배지
    # ========================================================
    st.markdown(f"### 🏢 현재 분석 중: **{company_name}**  `({ticker})`")

    if mismatch_warning:
        st.warning(mismatch_warning)
        if st.button("🔄 데이터 다시 불러오기 (캐시 지우기)"):
            st.cache_data.clear()
            st.rerun()

    # ---- 종합 상태 판정 (배지용) ----
    signal_score = 0
    if isinstance(target_mean, (int, float)) and target_mean > 0 and current_price > 0:
        if target_mean > current_price * 1.10:
            signal_score += 1
        elif target_mean < current_price * 0.95:
            signal_score -= 1
    if recommendation_key in ("STRONG_BUY", "BUY"):
        signal_score += 1
    elif recommendation_key in ("SELL", "UNDERPERFORM"):
        signal_score -= 1
    if rsi_val < 30:
        signal_score += 1
    elif rsi_val > 70:
        signal_score -= 1
    if macd_val > macd_signal:
        signal_score += 1
    else:
        signal_score -= 1

    if signal_score >= 2:
        badge_bg, badge_txt, badge_label = "#0b3d2e", "#00e676", "🟢 긍정 신호 우위"
    elif signal_score <= -2:
        badge_bg, badge_txt, badge_label = "#3d0b12", "#ff5252", "🔴 주의 신호 우위"
    else:
        badge_bg, badge_txt, badge_label = "#3d360b", "#ffd54f", "🟡 중립 / 혼조"

    st.markdown(
        f"""
        <div style="background:{badge_bg};padding:14px 18px;border-radius:12px;
        border:1px solid {badge_txt}33;margin:6px 0 14px 0;">
          <span style="font-size:20px;font-weight:700;color:{badge_txt};">
          현재 종합 신호: {badge_label}</span>
          <span style="color:#aaa;font-size:13px;margin-left:10px;">
          (목표가·기관의견·RSI·MACD를 기계적으로 합산한 참고용 신호예요)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if current_price == 0.0:
        price_str = "N/A"
    elif region_hint == "KR":
        price_str = f"{currency_symbol}{int(current_price):,}"
    else:
        price_str = f"{currency_symbol}{current_price:,.2f}"

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("현재 주가", price_str)
    h2.metric("시가총액", f"{currency_symbol}{market_cap:,.2f}{cap_unit}" if market_cap else "N/A")
    h3.metric("PER", per_display)
    h4.metric("ROE", f"{roe:.2f}%" if roe != 0 else "N/A")

    st.write("")

    tab_chart, tab_tech, tab_analyst, tab_check, tab_ai = st.tabs(
        ["📈 차트", "📊 기술지표", "🏛️ 애널리스트", "🎯 조건검증", "🧠 AI 리포트"]
    )

    # ---------- 탭1: 차트 ----------
    with tab_chart:
        period_label = st.radio("표시 기간", ["1개월", "3개월", "6개월", "1년"], index=2, horizontal=True)
        period_map = {"1개월": 21, "3개월": 63, "6개월": 126, "1년": 252}
        n = period_map[period_label]
        view = hist.tail(n)

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            row_heights=[0.50, 0.14, 0.18, 0.18], vertical_spacing=0.03)

        fig.add_trace(go.Candlestick(
            x=view.index, open=view['Open'], high=view['High'],
            low=view['Low'], close=view['Close'], name="가격",
            increasing_line_color="#26a69a", increasing_fillcolor="#26a69a",
            decreasing_line_color="#ef5350", decreasing_fillcolor="#ef5350",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=view.index, y=view['MA20'], name="MA20",
                                 line=dict(color="#f0b90b", width=1.3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=view.index, y=view['MA60'], name="MA60",
                                 line=dict(color="#2962ff", width=1.3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=view.index, y=view['MA120'], name="MA120",
                                 line=dict(color="#e040fb", width=1.5)), row=1, col=1)

        gc = view[view['GoldenCross'] == True]
        dc = view[view['DeadCross'] == True]
        if not gc.empty:
            fig.add_trace(go.Scatter(x=gc.index, y=gc['MA20'], mode='markers', name='골든크로스',
                marker=dict(symbol='triangle-up', size=13, color='#00e676',
                            line=dict(width=1, color='white'))), row=1, col=1)
        if not dc.empty:
            fig.add_trace(go.Scatter(x=dc.index, y=dc['MA20'], mode='markers', name='데드크로스',
                marker=dict(symbol='triangle-down', size=13, color='#ff1744',
                            line=dict(width=1, color='white'))), row=1, col=1)

        vol_colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(view['Close'], view['Open'])]
        fig.add_trace(go.Bar(x=view.index, y=view['Volume'], name="거래량",
                             marker_color=vol_colors, opacity=0.6), row=2, col=1)

        fig.add_trace(go.Scatter(x=view.index, y=view['RSI'], name="RSI",
                                 line=dict(color="#ab47bc", width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line=dict(color="#ef5350", width=1, dash="dash"), row=3, col=1)
        fig.add_hline(y=30, line=dict(color="#26a69a", width=1, dash="dash"), row=3, col=1)

        macd_hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in view['Histogram']]
        fig.add_trace(go.Bar(x=view.index, y=view['Histogram'], name="MACD Hist",
                             marker_color=macd_hist_colors, opacity=0.5), row=4, col=1)
        fig.add_trace(go.Scatter(x=view.index, y=view['MACD'], name="MACD",
                                 line=dict(color="#2962ff", width=1.3)), row=4, col=1)
        fig.add_trace(go.Scatter(x=view.index, y=view['Signal'], name="Signal",
                                 line=dict(color="#f0b90b", width=1.3)), row=4, col=1)

        mg = view[view['MacdGolden'] == True]
        md = view[view['MacdDead'] == True]
        if not mg.empty:
            fig.add_trace(go.Scatter(x=mg.index, y=mg['MACD'], mode='markers', name='MACD 골든',
                marker=dict(symbol='circle', size=9, color='#00e676',
                            line=dict(width=1, color='white')), showlegend=False), row=4, col=1)
        if not md.empty:
            fig.add_trace(go.Scatter(x=md.index, y=md['MACD'], mode='markers', name='MACD 데드',
                marker=dict(symbol='circle', size=9, color='#ff1744',
                            line=dict(width=1, color='white')), showlegend=False), row=4, col=1)

        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#131722", plot_bgcolor="#131722",
            height=900, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
            hovermode="x unified", font=dict(color="#d1d4dc"), barmode="overlay",
        )
        for r in [1, 2, 3, 4]:
            fig.update_yaxes(gridcolor="#2a2e39", row=r, col=1, side="right")
            fig.update_xaxes(gridcolor="#2a2e39", row=r, col=1,
                             rangebreaks=[dict(bounds=["sat", "mon"])])
        fig.update_yaxes(title_text="가격", row=1, col=1)
        fig.update_yaxes(title_text="거래량", row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
        fig.update_yaxes(title_text="MACD", row=4, col=1)

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "💡 드래그하면 확대, 더블클릭하면 원위치. "
            "▲초록 = 골든크로스(MA20이 MA60 상향돌파, 상승 신호), "
            "▼빨강 = 데드크로스(하향돌파, 하락 신호). "
            "RSI 점선은 과매수(70)·과매도(30) 기준선입니다."
        )

    # ---------- 탭2: 기술지표 (게이지 추가) ----------
    with tab_tech:
        recent_gc = hist[hist['GoldenCross'] == True].index
        recent_dc = hist[hist['DeadCross'] == True].index
        last_gc = recent_gc[-1].strftime("%Y-%m-%d") if len(recent_gc) else None
        last_dc = recent_dc[-1].strftime("%Y-%m-%d") if len(recent_dc) else None
        cross_msg = []
        if last_gc:
            cross_msg.append(f"🟢 최근 골든크로스: **{last_gc}**")
        if last_dc:
            cross_msg.append(f"🔴 최근 데드크로스: **{last_dc}**")
        st.info("　|　".join(cross_msg) if cross_msg else "최근 1년 내 MA20/MA60 교차(크로스)가 없습니다.")

        # ---- 게이지 3종 (RSI / 스토캐스틱 / 거래량) ----
        def make_gauge(value, title, vmin, vmax, good_low, good_high, suffix=""):
            # NaN/None 방어 + 범위 보정
            try:
                value = float(value)
                if pd.isna(value):
                    value = vmin
            except (TypeError, ValueError):
                value = vmin
            value = max(vmin, min(value, vmax))
            g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                number={"suffix": suffix, "font": {"size": 26, "color": "#d1d4dc"}},
                title={"text": title, "font": {"size": 15, "color": "#d1d4dc"}},
                gauge={
                    "axis": {"range": [vmin, vmax], "tickcolor": "#888"},
                    "bar": {"color": "#2962ff"},
                    "bgcolor": "#131722",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [vmin, good_low], "color": "rgba(38,166,154,0.33)"},
                        {"range": [good_low, good_high], "color": "rgba(92,99,110,0.20)"},
                        {"range": [good_high, vmax], "color": "rgba(239,83,80,0.33)"},
                    ],
                },
            ))
            g.update_layout(paper_bgcolor="#131722", height=230,
                            margin=dict(l=20, r=20, t=50, b=10),
                            font=dict(color="#d1d4dc"))
            return g

        gcol1, gcol2, gcol3 = st.columns(3)
        with gcol1:
            st.plotly_chart(make_gauge(rsi_val, "RSI (14일)", 0, 100, 30, 70),
                            use_container_width=True)
            st.caption("초록=과매도(살 만함) · 빨강=과매수(비쌈)")
        with gcol2:
            st.plotly_chart(make_gauge(stoch_k, "스토캐스틱 %K", 0, 100, 20, 80),
                            use_container_width=True)
            st.caption("초록=바닥권 · 빨강=천장권")
        with gcol3:
            vr_capped = min(volume_ratio, 300)
            st.plotly_chart(make_gauge(vr_capped, "거래량 (평균대비)", 0, 300, 80, 150, suffix="%"),
                            use_container_width=True)
            st.caption("100% = 평소 수준 · 높으면 관심 급증")

        st.write("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("120일선 이격도", f"{((current_price/ma120_val)-1)*100:.1f}%" if ma120_val else "0.0%")
        c2.metric("스토캐스틱 K / D", f"{stoch_k:.1f} / {stoch_d:.1f}")
        c3.metric("RSI (14일)", f"{rsi_val:.1f}")
        c4.metric("거래량 (평균대비)", f"{volume_ratio:.1f}%")
        m1, m2, m3 = st.columns(3)
        m1.metric("MACD", f"{macd_val:.4f}")
        m2.metric("MACD Signal", f"{macd_signal:.4f}")
        m3.metric("MACD Histogram", f"{(macd_val-macd_signal):.4f}")
        st.caption("RSI 70↑ 과매수 / 30↓ 과매도 · 스토캐스틱 80↑ 과매수 / 20↓ 과매도 · MACD가 Signal 위면 상승 모멘텀")

    # ---------- 탭3: 애널리스트 (목표주가 바) ----------
    with tab_analyst:
        a1, a2, a3 = st.columns(3)
        a1.metric("기관 종합 의견", recommendation_kor)
        if isinstance(target_mean, (int, float)) and target_mean > 0 and current_price > 0:
            upside = ((target_mean / current_price) - 1) * 100
            tgt = f"{currency_symbol}{int(target_mean):,}" if region_hint == "KR" else f"{currency_symbol}{target_mean:,.2f}"
            a2.metric("평균 목표주가", tgt, delta=f"상승 여력 {upside:.1f}%")
        else:
            a2.metric("평균 목표주가", "N/A")
        a3.metric("참여 애널리스트", f"{num_analysts} 명")

        # ---- 목표주가 범위 바 차트 ----
        if isinstance(target_mean, (int, float)) and target_mean > 0 and current_price > 0:
            lo = target_low if isinstance(target_low, (int, float)) and target_low > 0 else min(current_price, target_mean) * 0.9
            hi = target_high if isinstance(target_high, (int, float)) and target_high > 0 else max(current_price, target_mean) * 1.1
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=[lo, hi], y=["목표가 범위", "목표가 범위"],
                mode="lines", line=dict(color="#5c636e", width=14),
                showlegend=False, hoverinfo="skip"))
            fig_t.add_trace(go.Scatter(
                x=[current_price], y=["목표가 범위"], mode="markers+text",
                marker=dict(color="#f0b90b", size=18, symbol="diamond",
                            line=dict(color="white", width=1)),
                text=["현재가"], textposition="top center",
                textfont=dict(color="#f0b90b"), showlegend=False))
            fig_t.add_trace(go.Scatter(
                x=[target_mean], y=["목표가 범위"], mode="markers+text",
                marker=dict(color="#00e676", size=18, symbol="star",
                            line=dict(color="white", width=1)),
                text=["평균 목표가"], textposition="bottom center",
                textfont=dict(color="#00e676"), showlegend=False))
            fig_t.update_layout(
                template="plotly_dark", paper_bgcolor="#131722", plot_bgcolor="#131722",
                height=200, margin=dict(l=10, r=10, t=30, b=10),
                font=dict(color="#d1d4dc"),
                xaxis=dict(title="주가", gridcolor="#2a2e39"),
                yaxis=dict(showticklabels=False))
            st.plotly_chart(fig_t, use_container_width=True)
            st.caption("◆노랑 = 현재가 · ⭐초록 = 기관 평균 목표가 · 회색 막대 = 최저~최고 목표가 범위")
        st.caption("야후 파이낸스가 집계한 기관 컨센서스입니다. 종목에 따라 데이터가 없을 수 있습니다.")

    # ---------- 탭4: 조건검증 (카드형) ----------
    with tab_check:
        is_small_cap = market_cap < small_cap_threshold if market_cap else False
        is_high_roe = roe >= 15.0
        is_reasonable_per = (isinstance(per, (int, float)) and per <= 30) or per_display.startswith("N/A")
        is_near_ma120 = (current_price / ma120_val) <= 1.10 if ma120_val else False
        is_rsi_oversold = rsi_val < 30
        pass_count = sum([is_small_cap, is_high_roe, is_reasonable_per])

        if pass_count == 3 and is_near_ma120:
            st.success("⚖️ 기계적 진단: 재무 조건 충족 및 장기 지지선 안착 구간.")
        elif pass_count >= 2:
            st.info("⚖️ 기계적 진단: 펀더멘털 양호. 단기 수급 불균형 여부 확인 필요.")
        else:
            st.warning("⚠️ 기계적 진단: 고평가 혹은 성장 임계치 도달. 보수적 관점 유지.")

        def check_card(label, ok, ok_txt, no_txt):
            color = "#0b3d2e" if ok else "#3d0b12"
            edge = "#00e676" if ok else "#ff5252"
            mark = "✅" if ok else "❌"
            txt = ok_txt if ok else no_txt
            return f"""<div style="background:{color};border:1px solid {edge}44;
            border-radius:10px;padding:12px 14px;margin:6px 0;">
            <span style="font-size:16px;font-weight:600;color:{edge};">{mark} {label}</span><br>
            <span style="color:#c8c8c8;font-size:13px;">{txt}</span></div>"""

        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.markdown(check_card("재무 펀더멘털", pass_count >= 2,
                        "ROE·PER 등 재무 기준을 충족했어요.",
                        "재무 기준을 일부 못 채웠어요."), unsafe_allow_html=True)
        with cc2:
            st.markdown(check_card("가격 안전마진 (120일선)", is_near_ma120,
                        "장기 평균선 근처라 과열이 덜해요.",
                        "장기 평균선보다 많이 올라 있어요."), unsafe_allow_html=True)
        with cc3:
            st.markdown(check_card("RSI 과매도 진입", is_rsi_oversold,
                        "과매도 구간이라 반등 여지가 있어요.",
                        "과매도까지는 아니에요 (정상 범위)."), unsafe_allow_html=True)
        st.caption("개인 투자 원칙을 기계적으로 점검한 결과입니다. 매수/매도 권유가 아닙니다.")

    # ---------- 탭5: AI 리포트 ----------
    with tab_ai:
        st.subheader("🌐 뉴스 / 촉매 입력")
        try:
            raw_news = stock_data.news
        except Exception:
            raw_news = []
        news_context = ""
        manual_news = st.text_area(
            "✍️ 최근 포착한 촉매·호재·검증하고 싶은 키워드를 입력하세요:",
            placeholder="예: '독점 공급 계약 수주 및 차세대 플랫폼 출시 확정' 등"
        )
        if raw_news:
            st.write("**시스템 추적 뉴스:**")
            for item in raw_news[:3]:
                title = item.get('title', '제목 없음')
                publisher = item.get('publisher', '출처 미상')
                st.write(f"- [{publisher}] {title}")
                news_context += f"출처: {publisher} / 제목: {title}\n"
        if manual_news:
            news_context += f"\n[투자자 직접 입력 이슈]\n{manual_news}\n"

        st.write("---")
        st.subheader("🧠 종합 리서치 보고서")
        if st.button("🎬 입체적 퀀트 + 실시간 구글 서치 종합 분석 시작"):
            if not api_key:
                st.warning("왼쪽 설정에서 Gemini API Key를 입력하세요.")
            else:
                with st.spinner("구글 검색·기관 컨센서스·수급 지표를 융합 중입니다..."):
                    try:
                        genai.configure(api_key=api_key)
                        model_candidates = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"]
                        response = None
                        used_search = False

                        prompt = f"""
너는 주식 투자를 오래 해온 친한 친구야. 지금 친구가 "이 회사 어때?"라고 물어봐서,
편하게, 하지만 아는 건 정확하게 짚어주면서 설명해주는 상황이야.

[말투 규칙 — 반드시 지킬 것]
- 인사말, 자기소개, "~분석해 드리겠습니다" 같은 서론 문장은 절대 쓰지 말고 바로 0번 항목부터 시작해.
- 격식체 대신, 친한 친구에게 설명하듯 편안한 존댓말("~예요", "~거든요", "~더라고요", "~해요")을 써.
- 전문용어(PER, ROE, RSI, MACD 등)가 나오면 괄호로 짧게 풀어줘. 예: "PER(주가가 회사 이익의 몇 배에 거래되는지 보는 지표)"
- 어려운 개념은 일상적인 비유를 들어 설명해. (예: 회사를 동네 가게에 비유, 경쟁을 시험 등수에 비유 등)
- 각 항목은 핵심만 던지고 끝내지 말고, 왜 그런지 배경까지 2~3문장 이상 충분히 풀어서 설명해. 너무 짧게 줄이지 마.
- 다만 불필요하게 늘어지지는 말고, 읽는 사람이 "아 그렇구나" 하고 이해되게.
- 감정을 배제하고 균형 잡힌 시각은 유지해(막연한 낙관·비관 금지). 문체만 편안하게.
- 특정 매수·매도 시점을 단정적으로 지시하지 마.

반드시 구글 검색을 활용해 '{company_name}'의 최신 사업 현황과 시장 평가를 확인하고 반영해.

[대상]: {company_name} ({ticker})
[퀀트]: 현재가 {current_price:.2f} / 시총 {market_cap:.2f}{cap_unit} / PER {per_display} / ROE {roe:.2f}%
[기술]: MA20 {ma20_val:.2f} / MA120 {ma120_val:.2f} / 스토캐스틱 {stoch_k:.1f} / RSI {rsi_val:.1f} / MACD {macd_val:.4f}
[기관]: 의견 {recommendation_kor} / 목표가 {target_mean} / 인원 {num_analysts}명
[단서]:
{news_context}

[출력 순서 — 위 말투 규칙을 지켜서]:
0. 🏢 **이 회사, 한마디로 뭐하는 곳이야?**: 무슨 사업을 하는 회사인지 쉽게 3~4문장으로 설명하고, 이어서 지금 어떻게 돈을 벌고 있는지(주력 사업·주요 고객 등), 아직 적자라면 앞으로 어떻게 벌 계획인지 구체적으로 설명해.
1. 📈 **오를 만한 이유 (Bull Case)**: 핵심 성장 동력과 강세 논리를 배경까지 풀어서.
2. 🚨 **내릴 만한 이유 (Bear Case)**: 리스크와 약점을 왜 위험한지까지.
3. 🏛️ **전문가들 생각 (기관 컨센서스)**: 목표가·의견이 얼마나 믿을만한지, 왜 그렇게 보는지.
4. 📊 **지금 차트는 어떤 상태야?**: 이격도·스토캐스틱·MACD·RSI를 종합해서 지금이 과열/침체/중립 중 어디인지, 시나리오별로 균형 있게.
5. ✅ **한 줄 정리**: 위 내용을 3~4줄로 최종 요약. 친구가 결론만 읽어도 감이 잡히게.
"""
                        for model_name in model_candidates:
                            try:
                                model = genai.GenerativeModel(model_name, tools=[{"google_search": {}}])
                                response = model.generate_content(prompt)
                                if response:
                                    used_search = True
                                    st.caption(f"ℹ️ `{model_name}` (구글 서치 포함) 구동 성공")
                                    break
                            except Exception:
                                try:
                                    model = genai.GenerativeModel(model_name)
                                    response = model.generate_content(prompt)
                                    if response:
                                        used_search = False
                                        st.caption(f"ℹ️ `{model_name}` (일반 모드, 구글 서치 미사용) 구동 성공")
                                        break
                                except Exception:
                                    continue
                        if response:
                            st.success("✅ 리서치 보고서 발급 완료")
                            if not used_search:
                                st.caption("⚠️ 이번 응답은 구글 검색 없이 생성되었습니다. 최신 이슈는 위 뉴스/촉매 입력란을 참고해주세요.")
                            st.markdown(response.text)
                        else:
                            st.error("모든 모델 호출에 실패했습니다.")
                    except Exception as e:
                        st.error(f"오류 발생: {e}")
else:
    st.error("종목 코드가 유효하지 않거나 데이터를 불러올 수 없습니다. 종목명을 다시 선택하거나 코드를 확인해주세요.")
