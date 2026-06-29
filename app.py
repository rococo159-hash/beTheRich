import streamlit as st
import yfinance as yf
import google.generativeai as genai
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(page_title="관심종목분석", page_icon="📊", layout="wide")

# ============================================================
# 국내/미국 종목명 → 티커 매핑
#  - 국내: 야후 파이낸스용 .KS(코스피)/.KQ(코스닥) 접미사 포함
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
# 입력부 — 종목명 검색 / 코드 직접입력 병행
# ============================================================
market_choice = st.radio("📊 분석 대상 시장 선택", ["미국 주식 (US)", "국내 주식 (KR)"], horizontal=True)

in_col1, in_col2 = st.columns(2)

if market_choice == "미국 주식 (US)":
    name_map = US_NAME_TO_TICKER
    with in_col1:
        chosen_name = st.selectbox("🔍 회사 이름으로 검색 (자동완성)", ["(직접 입력)"] + list(name_map.keys()))
    with in_col2:
        typed = st.text_input("⌨️ 또는 티커 직접 입력", "NVDA" if chosen_name == "(직접 입력)" else "")
    ticker = (name_map.get(chosen_name) if chosen_name != "(직접 입력)" else typed.upper().strip()) or "NVDA"
    currency_symbol, cap_unit, cap_divider, small_cap_threshold = "$", "B", 1_000_000_000, 10.0
else:
    name_map = KR_NAME_TO_TICKER
    with in_col1:
        chosen_name = st.selectbox("🔍 회사 이름으로 검색 (자동완성)", ["(직접 입력)"] + list(name_map.keys()))
    with in_col2:
        typed = st.text_input("⌨️ 또는 종목코드 6자리 직접 입력", "" if chosen_name != "(직접 입력)" else "005930")
    if chosen_name != "(직접 입력)":
        ticker = name_map[chosen_name]
    else:
        raw = typed.strip()
        ticker = f"{raw}.KS" if raw else "005930.KS"
    currency_symbol, cap_unit, cap_divider, small_cap_threshold = "₩", "조 원", 1_000_000_000_000, 1.5

# ============================================================
# 데이터 로드
# ============================================================
@st.cache_data(ttl=300)
def load_history(tk):
    return yf.Ticker(tk).history(period="1y")

stock_data = yf.Ticker(ticker)
hist = load_history(ticker)

# 코스피 실패 시 코스닥 재시도 (직접입력 6자리인 경우)
if len(hist) == 0 and market_choice == "국내 주식 (KR)" and ticker.endswith(".KS"):
    ticker = ticker.replace(".KS", ".KQ")
    stock_data = yf.Ticker(ticker)
    hist = load_history(ticker)

if len(hist) > 0:
    info = stock_data.info
    company_name = info.get('longName', info.get('shortName', ticker))

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

    # ---- 애널리스트 ----
    target_mean = info.get('targetMeanPrice')
    recommendation_key = info.get('recommendationKey', 'N/A').upper()
    num_analysts = info.get('numberOfAnalystOpinions', 'N/A')
    recommendation_mapping = {
        'STRONG_BUY': '🔥 강력 매수', 'BUY': '🟢 매수', 'HOLD': '🟡 보유',
        'UNDERPERFORM': '🟠 비중축소', 'SELL': '🔴 매도', 'N/A': '데이터 공백'
    }
    recommendation_kor = recommendation_mapping.get(recommendation_key, recommendation_key)

    # ---- 가치지표 ----
    market_cap = info.get('marketCap', 0) / cap_divider
    per = info.get('trailingPE') or info.get('forwardPE')
    if not isinstance(per, (int, float)) or pd.isna(per):
        eps = info.get('trailingEps') or info.get('forwardEps')
        if isinstance(eps, (int, float)) and eps > 0 and current_price > 0:
            per = current_price / eps
    per_display = f"{per:.2f}" if isinstance(per, (int, float)) and per > 0 else "N/A (적자)"
    roe = info.get('returnOnEquity', 0) * 100

    # ========================================================
    # 헤더: 현재 분석 대상 + 핵심 요약 메트릭
    # ========================================================
    st.markdown(f"### 🏢 현재 분석 중: **{company_name}**  `({ticker})`")

    if current_price == 0.0:
        price_str = "N/A"
    elif market_choice == "국내 주식 (KR)":
        price_str = f"{currency_symbol}{int(current_price):,}"
    else:
        price_str = f"{currency_symbol}{current_price:,.2f}"

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("현재 주가", price_str)
    h2.metric("시가총액", f"{currency_symbol}{market_cap:,.2f}{cap_unit}")
    h3.metric("PER", per_display)
    h4.metric("ROE", f"{roe:.2f}%" if roe != 0 else "N/A")

    st.write("")

    # ========================================================
    # 탭 구조로 가독성 개선
    # ========================================================
    tab_chart, tab_tech, tab_analyst, tab_check, tab_ai = st.tabs(
        ["📈 차트", "📊 기술지표", "🏛️ 애널리스트", "🎯 조건검증", "🧠 AI 리포트"]
    )

    # ---------- 탭1: 트레이딩뷰 스타일 차트 ----------
    with tab_chart:
        period_label = st.radio(
            "표시 기간", ["1개월", "3개월", "6개월", "1년"],
            index=2, horizontal=True
        )
        period_map = {"1개월": 21, "3개월": 63, "6개월": 126, "1년": 252}
        n = period_map[period_label]
        view = hist.tail(n)

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.78, 0.22], vertical_spacing=0.03,
            subplot_titles=("", "")
        )

        # 캔들
        fig.add_trace(go.Candlestick(
            x=view.index, open=view['Open'], high=view['High'],
            low=view['Low'], close=view['Close'], name="가격",
            increasing_line_color="#26a69a", increasing_fillcolor="#26a69a",
            decreasing_line_color="#ef5350", decreasing_fillcolor="#ef5350",
        ), row=1, col=1)

        # 이동평균선
        fig.add_trace(go.Scatter(x=view.index, y=view['MA20'], name="MA20",
                                 line=dict(color="#f0b90b", width=1.3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=view.index, y=view['MA60'], name="MA60",
                                 line=dict(color="#2962ff", width=1.3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=view.index, y=view['MA120'], name="MA120",
                                 line=dict(color="#e040fb", width=1.5)), row=1, col=1)

        # 거래량 (상승/하락 색 구분)
        vol_colors = ["#26a69a" if c >= o else "#ef5350"
                      for c, o in zip(view['Close'], view['Open'])]
        fig.add_trace(go.Bar(x=view.index, y=view['Volume'], name="거래량",
                             marker_color=vol_colors, opacity=0.6), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#131722", plot_bgcolor="#131722",
            height=620, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
            hovermode="x unified",
            font=dict(color="#d1d4dc"),
        )
        fig.update_yaxes(gridcolor="#2a2e39", row=1, col=1, side="right", title_text="가격")
        fig.update_yaxes(gridcolor="#2a2e39", row=2, col=1, side="right", title_text="거래량")
        fig.update_xaxes(gridcolor="#2a2e39", row=2, col=1, rangebreaks=[dict(bounds=["sat", "mon"])])
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], row=1, col=1)

        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 차트를 드래그하면 확대, 더블클릭하면 원위치됩니다. 캔들에 마우스를 올리면 OHLC가 표시됩니다.")

    # ---------- 탭2: 기술지표 ----------
    with tab_tech:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("120일선 이격도", f"{((current_price/ma120_val)-1)*100:.1f}%" if ma120_val else "0.0%")
        c2.metric("스토캐스틱 K / D", f"{stoch_k:.1f} / {stoch_d:.1f}")
        c3.metric("RSI (14일)", f"{rsi_val:.1f}")
        c4.metric("거래량 (평균대비)", f"{volume_ratio:.1f}%")

        st.write("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("MACD", f"{macd_val:.4f}")
        m2.metric("MACD Signal", f"{macd_signal:.4f}")
        m3.metric("MACD Histogram", f"{(macd_val-macd_signal):.4f}")

        st.caption("RSI 70↑ 과매수 / 30↓ 과매도 · 스토캐스틱 80↑ 과매수 / 20↓ 과매도 · MACD가 Signal 위면 상승 모멘텀")

    # ---------- 탭3: 애널리스트 ----------
    with tab_analyst:
        a1, a2, a3 = st.columns(3)
        a1.metric("기관 종합 의견", recommendation_kor)
        if isinstance(target_mean, (int, float)) and target_mean > 0 and current_price > 0:
            upside = ((target_mean / current_price) - 1) * 100
            tgt = f"{currency_symbol}{int(target_mean):,}" if market_choice == "국내 주식 (KR)" else f"{currency_symbol}{target_mean:,.2f}"
            a2.metric("평균 목표주가", tgt, delta=f"상승 여력 {upside:.1f}%")
        else:
            a2.metric("평균 목표주가", "N/A")
        a3.metric("참여 애널리스트", f"{num_analysts} 명")
        st.caption("야후 파이낸스가 집계한 기관 컨센서스입니다. 종목에 따라 데이터가 없을 수 있습니다.")

    # ---------- 탭4: 조건검증 ----------
    with tab_check:
        is_small_cap = market_cap < small_cap_threshold
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

        st.markdown(f"- **재무 펀더멘털:** {'✅ 합격' if pass_count >= 2 else '❌ 기준 미달'}")
        st.markdown(f"- **가격 안전마진 (120일선):** {'✅ 확보' if is_near_ma120 else '❌ 이격 과열'}")
        st.markdown(f"- **RSI 과매도 진입:** {'✅ 과매도 신호' if is_rsi_oversold else '❌ 정상 범위'}")
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
                        prompt = f"""
너는 글로벌 대형 자산운용사의 수석 투자전략가야. 개인의 전 재산이 걸린 무거운 결정이므로, 맹목적 낙관도 비관도 배제하고 철저히 균형 잡힌 시각으로 분석해.
반드시 구글 검색을 활용해 '{company_name}'의 최신 사업 현황과 시장 평가를 확인해.

[대상]: {company_name} ({ticker})
[퀀트]: 현재가 {current_price:.2f} / 시총 {market_cap:.2f}{cap_unit} / PER {per_display} / ROE {roe:.2f}%
[기술]: MA20 {ma20_val:.2f} / MA120 {ma120_val:.2f} / 스토캐스틱 {stoch_k:.1f} / RSI {rsi_val:.1f} / MACD {macd_val:.4f}
[기관]: 의견 {recommendation_kor} / 목표가 {target_mean} / 인원 {num_analysts}명
[단서]:
{news_context}

[출력 — 존댓말, 감정 배제, 균형 잡힌 애널리스트 톤]:
1. 📈 **상승 모멘텀 (Bull Case)**: 핵심 성장 동력과 강세 논리
2. 🚨 **하방 리스크 (Bear Case)**: 치명적 리스크와 재무·경쟁 약점
3. 🏛️ **기관 컨센서스 평가**: 목표가·의견의 신뢰도 평가
4. 📊 **기술적 위치 종합 해설**: 이격도·스토캐스틱·MACD·RSI를 종합해 현재가 과열/침체/중립 중 어디인지 객관적으로 서술. 특정 매수·매도 시점을 단정적으로 지시하지 말고, 시나리오별 관점과 유의점을 균형 있게 제시.
"""
                        for model_name in model_candidates:
                            try:
                                model = genai.GenerativeModel(model_name, tools=[{"google_search": {}}])
                                response = model.generate_content(prompt)
                                if response:
                                    st.caption(f"ℹ️ `{model_name}` (구글 서치 포함) 구동 성공")
                                    break
                            except Exception:
                                try:
                                    model = genai.GenerativeModel(model_name)
                                    response = model.generate_content(prompt)
                                    if response:
                                        st.caption(f"ℹ️ `{model_name}` (일반 모드) 구동 성공")
                                        break
                                except Exception:
                                    continue
                        if response:
                            st.success("✅ 리서치 보고서 발급 완료")
                            st.markdown(response.text)
                        else:
                            st.error("모든 모델 호출에 실패했습니다.")
                    except Exception as e:
                        st.error(f"오류 발생: {e}")
else:
    st.error("종목 코드가 유효하지 않거나 데이터를 불러올 수 없습니다. 종목명을 다시 선택하거나 코드를 확인해주세요.")
