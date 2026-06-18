import streamlit as st
import yfinance as yf
import google.generativeai as genai
import pandas as pd
import plotly.graph_objects as go

# 1. 웹페이지 제목 및 레이아웃 설정
st.set_page_config(
    page_title="관심종목분석",
    page_icon="📊",
    layout="wide"
)

# 2. [핵심 보완] 사이드바 UI — Secrets 비밀 금고 자동 연동 및 백업 이중화 🔐
st.sidebar.title("🔐 시스템 보안 통제")

# 스트림릿 클라우드 Secrets 금고에서 GEMINI_API_KEY가 있는지 우선 탐색
secret_key = st.secrets.get("GEMINI_API_KEY", "")

if secret_key:
    # 금고에서 키를 발견한 경우 자동 로그인 처리
    api_key = secret_key
    st.sidebar.success("🔒 Secrets 금고에서 API Key를 자동으로 불러왔습니다.")
    
    # 백업용으로 다른 키를 입력해 볼 수 있는 숨김 창 제공
    with st.sidebar.expander("🔑 다른 API Key로 대체하기 (선택)", expanded=False):
        user_override = st.text_input("대체 API Key 입력:", type="password")
        if user_override:
            api_key = user_override
else:
    # 금고에 키가 없는 경우 (Local PC 개발 환경 등) 수동 입력창 강제 활성화
    with st.sidebar.expander("🔑 구글 Gemini API Key 입력 (수동)", expanded=True):
        api_key = st.text_input("API Key:", type="password")
    st.sidebar.caption("※ 클라우드 Secrets 설정 전이므로 수동 입력이 필요합니다.")

# 3. 메인 관제 패널 🎛️
st.title("📊 관심종목분석")
st.write("시장 컨센서스, 기술적 수급, 실시간 성장 촉매를 융합하여 균형 잡힌 투자 결정을 지원합니다.")
st.write("---")

input_col1, input_col2 = st.columns([1, 2])

with input_col1:
    market_choice = st.radio("📊 분석 대상 시장 선택", ["미국 주식 (US)", "국내 주식 (KR)"], horizontal=True)

with input_col2:
    if market_choice == "미국 주식 (US)":
        user_input = st.text_input("🔍 분석 종목 티커 입력 (영문 기호):", "NVDA")
        ticker = user_input.upper().strip()
        currency_symbol = "$"
        cap_unit = "B"
        cap_divider = 1000000000
        small_cap_threshold = 10.0
    else:
        user_input = st.text_input("🔍 분석 종목 코드 입력 (숫자 6자리):", "006800")
        raw_ticker = user_input.strip()
        currency_symbol = "₩"
        cap_unit = "조 원"
        cap_divider = 1000000000000
        small_cap_threshold = 1.5
        ticker = f"{raw_ticker}.KS" 

# 4. 데이터 로드 및 기술적 지표 계산 ⚙
stock_data = yf.Ticker(ticker)
hist = stock_data.history(period="1y") 

if len(hist) == 0 and market_choice == "국내 주식 (KR)":
    ticker = f"{raw_ticker}.KQ"
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period="1y")

if len(hist) > 0:
    info = stock_data.info
    company_name = info.get('longName', info.get('shortName', ticker))
    
    # 결측치 제거 후 최신 유효 가격 추출
    valid_close_series = hist['Close'].dropna()
    if not valid_close_series.empty:
        current_price = valid_close_series.iloc[-1]
    else:
        current_price = info.get('regularMarketPrice', info.get('previousClose', 0.0))
        
    if pd.isna(current_price) or current_price is None:
        current_price = 0.0

    # 보조지표 연산
    hist['MA20'] = hist['Close'].rolling(window=20, min_periods=1).mean()
    hist['MA120'] = hist['Close'].rolling(window=120, min_periods=1).mean()
    
    low_14 = hist['Low'].rolling(window=14, min_periods=1).min()
    high_14 = hist['High'].rolling(window=14, min_periods=1).max()
    denom = high_14 - low_14
    hist['%K'] = 100 * ((hist['Close'] - low_14) / denom.replace(0, 1))
    hist['%D'] = hist['%K'].rolling(window=3, min_periods=1).mean()
    
    # MACD 계산
    exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
    exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
    hist['MACD'] = exp1 - exp2
    hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
    hist['Histogram'] = hist['MACD'] - hist['Signal']
    
    # RSI 계산 (14일)
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss.replace(0, 1)
    hist['RSI'] = 100 - (100 / (1 + rs))
    
    ma20_val = hist['MA20'].dropna().iloc[-1] if not hist['MA20'].dropna().empty else current_price
    ma120_val = hist['MA120'].dropna().iloc[-1] if not hist['MA120'].dropna().empty else current_price
    stoch_k = hist['%K'].dropna().iloc[-1] if not hist['%K'].dropna().empty else 50
    stoch_d = hist['%D'].dropna().iloc[-1] if not hist['%D'].dropna().empty else 50
    macd_val = hist['MACD'].dropna().iloc[-1] if not hist['MACD'].dropna().empty else 0.0
    macd_signal = hist['Signal'].dropna().iloc[-1] if not hist['Signal'].dropna().empty else 0.0
    rsi_val = hist['RSI'].dropna().iloc[-1] if not hist['RSI'].dropna().empty else 50.0
    volume_ratio = (hist['Volume'].iloc[-1] / hist['Volume'].mean()) * 100

    # 애널리스트 컨센서스 데이터 가공
    target_mean = info.get('targetMeanPrice')
    recommendation_key = info.get('recommendationKey', 'N/A').upper()
    num_analysts = info.get('numberOfAnalystOpinions', 'N/A')
    
    recommendation_mapping = {
        'STRONG_BUY': '🔥 강력 매수 (Strong Buy)',
        'BUY': '🟢 매수 (Buy)',
        'HOLD': '🟡 보유 (Hold)',
        'UNDERPERFORM': '🟠 비중축소 (Underperform)',
        'SELL': '🔴 매도 (Sell)',
        'N/A': '데이터 공백'
    }
    recommendation_kor = recommendation_mapping.get(recommendation_key, recommendation_key)

    # 🏢 현재 검증 중인 기업 명세 전면 노출
    st.write("---")
    st.markdown(f"### 🏢 현재 분석 중인 대상: **{company_name}** ({ticker})")
    
    # 5. 메인 대시보드 인터페이스
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.subheader("📊 1개년 일봉 캔들 차트 (이평선 융합)")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'], high=hist['High'],
            low=hist['Low'], close=hist['Close'],
            name='일봉(Candle)'
        ))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], name='20일선(단기)', line=dict(width=1.5)))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA120'], name='120일선(장기 지지)', line=dict(width=2)))
        
        fig.update_layout(xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10), height=450)
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("📈 **기술적 수급 지표 데이터**")
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        with t_col1:
            st.metric(label="120일선 이격도", value=f"{((current_price/ma120_val)-1)*100:.1f}%" if ma120_val != 0 else "0.0%")
        with t_col2:
            st.metric(label="스토캐스틱 K/D", value=f"{stoch_k:.1f} / {stoch_d:.1f}")
        with t_col3:
            st.metric(label="RSI (14일)", value=f"{rsi_val:.1f}")
        with t_col4:
            st.metric(label="당일 거래량 (평균대비)", value=f"{volume_ratio:.1f}%")

    with right_col:
        st.subheader("💡 펀더멘털 및 가치평가 데이터")
        pr_col, m_col, p_col, r_col = st.columns(4)
        
        with pr_col:
            if current_price == 0.0:
                st.metric(label="현재 주가", value="N/A (데이터 공백)")
            else:
                if market_choice == "국내 주식 (KR)":
                    st.metric(label="현재 주가", value=f"{currency_symbol}{int(current_price):,}원")
                else:
                    st.metric(label="현재 주가", value=f"{currency_symbol}{current_price:.2f}")
                
        with m_col:
            market_cap = info.get('marketCap', 0) / cap_divider
            st.metric(label="시가총액", value=f"{currency_symbol}{market_cap:.2f}{cap_unit}")
            
        with p_col:
            per = info.get('trailingPE') or info.get('forwardPE')
            if not isinstance(per, (int, float)) or pd.isna(per):
                eps = info.get('trailingEps') or info.get('forwardEps')
                if isinstance(eps, (int, float)) and eps > 0 and current_price > 0:
                    per = current_price / eps
            
            if isinstance(per, (int, float)) and per > 0:
                per_display = f"{per:.2f}"
            else:
                per_display = "N/A (적자 기업)"
            st.metric(label="현재 PER", value=per_display)
            
        with r_col:
            roe = info.get('returnOnEquity', 0) * 100
            st.metric(label="현재 ROE", value=f"{roe:.2f}%" if roe != 0 else "N/A")
            
        st.write("---")
        st.subheader("🏛️ 기관 애널리스트 투자 합산 뷰")
        an_col1, an_col2, an_col3 = st.columns(3)
        
        with an_col1:
            st.metric(label="기관 종합 의견", value=recommendation_kor)
        with an_col2:
            if isinstance(target_mean, (int, float)) and target_mean > 0:
                upside = ((target_mean / current_price) - 1) * 100
                if market_choice == "국내 주식 (KR)":
                    st.metric(label="평균 목표주가", value=f"{currency_symbol}{int(target_mean):,}원", delta=f"상승 여력 {upside:.1f}%")
                else:
                    st.metric(label="평균 목표주가", value=f"{currency_symbol}{target_mean:.2f}", delta=f"상승 여력 {upside:.1f}%")
            else:
                st.metric(label="평균 목표주가", value="N/A (의견 없음)")
        with an_col3:
            st.metric(label="참여 애널리스트 수", value=f"{num_analysts} 명")
            
        st.write("---")
        st.subheader("📊 MACD 및 모멘텀 지표")
        macd_col1, macd_col2 = st.columns(2)
        
        with macd_col1:
            st.metric(label="MACD", value=f"{macd_val:.4f}")
        with macd_col2:
            st.metric(label="MACD Signal", value=f"{macd_signal:.4f}")
            
        st.write("---")
        st.subheader("🎯 개인 투자 원칙 기반 조건 검증")
        
        is_small_cap = market_cap < small_cap_threshold
        is_high_roe = roe >= 15.0
        is_reasonable_per = (isinstance(per, (int, float)) and per <= 30) or per_display == "N/A (적자 기업)"
        is_near_ma120 = (current_price / ma120_val) <= 1.10 if ma120_val != 0 else False
        is_stoch_low = stoch_k < 30
        is_rsi_oversold = rsi_val < 30

        pass_count = sum([is_small_cap, is_high_roe, is_reasonable_per])
        
        if pass_count == 3 and is_near_ma120:
            st.success("⚖️ 기계적 진단: 재무 조건 충족 및 장기 지지선 안착 구간 진입.")
        elif pass_count >= 2:
            st.info("⚖️ 기계적 진단: 기업 펀더멘털 우수함. 단기 수급 불균형 여부 확인 필요.")
        else:
            st.warning("⚠️ 기계적 진단: 고평가 혹은 성장 임계치 도달 상태. 보수적 관점 유지 요망.")
            
        st.markdown(f"- **텐베거 재무 펀더멘털 통과 여부:** {'✅ 합격' if pass_count >= 2 else '❌ 기준 미달'}")
        st.markdown(f"- **가격 안전마진 (120일선 기준):** {'✅ 확보 (진입 유리)' if is_near_ma120 else '❌ 이격 과열 (눌림목 대기 요망)'}")
        st.markdown(f"- **RSI 과매도 진입 여부:** {'✅ 과매도 진입 신호' if is_rsi_oversold else '❌ 정상 범위'}")

    # 6. 실시간 뉴스 및 정보 수집 컨텍스트
    st.write("---")
    st.subheader("🌐 투자자 수집 정보 및 뉴스 피드")
    try:
        raw_news = stock_data.news
    except Exception:
        raw_news = []
        
    news_context = ""
    manual_news = st.text_area(
        "✍️ 최근 포착한 긍정적 촉매제, 호재 기사 내용 또는 검증하고 싶은 성장 키워드를 입력하세요:",
        placeholder="예: '독점 공급 계약 수주 성공 및 차세대 플랫폼 출시 확정' 등"
    )
    
    if raw_news:
        st.write("**기본 시스템 추적 정보:**")
        for item in raw_news[:3]:
            title = item.get('title', '제목 없음')
            publisher = item.get('publisher', '출처 미상')
            st.write(f"- [{publisher}] {title}")
            news_context += f"출처: {publisher} / 제목: {title}\n"
            
    if manual_news:
        news_context += f"\n[투자자 직접 입력 이슈]\n{manual_news}\n"

    # 7. 균형 잡힌 투자전략 리포트 발급 엔진
    st.write("---")
    st.subheader("🧠 글로벌 자산운용사 전략가 종합 리서치 보고서 (개인 투자용)")
    
    if st.button("🎬 입체적 퀀트 데이터 및 실시간 구글 서치 종합 분석 시작"):
        if not api_key:
            st.warning("왼쪽 설정 컨트롤러를 열어 Gemini API Key를 입력하세요.")
        else:
            with st.spinner("구글 검색 네트워크, 기관 컨센서스, 수급 지표를 다각도로 융합 중입니다..."):
                try:
                    genai.configure(api_key=api_key)
                    model_candidates = ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-2.5-pro"]
                    model = None
                    response = None
                    
                    prompt = f"""
                    너는 글로벌 대형 자산운용사의 최고투자책임자(CIO)이자 균형 잡힌 시각으로 자본 효율성을 극대화하는 수석 투자전략가야.
                    개인의 전 재산이 걸린 무거운 투자 결정이므로, 근거 없는 맹목적 낙관론도 배제해야 하지만, 기업 고유의 파괴적 가치와 촉매제를[...]

                    반드시 구글 검색 기능(Google Search Tool)을 활용하여 이 기업 '{company_name}'의 핵심 사업 매력도와 월가/여의도의 긍정적 리포트 논리를 [...]

                    [대상 종목]: {company_name} ({ticker})
                    [정량 퀀트 데이터]: 현재가 {current_price:.2f} / 시총 {market_cap:.2f}{cap_unit} / PER {per_display} / ROE {roe:.2f}%
                    [기술적 포지션]: 20일선 {ma20_val:.2f} / 120일선 {ma120_val:.2f} / 스토캐스틱 {stoch_k:.1f} / RSI {rsi_val:.1f} / MACD {macd_val:.4f}
                    [기관 수급 정보]: 의견 {recommendation_kor} / 목표가 {target_mean} / 참여 인원 {num_analysts}명
                    [투자자 제공 단서]:
                    {news_context}

                    [출력 필수 서식 - 무조건 존댓말을 쓰되, 감정을 배제하고 완벽하게 이성적이고 균형 잡힌 애널리스트 톤을 유지]:
                    1. 📈 **상승 촉매제 및 핵심 성장 모멘텀 분석 (Bull Case)**:
                       - 구글 실시간 서치 및 컨센서스 분석을 기초로, 이 기업이 미래에 주가 폭발(텐베거)을 일으킬 수밖에 없다고 판단되는 핵심 기술[...]
                    2. 🚨 **실시간 다운사이드 리스크 및 방어 기전 (Bear Case)**:
                       - 동전의 뒷면으로서, 이 기업의 상승 스토리를 정면으로 가로막을 수 있는 치명적인 대외 리스크, 경쟁사의 추격, 재무적 한계(적[...]
                    3. 🏛️ **기관 및 애널리스트 컨센서스 평가**:
                       - 현재 제시된 기관들의 평균 목표주가와 종합 매수 의견의 신뢰도를 평가하고, 시장 참여자들이 이 주식의 미래 가치를 대략 얼마[...]
                    4. 📊 **투자 결정을 위한 최종 전략 제언**:
                       - 120일선 이격도와 스토캐스틱 위치, MACD, RSI 가치 평가를 총합하여 지금 진입하는 행위의 손익비를 평가하십시오. 구체적인 행동 [...]
                    """
                    
                    for model_name in model_candidates:
                        try:
                            model = genai.GenerativeModel(model_name, tools=[{"google_search": {}}])
                            response = model.generate_content(prompt)
                            if response:
                                st.write(f"ℹ️ `{model_name}` 모델 (구글 서치 포함) 구동 성공.")
                                break
                        except Exception:
                            try:
                                model = genai.GenerativeModel(model_name)
                                response = model.generate_content(prompt)
                                if response:
                                    st.write(f"ℹ️ `{model_name}` 모델 (일반 모드) 백업 구동 성공.")
                                    break
                            except Exception as e2:
                                st.write(f"❌ `{model_name}` 가동 실패 원인: {e2}")
                                continue
                    
                    if response:
                        st.success("✅ 균형 분석 리서치 보고서 발급이 완료되었습니다.")
                        st.markdown(response.text)
                    else:
                        st.error("구글 서버의 모든 모델이 호출에 실패했습니다.")
                        
                except Exception as e:
                    st.error(f"오류 발생: {e}")
else:
    st.error("종목 코드가 유효하지 않습니다.")
