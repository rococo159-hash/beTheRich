import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def main():
    st.title("투자 전략 분석 도우미")

    # 1. 기본 정보 입력
    st.header("1. 기본 정보")
    stock_name = st.text_input("종목 티커 (예: AAPL, TSLA, 삼성전자 005930.KQ)")
    period = st.selectbox("데이터 기간", ["1mo", "3mo", "6mo", "1y", "2y"])
    interval = st.selectbox("데이터 간격", ["1d", "1wk", "1mo"])

    if stock_name:
        # 데이터 불러오기
        data = yf.download(stock_name, period=period, interval=interval)
        st.write("📈 가격 데이터", data.tail())

        # 이동평균선
        data["MA20"] = data["Close"].rolling(window=20).mean()
        data["MA60"] = data["Close"].rolling(window=60).mean()
        data["MA120"] = data["Close"].rolling(window=120).mean()

        # MACD
        macd_indicator = ta.trend.MACD(data["Close"])
        data["MACD"] = macd_indicator.macd()
        data["MACD_Signal"] = macd_indicator.macd_signal()
        data["MACD_Hist"] = macd_indicator.macd_diff()

        # RSI
        rsi_indicator = ta.momentum.RSIIndicator(data["Close"])
        data["RSI"] = rsi_indicator.rsi()

        st.header("2. 기술적 지표 시각화")

        # 📊 Plotly 차트: 가격 + 이동평균선
        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"],
            name="Candlestick"
        ))
        fig_price.add_trace(go.Scatter(x=data.index, y=data["MA20"], line=dict(color="blue"), name="MA20"))
        fig_price.add_trace(go.Scatter(x=data.index, y=data["MA60"], line=dict(color="orange"), name="MA60"))
        fig_price.add_trace(go.Scatter(x=data.index, y=data["MA120"], line=dict(color="green"), name="MA120"))
        fig_price.update_layout(title="가격 및 이동평균선", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_price, use_container_width=True)

        # 📊 MACD 차트
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACD"], line=dict(color="blue"), name="MACD"))
        fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACD_Signal"], line=dict(color="red"), name="Signal"))
        fig_macd.add_trace(go.Bar(x=data.index, y=data["MACD_Hist"], name="Histogram"))
        fig_macd.update_layout(title="MACD", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_macd, use_container_width=True)

        # 📊 RSI 차트
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data["RSI"], line=dict(color="purple"), name="RSI"))
        fig_rsi.add_hline(y=70, line=dict(color="red", dash="dash"))
        fig_rsi.add_hline(y=30, line=dict(color="green", dash="dash"))
        fig_rsi.update_layout(title="RSI", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_rsi, use_container_width=True)

        # 3. 최종 전략 제언
        st.header("3. 최종 전략 제언")
        if st.button("전략 분석 실행"):
            latest = data.iloc[-1]
            st.subheader("📌 종합 분석 결과")
            st.write(f"종목: {stock_name}")
            st.write(f"- MACD: {latest['MACD']:.2f}, Signal: {latest['MACD_Signal']:.2f}, Hist: {latest['MACD_Hist']:.2f}")
            st.write(f"- RSI: {latest['RSI']:.2f}")

            if latest["MACD"] > latest["MACD_Signal"] and latest["RSI"] < 70:
                st.success("✅ 매수 신호: MACD 골든크로스 + RSI 과매수 아님")
            elif latest["MACD"] < latest["MACD_Signal"] and latest["RSI"] > 30:
                st.warning("⚠️ 관망 권장: 추세 불확실")
            else:
                st.error("❌ 매도 신호: MACD 데드크로스 또는 RSI 과매수")

if __name__ == "__main__":
    main()
