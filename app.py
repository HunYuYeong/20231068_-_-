
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
import scipy.optimize as sco
from datetime import date, timedelta

# ==========================================
# 1. UI 기본 설정 및 폰트 세팅
# ==========================================
st.set_page_config(page_title="20231068유영훈 금프 프로젝트 포트폴리오 진단", layout="wide")

plt.rcParams['font.family'] = 'Malgun Gothic' # Mac은 'AppleGothic'으로 변경
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 데이터 캐싱 (종목 리스트를 매번 새로 불러오지 않도록 메모리에 저장)
# ==========================================
@st.cache_data
def load_tickers():
    kospi = fdr.StockListing('KOSPI')
    kosdaq = fdr.StockListing('KOSDAQ')
    
    t_map = {}
    for _, row in kospi.iterrows():
        t_map[row['Name']] = f"{row['Code']}.KS"
    for _, row in kosdaq.iterrows():
        t_map[row['Name']] = f"{row['Code']}.KQ"
    return t_map

ticker_map = load_tickers()

# 💡 여기서 딕셔너리의 키(종목명)들을 가져와 '가나다순'으로 정렬
stock_list = sorted(list(ticker_map.keys())) 

# ==========================================
# 3. 사이드바 (사용자 입력 UI)
# ==========================================
st.sidebar.header("⚙️ 포트폴리오 설정")

# options에 가나다순으로 정렬된 stock_list
selected_stocks = st.sidebar.multiselect(
    "1. 투자 종목 선택",
    options=stock_list,
    default=["삼성전자", "SK하이닉스", "NAVER"]
)

# 선택된 종목에 맞춰 비중 입력 UI를 동적으로 생성
st.sidebar.subheader("2. 종목별 비중 (%)")
user_weights_input = []
if selected_stocks:
    default_w = 100.0 / len(selected_stocks)
    for stock in selected_stocks:
        w = st.sidebar.number_input(f"{stock} 비중", min_value=0.0, max_value=100.0, value=default_w, step=1.0)
        user_weights_input.append(w)

st.sidebar.subheader("3. 분석 기간 설정")
start_date = st.sidebar.date_input("매수일 (시작일)", date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("매도일 (종료일)", date.today())

run_button = st.sidebar.button("🚀 포트폴리오 진단 시작")

# ==========================================
# 4. 메인 화면 (분석 및 결과 출력)
# ==========================================
st.title("📊 20231068 유영훈 포트폴리오 진단")
st.markdown("현재 투자 중인 비중과 수학적 최적 비중 비교.")

if run_button:
    if len(selected_stocks) < 2:
        st.error("포트폴리오 분석을 위해 최소 2개 이상의 종목을 선택.")
        st.stop()

    # 비중 스케일링 (합이 100%가 되도록)
    total_weight = sum(user_weights_input)
    if total_weight == 0:
        st.error("비중의 합이 0일 수 없습니다.")
        st.stop()
        
    user_weights = np.array([w / total_weight for w in user_weights_input])
    tickers = [ticker_map[s] for s in selected_stocks]

    with st.spinner('주가 데이터를 수집하고 비중 최적화를 진행 중...'):
        # 데이터 수집
        all_data = []
        for name, ticker in zip(selected_stocks, tickers):
            df = yf.Ticker(ticker).history(start=start_date, end=end_date)
            if not df.empty:
                df = df.reset_index()
                df['Date'] = pd.to_datetime(df['Date']).dt.date
                df = df[['Date', 'Close']]
                df.columns = ['date', name]
                df.set_index('date', inplace=True)
                all_data.append(df)

        price_df = pd.concat(all_data, axis=1).dropna()
        returns_df = price_df.pct_change().dropna()

        # 성과 계산 함수
        annual_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        risk_free_rate = 0.02

        def get_performance(weights):
            p_ret = np.sum(annual_returns * weights)
            p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            p_sr = (p_ret - risk_free_rate) / p_std
            return p_ret, p_std, p_sr

        # [내 포트폴리오 성과]
        u_ret, u_std, u_sr = get_performance(user_weights)
        u_cum = (1 + returns_df.dot(user_weights)).cumprod()
        u_mdd = ((u_cum / u_cum.cummax()) - 1.0).min()

        # [최적화 성과]
        def neg_sharpe(w): return -get_performance(w)[2]
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(len(selected_stocks)))
        init_guess = [1./len(selected_stocks)] * len(selected_stocks)
        
        opt_res = sco.minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        opt_weights = np.round(opt_res.x, 3)
        
        o_ret, o_std, o_sr = get_performance(opt_weights)
        o_cum = (1 + returns_df.dot(opt_weights)).cumprod()
        o_mdd = ((o_cum / o_cum.cummax()) - 1.0).min()

    # ==========================================
    # 5. 결과 화면 렌더링
    # ==========================================
    st.subheader("💡 성과 요약 비교")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("누적 수익률", f"{(u_cum.iloc[-1]-1)*100:.2f}%", f"최적: {(o_cum.iloc[-1]-1)*100:.2f}%")
    col2.metric("연환산 위험(변동성)", f"{u_std*100:.2f}%", f"최적: {o_std*100:.2f}%", delta_color="inverse")
    col3.metric("최대 낙폭(MDD)", f"{u_mdd*100:.2f}%", f"최적: {o_mdd*100:.2f}%", delta_color="inverse")
    col4.metric("샤프 지수(효율)", f"{u_sr:.2f}", f"최적: {o_sr:.2f}")

    st.divider()

    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        st.subheader("📈 수익률 추이 그래프")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(u_cum, color='gray', linestyle='--', label=f'내 비중 (Sharpe: {u_sr:.2f})')
        ax.plot(o_cum, color='blue', linewidth=2, label=f'최적 비중 (Sharpe: {o_sr:.2f})')
        ax.set_ylabel('수익률 (1.0 = 원금)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    with col_table:
        st.subheader("⚖️ 리밸런싱 제안")
        
        result_data = []
        for i, name in enumerate(selected_stocks):
            u_w = user_weights[i] * 100
            o_w = opt_weights[i] * 100
            diff = o_w - u_w
            
            if diff > 5: action = "🔺 비중 확대"
            elif diff < -5: action = "🔻 비중 축소"
            else: action = "✅ 유지"
                
            result_data.append({
                "종목명": name,
                "현재 비중(%)": round(u_w, 1),
                "최적 비중(%)": round(o_w, 1),
                "의견": action
            })
            
        st.dataframe(pd.DataFrame(result_data), hide_index=True)

else:
    st.info("👈 왼쪽 사이드바에서 종목과 비중을 설정한 후 '진단 시작' 버튼을 눌러주세요.")