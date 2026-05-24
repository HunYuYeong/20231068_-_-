"""
AI 포트폴리오 진단 및 최적화 대시보드
작성자: 20231068 유영훈
설명: 사용자의 포트폴리오 비중과 MPT(현대 포트폴리오 이론) 기반의 최적 비중을 비교 분석하는 Streamlit 앱입니다.
"""

import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
import streamlit as st
import yfinance as yf
import FinanceDataReader as fdr

# ==========================================
# 전역 상수(Constants) 설정
# ==========================================
TRADING_DAYS = 252       # 1년 평균 주식 시장 개장일
RISK_FREE_RATE = 0.02    # 무위험 이자율 (2% 가정)

# ==========================================
# 1. UI 기본 설정 및 폰트 세팅
# ==========================================
st.set_page_config(page_title="20231068유영훈 금프 프로젝트 포트폴리오 진단", layout="wide")

plt.rcParams['font.family'] = 'Malgun Gothic' # Mac은 'AppleGothic'으로 변경
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 핵심 비즈니스 로직 함수 정의
# ==========================================
@st.cache_data
def load_tickers() -> Dict[str, str]:
    """
    한국거래소(KRX)의 코스피 및 코스닥 상장 종목 정보를 불러와 
    종목명과 yfinance 티커(Ticker) 형식으로 매핑된 딕셔너리를 반환합니다.
    
    Returns:
        Dict[str, str]: {종목명: yfinance 티커} 형태의 딕셔너리
    """
    kospi = fdr.StockListing('KOSPI')
    kosdaq = fdr.StockListing('KOSDAQ')
    
    t_map = {}
    for _, row in kospi.iterrows():
        t_map[row['Name']] = f"{row['Code']}.KS"
    for _, row in kosdaq.iterrows():
        t_map[row['Name']] = f"{row['Code']}.KQ"
    return t_map

def fetch_stock_data(selected_stocks: List[str], tickers: List[str], start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    선택된 종목들의 과거 일간 주가(종가) 데이터를 yfinance를 통해 수집합니다.
    
    Args:
        selected_stocks (List[str]): 종목명 리스트
        tickers (List[str]): yfinance 티커 리스트
        start_date (datetime.date): 수집 시작일
        end_date (datetime.date): 수집 종료일
        
    Returns:
        pd.DataFrame: 날짜를 인덱스로 하고 각 종목의 종가를 컬럼으로 갖는 데이터프레임
    """
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
            
    return pd.concat(all_data, axis=1).dropna()

def calculate_portfolio_performance(weights: np.ndarray, annual_returns: pd.Series, cov_matrix: pd.DataFrame) -> Tuple[float, float, float]:
    """
    주어진 비중(weights)을 바탕으로 포트폴리오의 기대 수익률, 변동성, 샤프 지수를 계산합니다.
    
    Args:
        weights (np.ndarray): 종목별 투자 비중 배열
        annual_returns (pd.Series): 종목별 연환산 수익률
        cov_matrix (pd.DataFrame): 종목 간의 연환산 공분산 행렬
        
    Returns:
        Tuple[float, float, float]: 연환산 기대수익률, 연환산 변동성, 샤프 지수
    """
    p_ret = np.sum(annual_returns * weights)
    p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    p_sr = (p_ret - RISK_FREE_RATE) / p_std
    return p_ret, p_std, p_sr

def optimize_portfolio(num_assets: int, annual_returns: pd.Series, cov_matrix: pd.DataFrame) -> np.ndarray:
    """
    SciPy의 SLSQP 알고리즘을 사용하여 샤프 지수를 극대화하는 최적의 자산 비중을 계산합니다.
    
    Args:
        num_assets (int): 포트폴리오에 포함된 자산의 개수
        annual_returns (pd.Series): 종목별 연환산 수익률
        cov_matrix (pd.DataFrame): 종목 간의 연환산 공분산 행렬
        
    Returns:
        np.ndarray: 최적화된 종목별 비중 배열
    """
    def neg_sharpe(w): 
        return -calculate_portfolio_performance(w, annual_returns, cov_matrix)[2]
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # 비중의 합은 1(100%)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))          # 공매도 금지 (비중은 0~1 사이)
    init_guess = [1. / num_assets] * num_assets                    # 초기값: 동일 비중 분배
    
    opt_res = sco.minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return np.round(opt_res.x, 3)

# ==========================================
# 3. 데이터 로드 및 초기화
# ==========================================
ticker_map = load_tickers()
stock_list = sorted(list(ticker_map.keys())) # 가나다순 정렬

# ==========================================
# 4. 사이드바 (사용자 입력 UI)
# ==========================================
st.sidebar.header("⚙️ 포트폴리오 설정")

selected_stocks = st.sidebar.multiselect(
    "1. 투자 종목 선택",
    options=stock_list,
    default=["삼성전자", "SK하이닉스", "NAVER"]
)

st.sidebar.subheader("2. 종목별 비중 (%)")
user_weights_input = []
if selected_stocks:
    default_w = 100.0 / len(selected_stocks)
    for stock in selected_stocks:
        w = st.sidebar.number_input(f"{stock} 비중", min_value=0.0, max_value=100.0, value=default_w, step=1.0)
        user_weights_input.append(w)

st.sidebar.subheader("3. 분석 기간 설정")
start_date = st.sidebar.date_input("매수일 (시작일)", datetime.date.today() - datetime.timedelta(days=365))
end_date = st.sidebar.date_input("매도일 (종료일)", datetime.date.today())

run_button = st.sidebar.button("🚀 포트폴리오 진단 시작")

# ==========================================
# 5. 메인 화면 (분석 및 결과 출력)
# ==========================================
st.title("📊 20231068 유영훈 포트폴리오 진단")
st.markdown("현재 투자 중인 비중과 수학적 최적 비중 비교.")

if run_button:
    # 5-1. 예외 처리 (방어적 프로그래밍)
    if len(selected_stocks) < 2:
        st.error("포트폴리오 분석을 위해 최소 2개 이상의 종목을 선택.")
        st.stop()

    total_weight = sum(user_weights_input)
    if total_weight == 0:
        st.error("비중의 합이 0일 수 없습니다.")
        st.stop()
        
    # 5-2. 데이터 준비
    user_weights = np.array([w / total_weight for w in user_weights_input])
    tickers = [ticker_map[s] for s in selected_stocks]

    with st.spinner('주가 데이터를 수집하고 비중 최적화를 진행 중...'):
        
        # 주가 데이터 수집 및 전처리
        price_df = fetch_stock_data(selected_stocks, tickers, start_date, end_date)
        returns_df = price_df.pct_change().dropna()

        # 공분산 및 연환산 수익률 계산
        annual_returns = returns_df.mean() * TRADING_DAYS
        cov_matrix = returns_df.cov() * TRADING_DAYS

        # [내 포트폴리오 성과 계산]
        u_ret, u_std, u_sr = calculate_portfolio_performance(user_weights, annual_returns, cov_matrix)
        u_cum = (1 + returns_df.dot(user_weights)).cumprod()
        u_mdd = ((u_cum / u_cum.cummax()) - 1.0).min()

        # [최적화 성과 계산]
        opt_weights = optimize_portfolio(len(selected_stocks), annual_returns, cov_matrix)
        o_ret, o_std, o_sr = calculate_portfolio_performance(opt_weights, annual_returns, cov_matrix)
        o_cum = (1 + returns_df.dot(opt_weights)).cumprod()
        o_mdd = ((o_cum / o_cum.cummax()) - 1.0).min()

    # ==========================================
    # 6. 결과 화면 렌더링
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