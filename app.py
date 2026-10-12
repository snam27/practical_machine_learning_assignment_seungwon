
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

# 제목
st.title("Bike Sharing 시계열 예측")

# 파일 업로드
uploaded = st.file_uploader("Bike_Sharing_Demand.csv를 업로드하세요", type="csv")

if uploaded is not None:
    # DataFrame 불러오기
    bike = pd.read_csv(uploaded)

    # datetime 변환
    bike["datetime"] = pd.to_datetime(bike["datetime"])

    # 시간순 정렬 후 월별 count 합계
    bike = bike.sort_values("datetime").reset_index(drop=True)
    bike["month"] = bike["datetime"].dt.to_period("M").dt.to_timestamp()
    monthly = bike.groupby("month", as_index=False)["count"].sum()
    monthly = monthly.set_index("month")

    # 월별 데이터 표 + 선 그래프
    st.subheader("월별 count 합계")
    st.dataframe(monthly)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly["count"], label="Monthly count")
    ax.set_xlabel("Month")
    ax.set_ylabel("Bike Rentals")
    ax.set_title("Monthly Bike Rentals")
    ax.legend()
    st.pyplot(fig)

    # 예측 기간 선택 (1, 3, 6개월)
    steps = st.selectbox("예측 기간(개월)", [1, 3, 6])

    # 전체 월별 count로 ARIMA(2, 1, 1) 학습
    model_fit = ARIMA(monthly["count"], order=(2, 1, 1)).fit()
    # 선택한 기간만큼 예측
    forecast = model_fit.forecast(steps=steps)

    # 마지막 관측 월의 다음 달부터 미래 날짜 생성
    future_dates = pd.date_range(
        start=monthly.index[-1] + pd.DateOffset(months=1),
        periods=steps,
        freq="MS")

    # 미래 날짜 + 예측값 표
    forecast_df = pd.DataFrame({"predicted": forecast.values}, index=future_dates)
    forecast_df.index.name = "month"
    st.subheader("미래 예측값")
    st.dataframe(forecast_df)

    # 실제값 + 미래 예측값 선 그래프
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(monthly.index, monthly["count"], label="Actual")
    ax2.plot(forecast_df.index, forecast_df["predicted"], label="Forecast")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Bike Rentals")
    ax2.set_title(f"ARIMA(2,1,1) Forecast ({steps} months)")
    ax2.legend()
    st.pyplot(fig2)
