<Yimport streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, value

# Page configuration
st.set_page_config(layout='wide')
st.title("📈 Demand Forecasting & Workforce Scheduling")

# Display local logo
st.markdown(
    """
    <div style='text-align: center;'>
        <a href='https://salasa.co' target='_blank'>
            <img src='logo.png' width='200'/>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Helper: Add Promotion Events as a Regressor ---
def add_promotion_factors(df):
    df['Promotion'] = 0
    for index, row in df.iterrows():
        if (row['ds'].month == 4 and row['ds'].year in [2023, 2024]) or            (row['ds'].month == 5 and row['ds'].year in [2020, 2021, 2022]) or            (row['ds'].month == 6 and row['ds'].year == 2019):
            df.at[index, 'Promotion'] = 1
        elif row['ds'].month == 9:
            df.at[index, 'Promotion'] = 1
        elif row['ds'].month == 2 and row['ds'].year >= 2022:
            df.at[index, 'Promotion'] = 1
        elif row['ds'].month == 11:
            df.at[index, 'Promotion'] = 1
        elif row['ds'].month == 12:
            df.at[index, 'Promotion'] = 1
    return df

# --- Upload Excel File ---
st.subheader("Upload Monthly Demand Excel File")
uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # --- Reshape the Data ---
    month_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df_long = df.melt(id_vars='Year', value_vars=month_cols,
                      var_name='Month', value_name='Demand')
    df_long['Date'] = pd.to_datetime(df_long['Year'].astype(str) + '-' + df_long['Month'], format='%Y-%b')
    df_long = df_long.sort_values('Date').reset_index(drop=True)
    df_long.set_index('Date', inplace=True)

    # --- Cross-validation for Forecasting ---
    initial_window = 36
    n_splits = 12
    actuals, sarima_preds, prophet_preds, hw_preds = [], [], [], []

    for i in range(n_splits):
        train_end = initial_window + i
        train = df_long.iloc[:train_end]
        test = df_long.iloc[train_end:train_end + 1]
        if len(test) == 0:
            break

        # SARIMA Forecast
        try:
            sarima_model = SARIMAX(train['Demand'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
            sarima_forecast = sarima_model.get_forecast(steps=1).predicted_mean.values[0]
        except:
            sarima_forecast = 0

        # Prophet Forecast
        df_prophet_train = train.reset_index().rename(columns={'Date': 'ds', 'Demand': 'y'})
        df_prophet_train['cap'] = df_prophet_train['y'].max() * 3
        df_prophet_train['floor'] = df_prophet_train['y'].min() * 0.5
        df_prophet_train['company_growth'] = df_prophet_train['ds'].dt.year - 2017
        df_prophet_train = add_promotion_factors(df_prophet_train)

        model_prophet = Prophet(growth='logistic', yearly_seasonality=True)
        model_prophet.add_regressor('company_growth')
        model_prophet.add_regressor('Promotion')
        model_prophet.fit(df_prophet_train[['ds', 'y', 'cap', 'floor', 'company_growth', 'Promotion']])

        future = model_prophet.make_future_dataframe(periods=1, freq='MS')
        future['cap'] = df_prophet_train['cap'].iloc[0]
        future['floor'] = df_prophet_train['floor'].iloc[0]
        future['company_growth'] = future['ds'].dt.year - 2017
        future = add_promotion_factors(future)
        prophet_forecast = model_prophet.predict(future)['yhat'].values[-1]

        # Holt-Winters Forecast
        try:
            hw_model = ExponentialSmoothing(train['Demand'], trend='add', seasonal='add', seasonal_periods=12).fit()
            hw_forecast = hw_model.forecast(1).values[0]
        except:
            hw_forecast = train['Demand'].mean()

        actuals.append(test['Demand'].values[0])
        sarima_preds.append(sarima_forecast)
        prophet_preds.append(prophet_forecast)
        hw_preds.append(hw_forecast)

    # --- Optimal Blending Weights ---
    best_mae = float('inf')
    best_weights = (1/3, 1/3, 1/3)
    for w1 in np.linspace(0, 1, 21):
        for w2 in np.linspace(0, 1 - w1, 21):
            w3 = 1 - w1 - w2
            blended = w1 * np.array(sarima_preds) + w2 * np.array(prophet_preds) + w3 * np.array(hw_preds)
            mae = mean_absolute_error(actuals, blended)
            if mae < best_mae:
                best_mae = mae
                best_weights = (w1, w2, w3)

    # --- Final Forecasts ---
    sarima_model = SARIMAX(df_long['Demand'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
    sarima_future = sarima_model.get_forecast(steps=12).predicted_mean
    future_index = pd.date_range(start=df_long.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
    sarima_future.index = future_index

    df_prophet = df_long.reset_index().rename(columns={'Date': 'ds', 'Demand': 'y'})
    df_prophet['cap'] = df_prophet['y'].max() * 3
    df_prophet['floor'] = df_prophet['y'].min() * 0.5
    df_prophet['company_growth'] = df_prophet['ds'].dt.year - 2017
    df_prophet = add_promotion_factors(df_prophet)

    model_prophet = Prophet(growth='logistic', yearly_seasonality=True)
    model_prophet.add_regressor('company_growth')
    model_prophet.add_regressor('Promotion')
    model_prophet.fit(df_prophet[['ds', 'y', 'cap', 'floor', 'company_growth', 'Promotion']])

    future = model_prophet.make_future_dataframe(periods=12, freq='MS')
    future['cap'] = df_prophet['cap'].iloc[0]
    future['floor'] = df_prophet['floor'].iloc[0]
    future['company_growth'] = future['ds'].dt.year - 2017
    future = add_promotion_factors(future)
    prophet_future = model_prophet.predict(future)['yhat'].values[-12:]

    hw_model_full = ExponentialSmoothing(df_long['Demand'], trend='add', seasonal='add', seasonal_periods=12).fit()
    hw_future = hw_model_full.forecast(12).values

    w1, w2, w3 = best_weights
    combined_forecast = w1 * sarima_future.values + w2 * prophet_future + w3 * hw_future

    # --- Workforce Optimization ---
    M = 12
    S = 3
    Productivity = 23
    Cost = 8.5
    Days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    Hours = [6, 6, 6]

    model = LpProblem("Workforce_Scheduling", LpMinimize)
    X = {(i, j): LpVariable(f"X_{i}_{j}", lowBound=0, cat='Integer') for i in range(M) for j in range(S)}
    model += lpSum(Cost * X[i, j] * Hours[j] * Days[i] for i in range(M) for j in range(S))

    for i in range(M):
        model += lpSum(Productivity * X[i, j] * Hours[j] * Days[i] for j in range(S)) >= combined_forecast[i]

    model.solve()

    # --- Display Results ---
    st.success(f"Optimal Weights: SARIMA={w1:.2f}, Prophet={w2:.2f}, HW={w3:.2f} | MAE (CV): {best_mae:.2f}")

    st.subheader("📈 2025 Forecast (Weighted Blend)")
    forecast_df = pd.DataFrame({
        'Month': [d.strftime('%b %Y') for d in future_index],
        'Forecasted Demand': combined_forecast,
        'Workers Required': [sum(value(X[i, j]) for j in range(S)) for i in range(M)]
    })
    st.dataframe(forecast_df)

    st.subheader("📉 Demand Forecast Plot")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_long.index, df_long['Demand'], label='Historical Demand', marker='o')
    ax.plot(future_index, combined_forecast, label='2025 Forecast (Blended)', marker='o')
    ax.set_title("Historical + 2025 Forecast")
    ax.set_ylabel("Demand")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
>