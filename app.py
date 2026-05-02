import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")

st.title("📊 Sales Forecasting Dashboard (SARIMA)")
st.write("Forecast future sales using time-series modeling")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("clean_sales_data.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data.asfreq('D')
    data['Sales'] = data['Sales'].fillna(method='ffill')
    return data

data = load_data()

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Model Settings")

p = st.sidebar.slider("AR (p)", 0, 5, 1)
d = st.sidebar.slider("I (d)", 0, 2, 1)
q = st.sidebar.slider("MA (q)", 0, 5, 1)

P = st.sidebar.slider("Seasonal AR (P)", 0, 2, 1)
D = st.sidebar.slider("Seasonal I (D)", 0, 2, 1)
Q = st.sidebar.slider("Seasonal MA (Q)", 0, 2, 1)

season_length = st.sidebar.selectbox("Season Length", [7, 12, 30])

forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 30)

# -----------------------------
# Train-Test Split
# -----------------------------
train_size = int(0.8 * len(data))
train = data[:train_size]
test = data[train_size:]

# -----------------------------
# Model Training
# -----------------------------
st.subheader("📈 Model Training")

model = SARIMAX(train['Sales'],
                order=(p, d, q),
                seasonal_order=(P, D, Q, season_length))

model_fit = model.fit(disp=False)

# -----------------------------
# Forecast
# -----------------------------
forecast = model_fit.forecast(steps=len(test))

# -----------------------------
# Metrics
# -----------------------------
mae = mean_absolute_error(test['Sales'], forecast)

col1, col2 = st.columns(2)
col1.metric("📉 MAE", round(mae, 2))
col2.metric("📊 Data Points", len(data))

# -----------------------------
# Plot: Train vs Test vs Forecast
# -----------------------------
st.subheader("📊 Sales Forecast Visualization")

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(train.index, train['Sales'], label='Train', color='blue')
ax.plot(test.index, test['Sales'], label='Actual', color='green')
ax.plot(test.index, forecast, label='Forecast', linestyle='dashed', color='red')

ax.set_title("SARIMA Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# -----------------------------
# Future Prediction
# -----------------------------
st.subheader("🔮 Future Prediction")

future = model_fit.forecast(steps=forecast_days)

fig2, ax2 = plt.subplots(figsize=(12,6))

ax2.plot(data.index, data['Sales'], label='Historical', color='blue')
ax2.plot(future.index, future, label='Future Forecast', linestyle='dashed', color='red')

ax2.set_title(f"Next {forecast_days} Days Forecast")
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)

# -----------------------------
# Show Data
# -----------------------------
st.subheader("📋 Dataset Preview")
st.dataframe(data.tail(20))

# -----------------------------
# Download Forecast
# -----------------------------
future_df = future.reset_index()
future_df.columns = ['Date', 'Predicted_Sales']

csv = future_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="⬇️ Download Forecast CSV",
    data=csv,
    file_name="future_sales_forecast.csv",
    mime="text/csv"
)