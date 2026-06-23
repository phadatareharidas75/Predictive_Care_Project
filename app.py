import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Predictive Care Analytics", layout="wide")

st.title("📊 Predictive Forecasting of Care Load & Placement Demand")
st.markdown("---")

st.sidebar.header("🎯 Navigation & Control Panel")
st.sidebar.write(f"Logged in as: **Haridas Phadtare**")

uploaded_file = st.sidebar.file_uploader("Upload your own Dataset (Excel or CSV)", type=["xlsx", "csv"])
forecast_days = st.sidebar.slider("Select Days to Forecast Future Demand", min_value=7, max_value=90, value=30)

if uploaded_file is not None:
    if uploaded_file.name.endswith('xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
        
    df.columns = df.columns.str.strip()
    
    if 'Date' not in df.columns:
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    if 'Actual_Care_Load' not in df.columns:
        df.rename(columns={df.columns[1]: 'Actual_Care_Load'}, inplace=True)
        
    df.dropna(subset=['Date', 'Actual_Care_Load'], inplace=True)
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Actual_Care_Load'] = pd.to_numeric(df['Actual_Care_Load'], errors='coerce')
    df.dropna(subset=['Date', 'Actual_Care_Load'], inplace=True)
    
    df['Actual_Care_Load'] = df['Actual_Care_Load'].astype(int)
    
    if 'Placement_Demand' not in df.columns:
        df['Placement_Demand'] = (df['Actual_Care_Load'] * 0.80).astype(int)
        
    st.sidebar.success("✅ Your Dataset Uploaded Successfully!")
else:
    st.sidebar.warning("Showing Sample Data. Please upload your file.")
    @st.cache_data
    def create_historical_data():
        start_date = datetime(2025, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(500)]
        np.random.seed(42)
        base_load = np.random.randint(50, 100, size=500)
        trend = np.arange(500) * 0.08
        final_load = (base_load + trend).astype(int)
        placement_demand = (final_load * 0.80).astype(int)
        return pd.DataFrame({"Date": dates, "Actual_Care_Load": final_load, "Placement_Demand": placement_demand})
    df = create_historical_data()

# Ensure dataframe is sorted by Date for proper graph plotting
df = df.sort_values('Date').reset_index(drop=True)

st.subheader("📌 Current Operational Status (Live Key Metrics)")

current_load = int(df["Actual_Care_Load"].iloc[-1])
current_placement = int(df["Placement_Demand"].iloc[-1])

col1, col2, col3 = st.columns(3)
col1.metric("Live Active Care Load", f"{current_load} Individuals")
col2.metric("Occupied Bed Placements", f"{current_placement} Beds")
col3.metric("System Risk Resource Threshold", "🚨 MONITORING", delta_color="inverse")

st.markdown("---")

st.subheader("🔮 Machine Learning Future Capacity Forecasting")

X = np.arange(len(df)).reshape(-1, 1)
y = df["Actual_Care_Load"].values

model = LinearRegression()
model.fit(X, y)

future_indices = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
future_predictions = model.predict(future_indices).astype(int)

last_date = df["Date"].iloc[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecasted_Care_Load": future_predictions,
    "Forecasted_Placement_Demand": (future_predictions * 0.80).astype(int)
})

# Complete Chart configuration
fig = px.line(df, x="Date", y="Actual_Care_Load", title="💡 Care Load Historical Trends vs. Future ML Forecast")
fig.add_scatter(x=forecast_df["Date"], y=forecast_df["Forecasted_Care_Load"], name="🔮 ML Predicted Care Load", mode="lines+markers", line=dict(color="red", dash="dash"))
fig.add_scatter(x=forecast_df["Date"], y=forecast_df["Forecasted_Placement_Demand"], name="🛏️ Predicted Bed Demand", mode="lines", line=dict(color="orange"))

st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 Forecasted Allocation Data Ledger")

# Displaying table and converting for viewing without breaks
display_df = forecast_df.copy()
display_df["Date"] = display_df["Date"].dt.strftime('%Y-%m-%d')
st.dataframe(display_df, use_container_width=True)

csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Export Forecast Report as CSV", data=csv, file_name="Care_Load_Future_Forecast.csv", mime="text/csv")