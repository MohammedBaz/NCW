import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sharaan Environmental Dashboard",
    page_icon="🌿",
    layout="wide",
)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    """
    Loads data directly from a GitHub URL.
    It includes error handling, data cleaning, and type conversion.
    Caches the data to avoid reloading on every interaction.
    """
    url = "https://raw.githubusercontent.com/MohammedBaz/NCW/refs/heads/main/Sharaan_Hourly_TimeSeries_1800day.csv"
    try:
        df = pd.read_csv(url)
        
        # --- Data Cleaning and Type Conversion ---
        if 'date_time_utc' in df.columns:
            df['date_time_utc'] = pd.to_datetime(df['date_time_utc'])
        else:
            st.error("CRITICAL ERROR: The data source must contain a 'date_time_utc' column.")
            return None
            
        df.set_index('date_time_utc', inplace=True)

        if '.geo' in df.columns:
            df = df.drop(columns=['.geo'])
        if 'system:index' in df.columns:
            df = df.drop(columns=['system:index'])
            
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading or processing data from GitHub: {e}")
        return None

# --- FORECASTING FUNCTION ---
@st.cache_data
def run_forecast(_data, variable_to_forecast, years_to_predict):
    """
    Runs a Prophet forecast on the selected variable.
    Caches the result to avoid re-computation.
    """
    prophet_df = _data.reset_index()[['date_time_utc', variable_to_forecast]].rename(
        columns={'date_time_utc': 'ds', variable_to_forecast: 'y'}
    )
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=365 * years_to_predict)
    forecast = model.predict(future)
    fig = plot_plotly(model, forecast)
    fig.update_layout(
        title=f'10-Year Forecast for {variable_to_forecast.replace("_", " ").title()}',
        xaxis_title='Date',
        yaxis_title='Predicted Value'
    )
    return forecast, fig

# --- MAIN APP LOGIC ---
df = load_data()

if df is not None:
    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose a view",
        ["Data Visualization", "Forecasting"]
    )
    st.sidebar.divider()

    available_cols = df.select_dtypes(include=['number']).columns.tolist()

    # --- View 1: Data Visualization ---
    if app_mode == "Data Visualization":
        st.sidebar.header("Visualization Controls")
        selected_variables = st.sidebar.multiselect(
            'Select variables for time-series plot:',
            options=available_cols,
            default=['air_temp_c', 'relative_humidity', 'ndvi']
        )
        resample_freq = st.sidebar.selectbox(
            "View trends by:",
            options=['Hourly (raw)', 'Daily', 'Weekly', 'Monthly'],
            index=1 
        )

        st.title("📊 Data Visualization & Analysis")
        st.markdown("Analyze historical trends and relationships in the environmental data.")

        freq_map = {'Hourly (raw)': 'H', 'Daily': 'D', 'Weekly': 'W', 'Monthly': 'ME'}
        resampled_df = df[available_cols].resample(freq_map[resample_freq]).mean() if resample_freq != 'Hourly (raw)' else df

        st.header("Key Performance Indicators (Overall Period)")
        avg_temp = df['air_temp_c'].mean()
        total_precip = df['precipitation_mm'].sum()
        max_wind = df['wind_speed_ms'].max()
        latest_ndvi = df['ndvi'].iloc[-1]

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(label="Avg. Air Temp (°C)", value=f"{avg_temp:.1f}")
        kpi2.metric(label="Total Precipitation (mm)", value=f"{total_precip:.1f}")
        kpi3.metric(label="Max Wind Speed (m/s)", value=f"{max_wind:.1f}")
        kpi4.metric(label="Latest NDVI", value=f"{latest_ndvi:.3f}")
        st.divider()

        st.header(f"Time-Series Analysis ({resample_freq} Averages)")
        if not selected_variables:
            st.warning("Please select at least one variable to plot.")
        else:
            fig = px.line(resampled_df, y=selected_variables, title=f'Trend of Selected Variables ({resample_freq})', labels={'value': 'Value', 'date_time_utc': 'Date'}, template='seaborn')
            fig.update_layout(legend_title_text='Variables', xaxis_title="Date", yaxis_title="Average Values")
            st.plotly_chart(fig, use_container_width=True)

        st.header("In-Depth Analysis")
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary Statistics", "🔗 Correlation Heatmap", "📅 Seasonal Distribution", "📋 Raw Data"])

        with tab1:
            st.subheader("Statistical Overview")
            st.dataframe(df[available_cols].describe(), use_container_width=True)
        with tab2:
            st.subheader("Variable Correlation Heatmap")
            corr_matrix = df[available_cols].corr()
            corr_fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", labels=dict(color="Correlation"), color_continuous_scale='RdBu_r', title="Variable Correlation Matrix")
            st.plotly_chart(corr_fig, use_container_width=True)
        with tab3:
            st.subheader("Seasonal Distribution Analysis")
            # *** FIX IS HERE: Made the default index selection robust ***
            default_dist_ix = 0
            if 'air_temp_c' in available_cols:
                default_dist_ix = available_cols.index('air_temp_c')
            dist_var = st.selectbox("Select variable for distribution plot:", options=available_cols, index=default_dist_ix)
            
            dist_df = df.copy()
            dist_df['month'] = dist_df.index.strftime('%B')
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            box_fig = px.box(dist_df, x='month', y=dist_var, title=f"Monthly Distribution of {dist_var}", category_orders={"month": month_order}, labels={'month': 'Month', dist_var: dist_var.replace('_', ' ').title()}, template='seaborn')
            st.plotly_chart(box_fig, use_container_width=True)
        with tab4:
            st.subheader("Complete Raw Hourly Data")
            st.dataframe(df, use_container_width=True)

    # --- View 2: Forecasting ---
    elif app_mode == "Forecasting":
        st.sidebar.header("Forecasting Controls")
        default_fc_ix = 0
        if 'air_temp_c' in available_cols:
            default_fc_ix = available_cols.index('air_temp_c')
        
        forecast_var = st.sidebar.selectbox(
            "Select variable to forecast:", options=available_cols,
            index=default_fc_ix
        )
        
        st.title("🔮 Future Forecasting")
        st.markdown("Select a variable to generate a 10-year forecast using the Prophet model. The model is trained on the daily average of the historical data.")

        if st.button(f"Generate 10-Year Forecast for {forecast_var}"):
            with st.spinner("Training model and generating forecast... This may take a minute."):
                daily_df = df[[forecast_var]].resample('D').mean().dropna()
                
                if len(daily_df) < 2:
                    st.error("Not enough historical data to generate a forecast for this variable. Please choose another.")
                else:
                    forecast_data, forecast_fig = run_forecast(daily_df, forecast_var, 10)
                    st.plotly_chart(forecast_fig, use_container_width=True)
                    st.subheader("Forecast Data")
                    st.markdown("The table below shows the predicted value (`yhat`), along with the lower and upper uncertainty intervals (`yhat_lower`, `yhat_upper`).")
                    st.dataframe(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(365), use_container_width=True)

else:
    st.title("🌿 Sharaan Environmental Dashboard")
    st.error("Failed to load data. Please check the GitHub URL and your internet connection.")
