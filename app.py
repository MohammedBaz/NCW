import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import math

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
        xaxis_title='Date', yaxis_title='Predicted Value'
    )
    return forecast, fig

# --- UI RENDERING FUNCTIONS ---
def render_visualization_ui(df, available_columns):
    st.sidebar.header("Visualization Controls")
    selected_variables = st.sidebar.multiselect(
        'Select variables for time-series plot:', options=available_columns,
        default=['air_temp_c', 'relative_humidity', 'ndvi']
    )
    resample_freq = st.sidebar.selectbox("View trends by:", options=['Hourly (raw)', 'Daily', 'Weekly', 'Monthly'], index=1)
    st.title("📊 Data Visualization & Analysis")
    st.markdown("Analyze historical trends and relationships in the environmental data.")
    freq_map = {'Hourly (raw)': 'H', 'Daily': 'D', 'Weekly': 'W', 'Monthly': 'ME'}
    resampled_df = df[available_columns].resample(freq_map[resample_freq]).mean() if resample_freq != 'Hourly (raw)' else df
    st.header("Key Performance Indicators (Overall Period)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(label="Avg. Air Temp (°C)", value=f"{df['air_temp_c'].mean():.1f}")
    kpi2.metric(label="Total Precipitation (mm)", value=f"{df['precipitation_mm'].sum():.1f}")
    kpi3.metric(label="Max Wind Speed (m/s)", value=f"{df['wind_speed_ms'].max():.1f}")
    kpi4.metric(label="Latest NDVI", value=f"{df['ndvi'].iloc[-1]:.3f}")
    st.divider()
    st.header(f"Time-Series Analysis ({resample_freq} Averages)")
    if not selected_variables: st.warning("Please select at least one variable to plot.")
    else:
        fig = px.line(resampled_df, y=selected_variables, title=f'Trend of Selected Variables ({resample_freq})', labels={'value': 'Value', 'date_time_utc': 'Date'}, template='seaborn')
        fig.update_layout(legend_title_text='Variables', xaxis_title="Date", yaxis_title="Average Values")
        st.plotly_chart(fig, use_container_width=True)
    st.header("In-Depth Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary Statistics", "🔗 Correlation Heatmap", "📅 Seasonal Distribution", "📋 Raw Data"])
    with tab1: st.dataframe(df[available_columns].describe(), use_container_width=True)
    with tab2:
        corr_matrix = df[available_columns].corr()
        corr_fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", labels=dict(color="Correlation"), color_continuous_scale='RdBu_r')
        st.plotly_chart(corr_fig, use_container_width=True)
    with tab3:
        default_dist_ix = available_columns.index('air_temp_c') if 'air_temp_c' in available_columns else 0
        dist_var = st.selectbox("Select variable for distribution plot:", options=available_columns, index=default_dist_ix)
        dist_df = df.copy(); dist_df['month'] = dist_df.index.strftime('%B')
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        box_fig = px.box(dist_df, x='month', y=dist_var, title=f"Monthly Distribution of {dist_var}", category_orders={"month": month_order}, labels={'month': 'Month', dist_var: dist_var.replace('_', ' ').title()}, template='seaborn')
        st.plotly_chart(box_fig, use_container_width=True)
    with tab4: st.dataframe(df, use_container_width=True)

def render_forecasting_ui(df, available_columns):
    st.sidebar.header("Forecasting Controls")
    default_fc_ix = available_columns.index('air_temp_c') if 'air_temp_c' in available_columns else 0
    forecast_var = st.sidebar.selectbox("Select variable to forecast:", options=available_columns, index=default_fc_ix)
    st.title("🔮 Future Forecasting")
    st.markdown("Select a variable to generate a 10-year forecast using the Prophet model.")
    if st.button(f"Generate 10-Year Forecast for {forecast_var}"):
        with st.spinner("Training model and generating forecast..."):
            daily_df = df[[forecast_var]].resample('D').mean().dropna()
            if len(daily_df) < 2: st.error("Not enough historical data to generate a forecast.")
            else:
                forecast_data, forecast_fig = run_forecast(daily_df, forecast_var, 10)
                st.plotly_chart(forecast_fig, use_container_width=True)
                st.subheader("Forecast Data")
                st.dataframe(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(365), use_container_width=True)

def render_capacity_calculator_ui(df):
    st.title("🐐 Carrying Capacity Calculator")
    st.markdown("This tool estimates the number of animals that can be sustainably released based on the annual green cover, proxied by the average NDVI.")
    
    # --- Species Data and Constants ---
    TOTAL_AREA_KM2 = 2257
    species_data = {
        'Gazella arabica': {'baseline': 24, 'consumption': 4},
        'Arabian Reem': {'baseline': 92, 'consumption': 0.5},
        'Arabian oryx': {'baseline': 33, 'consumption': 0.2},
        'Nubian ibex': {'baseline': 34, 'consumption': 10},
    }
    
    # --- Sidebar Controls ---
    st.sidebar.header("Calculator Controls")
    available_years = sorted(df.index.year.unique())
    selected_year = st.sidebar.selectbox("Select year for analysis:", options=available_years, index=len(available_years)-1)
    
    # --- Calculations ---
    annual_means = df['ndvi'].resample('YE').mean()
    min_hist_ndvi, max_hist_ndvi = annual_means.min(), annual_means.max()
    current_year_ndvi = annual_means[annual_means.index.year == selected_year].values[0]
    
    # Determine scenario
    ndvi_ratio = (current_year_ndvi - min_hist_ndvi) / (max_hist_ndvi - min_hist_ndvi) * 100
    if ndvi_ratio > 60: scenario, color = "High", "green"
    elif 41 <= ndvi_ratio <= 60: scenario, color = "Medium", "orange"
    elif 20 <= ndvi_ratio <= 40: scenario, color = "Low", "red"
    else: scenario, color = "Very Low", "darkred"

    # Effective green area calculation
    effective_green_area = TOTAL_AREA_KM2 * (current_year_ndvi / max_hist_ndvi)
    
    # --- Display Results ---
    st.header(f"Analysis for {selected_year}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Annual Average NDVI", f"{current_year_ndvi:.4f}")
    col2.metric("Vegetation Scenario", scenario)
    st.markdown(f"**Scenario:** <span style='color:{color}; font-weight:bold;'>{scenario}</span> ({ndvi_ratio:.1f}% of historical maximum productivity)", unsafe_allow_html=True)

    st.subheader("Recommended Animal Release Plan")
    results = []
    for species, data in species_data.items():
        max_capacity = effective_green_area / data['consumption'] if data['consumption'] > 0 else 0
        
        if scenario == "High": release_count = max(0, math.floor(max_capacity - data['baseline']))
        elif scenario == "Medium": release_count = max(0, math.floor(0.5 * (max_capacity - data['baseline'])))
        else: release_count = 0 # Low or Very Low scenario
            
        results.append({
            "Species": species,
            "Current Population (Baseline)": data['baseline'],
            "Max Sustainable Population": math.floor(max_capacity),
            "Recommended Release for Next Year": release_count
        })
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    with st.expander("ℹ️ See Explanation of Methodology"):
        st.markdown(f"""
        The recommendation is based on the following model:
        1.  **Total Area:** The Sharaan Nature Reserve is assumed to be **{TOTAL_AREA_KM2} km²**.
        2.  **Annual Green Cover:** The average NDVI for the selected year ({current_year_ndvi:.4f}) is used as a proxy for vegetation health.
        3.  **Effective Forage Area:** The total area is multiplied by the ratio of the current year's NDVI to the historical maximum annual NDVI. For {selected_year}, this results in an effective green area of **{effective_green_area:,.0f} km²**.
        4.  **Max Capacity:** The effective area is divided by each species' consumption rate to find the maximum sustainable population.
        5.  **Release Scenario:**
            - **High (>60% productivity):** Release up to the maximum calculated capacity.
            - **Medium (41-60%):** Release up to 50% of the difference between max capacity and baseline.
            - **Low/Very Low (<41%):** No releases are recommended to allow the environment to recover.
        """)

# --- MAIN APP LOGIC ---
df = load_data()

if df is not None:
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose a view",
        ["Data Visualization", "Carrying Capacity Calculator", "Forecasting"]
    )
    st.sidebar.divider()
    available_cols = df.select_dtypes(include=['number']).columns.tolist()

    if app_mode == "Data Visualization": render_visualization_ui(df, available_cols)
    elif app_mode == "Forecasting": render_forecasting_ui(df, available_cols)
    elif app_mode == "Carrying Capacity Calculator": render_capacity_calculator_ui(df)
else:
    st.title("🌿 Sharaan Environmental Dashboard")
    st.error("Failed to load data. Please check the GitHub URL and your internet connection.")
