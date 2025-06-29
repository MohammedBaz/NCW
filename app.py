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
    st.title("🐐 Predictive Carrying Capacity Calculator")
    st.markdown("This tool **forecasts** future environmental conditions to estimate the number of animals that can be sustainably released.")

    with st.spinner("Generating 10-year NDVI forecast... This is needed for the calculation."):
        daily_ndvi_df = df[['ndvi']].resample('D').mean().dropna()
        if len(daily_ndvi_df) < 2:
            st.error("Not enough historical NDVI data to generate a forecast for the calculator.")
            return
        ndvi_forecast, _ = run_forecast(daily_ndvi_df, 'ndvi', 10)

    TOTAL_AREA_KM2 = 2257
    species_data = {
        'Gazella arabica': {'baseline': 24, 'consumption': 4, 'limit_factor': 1.0},
        'Arabian Reem': {'baseline': 92, 'consumption': 0.5, 'limit_factor': 1.0},
        'Arabian oryx': {'baseline': 33, 'consumption': 0.2, 'limit_factor': 0.25}, # Limiting factor for Oryx
        'Nubian ibex': {'baseline': 34, 'consumption': 10, 'limit_factor': 1.0},
    }
    
    st.sidebar.header("Calculator Controls")
    last_hist_year = df.index.year.max()
    future_years = [last_hist_year + i for i in range(1, 11)]
    selected_year = st.sidebar.selectbox("Select a future year for recommendation:", options=future_years)
    
    hist_annual_means = df['ndvi'].resample('YE').mean()
    min_hist_ndvi, max_hist_ndvi = hist_annual_means.min(), hist_annual_means.max()

    forecast_df_indexed = ndvi_forecast.set_index('ds')
    predicted_annual_mean_df = forecast_df_indexed[forecast_df_indexed.index.year == selected_year]
    
    if predicted_annual_mean_df.empty:
        st.error(f"Could not retrieve forecast for the year {selected_year}.")
        return
        
    predicted_year_ndvi = predicted_annual_mean_df['yhat'].mean()

    ndvi_ratio = (predicted_year_ndvi - min_hist_ndvi) / (max_hist_ndvi - min_hist_ndvi) * 100
    if ndvi_ratio > 60: scenario, color = "High", "green"
    elif 41 <= ndvi_ratio <= 60: scenario, color = "Medium", "orange"
    elif 20 <= ndvi_ratio <= 40: scenario, color = "Low", "red"
    else: scenario, color = "Very Low", "darkred"

    effective_green_area = TOTAL_AREA_KM2 * predicted_year_ndvi
    
    st.header(f"Predicted Scenario for {selected_year}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Annual Average NDVI", f"{predicted_year_ndvi:.4f}")
    col2.metric("Predicted Vegetation Scenario", scenario)
    st.markdown(f"**Predicted Scenario:** <span style='color:{color}; font-weight:bold;'>{scenario}</span> ({ndvi_ratio:.1f}% of historical maximum productivity)", unsafe_allow_html=True)

    st.subheader(f"Recommended Animal Release Plan for {selected_year+1}")
    results = []
    for species, data in species_data.items():
        max_capacity = (effective_green_area / data['consumption']) * data['limit_factor'] if data['consumption'] > 0 else 0
        
        if scenario == "High": release_count = max(0, math.floor(max_capacity - data['baseline']))
        elif scenario == "Medium": release_count = max(0, math.floor(0.5 * (max_capacity - data['baseline'])))
        else: release_count = 0
            
        results.append({
            "Species": species,
            "Current Population (Baseline)": data['baseline'],
            "Predicted Max Population": math.floor(max_capacity),
            "Recommended Release": release_count
        })
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # --- 5-Year Predicted Population Graph ---
    st.subheader("5-Year Population Forecast")
    five_year_data = []
    for year in future_years[:5]:
        pred_ndvi_for_year = forecast_df_indexed[forecast_df_indexed.index.year == year]['yhat'].mean()
        eff_area = TOTAL_AREA_KM2 * pred_ndvi_for_year
        for species, data in species_data.items():
            max_pop = (eff_area / data['consumption']) * data['limit_factor'] if data['consumption'] > 0 else 0
            five_year_data.append({
                'Year': year,
                'Species': species,
                'Predicted Max Population': math.floor(max_pop)
            })
    five_year_df = pd.DataFrame(five_year_data)

    fig_5_year = px.bar(five_year_df, x='Year', y='Predicted Max Population', color='Species',
                        title='Predicted Maximum Population by Species (Next 5 Years)',
                        labels={'Predicted Max Population': 'Max Population', 'Year': 'Year'},
                        barmode='group')
    st.plotly_chart(fig_5_year, use_container_width=True)

    
    with st.expander("ℹ️ See Explanation of Methodology"):
        st.markdown(f"""
        The recommendation is based on a **predictive model**:
        1.  **NDVI Forecast:** A 10-year forecast for NDVI is generated.
        2.  **Predicted Green Cover:** The average **predicted** NDVI for the selected year is used as a proxy for future vegetation health.
        3.  **Predicted Forage Area:** The total area ({TOTAL_AREA_KM2} km²) is multiplied by the *predicted NDVI value*.
        4.  **Species Limiting Factor:** A species-specific factor is applied to the calculation. For the **Arabian Oryx**, this is set to **0.25** to account for other habitat constraints, ensuring more realistic numbers.
        5.  **Predicted Max Capacity:** The forage area is divided by consumption rate and multiplied by the limiting factor to find the sustainable population.
        6.  **Release Scenario:**
            - **High (>60% productivity):** Release up to the predicted maximum capacity.
            - **Medium (41-60%):** Release up to 50% of the difference between predicted max capacity and baseline.
            - **Low/Very Low (<41%):** No releases are recommended.
        """)

# --- MAIN APP LOGIC ---
df = load_data()

if df is not None:
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose a view",
        ["Data Visualization", "Forecasting", "Carrying Capacity Calculator"]
    )
    st.sidebar.divider()
    available_cols = df.select_dtypes(include=['number']).columns.tolist()

    if app_mode == "Data Visualization": render_visualization_ui(df, available_cols)
    elif app_mode == "Forecasting": render_forecasting_ui(df, available_cols)
    elif app_mode == "Carrying Capacity Calculator": render_capacity_calculator_ui(df)
else:
    st.title("🌿 Sharaan Environmental Dashboard")
    st.error("Failed to load data. Please check the GitHub URL and your internet connection.")
