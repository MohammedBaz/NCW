import streamlit as st
import pandas as pd
import plotly.express as px

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

# --- LOAD DATA ---
df = load_data()

# --- SIDEBAR FOR CONTROLS ---
st.sidebar.title("Dashboard Controls")

if df is not None:
    # Get the list of available numeric columns for plotting
    available_columns = df.select_dtypes(include=['number']).columns.tolist()

    st.sidebar.header("Plotting Options")
    selected_variables = st.sidebar.multiselect(
        'Select variables for time-series plot:',
        options=available_columns,
        default=['air_temp_c', 'relative_humidity', 'ndvi']
    )
    
    st.sidebar.header("Data Resampling")
    resample_freq = st.sidebar.selectbox(
        "View trends by:",
        options=['Hourly (raw)', 'Daily', 'Weekly', 'Monthly'],
        index=1 # Default to Daily
    )

# --- MAIN PANEL ---
st.title("🌿 Sharaan Environmental Dashboard")
st.markdown("An interactive dashboard for analyzing environmental and vegetation data for the Sharaan region.")

if df is not None:
    # --- RESAMPLE DATA BASED ON SELECTION ---
    freq_map = {
        'Hourly (raw)': 'H',
        'Daily': 'D',
        'Weekly': 'W',
        'Monthly': 'ME'
    }
    if resample_freq != 'Hourly (raw)':
        resampled_df = df[available_columns].resample(freq_map[resample_freq]).mean()
    else:
        resampled_df = df

    # --- KPI METRICS ---
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

    # --- VISUALIZATION SECTION ---
    st.header(f"Time-Series Analysis ({resample_freq} Averages)")

    if not selected_variables:
        st.warning("Please select at least one variable from the sidebar to plot.")
    else:
        # Create an interactive line chart on the resampled data
        fig = px.line(
            resampled_df,
            y=selected_variables,
            title=f'Trend of Selected Variables ({resample_freq})',
            labels={'value': 'Value', 'date_time_utc': 'Date'},
            template='seaborn'
        )
        fig.update_layout(
            legend_title_text='Variables',
            xaxis_title="Date",
            yaxis_title="Average Values"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- ANALYSIS TABS ---
    st.header("In-Depth Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary Statistics", "🔗 Correlation Heatmap", "📅 Seasonal Distribution", "📋 Raw Data"])

    with tab1:
        st.subheader("Statistical Overview")
        st.markdown("Key statistical measures for each variable in the raw dataset.")
        st.dataframe(df[available_columns].describe(), use_container_width=True)

    with tab2:
        st.subheader("Variable Correlation Heatmap")
        st.markdown("Shows how variables are related. A value near 1 or -1 indicates a strong correlation.")
        corr_matrix = df[available_columns].corr()
        corr_fig = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto",
            labels=dict(color="Correlation"), color_continuous_scale='RdBu_r',
            title="Variable Correlation Matrix"
        )
        st.plotly_chart(corr_fig, use_container_width=True)
            
    with tab3:
        st.subheader("Seasonal Distribution Analysis")
        st.markdown("Explore the distribution of a variable across different months to identify seasonal patterns.")
        dist_var = st.selectbox("Select variable for distribution plot:", options=available_columns, index=available_columns.index('air_temp_c'))
        
        dist_df = df.copy()
        dist_df['month'] = dist_df.index.strftime('%B')
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        
        box_fig = px.box(
            dist_df,
            x='month',
            y=dist_var,
            title=f"Monthly Distribution of {dist_var}",
            category_orders={"month": month_order},
            labels={'month': 'Month', dist_var: dist_var.replace('_', ' ').title()},
            template='seaborn'
        )
        st.plotly_chart(box_fig, use_container_width=True)
        
    with tab4:
        st.subheader("Complete Raw Hourly Data")
        st.markdown(f"Displaying the full dataset with {len(df)} records.")
        st.dataframe(df, use_container_width=True)

else:
    st.error("Failed to load data. Please check the GitHub URL and your internet connection.")
