import streamlit as st
import pandas as pd
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sharaan Environmental Data Explorer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_data(uploaded_file):
    """
    Loads data from the uploaded CSV file.
    It includes error handling for file format and required columns.
    Caches the data to avoid reloading on every interaction.
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # --- Data Cleaning and Type Conversion ---
        # Convert date column to datetime objects
        if 'date_time_utc' in df.columns:
            df['date_time_utc'] = pd.to_datetime(df['date_time_utc'])
        else:
            st.error("Error: The CSV must contain a 'date_time_utc' column.")
            return None
        
        # Set the datetime column as the index
        df.set_index('date_time_utc', inplace=True)

        # Drop non-essential or problematic columns
        if '.geo' in df.columns:
            df = df.drop(columns=['.geo'])
        if 'system:index' in df.columns:
            df = df.drop(columns=['system:index'])
            
        # Ensure all numeric columns are of float type for calculations
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with any NaN values that might have been created
        df.dropna(inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading or processing the file: {e}")
        return None

# --- APP TITLE AND DESCRIPTION ---
st.title("🌿 Sharaan Environmental Data Explorer")
st.markdown("""
This interactive application allows you to visualize and analyze the hourly environmental time-series data 
exported from Google Earth Engine for the Sharaan region.

**How to use:**
1.  **Upload your CSV file** using the sidebar on the left.
2.  Select the **variables you want to plot** from the multi-select box.
3.  The interactive charts, summary statistics, and data table will update automatically.
""")

# --- SIDEBAR FOR CONTROLS ---
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload your hourly data CSV", type=["csv"])

# --- MAIN PANEL ---
if uploaded_file is not None:
    df = load_data(uploaded_file)

    if df is not None:
        st.sidebar.success(f"Successfully loaded {uploaded_file.name} with {len(df)} rows.")
        
        # Get the list of available numeric columns for plotting
        available_columns = df.select_dtypes(include=['number']).columns.tolist()

        # --- Variable Selection ---
        st.sidebar.subheader("Plotting Options")
        selected_variables = st.sidebar.multiselect(
            'Select variables to plot:',
            options=available_columns,
            default=['air_temp_c', 'relative_humidity', 'ndvi'] # Sensible defaults
        )

        # --- VISUALIZATION SECTION ---
        st.header("Time-Series Visualization")

        if not selected_variables:
            st.warning("Please select at least one variable from the sidebar to plot.")
        else:
            # Create an interactive line chart
            fig = px.line(
                df,
                y=selected_variables,
                title='Interactive Time-Series Plot',
                labels={'value': 'Value', 'date_time_utc': 'Date and Time'},
                template='plotly_white'
            )
            fig.update_layout(
                legend_title_text='Variables',
                xaxis_title="Date/Time",
                yaxis_title="Values"
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- ANALYSIS SECTION ---
        st.header("Data Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📊 Summary Statistics", "🔗 Correlation Heatmap", "📋 Raw Data Table"])

        with tab1:
            st.subheader("Summary Statistics")
            st.markdown("Key statistical measures for each variable in the dataset.")
            st.dataframe(df[available_columns].describe(), use_container_width=True)

        with tab2:
            st.subheader("Correlation Heatmap")
            st.markdown("Shows the correlation between different variables. A value of 1 means a strong positive correlation, and -1 means a strong negative correlation.")
            
            # Calculate correlation matrix
            corr_matrix = df[available_columns].corr()
            
            # Create heatmap
            corr_fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                labels=dict(color="Correlation"),
                color_continuous_scale='RdBu_r', # Red-Blue diverging scale
                title="Variable Correlation Matrix"
            )
            st.plotly_chart(corr_fig, use_container_width=True)
            
        with tab3:
            st.subheader("Raw Data View")
            st.markdown(f"Displaying the full dataset with {len(df)} records.")
            st.dataframe(df, use_container_width=True)

else:
    st.info("Awaiting for a CSV file to be uploaded.")
