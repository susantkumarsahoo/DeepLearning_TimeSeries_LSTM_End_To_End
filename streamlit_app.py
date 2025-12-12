import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Time Series Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Session state initialization
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'last_request' not in st.session_state:
    st.session_state.last_request = None


def check_api_health():
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None


def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def generate_predictions(start_date, end_date):
    """Call API to generate predictions"""
    try:
        payload = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }
        
        # Calculate expected time based on date range
        days_diff = (end_date - start_date).days
        estimated_seconds = max(5, days_diff * 0.15)  # Rough estimate
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(f"🔄 Generating predictions for {days_diff} days...")
        progress_bar.progress(10)
        
        # Make request with extended timeout
        timeout_seconds = max(300, days_diff * 2)  # At least 5 minutes, scale with days
        
        status_text.text(f"⏳ Processing... This may take {int(estimated_seconds)} seconds for large ranges")
        progress_bar.progress(30)
        
        import time
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json=payload,
                timeout=timeout_seconds
            )
            
            elapsed = time.time() - start_time
            progress_bar.progress(90)
            status_text.text(f"✅ Completed in {elapsed:.1f} seconds")
            
            time.sleep(0.5)  # Brief pause to show completion
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            if response.status_code == 200:
                return True, response.json()
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                return False, error_detail
                
        except requests.exceptions.Timeout:
            progress_bar.empty()
            status_text.empty()
            return False, f"Request timed out after {timeout_seconds} seconds. Try a smaller date range."
        except requests.exceptions.ConnectionError:
            progress_bar.empty()
            status_text.empty()
            return False, "Connection error. Make sure FastAPI is running on http://localhost:8000"
            
    except Exception as e:
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        return False, str(e)


def get_prediction_history():
    """Get list of previous predictions"""
    try:
        response = requests.get(f"{API_BASE_URL}/predictions/history", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def create_prediction_chart(df):
    """Create interactive prediction chart using Plotly"""
    fig = go.Figure()
    
    # Add prediction line
    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['predicted_value'],
        mode='lines+markers',
        name='Predictions',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4),
        hovertemplate='<b>Date</b>: %{x}<br><b>Value</b>: %{y:.2f}<extra></extra>'
    ))
    
    # Add moving average
    if len(df) > 24:
        df['ma_24h'] = df['predicted_value'].rolling(window=24).mean()
        fig.add_trace(go.Scatter(
            x=df['ds'],
            y=df['ma_24h'],
            mode='lines',
            name='24h Moving Avg',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            hovertemplate='<b>Date</b>: %{x}<br><b>MA</b>: %{y:.2f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title='Time Series Forecast',
        xaxis_title='Date',
        yaxis_title='Predicted Value',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_distribution_chart(df):
    """Create distribution histogram"""
    fig = px.histogram(
        df,
        x='predicted_value',
        nbins=50,
        title='Distribution of Predicted Values',
        labels={'predicted_value': 'Predicted Value', 'count': 'Frequency'},
        color_discrete_sequence=['#1f77b4']
    )
    
    fig.update_layout(
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig


def create_hourly_pattern_chart(df):
    """Create hourly pattern chart"""
    df['hour'] = pd.to_datetime(df['ds']).dt.hour
    hourly_avg = df.groupby('hour')['predicted_value'].mean().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=hourly_avg['hour'],
        y=hourly_avg['predicted_value'],
        marker_color='#2ca02c',
        hovertemplate='<b>Hour</b>: %{x}<br><b>Avg Value</b>: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Average Predicted Value by Hour',
        xaxis_title='Hour of Day',
        yaxis_title='Average Predicted Value',
        template='plotly_white',
        height=400,
        xaxis=dict(tickmode='linear', tick0=0, dtick=1)
    )
    
    return fig


def main():
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0 2rem 0;'>
            <h1 style='color: #1f77b4;'>📈 Time Series Forecasting Dashboard</h1>
            <p style='font-size: 1.2rem; color: #666;'>Generate and visualize LSTM-based predictions</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Health Status
        st.subheader("🏥 API Status")
        is_healthy, health_info = check_api_health()
        
        if is_healthy:
            st.success("✅ API is running")
            if health_info:
                st.info(f"Model: {health_info.get('model_status', 'unknown')}")
                st.info(f"Preprocessor: {health_info.get('preprocessor_status', 'unknown')}")
        else:
            st.error("❌ API is not reachable")
            st.warning("Please ensure the FastAPI server is running on http://localhost:8000")
            st.code("uvicorn app:app --reload --host 0.0.0.0 --port 8000")
            st.stop()
        
        st.divider()
        
        # Model Information
        st.subheader("🤖 Model Info")
        model_info = get_model_info()
        if model_info:
            st.success(f"✅ Model loaded" if model_info['model_exists'] else "❌ Model not found")
            st.success(f"✅ Preprocessor loaded" if model_info['preprocessor_exists'] else "❌ Preprocessor not found")
        
        st.divider()
        
        # Date Range Selection
        st.subheader("📅 Prediction Settings")
        
        # Initialize session state for dates if not exists
        if 'selected_start_date' not in st.session_state:
            st.session_state.selected_start_date = datetime.now().date()
        if 'selected_end_date' not in st.session_state:
            st.session_state.selected_end_date = datetime.now().date() + timedelta(days=30)
        
        # Quick date range buttons (at top for better UX)
        st.write("Quick Select:")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📅 7 Days", use_container_width=True):
                st.session_state.selected_start_date = datetime.now().date()
                st.session_state.selected_end_date = st.session_state.selected_start_date + timedelta(days=7)
                st.rerun()
        
        with col2:
            if st.button("📅 30 Days", use_container_width=True):
                st.session_state.selected_start_date = datetime.now().date()
                st.session_state.selected_end_date = st.session_state.selected_start_date + timedelta(days=30)
                st.rerun()
        
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("📅 60 Days", use_container_width=True):
                st.session_state.selected_start_date = datetime.now().date()
                st.session_state.selected_end_date = st.session_state.selected_start_date + timedelta(days=60)
                st.rerun()
        
        with col4:
            if st.button("📅 90 Days", use_container_width=True):
                st.session_state.selected_start_date = datetime.now().date()
                st.session_state.selected_end_date = st.session_state.selected_start_date + timedelta(days=90)
                st.rerun()
        
        st.write("") # Spacer
        
        # Date inputs (will update with button clicks)
        start_date = st.date_input(
            "Start Date",
            value=st.session_state.selected_start_date,
            help="Select the start date for predictions"
        )
        
        end_date = st.date_input(
            "End Date",
            value=st.session_state.selected_end_date,
            help="Select the end date for predictions"
        )
        
        # Update session state if manually changed
        if start_date != st.session_state.selected_start_date:
            st.session_state.selected_start_date = start_date
        if end_date != st.session_state.selected_end_date:
            st.session_state.selected_end_date = end_date
        
        st.divider()
        
        # Generate Predictions Button
        if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
            if start_date >= end_date:
                st.error("End date must be after start date!")
            else:
                days_diff = (end_date - start_date).days
                
                # Warning for large date ranges
                if days_diff > 90:
                    st.warning(f"⚠️ Large date range ({days_diff} days) will take longer to process.")
                    st.info("💡 Tip: For faster results, try smaller ranges (7-30 days)")
                
                success, result = generate_predictions(
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.min.time())
                )
                
                if success:
                    st.session_state.predictions = result
                    st.session_state.last_request = {
                        'start_date': start_date,
                        'end_date': end_date,
                        'timestamp': datetime.now()
                    }
                    st.success("✅ Predictions generated successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Error: {result}")
        
        st.divider()
        
        # Prediction History
        st.subheader("📜 History")
        history = get_prediction_history()
        if history and history.get('total_files', 0) > 0:
            st.info(f"Total predictions: {history['total_files']}")
            
            with st.expander("View History"):
                for file in history['files'][:5]:  # Show last 5
                    st.text(f"📄 {file['filename']}")
                    st.caption(f"Created: {file['modified'][:19]}")
        else:
            st.info("No prediction history yet")
    
    # Main Content Area
    if st.session_state.predictions:
        result = st.session_state.predictions
        
        # Success Message
        if st.session_state.last_request:
            req = st.session_state.last_request
            st.success(f"✅ Predictions generated from {req['start_date']} to {req['end_date']}")
        
        # Statistics Cards
        st.subheader("📊 Summary Statistics")
        
        stats = result['statistics']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Predictions", result['total_predictions'])
        
        with col2:
            st.metric("Mean Value", f"{stats['mean']:.2f}")
        
        with col3:
            st.metric("Std Deviation", f"{stats['std']:.2f}")
        
        with col4:
            st.metric("Min Value", f"{stats['min']:.2f}")
        
        with col5:
            st.metric("Max Value", f"{stats['max']:.2f}")
        
        st.divider()
        
        # Convert predictions to DataFrame
        df = pd.DataFrame(result['predictions'])
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "📊 Distribution", "⏰ Hourly Pattern", "📋 Data Table"])
        
        with tab1:
            st.plotly_chart(create_prediction_chart(df), use_container_width=True)
            
            # Date range info
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Start:** {result['date_range']['start']}")
            with col2:
                st.info(f"**End:** {result['date_range']['end']}")
        
        with tab2:
            st.plotly_chart(create_distribution_chart(df), use_container_width=True)
            
            # Quartile information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("25th Percentile", f"{stats['q25']:.2f}")
            with col2:
                st.metric("Median", f"{stats['median']:.2f}")
            with col3:
                st.metric("75th Percentile", f"{stats['q75']:.2f}")
        
        with tab3:
            st.plotly_chart(create_hourly_pattern_chart(df), use_container_width=True)
        
        with tab4:
            st.subheader("Prediction Data")
            
            # Search and filter
            col1, col2 = st.columns([3, 1])
            with col1:
                search_date = st.text_input("🔍 Search by date (YYYY-MM-DD)", "")
            with col2:
                num_rows = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
            
            # Filter data
            display_df = df.copy()
            if search_date:
                display_df = display_df[display_df['ds'].astype(str).str.contains(search_date)]
            
            # Display table
            st.dataframe(
                display_df[['ds', 'predicted_value']].head(num_rows),
                use_container_width=True,
                hide_index=True
            )
            
            st.info(f"Showing {min(num_rows, len(display_df))} of {len(display_df)} rows")
            
            # Download button
            csv = df[['ds', 'predicted_value']].to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions (CSV)",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        # Welcome message when no predictions yet
        st.info("👈 Configure your settings in the sidebar and click '🚀 Generate Predictions' to get started!")
        
        # Example usage
        with st.expander("ℹ️ How to use this dashboard"):
            st.markdown("""
            ### Steps to Generate Predictions:
            
            1. **Check API Status**: Ensure the API is running (green checkmark in sidebar)
            2. **Select Date Range**: Choose start and end dates for your predictions
            3. **Generate Predictions**: Click the blue button to start the prediction process
            4. **Explore Results**: View charts, statistics, and download your predictions
            
            ### Features:
            
            - 📈 **Interactive Time Series Chart**: Visualize predictions over time
            - 📊 **Distribution Analysis**: Understand the spread of predicted values
            - ⏰ **Hourly Patterns**: See average predictions by hour of day
            - 📋 **Data Table**: Search, filter, and download prediction data
            - 📜 **History**: Track previous prediction runs
            
            ### Tips:
            
            - Use quick select buttons for common date ranges (7, 30, 60, 90 days)
            - Larger date ranges will take longer to process
            - Maximum recommended range: 365 days
            """)
        
        # Quick start
        st.markdown("### 🚀 Quick Start")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **1. Select Dates**  
            Choose your prediction range in the sidebar
            """)
        
        with col2:
            st.markdown("""
            **2. Generate**  
            Click the Generate Predictions button
            """)
        
        with col3:
            st.markdown("""
            **3. Analyze**  
            Explore charts and download results
            """)


if __name__ == "__main__":
    main()

# Run with: streamlit run streamlit_app.py

# Run with: streamlit run streamlit_app.py