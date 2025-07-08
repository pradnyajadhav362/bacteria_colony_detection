# admin_dashboard.py
# admin web interface for viewing all user uploads and analysis results
# provides comprehensive data access and export capabilities

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from admin_logger import admin_logger
import json
from datetime import datetime, timedelta
from pathlib import Path

def admin_login():
    # simple admin authentication
    st.sidebar.header("Admin Access")
    
    admin_password = st.sidebar.text_input("Admin Password", type="password")
    
    # check if admin is logged in
    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False
    
    # Get admin password from environment variable for security
    import os
    correct_admin_password = os.getenv('ADMIN_PASSWORD', 'default_admin_123')
    
    if admin_password == correct_admin_password:
        st.session_state.admin_logged_in = True
        st.sidebar.success("Admin logged in")
        return True
    elif admin_password:
        st.sidebar.error("Invalid password")
        return False
    
    return st.session_state.admin_logged_in

def display_admin_overview():
    # main admin dashboard overview
    st.header("Admin Dashboard - All User Data")
    
    # get all session data
    all_sessions = admin_logger.get_all_sessions()
    
    if all_sessions.empty:
        st.info("No user sessions recorded yet")
        return
    
    # overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sessions", len(all_sessions))
    
    with col2:
        total_uploads = all_sessions['total_uploads'].sum()
        st.metric("Total Uploads", total_uploads)
    
    with col3:
        total_analyses = all_sessions['total_analyses'].sum()
        st.metric("Total Analyses", total_analyses)
    
    with col4:
        total_downloads = all_sessions['total_downloads'].sum()
        st.metric("Total Downloads", total_downloads)
    
    # usage over time
    st.subheader("Usage Over Time")
    
    # convert timestamp to datetime
    all_sessions['date'] = pd.to_datetime(all_sessions['timestamp']).dt.date
    daily_usage = all_sessions.groupby('date').agg({
        'session_id': 'count',
        'total_uploads': 'sum',
        'total_analyses': 'sum'
    }).reset_index()
    
    fig_usage = px.line(daily_usage, x='date', y=['session_id', 'total_uploads', 'total_analyses'],
                       title="Daily Usage Metrics")
    st.plotly_chart(fig_usage, use_container_width=True)
    
    # colony detection statistics
    st.subheader("Colony Detection Statistics")
    
    successful_analyses = all_sessions[all_sessions['analysis_successful'] == True]
    if not successful_analyses.empty:
        fig_colonies = px.histogram(successful_analyses, x='colony_count', nbins=20,
                                   title="Distribution of Colony Counts")
        st.plotly_chart(fig_colonies, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Colonies per Image", f"{successful_analyses['colony_count'].mean():.1f}")
        with col2:
            st.metric("Max Colonies Detected", successful_analyses['colony_count'].max())

def display_session_browser():
    # browse and view individual sessions
    st.header("Session Browser")
    
    all_sessions = admin_logger.get_all_sessions()
    
    if all_sessions.empty:
        st.info("No sessions to display")
        return
    
    # session selection
    st.subheader("All User Sessions")
    
    # add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_filter = st.date_input("Filter by date (optional)")
        if date_filter:
            all_sessions['date'] = pd.to_datetime(all_sessions['timestamp']).dt.date
            all_sessions = all_sessions[all_sessions['date'] >= date_filter]
    
    with col2:
        min_colonies = st.number_input("Min colonies", min_value=0, value=0)
        all_sessions = all_sessions[all_sessions['colony_count'] >= min_colonies]
    
    with col3:
        success_filter = st.selectbox("Analysis status", ["All", "Successful only", "Failed only"])
        if success_filter == "Successful only":
            all_sessions = all_sessions[all_sessions['analysis_successful'] == True]
        elif success_filter == "Failed only":
            all_sessions = all_sessions[all_sessions['analysis_successful'] == False]
    
    # display sessions table
    if not all_sessions.empty:
        # make table interactive
        selected_session = st.selectbox(
            "Select session to view details:",
            options=all_sessions['session_id'].tolist(),
            format_func=lambda x: f"{x} - {all_sessions[all_sessions['session_id']==x]['image_name'].iloc[0]} ({all_sessions[all_sessions['session_id']==x]['colony_count'].iloc[0]} colonies)"
        )
        
        # display session details
        if selected_session:
            display_session_details(selected_session)
        
        # display sessions table
        st.subheader("Sessions Table")
        st.dataframe(all_sessions, use_container_width=True)

def display_session_details(session_id):
    # show detailed information for specific session
    st.subheader(f"Session Details: {session_id}")
    
    details = admin_logger.get_session_details(session_id)
    
    if not details:
        st.error("Session not found")
        return
    
    # session timeline
    if 'logs' in details:
        st.write("**Session Timeline:**")
        for log in details['logs']:
            timestamp = log['timestamp']
            action = log['action']
            if action == 'upload':
                st.write(f"• {timestamp}: Uploaded {log['image_name']} ({log['file_size']} bytes)")
            elif action == 'analysis':
                st.write(f"• {timestamp}: Analysis completed - {log['colony_count']} colonies detected")
            elif action == 'download':
                st.write(f"• {timestamp}: Downloaded {log['download_type']}: {log['filename']}")
    
    # parameters used
    if 'parameters' in details:
        st.write("**Analysis Parameters:**")
        
        with st.expander("View all parameters"):
            st.json(details['parameters'])
        
        # key parameters summary
        params = details['parameters']
        key_params = {
            'bilateral_d': params.get('bilateral_d'),
            'gamma': params.get('gamma'),
            'min_colony_size': params.get('min_colony_size'),
            'max_colony_size': params.get('max_colony_size'),
            'adaptive_block_size': params.get('adaptive_block_size')
        }
        st.write("Key parameters:", key_params)
    
    # results summary
    if 'results_summary' in details:
        st.write("**Results Summary:**")
        results = details['results_summary']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Colonies Detected", results.get('colony_count', 0))
        with col2:
            st.metric("Has Morphology Data", "Yes" if results.get('has_morphology') else "No")
        with col3:
            st.metric("Analysis Success", "Yes" if results.get('analysis_successful') else "No")
    
    # colony data preview
    if 'colony_data' in details:
        st.write("**Colony Data:**")
        colony_df = details['colony_data']
        
        st.write(f"Found {len(colony_df)} colonies with detailed measurements")
        
        # show preview
        st.dataframe(colony_df.head(10), use_container_width=True)
        
        # download full data
        csv_data = colony_df.to_csv(index=False)
        st.download_button(
            label="Download Complete Colony Data CSV",
            data=csv_data,
            file_name=f"session_{session_id}_colony_data.csv",
            mime="text/csv"
        )
    
    # image files
    st.write("**Available Files:**")
    
    # check for uploaded image
    upload_path = Path("admin_logs/uploads")
    for img_file in upload_path.glob(f"{session_id}_*"):
        st.write(f"• Original upload: {img_file.name}")
        
        # display image
        try:
            from PIL import Image
            img = Image.open(img_file)
            st.image(img, caption="Original Upload", width=400)
        except:
            st.write("Could not display image")
    
    # check for processed image
    processed_path = Path("admin_logs/processed_images")
    for img_file in processed_path.glob(f"{session_id}_*"):
        st.write(f"• Processed image: {img_file.name}")
        
        try:
            from PIL import Image
            img = Image.open(img_file)
            st.image(img, caption="Processed Image", width=400)
        except:
            st.write("Could not display processed image")

def display_data_export():
    # data export and download options
    st.header("Data Export")
    
    st.write("Export all user data for analysis or backup.")
    
    # quick stats
    all_sessions = admin_logger.get_all_sessions()
    if not all_sessions.empty:
        st.write(f"**Available data:**")
        st.write(f"• {len(all_sessions)} user sessions")
        st.write(f"• {all_sessions['total_uploads'].sum()} uploaded images")
        st.write(f"• {all_sessions['total_analyses'].sum()} completed analyses")
        st.write(f"• {all_sessions[all_sessions['analysis_successful']]['colony_count'].sum()} total colonies detected")
    
    # export options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export All Data"):
            with st.spinner("Exporting all data..."):
                export_path = admin_logger.export_all_data()
                st.success(f"Data exported to: {export_path}")
                st.write("**Exported files:**")
                st.write("• admin_summary.csv - Session overview")
                st.write("• session_log.jsonl - Detailed logs")
                st.write("• all_colony_data.csv - Combined colony measurements")
                st.write("• uploads/ - All original images")
                st.write("• processed_images/ - All processed images")
                st.write("• parameters/ - All analysis parameters")
    
    with col2:
        # download summary csv directly
        if not all_sessions.empty:
            csv_data = all_sessions.to_csv(index=False)
            st.download_button(
                label="Download Session Summary CSV",
                data=csv_data,
                file_name=f"admin_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def main():
    st.set_page_config(
        page_title="Admin Dashboard",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("Bacterial Colony Analyzer - Admin Dashboard")
    
    # check admin login
    if not admin_login():
        st.warning("Please enter admin password to access dashboard")
        return
    
    # admin navigation
    tab1, tab2, tab3 = st.tabs(["Overview", "Session Browser", "Data Export"])
    
    with tab1:
        display_admin_overview()
    
    with tab2:
        display_session_browser()
    
    with tab3:
        display_data_export()

if __name__ == "__main__":
    main() 