# app.py
# bacterial colony analysis visualization app
# provides interactive interface for analyzing petri dish images

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import io
import base64
from colony_analyzer import ColonyAnalyzer
from auth import init_auth

st.set_page_config(
    page_title="Bacterial Colony Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Initialize authentication
    auth = init_auth()
    
    st.title("🦠 Bacterial Colony Analyzer")
    st.caption("Advanced image analysis for petri dish colony detection and characterization")
    
    # sidebar for parameters
    with st.sidebar:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose a petri dish image", 
                                        type=['png', 'jpg', 'jpeg'],
                                        help="Upload a high-resolution image of a petri dish")
        
        st.header("Analysis Parameters")
        st.caption("Adjust image processing settings")
        
        bilateral_d = st.slider("Bilateral filter diameter", 3, 21, 9, step=2, 
                               help="Diameter of pixel neighborhood for noise reduction (higher = smoother)")
        bilateral_sigma_color = st.slider("Bilateral sigmaColor", 10, 150, 75,
                                         help="Color space sigma for noise filtering (higher = more blur)")
        bilateral_sigma_space = st.slider("Bilateral sigmaSpace", 10, 150, 75,
                                         help="Coordinate space sigma for noise filtering (higher = more blur)")
        clahe_clip_limit = st.slider("CLAHE clip limit", 1.0, 10.0, 3.0,
                                    help="Contrast enhancement limit (higher = more contrast)")
        clahe_tile_grid = st.slider("CLAHE tile grid size", 2, 32, 8,
                                   help="Grid size for contrast enhancement (smaller = more local)")
        gamma = st.slider("Gamma correction", 0.5, 2.5, 1.2,
                         help="Brightness adjustment (1.0 = normal, <1 = brighter, >1 = darker)")
        sharpen_strength = st.slider("Sharpen strength", 0.0, 2.0, 1.0,
                                    help="Edge sharpening intensity (0 = no sharpening)")
        
        st.header("Colony Detection")
        st.caption("Configure colony segmentation parameters")
        
        margin_percent = st.slider("Plate margin percent", 0.05, 0.20, 0.08, 0.01,
                                  help="Percentage of image edges to exclude from plate detection")
        
        min_colony_size = st.slider("Min colony size", 10, 50, 15,
                                   help="Minimum area in pixels for a colony to be considered valid")
        max_colony_size = st.slider("Max colony size", 5000, 20000, 10000,
                                   help="Maximum area in pixels for a colony to be considered valid")
        adaptive_block_size = st.slider("Adaptive threshold block size", 11, 25, 15, step=2,
                                       help="Block size for adaptive thresholding (must be odd)")
        adaptive_c = st.slider("Adaptive threshold C", 1, 10, 3,
                              help="Constant subtracted from mean for adaptive thresholding")
        watershed_min_distance = st.slider("Watershed min distance", 5, 15, 8,
                                          help="Minimum distance between colony centers for watershed")
        watershed_threshold = st.slider("Watershed threshold", 0.1, 0.5, 0.3, 0.05,
                                       help="Threshold for watershed peak detection")
        
        st.header("Color Analysis")
        st.caption("Configure color clustering parameters")
        
        color_n_clusters = st.number_input("Number of color clusters (0=auto)", 0, 10, 0,
                                          help="Number of color groups (0 = automatic detection)")
        color_random_state = st.number_input("KMeans random state", 0, 100, 42,
                                            help="Random seed for consistent color clustering")
        color_n_init = st.slider("KMeans n_init", 1, 20, 10,
                                help="Number of times to run K-means with different seeds")
        
        st.header("Top Colonies & Scoring")
        st.caption("Select and rank the most interesting colonies")
        
        n_top_colonies = st.slider("Number of top colonies to display", 1, 50, 20,
                                  help="How many highest-scoring colonies to show")
        penalty_factor = st.slider("Penalty factor for diversity", 0.0, 1.0, 0.5,
                                  help="Reduce score for similar colonies (higher = more diverse selection)")
        
        st.header("Analysis Options")
        st.caption("Choose which analyses to run")
        
        run_morphology = st.checkbox("Analyze morphology", value=True,
                                    help="Measure colony shape, size, and edge characteristics")
        run_color_analysis = st.checkbox("Analyze colors", value=True,
                                        help="Extract and cluster colony colors")
        run_density_analysis = st.checkbox("Analyze density", value=True,
                                          help="Measure colony opacity and texture")
        
        # Store current parameters in session state
        current_params = dict(
                    bilateral_d=bilateral_d,
                    bilateral_sigma_color=bilateral_sigma_color,
                    bilateral_sigma_space=bilateral_sigma_space,
                    clahe_clip_limit=clahe_clip_limit,
                    clahe_tile_grid=(clahe_tile_grid, clahe_tile_grid),
                    gamma=gamma,
                    sharpen_strength=sharpen_strength,
                    margin_percent=margin_percent,
                    adaptive_block_size=adaptive_block_size,
                    adaptive_c=adaptive_c,
                    min_colony_size=min_colony_size,
                    max_colony_size=max_colony_size,
                    watershed_min_distance=watershed_min_distance,
                    watershed_threshold=watershed_threshold,
                    color_n_clusters=(color_n_clusters if color_n_clusters > 0 else None),
                    color_random_state=color_random_state,
                    color_n_init=color_n_init,
                    n_top_colonies=n_top_colonies,
                    penalty_factor=penalty_factor
                )
        
        if st.button(" Run Analysis", type="primary"):
            if uploaded_file is not None:
                st.session_state.run_analysis = True
                st.session_state.uploaded_file = uploaded_file
                st.session_state.params = current_params
                # Clear cached results to force re-analysis
                if 'analysis_results' in st.session_state:
                    del st.session_state.analysis_results
            else:
                st.error("Please upload an image first!")
        
        # Add a button to re-run analysis with current parameters
        if 'run_analysis' in st.session_state and st.session_state.run_analysis:
            if st.button(" Update Display", type="secondary"):
                # Update parameters and clear cache
                st.session_state.params = current_params
                if 'analysis_results' in st.session_state:
                    del st.session_state.analysis_results
                st.rerun()
    
    # main content area
    if 'run_analysis' in st.session_state and st.session_state.run_analysis:
        if 'uploaded_file' in st.session_state:
            uploaded_file = st.session_state.uploaded_file
            stored_params = st.session_state.get('params', {})
            
            # Use stored parameters
            params = stored_params.copy()
            
            # save uploaded file temporarily
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Check if we need to re-run analysis or use cached results
            if 'analysis_results' not in st.session_state:
                # run analysis
                with st.spinner(" Analyzing bacterial colonies..."):
                    analyzer = ColonyAnalyzer(**params)
                    results = analyzer.run_full_analysis("temp_image.jpg")
                    # add binary mask to results
                    if hasattr(analyzer, 'final_binary_mask'):
                        results['final_binary_mask'] = analyzer.final_binary_mask
                    # Cache the results
                    st.session_state.analysis_results = results
                    st.session_state.params = params
            else:
                # Use cached results
                results = st.session_state.analysis_results
            
            if results is not None:
                display_results(results, params.get('n_top_colonies', 20))
            else:
                st.error(" Analysis failed. Please check your image and try again.")
            
            # cleanup
            import os
            if os.path.exists("temp_image.jpg"):
                os.remove("temp_image.jpg")
    
    else:
        # welcome screen
        st.markdown("""
        ## Welcome to the Bacterial Colony Analyzer! 
        
        This app analyzes petri dish images to detect, characterize, and score bacterial colonies.
        
        ### What it does:
        - **Detects colonies** using advanced image processing
        - **Analyzes morphology** (size, shape, roundness)
        - **Clusters by color** to group similar colonies
        - **Measures density** and opacity characteristics
        - **Scores colonies** based on multiple factors
        - **Visualizes results** with interactive plots
        
        ### How to use:
        1. Upload a petri dish image using the sidebar
        2. Adjust analysis parameters if needed
        3. Click "Run Analysis" to start processing
        4. Explore the results in the tabs below
        
        ### Supported formats:
        - PNG, JPG, JPEG images
        - High resolution recommended for best results
        """)
        
        # example image
        st.subheader(" Example Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original Image**")
            st.image("https://via.placeholder.com/300x200/4CAF50/FFFFFF?text=Petri+Dish")
        
        with col2:
            st.markdown("**Detected Colonies**")
            st.image("https://via.placeholder.com/300x200/2196F3/FFFFFF?text=Colony+Detection")
        
        with col3:
            st.markdown("**Analysis Results**")
            st.image("https://via.placeholder.com/300x200/FF9800/FFFFFF?text=Results")

def display_results(results, n_top_colonies):
    # display analysis results in organized tabs
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        " Overview", 
        " Colony Details", 
        " Color Analysis", 
        " Morphology", 
        " Top Colonies",
        " Binary Mask & Grid"
    ])
    
    with tab1:
        display_overview(results)
    
    with tab2:
        display_colony_details(results)
    
    with tab3:
        display_color_analysis(results)
    
    with tab4:
        display_morphology_analysis(results)
    
    with tab5:
        display_top_colonies(results, n_top_colonies)
    
    with tab6:
        display_binary_mask(results, n_top_colonies)

def display_overview(results):
    # display overview statistics and key metrics
    st.header(" Analysis Overview")
    
    # key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Colonies", len(results['colony_properties']))
    
    with col2:
        avg_area = np.mean([prop.area for prop in results['colony_properties']])
        st.metric("Average Area", f"{avg_area:.0f} px²")
    
    with col3:
        if 'density_df' in results and not results['density_df'].empty:
            dense_count = len(results['density_df'][results['density_df']['density_class'] == 'dense'])
            st.metric("Dense Colonies", dense_count)
        else:
            st.metric("Dense Colonies", "N/A")
    
    with col4:
        if 'morph_df' in results and not results['morph_df'].empty:
            circular_count = len(results['morph_df'][results['morph_df']['form'] == 'circular'])
            st.metric("Circular Colonies", circular_count)
        else:
            st.metric("Circular Colonies", "N/A")
    
    # image comparison
    st.subheader(" Image Processing Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Image**")
        st.image(results['original_image'])
    
    with col2:
        st.markdown("**Processed Image**")
        st.image(results['processed_image'])
    
    # colony detection visualization
    st.subheader(" Colony Detection")
    
    # create visualization of detected colonies
    colony_viz = results['processed_image'].copy()
    for prop in results['colony_properties']:
        mask = (results['colony_labels'] == prop.label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(colony_viz, contours, -1, (0, 255, 0), 2)
    
    st.image(colony_viz, caption="Detected colonies highlighted in green")

def display_colony_details(results):
    # display detailed colony information in tables
    st.header(" Colony Details")
    
    if 'combined_df' in results and not results['combined_df'].empty:
        df = results['combined_df']
        
        # filters
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_forms = st.multiselect(
                "Filter by form",
                options=df['form'].unique() if 'form' in df else [],
                default=df['form'].unique() if 'form' in df else []
            )
        
        with col2:
            if 'density_class' in df:
                selected_densities = st.multiselect(
                    "Filter by density",
                    options=df['density_class'].unique(),
                    default=df['density_class'].unique()
                )
            else:
                selected_densities = []
        
        with col3:
            if 'color_cluster' in df:
                selected_colors = st.multiselect(
                    "Filter by color cluster",
                    options=sorted(df['color_cluster'].unique()),
                    default=sorted(df['color_cluster'].unique())
                )
            else:
                selected_colors = []
        
        # apply filters
        filtered_df = df.copy()
        if selected_forms and 'form' in df:
            filtered_df = filtered_df[filtered_df['form'].isin(selected_forms)]
        if selected_densities and 'density_class' in df:
            filtered_df = filtered_df[filtered_df['density_class'].isin(selected_densities)]
        if selected_colors and 'color_cluster' in df:
            filtered_df = filtered_df[filtered_df['color_cluster'].isin(selected_colors)]
        
        # display table
        st.dataframe(
            filtered_df.round(3),
            use_container_width=True,
            hide_index=True
        )
        
        # download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label=" Download CSV",
            data=csv,
            file_name="colony_analysis_results.csv",
            mime="text/csv"
        )
    
    else:
        st.warning("No colony data available")

def display_color_analysis(results):
    # display color clustering results
    st.header(" Color Analysis")
    
    if 'colony_data' in results and results['colony_data']:
        colony_data = results['colony_data']
        
        # color cluster distribution
        if colony_data and 'color_cluster' in colony_data[0]:
            color_counts = {}
            for data in colony_data:
                cluster = data['color_cluster']
                color_counts[cluster] = color_counts.get(cluster, 0) + 1
            
            # create pie chart
            fig = px.pie(
                values=list(color_counts.values()),
                names=[f"Cluster {k}" for k in color_counts.keys()],
                title="Color Cluster Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # color visualization
        st.subheader(" Colony Colors by Cluster")
        
        # create color visualization
        if 'colony_labels' in results and 'processed_image' in results:
            color_viz = results['processed_image'].copy()
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            
            for data in colony_data:
                if 'color_cluster' in data:
                    cluster = data['color_cluster']
                    color = colors[cluster % len(colors)]
                    
                    # find colony mask
                    for prop in results['colony_properties']:
                        if prop.label == data['label']:
                            mask = (results['colony_labels'] == prop.label).astype(np.uint8)
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(color_viz, contours, -1, color, 3)
                            break
            
            st.image(color_viz, caption="Colonies colored by cluster")
    
    else:
        st.warning("No color analysis data available")

def display_morphology_analysis(results):
    # display morphology analysis results
    st.header(" Morphology Analysis")
    
    if 'morph_df' in results and not results['morph_df'].empty:
        df = results['morph_df']
        
        # form distribution
        col1, col2 = st.columns(2)
        
        with col1:
            form_counts = df['form'].value_counts()
            fig = px.bar(
                x=form_counts.index,
                y=form_counts.values,
                title="Colony Form Distribution",
                labels={'x': 'Form', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            margin_counts = df['margin'].value_counts()
            fig = px.pie(
                values=margin_counts.values,
                names=margin_counts.index,
                title="Margin Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # morphology scatter plots
        st.subheader(" Morphology Relationships")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                df,
                x='area',
                y='circularity',
                color='form',
                title="Area vs Circularity",
                labels={'area': 'Area (pixels²)', 'circularity': 'Circularity'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                df,
                x='aspect_ratio',
                y='solidity',
                color='form',
                title="Aspect Ratio vs Solidity",
                labels={'aspect_ratio': 'Aspect Ratio', 'solidity': 'Solidity'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # morphology statistics
        st.subheader(" Morphology Statistics")
        morph_stats = df[['area', 'circularity', 'aspect_ratio', 'solidity']].describe()
        st.dataframe(morph_stats, use_container_width=True)
    
    else:
        st.warning("No morphology data available")

def display_top_colonies(results, n_top_colonies):
    # display top-scoring colonies with visualizations
    st.header(f" Top {n_top_colonies} Colonies")
    
    if 'top_colonies' in results and not results['top_colonies'].empty:
        # Use the current slider value to limit display, not the backend selection
        top_df = results['top_colonies'].head(n_top_colonies)
        
        # top colonies table
        st.subheader(" Top Colonies by Interest Score")
        st.dataframe(
            top_df[['colony_id', 'bio_interest', 'form', 'density_class', 'area']].round(3),
            use_container_width=True,
            hide_index=True
        )
        
        # score distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                top_df,
                x='bio_interest',
                title="Interest Score Distribution",
                labels={'bio_interest': 'Interest Score', 'count': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                top_df,
                x='area',
                y='bio_interest',
                color='form',
                size='bio_interest',
                title="Area vs Interest Score",
                labels={'area': 'Area (pixels²)', 'bio_interest': 'Interest Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # visualize top colonies on image
        st.subheader(" Top Colonies Visualization")
        
        if 'processed_image' in results and 'colony_properties' in results:
            # Original vs marked image comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image**")
                st.image(results['processed_image'], caption="Original processed image")
            
            with col2:
                st.markdown("**Top Colonies Highlighted**")
                marked_image = results['processed_image'].copy()
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                
                for idx, row in top_df.iterrows():
                    colony_id = row['colony_id']
                    rank = idx + 1
                    color = colors[idx % len(colors)]
                    
                    prop = results['colony_properties'][colony_id]
                    colony_mask = (results['colony_labels'] == prop.label).astype(np.uint8)
                    contours, _ = cv2.findContours(colony_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        cv2.drawContours(marked_image, contours, -1, color, thickness=6)
                        
                        y_center, x_center = prop.centroid
                        center_point = (int(x_center), int(y_center))
                        cv2.circle(marked_image, center_point, 25, (255, 255, 255), -1)
                        cv2.circle(marked_image, center_point, 25, color, 3)
                        cv2.putText(marked_image, str(rank),
                                   (int(x_center-10), int(y_center+8)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                
                st.image(marked_image, caption="Top colonies highlighted with rankings")
            
            # Individual zoomed views of top colonies
            st.subheader(" Zoomed Views of Top Colonies")
            
            zoom_size = 100  # pixels around colony
            
            # Create columns for zoomed views
            cols = st.columns(n_top_colonies)
            
            for idx, (col_idx, row) in enumerate(zip(cols, top_df.iterrows())):
                colony_id = row[1]['colony_id']
                rank = idx + 1
                color = colors[idx % len(colors)]
                
                prop = results['colony_properties'][colony_id]
                y_center, x_center = prop.centroid
                
                # Calculate zoom boundaries
                y_min = max(0, int(y_center - zoom_size))
                y_max = min(results['processed_image'].shape[0], int(y_center + zoom_size))
                x_min = max(0, int(x_center - zoom_size))
                x_max = min(results['processed_image'].shape[1], int(x_center + zoom_size))
                
                # Extract zoomed region
                zoomed_region = results['processed_image'][y_min:y_max, x_min:x_max].copy()
                zoomed_labels = results['colony_labels'][y_min:y_max, x_min:x_max]
                
                # Mark the specific colony in zoom
                colony_mask_zoom = (zoomed_labels == prop.label).astype(np.uint8)
                contours_zoom, _ = cv2.findContours(colony_mask_zoom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours_zoom:
                    # Thick outline on zoomed view
                    cv2.drawContours(zoomed_region, contours_zoom, -1, color, thickness=4)
                    
                    # Crosshairs at center
                    center_y_zoom = int(y_center - y_min)
                    center_x_zoom = int(x_center - x_min)
                    
                    # Draw crosshairs
                    cv2.line(zoomed_region, (center_x_zoom-30, center_y_zoom),
                            (center_x_zoom+30, center_y_zoom), color, 3)
                    cv2.line(zoomed_region, (center_x_zoom, center_y_zoom-30),
                            (center_x_zoom, center_y_zoom+30), color, 3)
                    
                    cv2.circle(zoomed_region, (center_x_zoom, center_y_zoom), 8, (255, 255, 255), -1)
                    cv2.circle(zoomed_region, (center_x_zoom, center_y_zoom), 8, color, 2)
                
                # Display zoomed view
                with col_idx:
                    st.image(zoomed_region, caption=f"Colony #{rank}")
                    
                    # Title with key info
                    score = row[1]['bio_interest']
                    form = row[1].get('form', 'unknown')
                    st.caption(f"Score: {score:.2f}")
                    st.caption(f"Form: {form}")
    
    else:
        st.warning("No top colonies data available")

def display_binary_mask(results, n_top_colonies):
    st.header(" Final Binary Mask (Colonies)")
    if 'final_binary_mask' in results and results['final_binary_mask'] is not None:
        st.image(results['final_binary_mask'], caption="Final binary mask (colonies=white)")
        
        # Optionally overlay a grid
        import cv2
        import numpy as np
        mask = results['final_binary_mask'].copy()
        h, w = mask.shape
        grid_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        grid_spacing = st.slider("Grid spacing (pixels)", 10, 100, 20)
        for x in range(0, w, grid_spacing):
            cv2.line(grid_img, (x, 0), (x, h), (200, 200, 200), 1)
        for y in range(0, h, grid_spacing):
            cv2.line(grid_img, (0, y), (w, y), (200, 200, 200), 1)
        st.image(grid_img, caption="Binary mask with grid overlay")
        
        # Show top colonies highlighted on binary mask
        if 'top_colonies' in results and not results['top_colonies'].empty:
            st.subheader(" Top Colonies Highlighted on Binary Mask")
            
            # Create highlighted binary mask
            highlighted_mask = grid_img.copy()
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            
            actual_n_top = min(n_top_colonies, len(results['top_colonies']))
            top_colonies = results['top_colonies'].head(actual_n_top)
            
            for idx, row in top_colonies.iterrows():
                colony_id = row['colony_id']
                rank = idx + 1
                color = colors[idx % len(colors)]
                
                prop = results['colony_properties'][colony_id]
                colony_mask = (results['colony_labels'] == prop.label).astype(np.uint8)
                contours, _ = cv2.findContours(colony_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    color = colors[idx % len(colors)]
                    cv2.drawContours(highlighted_mask, contours, -1, color, thickness=6)
                    y, x = prop.centroid
                    cv2.putText(highlighted_mask, str(idx+1),
                                (int(x-10), int(y+10)), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, color, 4)
            
            st.image(highlighted_mask, caption="top colonies highlighted")
        
        # Show zoomed views of top colonies on binary mask
        if 'top_colonies' in results and not results['top_colonies'].empty:
            st.subheader(" Zoomed Views of Top Colonies")
            
            # Use the n_top_colonies parameter passed to the function
            # Get the actual number of colonies to display (don't cap at 5)
            actual_n_top = min(n_top_colonies, len(results['top_colonies']))
            top_colonies = results['top_colonies'].head(actual_n_top)
            zoom_size = 100  # pixels around colony
            
            # Create columns for zoomed views
            cols = st.columns(actual_n_top)
            
            for idx, (col_idx, row) in enumerate(zip(cols, top_colonies.iterrows())):
                prop = results['colony_properties'][row[1]['colony_id']]
                y_center, x_center = prop.centroid
                y_min = max(0, int(y_center - zoom_size))
                y_max = min(mask.shape[0], int(y_center + zoom_size))
                x_min = max(0, int(x_center - zoom_size))
                x_max = min(mask.shape[1], int(x_center + zoom_size))

                crop = mask[y_min:y_max, x_min:x_max].copy()
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                mask_crop = results['colony_labels'][y_min:y_max, x_min:x_max]
                contours, _ = cv2.findContours((mask_crop==prop.label).astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(crop_bgr, contours, -1, colors[idx%len(colors)], thickness=2)

                # draw light grid
                h, w = crop_bgr.shape[:2]
                for gx in range(0, w, 20):
                    cv2.line(crop_bgr, (gx, 0), (gx, h), (200, 200, 200), 1)
                for gy in range(0, h, 20):
                    cv2.line(crop_bgr, (0, gy), (w, gy), (200, 200, 200), 1)
                
                # Display in column
                with col_idx:
                    st.image(crop_bgr, caption=f'Rank {idx+1}')
                    st.caption(f"Score: {row[1]['bio_interest']:.2f}")
    else:
        st.warning("No binary mask available.")

if __name__ == "__main__":
    main() 