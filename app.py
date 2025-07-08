# bacterial colony analysis app - updated 2025-07-08 14:00
# admin functionality removed for clean user experience

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
import json
import datetime
from colony_analyzer import ColonyAnalyzer
# Authentication removed for direct access

st.set_page_config(
    page_title="Bacterial Colony Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# helper functions for downloading images and charts
def create_download_link(data, filename, mime_type):
    # creates a download link for any file type
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def image_to_bytes(image, format='PNG'):
    # converts numpy array or PIL image to bytes for download
    if isinstance(image, np.ndarray):
        # ensure correct data type for PIL
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        image = Image.fromarray(image)
    
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()

def plotly_to_bytes(fig, format='PNG', width=1200, height=800):
    # converts plotly figure to bytes for download - disabled to avoid kaleido dependency
    return None

def matplotlib_to_bytes(fig, format='PNG', dpi=150):
    # converts matplotlib figure to bytes for download
    buf = io.BytesIO()
    fig.savefig(buf, format=format.lower(), dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()

def initialize_run_history():
    # initialize run history in session state
    if 'run_history' not in st.session_state:
        st.session_state.run_history = []
    if 'current_run_id' not in st.session_state:
        st.session_state.current_run_id = 0

def add_run_to_history(params, results, image_name):
    # add a new run to history with parameters and results
    initialize_run_history()
    
    run_id = st.session_state.current_run_id + 1
    st.session_state.current_run_id = run_id
    
    run_data = {
        'run_id': run_id,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image_name': image_name,
        'parameters': params.copy(),
        'results': results,
        'colony_count': len(results['colony_properties']) if results and 'colony_properties' in results else 0
    }
    
    st.session_state.run_history.append(run_data)

def get_parameter_changes(current_params, previous_params):
    # compare parameters and return what changed
    changes = []
    
    for key, current_val in current_params.items():
        if key in previous_params:
            prev_val = previous_params[key]
            if current_val != prev_val:
                changes.append(f"{key}: {prev_val} → {current_val}")
        else:
            changes.append(f"{key}: (new) {current_val}")
    
    return changes

def display_run_history():
    # display run history with parameter changes and download options
    initialize_run_history()
    
    if not st.session_state.run_history:
        st.info("No analysis runs yet. Upload an image and run analysis to start building history.")
        return
    
    st.header("Analysis Run History")
    st.caption(f"Track parameter changes and compare results across {len(st.session_state.run_history)} runs")
    
    # Create expandable sections for each run
    for i, run in enumerate(reversed(st.session_state.run_history)):
        run_id = run['run_id']
        
        with st.expander(f"Run {run_id} - {run['timestamp']} ({run['colony_count']} colonies)", 
                        expanded=(i == 0)):  # Latest run expanded by default
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.write(f"**Image:** {run['image_name']}")
                st.write(f"**Colonies Detected:** {run['colony_count']}")
                
                # Show parameter changes compared to previous run
                if i < len(st.session_state.run_history) - 1:
                    prev_run = st.session_state.run_history[-(i+2)]  # Previous run
                    changes = get_parameter_changes(run['parameters'], prev_run['parameters'])
                    
                    if changes:
                        st.write("**Parameters Changed:**")
                        for change in changes[:5]:  # Show first 5 changes
                            st.write(f"• {change}")
                        if len(changes) > 5:
                            st.write(f"... and {len(changes) - 5} more changes")
                    else:
                        st.write("**No parameter changes from previous run**")
                else:
                    st.write("**First run - baseline parameters**")
            
            with col2:
                # Key results summary
                results = run['results']
                if results and 'combined_df' in results and not results['combined_df'].empty:
                    df = results['combined_df']
                    avg_area = df['area'].mean() if 'area' in df else 0
                    
                    st.write("**Results Summary:**")
                    st.write(f"• Average Colony Area: {avg_area:.1f} px²")
                    
                    if 'form' in df:
                        most_common_form = df['form'].mode().iloc[0] if len(df['form'].mode()) > 0 else "N/A"
                        st.write(f"• Most Common Form: {most_common_form}")
                    
                    if 'density_class' in df:
                        dense_count = len(df[df['density_class'] == 'dense'])
                        st.write(f"• Dense Colonies: {dense_count}")
            
            with col3:
                st.write("**Downloads:**")
                
                # CSV download for this run
                if results and 'combined_df' in results and not results['combined_df'].empty:
                    csv_data = results['combined_df'].to_csv(index=False)
                    st.download_button(
                        label="CSV Data",
                        data=csv_data,
                        file_name=f"run_{run_id}_analysis.csv",
                        mime="text/csv",
                        key=f"csv_{run_id}"
                    )
                
                # Original image download
                if results and 'original_image' in results:
                    img_bytes = image_to_bytes(results['original_image'])
                    st.download_button(
                        label="Original Image",
                        data=img_bytes,
                        file_name=f"run_{run_id}_original.png",
                        mime="image/png",
                        key=f"orig_{run_id}"
                    )
                
                # Processed image download
                if results and 'processed_image' in results:
                    img_bytes = image_to_bytes(results['processed_image'])
                    st.download_button(
                        label="Processed Image",
                        data=img_bytes,
                        file_name=f"run_{run_id}_processed.png",
                        mime="image/png",
                        key=f"proc_{run_id}"
                    )
                
                # Parameters JSON download
                params_json = json.dumps(run['parameters'], indent=2)
                st.download_button(
                    label="Parameters",
                    data=params_json,
                    file_name=f"run_{run_id}_parameters.json",
                    mime="application/json",
                    key=f"params_{run_id}"
                )
    
    # Bulk operations
    if len(st.session_state.run_history) > 1:
        st.subheader("Bulk Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export All Run Data"):
                # Create comprehensive export of all runs
                all_data = []
                for run in st.session_state.run_history:
                    run_summary = {
                        'Run_ID': run['run_id'],
                        'Timestamp': run['timestamp'],
                        'Image_Name': run['image_name'],
                        'Colony_Count': run['colony_count']
                    }
                    # Add key parameters
                    run_summary.update(run['parameters'])
                    all_data.append(run_summary)
                
                df_export = pd.DataFrame(all_data)
                csv_export = df_export.to_csv(index=False)
                
                st.download_button(
                    label="Download Run Comparison CSV",
                    data=csv_export,
                    file_name=f"all_runs_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Clear Run History"):
                st.session_state.run_history = []
                st.session_state.current_run_id = 0
                st.success("Run history cleared!")
                st.rerun()

# Add custom CSS to improve file uploader appearance
st.markdown("""
<style>
.uploadedFile {
    border: 2px dashed #1f77b4;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin: 10px 0;
    background-color: #f0f2f6;
}

.stFileUploader > div > div > div > div {
    border: 2px dashed #1f77b4 !important;
    border-radius: 10px !important;
    padding: 20px !important;
    background-color: #f8f9fa !important;
}

.stFileUploader > div > div > div > div > p {
    font-size: 16px !important;
    font-weight: 500 !important;
    color: #1f77b4 !important;
}

/* Hide the browse files button */
.stFileUploader > div > div > div > div > button {
    display: none !important;
}

/* Make the entire drop zone clickable */
.stFileUploader > div > div > div {
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Simple user tracking with group ID
    if 'user_logged_in' not in st.session_state:
        st.title("Bacterial Colony Analysis")
        st.markdown("Enter your Group ID to access the analysis tool")
        
        group_id = st.text_input("Group ID:", placeholder="A1, A2")
        
        if st.button("Access Analysis Tool"):
            if group_id.strip():
                st.session_state.user_logged_in = True
                st.session_state.group_id = group_id.strip()
                st.rerun()
            else:
                st.error("Please enter a valid Group ID")
        return

    # Initialize session tracking with group ID
    if 'session_id' not in st.session_state:
        import hashlib
        import time
        st.session_state.session_id = hashlib.md5(f"{st.session_state.group_id}_{time.time()}".encode()).hexdigest()[:8]
    
    # Main app interface
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Bacterial Colony Analyzer")
        st.caption("Advanced image analysis for petri dish colony detection and characterization")
    with col2:
        st.write(f"**User:** {st.session_state.group_id}")
        if st.button("Change User", help="Switch to different Group ID"):
            st.session_state.user_logged_in = False
            st.session_state.group_id = None
            st.session_state.session_id = None
            st.rerun()
    
    # Add mode selection
    analysis_mode = st.radio(
        "Choose Analysis Mode:",
        ["Single Image Analysis", "Multi-Image Comparison"],
        horizontal=True
    )
    
    # sidebar for parameters
    with st.sidebar:
        if analysis_mode == "Single Image Analysis":
            st.header("Upload Image")
            
            uploaded_file = st.file_uploader("Upload petri dish image", 
                                            type=['png', 'jpg', 'jpeg'],
                                            label_visibility="hidden")
            
            if uploaded_file is not None:
                st.success(f"✓ {uploaded_file.name}")
                # Show image preview
                image = Image.open(uploaded_file)
                st.image(image, caption="Preview", width=200)
                
                # Upload logged to session
        
        elif analysis_mode == "Multi-Image Comparison":
            st.header("Upload Multiple Images")
            
            # Upload method selection
            upload_method = st.radio(
                "Choose upload method:",
                ["Select Individual Files", "Upload Folder (Drag & Drop)"],
                horizontal=True
            )
            
            if upload_method == "Select Individual Files":
                uploaded_files = st.file_uploader(
                    "Select up to 100 images", 
                    type=['png', 'jpg', 'jpeg'],
                    accept_multiple_files=True
                )
            else:
                st.info("📁 **Folder Upload Instructions:**\n"
                       "1. Create a folder with your images (PNG, JPG, JPEG)\n"
                       "2. Drag the entire folder to the upload area below\n"
                       "3. All images in the folder will be processed automatically")
                
                uploaded_files = st.file_uploader(
                    "Drag folder here or click to browse", 
                    type=['png', 'jpg', 'jpeg'],
                    accept_multiple_files=True,
                    help="You can drag a folder containing images directly to this area"
                )
            
            if uploaded_files:
                if len(uploaded_files) > 100:
                    st.error("Please select no more than 100 images")
                    uploaded_files = uploaded_files[:100]
                
                st.success(f"✓ {len(uploaded_files)} images uploaded")
                
                # Auto-generate labels or allow custom labeling
                label_method = st.radio(
                    "Sample labeling method:",
                    ["Auto-generate from filenames", "Custom labels"],
                    horizontal=True
                )
                
                sample_labels = {}
                if label_method == "Auto-generate from filenames":
                    for file in uploaded_files:
                        # Use filename without extension as label
                        clean_name = file.name.rsplit('.', 1)[0]
                        sample_labels[file.name] = clean_name
                    
                    st.write("**Auto-generated labels:**")
                    for file in uploaded_files[:5]:  # Show first 5
                        st.write(f"• {file.name} → {sample_labels[file.name]}")
                    if len(uploaded_files) > 5:
                        st.write(f"... and {len(uploaded_files) - 5} more")
                
                else:  # Custom labels
                    st.subheader("Custom Sample Labels")
                    
                    # Bulk labeling options
                    with st.expander("Bulk Labeling Options"):
                        bulk_prefix = st.text_input("Common prefix for all samples:", value="Sample")
                        if st.button("Apply bulk prefix"):
                            for i, file in enumerate(uploaded_files):
                                st.session_state[f"label_{i}"] = f"{bulk_prefix}_{i+1}"
                            st.rerun()
                    
                    # Individual labels
                    for i, file in enumerate(uploaded_files):
                        sample_labels[file.name] = st.text_input(
                            f"Label for {file.name[:20]}...", 
                            value=f"Sample_{i+1}",
                            key=f"label_{i}"
                        )
        
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
                                       help="Higher = smoother detection (less noise), Lower = more detailed detection (captures small features)")
        adaptive_c = st.slider("Adaptive threshold C", 1, 10, 3,
                              help="Higher = detects fewer colonies (stricter), Lower = detects more colonies (more sensitive)")
        watershed_min_distance = st.slider("Watershed min distance", 5, 15, 8,
                                          help="Higher = merges touching colonies together, Lower = separates colonies more aggressively")
        watershed_threshold = st.slider("Watershed threshold", 0.1, 0.5, 0.3, 0.05,
                                       help="Higher = detects fewer peaks (stricter separation), Lower = detects more peaks (more separation)")
        
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
                                  help="Higher = more diverse top colonies (avoids similar ones), Lower = shows highest scores regardless of similarity")
        
        st.header("Analysis Options")
        st.caption("Choose which analyses to run")
        
        run_morphology = st.checkbox("Analyze morphology", value=True,
                                    help="Measure colony shape, size, and edge characteristics")
        run_color_analysis = st.checkbox("Analyze colors", value=True,
                                        help="Extract and cluster colony colors")
        run_density_analysis = st.checkbox("Analyze density", value=True,
                                          help="Measure colony opacity and texture")
        
        # Add guide button in sidebar
        if st.button("User Guide", help="Click to open the complete package guide on the right"):
            st.session_state.show_guide = True
        
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
                    min_distance=watershed_min_distance,
                    watershed=True,
                    color_n_clusters=(color_n_clusters if color_n_clusters > 0 else None),
                    color_random_state=color_random_state,
                    color_n_init=color_n_init,
                    n_top_colonies=n_top_colonies,
                    penalty_factor=penalty_factor
                )
        
        if analysis_mode == "Single Image Analysis":
            if st.button(" Run Analysis", type="primary"):
                if uploaded_file is not None:
                    st.session_state.run_analysis = True
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.params = current_params
                    st.session_state.analysis_mode = "single"
                    # Clear cached results to force re-analysis
                    if 'analysis_results' in st.session_state:
                        del st.session_state.analysis_results
                else:
                    st.error("Please upload an image first!")
        
        elif analysis_mode == "Multi-Image Comparison":
            if st.button(" Run Multi-Image Analysis", type="primary"):
                if uploaded_files and len(uploaded_files) >= 2:
                    st.session_state.run_analysis = True
                    st.session_state.uploaded_files = uploaded_files
                    st.session_state.sample_labels = sample_labels
                    st.session_state.params = current_params
                    st.session_state.analysis_mode = "multi"
                    # Clear cached results to force re-analysis
                    if 'multi_analysis_results' in st.session_state:
                        del st.session_state.multi_analysis_results
                else:
                    st.error("Please upload at least 2 images for comparison!")
        


    
    # Create main layout with guide on right if activated
    if st.session_state.get('show_guide', False):
        col1, col2 = st.columns([2, 1])  # Main content gets 2/3, guide gets 1/3
        main_container = col1
        guide_container = col2
    else:
        main_container = st.container()
        guide_container = None
    
    # Show guide in right column if activated
    if guide_container is not None:
        with guide_container:
            if st.button("Close Guide", key="close_guide_main"):
                st.session_state.show_guide = False
                st.rerun()
            
            st.markdown("""
            #### Bacterial Colony Detection: Image Analysis Pipeline
            
            Bacterial colony counting and characterization is a fundamental task in microbiology. Manual analysis of petri dish images is time-consuming and prone to error, especially when dealing with large datasets or high-throughput experiments. This app provides a complete, 12-step workflow for analyzing photographs of petri dishes with bacterial colonies.
            
            ### Core Packages & Their Purpose
            
            #### **Image Processing**
            - **OpenCV** (`cv2`) - Core image operations, bilateral filtering, contour detection, morphological operations
            - **scikit-image** (`skimage`) - Advanced segmentation, watershed algorithm, morphology analysis, peak detection
            - **Pillow** (`PIL`) - Image loading and basic manipulations in streamlit interface
            
            #### **Scientific Computing** 
            - **NumPy** (`numpy`) - Array operations, mathematical computations, distance calculations
            - **SciPy** (`scipy`) - Distance transforms, morphological operations, texture analysis
            - **pandas** - Data organization, analysis results storage, dataframe operations
            
            #### **Machine Learning**
            - **scikit-learn** (`sklearn`) - K-means clustering for color analysis, standardscaler for data normalization
            - **KMeans clustering** - Groups colonies by color similarity using lab color space
            - **DBSCAN** - Alternative clustering method for density-based grouping
            
            #### **Visualization**
            - **Matplotlib** (`plt`) - Basic plotting, image display, histograms, scatter plots
            - **Seaborn** (`sns`) - Statistical plots, distribution charts, enhanced visualizations
            - **Plotly** (`px`) - Interactive charts, 3d visualizations, pie charts
            
            ### Complete 12-Step Analysis Pipeline
            
            #### **Section 1: Environment Setup**
            Prepares the environment by installing and importing all necessary image-processing and data-analysis libraries
            
            #### **Section 2: Image Upload** 
            Lets you upload a dish photograph via streamlit file picker
            
            #### **Section 3: Image Display**
            Shows the loaded photograph and prints basic properties (dimensions, color channels) so you can confirm it loaded correctly
            
            #### **Section 4: Image Preprocessing**
            - **Bilateral Filtering** - Denoise while keeping edges sharp using cv2.bilateralFilter with diameter=9, sigmaColor=75, sigmaSpace=75
            - **CLAHE Enhancement** - Enhances local contrast using cv2.createCLAHE with clipLimit=3.0, tileGridSize=(8,8)
            - **Gamma Correction** - Adjusts brightness for optimal detection using gamma=1.2 
            - **Sharpening** - Enhances colony boundaries using convolution kernel [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]
            
            #### **Section 5: Plate Detection**
            - Locates the inner region of the petri dish and creates a binary mask of the plate area
            - Creates inner margin to exclude plate edges (8% margin from edges)
            - Uses otsu's thresholding and contour analysis to refine the region
            - Finds largest contour in inner area and combines with inner margin for final mask
            
            #### **Section 6: Colony Segmentation**
            - **Adaptive Thresholding** - Separates colonies from background using cv2.adaptiveThreshold with ADAPTIVE_THRESH_GAUSSIAN_C
            - **Morphological Operations** - Cleans up detected regions using opening and closing operations
            - **Watershed Algorithm** - Separates touching/overlapping colonies using scipy.ndimage.distance_transform_edt and skimage.segmentation.watershed
            - **Size Filtering** - Removes artifacts by filtering colonies between min_size=15 and max_size=10000 pixels
            
            #### **Section 7: Morphology Analysis**
            Measures size, roundness, elongation, solidity, and classifies edge style and form for every colony:
            - **Area & Perimeter** - Basic size measurements from regionprops
            - **Circularity** - Calculated as 4*π*area/perimeter² to measure roundness
            - **Aspect Ratio** - Major_axis/minor_axis to measure elongation
            - **Solidity** - Measures how well-filled the colony shape is
            - **Convexity** - Measures edge smoothness by comparing area to convex hull area
            - **Margin Classification** - Categorizes edge types (entire, undulate, lobate, serrate) based on convexity and complexity
            - **Form Classification** - Categorizes overall shape (circular, oval, irregular, filamentous) based on circularity and aspect ratio
            
            #### **Section 8: Color Analysis**
            Extracts each colony's dominant hue, converts to lab color space, and groups similar-looking colonies:
            - **Dominant Color Extraction** - Finds main color using k-means clustering within each colony
            - **RGB to LAB Conversion** - Uses perceptually uniform color space for better comparison
            - **K-means Clustering** - Groups colonies with similar colors using sklearn.KMeans
            - **Elbow Method** - Automatically determines optimal number of color groups by analyzing inertia curves
            - **Color Visualization** - Draws contours in distinct colors to show each group
            
            #### **Section 9: Density Analysis**
            Quantifies brightness, texture, and saturation metrics to classify how "dense" or "translucent" each colony appears:
            - **Opacity Scoring** - Measures how different colony is from background using abs(mean_intensity - background_mean)/background_std
            - **Texture Analysis** - Quantifies surface roughness using local variance filters
            - **Density Gradient** - Compares center vs edge density using distance transforms
            - **Density Classification** - Categories from very_sparse to very_dense based on opacity thresholds
            - **Saturation Analysis** - Measures color intensity using hsv color space
            
            #### **Section 10: Combined Scoring**
            Merges shape, color, and density features into a single "interest" score, highlighting the most distinctive colonies:
            - **Morphology Scores** - Rewards interesting shapes using percentile-based metrics
            - **Form Rarity** - Prioritizes uncommon colony types by calculating frequency-based scores
            - **Size Preferences** - Favors medium-sized colonies using logarithmic ranking
            - **Density Bonuses** - Rewards high-density colonies with enhanced weighting
            - **Novelty Combinations** - Finds unique feature combinations and rare form-color pairs
            - **Diversity Penalties** - Reduces scores for too-similar colonies to encourage variety
            
            #### **Section 11: Top Colony Visualization**
            Draws colored outlines around the highest-scoring colonies on the full dish and zooms in on each:
            - Highlights top 5 colonies on full image with colored outlines and rank labels
            - Displays original and marked images side by side for context
            - Extracts zoomed regions around each top colony center (100 pixel radius)
            - Adds crosshairs and detailed annotations for easy inspection
            
            #### **Section 12: Binary Mask View**
            Provides alternate view by outlining top colonies on simple black-and-white mask:
            - Creates binary image where colonies are white (255) and background is black (0)
            - Outlines top colonies with colored contours and measurement grids
            - Shows side-by-side views for precise sizing and validation
            
            ### App Results & Outputs
            
            #### **Overview Tab**
            - Summary statistics (total colonies, average area, dense/circular counts)
            - Before/after image comparison showing preprocessing effects
            - Colony detection visualization with green outlines
            
            #### **Colony Details Tab**
            - Complete morphology measurements table with sortable columns
            - Individual colony characteristics including area, circularity, form, margin
            - Filterable data for detailed analysis
            
            #### **Color Analysis Tab**
            - Color cluster distribution pie chart showing group percentages
            - Colonies colored by cluster group with distinct outline colors
            - Dominant color visualization for each detected group
            
            #### **Morphology Tab**
            - Shape distribution charts showing form and margin counts
            - Circularity vs area scatter plots for relationship analysis
            - Statistical breakdowns of morphological features
            
            #### **Top Colonies Tab**
            - Highest-scoring colonies with thumbnail images
            - Detailed scoring breakdown showing individual metric contributions
            - Reasons for high interest scores with feature explanations
            
            #### **Binary Mask Tab**
            - Raw detection mask overlay for technical validation
            - Binary view showing exactly what was detected as colonies
            - Useful for adjusting parameters and troubleshooting
            
            ### Parameter Guide
            
            #### **Image Processing Parameters**
            - **Bilateral Filter Diameter** - Pixel neighborhood size for noise reduction (3-21, default 9)
            - **Bilateral SigmaColor** - Color space sigma for filtering (10-150, default 75)
            - **Bilateral SigmaSpace** - Coordinate space sigma (10-150, default 75)
            - **CLAHE Clip Limit** - Contrast enhancement strength (1-10, default 3)
            - **CLAHE Tile Grid** - Local enhancement grid size (2-32, default 8)
            - **Gamma Correction** - Brightness adjustment (0.5-2.5, default 1.2)
            - **Sharpen Strength** - Edge enhancement intensity (0-2, default 1.0)
            
            #### **Colony Detection Parameters**
            - **Plate Margin Percent** - Edge exclusion percentage (0.05-0.20, default 0.08)
            - **Min Colony Size** - Minimum area in pixels (10-50, default 15)
            - **Max Colony Size** - Maximum area in pixels (5000-20000, default 10000)
            - **Adaptive Block Size** - Threshold calculation window (11-25, must be odd, default 15)
            - **Adaptive C** - Threshold adjustment constant (1-10, default 3)
            - **Watershed Min Distance** - Minimum separation between peaks (5-15, default 8)
            
            #### **Advanced Options**
            - **Color Clustering** - Number of color groups (0=auto, 1-10)
            - **Scoring Weights** - Penalty factor for diversity (0-1, default 0.5)
            - **Display Settings** - Number of top colonies to show (1-50, default 20)
            
            ### Technical Implementation Notes
            - Uses reproducible random seeds (42) for consistent k-means results
            - Caches analysis results in streamlit session state for parameter adjustments
            - Processes images up to 20MB with automatic format conversion
            - Optimized for petri dish images with good contrast and lighting
            - Implements elbow method for automatic cluster detection
            - Uses lab color space for perceptually uniform color analysis
            
            ### Tips for Best Results
            - Use well-lit, high-contrast images with uniform illumination
            - Ensure colonies are clearly visible against plate background
            - Adjust bilateral filter parameters if image is very noisy
            - Increase CLAHE clip limit for low-contrast images
            - Adjust gamma correction for over/under-exposed images
            - Fine-tune size limits to exclude debris and large artifacts
            - Use color clustering when colonies have distinct pigmentation
            - Experiment with watershed parameters for touching colonies
            - Check binary mask tab to validate detection accuracy
            """)

    # main content area
    with main_container:
        if 'run_analysis' in st.session_state and st.session_state.run_analysis:
            analysis_mode = st.session_state.get('analysis_mode', 'single')
            
            if analysis_mode == 'single' and 'uploaded_file' in st.session_state:
                uploaded_file = st.session_state.uploaded_file
                stored_params = st.session_state.get('params', {})
                
                # Use stored parameters and convert old parameter names to new ones
                params = stored_params.copy()
                
                # Convert old parameter names to new ones for ColonyAnalyzer compatibility
                if 'watershed_min_distance' in params:
                    params['min_distance'] = params.pop('watershed_min_distance')
                if 'watershed_threshold' in params:
                    params.pop('watershed_threshold')  # Remove unused parameter
                if 'watershed' not in params:
                    params['watershed'] = True  # Add required parameter
                
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
                        
                        # Add to run history
                        if results is not None:
                            add_run_to_history(params, results, uploaded_file.name)
                            
                            # Analysis completed successfully
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
            
            elif analysis_mode == 'multi' and 'uploaded_files' in st.session_state:
                # Multi-image analysis
                uploaded_files = st.session_state.uploaded_files
                sample_labels = st.session_state.get('sample_labels', {})
                stored_params = st.session_state.get('params', {})
                
                # Check if we need to re-run analysis or use cached results
                if 'multi_analysis_results' not in st.session_state:
                    # Run multi-image analysis
                    with st.spinner(f"Analyzing {len(uploaded_files)} images..."):
                        results = run_multi_image_analysis(uploaded_files, sample_labels, stored_params)
                        st.session_state.multi_analysis_results = results
                else:
                    # Use cached results
                    results = st.session_state.multi_analysis_results
                
                if results is not None:
                    display_multi_image_results(results)
                else:
                    st.error("Multi-image analysis failed. Please check your images and try again.")

        else:
            # welcome screen
            st.markdown("""
            ## Welcome to the Bacterial Colony Analyzer! 
            
            This app analyzes petri dish images to detect, characterize, and score bacterial colonies.
            
            ### Analysis Modes:
            
            #### **Single Image Analysis**
            - **Detects colonies** using advanced image processing
            - **Analyzes morphology** (size, shape, roundness)
            - **Clusters by color** to group similar colonies
            - **Measures density** and opacity characteristics
            - **Scores colonies** based on multiple factors
            - **Visualizes results** with interactive plots
            
            #### **Multi-Image Comparison** 
            - **Compare multiple samples** (up to 100 images)
            - **PCA analysis** to identify microbiome variability
            - **Similarity analysis** between different samples
            - **Statistical comparisons** (ANOVA, t-tests)
            - **Variance ranking** to find most/least variable microbiomes
            - **Interactive visualizations** for comparative analysis
            
            ### How to use:
            1. **Choose analysis mode** above (Single or Multi-Image)
            2. **Upload image(s)** using the sidebar
            3. **Label samples** (for multi-image mode)
            4. **Adjust analysis parameters** if needed
            5. **Click "Run Analysis"** to start processing
            6. **Explore results** in the organized tabs
            
            ### Supported formats:
            - PNG, JPG, JPEG images
            - High resolution recommended for best results
            - For multi-image: consistent lighting and setup recommended
            """)

def display_results(results, n_top_colonies):
    # display analysis results in organized tabs
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Overview", 
        "Colony Details", 
        "Color Analysis", 
        "Morphology", 
        "Top Colonies",
        "Binary Mask & Grid",
        "Run History"
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
    
    with tab7:
        display_run_history()

def display_overview(results):
    # display overview statistics and key metrics
    st.header("Analysis Overview")
    
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
    st.subheader("Image Processing Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Image**")
        st.image(results['original_image'])
        # download button
        original_bytes = image_to_bytes(results['original_image'])
        if st.download_button(
            label="Download Original Image",
            data=original_bytes,
            file_name="original_image.png",
            mime="image/png"
        ):
            admin_logger.log_download(st.session_state.session_id, "original_image", "original_image.png")
    
    with col2:
        st.markdown("**Processed Image**")
        st.image(results['processed_image'])
        # download button  
        processed_bytes = image_to_bytes(results['processed_image'])
        if st.download_button(
            label="Download Processed Image",
            data=processed_bytes,
            file_name="processed_image.png",
            mime="image/png"
        ):
            admin_logger.log_download(st.session_state.session_id, "processed_image", "processed_image.png")
    
    # colony detection visualization
    st.subheader("Colony Detection")
    
    # create visualization of detected colonies
    colony_viz = results['processed_image'].copy()
    for prop in results['colony_properties']:
        mask = (results['colony_labels'] == prop.label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(colony_viz, contours, -1, (0, 255, 0), 2)
    
    # store for admin logging
    results['colony_viz'] = colony_viz
    
    st.image(colony_viz, caption="Detected colonies highlighted in green")
    
    # download button for colony detection
    colony_viz_bytes = image_to_bytes(colony_viz)
    st.download_button(
        label="Download Colony Detection Image",
        data=colony_viz_bytes,
        file_name="colony_detection.png",
        mime="image/png"
    )

def display_colony_details(results):
    # display detailed colony information in tables
    st.header("Colony Details")
    
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
        
        # download buttons
        col1, col2 = st.columns(2)
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label=" Download CSV",
                data=csv,
                file_name="colony_analysis_results.csv",
                mime="text/csv"
            )
        
        with col2:
            # create summary report
            summary_text = f"""Bacterial Colony Analysis Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

ANALYSIS SUMMARY:
- Total Colonies Detected: {len(results['colony_properties'])}
- Colonies in Filtered View: {len(filtered_df)}
- Average Colony Area: {np.mean([prop.area for prop in results['colony_properties']]):.1f} pixels²

MORPHOLOGY DISTRIBUTION:
{filtered_df['form'].value_counts().to_string() if 'form' in filtered_df else 'No form data available'}

COLOR CLUSTERS:
{len(set(filtered_df['color_cluster'])) if 'color_cluster' in filtered_df else 'No color data'} distinct color groups detected

DENSITY ANALYSIS:
{filtered_df['density_class'].value_counts().to_string() if 'density_class' in filtered_df else 'No density data available'}

TOP SCORING COLONIES:
{results['top_colonies'][['colony_id', 'bio_interest', 'form']].head(10).to_string(index=False) if 'top_colonies' in results and not results['top_colonies'].empty else 'No scoring data available'}
"""
            st.download_button(
                label=" Download Report",
                data=summary_text,
                file_name="analysis_report.txt",
                mime="text/plain"
            )
    
    else:
        st.warning("No colony data available")

def display_color_analysis(results):
    # display color clustering results
    st.header("Color Analysis")
    
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
            
            # download button for pie chart
            pie_chart_bytes = plotly_to_bytes(fig)
            if pie_chart_bytes:
                st.download_button(
                    label="Download Color Distribution Chart",
                    data=pie_chart_bytes,
                    file_name="color_distribution.png",
                    mime="image/png"
                )
        
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
            
            # download button for color visualization
            color_viz_bytes = image_to_bytes(color_viz)
            st.download_button(
                label="Download Color Cluster Image",
                data=color_viz_bytes,
                file_name="color_clusters.png",
                mime="image/png"
            )
    
    else:
        st.warning("No color analysis data available")

def display_morphology_analysis(results):
    # display morphology analysis results
    st.header("Morphology Analysis")
    
    if 'morph_df' in results and not results['morph_df'].empty:
        df = results['morph_df']
        
        # form distribution
        col1, col2 = st.columns(2)
        
        with col1:
            form_counts = df['form'].value_counts()
            fig_form = px.bar(
                x=form_counts.index,
                y=form_counts.values,
                title="Colony Form Distribution",
                labels={'x': 'Form', 'y': 'Count'}
            )
            st.plotly_chart(fig_form, use_container_width=True)
            
            # download button
            form_chart_bytes = plotly_to_bytes(fig_form)
            if form_chart_bytes:
                st.download_button(
                    label="Download Form Chart",
                    data=form_chart_bytes,
                    file_name="form_distribution.png",
                    mime="image/png"
                )
        
        with col2:
            margin_counts = df['margin'].value_counts()
            fig_margin = px.pie(
                values=margin_counts.values,
                names=margin_counts.index,
                title="Margin Type Distribution"
            )
            st.plotly_chart(fig_margin, use_container_width=True)
            
            # download button
            margin_chart_bytes = plotly_to_bytes(fig_margin)
            if margin_chart_bytes:
                st.download_button(
                    label="Download Margin Chart",
                    data=margin_chart_bytes,
                    file_name="margin_distribution.png",
                    mime="image/png"
                )
        
        # morphology scatter plots
        st.subheader(" Morphology Relationships")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_area = px.scatter(
                df,
                x='area',
                y='circularity',
                color='form',
                title="Area vs Circularity",
                labels={'area': 'Area (pixels²)', 'circularity': 'Circularity'}
            )
            st.plotly_chart(fig_area, use_container_width=True)
            
            # download button
            area_chart_bytes = plotly_to_bytes(fig_area)
            if area_chart_bytes:
                st.download_button(
                    label="Download Area Chart",
                    data=area_chart_bytes,
                    file_name="area_vs_circularity.png",
                    mime="image/png"
                )
        
        with col2:
            fig_aspect = px.scatter(
                df,
                x='aspect_ratio',
                y='solidity',
                color='form',
                title="Aspect Ratio vs Solidity",
                labels={'aspect_ratio': 'Aspect Ratio', 'solidity': 'Solidity'}
            )
            st.plotly_chart(fig_aspect, use_container_width=True)
            
            # download button
            aspect_chart_bytes = plotly_to_bytes(fig_aspect)
            if aspect_chart_bytes:
                st.download_button(
                    label="Download Aspect Chart",
                    data=aspect_chart_bytes,
                    file_name="aspect_vs_solidity.png",
                    mime="image/png"
                )
        
        # morphology statistics
        st.subheader(" Morphology Statistics")
        morph_stats = df[['area', 'circularity', 'aspect_ratio', 'solidity']].describe()
        st.dataframe(morph_stats, use_container_width=True)
    
    else:
        st.warning("No morphology data available")

def display_top_colonies(results, n_top_colonies):
    # display top-scoring colonies with visualizations
    st.header(f"Top {n_top_colonies} Colonies")
    
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
            fig_hist = px.histogram(
                top_df,
                x='bio_interest',
                title="Interest Score Distribution",
                labels={'bio_interest': 'Interest Score', 'count': 'Count'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # download button
            hist_bytes = plotly_to_bytes(fig_hist)
            if hist_bytes:
                st.download_button(
                    label="Download Score Distribution",
                    data=hist_bytes,
                    file_name="score_distribution.png",
                    mime="image/png"
                )
        
        with col2:
            fig_scatter = px.scatter(
                top_df,
                x='area',
                y='bio_interest',
                color='form',
                size='bio_interest',
                title="Area vs Interest Score",
                labels={'area': 'Area (pixels²)', 'bio_interest': 'Interest Score'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # download button
            scatter_bytes = plotly_to_bytes(fig_scatter)
            if scatter_bytes:
                st.download_button(
                    label="Download Score Scatter Plot",
                    data=scatter_bytes,
                    file_name="area_vs_score.png",
                    mime="image/png"
                )
        
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
                
                # download button for marked image
                marked_bytes = image_to_bytes(marked_image)
                st.download_button(
                    label="Download Highlighted Image",
                    data=marked_bytes,
                    file_name="top_colonies_highlighted.png",
                    mime="image/png"
                )
            
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
    st.header("Final Binary Mask (Colonies)")
    if 'final_binary_mask' in results and results['final_binary_mask'] is not None:
        st.image(results['final_binary_mask'], caption="Final binary mask (colonies=white)")
        
        # download button for binary mask
        mask_bytes = image_to_bytes(results['final_binary_mask'])
        st.download_button(
            label="Download Binary Mask",
            data=mask_bytes,
            file_name="binary_mask.png",
            mime="image/png"
        )
        
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
        
        # download button for grid overlay
        grid_bytes = image_to_bytes(grid_img)
        st.download_button(
            label="Download Grid Overlay",
            data=grid_bytes,
            file_name="binary_mask_with_grid.png",
            mime="image/png"
        )
        
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
            
            # download button for highlighted mask
            highlighted_bytes = image_to_bytes(highlighted_mask)
            st.download_button(
                label="Download Highlighted Mask",
                data=highlighted_bytes,
                file_name="highlighted_binary_mask.png",
                mime="image/png"
            )
        
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

def run_multi_image_analysis(uploaded_files, sample_labels, params):
    # runs analysis on multiple images and combines results for comparison
    print("starting multi image analysis")
    
    all_results = {}
    combined_data = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # save temporary file
            temp_filename = f"temp_image_{i}.jpg"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # run single analysis
            analyzer = ColonyAnalyzer(**params)
            results = analyzer.run_full_analysis(temp_filename)
            
            # get sample label
            sample_name = sample_labels.get(uploaded_file.name, f"Sample_{i+1}")
            
            # store results
            all_results[sample_name] = results
            
            # extract features for comparison
            if results and 'combined_df' in results and not results['combined_df'].empty:
                df = results['combined_df'].copy()
                df['sample'] = sample_name
                df['image_name'] = uploaded_file.name
                combined_data.append(df)
            
            # cleanup
            import os
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
        except Exception as e:
            print(f"error processing {uploaded_file.name}: {e}")
            continue
    
    if not combined_data:
        return None
    
    # combine all data
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # run comparative analysis
    comparison_results = {
        'individual_results': all_results,
        'combined_df': combined_df,
        'sample_count': len(all_results),
        'total_colonies': len(combined_df)
    }
    
    # add pca analysis
    try:
        comparison_results['pca_results'] = run_pca_analysis(combined_df)
    except Exception as e:
        print(f"pca analysis failed: {e}")
        comparison_results['pca_results'] = None
    
    # add similarity analysis
    try:
        comparison_results['similarity_results'] = run_similarity_analysis(combined_df)
    except Exception as e:
        print(f"similarity analysis failed: {e}")
        comparison_results['similarity_results'] = None
    
    return comparison_results

def run_pca_analysis(combined_df):
    # performs pca on colony features to identify variability patterns
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # select numeric features for pca
    numeric_cols = ['area', 'perimeter', 'circularity', 'aspect_ratio', 'solidity']
    available_cols = [col for col in numeric_cols if col in combined_df.columns]
    
    if len(available_cols) < 2:
        return None
    
    # prepare data
    feature_data = combined_df[available_cols].dropna()
    samples = combined_df.loc[feature_data.index, 'sample']
    
    # standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(feature_data)
    
    # run pca
    pca = PCA(n_components=min(len(available_cols), 5))
    pca_data = pca.fit_transform(scaled_data)
    
    # create pca dataframe
    pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])
    pca_df['sample'] = samples.values
    
    # calculate variance by sample
    sample_variance = {}
    for sample in pca_df['sample'].unique():
        sample_data = pca_df[pca_df['sample'] == sample]
        # calculate variance across first 2 pcs
        variance = sample_data[['PC1', 'PC2']].var().sum()
        sample_variance[sample] = variance
    
    return {
        'pca_df': pca_df,
        'explained_variance': pca.explained_variance_ratio_,
        'feature_names': available_cols,
        'sample_variance': sample_variance,
        'most_variable': max(sample_variance, key=sample_variance.get),
        'least_variable': min(sample_variance, key=sample_variance.get)
    }

def run_similarity_analysis(combined_df):
    # calculates similarity between samples based on colony characteristics
    import scipy.stats as stats
    from scipy.spatial.distance import pdist, squareform
    
    # group by sample and calculate summary statistics
    sample_stats = combined_df.groupby('sample').agg({
        'area': ['mean', 'std', 'count'],
        'circularity': ['mean', 'std'] if 'circularity' in combined_df.columns else ['count'],
        'aspect_ratio': ['mean', 'std'] if 'aspect_ratio' in combined_df.columns else ['count']
    }).round(3)
    
    # flatten column names
    sample_stats.columns = ['_'.join(col).strip() for col in sample_stats.columns]
    
    # calculate pairwise distances
    numeric_cols = sample_stats.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        distances = pdist(sample_stats[numeric_cols].fillna(0), metric='euclidean')
        similarity_matrix = 1 / (1 + squareform(distances))
        np.fill_diagonal(similarity_matrix, 1.0)
        
        similarity_df = pd.DataFrame(
            similarity_matrix, 
            index=sample_stats.index, 
            columns=sample_stats.index
        )
    else:
        similarity_df = None
    
    return {
        'sample_stats': sample_stats,
        'similarity_matrix': similarity_df
    }

def display_multi_image_results(results):
    # displays multi-image comparison results with pca and similarity plots
    st.header("Multi-Image Analysis Results")
    
    # overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", results['sample_count'])
    with col2:
        st.metric("Total Colonies", results['total_colonies'])
    with col3:
        avg_colonies = results['total_colonies'] / results['sample_count']
        st.metric("Avg Colonies/Sample", f"{avg_colonies:.1f}")
    with col4:
        if results['pca_results']:
            most_variable = results['pca_results']['most_variable']
            st.metric("Most Variable", most_variable)
    
    # create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Sample Overview",
        "PCA Analysis", 
        "Similarity Analysis",
        "Colony Comparison",
        "Statistical Tests"
    ])
    
    with tab1:
        display_sample_overview(results)
    
    with tab2:
        display_pca_results(results)
    
    with tab3:
        display_similarity_results(results)
    
    with tab4:
        display_colony_comparison(results)
    
    with tab5:
        display_statistical_tests(results)

def display_sample_overview(results):
    # shows overview of each sample
    st.subheader("Sample Summary")
    
    # sample statistics table
    sample_summary = []
    for sample_name, sample_results in results['individual_results'].items():
        if sample_results and 'colony_properties' in sample_results:
            colony_count = len(sample_results['colony_properties'])
            avg_area = np.mean([prop.area for prop in sample_results['colony_properties']])
            sample_summary.append({
                'Sample': sample_name,
                'Colony Count': colony_count,
                'Average Area': f"{avg_area:.1f}",
                'Status': 'Analyzed'
            })
    
    if sample_summary:
        summary_df = pd.DataFrame(sample_summary)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # show sample images grid
    st.subheader("Sample Images")
    if len(results['individual_results']) > 0:
        cols = st.columns(min(3, len(results['individual_results'])))
        for i, (sample_name, sample_results) in enumerate(results['individual_results'].items()):
            with cols[i % 3]:
                if sample_results and 'original_image' in sample_results:
                    st.image(sample_results['original_image'], caption=sample_name, use_container_width=True)

def display_pca_results(results):
    # displays pca analysis results
    import plotly.express as px
    
    st.subheader("Principal Component Analysis")
    
    if not results['pca_results']:
        st.warning("PCA analysis not available - insufficient numeric features")
        return
    
    pca_data = results['pca_results']
    
    # variance explained
    st.write("**Variance Explained by Components:**")
    for i, var in enumerate(pca_data['explained_variance']):
        st.write(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")
    
    # pca scatter plot
    st.subheader("PCA Scatter Plot")
    fig_pca = px.scatter(
        pca_data['pca_df'], 
        x='PC1', y='PC2', 
        color='sample',
        title="Sample Variability in Colony Feature Space",
        labels={'PC1': f"PC1 ({pca_data['explained_variance'][0]*100:.1f}%)",
                'PC2': f"PC2 ({pca_data['explained_variance'][1]*100:.1f}%)"}
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # download button
    pca_bytes = plotly_to_bytes(fig_pca)
    if pca_bytes:
        st.download_button(
            label="Download PCA Plot",
            data=pca_bytes,
            file_name="pca_analysis.png",
            mime="image/png"
        )
    
    # variability ranking
    st.subheader("Sample Variability Ranking")
    variance_df = pd.DataFrame.from_dict(
        pca_data['sample_variance'], 
        orient='index', 
        columns=['Variance Score']
    ).sort_values('Variance Score', ascending=False)
    
    fig_var = px.bar(
        variance_df.reset_index(),
        x='index', y='Variance Score',
        title="Microbiome Variability by Sample",
        labels={'index': 'Sample', 'Variance Score': 'Variance Score'}
    )
    st.plotly_chart(fig_var, use_container_width=True)
    
    # download button
    variance_bytes = plotly_to_bytes(fig_var)
    if variance_bytes:
        st.download_button(
            label="Download Variability Chart",
            data=variance_bytes,
            file_name="sample_variability.png",
            mime="image/png"
        )
    
    st.write(f"**Most variable microbiome:** {pca_data['most_variable']}")
    st.write(f"**Least variable microbiome:** {pca_data['least_variable']}")

def display_similarity_results(results):
    # displays similarity analysis results
    import plotly.express as px
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    st.subheader("Sample Similarity Analysis")
    
    if not results['similarity_results'] or results['similarity_results']['similarity_matrix'] is None:
        st.warning("Similarity analysis not available")
        return
    
    similarity_data = results['similarity_results']
    
    # similarity heatmap
    st.subheader("Sample Similarity Matrix")
    fig_sim, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        similarity_data['similarity_matrix'], 
        annot=True, 
        cmap='viridis', 
        fmt='.3f',
        ax=ax
    )
    ax.set_title("Pairwise Sample Similarity")
    st.pyplot(fig_sim)
    
    # download button
    similarity_bytes = matplotlib_to_bytes(fig_sim)
    st.download_button(
        label="Download Similarity Matrix",
        data=similarity_bytes,
        file_name="similarity_matrix.png",
        mime="image/png"
    )
    
    # sample statistics
    st.subheader("Sample Statistics")
    st.dataframe(similarity_data['sample_stats'].round(3), use_container_width=True)

def display_colony_comparison(results):
    # displays colony feature comparisons across samples
    import plotly.express as px
    
    st.subheader("Colony Feature Comparison")
    
    combined_df = results['combined_df']
    
    # feature distribution plots
    numeric_features = ['area', 'perimeter', 'circularity', 'aspect_ratio', 'solidity']
    available_features = [f for f in numeric_features if f in combined_df.columns]
    
    if available_features:
        selected_feature = st.selectbox("Select feature to compare:", available_features)
        
        # box plot
        fig_box = px.box(
            combined_df, 
            x='sample', y=selected_feature,
            title=f"{selected_feature.title()} Distribution by Sample"
        )
        fig_box.update_xaxes(tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # download button for box plot
        box_bytes = plotly_to_bytes(fig_box)
        if box_bytes:
            st.download_button(
                label="Download Box Plot",
                data=box_bytes,
                file_name=f"{selected_feature}_boxplot.png",
                mime="image/png"
            )
        
        # violin plot
        fig_violin = px.violin(
            combined_df, 
            x='sample', y=selected_feature,
            title=f"{selected_feature.title()} Density by Sample"
        )
        fig_violin.update_xaxes(tickangle=45)
        st.plotly_chart(fig_violin, use_container_width=True)
        
        # download button for violin plot
        violin_bytes = plotly_to_bytes(fig_violin)
        if violin_bytes:
            st.download_button(
                label="Download Violin Plot",
                data=violin_bytes,
                file_name=f"{selected_feature}_violin.png",
                mime="image/png"
            )

def display_statistical_tests(results):
    # displays statistical test results between samples
    import scipy.stats as stats
    
    st.subheader("Statistical Comparisons")
    
    combined_df = results['combined_df']
    samples = combined_df['sample'].unique()
    
    if len(samples) < 2:
        st.warning("Need at least 2 samples for statistical comparison")
        return
    
    # anova for area differences
    if 'area' in combined_df.columns:
        sample_groups = [combined_df[combined_df['sample'] == sample]['area'].dropna() for sample in samples]
        
        if all(len(group) > 0 for group in sample_groups):
            f_stat, p_value = stats.f_oneway(*sample_groups)
            st.write("**One-way ANOVA for Colony Area:**")
            st.write(f"P-value: {p_value:.6f}")
    
    # pairwise t-tests
    if len(samples) == 2 and 'area' in combined_df.columns:
        group1 = combined_df[combined_df['sample'] == samples[0]]['area'].dropna()
        group2 = combined_df[combined_df['sample'] == samples[1]]['area'].dropna()
        
        if len(group1) > 0 and len(group2) > 0:
            t_stat, p_value = stats.ttest_ind(group1, group2)
            st.write("**Independent t-test for Colony Area:**")
            st.write(f"P-value: {p_value:.6f}")

if __name__ == "__main__":
    main() 