# debug_parameters.py
# script to debug parameter differences between notebook and app

import cv2
import numpy as np
from colony_analyzer import ColonyAnalyzer

def debug_analysis(image_path):
    """Run analysis with default parameters and print all intermediate results"""
    
    print("=== COLONY ANALYZER PARAMETERS ===")
    
    # Default parameters from ColonyAnalyzer
    default_params = {
        'bilateral_d': 9,
        'bilateral_sigma_color': 75,
        'bilateral_sigma_space': 75,
        'clahe_clip_limit': 3.0,
        'clahe_tile_grid': (8,8),
        'gamma': 1.2,
        'sharpen_strength': 1.0,
        'margin_percent': 0.08,
        'adaptive_block_size': 15,
        'adaptive_c': 3,
        'min_colony_size': 15,
        'max_colony_size': 10000,
        'min_distance': 8,
        'watershed': True,
        'color_n_clusters': None,
        'color_random_state': 42,
        'color_n_init': 10,
        'n_top_colonies': 50,
        'penalty_factor': 0.5
    }
    
    print("Default parameters:")
    for key, value in default_params.items():
        print(f"  {key}: {value}")
    
    print("\n=== RUNNING ANALYSIS ===")
    
    # Create analyzer with default parameters
    analyzer = ColonyAnalyzer(**default_params)
    
    # Load image
    print("Loading image...")
    original_image = analyzer.load_image(image_path)
    if original_image is None:
        print("Failed to load image")
        return
    
    print(f"Image shape: {original_image.shape}")
    print(f"Image dtype: {original_image.dtype}")
    print(f"Image min/max: {original_image.min()}/{original_image.max()}")
    
    # Preprocess image
    print("Preprocessing image...")
    processed_image = analyzer.preprocess_image(original_image)
    print(f"Processed image shape: {processed_image.shape}")
    print(f"Processed image min/max: {processed_image.min()}/{processed_image.max()}")
    
    # Detect plate
    print("Detecting plate...")
    plate_mask, plate_info = analyzer.detect_plate(processed_image)
    print(f"Plate info: {plate_info}")
    
    # Segment colonies
    print("Segmenting colonies...")
    colony_labels, colony_properties = analyzer.segment_colonies(processed_image, plate_mask)
    print(f"Found {len(colony_properties)} colonies")
    
    # Print first few colony properties
    print("\nFirst 5 colonies:")
    for i, prop in enumerate(colony_properties[:5]):
        print(f"  Colony {i}: area={prop.area}, centroid={prop.centroid}")
    
    # Run full analysis
    print("\nRunning full analysis...")
    results = analyzer.run_full_analysis(image_path)
    
    if results is not None:
        print(f"\n=== ANALYSIS RESULTS ===")
        print(f"Total colonies: {len(results['colony_properties'])}")
        
        if 'morph_df' in results and not results['morph_df'].empty:
            print(f"Morphology data: {len(results['morph_df'])} rows")
            print("First 5 morphology rows:")
            print(results['morph_df'].head())
        
        if 'density_df' in results and not results['density_df'].empty:
            print(f"Density data: {len(results['density_df'])} rows")
            print("First 5 density rows:")
            print(results['density_df'].head())
        
        if 'combined_df' in results and not results['combined_df'].empty:
            print(f"Combined data: {len(results['combined_df'])} rows")
            print("First 5 combined rows:")
            print(results['combined_df'].head())
        
        if 'scores_df' in results and not results['scores_df'].empty:
            print(f"Scores data: {len(results['scores_df'])} rows")
            print("First 5 scores rows:")
            print(results['scores_df'].head())
        
        if 'top_colonies' in results and not results['top_colonies'].empty:
            print(f"Top colonies: {len(results['top_colonies'])} rows")
            print("Top 10 colonies:")
            print(results['top_colonies'].head(10))
    
    return results

if __name__ == "__main__":
    # Replace with your image path
    image_path = "your_image.jpg"  # Change this to your actual image path
    results = debug_analysis(image_path) 