# compare_results.py
# script to compare results between notebook and app

import pandas as pd
import numpy as np

def compare_results(notebook_results, app_results):
    """Compare results between notebook and app outputs"""
    
    print("=== COMPARING NOTEBOOK vs APP RESULTS ===")
    
    # Notebook results (from your example)
    notebook_data = {
        'colony_id': [0, 32, 71, 90, 44, 305],
        'morphology_score': [0.515, 0.454, 0.581, 0.222, 0.728, 0.204],
        'form_score': [0.774, 0.573, 0.871, 0.947, 0.774, 0.947],
        'size_score': [0.5, 1.0, 1.0, 0.5, 1.0, 0.5],
        'density_score': [1.000, 1.000, 1.000, 1.000, 1.000, 0.869],
        'density_bonus': [0.3, 0.3, 0.0, 0.0, 0.3, 0.0],
        'density_class_bonus': [0.1, 0.4, 0.1, 0.1, 0.1, 0.0],
        'margin_score': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        'bio_base': [0.708, 0.716, 0.790, 0.622, 0.832, 0.570],
        'novelty_combo': [0.293, 0.000, 0.293, 0.298, 0.000, 0.293],
        'penalty': [0.018, 0.085, 0.009, 0.002, 0.026, 0.019],
        'bio_interest': [1.000, 1.000, 1.000, 1.000, 1.000, 0.845],
        'color_cluster': [4, 1, 3, 5, 2, 0],
        'form': ['irregular', 'filamentous', 'oval', 'circular', 'irregular', 'circular'],
        'margin': ['serrate', 'undulate', 'undulate', 'undulate', 'serrate', 'undulate'],
        'area': [445.0, 151.0, 68.0, 52.0, 114.0, 34.0],
        'density_class': ['medium', 'dense', 'medium', 'medium', 'medium', 'sparse']
    }
    
    notebook_df = pd.DataFrame(notebook_data)
    
    # App results (from your example) - simplified format
    app_data = {
        'colony_id': [0, 8, 55, 71, 235, 314, 7, 62, 2],
        'bio_interest': [1, 1, 1, 1, 1, 0.945, 1, 1, 1],
        'form': ['irregular', 'filamentous', 'circular', 'oval', 'irregular', 'circular', 'filamentous', 'oval', 'irregular'],
        'density_class': ['medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'dense', 'medium', 'medium'],
        'area': [445, 221, 67, 68, 71, 25, 201, 81, 287]
    }
    
    app_df = pd.DataFrame(app_data)
    
    print(f"Notebook colonies: {len(notebook_df)}")
    print(f"App colonies: {len(app_df)}")
    
    print("\n=== NOTEBOOK TOP COLONIES ===")
    print(notebook_df[['colony_id', 'bio_interest', 'form', 'density_class', 'area']].head(10))
    
    print("\n=== APP TOP COLONIES ===")
    print(app_df[['colony_id', 'bio_interest', 'form', 'density_class', 'area']].head(10))
    
    # Compare colony IDs
    notebook_ids = set(notebook_df['colony_id'])
    app_ids = set(app_df['colony_id'])
    
    print(f"\n=== COLONY ID COMPARISON ===")
    print(f"Colonies in notebook only: {notebook_ids - app_ids}")
    print(f"Colonies in app only: {app_ids - notebook_ids}")
    print(f"Colonies in both: {notebook_ids & app_ids}")
    
    # Compare forms
    print(f"\n=== FORM DISTRIBUTION ===")
    print("Notebook forms:")
    print(notebook_df['form'].value_counts())
    print("\nApp forms:")
    print(app_df['form'].value_counts())
    
    # Compare density classes
    print(f"\n=== DENSITY CLASS DISTRIBUTION ===")
    print("Notebook density classes:")
    print(notebook_df['density_class'].value_counts())
    print("\nApp density classes:")
    print(app_df['density_class'].value_counts())
    
    # Compare areas
    print(f"\n=== AREA COMPARISON ===")
    print(f"Notebook area range: {notebook_df['area'].min()} - {notebook_df['area'].max()}")
    print(f"App area range: {app_df['area'].min()} - {app_df['area'].max()}")
    
    # Find matching colonies by area (approximate)
    print(f"\n=== MATCHING COLONIES BY AREA ===")
    for _, notebook_row in notebook_df.iterrows():
        notebook_area = notebook_row['area']
        # Find app colonies with similar area (±10%)
        similar_app = app_df[abs(app_df['area'] - notebook_area) / notebook_area < 0.1]
        if not similar_app.empty:
            print(f"Notebook colony {notebook_row['colony_id']} (area={notebook_area}) matches:")
            for _, app_row in similar_app.iterrows():
                print(f"  App colony {app_row['colony_id']} (area={app_row['area']}, form={app_row['form']})")

def analyze_differences():
    """Analyze potential causes of differences"""
    
    print("\n=== POTENTIAL CAUSES OF DIFFERENCES ===")
    
    causes = [
        "1. Different random seeds in color clustering",
        "2. Different image preprocessing (OpenCV version differences)",
        "3. Different colony detection thresholds",
        "4. Different watershed parameters",
        "5. Different scoring algorithm implementation",
        "6. Different parameter values (even if they look the same)",
        "7. Different image loading/format handling",
        "8. Different numpy/scipy versions affecting calculations"
    ]
    
    for cause in causes:
        print(cause)
    
    print("\n=== RECOMMENDED DEBUGGING STEPS ===")
    steps = [
        "1. Print all parameter values in both notebook and app",
        "2. Save intermediate images (preprocessed, binary mask) from both",
        "3. Compare colony detection results step by step",
        "4. Check random seeds in color clustering",
        "5. Verify image loading is identical",
        "6. Compare scoring calculations line by line"
    ]
    
    for step in steps:
        print(step)

if __name__ == "__main__":
    # Run comparison
    compare_results(None, None)
    analyze_differences() 