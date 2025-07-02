# demo.py
# demo script showing the bacterial colony analysis functionality
# creates sample data to demonstrate the app features

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from colony_analyzer import ColonyAnalyzer

def create_sample_petri_dish():
    """create a sample petri dish image with simulated colonies"""
    print("creating sample petri dish image")
    
    # create base image (petri dish)
    width, height = 800, 600
    image = np.ones((height, width, 3), dtype=np.uint8) * 240  # light gray background
    
    # add petri dish circle
    center = (width // 2, height // 2)
    radius = min(width, height) // 2 - 50
    
    # draw petri dish
    cv2.circle(image, center, radius, (200, 200, 200), 3)
    cv2.circle(image, center, radius - 20, (180, 180, 180), 1)
    
    # create sample colonies
    colonies = []
    np.random.seed(42)  # for reproducible results
    
    # add different types of colonies
    colony_types = [
        {'color': (100, 150, 100), 'size_range': (20, 40), 'count': 15},  # green colonies
        {'color': (150, 100, 100), 'size_range': (15, 35), 'count': 12},  # red colonies
        {'color': (100, 100, 150), 'size_range': (25, 45), 'count': 10},  # blue colonies
        {'color': (150, 150, 100), 'size_range': (10, 30), 'count': 8},   # yellow colonies
    ]
    
    for colony_type in colony_types:
        for _ in range(colony_type['count']):
            # random position within petri dish
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(50, radius - 50)
            x = int(center[0] + distance * np.cos(angle))
            y = int(center[1] + distance * np.sin(angle))
            
            # random size
            size = np.random.uniform(*colony_type['size_range'])
            
            # add some variation to color
            color = list(colony_type['color'])
            color = [max(0, min(255, c + np.random.normal(0, 20))) for c in color]
            
            # draw colony
            cv2.circle(image, (x, y), int(size), color, -1)
            
            # add some texture
            for _ in range(3):
                offset_x = np.random.normal(0, size * 0.3)
                offset_y = np.random.normal(0, size * 0.3)
                cv2.circle(image, (int(x + offset_x), int(y + offset_y)), 
                          int(size * 0.3), color, -1)
            
            colonies.append({
                'x': x, 'y': y, 'size': size, 'color': color
            })
    
    return image, colonies

def create_sample_analysis_results(colonies):
    """create sample analysis results for demonstration"""
    print("creating sample analysis results")
    
    # morphology data
    morph_data = []
    for i, colony in enumerate(colonies):
        area = np.pi * colony['size'] ** 2
        perimeter = 2 * np.pi * colony['size']
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # add some variation
        aspect_ratio = 1 + np.random.normal(0, 0.2)
        solidity = 0.8 + np.random.normal(0, 0.1)
        
        # classify form
        if circularity > 0.85:
            form = 'circular'
        elif aspect_ratio > 1.5:
            form = 'oval'
        else:
            form = 'irregular'
        
        # classify margin
        margin_options = ['entire', 'undulate', 'lobate', 'serrate']
        margin = np.random.choice(margin_options, p=[0.4, 0.3, 0.2, 0.1])
        
        morph_data.append({
            'colony_id': i,
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'extent': 0.7 + np.random.normal(0, 0.1),
            'convexity': 0.9 + np.random.normal(0, 0.05),
            'margin_complexity': np.random.exponential(0.1),
            'form': form,
            'margin': margin
        })
    
    # color data
    color_data = []
    color_clusters = {}
    cluster_id = 0
    
    for i, colony in enumerate(colonies):
        color = colony['color']
        color_key = tuple(color)
        
        if color_key not in color_clusters:
            color_clusters[color_key] = cluster_id
            cluster_id += 1
        
        color_data.append({
            'colony_id': i,
            'area': morph_data[i]['area'],
            'color_cluster': color_clusters[color_key],
            'color_class': f"color_group_{color_clusters[color_key]}"
        })
    
    # density data
    density_data = []
    for i, colony in enumerate(colonies):
        # simulate density based on color intensity
        color = colony['color']
        intensity = sum(color) / 3
        
        if intensity < 120:
            density_class = 'dense'
            opacity_score = 2.0 + np.random.normal(0, 0.3)
        elif intensity < 140:
            density_class = 'medium'
            opacity_score = 1.5 + np.random.normal(0, 0.3)
        else:
            density_class = 'sparse'
            opacity_score = 0.8 + np.random.normal(0, 0.3)
        
        density_data.append({
            'colony_id': i,
            'area': morph_data[i]['area'],
            'mean_intensity': intensity,
            'opacity_score': opacity_score,
            'density_uniformity': 0.8 + np.random.normal(0, 0.1),
            'density_gradient': np.random.normal(0, 0.5),
            'texture_score': np.random.exponential(10),
            'mean_saturation': np.random.uniform(0.3, 0.8),
            'density_class': density_class
        })
    
    return pd.DataFrame(morph_data), color_data, pd.DataFrame(density_data)

def run_demo():
    """run the complete demo"""
    print("🔬 Bacterial Colony Analysis Demo")
    print("=" * 50)
    
    # create sample image
    sample_image, colonies = create_sample_petri_dish()
    
    # save sample image
    cv2.imwrite("sample_petri_dish.jpg", cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR))
    print("✅ created sample petri dish image: sample_petri_dish.jpg")
    
    # create sample analysis results
    morph_df, color_data, density_df = create_sample_analysis_results(colonies)
    
    # combine data
    combined_df = morph_df.copy()
    color_df = pd.DataFrame(color_data)
    combined_df = combined_df.merge(color_df, on='colony_id', how='outer')
    combined_df = combined_df.merge(density_df, on='colony_id', how='outer')
    combined_df = combined_df.fillna(0)
    
    # calculate scores (simplified version)
    scores = pd.DataFrame({'colony_id': combined_df['colony_id']})
    
    # morphology score
    morph_vals = []
    for f in ['circularity', 'aspect_ratio', 'solidity', 'convexity']:
        v = combined_df[f].fillna(combined_df[f].median())
        pct = v.rank(pct=True)
        morph_vals.append(np.minimum(pct, 1-pct) * 2)
    scores['morphology_score'] = np.mean(morph_vals, axis=0) if morph_vals else 0.5
    
    # density score
    if 'opacity_score' in combined_df:
        norm = combined_df['opacity_score'] / combined_df['opacity_score'].max()
        scores['density_score'] = np.clip(norm ** 0.5 * 1.5, 0, 1)
    else:
        scores['density_score'] = 0.5
    
    # final interest score
    scores['bio_interest'] = (scores['morphology_score'] * 0.6 + scores['density_score'] * 0.4)
    
    # attach metadata
    for c in ['color_cluster', 'form', 'area', 'density_class']:
        if c in combined_df:
            scores[c] = combined_df[c]
    
    # select top colonies
    top_colonies = scores.nlargest(10, 'bio_interest')
    
    # display results
    print(f"\n📊 Analysis Results:")
    print(f"Total colonies analyzed: {len(colonies)}")
    print(f"Average colony area: {combined_df['area'].mean():.0f} px²")
    print(f"Colony forms found: {combined_df['form'].value_counts().to_dict()}")
    print(f"Color clusters: {len(combined_df['color_cluster'].unique())}")
    print(f"Density classes: {combined_df['density_class'].value_counts().to_dict()}")
    
    print(f"\n🏆 Top 5 Colonies by Interest Score:")
    for i, (_, row) in enumerate(top_colonies.head().iterrows()):
        print(f"{i+1}. Colony {row['colony_id']}: Score {row['bio_interest']:.3f}, "
              f"Form: {row['form']}, Density: {row['density_class']}")
    
    # create visualizations
    print("\n📈 Creating visualizations...")
    
    # form distribution
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    form_counts = combined_df['form'].value_counts()
    plt.bar(form_counts.index, form_counts.values)
    plt.title('Colony Form Distribution')
    plt.ylabel('Count')
    
    # density distribution
    plt.subplot(2, 3, 2)
    density_counts = combined_df['density_class'].value_counts()
    plt.pie(density_counts.values, labels=density_counts.index, autopct='%1.1f%%')
    plt.title('Density Class Distribution')
    
    # area vs circularity
    plt.subplot(2, 3, 3)
    plt.scatter(combined_df['area'], combined_df['circularity'], 
               c=combined_df['color_cluster'], alpha=0.6)
    plt.xlabel('Area (px²)')
    plt.ylabel('Circularity')
    plt.title('Area vs Circularity')
    plt.colorbar(label='Color Cluster')
    
    # interest score distribution
    plt.subplot(2, 3, 4)
    plt.hist(scores['bio_interest'], bins=15, alpha=0.7)
    plt.xlabel('Interest Score')
    plt.ylabel('Count')
    plt.title('Interest Score Distribution')
    
    # area vs interest score
    plt.subplot(2, 3, 5)
    plt.scatter(combined_df['area'], scores['bio_interest'], 
               c=combined_df['color_cluster'], alpha=0.6)
    plt.xlabel('Area (px²)')
    plt.ylabel('Interest Score')
    plt.title('Area vs Interest Score')
    plt.colorbar(label='Color Cluster')
    
    # sample image
    plt.subplot(2, 3, 6)
    plt.imshow(sample_image)
    plt.title('Sample Petri Dish')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
    print("✅ saved demo results visualization: demo_results.png")
    
    # save data
    combined_df.to_csv('demo_colony_data.csv', index=False)
    scores.to_csv('demo_scores.csv', index=False)
    print("✅ saved demo data: demo_colony_data.csv, demo_scores.csv")
    
    print("\n🎉 Demo completed successfully!")
    print("Files created:")
    print("- sample_petri_dish.jpg (sample image)")
    print("- demo_results.png (visualizations)")
    print("- demo_colony_data.csv (colony data)")
    print("- demo_scores.csv (scoring results)")
    
    return {
        'sample_image': sample_image,
        'colonies': colonies,
        'morph_df': morph_df,
        'color_data': color_data,
        'density_df': density_df,
        'combined_df': combined_df,
        'scores': scores,
        'top_colonies': top_colonies
    }

if __name__ == "__main__":
    results = run_demo() 