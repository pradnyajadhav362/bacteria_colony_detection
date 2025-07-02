# colony_analyzer.py
# bacterial colony detection and analysis pipeline
# converts petri dish images into detailed colony characterizations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import ndimage
from skimage import filters, morphology, measure, segmentation, color
from skimage.feature import peak_local_max
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ColonyAnalyzer:
    def __init__(self,
                 bilateral_d=9,
                 bilateral_sigma_color=75,
                 bilateral_sigma_space=75,
                 clahe_clip_limit=3.0,
                 clahe_tile_grid=(8,8),
                 gamma=1.2,
                 sharpen_strength=1.0,
                 margin_percent=0.08,
                 adaptive_block_size=15,
                 adaptive_c=3,
                 min_colony_size=15,
                 max_colony_size=10000,
                 min_distance=8,
                 watershed=True,
                 color_n_clusters=None,
                 color_random_state=42,
                 color_n_init=10,
                 n_top_colonies=50,  # Always select more colonies than needed for display flexibility
                 penalty_factor=0.5):
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.gamma = gamma
        self.sharpen_strength = sharpen_strength
        self.margin_percent = margin_percent
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.min_colony_size = min_colony_size
        self.max_colony_size = max_colony_size
        self.min_distance = min_distance
        self.watershed = watershed
        self.color_n_clusters = color_n_clusters
        self.color_random_state = color_random_state
        self.color_n_init = color_n_init
        self.n_top_colonies = n_top_colonies
        self.penalty_factor = penalty_factor
        # results
        self.original_image = None
        self.processed_image = None
        self.plate_mask = None
        self.plate_info = None
        self.colony_labels = None
        self.colony_properties = None
        self.morph_df = None
        self.colony_data = None
        self.density_df = None
        self.combined_df = None
        self.scores_df = None
        self.top_colonies = None
        self.final_binary_mask = None
        
    def load_image(self, image_path):
        # load and convert image to rgb format
        print("loading microbiome plate image")
        
        original_image = cv2.imread(image_path)
        if original_image is None:
            print("error loading image")
            return None
            
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        self.original_image = original_image
        
        h, w, c = original_image.shape
        print(f"image dimensions: {w}x{h}, channels: {c}")
        
        return original_image
    
    def preprocess_image(self, original_image):
        # denoise, enhance contrast, apply gamma correction, and sharpen
        print("cleaning and enhancing image quality")
        
        img = original_image.copy()
        
        # denoise while keeping edges sharp
        img_denoised = cv2.bilateralFilter(img, self.bilateral_d, self.bilateral_sigma_color, self.bilateral_sigma_space)
        
        # enhance contrast using CLAHE
        lab = cv2.cvtColor(img_denoised, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # gamma correction for better colony visibility
        img_gamma = np.power(img_enhanced / 255.0, self.gamma) * 255
        img_gamma = img_gamma.astype(np.uint8)
        
        # sharpen image slightly
        if self.sharpen_strength > 0:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * self.sharpen_strength
            img_sharpened = cv2.filter2D(img_gamma, -1, kernel)
        else:
            img_sharpened = img_gamma
        
        self.processed_image = img_sharpened
        print("preprocessing complete")
        return img_sharpened
    
    def detect_plate(self, processed_image):
        # find inner rectangular region of plate and compute metrics
        print("detecting inner plate area without edges")
        
        gray = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # create inner margin to exclude plate edges
        margin_h = int(h * self.margin_percent)
        margin_w = int(w * self.margin_percent)
        
        # create inner rectangular mask
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        inner_mask[margin_h:h-margin_h, margin_w:w-margin_w] = 255
        
        # refine using intensity analysis
        masked_gray = cv2.bitwise_and(gray, gray, mask=inner_mask)
        
        # find actual plate content area
        _, thresh = cv2.threshold(masked_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # find largest contour in inner area
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            refined_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(refined_mask, [largest_contour], 255)
            
            # combine with inner margin
            final_mask = cv2.bitwise_and(inner_mask, refined_mask)
        else:
            final_mask = inner_mask
            
        # get bounding box info
        x, y, rect_w, rect_h = margin_w, margin_h, w-2*margin_w, h-2*margin_h
        
        plate_info = {
            'center': (w//2, h//2),
            'bbox': (x, y, rect_w, rect_h),
            'area_pixels': rect_w * rect_h,
            'inner_area': np.sum(final_mask > 0)
        }
        
        self.plate_mask = final_mask
        self.plate_info = plate_info
        
        print(f"inner plate area: {rect_w}x{rect_h}, excluding {self.margin_percent*100}% edge margin")
        return final_mask, plate_info
    
    def segment_colonies(self, processed_image, plate_mask):
        # identify each bacterial colony as separate blob within dish boundary
        print("segmenting colonies in rectangular plate")
        
        # apply rectangular plate mask first
        masked_img = cv2.bitwise_and(processed_image, processed_image, mask=plate_mask)
        
        gray = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)
        
        # adaptive threshold with larger block size for rectangular plates
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, self.adaptive_block_size, self.adaptive_c)
        
        # apply plate mask to binary
        binary = cv2.bitwise_and(binary, plate_mask)
        
        # morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel_medium)
        
        # watershed for touching colonies
        distance = ndimage.distance_transform_edt(binary_clean)
        coords = peak_local_max(distance, min_distance=self.min_distance, threshold_abs=0.3*distance.max())
        
        if self.watershed and len(coords) > 0:
            markers = np.zeros_like(binary_clean, dtype=np.int32)
            for i, coord in enumerate(coords):
                markers[coord[0], coord[1]] = i + 1
            labels = segmentation.watershed(-distance, markers, mask=binary_clean)
        else:
            num_labels, labels = cv2.connectedComponents(binary_clean)
        
        # filter by size
        min_colony_size = self.min_colony_size
        max_colony_size = self.max_colony_size
        colony_props = measure.regionprops(labels)
        
        valid_colonies = []
        valid_label_mask = np.zeros_like(labels)
        
        for new_id, prop in enumerate(colony_props, start=1):
            if min_colony_size <= prop.area <= max_colony_size:
                valid_colonies.append(prop)
                valid_label_mask[labels == prop.label] = new_id
        
        self.colony_labels = valid_label_mask
        self.colony_properties = valid_colonies
        self.final_binary_mask = (valid_label_mask > 0).astype(np.uint8) * 255
        
        print(f"found {len(valid_colonies)} colonies in rectangular plate")
        return valid_label_mask, valid_colonies
    
    def analyze_morphology(self, colony_labels, colony_properties):
        # measure each colony's shape and classify edge style
        print("analyzing colony morphology")
        
        data = []
        for idx, prop in enumerate(colony_properties):
            area = prop.area
            perimeter = prop.perimeter
            major_axis = prop.major_axis_length
            minor_axis = prop.minor_axis_length
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1
            solidity = prop.solidity
            extent = prop.extent
            
            mask = (colony_labels == prop.label).astype('uint8')
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                convexity = area / hull_area if hull_area > 0 else 1
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                margin_complexity = len(approx) / area * 1000
                
                if convexity > 0.95 and margin_complexity < 0.1:
                    margin = 'entire'
                elif convexity > 0.85:
                    margin = 'undulate'
                elif margin_complexity > 0.3:
                    margin = 'serrate'
                else:
                    margin = 'lobate'
            else:
                convexity = 1
                margin_complexity = 0
                margin = 'unknown'
                
            if circularity > 0.85:
                form = 'circular'
            elif aspect_ratio > 2.0:
                form = 'filamentous'
            elif circularity < 0.6:
                form = 'irregular'
            else:
                form = 'oval'
                
            data.append({
                'colony_id': idx,
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'extent': extent,
                'convexity': convexity,
                'margin_complexity': margin_complexity,
                'form': form,
                'margin': margin
            })
        
        self.morph_df = pd.DataFrame(data)
        return self.morph_df
    
    def extract_dominant_colors(self, colony_pixels, n_colors=3):
        # get dominant colors instead of mean
        if len(colony_pixels) < 10:
            return np.mean(colony_pixels, axis=0)
        
        pixels_reshaped = colony_pixels.reshape(-1, 3)
        kmeans = KMeans(n_clusters=min(n_colors, len(pixels_reshaped)),
                       random_state=self.color_random_state, n_init=self.color_n_init)
        kmeans.fit(pixels_reshaped)
        
        labels = kmeans.labels_
        from collections import Counter
        label_counts = Counter(labels)
        dominant_label = label_counts.most_common(1)[0][0]
        
        return kmeans.cluster_centers_[dominant_label]
    
    def rgb_to_lab_batch(self, rgb_colors):
        # convert rgb to lab color space
        rgb_normalized = rgb_colors / 255.0
        lab_colors = []
        
        for rgb in rgb_normalized:
            rgb_img = rgb.reshape(1, 1, 3)
            lab_img = color.rgb2lab(rgb_img)
            lab_colors.append(lab_img[0, 0])
        
        return np.array(lab_colors)
    
    def analyze_colors(self, processed_image, colony_labels, colony_properties):
        # pick dominant color of each colony and group similar ones
        print("analyzing colony colors with kmeans clustering")
        
        if len(colony_properties) == 0:
            return [], []
        
        colony_data = []
        rgb_colors = []
        
        # extract dominant colors
        for i, prop in enumerate(colony_properties):
            mask = (colony_labels == prop.label)
            colony_pixels = processed_image[mask]
            
            if len(colony_pixels) > 0:
                dominant_rgb = self.extract_dominant_colors(colony_pixels)
                
                colony_info = {
                    'colony_id': i,
                    'label': prop.label,
                    'area': prop.area,
                    'centroid': prop.centroid,
                    'dominant_color': dominant_rgb
                }
                
                colony_data.append(colony_info)
                rgb_colors.append(dominant_rgb)
        
        if len(rgb_colors) == 0:
            print("no valid colonies found")
            return [], []
        
        rgb_colors = np.array(rgb_colors)
        lab_colors = self.rgb_to_lab_batch(rgb_colors)
        
        print(f"extracted colors from {len(colony_data)} colonies")
        
        # find optimal number of clusters using elbow method
        if len(lab_colors) > 1:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            lab_scaled = scaler.fit_transform(lab_colors)
            
            best_k = 3
            if len(lab_colors) > 4:
                inertias = []
                k_range = range(2, min(8, len(lab_colors)))
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=self.color_random_state, n_init=self.color_n_init)
                    kmeans.fit(lab_scaled)
                    inertias.append(kmeans.inertia_)
                
                # find elbow point
                if len(inertias) > 2:
                    diffs = np.diff(inertias)
                    best_k = k_range[np.argmax(diffs)] if len(diffs) > 0 else 3
            
            # final clustering
            kmeans = KMeans(n_clusters=best_k, random_state=self.color_random_state, n_init=self.color_n_init)
            clusters = kmeans.fit_predict(lab_scaled)
            
            print(f"kmeans found {best_k} color groups")
        else:
            clusters = np.zeros(len(colony_data))
            best_k = 1
        
        # assign clusters to colony data
        for i, cluster in enumerate(clusters):
            colony_data[i]['color_cluster'] = int(cluster)
        
        self.colony_data = colony_data
        return colony_data, clusters
    
    def analyze_density(self, processed_image, colony_labels, colony_properties, plate_mask):
        # quantify how dense or see-through each colony appears
        print("analyzing colony density")
        
        gray_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
        hsv_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2HSV)
        
        background_mask = plate_mask.copy()
        background_mask[colony_labels > 0] = 0
        background_pixels = gray_image[background_mask > 0]
        background_mean = np.mean(background_pixels) if len(background_pixels) > 0 else 128
        background_std = np.std(background_pixels) if len(background_pixels) > 0 else 30
        
        colony_density_data = []
        
        for i, prop in enumerate(colony_properties):
            minr, minc, maxr, maxc = prop.bbox
            colony_region_gray = gray_image[minr:maxr, minc:maxc]
            colony_region_hsv = hsv_image[minr:maxr, minc:maxc]
            colony_mask_region = (colony_labels[minr:maxr, minc:maxc] == prop.label)
            
            if not np.any(colony_mask_region):
                continue
            
            colony_pixels_gray = colony_region_gray[colony_mask_region]
            colony_pixels_hsv = colony_region_hsv[colony_mask_region]
            
            mean_intensity = np.mean(colony_pixels_gray)
            std_intensity = np.std(colony_pixels_gray)
            opacity_score = abs(mean_intensity - background_mean) / background_std
            density_uniformity = 1.0 / (std_intensity + 1)
            
            distance_transform = ndimage.distance_transform_edt(colony_mask_region)
            max_distance = np.max(distance_transform)
            
            if max_distance > 3:
                center_mask = distance_transform >= max_distance * 0.5
                edge_mask = distance_transform <= max_distance * 0.3
                center_pixels = colony_region_gray[colony_mask_region & center_mask]
                edge_pixels = colony_region_gray[colony_mask_region & edge_mask]
                center_density = np.mean(center_pixels) if len(center_pixels) > 0 else mean_intensity
                edge_density = np.mean(edge_pixels) if len(edge_pixels) > 0 else mean_intensity
                density_gradient = (center_density - edge_density) / background_std
            else:
                center_density = edge_density = mean_intensity
                density_gradient = 0
            
            kernel_size = max(3, min(7, int(np.sqrt(prop.area) // 3)))
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            local_var = ndimage.generic_filter(colony_region_gray.astype(float), np.var, size=kernel_size)
            texture_score = np.mean(local_var[colony_mask_region])
            mean_saturation = np.mean(colony_pixels_hsv[:, 1])
            
            if opacity_score > 2.0 and density_uniformity > 0.1:
                density_class = "very_dense"
            elif opacity_score > 1.5:
                density_class = "dense"
            elif opacity_score > 0.8:
                density_class = "medium"
            elif opacity_score > 0.4:
                density_class = "sparse"
            else:
                density_class = "very_sparse"
            
            colony_info = {
                'colony_id': i,
                'area': prop.area,
                'mean_intensity': mean_intensity,
                'opacity_score': opacity_score,
                'density_uniformity': density_uniformity,
                'density_gradient': density_gradient,
                'texture_score': texture_score,
                'mean_saturation': mean_saturation,
                'density_class': density_class
            }
            
            colony_density_data.append(colony_info)
        
        self.density_df = pd.DataFrame(colony_density_data)
        print("done with density analysis")
        return self.density_df
    
    def combine_analyses(self, morph_df, colony_data, density_df):
        # combine morphology, color, density data and calculate comprehensive scores
        print("combining all colony analysis data")
        
        # convert color data to dataframe if needed
        if isinstance(colony_data, list):
            color_df = pd.DataFrame([{
                'colony_id': data['colony_id'],
                'area': data['area'],
                'color_cluster': data.get('color_cluster', 0),
                'color_class': f"color_group_{data.get('color_cluster', 0)}"
            } for data in colony_data])
        else:
            color_df = colony_data
        
        # merge all dataframes
        combined_df = morph_df.copy()
        if not color_df.empty:
            combined_df = combined_df.merge(color_df, on='colony_id', how='outer')
        if not density_df.empty:
            combined_df = combined_df.merge(density_df, on='colony_id', how='outer')
        
        # fill missing values
        combined_df = combined_df.fillna(0)
        
        self.combined_df = combined_df
        print(f"combined data for {len(combined_df)} colonies")
        return combined_df
    
    def calculate_scores(self, combined_df):
        # compute base scores penalizing common features and rewarding rare combos
        scores = pd.DataFrame({'colony_id': combined_df['colony_id']})
        
        # morphological complexity
        morph_vals = []
        for f in ['circularity','aspect_ratio','solidity','convexity','margin_complexity']:
            if f in combined_df:
                v = combined_df[f].fillna(combined_df[f].median())
                pct = v.rank(pct=True)
                morph_vals.append(np.minimum(pct,1-pct)*2)
        scores['morphology_score'] = np.mean(morph_vals, axis=0) if morph_vals else 0.5
        
        # form rarity score
        if 'form' in combined_df:
            freq_f = combined_df['form'].value_counts(normalize=True)
            scores['form_score'] = combined_df['form'].map(lambda x: (1-freq_f[x])**0.8)
        else:
            scores['form_score'] = 0.5
        
        # size preference
        if 'area' in combined_df:
            pct = np.log1p(combined_df['area']).rank(pct=True)
            scores['size_score'] = np.where((pct>=0.2)&(pct<=0.8),1,0.5)
        else:
            scores['size_score'] = 0.5
        
        # enhanced density complexity
        if 'opacity_score' in combined_df:
            norm = combined_df['opacity_score']/combined_df['opacity_score'].max()
            scores['density_score'] = np.clip(norm**0.5 * 1.5, 0, 1)
            
            density_threshold = np.percentile(combined_df['opacity_score'], 75)
            density_bonus = np.where(combined_df['opacity_score'] >= density_threshold, 0.3, 0)
            scores['density_bonus'] = density_bonus
        else:
            scores['density_score'] = 0.5
            scores['density_bonus'] = 0
        
        # density class bonus
        if 'density_class' in combined_df:
            density_class_bonus = combined_df['density_class'].map({
                'dense': 0.4,
                'medium': 0.1,
                'sparse': 0.0
            }).fillna(0)
            scores['density_class_bonus'] = density_class_bonus
        else:
            scores['density_class_bonus'] = 0
        
        # margin preference
        if 'margin' in combined_df:
            interest = {'entire':0.2,'undulate':0.6,'serrate':1.0,'lobate':0.8,'unknown':0.5}
            scores['margin_score'] = combined_df['margin'].map(lambda x: interest.get(x,0.5))
        else:
            scores['margin_score'] = 0.5
        
        # base interest
        weights = {
            'morphology_score': 0.35,
            'form_score': 0.1,
            'size_score': 0.1,
            'density_score': 0.35,
            'margin_score': 0.1
        }
        scores['bio_base'] = sum(scores[f]*w for f,w in weights.items())
        
        # combo novelty reward
        combo = None
        if 'color_cluster' in combined_df and 'form' in combined_df:
            combo = combined_df['color_cluster'].astype(str)+'_'+combined_df['form'].astype(str)
            freq_combo = combo.value_counts(normalize=True)
            scores['novelty_combo'] = combo.map(lambda x: (1-freq_combo[x])*0.3 if freq_combo[x]<0.03 else 0)
        else:
            scores['novelty_combo'] = 0
        
        # penalties on common features
        pen_vals = np.zeros(len(combined_df))
        if 'color_cluster' in combined_df:
            freq_c = combined_df['color_cluster'].value_counts(normalize=True)
            pen_vals += combined_df['color_cluster'].map(lambda x: freq_c[x]**2)
        if 'form' in combined_df:
            pen_vals += combined_df['form'].map(lambda x: freq_f[x]**2)
        if combo is not None:
            pen_vals += combo.map(lambda x: freq_combo[x]**2)
        scores['penalty'] = (pen_vals/3) * 0.7
        
        # final interest score
        total_density_bonus = scores['density_bonus'] + scores['density_class_bonus']
        scores['bio_interest'] = np.clip(
            scores['bio_base'] + scores['novelty_combo'] - scores['penalty'] + total_density_bonus,
            0, 1
        )
        
        # attach metadata
        for c in ['color_cluster','form','margin','area','density_class']:
            if c in combined_df:
                scores[c] = combined_df[c]
        
        self.scores_df = scores
        return scores
    
    def select_top_colonies(self, scores_df, n=20, penalty_factor=0.5):
        # pick top scoring colonies while enforcing minimum per-cluster quota
        selected = []
        pool = scores_df.copy().set_index('colony_id')
        
        # determine quotas per unique color cluster
        clusters = pool['color_cluster'].unique() if 'color_cluster' in pool else []
        k = len(clusters)
        min_quota = n // k if k>0 else 0
        quota = {c:0 for c in clusters}
        
        for _ in range(n):
            if pool.empty:
                break
            # enforce quota: select from clusters under quota first
            under = [c for c,count in quota.items() if count < min_quota]
            if under:
                cand = pool[pool['color_cluster'].isin(under)]
            else:
                cand = pool
            # pick best candidate
            best = cand['bio_interest'].idxmax()
            selected.append(best)
            # update quota
            if 'color_cluster' in pool:
                c = pool.loc[best,'color_cluster']
                quota[c] = quota.get(c,0) + 1
            # penalize same cluster/form in pool
            attrs = ['color_cluster','form']
            row = pool.loc[best]
            for attr in attrs:
                if attr in row:
                    mask = pool[attr]==row[attr]
                    pool.loc[mask,'bio_interest'] *= (1-penalty_factor)
            # remove picked
            pool = pool.drop(best)
        
        self.top_colonies = scores_df.set_index('colony_id').loc[selected].reset_index()
        return self.top_colonies
    
    def run_full_analysis(self, image_path):
        # run complete analysis pipeline
        print("starting full colony analysis pipeline")
        
        # load and preprocess
        original = self.load_image(image_path)
        if original is None:
            return None
        
        processed = self.preprocess_image(original)
        
        # detect plate and segment colonies
        plate_mask, plate_info = self.detect_plate(processed)
        colony_labels, colony_props = self.segment_colonies(processed, plate_mask)
        
        if len(colony_props) == 0:
            print("no colonies detected")
            return None
        
        # analyze colonies
        morph_df = self.analyze_morphology(colony_labels, colony_props)
        colony_data, clusters = self.analyze_colors(processed, colony_labels, colony_props)
        density_df = self.analyze_density(processed, colony_labels, colony_props, plate_mask)
        
        # combine and score
        combined_df = self.combine_analyses(morph_df, colony_data, density_df)
        scores_df = self.calculate_scores(combined_df)
        top_colonies = self.select_top_colonies(scores_df, n=self.n_top_colonies)
        
        print("analysis complete")
        return {
            'original_image': original,
            'processed_image': processed,
            'plate_mask': plate_mask,
            'colony_labels': colony_labels,
            'colony_properties': colony_props,
            'morph_df': morph_df,
            'colony_data': colony_data,
            'density_df': density_df,
            'combined_df': combined_df,
            'scores_df': scores_df,
            'top_colonies': top_colonies
        } 