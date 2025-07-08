# test_admin_system.py
# this script tests the admin logging system to show what files get created

import sys
import os
sys.path.append('/Users/pradnyajadhav/Desktop/Image_analysis')

from admin_logger import AdminLogger
from PIL import Image
import numpy as np
import pandas as pd

# create test admin logger
admin_logger = AdminLogger("test_admin_logs")

print("testing admin logging system")

# test 1: create a mock session
user_id = "TestGroup_A1"
session_id = admin_logger.generate_session_id(user_id)
print(f"generated session id: {session_id} for user: {user_id}")

# log session start
admin_logger.log_session(session_id, user_id)

# test 2: create a dummy image upload
print("creating dummy uploaded image")
test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
test_pil_image = Image.fromarray(test_image)

# save as temporary file to simulate upload
temp_image_path = "temp_test_image.png"
test_pil_image.save(temp_image_path)

# log the upload
with open(temp_image_path, 'rb') as f:
    class MockUploadedFile:
        def __init__(self, file_path):
            self.name = "test_petri_dish.png"
            with open(file_path, 'rb') as f:
                self._content = f.read()
        
        def getbuffer(self):
            return self._content
    
    mock_file = MockUploadedFile(temp_image_path)
    upload_path = admin_logger.log_upload(session_id, mock_file, mock_file.name, user_id)
    print(f"logged upload: {upload_path}")

# test 3: create mock analysis results
print("creating mock analysis results")
mock_results = {
    'colony_properties': ['prop1', 'prop2', 'prop3'],  # 3 colonies detected
    'combined_df': pd.DataFrame({
        'colony_id': [1, 2, 3],
        'area': [150, 200, 175],
        'perimeter': [45, 52, 48],
        'color_r': [120, 135, 125],
        'color_g': [110, 125, 115],
        'color_b': [95, 105, 100]
    }),
    'morph_df': pd.DataFrame({
        'colony_id': [1, 2, 3],
        'form': ['circular', 'irregular', 'circular'],
        'roundness': [0.85, 0.65, 0.90],
        'aspect_ratio': [1.1, 1.4, 1.0]
    }),
    'colony_data': [
        {'colony_id': 1, 'dominant_color': [120, 110, 95], 'color_cluster': 0},
        {'colony_id': 2, 'dominant_color': [135, 125, 105], 'color_cluster': 1},
        {'colony_id': 3, 'dominant_color': [125, 115, 100], 'color_cluster': 0}
    ],
    'density_df': pd.DataFrame({
        'colony_id': [1, 2, 3],
        'density_score': [0.75, 0.60, 0.80],
        'density_class': ['dense', 'moderate', 'dense']
    }),
    'top_colonies_df': pd.DataFrame({
        'colony_id': [3, 1, 2],
        'total_score': [8.5, 8.2, 7.1],
        'rank': [1, 2, 3]
    }),
    'processed_image': test_image,
    'original_image': test_image
}

# mock parameters
mock_params = {
    'bilateral_d': 9,
    'gamma': 1.2,
    'min_colony_size': 15,
    'n_top_colonies': 20
}

# log the analysis
admin_logger.log_analysis(session_id, mock_params, mock_results, 15.5)
print("logged analysis results")

# save analysis images
highlighted_image = test_image.copy()
highlighted_image[50:100, 50:100] = [255, 0, 0]  # red square to simulate highlighted colony

detection_image = test_image.copy()
detection_image[100:150, 100:150] = [0, 255, 0]  # green square to simulate detection

admin_logger.save_analysis_image(session_id, highlighted_image, "test_petri_dish.png_highlighted.png")
admin_logger.save_analysis_image(session_id, detection_image, "test_petri_dish.png_detection.png")
print("saved analysis images")

# test 4: log some downloads
admin_logger.log_download(session_id, "csv", "colonies_complete.csv")
admin_logger.log_download(session_id, "image", "highlighted.png")

print("\ntest complete files created in test_admin_logs directory")

# show directory structure
import glob
if os.path.exists("test_admin_logs"):
    print("\ndirectory structure created:")
    for root, dirs, files in os.walk("test_admin_logs"):
        level = root.replace("test_admin_logs", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

print(f"\nyou can now view session {session_id} in your admin dashboard")
print("the files created above show exactly what users will generate when they analyze images")

# cleanup
os.remove(temp_image_path) 