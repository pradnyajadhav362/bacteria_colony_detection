# create_sample_data.py
# creates sample user data in the main admin_logs directory for testing

from admin_logger import AdminLogger
from PIL import Image
import numpy as np
import pandas as pd
import os

# create admin logger using the main directory
admin_logger = AdminLogger("admin_logs")

print("creating sample user data for admin dashboard testing")

# create multiple test sessions
test_users = ["A1", "A2", "B1", "B2"]

for i, user_id in enumerate(test_users):
    print(f"creating data for user {user_id}")
    
    # generate session
    session_id = admin_logger.generate_session_id(user_id)
    admin_logger.log_session(session_id, user_id)
    
    # create dummy petri dish image
    test_image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    
    # add some fake circular colonies
    for j in range(3):
        center_x = 150 + j * 100
        center_y = 200 + (j % 2) * 100
        radius = 30 + j * 5
        
        # draw circular colony
        y, x = np.ogrid[:500, :500]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        test_image[mask] = [100 + j*30, 80 + j*20, 60 + j*15]  # different colors
    
    # save as temporary file to simulate upload
    temp_image_path = f"temp_petri_{user_id}.png"
    Image.fromarray(test_image).save(temp_image_path)
    
    # mock upload
    class MockUploadedFile:
        def __init__(self, file_path, user_id):
            self.name = f"petri_dish_{user_id}.png"
            with open(file_path, 'rb') as f:
                self._content = f.read()
        
        def getbuffer(self):
            return self._content
    
    mock_file = MockUploadedFile(temp_image_path, user_id)
    admin_logger.log_upload(session_id, mock_file, mock_file.name, user_id)
    
    # create mock analysis results
    num_colonies = 2 + i  # different numbers of colonies
    
    mock_results = {
        'colony_properties': [f'colony_{j}' for j in range(num_colonies)],
        'combined_df': pd.DataFrame({
            'colony_id': list(range(1, num_colonies + 1)),
            'area': [150 + j*25 for j in range(num_colonies)],
            'perimeter': [45 + j*5 for j in range(num_colonies)],
            'color_r': [120 + j*10 for j in range(num_colonies)],
            'color_g': [110 + j*8 for j in range(num_colonies)],
            'color_b': [95 + j*6 for j in range(num_colonies)]
        }),
        'morph_df': pd.DataFrame({
            'colony_id': list(range(1, num_colonies + 1)),
            'form': ['circular'] * num_colonies,
            'roundness': [0.85 + j*0.05 for j in range(num_colonies)],
            'aspect_ratio': [1.0 + j*0.1 for j in range(num_colonies)]
        }),
        'colony_data': [
            {'colony_id': j+1, 'dominant_color': [120+j*10, 110+j*8, 95+j*6], 'color_cluster': j%2}
            for j in range(num_colonies)
        ],
        'density_df': pd.DataFrame({
            'colony_id': list(range(1, num_colonies + 1)),
            'density_score': [0.7 + j*0.05 for j in range(num_colonies)],
            'density_class': (['dense', 'moderate'] * (num_colonies//2 + 1))[:num_colonies]
        }),
        'top_colonies_df': pd.DataFrame({
            'colony_id': list(range(1, num_colonies + 1)),
            'total_score': [8.0 + j*0.3 for j in range(num_colonies)],
            'rank': list(range(1, num_colonies + 1))
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
    
    # log analysis
    admin_logger.log_analysis(session_id, mock_params, mock_results, 12.5 + i*2)
    
    # create highlighted image with colored circles
    highlighted_image = test_image.copy()
    for j in range(num_colonies):
        center_x = 150 + j * 100
        center_y = 200 + (j % 2) * 100
        
        # draw red outline around colony
        y, x = np.ogrid[:500, :500]
        mask = ((x - center_x)**2 + (y - center_y)**2 <= (35 + j*5)**2) & ((x - center_x)**2 + (y - center_y)**2 >= (30 + j*5)**2)
        highlighted_image[mask] = [255, 0, 0]  # red outline
    
    # save analysis images
    admin_logger.save_analysis_image(session_id, highlighted_image, f"petri_dish_{user_id}.png_highlighted.png")
    admin_logger.save_analysis_image(session_id, test_image, f"petri_dish_{user_id}.png_detection.png")
    
    # log some downloads
    admin_logger.log_download(session_id, "csv", "colonies_complete.csv")
    admin_logger.log_download(session_id, "image", "highlighted.png")
    
    # cleanup temp file
    os.remove(temp_image_path)
    
    print(f"created session {session_id} for user {user_id} with {num_colonies} colonies")

print(f"\nsample data created for {len(test_users)} users")
print("now your admin dashboard will show data when you access it")
print("you can view uploaded images and download CSV files")

# show what was created
print("\nfiles created in admin_logs directory:")
if os.path.exists("admin_logs"):
    for root, dirs, files in os.walk("admin_logs"):
        level = root.replace("admin_logs", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files[:3]:  # limit output
            print(f"{subindent}{file}")
        if len(files) > 3:
            print(f"{subindent}... and {len(files)-3} more files") 