# export all user data and results to easily accessible folder
# runs independently of the main app - no admin interface needed
# collects uploaded images, analysis results, csvs, and summary stats

import os
import shutil
import pandas as pd
import json
from datetime import datetime
import glob

def export_all_user_data():
    print("starting data export")
    
    source_dir = "/Users/pradnyajadhav/Desktop/Image_analysis/admin_logs"
    export_dir = "/Users/pradnyajadhav/Desktop/Image_analysis/exported_user_data"
    
    if not os.path.exists(source_dir):
        print("no admin logs found - users haven't uploaded any data yet")
        return
    
    # create fresh export directory
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir)
    
    # copy all session folders
    session_count = 0
    total_images = 0
    total_results = 0
    
    for session_dir in glob.glob(f"{source_dir}/session_*"):
        if os.path.isdir(session_dir):
            session_name = os.path.basename(session_dir)
            dest_session = os.path.join(export_dir, session_name)
            shutil.copytree(session_dir, dest_session)
            session_count += 1
            
            # count files in this session
            uploads_dir = os.path.join(dest_session, "uploads")
            if os.path.exists(uploads_dir):
                total_images += len(glob.glob(f"{uploads_dir}/*"))
            
            results_dir = os.path.join(dest_session, "results")  
            if os.path.exists(results_dir):
                total_results += len(glob.glob(f"{results_dir}/*"))
    
    # copy summary files
    summary_files = ["session_log.jsonl", "admin_summary.csv"]
    for file in summary_files:
        source_file = os.path.join(source_dir, file)
        if os.path.exists(source_file):
            shutil.copy2(source_file, export_dir)
    
    # copy other directories
    other_dirs = ["results", "parameters", "processed_images", "uploads"]
    for dir_name in other_dirs:
        source_path = os.path.join(source_dir, dir_name)
        if os.path.exists(source_path):
            dest_path = os.path.join(export_dir, dir_name)
            shutil.copytree(source_path, dest_path)
    
    # create summary report
    summary_report = {
        "export_date": datetime.now().isoformat(),
        "total_sessions": session_count,
        "total_uploaded_images": total_images,
        "total_result_files": total_results,
        "export_location": export_dir
    }
    
    with open(os.path.join(export_dir, "export_summary.json"), "w") as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"export complete")
    print(f"sessions: {session_count}")
    print(f"uploaded images: {total_images}")
    print(f"result files: {total_results}")
    print(f"location: {export_dir}")
    
    return export_dir

if __name__ == "__main__":
    export_all_user_data() 