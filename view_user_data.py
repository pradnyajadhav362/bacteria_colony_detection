# browse and view all user uploaded images and analysis results
# displays session info, image counts, and file paths for easy access
# no web interface needed - just run this script

import os
import glob
import json
import pandas as pd
from datetime import datetime

def view_user_data():
    print("scanning user data")
    
    admin_logs_dir = "/Users/pradnyajadhav/Desktop/Image_analysis/admin_logs"
    
    if not os.path.exists(admin_logs_dir):
        print("no user data found yet")
        return
    
    # get all sessions
    sessions = []
    for session_dir in glob.glob(f"{admin_logs_dir}/session_*"):
        if os.path.isdir(session_dir):
            session_id = os.path.basename(session_dir)
            
            # count files
            uploads_dir = os.path.join(session_dir, "uploads")
            results_dir = os.path.join(session_dir, "results")
            
            upload_count = 0
            result_count = 0
            
            if os.path.exists(uploads_dir):
                upload_files = glob.glob(f"{uploads_dir}/*")
                upload_count = len(upload_files)
            
            if os.path.exists(results_dir):
                result_files = glob.glob(f"{results_dir}/*")
                result_count = len(result_files)
            
            sessions.append({
                "session_id": session_id,
                "upload_count": upload_count,
                "result_count": result_count,
                "session_path": session_dir
            })
    
    if not sessions:
        print("no sessions found")
        return
    
    # display summary
    total_uploads = sum(s["upload_count"] for s in sessions)
    total_results = sum(s["result_count"] for s in sessions)
    
    print(f"\nfound {len(sessions)} user sessions")
    print(f"total uploaded images: {total_uploads}")
    print(f"total result files: {total_results}")
    print("-" * 50)
    
    # show each session
    for session in sessions:
        print(f"\nsession: {session['session_id']}")
        print(f"  uploads: {session['upload_count']}")
        print(f"  results: {session['result_count']}")
        print(f"  location: {session['session_path']}")
        
        # show actual files
        uploads_dir = os.path.join(session['session_path'], "uploads")
        if os.path.exists(uploads_dir) and session['upload_count'] > 0:
            print("  uploaded images:")
            for img_file in glob.glob(f"{uploads_dir}/*"):
                print(f"    - {os.path.basename(img_file)}")
        
        results_dir = os.path.join(session['session_path'], "results")
        if os.path.exists(results_dir) and session['result_count'] > 0:
            print("  analysis results:")
            for result_file in glob.glob(f"{results_dir}/*"):
                print(f"    - {os.path.basename(result_file)}")
    
    # check for session log
    session_log_file = os.path.join(admin_logs_dir, "session_log.jsonl")
    if os.path.exists(session_log_file):
        print(f"\ndetailed session log available at:")
        print(f"  {session_log_file}")
    
    print(f"\nall data stored in: {admin_logs_dir}")

if __name__ == "__main__":
    view_user_data() 