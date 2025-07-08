# view_admin_data.py
# script to view admin data programmatically
# access all user uploads and results without web interface

from admin_logger import admin_logger
import pandas as pd
import json

def show_summary():
    # show overview of all user activity
    print("admin data summary")
    print("=" * 40)
    
    sessions = admin_logger.get_all_sessions()
    
    if sessions.empty:
        print("no user sessions recorded yet")
        print("\nto generate data:")
        print("1. run the main app: python3 app.py") 
        print("2. upload images and run analysis")
        print("3. check back here for logged data")
        return
    
    print(f"total sessions: {len(sessions)}")
    print(f"total uploads: {sessions['total_uploads'].sum()}")
    print(f"total analyses: {sessions['total_analyses'].sum()}")
    print(f"total downloads: {sessions['total_downloads'].sum()}")
    print(f"total colonies detected: {sessions['colony_count'].sum()}")
    
    print("\nrecent sessions:")
    recent = sessions.tail(5)
    for _, session in recent.iterrows():
        print(f"• {session['session_id']}: {session['image_name']} ({session['colony_count']} colonies)")

def show_session_details(session_id):
    # show detailed info for specific session
    print(f"\nsession details: {session_id}")
    print("-" * 40)
    
    details = admin_logger.get_session_details(session_id)
    
    if not details:
        print("session not found")
        return
    
    # timeline
    if 'logs' in details:
        print("activity timeline:")
        for log in details['logs']:
            action = log['action']
            timestamp = log['timestamp']
            if action == 'upload':
                print(f"  {timestamp}: uploaded {log['image_name']}")
            elif action == 'analysis':
                print(f"  {timestamp}: analyzed - {log['colony_count']} colonies")
            elif action == 'download':
                print(f"  {timestamp}: downloaded {log['download_type']}")
    
    # parameters
    if 'parameters' in details:
        print(f"\nanalysis parameters:")
        params = details['parameters']
        key_params = ['bilateral_d', 'gamma', 'min_colony_size', 'max_colony_size']
        for param in key_params:
            if param in params:
                print(f"  {param}: {params[param]}")
    
    # results
    if 'results_summary' in details:
        results = details['results_summary']
        print(f"\nresults:")
        print(f"  colonies detected: {results.get('colony_count', 0)}")
        print(f"  analysis success: {results.get('analysis_successful', False)}")
    
    # colony data
    if 'colony_data' in details:
        df = details['colony_data']
        print(f"\ncolony data: {len(df)} rows")
        if not df.empty:
            print("  columns:", list(df.columns))
            print("  sample data:")
            print(df.head(3).to_string(index=False))

def export_all_data():
    # export everything for analysis
    print("\nexporting all admin data...")
    export_path = admin_logger.export_all_data()
    print(f"data exported to: {export_path}")
    
    # show what was exported
    import os
    for root, dirs, files in os.walk(export_path):
        level = root.replace(str(export_path), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

def main():
    print("bacterial colony analyzer - admin data viewer")
    print("=" * 50)
    
    while True:
        print("\noptions:")
        print("1. show summary")
        print("2. view session details")
        print("3. export all data")
        print("4. list all sessions")
        print("5. exit")
        
        choice = input("\nselect option (1-5): ").strip()
        
        if choice == '1':
            show_summary()
        
        elif choice == '2':
            sessions = admin_logger.get_all_sessions()
            if sessions.empty:
                print("no sessions available")
                continue
            
            print("\navailable sessions:")
            for i, session_id in enumerate(sessions['session_id'], 1):
                print(f"{i}. {session_id}")
            
            try:
                selection = int(input("select session number: ")) - 1
                if 0 <= selection < len(sessions):
                    session_id = sessions.iloc[selection]['session_id']
                    show_session_details(session_id)
                else:
                    print("invalid selection")
            except ValueError:
                print("please enter a number")
        
        elif choice == '3':
            export_all_data()
        
        elif choice == '4':
            sessions = admin_logger.get_all_sessions()
            if sessions.empty:
                print("no sessions available")
            else:
                print(f"\nall {len(sessions)} sessions:")
                print(sessions.to_string(index=False))
        
        elif choice == '5':
            print("goodbye")
            break
        
        else:
            print("invalid option")

if __name__ == "__main__":
    main() 