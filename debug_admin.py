# debug_admin.py
# quick test to see what the admin logger is returning

from admin_logger import admin_logger
import pandas as pd

print("testing admin logger data retrieval")

try:
    # test getting all sessions
    all_sessions = admin_logger.get_all_sessions()
    print(f"admin_logger.get_all_sessions() returned:")
    print(f"type: {type(all_sessions)}")
    print(f"empty: {all_sessions.empty}")
    print(f"shape: {all_sessions.shape}")
    
    if not all_sessions.empty:
        print("columns:", list(all_sessions.columns))
        print("first 3 rows:")
        print(all_sessions.head(3))
        
        print("\nuser_id column exists:", 'user_id' in all_sessions.columns)
        if 'user_info' in all_sessions.columns:
            print("user_info values:", list(all_sessions['user_info'].unique()))
    else:
        print("dataframe is empty - this would cause buttons not to show")
        
except Exception as e:
    print(f"error getting sessions: {e}")
    print("this exception would prevent buttons from showing")

print("\nchecking admin_summary.csv directly:")
try:
    df = pd.read_csv("admin_logs/admin_summary.csv")
    print(f"direct csv read successful: {df.shape}")
    print("columns:", list(df.columns))
    print("user_info column:", 'user_info' in df.columns)
except Exception as e:
    print(f"error reading csv directly: {e}")

print("\nchecking if admin_logs directory exists:")
import os
print(f"admin_logs exists: {os.path.exists('admin_logs')}")
print(f"admin_summary.csv exists: {os.path.exists('admin_logs/admin_summary.csv')}")

print("\ntesting admin logger initialization:")
try:
    from admin_logger import AdminLogger
    test_logger = AdminLogger("admin_logs")
    print("admin logger initialization successful")
    
    sessions = test_logger.get_all_sessions()
    print(f"test logger get_all_sessions: {type(sessions)}, empty: {sessions.empty}")
    
except Exception as e:
    print(f"admin logger initialization failed: {e}") 