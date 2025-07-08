# run_admin.py
# launcher script for admin dashboard
# provides separate admin interface to view all user data

import subprocess
import sys

def main():
    print("starting admin dashboard")
    print("admin password: admin123")
    print("=" * 40)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "admin_dashboard.py", "--server.port", "8502"])
    except KeyboardInterrupt:
        print("admin dashboard stopped")
    except Exception as e:
        print(f"error starting admin dashboard: {e}")

if __name__ == "__main__":
    main() 