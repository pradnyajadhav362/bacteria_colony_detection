# run_app.py
# simple launcher for the bacterial colony analysis app

import subprocess
import sys
import os

def main():
    print("🔬 Bacterial Colony Analysis Pipeline")
    print("=" * 50)
    
    # check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed successfully")
    
    # check if other dependencies are installed
    required_packages = [
        "opencv-python", "scikit-image", "scikit-learn", 
        "pandas", "matplotlib", "plotly", "PIL"
    ]
    
    print("\n📦 Checking dependencies...")
    for package in required_packages:
        try:
            if package == "PIL":
                import PIL
            elif package == "opencv-python":
                import cv2
            elif package == "scikit-image":
                import skimage
            elif package == "scikit-learn":
                import sklearn
            elif package == "pandas":
                import pandas
            elif package == "matplotlib":
                import matplotlib
            elif package == "plotly":
                import plotly
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not found")
            print("Please run: pip install -r requirements.txt")
            return
    
    print("\n🚀 Starting the application...")
    print("The app will open in your default web browser")
    print("Press Ctrl+C to stop the application")
    print("-" * 50)
    
    # run the streamlit app
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        print("Please check that all files are in the correct directory")

if __name__ == "__main__":
    main() 