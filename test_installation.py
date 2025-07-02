# test_installation.py
# test script to verify all dependencies are working correctly

import sys
import traceback

def test_imports():
    # test all required imports
    print(" Testing Bacterial Colony Analysis Dependencies")
    print("=" * 60)
    
    tests = [
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("Matplotlib", "matplotlib"),
        ("scikit-image", "skimage"),
        ("scikit-learn", "sklearn"),
        ("scipy", "scipy"),
        ("Streamlit", "streamlit"),
        ("Plotly", "plotly"),
        ("PIL/Pillow", "PIL")
    ]
    
    failed_imports = []
    
    for name, module in tests:
        try:
            __import__(module)
            print(f" {name} - OK")
        except ImportError as e:
            print(f" {name} - FAILED: {e}")
            failed_imports.append(name)
    
    return failed_imports

def test_colony_analyzer():
    # test the colony analyzer module
    print("\n🧪 Testing Colony Analyzer Module")
    print("-" * 40)
    
    try:
        from colony_analyzer import ColonyAnalyzer
        analyzer = ColonyAnalyzer()
        print(" ColonyAnalyzer class imported successfully")
        
        # test basic functionality
        if hasattr(analyzer, 'load_image'):
            print(" load_image method found")
        else:
            print(" load_image method missing")
            
        if hasattr(analyzer, 'preprocess_image'):
            print(" preprocess_image method found")
        else:
            print(" preprocess_image method missing")
            
        if hasattr(analyzer, 'run_full_analysis'):
            print(" run_full_analysis method found")
        else:
            print(" run_full_analysis method missing")
            
        return True
        
    except Exception as e:
        print(f" ColonyAnalyzer test failed: {e}")
        traceback.print_exc()
        return False

def test_streamlit_app():
    # test the streamlit app module
    print("\n Testing Streamlit App Module")
    print("-" * 40)
    
    try:
        import app
        print(" app.py imported successfully")
        
        # test if main function exists
        if hasattr(app, 'main'):
            print(" main function found")
        else:
            print(" main function missing")
            
        # test if display functions exist
        display_functions = [
            'display_results', 'display_overview', 'display_colony_details',
            'display_color_analysis', 'display_morphology_analysis', 'display_top_colonies'
        ]
        
        for func_name in display_functions:
            if hasattr(app, func_name):
                print(f" {func_name} function found")
            else:
                print(f" {func_name} function missing")
                
        return True
        
    except Exception as e:
        print(f" Streamlit app test failed: {e}")
        traceback.print_exc()
        return False

def main():
    # run all tests
    print("Starting dependency and functionality tests...\n")
    
    # test imports
    failed_imports = test_imports()
    
    # test modules
    analyzer_ok = test_colony_analyzer()
    app_ok = test_streamlit_app()
    
    # summary
    print("\n" + "=" * 60)
    print(" TEST SUMMARY")
    print("=" * 60)
    
    if failed_imports:
        print(f" Failed imports: {', '.join(failed_imports)}")
        print(" Run: pip install -r requirements.txt")
    else:
        print(" All imports successful")
    
    if analyzer_ok:
        print(" ColonyAnalyzer module working")
    else:
        print(" ColonyAnalyzer module has issues")
    
    if app_ok:
        print(" Streamlit app module working")
    else:
        print(" Streamlit app module has issues")
    
    if not failed_imports and analyzer_ok and app_ok:
        print("\n All tests passed! The app should work correctly.")
        print(" Run the app with: python run_app.py")
    else:
        print("\n  Some tests failed. Please fix the issues before running the app.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 