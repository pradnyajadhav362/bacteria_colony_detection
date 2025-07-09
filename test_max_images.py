# test_max_images.py
# simple test to find max images your system can handle

import psutil
import gc
from colony_analyzer import ColonyAnalyzer
import cv2
import numpy as np

def check_memory():
    # get current memory usage
    memory = psutil.virtual_memory()
    print(f"memory usage: {memory.percent}% ({memory.used / (1024**3):.2f}GB / {memory.total / (1024**3):.2f}GB)")
    return memory.percent

def create_test_image(filename):
    # create a simple test image with random colonies
    test_image = np.zeros((800, 800, 3), dtype=np.uint8)
    # add some random white circles as fake colonies
    for _ in range(np.random.randint(5, 20)):
        x = np.random.randint(100, 700)
        y = np.random.randint(100, 700)
        radius = np.random.randint(10, 30)
        cv2.circle(test_image, (x, y), radius, (255, 255, 255), -1)
    cv2.imwrite(filename, test_image)

print("testing maximum images your system can handle")
print("creating test images...")

# create test images
for i in range(30):
    create_test_image(f"test_img_{i}.jpg")

print("starting memory test")
check_memory()

analyzer = ColonyAnalyzer()
results_count = 0
max_tested = 0

try:
    for i in range(30):
        print(f"\nprocessing image {i+1}/30")
        
        # check memory before processing
        mem_before = check_memory()
        if mem_before > 85:
            print(f"memory too high ({mem_before}%), stopping at {i} images")
            break
        
        # analyze image
        results = analyzer.run_full_analysis(f"test_img_{i}.jpg")
        if results:
            results_count += 1
            
        # force cleanup
        del results
        gc.collect()
        
        max_tested = i + 1
        
        # check memory after processing
        mem_after = check_memory()
        print(f"memory change: {mem_after - mem_before:+.1f}%")
        
except Exception as e:
    print(f"crashed at image {max_tested}: {e}")

print(f"\ntest complete!")
print(f"successfully processed: {results_count} images")
print(f"max tested without crash: {max_tested}")
print(f"recommended max for 25 images: should work if max_tested >= 25")

# cleanup test files
import os
for i in range(30):
    if os.path.exists(f"test_img_{i}.jpg"):
        os.remove(f"test_img_{i}.jpg")
        
print("test files cleaned up") 