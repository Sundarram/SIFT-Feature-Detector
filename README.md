# SIFT-Feature-Detector
Program to stitch together two images into a single image (OpenCV - Python) by matching SIFT features. 
Steps: Detect SIFT features and compute descriptors separately for the two images. Find robust corresponding features using Flann-based matcher. Calculate homography matrix and merge images after perspective transformation. The two images that I have merged are 'Imag1.jpg' and 'Image2.jpg'. 'correspondences.jpg' contains the matching robust SIFT  features. 'panorama.jpg' contains the final merged output image. 
Used OpenCV in Python.


