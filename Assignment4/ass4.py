# ===============================================
# Name: Alya
# Roll No: XXXXX
# Course: Image Processing & Computer Vision
# Assignment: Feature-Based Traffic Monitoring
# Date: XXXXX
# ===============================================

import cv2
import numpy as np
import os

# Create output folder
if not os.path.exists("outputs"):
    os.makedirs("outputs")

print("Loading traffic image...")

img = cv2.imread("image1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("outputs/original.png", img)

# =========================
# Task 1: Edge Detection
# =========================

print("\nPerforming Edge Detection...")

# Sobel
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobel = cv2.magnitude(sobelx, sobely)

# Normalize Sobel
sobel = np.uint8(np.clip(sobel, 0, 255))

# Canny
canny = cv2.Canny(gray, 100, 200)

cv2.imwrite("outputs/sobel.png", sobel)
cv2.imwrite("outputs/canny.png", canny)

# =========================
# Task 2: Contours & Bounding Boxes
# =========================

print("\nDetecting objects...")

contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_img = img.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    
    if area > 500:  # ignore tiny noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_img, (x,y), (x+w, y+h), (0,255,0), 2)
        
        perimeter = cv2.arcLength(cnt, True)
        
        print(f"Object -> Area: {area}, Perimeter: {perimeter}")

cv2.imwrite("outputs/contours.png", contour_img)

# =========================
# Task 3: Feature Extraction (ORB)
# =========================

print("\nExtracting features using ORB...")

orb = cv2.ORB_create()

keypoints, descriptors = orb.detectAndCompute(gray, None)

feature_img = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0))

cv2.imwrite("outputs/orb_features.png", feature_img)

print("Total keypoints detected:", len(keypoints))

# =========================
# Task 4: Analysis
# =========================

print("\n=== ANALYSIS ===")

print("\nEdge Detection:")
print("Sobel detects gradients but is noisy.")
print("Canny provides sharper and cleaner edges.")

print("\nObject Representation:")
print("Contours help identify vehicles and objects.")
print("Bounding boxes locate objects in the image.")

print("\nFeature Extraction:")
print("ORB detects keypoints useful for tracking and recognition.")

print("\nConclusion:")
print("Canny + ORB gives best performance for traffic monitoring.")