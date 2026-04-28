# ===============================================
# Name: Alya
# Roll No: XXXXX
# Course: Image Processing & Computer Vision
# Assignment: Medical Image Compression & Segmentation
# Date: XXXXX
# ===============================================

import cv2
import numpy as np
import os

# Create output folder
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# =========================
# Task 1: Load Image
# =========================
print("Loading medical image...")

img = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: image not found. Check filename and path.")
    exit()

cv2.imwrite("outputs/original.png", img)

# =========================
# Task 1: RLE Compression
# =========================

def rle_encode(image):
    pixels = image.flatten()
    encoded = []
    prev = pixels[0]
    count = 1

    for pixel in pixels[1:]:
        if pixel == prev:
            count += 1
        else:
            encoded.append((prev, count))
            prev = pixel
            count = 1

    encoded.append((prev, count))
    return encoded

rle = rle_encode(img)

original_size = img.size
compressed_size = len(rle) * 2

compression_ratio = original_size / compressed_size
saving = (1 - compressed_size / original_size) * 100

print("\n=== COMPRESSION RESULTS ===")
print("Original size:", original_size)
print("Compressed size:", compressed_size)
print("Compression Ratio:", compression_ratio)
print("Storage Saving (%):", saving)

# =========================
# Task 2: Segmentation
# =========================

# Global Threshold
_, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu Threshold
_, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite("outputs/global_threshold.png", global_thresh)
cv2.imwrite("outputs/otsu_threshold.png", otsu_thresh)

# =========================
# Task 3: Morphological Processing
# =========================

kernel = np.ones((3, 3), np.uint8)

# Dilation
dilated = cv2.dilate(otsu_thresh, kernel, iterations=1)

# Erosion
eroded = cv2.erode(otsu_thresh, kernel, iterations=1)

cv2.imwrite("outputs/dilated.png", dilated)
cv2.imwrite("outputs/eroded.png", eroded)

# =========================
# Task 4: Analysis
# =========================

print("\n=== SEGMENTATION ANALYSIS ===")

print("\nGlobal Threshold:")
print("Uses fixed value, may fail if lighting varies.")

print("\nOtsu Threshold:")
print("Automatically selects best threshold, better segmentation.")

print("\nMorphological Operations:")
print("Dilation expands white regions.")
print("Erosion removes noise and shrinks regions.")

print("\nFinal Conclusion:")
print("Otsu + Morphology gives best results for medical images.")