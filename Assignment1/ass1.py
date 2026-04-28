# ==========================================
# Name: Alya
# Roll No: XXXXX
# Course: Image Processing & Computer Vision
# Assignment: Smart Document Scanner
# ==========================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Welcome to Smart Document Scanner & Quality Analysis System")

# -------------------------------
# LOAD IMAGE (FINAL FIXED)
# -------------------------------
img = cv2.imread('image1.jpg')

if img is None:
    print("Error: Image not found. Check file name or location.")
    exit()

# Resize to 512x512
img_resized = cv2.resize(img, (512, 512))

# Convert to grayscale
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# -------------------------------
# SAMPLING (Resolution)
# -------------------------------
img_256 = cv2.resize(gray, (256, 256))
img_128 = cv2.resize(gray, (128, 128))

img_256_up = cv2.resize(img_256, (512, 512))
img_128_up = cv2.resize(img_128, (512, 512))

# -------------------------------
# QUANTIZATION
# -------------------------------
def quantize(image, levels):
    factor = 256 // levels
    return (image // factor) * factor

img_8bit = gray
img_4bit = quantize(gray, 16)
img_2bit = quantize(gray, 4)

# -------------------------------
# SAVE OUTPUTS
# -------------------------------
cv2.imwrite("outputs/original.png", gray)
cv2.imwrite("outputs/sample_256.png", img_256_up)
cv2.imwrite("outputs/sample_128.png", img_128_up)
cv2.imwrite("outputs/quant_8bit.png", img_8bit)
cv2.imwrite("outputs/quant_4bit.png", img_4bit)
cv2.imwrite("outputs/quant_2bit.png", img_2bit)

# -------------------------------
# DISPLAY RESULTS
# -------------------------------
plt.figure(figsize=(12,10))

plt.subplot(2,3,1)
plt.title("Original")
plt.imshow(gray, cmap='gray')

plt.subplot(2,3,2)
plt.title("256x256")
plt.imshow(img_256_up, cmap='gray')

plt.subplot(2,3,3)
plt.title("128x128")
plt.imshow(img_128_up, cmap='gray')

plt.subplot(2,3,4)
plt.title("8-bit (256 levels)")
plt.imshow(img_8bit, cmap='gray')

plt.subplot(2,3,5)
plt.title("4-bit (16 levels)")
plt.imshow(img_4bit, cmap='gray')

plt.subplot(2,3,6)
plt.title("2-bit (4 levels)")
plt.imshow(img_2bit, cmap='gray')

for i in range(1,7):
    plt.subplot(2,3,i).axis('off')

plt.tight_layout()
plt.show()

# -------------------------------
# OBSERVATIONS
# -------------------------------
print("\n--- OBSERVATIONS ---")
print("High resolution retains sharp text and edges.")
print("Lower resolution reduces clarity and readability.")
print("8-bit images preserve full detail.")
print("4-bit shows slight quality loss.")
print("2-bit severely degrades image quality.")
print("OCR works best with high resolution and high bit depth.")