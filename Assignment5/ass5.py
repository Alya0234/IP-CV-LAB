# ===============================================
# Name: Alya
# Roll No: XXXXX
# Course: Image Processing & Computer Vision
# Assignment: Intelligent Image Enhancement System
# Date: XXXXX
# ===============================================

import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Create output folder
if not os.path.exists("outputs"):
    os.makedirs("outputs")

print("\n=== Intelligent Image Processing System ===")
print("Pipeline: Load → Noise → Restore → Segment → Features → Evaluate\n")

# =========================
# Task 2: Load & Preprocess
# =========================

img = cv2.imread("image1.jpg")

if img is None:
    print("Error: image not found")
    exit()

img = cv2.resize(img, (512, 512))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("outputs/original.png", img)
cv2.imwrite("outputs/grayscale.png", gray)

# =========================
# Task 3: Noise + Restoration
# =========================

def add_gaussian(img):
    noise = np.random.normal(0, 25, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def add_sp(img):
    noisy = img.copy()
    prob = 0.05
    salt = np.random.rand(*img.shape) < prob
    pepper = np.random.rand(*img.shape) < prob
    noisy[salt] = 255
    noisy[pepper] = 0
    return noisy

g_noise = add_gaussian(gray)
sp_noise = add_sp(gray)

cv2.imwrite("outputs/gaussian_noise.png", g_noise)
cv2.imwrite("outputs/sp_noise.png", sp_noise)

# Filters
mean = cv2.blur(g_noise, (5,5))
median = cv2.medianBlur(sp_noise, 5)
gaussian = cv2.GaussianBlur(g_noise, (5,5), 0)

cv2.imwrite("outputs/mean.png", mean)
cv2.imwrite("outputs/median.png", median)
cv2.imwrite("outputs/gaussian.png", gaussian)

# Enhancement
equalized = cv2.equalizeHist(gray)
cv2.imwrite("outputs/equalized.png", equalized)

# =========================
# Task 4: Segmentation
# =========================

_, global_th = cv2.threshold(equalized, 127, 255, cv2.THRESH_BINARY)
_, otsu = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite("outputs/global.png", global_th)
cv2.imwrite("outputs/otsu.png", otsu)

# Morphology
kernel = np.ones((3,3), np.uint8)
dilate = cv2.dilate(otsu, kernel, 1)
erode = cv2.erode(otsu, kernel, 1)

cv2.imwrite("outputs/dilate.png", dilate)
cv2.imwrite("outputs/erode.png", erode)

# =========================
# Task 5: Features
# =========================

# Edges
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
sobel = cv2.magnitude(sobelx, sobely)
sobel = np.uint8(np.clip(sobel, 0, 255))

canny = cv2.Canny(gray, 100, 200)

cv2.imwrite("outputs/sobel.png", sobel)
cv2.imwrite("outputs/canny.png", canny)

# Contours
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()

for cnt in contours:
    if cv2.contourArea(cnt) > 500:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_img, (x,y), (x+w,y+h), (0,255,0), 2)

cv2.imwrite("outputs/contours.png", contour_img)

# ORB
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(gray, None)
feature_img = cv2.drawKeypoints(img, kp, None, color=(0,255,0))

cv2.imwrite("outputs/orb.png", feature_img)

# =========================
# Task 6: Evaluation
# =========================

def mse(a,b):
    return np.mean((a-b)**2)

print("\n=== PERFORMANCE ===")

print("MSE (Original vs Gaussian Noise):", mse(gray, g_noise))
print("PSNR:", psnr(gray, g_noise))
print("SSIM:", ssim(gray, g_noise))

print("\nMSE (Original vs Restored):", mse(gray, gaussian))
print("PSNR:", psnr(gray, gaussian))
print("SSIM:", ssim(gray, gaussian))

# =========================
# Task 7: Conclusion
# =========================

print("\n=== FINAL CONCLUSION ===")

print("Enhancement improves visibility.")
print("Filtering removes noise effectively.")
print("Otsu + Morphology improves segmentation.")
print("Canny + ORB provides strong feature detection.")
print("System successfully converts raw image into meaningful data.")