# ===============================================
# Name: Alya
# Roll No: XXXXX
# Course: Image Processing & Computer Vision
# Assignment: Image Restoration for Surveillance
# Date: 14-Feb-2026
# ===============================================

import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr

# Create output folder
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# =========================
# Task 1: Load Image
# =========================
print("Loading image...")

image = cv2.imread("image1.jpg")   # put your image here
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite("outputs/original.png", gray)

# =========================
# Task 2: Add Noise
# =========================

def add_gaussian_noise(img):
    mean = 0
    sigma = 25
    noise = np.random.normal(mean, sigma, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(img):
    noisy = img.copy()
    prob = 0.05

    # Salt
    salt = np.random.rand(*img.shape) < prob
    noisy[salt] = 255

    # Pepper
    pepper = np.random.rand(*img.shape) < prob
    noisy[pepper] = 0

    return noisy

gaussian_noisy = add_gaussian_noise(gray)
sp_noisy = add_salt_pepper_noise(gray)

cv2.imwrite("outputs/gaussian_noise.png", gaussian_noisy)
cv2.imwrite("outputs/salt_pepper_noise.png", sp_noisy)

# =========================
# Task 3: Filtering
# =========================

def mean_filter(img):
    return cv2.blur(img, (5,5))

def median_filter(img):
    return cv2.medianBlur(img, 5)

def gaussian_filter(img):
    return cv2.GaussianBlur(img, (5,5), 0)

# Apply filters on Gaussian noise
mean_g = mean_filter(gaussian_noisy)
median_g = median_filter(gaussian_noisy)
gaussian_g = gaussian_filter(gaussian_noisy)

# Apply filters on Salt-Pepper noise
mean_sp = mean_filter(sp_noisy)
median_sp = median_filter(sp_noisy)
gaussian_sp = gaussian_filter(sp_noisy)

# Save outputs
cv2.imwrite("outputs/mean_gaussian.png", mean_g)
cv2.imwrite("outputs/median_gaussian.png", median_g)
cv2.imwrite("outputs/gaussian_gaussian.png", gaussian_g)

cv2.imwrite("outputs/mean_sp.png", mean_sp)
cv2.imwrite("outputs/median_sp.png", median_sp)
cv2.imwrite("outputs/gaussian_sp.png", gaussian_sp)

# =========================
# Task 4: Metrics
# =========================

def mse(original, restored):
    return np.mean((original - restored) ** 2)

print("\n===== PERFORMANCE METRICS =====")

def evaluate(name, original, restored):
    print(f"{name}")
    print("MSE:", mse(original, restored))
    print("PSNR:", psnr(original, restored))
    print("--------------------------")

print("\n--- Gaussian Noise Results ---")
evaluate("Mean Filter", gray, mean_g)
evaluate("Median Filter", gray, median_g)
evaluate("Gaussian Filter", gray, gaussian_g)

print("\n--- Salt & Pepper Noise Results ---")
evaluate("Mean Filter", gray, mean_sp)
evaluate("Median Filter", gray, median_sp)
evaluate("Gaussian Filter", gray, gaussian_sp)