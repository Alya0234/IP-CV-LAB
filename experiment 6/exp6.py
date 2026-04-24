import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

# Read image
img = cv2.imread('image1.jpg', 0)

if img is None:
    print("Error: Image not found")
else:
    # -------- Add Periodic Noise --------
    rows, cols = img.shape

    x = np.linspace(0, 50, cols)
    noise = 20 * np.sin(x)
    noise = np.tile(noise, (rows, 1))

    noisy = img + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    # -------- Apply Wiener Filter --------
    wiener_img = wiener(noisy)

    # -------- Display --------
    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(noisy, cmap='gray')
    plt.title("Periodic Noise")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(wiener_img, cmap='gray')
    plt.title("Wiener Filter")
    plt.axis('off')

    plt.savefig("exp6_output.png")
    plt.show()