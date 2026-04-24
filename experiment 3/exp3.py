import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread('image1.jpg', 0)

if img is None:
    print("Error: Image not found")
else:
    # Contrast Stretching
    min_val = np.min(img)
    max_val = np.max(img)
    cs_img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    # Histogram Equalization
    he_img = cv2.equalizeHist(img)

    # Display images
    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(cs_img, cmap='gray')
    plt.title("Contrast Stretch")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(he_img, cmap='gray')
    plt.title("Histogram Equalized")
    plt.axis('off')

    plt.savefig("exp3_output.png")
    plt.show()