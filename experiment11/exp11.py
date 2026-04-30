import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread('image1.jpg', 0)

if img is None:
    print("Error: Image not found")
else:
    # Convert to binary (thresholding)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Create kernel (structuring element)
    kernel = np.ones((5,5), np.uint8)

    # -------- Morphological Operations --------
    dilation = cv2.dilate(binary, kernel, iterations=1)
    erosion = cv2.erode(binary, kernel, iterations=1)

    # -------- Display --------
    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.imshow(binary, cmap='gray')
    plt.title("Binary Image")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(dilation, cmap='gray')
    plt.title("Dilation")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(erosion, cmap='gray')
    plt.title("Erosion")
    plt.axis('off')

    plt.savefig("exp11_output.png")
    plt.show()