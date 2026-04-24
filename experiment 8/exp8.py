import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('image1.jpg')

if img is None:
    print("Error: Image not found")
else:
    # Convert BGR → RGB (for display)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert BGR → HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # -------- Define color range (Green example) --------
    lower = np.array([30, 50, 50])
    upper = np.array([90, 255, 255])

    # -------- Create mask --------
    mask = cv2.inRange(hsv, lower, upper)

    # -------- Apply mask --------
    result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    # -------- Display --------
    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(result)
    plt.title("Segmented (Green)")
    plt.axis('off')

    plt.savefig("exp8_output.png")
    plt.show()