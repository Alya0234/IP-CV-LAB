import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('image1.jpg')

if img is None:
    print("Error: Image not found")
else:
    rows, cols = img.shape[:2]

    # -------- Translation --------
    M_trans = np.float32([[1, 0, 50], [0, 1, 50]])
    translated = cv2.warpAffine(img, M_trans, (cols, rows))

    # -------- Scaling --------
    scaled = cv2.resize(img, None, fx=0.5, fy=0.5)

    # -------- Rotation --------
    M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
    rotated = cv2.warpAffine(img, M_rot, (cols, rows))

    # Convert BGR → RGB for display
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    translated = cv2.cvtColor(translated, cv2.COLOR_BGR2RGB)
    scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
    rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

    # -------- Display --------
    plt.figure(figsize=(10,6))

    plt.subplot(2,2,1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(translated)
    plt.title("Translated")
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(scaled)
    plt.title("Scaled")
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(rotated)
    plt.title("Rotated")
    plt.axis('off')

    plt.savefig("exp7_output.png")
    plt.show()