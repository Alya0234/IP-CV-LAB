import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread('image1.jpg', 0)

if img is None:
    print("Error: Image not found")
else:
    # Step 1: Convert to frequency domain
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = rows//2, cols//2

    # Step 2: Create Low Pass Filter mask
    mask_lp = np.zeros((rows, cols), np.uint8)
    mask_lp[crow-30:crow+30, ccol-30:ccol+30] = 1

    # Step 3: High Pass Filter mask
    mask_hp = 1 - mask_lp

    # Step 4: Apply filters
    lp = np.fft.ifft2(np.fft.ifftshift(fshift * mask_lp))
    hp = np.fft.ifft2(np.fft.ifftshift(fshift * mask_hp))

    # Step 5: Display
    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(np.abs(lp), cmap='gray')
    plt.title("Low Pass (Blur)")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(np.abs(hp), cmap='gray')
    plt.title("High Pass (Edges)")
    plt.axis('off')

    plt.savefig("exp4_output.png")
    plt.show()