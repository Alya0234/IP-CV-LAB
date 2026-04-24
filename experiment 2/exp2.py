import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read image in grayscale
image = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)

# Step 2: Check image
if image is None:
    print("Error: Image not found")
else:
    # -------- Sampling --------
    def sampling(img, factor):
        return img[::factor, ::factor]

    # -------- Quantization --------
    def quantization(img, levels):
        img_norm = img / 255.0
        img_quant = np.floor(img_norm * levels) / levels
        return (img_quant * 255).astype(np.uint8)

    # -------- Sampling Results --------
    factors = [1, 2, 4]

    plt.figure(figsize=(10,4))
    for i, f in enumerate(factors):
        sampled = sampling(image, f)
        plt.subplot(1,3,i+1)
        plt.imshow(sampled, cmap='gray')
        plt.title(f'Sampling = {f}')
        plt.axis('off')

    plt.savefig("sampling_output.png")
    plt.show()

    # -------- Quantization Results --------
    levels = [256, 64, 16]

    plt.figure(figsize=(10,4))
    for i, q in enumerate(levels):
        quant = quantization(image, q)
        plt.subplot(1,3,i+1)
        plt.imshow(quant, cmap='gray')
        plt.title(f'Levels = {q}')
        plt.axis('off')

    plt.savefig("quantization_output.png")
    plt.show()