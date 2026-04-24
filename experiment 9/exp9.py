import cv2
import numpy as np

# Read image
img = cv2.imread('image1.jpg', 0)

if img is None:
    print("Error: Image not found")
else:
    # Flatten image
    flat = img.flatten()

    # -------- Run Length Encoding --------
    def rle_encode(data):
        encoding = []
        prev = data[0]
        count = 1

        for i in data[1:]:
            if i == prev:
                count += 1
            else:
                encoding.append((prev, count))
                prev = i
                count = 1

        encoding.append((prev, count))
        return encoding

    # Apply RLE
    encoded = rle_encode(flat)

    # Results
    print("Original size:", len(flat))
    print("Encoded size:", len(encoded))