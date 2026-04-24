import cv2
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('image1.jpg')

if img is None:
    print("Error: Image not found")
else:
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge Detection
    edges = cv2.Canny(gray, 100, 200)

    # Display
    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(edges, cmap='gray')
    plt.title("Edges")
    plt.axis('off')

    plt.savefig("exp10_output.png")
    plt.show()