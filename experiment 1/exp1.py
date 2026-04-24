import cv2
import matplotlib.pyplot as plt

# Step 1: Read image (use correct name + quotes)
image = cv2.imread('image1.jpg')

# Step 2: Check if image loaded
if image is None:
    print("Error: Image not found. Check file name.")
else:
    # Convert BGR → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 3: Display
    plt.imshow(image)
    plt.title("Acquired Image")
    plt.axis('off')

    # Save output
    plt.imsave("output.png", image)

    plt.show()