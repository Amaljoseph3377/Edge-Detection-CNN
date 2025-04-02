import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and verify the image
image_path = r"C:\Users\Amal\Downloads\download.jpg"  # Provide the full path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Error: Unable to load image '{image_path}'. Check the file path.")
else:
    print("Image loaded successfully!")

    # Resize the image for consistency
    image = cv2.resize(image, (300, 300))

    # Define the kernel for edge detection
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])

    # Apply convolution
    image_filter = cv2.filter2D(image, -1, kernel)

    # Apply ReLU activation (set negative values to 0)
    image_detect = np.maximum(image_filter, 0)

    # Max Pooling (2x2 window)
    def max_pooling(img, size=2):
        h, w = img.shape
        pooled_img = np.zeros((h // size, w // size))

        for i in range(0, h, size):
            for j in range(0, w, size):
                pooled_img[i // size, j // size] = np.max(img[i:i+size, j:j+size])

        return pooled_img

    image_condense = max_pooling(image_detect)

    # Plot results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(1, 4, 2)
    plt.imshow(image_filter, cmap='gray')
    plt.axis('off')
    plt.title('Convolution')

    plt.subplot(1, 4, 3)
    plt.imshow(image_detect, cmap='gray')
    plt.axis('off')
    plt.title('Activation (ReLU)')

    plt.subplot(1, 4, 4)
    plt.imshow(image_condense, cmap='gray')
    plt.axis('off')
    plt.title('Pooling')

    plt.show()
