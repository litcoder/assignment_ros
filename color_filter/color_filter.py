import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

# Show usage
if len(sys.argv) < 2:
    print(f"{sys.argv[0]} <image_file>")
    exit(1)

# Load image
img = cv2.imread(sys.argv[1])
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Kernels
kernels = {
    "Original": np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]]),
    "Blur": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
    "Gausian Blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16,
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "Sobel (X)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "Sobel (Y)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    "Edge detection": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
}

# Apply kernels
results = [img_rgb]
for name, kernel in list(kernels.items())[1:]:
    filtered = cv2.filter2D(img_rgb, -1, kernel)
    results.append(filtered)

# Plot results
plt.figure(figsize=(15, 8))
for i, (name, _) in enumerate(kernels.items()):
    plt.subplot(3, 3, i+1)
    plt.imshow(results[i])
    plt.title(name)
    plt.axis("off")
plt.tight_layout()

plt.show()
