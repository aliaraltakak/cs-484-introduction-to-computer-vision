# Import the libraries that will be used.
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define a function to execute automated Otsu's Thresholding on a given grayscale image.
def otsu_method(image):

    # Compute the histogram of the input image.
    histogram = np.zeros(256)
    for pixel_value in image.ravel():
        histogram[int(pixel_value)] += 1
    
    # Determine the total number of pixels.
    total_pixels = image.size
    
    # Normalize the histogram to determine the probabilities.
    probability = histogram / total_pixels
    
    # Compute the cumulative sums and cumulative means.
    cumulative_sum = np.cumsum(probability)  
    cumulative_mean = np.cumsum(np.arange(256) * probability)  
    global_mean = cumulative_mean[-1]  
    
    # Compute the between-class variance for each threshold.
    between_class_variance = np.zeros(256)
    for t in range(256):
        if cumulative_sum[t] == 0 or cumulative_sum[t] == 1:  # Avoid division by zero
            continue
        between_class_variance[t] = ((global_mean * cumulative_sum[t] - cumulative_mean[t]) ** 2) / (cumulative_sum[t] * (1 - cumulative_sum[t]))
    
    # Find the optimal threshold that maximizes between-class variance.
    optimal_threshold = np.argmax(between_class_variance)
    
    # Apply the threshold to convert the image to binary.
    binary_image = np.where(image >= optimal_threshold, 255, 0).astype(np.uint8)
    
    
    return binary_image, optimal_threshold

# Define images paths.
image_path_1 = "/Users/aral/Documents/Homeworks/Homework 1/OtsuThresholding/otsu_1.png"
image_path_2 = "/Users/aral/Documents/Homeworks/Homework 1/OtsuThresholding/otsu_2.jpg"

# Load the images.
image_1 = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
image_2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)

# Apply Otsu's method to the image
binary_image_1, optimal_threshold_1 = otsu_method(image_1)
binary_image_2, optimal_threshold_2 = otsu_method(image_2)


# Display the original and binary images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_1, cmap='gray')
plt.title('First Original Grayscale Image')

plt.subplot(1, 2, 2)
plt.imshow(binary_image_1, cmap='gray')
plt.title(f'First Binary Image (Threshold = {optimal_threshold_1})')

plt.show()

# Plot the histogram of the image with the threshold marked
plt.hist(image_1.ravel(), bins=256, range=(0, 256))
plt.axvline(optimal_threshold_1, color='red', linestyle='dashed', linewidth=3)
plt.title(f'Histogram of First Grayscale Image with Otsu Threshold = {optimal_threshold_1}')
plt.xlabel("Pixel Values")
plt.ylabel("Intensity")
plt.show()

# Display the original and binary images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_2, cmap='gray')
plt.title('Second Original Grayscale Image')

plt.subplot(1, 2, 2)
plt.imshow(binary_image_2, cmap='gray')
plt.title(f'Second Binary Image (Threshold = {optimal_threshold_2})')

plt.show()

# Plot the histogram of the image with the threshold marked
plt.hist(image_2.ravel(), bins=256, range=(0, 256))
plt.axvline(optimal_threshold_2, color='green', linestyle='dashed', linewidth=3)
plt.title(f'Histogram of Second Grayscale Image with Otsu Threshold = {optimal_threshold_2}')
plt.xlabel("Pixel Values")
plt.ylabel("Intensity")
plt.show()






