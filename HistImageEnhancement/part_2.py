# Import the libraries that will be used.
import numpy as np
import cv2 
import matplotlib.pyplot as plt 

# Define a function to generate the histogram plot of the grayscale image.
def histogram(source_image):
    
    # Initialize an array of 256 zeros for intensity values.
    histogram_array = [0] * 256
    
    # Traverse the image and increase the pixel values for each pixel.
    for row in source_image:
        for pixel in row:
            histogram_array[pixel] += 1
    
    # Generate a histogram plot.
    plt.bar(range(256), histogram_array)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Grayscale Image')
    plt.show()

    return histogram_array

# Define a function to execute the contrast stretching operation on a grayscale image.
def contrast_stretching(source_image):
    
    # Detect the minimum and maximum pixel values in an image.
    pixel_min_val = np.min(source_image)
    pixel_max_val = np.max(source_image)
    
    # Apply the given contrast stretching formula.
    new_image = ((source_image - pixel_min_val) / (pixel_max_val - pixel_min_val)) * 127 + 128
    new_image = new_image.astype(np.uint8)  
    
    return new_image

# Define the image paths.
image_path_1 = "/Users/aral/Documents/Homeworks/Homework 1/HistImageEnhancement/hist1.jpg"
image_path_2 = "/Users/aral/Documents/Homeworks/Homework 1/HistImageEnhancement/hist2.jpg"
image_path_3 = "/Users/aral/Documents/Homeworks/Homework 1/HistImageEnhancement/contrastive_strecth.png"

# Load the images and read them on grayscale.
image_1 = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
image_2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)
image_3 = cv2.imread(image_path_3, cv2.IMREAD_GRAYSCALE)

# Test with the image (assuming image is a grayscale numpy array)
stretched_img = contrast_stretching(image_3)

# Plot the contrast stretched image
plt.imshow(stretched_img, cmap='gray')
plt.title('Contrast Stretched Image')
plt.show()

# You can also plot the histogram to compare with the original
new_hist = histogram(stretched_img)


