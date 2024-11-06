# Import the necessary libraries.
import numpy as np
import cv2 
import matplotlib.pyplot as plt 

# Define a function for the dilation morphological operation. 
def dilation(image, structuring_element):

    # Obtain the dimensions of the source image.
    image_height, image_width = image.shape
    se_height, se_width = structuring_element.shape
    pad_h, pad_w = se_height // 2, se_width // 2
    
    # Perform padding.
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    dilated_image = np.zeros_like(image)
    
    # Perform dilation.
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+se_height, j:j+se_width]
            dilated_image[i, j] = (region * structuring_element).max()
    
    return dilated_image

# Define a function for the erosion morphological operation. 
def erosion(image, structuring_element):

    # Obtain the dimensions of the source image.
    image_height, image_width = image.shape
    se_height, se_width = structuring_element.shape
    pad_h, pad_w = se_height // 2, se_width // 2
    
    # Perform padding.
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    eroded_image = np.zeros_like(image)
    
    # Perform erosion.
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+se_height, j:j+se_width]
            eroded_image[i, j] = (region * structuring_element).min()
    
    return eroded_image

# Define the image path to a variable.
img_path = "/Users/aral/Documents/Homeworks/Homework 1/MorphologicalOperations/morphological_operations.png"

# Define an arbitrary structuring element.
structuring_element = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

# Read the example image as grayscale. 
binary_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply erosion to the original image.
eroded_image = erosion(binary_image, structuring_element)
eroded_image = erosion(eroded_image, structuring_element)
eroded_image = erosion(eroded_image, structuring_element)
eroded_image = erosion(eroded_image, structuring_element)
eroded_image = erosion(eroded_image, structuring_element)
eroded_image = erosion(eroded_image, structuring_element)


# Apply dilation to the eroded image.
dilated_image = dilation(eroded_image, structuring_element)
dilated_image = dilation(eroded_image, structuring_element)
dilated_image = dilation(eroded_image, structuring_element)
dilated_image = dilation(eroded_image, structuring_element)
dilated_image = dilation(eroded_image, structuring_element)
dilated_image = dilation(eroded_image, structuring_element)


# Plot the results.
plt.figure(figsize=(10, 3))

# Display the original image.
plt.subplot(1, 3, 1)
plt.imshow(binary_image, cmap='gray')
plt.title('Original Image.')


# Display the eroded image.
plt.subplot(1, 3, 2)
plt.imshow(dilated_image, cmap='gray')
plt.title('Eroded Image.')

# Display the dilated image.
plt.subplot(1, 3, 3)
plt.imshow(eroded_image, cmap='gray')
plt.title('Dilated Image.')

plt.show()
