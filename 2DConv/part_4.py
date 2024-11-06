# Import the required libraries.
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift


# Define spatial domain convolution function.
def spatial_convolution(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    kernel = np.flipud(np.fliplr(kernel))
    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    output = np.zeros_like(image)
    
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)
    
    return output

# Define a function to create a Gaussian kernel in the frequency domain.
def gaussian_kernel(shape, sigma):

    m, n = shape
    y, x = np.indices((m, n))
    center_y, center_x = m // 2, n // 2
    kernel = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    return kernel / kernel.sum()


# Define Sobel operators.
sobel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Define Prewitt operators.
prewitt_horizontal = np.array([[-1, 0, 1], 
                               [-1, 0, 1], 
                               [-1, 0, 1]])

prewitt_vertical = np.array([[-1, -1, -1], 
                             [0, 0, 0], 
                             [1, 1, 1]])


# Read the image in grayscale and normalize.
spatial_image_path = "/Users/aral/Documents/Homeworks/Homework 1/2DConv/convolution_spatial_domain.jpg"
spatial_image = cv2.imread(spatial_image_path, cv2.IMREAD_GRAYSCALE)

gaussian_image_path = "/Users/aral/Documents/Homeworks/Homework 1/2DConv/convolution_freq_domain.jpg"
gaussian_image = cv2.imread(gaussian_image_path, cv2.IMREAD_GRAYSCALE)

spatial_image = spatial_image.astype(np.float32) / 255.0
gaussian_image = gaussian_image.astype(np.float32) / 255.0

# Apply Sobel operator.
sobel_horizontal_result = spatial_convolution(spatial_image, sobel_horizontal)
sobel_vertical_result = spatial_convolution(spatial_image, sobel_vertical)

# Take absolute values and combine results for gradient magnitude for Sobel operator.
sobel_magnitude = np.sqrt(np.abs(sobel_horizontal_result)**2 + np.abs(sobel_vertical_result)**2)
sobel_magnitude = (sobel_magnitude / sobel_magnitude.max()) * 255  # Normalize to [0, 255]
sobel_magnitude = sobel_magnitude.astype(np.uint8)

# Apply Prewitt operator.
prewitt_horizontal_result = spatial_convolution(spatial_image, prewitt_horizontal)
prewitt_vertical_result = spatial_convolution(spatial_image, prewitt_vertical)

# Take absolute values and combine results for gradient magnitude for Prewitt operator.
prewitt_magnitude = np.sqrt(np.abs(prewitt_horizontal_result)**2 + np.abs(prewitt_vertical_result)**2)
prewitt_magnitude = (prewitt_magnitude / prewitt_magnitude.max()) * 255  # Normalize to [0, 255]
prewitt_magnitude = prewitt_magnitude.astype(np.uint8)


# Plot results for Sobel operator.
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(spatial_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(np.abs(sobel_horizontal_result), cmap='gray')
plt.title('Sobel operator in horizontal')

plt.subplot(1, 3, 3)
plt.imshow(sobel_magnitude, cmap='gray')
plt.title('Sobel Edge Detection')

plt.show()


# Plot results for Prewitt operator.
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(spatial_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(np.abs(prewitt_horizontal_result), cmap='gray')
plt.title('Prewitt operator in horizontal')

plt.subplot(1, 3, 3)
plt.imshow(prewitt_magnitude, cmap='gray')
plt.title('Prewitt Edge Detection')

plt.show()

# Apply the Fourier Transform to the image.
image_fft = fft2(gaussian_image)
image_fft_shifted = fftshift(image_fft)

# Define the Gaussian kernel in the frequency domain.
sigma = 15  # Adjust the sigma value for more or less blurring.
gaussian_filter = gaussian_kernel(gaussian_image.shape, sigma)

# Apply the Gaussian filter in the frequency domain.
filtered_fft = image_fft_shifted * gaussian_filter

# Inverse shift and inverse Fourier Transform to convert back to spatial domain.
filtered_image = np.real(ifft2(np.fft.ifftshift(filtered_fft)))

# Plot the original and the filtered images.
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(gaussian_image, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title("Gaussian Blurred Image in Frequency Domain")

plt.show()