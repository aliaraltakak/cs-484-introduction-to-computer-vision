# CS-484: Introduction to Computer Vision
## Homework I

The assignment explores various image processing techniques and algorithms implemented from scratch in Python.

### Homework Contents

- **Part I: Morphological Operations**
  - **Description**: Implements basic morphological operations, dilation and erosion, to clean up and enhance binary images by removing noise and isolating circular objects.
  - **Implementation Details**: 
    - Dilation and erosion functions are implemented with customizable structuring elements.
    - The structuring element is created externally and passed to the functions, making them flexible for different shapes.
  - **Results**: Six erosion operations followed by six dilation operations effectively removed noise and restored object sizes.

- **Part II: Histogram-Based Image Enhancement**
  - **Part II.I: Histogram of a Grayscale Image**
    - **Description**: Generates histograms for grayscale images to visualize pixel intensity distributions.
    - **Results**: Accurate histogram plots highlight differences in image pixel distributions.
  - **Part II.II: Contrastive Stretching**
    - **Description**: Applies contrast stretching using a linear transformation, enhancing contrast by adjusting intensity values.
    - **Results**: Tested on images with varying ranges (0-255, 0-128, 128-255) to demonstrate how contrast stretching affects image clarity.

- **Part III: Otsu’s Thresholding**
  - **Description**: Implements Otsu’s method for automated image thresholding, converting grayscale images into binary images based on pixel intensity.
  - **Implementation Details**: Calculates optimal threshold values using between-class variance, converting images to binary without additional image processing libraries.
  - **Results**: Segmented images reveal effective separation of foreground and background, though some small details may be affected by lighting and uniform pixel distributions.

- **Part IV: 2-D Convolution in Spatial and Frequency Domains**
  - **Spatial Domain Convolution**
    - **Description**: Applies Sobel and Prewitt operators for edge detection, highlighting differences in sensitivity and robustness between the two methods.
    - **Results**: Sobel provides stronger, noise-resistant edges, while Prewitt produces a simpler approximation.
  - **Frequency Domain Convolution**
    - **Description**: Applies Gaussian filtering in the frequency domain to blur images using Fourier Transform methods.
    - **Results**: Gaussian kernel smoothing effectively reduces detail, resulting in a blurred image in the spatial domain.

