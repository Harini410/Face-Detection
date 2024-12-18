import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import hadamard
from scipy.ndimage import median_filter


# Load and preprocess image
image = cv2.imread('face.jpg')

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Unable to open image. Check the file path and format.")
    exit(1)  # Exit if the image cannot be opened

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize the grayscale image
resized_image = cv2.resize(gray, (256, 256))


#Image Transforms

# Discrete Fourier Transform (DFT)
dft_shift = np.fft.fftshift(cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT))
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

# Discrete Cosine Transform (DCT)
dct_image = cv2.normalize(cv2.dct(np.float32(gray)), None, 0, 255, cv2.NORM_MINMAX)

# Hadamard Transform
resized_image = cv2.resize(gray, (256, 256))  # Resize to 256x256 for Hadamard
H = hadamard(256)
hadamard_image = cv2.normalize(np.dot(H, np.dot(resized_image, H.T)), None, 0, 255, cv2.NORM_MINMAX)

# Slant Transform
S = np.array([[np.cos(np.pi * (2 * i + 1) * j / (2 * 256)) for j in range(256)] for i in range(256)]) * (2 / 256) ** 0.5
slant_image = cv2.normalize(np.dot(S, np.dot(resized_image, S.T)), None, 0, 255, cv2.NORM_MINMAX)

# Karhunen-Loève Transform (PCA-based)
mean, eigenvectors = cv2.PCACompute(gray.astype(np.float32), mean=None)
kl_image = cv2.normalize(cv2.PCAProject(gray.astype(np.float32), mean, eigenvectors), None, 0, 255, cv2.NORM_MINMAX)

# Downsampled Image (reduce resolution)
downsampled_image = gray[::2, ::2]

# Quantization to 4 intensity levels
def quantize_image(image, levels):
    max_val = image.max()
    return np.floor(image / max_val * (levels - 1)) * (max_val / (levels - 1))

quantized_image = quantize_image(gray, 4)

# Calculate the histogram
histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Combined layout (2 rows, 3 columns)
plt.figure(figsize=(18, 12))

# Plotting each image
plt.subplot(2, 3, 1), plt.imshow(gray, cmap='gray'), plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('DFT'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 3), plt.imshow(dct_image, cmap='gray'), plt.title('DCT'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 4), plt.imshow(hadamard_image, cmap='gray'), plt.title('Hadamard Transform'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 5), plt.imshow(slant_image, cmap='gray'), plt.title('Slant Transform'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 6), plt.imshow(kl_image, cmap='gray'), plt.title('KL Transform'), plt.xticks([]), plt.yticks([])

# Add a new figure for the remaining images
plt.figure(figsize=(18, 12))
plt.subplot(2, 3, 1), plt.imshow(downsampled_image, cmap='gray'), plt.title('Downsampled Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2), plt.imshow(quantized_image, cmap='gray'), plt.title('Quantized Image (4 levels)'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 3), plt.plot(histogram, color='black'), plt.title('Histogram'), plt.xlabel('Pixel Intensity'), plt.ylabel('Frequency')
plt.xlim([0, 256])  # Adjust x-axis to fit pixel intensity range
plt.grid(True)  # Add grid for better visibility

plt.tight_layout(pad=3.0)  # Add padding for better spacing
plt.show()

def negative_image(image):
    # Generate the negative image by subtracting from 255
    return 255 - image

# Assuming 'gray' is your original grayscale image
negative_gray = negative_image(gray)

# Create a single plot for original and negative images
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

# Negative Image
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.imshow(negative_gray, cmap='gray')
plt.title('Negative Image')
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()


def power_law_transformation(image, gamma):
    # Normalize the image to the range [0, 1]
    normalized_image = image / 255.0
    # Apply the power-law transformation
    power_law_image = cv2.pow(normalized_image, gamma)
    # Scale back to range [0, 255] and convert to uint8
    power_law_image = np.uint8(power_law_image * 255)
    return power_law_image

gamma_corrected_image = power_law_transformation(gray, gamma=2.2)  # Example with gamma=2.2

def logarithmic_transformation(image):
    # Convert to float32 for precision and avoid log(0)
    log_image = np.float32(image) + 1
    # Apply the log transformation
    log_transformed_image = np.log(log_image)
    # Normalize to the range [0, 255]
    log_transformed_image = cv2.normalize(log_transformed_image, None, 0, 255, cv2.NORM_MINMAX)
    # Convert back to uint8
    return np.uint8(log_transformed_image)

log_image = logarithmic_transformation(gray)

def exponential_transformation(image, constant=1):
    # Normalize image to range [0, 1]
    normalized_image = image / 255.0
    # Apply exponential transformation
    exp_image = np.exp(constant * normalized_image) - 1
    # Normalize the result to [0, 255] and convert to uint8
    exp_image = np.uint8(cv2.normalize(exp_image, None, 0, 255, cv2.NORM_MINMAX))
    return exp_image

exp_image = exponential_transformation(gray, constant=1)

def inverse_logarithmic_transformation(image):
    # Normalize image to range [0, 1]
    normalized_image = image / 255.0
    # Apply inverse log transformation
    inv_log_image = np.exp(normalized_image) - 1
    # Scale back to range [0, 255] and convert to uint8
    inv_log_image = np.uint8(cv2.normalize(inv_log_image, None, 0, 255, cv2.NORM_MINMAX))
    return inv_log_image

inv_log_image = inverse_logarithmic_transformation(gray)

def sqrt_transformation(image):
    # Normalize the image to the range [0, 1]
    normalized_image = image / 255.0
    # Apply the square root transformation
    sqrt_image = np.sqrt(normalized_image)
    # Scale back to range [0, 255] and convert to uint8
    sqrt_image = np.uint8(sqrt_image * 255)
    return sqrt_image

sqrt_image = sqrt_transformation(gray)

# Display the transformations in one layout
transforms = [
    (gray, 'Original Image'),
    (gamma_corrected_image, 'Gamma Correction'),
    (log_image, 'Logarithmic Transformation'),
    (exp_image, 'Exponential Transformation'),
    (inv_log_image, 'Inverse Logarithmic Transformation'),
    (sqrt_image, 'Square Root Transformation')
]

plt.figure(figsize=(18, 8))
for i, (img, title) in enumerate(transforms):
    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

def bitwise_and(image1, image2):
    return cv2.bitwise_and(image1, image2)

and_image = bitwise_and(gray, gamma_corrected_image)

def bitwise_or(image1, image2):
    return cv2.bitwise_or(image1, image2)

or_image = bitwise_or(gray, gamma_corrected_image)

def bitwise_xor(image1, image2):
    return cv2.bitwise_xor(image1, image2)

xor_image = bitwise_xor(gray, gamma_corrected_image)

def bitwise_not(image):
    return cv2.bitwise_not(image)

not_image = bitwise_not(gray)

# Perform bitwise operations on the grayscale and gamma-corrected images
and_image = bitwise_and(gray, gamma_corrected_image)
or_image = bitwise_or(gray, gamma_corrected_image)
xor_image = bitwise_xor(gray, gamma_corrected_image)
not_image = bitwise_not(gray)

# Display the results
operations = [
    (and_image, 'Bitwise AND'),
    (or_image, 'Bitwise OR'),
    (xor_image, 'Bitwise XOR'),
    (not_image, 'Bitwise NOT')
]

plt.figure(figsize=(12, 8))
for i, (img, title) in enumerate(operations):
    plt.subplot(2, 2, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

def add_images(image1, image2):
    # Add two images and clip values to ensure they stay within [0, 255]
    added_image = cv2.add(image1, image2)
    return added_image

# Example: Adding gamma corrected image and logarithmic transformed image
added_image = add_images(gamma_corrected_image, log_image)

def subtract_images(image1, image2):
    # Subtract image2 from image1 and ensure pixel values stay within [0, 255]
    subtracted_image = cv2.subtract(image1, image2)
    return subtracted_image

# Example: Subtracting logarithmic transformed image from the gamma corrected image
subtracted_image = subtract_images(gamma_corrected_image, log_image)

def multiply_images(image1, image2):
    # Multiply two images and normalize the result to keep values in [0, 255]
    multiplied_image = cv2.multiply(image1, image2)
    return multiplied_image

# Example: Multiplying gamma corrected image and logarithmic transformed image
multiplied_image = multiply_images(gamma_corrected_image, log_image)

def divide_images(image1, image2):
    # Divide image1 by image2 and normalize the result
    image2 = np.where(image2 == 0, 1, image2)  # To avoid division by zero
    divided_image = cv2.divide(image1, image2)
    return divided_image

# Example: Dividing gamma corrected image by logarithmic transformed image
divided_image = divide_images(gamma_corrected_image, log_image)
# Operations between gamma corrected and log transformed images
added_image = add_images(gamma_corrected_image, log_image)
subtracted_image = subtract_images(gamma_corrected_image, log_image)
multiplied_image = multiply_images(gamma_corrected_image, log_image)
divided_image = divide_images(gamma_corrected_image, log_image)

# Displaying the results
operations = [
    (added_image, 'Addition'),
    (subtracted_image, 'Subtraction'),
    (multiplied_image, 'Multiplication'),
    (divided_image, 'Division')
]

plt.figure(figsize=(12, 8))
for i, (img, title) in enumerate(operations):
    plt.subplot(2, 2, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()


# Image Enhancement

# Contrast stretching
contrast_stretched = cv2.normalize(gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)

# High-pass filtering (Laplacian)
laplacian = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))

# Low-pass filtering (Gaussian)
low_pass_filtered = cv2.GaussianBlur(gray, (9, 9), 0)

# Butterworth low-pass filter
def butterworth_lowpass_filter(img, D0, n):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.array([[1 / (1 + ((np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) / D0) ** (2 * n)))
                      for j in range(cols)] for i in range(rows)])
    return mask

# Apply Butterworth filter
butterworth_filter = butterworth_lowpass_filter(gray, 30, 2)
butterworth_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(gray)) * butterworth_filter)))



# Display all enhanced images together
enhancements = [
    (gray, 'Original Image'),
    (contrast_stretched, 'Contrast Stretched'),
    (laplacian, 'High-pass (Laplacian)'),
    (low_pass_filtered, 'Low-pass (Gaussian)'),
    (butterworth_filtered, 'Butterworth Filtered')
]

# Plotting all enhancements in one layout
plt.figure(figsize=(18, 8))
for i, (img, title) in enumerate(enhancements):
    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

def histogram_equalization(image):
    # Calculate histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    
    # Normalize the histogram
    cdf = hist.cumsum()  # Cumulative distribution function
    cdf_normalized = cdf * hist.max() / cdf.max()  # Normalize for plotting

    # Mask all pixels with value 0 to avoid division by zero
    cdf_m = np.ma.masked_equal(cdf, 0)

    # Normalize the cumulative distribution function
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

    # Fill in the masked values with 0
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Apply histogram equalization
    equalized_image = cdf[image]

    return equalized_image, hist, bins



# Perform histogram equalization
equalized_image, hist, bins = histogram_equalization(gray)

# Calculate histogram for the equalized image
equalized_hist, _ = np.histogram(equalized_image.flatten(), 256, [0, 256])

# Create a figure to display the images and histograms
plt.figure(figsize=(18, 8))

# Original Image
plt.subplot(2, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')  # Hide axis ticks

# Histogram of Original Image
plt.subplot(2, 3, 2)
plt.plot(bins[:-1], hist, color='black')
plt.title('Histogram of Original Image')
plt.xlim([0, 256])
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# Equalized Image
plt.subplot(2, 3, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Histogram Equalized Image')
plt.axis('off')  # Hide axis ticks

# Histogram of Equalized Image
plt.subplot(2, 3, 4)
plt.plot(np.arange(256), equalized_hist, color='black')
plt.title('Histogram of Equalized Image')
plt.xlim([0, 256])
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


#Image Restoration

# Add Gaussian Noise
gaussian_noise = np.random.normal(0, 25, gray.shape).astype('uint8')
gaussian_noisy_image = cv2.add(gray, gaussian_noise)

# Add Poisson Noise
poisson_noisy_image = np.random.poisson(gray).astype('uint8')

# Add Motion Blur
motion_kernel = np.ones((15, 15)) / 225
motion_blurred = cv2.filter2D(gray, -1, motion_kernel)

# Add Salt-and-Pepper Noise
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy = image.copy()
    num_salt = np.ceil(salt_prob * image.size)
    salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[salt_coords[0], salt_coords[1]] = 255  # Salt
    num_pepper = np.ceil(pepper_prob * image.size)
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[pepper_coords[0], pepper_coords[1]] = 0  # Pepper
    return noisy

salt_pepper_noisy_image = add_salt_and_pepper_noise(gray, 0.01, 0.01)

# Add Speckle Noise
speckle_noise = gray + gray * np.random.normal(0, 0.1, gray.shape)
speckle_noisy_image = np.clip(speckle_noise, 0, 255).astype('uint8')

# Add Uniform Noise
uniform_noise = np.random.uniform(-20, 20, gray.shape).astype('uint8')
uniform_noisy_image = cv2.add(gray, uniform_noise)

# Apply Canny Edge Detection
edges = cv2.Canny(gray, 100, 200)

# Display all images in one layout
noisy_images = [
    (gray, 'Input Image'),
    (gaussian_noisy_image, 'Gaussian Noisy Image'),
    (poisson_noisy_image, 'Poisson Noisy Image'),
    (motion_blurred, 'Motion Blurred Image'),
    (salt_pepper_noisy_image, 'Salt and Pepper Noisy Image'),
    (speckle_noisy_image, 'Speckle Noisy Image'),
    (uniform_noisy_image, 'Uniform Noisy Image'),
    (edges, 'Canny Edge Detection')
]

plt.figure(figsize=(18, 12))
for i, (img, title) in enumerate(noisy_images):
    plt.subplot(2, 4, i + 1)  # 2 rows and 4 columns
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()



# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, sigma=25):
    noisy = image + np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    return noisy

# Function to add salt-and-pepper noise
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy = image.copy()
    num_salt = np.ceil(salt_prob * image.size)
    salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[salt_coords[0], salt_coords[1]] = 255
    num_pepper = np.ceil(pepper_prob * image.size)
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[pepper_coords[0], pepper_coords[1]] = 0
    return noisy

# Add noise to the image
gaussian_noisy_image = add_gaussian_noise(gray)
salt_pepper_noisy_image = add_salt_and_pepper_noise(gray, 0.01, 0.01)
motion_blurred = cv2.GaussianBlur(gray, (15, 15), 0)

# Apply different filters
def apply_filters(noisy_image):
    wiener_filtered = cv2.fastNlMeansDenoising(noisy_image, None, 30, 7, 21)
    median_filtered = cv2.medianBlur(noisy_image, 3)
    bilateral_filtered = cv2.bilateralFilter(noisy_image, 9, 75, 75)
    gaussian_filtered = cv2.GaussianBlur(noisy_image, (5, 5), 0)
    return wiener_filtered, median_filtered, bilateral_filtered, gaussian_filtered

# Define noisy images and their names
noisy_images = [gaussian_noisy_image, salt_pepper_noisy_image, motion_blurred]
noise_names = ['Gaussian Noise', 'Salt-and-Pepper Noise', 'Motion Blur']

# Filter names for different layouts
filter_names = ['Wiener', 'Median', 'Bilateral', 'Gaussian']

# Create separate layouts for each filter
for filter_name in filter_names:
    plt.figure(figsize=(15, 10))
    
    for i, (noisy_image, noise_name) in enumerate(zip(noisy_images, noise_names)):
        filtered_images = apply_filters(noisy_image)

        # Determine the index for filtered image based on filter name
        if filter_name == 'Wiener':
            filtered_image = filtered_images[0]
        elif filter_name == 'Median':
            filtered_image = filtered_images[1]
        elif filter_name == 'Bilateral':
            filtered_image = filtered_images[2]
        elif filter_name == 'Gaussian':
            filtered_image = filtered_images[3]

        # Show noisy image
        plt.subplot(len(noisy_images), 3, 3 * i + 1)
        plt.imshow(noisy_image, cmap='gray')
        plt.title(f'{noise_name}\nNoisy Image')
        plt.xticks([]), plt.yticks([])

        # Show filtered image
        plt.subplot(len(noisy_images), 3, 3 * i + 2)
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f'{filter_name} Filtered')
        plt.xticks([]), plt.yticks([])

        # Show restored image
        plt.subplot(len(noisy_images), 3, 3 * i + 3)
        plt.imshow(cv2.convertScaleAbs(filtered_image), cmap='gray')  # Ensure uint8 format
        plt.title('Restored Image')
        plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()




# IMAGE SEGMENTATION AND REPRESENTATION


# Load the image in grayscale
image = cv2.imread('face.jpg', cv2.IMREAD_GRAYSCALE)

# Edge Detection Techniques
# Sobel
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel_edges = cv2.magnitude(sobel_x, sobel_y)

# Laplacian
laplacian_edges = cv2.Laplacian(image, cv2.CV_64F)

# Canny
canny_edges = cv2.Canny(image, 100, 200)

# Thresholding Techniques
# Global Thresholding
_, global_thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# Otsu's Thresholding
_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive Thresholding
adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Local Thresholding
def local_threshold(image, block_size=15, C=5):
    mean = cv2.blur(image, (block_size, block_size))
    local_thresh = np.where(image > mean - C, 255, 0).astype(np.uint8)
    return local_thresh

local_thresh_image = local_threshold(image)

# Morphological Operations
kernel = np.ones((5, 5), np.uint8)

# Erosion
eroded_image = cv2.erode(global_thresh, kernel, iterations=1)

# Dilation
dilated_image = cv2.dilate(global_thresh, kernel, iterations=1)

# Opening
opened_image = cv2.morphologyEx(global_thresh, cv2.MORPH_OPEN, kernel)

# Closing
closed_image = cv2.morphologyEx(global_thresh, cv2.MORPH_CLOSE, kernel)

# Skeletonization
def skeletonize(image):
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    
    while not done:
        eroded = cv2.erode(image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, temp)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()

        zeros = size - cv2.countNonZero(image)
        if zeros == size:
            done = True
    
    return skel

skeleton_image = skeletonize(global_thresh)

# Polygonization
def polygonize(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygon_image = np.zeros_like(image)
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(polygon_image, [approx], -1, 255, -1)
    return polygon_image

polygon_image = polygonize(global_thresh)

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('face.jpg', cv2.IMREAD_GRAYSCALE)  # Load the image as grayscale

# Region Growing
def region_growing(img, seed):
    h, w = img.shape
    segmented = np.zeros_like(img)
    seeds = [seed]
    visited = np.zeros_like(img, dtype=bool)

    # Get the intensity of the seed point
    seed_value = img[seed]

    while seeds:
        x, y = seeds.pop(0)
        if visited[x, y]:
            continue
        visited[x, y] = True
        segmented[x, y] = img[x, y]

        # Iterate through 8-connected neighbors
        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = x + i, y + j
                if 0 <= nx < h and 0 <= ny < w:
                    if not visited[nx, ny] and abs(int(img[nx, ny]) - int(seed_value)) < 10:
                        seeds.append((nx, ny))

    return segmented

# Region Splitting
def split_segment(image):
    h, w = image.shape
    segments = [
        image[:h//2, :w//2],  # Top-left
        image[:h//2, w//2:],  # Top-right
        image[h//2:, :w//2],  # Bottom-left
        image[h//2:, w//2:]   # Bottom-right
    ]
    return segments

def region_merging(segments, threshold=10):
    h, w = image.shape
    merged = np.zeros_like(image)

    # Define the positions for each segment in the merged image
    positions = [
        (0, 0),           # Top-left
        (0, w // 2),     # Top-right
        (h // 2, 0),     # Bottom-left
        (h // 2, w // 2) # Bottom-right
    ]

    for idx, segment in enumerate(segments):
        if segment.size > 0:
            mean_val = np.mean(segment)

            pos_y, pos_x = positions[idx]
            h_seg, w_seg = segment.shape

            condition = segment > mean_val - threshold
            
            merged[pos_y:pos_y + h_seg, pos_x:pos_x + w_seg][condition] = mean_val

    return merged

# Choose a seed point (make sure it's within the bounds of the image)
seed_point = (100, 100)  # Adjust this point as needed

# Perform region growing
region_grow_segmented = region_growing(image, seed_point)

# Split the image into segments
split_segmented = split_segment(image)

# Merge the segmented regions
merged_image = region_merging(split_segmented, threshold=10)

# Display results
plt.figure(figsize=(15, 10))

# Display original image
plt.subplot(3, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Display segmented image
plt.subplot(3, 2, 2)
plt.imshow(region_grow_segmented, cmap='gray')
plt.title('Region Growing Segmentation')
plt.axis('off')

# Display split segments
for i, segment in enumerate(split_segmented):
    plt.subplot(3, 2, i + 3)  # Change this line to fit all segments in the next rows
    plt.imshow(segment, cmap='gray')
    plt.title(f'Segment {i + 1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Display the merged image
plt.figure(figsize=(5, 5))
plt.imshow(merged_image, cmap='gray')
plt.title('Merged Image')
plt.axis('off')
plt.show()

# Boundary Detection (Contours)
contours, _ = cv2.findContours(global_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = np.zeros_like(image)
cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)

# Connected Components Analysis
num_labels, components_image = cv2.connectedComponents(global_thresh)

# Hough Transform for Line Detection
lines = cv2.HoughLinesP(global_thresh, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
hough_image = np.zeros_like(image)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(hough_image, (x1, y1), (x2, y2), (255, 255, 255), 1)

# Convex Hull
hull_image = np.zeros_like(image)
for cnt in contours:
    hull = cv2.convexHull(cnt)
    cv2.drawContours(hull_image, [hull], -1, (255, 255, 255), 1)


# Assuming the following variables are defined with appropriate image data
# image, sobel_edges, laplacian_edges, canny_edges, global_thresh, otsu_thresh,
# adaptive_thresh, local_thresh_image, eroded_image, dilated_image, opened_image,
# closed_image, skeleton_image, polygon_image, region_grow_segmented, merged_image,
# contour_image, components_image, chain_code_result, shape_context_placeholder,
# fourier_descriptor_placeholder

# List to hold the titles and images for easy access
# Organize images into groups for separate plots

# Assuming all image variables are defined (image, sobel_edges, laplacian_edges, etc.)

# Organize images into groups for separate plots
image_groups = [
    [  # Plot 1: Edge Detection Techniques
        ('Original Image', image),
        ('Sobel Edge Detection', sobel_edges),
        ('Laplacian Edge Detection', laplacian_edges),
        ('Canny Edge Detection', canny_edges)
    ],
    [  # Plot 2: Thresholding Techniques
        ('Global Thresholding', global_thresh),
        ('Otsu\'s Thresholding', otsu_thresh),
        ('Adaptive Thresholding', adaptive_thresh),
        ('Local Thresholding', local_thresh_image)
    ],
    [  # Plot 3: Morphological Operations segmentation
        ('Eroded Image', eroded_image),
        ('Dilated Image', dilated_image),
        ('Opened Image', opened_image),
        ('Closed Image', closed_image),
    ],
    [  # Plot 4: Region Growing Segmentation
        ('Region Growing Segmentation', region_grow_segmented),
        ('Splitted Regions', split_segmented),
        ('Merged Regions', merged_image)
    ],
      [ #representation
      ('Contours',contour_image ),
      ('Connected Components', components_image ),
      ('Hough Transform', hough_image),
      ('Convex Hull',hull_image),
      ('Skeletonized Image', skeleton_image),
      ('Polygonized Image', polygon_image)
      ],

    
]
  
# Create each plot
for plot_index, group in enumerate(image_groups):
    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

    # Display each image in the current group
    num_images = len(group)  # Get the number of images in the current group
    for i, (title, img) in enumerate(group):
        plt.subplot(2, (num_images + 1) // 2, i + 1)  # Adjust the layout dynamically

        # Check if img is valid (non-empty and a numpy array)
        if img is not None and isinstance(img, np.ndarray) and img.size > 0:
            plt.title(title)
            plt.imshow(img, cmap='gray')
        else:
            plt.title(f'{title} (No Image)')
            plt.axis('off')

        plt.axis('off')  # Turn off axis for a cleaner look

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()  # Display the plot
image = cv2.imread('face.jpg')
if image is None:
    raise FileNotFoundError("Image not found. Please check the path.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours (boundaries)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Check if contours are found
if contours:
    main_contour = max(contours, key=cv2.contourArea)

    # Boundary Descriptors
    length = cv2.arcLength(main_contour, True)  # Perimeter
    area = cv2.contourArea(main_contour)  # Area enclosed
    compactness = (4 * np.pi * area) / (length ** 2) if length != 0 else 0  # Compactness
    convex_hull = cv2.convexHull(main_contour)
    convexity = area / cv2.contourArea(convex_hull) if cv2.contourArea(convex_hull) != 0 else 0  # Convexity
    circularity = (4 * np.pi * area) / (length ** 2) if length != 0 else 0  # Circularity
    (x, y), (MA, ma), angle = cv2.fitEllipse(main_contour)  # Major and Minor axis lengths
    eccentricity = np.sqrt(1 - (min(MA, ma) / max(MA, ma)) ** 2) if max(MA, ma) != 0 else 0  # Eccentricity

    # Regional Descriptors
    mean_intensity = cv2.mean(gray, mask=binary)[0]  # Mean intensity
    std_dev = np.std(gray[binary == 255])  # Standard deviation
    shape_features = {
        'Area': area,
        'Aspect Ratio': float(MA / ma) if ma != 0 else 0
    }

    # Print descriptor values for debugging
    print("Boundary Descriptors:")
    print(f"Length: {length}, Area: {area}, Compactness: {compactness}, "
          f"Convexity: {convexity}, Circularity: {circularity}, Eccentricity: {eccentricity}")

    print("\nRegional Descriptors:")
    print(f"Mean Intensity: {mean_intensity}, Standard Deviation: {std_dev}, "
          f"Area: {shape_features['Area']}, Aspect Ratio: {shape_features['Aspect Ratio']}")

    # Plotting results
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    # Original Image
    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    # Binary Image
    axs[0, 1].imshow(binary, cmap='gray')
    axs[0, 1].set_title('Binary Image')
    axs[0, 1].axis('off')

    # Detected Contours
    axs[1, 0].imshow(cv2.drawContours(image.copy(), [main_contour], -1, (0, 255, 0), 3))
    axs[1, 0].set_title('Detected Contours')
    axs[1, 0].axis('off')

    # Boundary Descriptors Plot
    boundary_titles = ['Length', 'Area', 'Compactness', 'Convexity', 'Circularity', 'Eccentricity']
    boundary_values = [length, area, compactness, convexity, circularity, eccentricity]

    axs[1, 1].bar(boundary_titles, boundary_values, color='cyan')
    axs[1, 1].set_title('Boundary Descriptors')
    axs[1, 1].set_ylabel('Values')
    axs[1, 1].tick_params(axis='x', rotation=45)

    # Regional Descriptors Plot
    regional_titles = ['Mean Intensity', 'Std Dev', 'Area', 'Aspect Ratio']
    regional_values = [mean_intensity, std_dev, shape_features['Area'], shape_features['Aspect Ratio']]

    # Using a 2x2 grid for the last two plots
    axs[2, 0].bar(regional_titles, regional_values, color='magenta')
    axs[2, 0].set_title('Regional Descriptors')
    axs[2, 0].set_ylabel('Values')
    axs[2, 0].tick_params(axis='x', rotation=45)

    # Remove the last unused subplot
    fig.delaxes(axs[2, 1])  # Delete the last subplot

    # Adjust layout
    plt.tight_layout()
    plt.show()
else:
    print("No contours found in the image.")











#IMAGE COMPRESSION

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load the image in grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    return image

def calculate_mse(original, compressed):
    """Calculate Mean Squared Error between original and compressed images."""
    return np.mean((original - compressed) ** 2)

def plot_images(original, compressed, title):
    """Plot original and compressed images with MSE value."""
    mse_value = calculate_mse(original, compressed)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Compressed Image")
    plt.imshow(compressed, cmap='gray')
    plt.axis('off')

    plt.suptitle(f"{title} - MSE: {mse_value:.4f}")
    plt.show()

def apply_dct(block):
    """Apply Discrete Cosine Transform (DCT) to a block."""
    return cv2.dct(block.astype(np.float32))

def apply_idct(block):
    """Apply Inverse Discrete Cosine Transform (IDCT) to a block."""
    return cv2.idct(block)

def jpeg_compress(image):
    """JPEG compression using DCT."""
    quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                     [12, 12, 14, 19, 26, 58, 60, 55],
                                     [14, 13, 16, 24, 40, 57, 69, 56],
                                     [14, 17, 22, 29, 51, 87, 80, 62],
                                     [18, 22, 37, 56, 68, 109, 103, 77],
                                     [24, 35, 55, 64, 81, 104, 113, 92],
                                     [49, 64, 78, 87, 103, 121, 120, 101],
                                     [72, 92, 95, 98, 112, 100, 103, 99]])

    compressed_image = np.zeros_like(image)

    for i in range(0, image.shape[0], 8):
        for j in range(0, image.shape[1], 8):
            block = image[i:i+8, j:j+8]
            if block.shape == (8, 8):  # Ensure it's a complete block
                dct_block = apply_dct(block)
                quantized_block = np.round(dct_block / quantization_matrix)
                # Perform IDCT on the quantized DCT block
                compressed_image[i:i+8, j:j+8] = np.clip(apply_idct(quantized_block * quantization_matrix), 0, 255)

    return compressed_image.astype(np.uint8)

def lossless_compress(image_path):
    """Simulate lossless compression (PNG)."""
    image = load_image(image_path)
    _, compressed_image = cv2.imencode('.png', image)
    return cv2.imdecode(compressed_image, cv2.IMREAD_GRAYSCALE)

def compare_image_compression(image_path):
    """Compare different image compression techniques."""
    original = load_image(image_path)

    # Lossless Compression
    lossless_compressed = lossless_compress(image_path)
    plot_images(original, lossless_compressed, "Lossless Compression (PNG)")

    # JPEG Compression
    jpeg_compressed = jpeg_compress(original)
    plot_images(original, jpeg_compressed, "Lossy Compression (JPEG)")

# Path to your input image
image_path = 'face.jpg'  # Change to your image path
compare_image_compression(image_path)






