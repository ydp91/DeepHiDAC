import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the file paths for the images
file_paths = {
    'a': './datasets/env/resources/a.jpg',
    'b': './datasets/env/resources/b.jpg',
    'c': './datasets/env/resources/c.jpg',
    'd': './datasets/env/resources/d.jpg',
    'e': './datasets/env/resources/e.jpg',
    'f': './datasets/env/resources/f.jpg'
}

# Function to calculate the Jaccard similarity index
def calculate_jaccard_similarity(mask1, mask2):
    intersection = np.sum((mask1 > 0) & (mask2 > 0))
    union = np.sum((mask1 > 0) | (mask2 > 0))
    return intersection / union if union != 0 else 0

# Function to create a mask for a given color range in HSV space
def create_color_mask(hsv_image, lower_range1, upper_range1, lower_range2=None, upper_range2=None):
    mask1 = cv2.inRange(hsv_image, lower_range1, upper_range1)
    if lower_range2 is not None and upper_range2 is not None:
        mask2 = cv2.inRange(hsv_image, lower_range2, upper_range2)
        return mask1 | mask2
    else:
        return mask1

# Adjust the HSV color ranges to capture larger areas of red, yellow, and green
red_lower1 = np.array([0, 50, 50])
red_upper1 = np.array([15, 255, 255])
red_lower2 = np.array([165, 50, 50])
red_upper2 = np.array([179, 255, 255])
yellow_lower = np.array([15, 50, 50])
yellow_upper = np.array([45, 255, 255])
green_lower = np.array([45, 50, 50])
green_upper = np.array([85, 255, 255])

# Function to process images and calculate similarities
def process_and_calculate_similarities(file_paths):
    reference_image = cv2.imread(file_paths['a'])
    reference_shape = reference_image.shape[:2]
    color_similarities = {}

    for label, path in file_paths.items():
        # Load, resize, and convert to HSV color space
        image = cv2.imread(path)
        resized_image = cv2.resize(image, (reference_shape[1], reference_shape[0]))
        hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

        # Create masks for the color ranges
        red_mask = create_color_mask(hsv_image, red_lower1, red_upper1, red_lower2, red_upper2)
        # cv2.imshow("red",red_mask)
        # cv2.waitKey()
        yellow_mask = create_color_mask(hsv_image, yellow_lower, yellow_upper)
        # cv2.imshow("yellow", yellow_mask)
        # cv2.waitKey()
        green_mask = create_color_mask(hsv_image, green_lower, green_upper)
        # cv2.imshow("green", green_mask)
        # cv2.waitKey()

        if label != 'a':  # Skip the reference image itself
            color_similarities[label] = {
                'Red Heat Zone Similarity': calculate_jaccard_similarity(red_mask, reference_red_mask),
                'Yellow Heat Zone Similarity': calculate_jaccard_similarity(yellow_mask, reference_yellow_mask),
                'Green Heat Zone Similarity': calculate_jaccard_similarity(green_mask, reference_green_mask)
            }

    return color_similarities

# Calculate the reference masks once, as they will be compared against multiple times
reference_image_hsv = cv2.cvtColor(cv2.imread(file_paths['a']), cv2.COLOR_BGR2HSV)
reference_red_mask = create_color_mask(reference_image_hsv, red_lower1, red_upper1, red_lower2, red_upper2)
reference_yellow_mask = create_color_mask(reference_image_hsv, yellow_lower, yellow_upper)
reference_green_mask = create_color_mask(reference_image_hsv, green_lower, green_upper)

# Run the processing and similarity calculation function
color_similarities = process_and_calculate_similarities(file_paths)

# Output the results
for label, similarities in color_similarities.items():
    print(f'Image {label}:')
    for color_zone, similarity in similarities.items():
        print(f'  {color_zone}: {similarity:.2%}')

