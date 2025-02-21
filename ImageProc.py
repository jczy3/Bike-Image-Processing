'''import cv2
print(cv2.__version__)
image_path = '/Users/josephcaraan/Downloads/guts-a-berserk-character-analysis.jpg'
image = cv2.imread(image_path) 
if image is None:
    print("Error: Unable to load image.")
else:
    # Display the image in a window
    cv2.imshow('Input Image', image)
    
    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import torch

# Check if PyTorch is working
x = torch.rand(5, 3)
print(x)'''

import cv2
import numpy as np

path = '/Users/josephcaraan/Downloads/'
img = cv2.imread(path + 'street-bicycle-rack-isolated-white-background-47283553.webp', 0)
cv2.imshow('Original Image', img)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Apply adaptive thresholding instead of fixed threshold
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

cv2.imshow('Thresholded Image', thresh)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by area (remove small contours)
min_contour_area = 500  # Adjust this value based on your image size
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Print the number of filtered contours (lines)
print('Number of lines detected:', len(filtered_contours))

# Optional: Draw the filtered contours on the image for visualization
output_img = cv2.drawContours(img.copy(), filtered_contours, -1, (0, 255, 0), 2)
cv2.imshow('Filtered Contours', output_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
