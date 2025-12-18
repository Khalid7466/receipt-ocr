import cv2
import numpy as np

def convert_to_grayscale(image):
    """Converts BGR image to Grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Applies Gaussian Blur to reduce sensor noise."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_binarization(image, c_value=3):
    """
    Applies Adaptive Thresholding + Morphological Opening.
    Args:
        image: Grayscale input.
        c_value: Sensitivity (3 is the sweet spot).
    Returns:
        Clean Binary Image (Black text on White background).
    """
    # 1. Adaptive Thresholding
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, c_value
    )
    
    # 2. Morphological Opening (Removes noise dots)
    kernel = np.ones((2, 2), np.uint8)
    clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return clean_binary