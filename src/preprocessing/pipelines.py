import cv2
import os
from .filters import convert_to_grayscale, apply_gaussian_blur, apply_binarization
from .geometry import detect_skew_hough, rotate_image, resize_if_needed

def yolo_preprocess(image_path):
    """
    Returns: Grayscale, Deskewed, Resized image (For Object Detection).
    """
    original = cv2.imread(image_path)
    if original is None: return None
    
    gray = convert_to_grayscale(original)
    
    # Calculate skew from a temp binary version
    temp_blur = apply_gaussian_blur(gray)
    temp_bin = apply_binarization(temp_blur)
    angle = detect_skew_hough(temp_bin)
    
    deskewed = rotate_image(gray, angle)
    final_img = resize_if_needed(deskewed)
    
    return final_img

def ocr_preprocess(image_path):
    """
    Returns: Binary, Deskewed, Resized image (For Text Recognition).
    """
    original = cv2.imread(image_path)
    if original is None: return None
    
    gray = convert_to_grayscale(original)
    
    # Clean and Binarize
    blurred = apply_gaussian_blur(gray)
    binary = apply_binarization(blurred)
    
    # Calculate skew
    angle = detect_skew_hough(binary)
    
    deskewed = rotate_image(binary, angle)
    final_img = resize_if_needed(deskewed)
    
    return final_img