import cv2
import os
from .filters import (
    convert_to_grayscale, 
    apply_gaussian_blur, 
    apply_binarization,
    apply_clahe,
    apply_denoising,
    apply_clahe_denoise
)
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
    Returns: CLAHE + Denoised image optimized for OCR (Text Recognition).
    
    Based on testing across CORD + SROIE datasets:
    - CLAHE + Denoise achieved 0.7030 avg confidence (BEST)
    - Original: 0.6825
    - Binarization: 0.4837 (WORST - hurts OCR!)
    
    The pipeline:
    1. Grayscale conversion
    2. CLAHE contrast enhancement (handles uneven lighting)
    3. Non-local means denoising (removes noise, preserves edges)
    """
    original = cv2.imread(image_path)
    if original is None: return None
    
    # Apply the winning preprocessing: CLAHE + Denoise
    processed = apply_clahe_denoise(original, clip_limit=2.0, denoise_h=8)
    
    # Resize if needed for OCR engine
    final_img = resize_if_needed(processed)
    
    return final_img

def ocr_preprocess_legacy(image_path):
    """
    LEGACY: Binary, Deskewed, Resized image.
    
    WARNING: Testing showed binarization hurts OCR accuracy!
    Use ocr_preprocess() instead for better results.
    Kept for backward compatibility.
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