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

def apply_clahe(gray_image, clip_limit=2.0, tile_size=8):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Best for: Receipts with uneven lighting, shadows, or low contrast.
    Tested as the #2 best preprocessing method (0.7006 avg confidence).
    
    Args:
        gray_image: Grayscale input image.
        clip_limit: Threshold for contrast limiting (2.0 is optimal).
        tile_size: Size of grid for histogram equalization.
        
    Returns:
        Contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray_image)

def apply_denoising(gray_image, h=8):
    """
    Applies Non-local Means Denoising for noise reduction.
    
    Best for: Scanner noise, camera noise while preserving text edges.
    
    Args:
        gray_image: Grayscale input image.
        h: Filter strength (higher = more denoising, may blur). 8-10 is optimal.
        
    Returns:
        Denoised grayscale image.
    """
    return cv2.fastNlMeansDenoising(gray_image, None, h, 7, 21)

def apply_clahe_denoise(image, clip_limit=2.0, denoise_h=8):
    """
    Applies CLAHE + Denoising - THE BEST preprocessing for OCR.
    
    Tested on CORD + SROIE datasets:
    - Overall confidence: 0.7030 (best)
    - CORD: 0.674
    - SROIE: 0.732
    
    Args:
        image: Input image (BGR or grayscale).
        clip_limit: CLAHE clip limit (2.0 optimal).
        denoise_h: Denoising strength (8 optimal).
        
    Returns:
        Preprocessed grayscale image optimized for OCR.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply CLAHE for contrast enhancement
    enhanced = apply_clahe(gray, clip_limit)
    
    # Apply denoising
    denoised = apply_denoising(enhanced, denoise_h)
    
    return denoised

def apply_binarization(image, c_value=3):
    """
    Applies Adaptive Thresholding + Morphological Opening.
    
    NOTE: Testing showed binarization HURTS OCR accuracy (0.56 vs 0.70).
    Use apply_clahe_denoise() instead for OCR tasks.
    This function is kept for YOLO detection preprocessing.
    
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