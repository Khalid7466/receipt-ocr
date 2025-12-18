import cv2
import numpy as np
import math
from .filters import convert_to_grayscale

def resize_if_needed(image, target_width=1280):
    """Resizes image to target width maintaining aspect ratio."""
    (h, w) = image.shape[:2]
    if w <= target_width:
        return image
    
    ratio = target_width / float(w)
    new_h = int(h * ratio)
    
    # INTER_AREA is best for shrinking text images
    return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)

def rotate_image(image, angle):
    """Rotates image around center with replicate padding."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def detect_skew_hough(image):
    """
    Detects skew angle using Hough Line Transform.
    Best for complex backgrounds (wood/bamboo).
    """
    # Ensure edges calculation
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLinesP(
        edges, 1, math.pi/180, threshold=100, 
        minLineLength=100, maxLineGap=20
    )
    
    if lines is None: return 0.0
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if -45 < angle < 45:
            angles.append(angle)
            
    if not angles: return 0.0
    
    return np.median(angles)