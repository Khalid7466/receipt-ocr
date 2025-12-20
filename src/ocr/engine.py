"""
Receipt OCR Engine using YOLO Detection + EasyOCR.

This module provides a complete OCR pipeline optimized for receipt images:
1. YOLO detection for text region localization
2. EasyOCR for text extraction
3. Preprocessing from src/preprocessing/pipelines.py
"""

import cv2
import numpy as np
import easyocr
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Import preprocessing from our module
from preprocessing.pipelines import ocr_preprocess
from preprocessing.filters import convert_to_grayscale, apply_gaussian_blur, apply_binarization
from preprocessing.geometry import detect_skew_hough, rotate_image, resize_if_needed


class ReceiptOCR:
    """
    OCR engine optimized for receipt text extraction.
    Uses EasyOCR with preprocessing from preprocessing/pipelines.py
    """
    
    def __init__(
        self, 
        languages: List[str] = ['en'],
        gpu: bool = True,
        model_storage_directory: str = 'models/easyocr'
    ):
        """
        Initialize the OCR engine.
        
        Args:
            languages: List of language codes to support
            gpu: Whether to use GPU acceleration
            model_storage_directory: Directory to store OCR models
        """
        self.reader = easyocr.Reader(
            languages,
            gpu=gpu,
            model_storage_directory=model_storage_directory,
            download_enabled=True
        )
        self.languages = languages
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy using our preprocessing pipeline.
        
        Args:
            image: Input image (BGR format from cv2)
            
        Returns:
            Preprocessed image (binary, deskewed, resized)
        """
        # Convert to grayscale
        gray = convert_to_grayscale(image)
        
        # Clean and binarize
        blurred = apply_gaussian_blur(gray)
        binary = apply_binarization(blurred)
        
        # Detect and correct skew
        angle = detect_skew_hough(binary)
        deskewed = rotate_image(binary, angle)
        
        # Resize if needed
        final_img = resize_if_needed(deskewed)
        
        return final_img
    
    def preprocess_from_path(self, image_path: str) -> np.ndarray:
        """
        Preprocess image from file path using the OCR preprocessing pipeline.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image ready for OCR
        """
        return ocr_preprocess(image_path)
    
    def extract_text(
        self, 
        image: np.ndarray,
        preprocess: bool = True,
        detail: int = 1,
        paragraph: bool = False,
        min_size: int = 10,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4
    ) -> List[Dict]:
        """
        Extract text from an image.
        
        Args:
            image: Input image (BGR or grayscale)
            preprocess: Whether to apply preprocessing
            detail: 0 for text only, 1 for detailed output with bbox & confidence
            paragraph: Whether to merge text into paragraphs
            min_size: Minimum text size to detect
            text_threshold: Text confidence threshold
            low_text: Low text bound
            link_threshold: Link threshold for connecting text
            
        Returns:
            List of detected text with bounding boxes and confidence scores
        """
        if preprocess:
            processed = self.preprocess_image(image)
        else:
            processed = image
        
        results = self.reader.readtext(
            processed,
            detail=detail,
            paragraph=paragraph,
            min_size=min_size,
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
            canvas_size=2560,
            mag_ratio=1.5
        )
        
        formatted_results = []
        for result in results:
            if detail == 1:
                bbox, text, confidence = result
                formatted_results.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': confidence
                })
            else:
                formatted_results.append({'text': result})
        
        return formatted_results
    
    def extract_from_regions(
        self,
        image: np.ndarray,
        regions: List[Tuple[int, int, int, int]],
        padding: int = 5
    ) -> List[Dict]:
        """
        Extract text from specific regions.
        
        Args:
            image: Full image
            regions: List of (x1, y1, x2, y2) bounding boxes
            padding: Pixels to add around each region
            
        Returns:
            List of extracted text for each region
        """
        results = []
        h, w = image.shape[:2]
        
        for i, (x1, y1, x2, y2) in enumerate(regions):
            # Add padding and clip
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Crop region
            region_img = image[y1:y2, x1:x2]
            
            if region_img.shape[0] < 10 or region_img.shape[1] < 10:
                continue
            
            text_results = self.extract_text(region_img, preprocess=True)
            combined_text = ' '.join([r['text'] for r in text_results])
            avg_confidence = np.mean([r['confidence'] for r in text_results]) if text_results else 0
            
            results.append({
                'region_id': i,
                'bbox': (x1, y1, x2, y2),
                'text': combined_text,
                'confidence': avg_confidence,
                'details': text_results
            })
        
        return results


class ReceiptPipeline:
    """
    Complete receipt processing pipeline combining:
    1. YOLO detection for text region localization
    2. EasyOCR for text extraction
    3. Preprocessing from src/preprocessing/pipelines.py
    """
    
    def __init__(
        self,
        yolo_model_path: str = 'models/yolo11n_receipt_detector.pt',
        languages: List[str] = ['en'],
        gpu: bool = True,
        model_storage_directory: str = 'models/easyocr'
    ):
        """
        Initialize the receipt processing pipeline.
        
        Args:
            yolo_model_path: Path to trained YOLO model weights
            languages: List of language codes for OCR
            gpu: Whether to use GPU acceleration
            model_storage_directory: Directory to store EasyOCR models
        """
        from ultralytics import YOLO
        
        # Load YOLO detector
        self.detector = YOLO(yolo_model_path)
        
        # Initialize OCR engine
        self.ocr = ReceiptOCR(
            languages=languages,
            gpu=gpu,
            model_storage_directory=model_storage_directory
        )
    
    def detect_regions(
        self, 
        image: np.ndarray, 
        conf_threshold: float = 0.25
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions using YOLO.
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of (x1, y1, x2, y2) bounding boxes sorted top-to-bottom, left-to-right
        """
        results = self.detector.predict(
            image, 
            conf=conf_threshold,
            verbose=False
        )
        
        regions = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    regions.append((x1, y1, x2, y2, conf))
        
        # Sort by y-coordinate (top to bottom), then x (left to right)
        regions.sort(key=lambda r: (r[1], r[0]))
        
        return [(r[0], r[1], r[2], r[3]) for r in regions]
    
    @staticmethod
    def _bbox_to_rect(bbox) -> Tuple[int, int, int, int]:
        """
        Convert various bbox formats to (x1, y1, x2, y2) rectangle.
        
        Handles:
        - Tuple/list of 4 ints: (x1, y1, x2, y2)
        - EasyOCR polygon: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        if isinstance(bbox, (list, np.ndarray)) and len(bbox) == 4:
            first_elem = bbox[0]
            if isinstance(first_elem, (list, np.ndarray)) and len(first_elem) == 2:
                # It's a polygon - extract bounding rectangle
                points = np.array(bbox)
                x1 = int(np.min(points[:, 0]))
                y1 = int(np.min(points[:, 1]))
                x2 = int(np.max(points[:, 0]))
                y2 = int(np.max(points[:, 1]))
                return (x1, y1, x2, y2)
            else:
                return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        
        return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    
    def process_image(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        use_detection: bool = True
    ) -> Dict:
        """
        Process a receipt image through the full pipeline.
        
        Args:
            image_path: Path to the receipt image
            conf_threshold: Confidence threshold for YOLO detection
            use_detection: If True, use YOLO detection; if False, OCR whole image
            
        Returns:
            Dictionary with:
            - image_path: Path to processed image
            - image_size: (width, height)
            - regions: List of detected regions with text
            - full_text: Combined text from all regions
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        result = {
            'image_path': str(image_path),
            'image_size': (image.shape[1], image.shape[0]),
            'regions': [],
            'full_text': ''
        }
        
        if use_detection:
            # Step 1: Detect text regions with YOLO
            regions = self.detect_regions(image, conf_threshold)
            result['num_regions'] = len(regions)
            
            # Step 2: Extract text from each region with EasyOCR
            for i, (x1, y1, x2, y2) in enumerate(regions):
                region_img = image[y1:y2, x1:x2]
                
                # Skip tiny regions
                if region_img.shape[0] < 10 or region_img.shape[1] < 10:
                    continue
                
                # OCR on region
                ocr_results = self.ocr.extract_text(region_img, preprocess=True)
                text = ' '.join([r['text'] for r in ocr_results])
                avg_conf = np.mean([r['confidence'] for r in ocr_results]) if ocr_results else 0
                
                result['regions'].append({
                    'id': i,
                    'bbox': (x1, y1, x2, y2),
                    'text': text,
                    'confidence': float(avg_conf)
                })
        else:
            # OCR on full image (fallback)
            ocr_results = self.ocr.extract_text(image, preprocess=True)
            for i, r in enumerate(ocr_results):
                rect_bbox = self._bbox_to_rect(r['bbox'])
                result['regions'].append({
                    'id': i,
                    'bbox': rect_bbox,
                    'text': r['text'],
                    'confidence': r['confidence']
                })
        
        # Combine all text (sorted by position)
        result['full_text'] = '\n'.join([r['text'] for r in result['regions'] if r['text']])
        
        return result
    
    def extract_text_only(self, image_path: str, use_detection: bool = True) -> str:
        """
        Convenience method to get just the extracted text.
        
        Args:
            image_path: Path to the receipt image
            use_detection: Whether to use YOLO detection
            
        Returns:
            Extracted text as a single string
        """
        result = self.process_image(image_path, use_detection=use_detection)
        return result['full_text']