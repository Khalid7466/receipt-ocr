"""
Convert SROIE2019 dataset to YOLO format.
SROIE2019 uses quad coordinates similar to CORD, but without explicit category labels.
We'll classify all detections as generic "text" regions.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class SROIEToYOLOConverter:
    """Convert SROIE2019 format to YOLO format."""
    
    # Map all text regions to a single class: "text"
    CATEGORY_MAPPING = {
        "text": 0,  # Generic text region
    }
    
    @staticmethod
    def extract_bbox_from_quad(quad: Dict[str, float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Extract bounding box from quad coordinates.
        
        Args:
            quad: Dict with keys x1, y1, x2, y2, x3, y3, x4, y4 (corner coordinates)
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            (x_min, y_min, x_max, y_max) normalized to [0, 1]
        """
        x_coords = [quad["x1"], quad["x2"], quad["x3"], quad["x4"]]
        y_coords = [quad["y1"], quad["y2"], quad["y3"], quad["y4"]]
        
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        # Normalize to [0, 1]
        x_min = max(0, min(1, x_min / img_width))
        x_max = max(0, min(1, x_max / img_width))
        y_min = max(0, min(1, y_min / img_height))
        y_max = max(0, min(1, y_max / img_height))
        
        return x_min, y_min, x_max, y_max
    
    @staticmethod
    def normalize_bbox(x_min: float, y_min: float, x_max: float, y_max: float) -> Tuple[float, float, float, float]:
        """Convert axis-aligned bbox to YOLO format (center_x, center_y, width, height)."""
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        return x_center, y_center, width, height
    
    @staticmethod
    def validate_normalized_values(x_center: float, y_center: float, width: float, height: float) -> bool:
        """Ensure values are within valid YOLO range [0, 1]."""
        return all(0 <= val <= 1 for val in [x_center, y_center, width, height])
    
    def parse_sroie_box_file(self, file_path: str, img_width: int = 800, img_height: int = 600) -> List[Tuple[int, float, float, float, float]]:
        """
        Parse SROIE box file format.
        Format: x1,y1,x2,y2,x3,y3,x4,y4,text
        Returns: List of (class_id, x_center, y_center, width, height)
        """
        annotations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(',')
                    if len(parts) < 9:
                        continue
                    
                    try:
                        # Extract quad coordinates
                        quad = {
                            "x1": float(parts[0]),
                            "y1": float(parts[1]),
                            "x2": float(parts[2]),
                            "y2": float(parts[3]),
                            "x3": float(parts[4]),
                            "y3": float(parts[5]),
                            "x4": float(parts[6]),
                            "y4": float(parts[7]),
                        }
                        
                        # Extract bounding box
                        x_min, y_min, x_max, y_max = self.extract_bbox_from_quad(quad, img_width, img_height)
                        x_center, y_center, width, height = self.normalize_bbox(x_min, y_min, x_max, y_max)
                        
                        # Validate
                        if not self.validate_normalized_values(x_center, y_center, width, height):
                            continue
                        
                        # All are "text" class
                        class_id = 0
                        annotations.append((class_id, x_center, y_center, width, height))
                    
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse line in {file_path}: {line}")
                        continue
        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        return annotations
    
    def convert_txt_to_yolo(self, box_file: str, output_file: str, img_width: int = 800, img_height: int = 600) -> bool:
        """Convert a single SROIE box file to YOLO format."""
        try:
            annotations = self.parse_sroie_box_file(box_file, img_width, img_height)
            
            if not annotations:
                return False
            
            # Write YOLO format
            with open(output_file, 'w') as f:
                for class_id, x_center, y_center, width, height in annotations:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            return True
        
        except Exception as e:
            print(f"Error converting {box_file}: {e}")
            return False
    
    def process_directory(self, input_dir: str, output_dir: str) -> Tuple[int, int, int]:
        """
        Process all box files in a directory.
        Returns: (processed, skipped, errors)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed = 0
        skipped = 0
        errors = 0
        
        box_files = list(input_path.glob("*.txt"))
        total = len(box_files)
        
        for i, box_file in enumerate(box_files):
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1} files...")
            
            output_file = output_path / box_file.name
            success = self.convert_txt_to_yolo(str(box_file), str(output_file))
            
            if success:
                processed += 1
            else:
                errors += 1
        
        return processed, skipped, errors
    
    def generate_classes_yaml(self, output_dir: str):
        """Generate data.yaml for single class (text)."""
        yaml_content = """names:
  0: text
nc: 1
"""
        yaml_path = Path(output_dir) / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)


def main():
    parser = argparse.ArgumentParser(description="Convert SROIE2019 dataset to YOLO format")
    parser.add_argument("input_dir", help="Input directory containing box files")
    parser.add_argument("output_dir", help="Output directory for YOLO txt files")
    parser.add_argument("--generate-yaml", action="store_true", help="Generate data.yaml")
    
    args = parser.parse_args()
    
    converter = SROIEToYOLOConverter()
    
    print(f"Converting SROIE2019 dataset from: {args.input_dir}")
    processed, skipped, errors = converter.process_directory(args.input_dir, args.output_dir)
    
    print(f"\n✓ Conversion complete!")
    print(f"  Processed: {processed} files")
    print(f"  Skipped: {skipped} files")
    print(f"  Errors: {errors} files")
    print(f"  Output directory: {args.output_dir}")
    
    if args.generate_yaml:
        converter.generate_classes_yaml(args.output_dir)
        print(f"  Generated data.yaml")


if __name__ == "__main__":
    main()
