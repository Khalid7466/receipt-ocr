"""
Conversion script to transform CORD dataset JSON annotations to YOLO11n format.

YOLO11n expects:
- One .txt file per image with same name
- Each line: class_id x_center y_center width height (normalized 0-1)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class CORDToYOLOConverter:
    """Convert CORD format annotations to YOLO11n txt format."""
    
    # Map CORD categories to class IDs
    # Customize this based on your needs
    CATEGORY_MAPPING = {
        # Menu items
        "menu.nm": 0,
        "menu.num": 1,
        "menu.cnt": 2,
        "menu.price": 3,
        "menu.itemsubtotal": 4,
        "menu.unitprice": 5,
        "menu.discountprice": 6,
        "menu.vatyn": 7,
        "menu.etc": 8,
        
        # Menu sub-items (combo/bundle items)
        "menu.sub.nm": 9,
        "menu.sub.cnt": 10,
        "menu.sub.price": 11,
        "menu.sub.unitprice": 12,
        
        # Void menu items (cancelled items)
        "void_menu.nm": 26,
        "void_menu.price": 27,
        
        # Sub totals
        "sub_total.subtotal_price": 13,
        "sub_total.discount_price": 14,
        "sub_total.tax_price": 15,
        "sub_total.service_price": 16,
        "sub_total.othersvc_price": 28,
        "sub_total.etc": 17,
        
        # Totals
        "total.total_price": 18,
        "total.menuqty_cnt": 19,
        "total.menutype_cnt": 20,
        "total.creditcardprice": 21,
        "total.cashprice": 22,
        "total.changeprice": 23,
        "total.emoneyprice": 24,
        "total.total_etc": 25,
    }
    
    def __init__(self, category_mapping: Dict[str, int] = None):
        """
        Initialize converter with optional custom category mapping.
        
        Args:
            category_mapping: Dictionary mapping category names to class IDs
        """
        if category_mapping:
            self.CATEGORY_MAPPING = category_mapping
    
    def extract_bbox_from_quad(self, quad: Dict) -> Tuple[float, float, float, float]:
        """
        Extract bounding box (x_min, y_min, x_max, y_max) from quad coordinates.
        
        Args:
            quad: Dictionary with x1, y1, x2, y2, x3, y3, x4, y4 coordinates
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        x_coords = [quad['x1'], quad['x2'], quad['x3'], quad['x4']]
        y_coords = [quad['y1'], quad['y2'], quad['y3'], quad['y4']]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        return x_min, y_min, x_max, y_max
    
    def normalize_bbox(self, bbox: Tuple[float, float, float, float], 
                      img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert absolute coordinates to normalized YOLO format.
        
        YOLO11n expects: x_center, y_center, width, height (all in range [0, 1])
        
        Args:
            bbox: Tuple of (x_min, y_min, x_max, y_max)
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            Tuple of (x_center, y_center, width, height) normalized to 0-1
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Calculate center and dimensions
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        
        # Normalize to 0-1
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return x_center, y_center, width, height
    
    def validate_normalized_values(self, x_center: float, y_center: float, 
                                   width: float, height: float) -> bool:
        """
        Validate that normalized values are within valid range [0, 1].
        YOLO11n is strict about this.
        
        Args:
            x_center, y_center, width, height: Normalized YOLO values
            
        Returns:
            True if valid, False otherwise
        """
        # Allow slight overflow due to floating point precision, clamp to [0, 1]
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                0 <= width <= 1 and 0 <= height <= 1):
            return False
        
        # Width and height should be > 0
        if width <= 0 or height <= 0:
            return False
        
        return True
    
    def convert_json_to_yolo(self, json_path: str) -> List[str]:
        """
        Convert a single CORD JSON file to YOLO11n format.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            List of YOLO format strings (one per detection)
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle case where entire content is a JSON string
        if isinstance(data, str):
            data = json.loads(data)
        
        # Extract image dimensions
        try:
            img_width = data['meta']['image_size']['width']
            img_height = data['meta']['image_size']['height']
        except KeyError as e:
            print(f"Error: Missing image dimensions in {json_path}: {e}")
            return []
        
        # Validate image dimensions
        if img_width <= 0 or img_height <= 0:
            print(f"Error: Invalid image dimensions in {json_path}: {img_width}x{img_height}")
            return []
        
        yolo_lines = []
        
        # Process each valid line (annotation group)
        for valid_line in data.get('valid_line', []):
            category = valid_line.get('category')
            
            # Skip if category not in mapping
            if category not in self.CATEGORY_MAPPING:
                print(f"Warning: Unknown category '{category}' in {json_path}")
                continue
            
            class_id = self.CATEGORY_MAPPING[category]
            
            # Process all words in this annotation
            for word in valid_line.get('words', []):
                quad = word.get('quad')
                if not quad:
                    continue
                
                try:
                    # Extract and convert coordinates
                    bbox = self.extract_bbox_from_quad(quad)
                    x_center, y_center, width, height = self.normalize_bbox(
                        bbox, img_width, img_height
                    )
                    
                    # Clamp values to [0, 1] for YOLO11n compatibility
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    # Validate
                    if not self.validate_normalized_values(x_center, y_center, width, height):
                        print(f"Warning: Invalid normalized bbox in {json_path}: "
                              f"({x_center}, {y_center}, {width}, {height})")
                        continue
                    
                    # YOLO11n format: class_id x_center y_center width height
                    # All coordinates normalized to [0, 1]
                    yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    yolo_lines.append(yolo_line)
                
                except Exception as e:
                    print(f"Error processing word in {json_path}: {e}")
                    continue
        
        return yolo_lines
    
    def process_directory(self, input_dir: str, output_dir: str, split: str = None):
        """
        Process all JSON files in a directory and save as YOLO11n txt files.
        
        Args:
            input_dir: Directory containing JSON files
            output_dir: Directory to save YOLO txt files
            split: Optional split name (e.g., 'train', 'val', 'test') to filter by
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        json_files = sorted(input_path.glob('*.json'))
        
        if not json_files:
            print(f"No JSON files found in {input_dir}")
            return
        
        processed = 0
        skipped = 0
        errors = 0
        
        for json_file in json_files:
            try:
                # Optionally filter by split
                if split:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, str):
                        data = json.loads(data)
                    if data.get('meta', {}).get('split') != split:
                        skipped += 1
                        continue
                
                # Convert JSON to YOLO format
                yolo_lines = self.convert_json_to_yolo(str(json_file))
                
                # Save to txt file
                txt_filename = json_file.stem + '.txt'
                txt_path = output_path / txt_filename
                
                with open(txt_path, 'w', encoding='utf-8') as f:
                    for line in yolo_lines:
                        f.write(line + '\n')
                
                processed += 1
                if processed % 50 == 0:
                    print(f"Processed {processed} files...")
            
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
                errors += 1
                continue
        
        print(f"\n✓ Conversion complete!")
        print(f"  Processed: {processed} files")
        print(f"  Skipped: {skipped} files")
        print(f"  Errors: {errors} files")
        print(f"  Output directory: {output_dir}")
        print(f"  Total classes: {len(self.CATEGORY_MAPPING)}")
        print(f"\nFormat: YOLO11n compatible (class_id x_center y_center width height)")
    
    def generate_classes_yaml(self, output_path: str):
        """
        Generate a data.yaml file for YOLO11n training.
        
        Args:
            output_path: Path to save the data.yaml file
        """
        import yaml
        
        # Create reverse mapping (class_id -> category name)
        nc = len(self.CATEGORY_MAPPING)
        names = {v: k for k, v in self.CATEGORY_MAPPING.items()}
        
        data = {
            'path': '/path/to/dataset',  # Placeholder, user should update
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': nc,
            'names': names
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        print(f"Generated data.yaml at {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert CORD dataset annotations to YOLO11n format'
    )
    parser.add_argument(
        'input_dir',
        help='Input directory containing JSON files'
    )
    parser.add_argument(
        'output_dir',
        help='Output directory for YOLO txt files'
    )
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test'],
        help='Filter by dataset split (optional)'
    )
    parser.add_argument(
        '--generate-yaml',
        action='store_true',
        help='Generate data.yaml configuration file'
    )
    
    args = parser.parse_args()
    
    converter = CORDToYOLOConverter()
    converter.process_directory(args.input_dir, args.output_dir, args.split)
    
    if args.generate_yaml:
        yaml_path = os.path.join(args.output_dir, 'data.yaml')
        converter.generate_classes_yaml(yaml_path)


if __name__ == '__main__':
    main()