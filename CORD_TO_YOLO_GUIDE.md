# CORD to YOLO11n Conversion - Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Basic Usage](#basic-usage)
5. [Understanding the Script](#understanding-the-script)
6. [Class Reference](#class-reference)
7. [Output Format](#output-format)
8. [Advanced Usage](#advanced-usage)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The `cord_to_yolo.py` script converts receipt OCR annotations from **CORD format** (JSON) to **YOLO11n format** (TXT) for object detection training.

### What it does:
- ✅ Reads CORD JSON annotations (quad coordinates)
- ✅ Extracts bounding boxes from receipt fields
- ✅ Converts pixel coordinates to normalized YOLO format
- ✅ Maps receipt field categories to class IDs
- ✅ Generates data.yaml for YOLOv11 training
- ✅ Handles multiple splits (train/val/test)

### Key Feature:
Instead of detecting generic objects, it detects **receipt-specific fields** like:
- Menu items and prices
- Subtotals, discounts, taxes
- Payment methods and amounts
- Void/cancelled items

---

## Prerequisites

### System Requirements:
- Python 3.8+
- 2+ GB free disk space (for converted dataset)
- Windows, macOS, or Linux
- `uv` package manager installed

### Python Dependencies:
```bash
# Required packages
pyyaml
# (json is built-in)
```

---

## Installation

### Step 1: Check Python Version
```bash
python --version
# Should be 3.8 or higher
```

### Step 2: Install UV (if not already installed)
```bash
# On Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 3: Verify UV Installation
```bash
uv --version
```

### Step 4: Install Dependencies
```bash
# Navigate to project directory
cd d:\College\sections\receipt-ocr

# Install dependencies using uv
uv pip install pyyaml
```

### Step 5: Verify the Script
```bash
# Test the script with help
uv run src/detection/cord_to_yolo.py --help
```

**Output:**
```
usage: cord_to_yolo.py [-h] [--split {train,val,test}] [--generate-yaml] input_dir output_dir

Convert CORD dataset annotations to YOLO11n format

positional arguments:
  input_dir             Input directory containing JSON files
  output_dir            Output directory for YOLO txt files

optional arguments:
  -h, --help            show this help message and exit
  --split {train,val,test}
                        Filter by dataset split (optional)
  --generate-yaml       Generate data.yaml configuration file
```

---

## Basic Usage

### Simple Conversion
```bash
uv run src/detection/cord_to_yolo.py <input_dir> <output_dir>
```

### With YAML Generation
```bash
uv run src/detection/cord_to_yolo.py <input_dir> <output_dir> --generate-yaml
```

### Filter by Split
```bash
uv run src/detection/cord_to_yolo.py <input_dir> <output_dir> --split train
```

---

## Understanding the Script

### Architecture

```
CORDToYOLOConverter (Main Class)
├── CATEGORY_MAPPING (Dict)
│   └── Maps CORD categories → Class IDs
├── extract_bbox_from_quad()
│   └── Converts 4-corner coords to axis-aligned bbox
├── normalize_bbox()
│   └── Converts pixel coords to normalized [0,1]
├── validate_normalized_values()
│   └── Ensures YOLO11n compliance
├── convert_json_to_yolo()
│   └── Converts single JSON file
├── process_directory()
│   └── Batch processes all JSON files
└── generate_classes_yaml()
    └── Creates data.yaml config
```

### Flow Diagram

```
Input: image_0.json (CORD format)
    ↓
Load JSON file
    ↓
Extract image dimensions (width × height)
    ↓
For each annotation in "valid_line":
    ├─ Get category (e.g., "menu.price")
    ├─ Map to class_id (e.g., 3)
    ├─ For each word:
    │  ├─ Extract quad: {x1, y1, x2, y2, x3, y3, x4, y4}
    │  ├─ Convert to bbox: (x_min, y_min, x_max, y_max)
    │  ├─ Normalize to YOLO: (x_center, y_center, width, height)
    │  ├─ Clamp values to [0, 1]
    │  ├─ Validate coordinates
    │  └─ Write to output
    ↓
Output: image_0.txt (YOLO format)
```

---

## Class Reference

**Total: 29 Classes (0-28)**

### Menu Items (Classes 0-12)

| Class | Name | Description | Example |
|-------|------|-------------|---------|
| 0 | menu.nm | Menu item name | "Coca Cola" |
| 1 | menu.num | Item number/ID | "901016" |
| 2 | menu.cnt | Item quantity | "2" |
| 3 | menu.price | Item unit price | "5000" |
| 4 | menu.itemsubtotal | Item total | "10000" |
| 5 | menu.unitprice | Unit price | "5000/unit" |
| 6 | menu.discountprice | Item discount | "-1000" |
| 7 | menu.vatyn | VAT indicator | "Y" or "N" |
| 8 | menu.etc | Other menu info | "Special notes" |
| 9 | menu.sub.nm | Sub-item name (bundles) | "Combo item 1" |
| 10 | menu.sub.cnt | Sub-item quantity | "1" |
| 11 | menu.sub.price | Sub-item price | "2500" |
| 12 | menu.sub.unitprice | Sub-item unit price | "2500/unit" |

### Void Items (Classes 26-27)

| Class | Name | Description |
|-------|------|-------------|
| 26 | void_menu.nm | Cancelled item name |
| 27 | void_menu.price | Cancelled item price |

### Subtotals (Classes 13-17, 28)

| Class | Name | Description | Example |
|-------|------|-------------|---------|
| 13 | sub_total.subtotal_price | Subtotal before tax | "45000" |
| 14 | sub_total.discount_price | Total discount | "-5000" |
| 15 | sub_total.tax_price | Tax amount | "4500" |
| 16 | sub_total.service_price | Service charge | "2250" |
| 28 | sub_total.othersvc_price | Other service charges | "1000" |
| 17 | sub_total.etc | Other subtotal info | "Misc charges" |

### Totals (Classes 18-25)

| Class | Name | Description | Example |
|-------|------|-------------|---------|
| 18 | total.total_price | Final total | "50000" |
| 19 | total.menuqty_cnt | Total items count | "5" |
| 20 | total.menutype_cnt | Total item types | "3" |
| 21 | total.creditcardprice | Credit card amount | "50000" |
| 22 | total.cashprice | Cash payment | "50000" |
| 23 | total.changeprice | Change amount | "0" |
| 24 | total.emoneyprice | E-money/digital payment | "50000" |
| 25 | total.total_etc | Other total info | "Rounding" |

---

## Output Format

### Example Input (CORD JSON)
```json
{
  "meta": {
    "image_size": {"width": 432, "height": 648},
    "split": "train",
    "image_id": 0
  },
  "valid_line": [
    {
      "category": "menu.nm",
      "words": [{
        "text": "Coca Cola",
        "quad": {
          "x1": 50, "y1": 100,
          "x2": 200, "y2": 100,
          "x3": 200, "y3": 130,
          "x4": 50, "y4": 130
        }
      }]
    },
    {
      "category": "menu.price",
      "words": [{
        "text": "5000",
        "quad": {
          "x1": 250, "y1": 100,
          "x2": 300, "y2": 100,
          "x3": 300, "y3": 130,
          "x4": 250, "y4": 130
        }
      }]
    }
  ]
}
```

### Example Output (YOLO TXT)
```
0 0.289352 0.177469 0.347222 0.046296
3 0.636574 0.177469 0.115741 0.046296
```

**Format:** `class_id x_center y_center width height`

- **class_id**: 0-28 (which field type)
- **x_center, y_center**: Center position (0-1 normalized)
- **width, height**: Box dimensions (0-1 normalized)

### Generated data.yaml
```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
nc: 29
names:
  0: menu.nm
  1: menu.num
  2: menu.cnt
  # ... all 29 classes
```

---

## Advanced Usage

### 1. Convert Multiple Splits Separately

```bash
# Navigate to project directory
cd d:\College\sections\receipt-ocr

# Train set
uv run src/detection/cord_to_yolo.py \
  data/cord/raw/train \
  data/yolo/train \
  --generate-yaml

# Validation set
uv run src/detection/cord_to_yolo.py \
  data/cord/raw/val \
  data/yolo/val

# Test set
uv run src/detection/cord_to_yolo.py \
  data/cord/raw/test \
  data/yolo/test
```

### 2. Filter by Split in Mixed Directory

```bash
# Extract only training files from mixed directory
uv run src/detection/cord_to_yolo.py \
  data/cord/raw/all_data \
  data/yolo/train \
  --split train
```

### 3. Custom Category Mapping

Edit `CATEGORY_MAPPING` in the script to use different class IDs:

```python
# Example: Group categories by importance
CATEGORY_MAPPING = {
    # High priority
    "menu.price": 0,
    "total.total_price": 1,
    
    # Medium priority
    "menu.nm": 2,
    "menu.cnt": 3,
    
    # ... etc
}
```

### 4. Batch Processing Script

Create a file `convert_all.bat` (Windows):

```batch
@echo off
cd d:\College\sections\receipt-ocr

echo Converting train set...
uv run src/detection/cord_to_yolo.py data/cord/raw/train data/yolo/train

echo Converting val set...
uv run src/detection/cord_to_yolo.py data/cord/raw/val data/yolo/val

echo Converting test set...
uv run src/detection/cord_to_yolo.py data/cord/raw/test data/yolo/test --generate-yaml

echo All conversions complete!
```

Or for Linux/Mac, create `convert_all.sh`:

```bash
#!/bin/bash

cd /path/to/receipt-ocr

echo "Converting train..."
uv run src/detection/cord_to_yolo.py data/cord/raw/train data/yolo/train

echo "Converting val..."
uv run src/detection/cord_to_yolo.py data/cord/raw/val data/yolo/val

echo "Converting test..."
uv run src/detection/cord_to_yolo.py data/cord/raw/test data/yolo/test --generate-yaml

echo "Done!"
```

Run with:
```bash
bash convert_all.sh
```

---

## Examples

### Example 1: Simple Conversion

**Command:**
```bash
cd d:\College\sections\receipt-ocr
uv run src/detection/cord_to_yolo.py data/cord/raw/train data/yolo/train
```

**Output:**
```
Processed 50 files...
Processed 100 files...
Processed 150 files...

✓ Conversion complete!
  Processed: 800 files
  Skipped: 0 files
  Errors: 0 files
  Output directory: data/yolo/train
  Total classes: 29

Format: YOLO11n compatible (class_id x_center y_center width height)
```

**Result:**
```
data/yolo/train/
├── image_0.txt
├── image_1.txt
├── image_2.txt
└── ... (800 files total)
```

### Example 2: Full Pipeline

```bash
cd d:\College\sections\receipt-ocr

# Step 1: Convert all splits
uv run src/detection/cord_to_yolo.py data/cord/raw/train data/yolo/train
uv run src/detection/cord_to_yolo.py data/cord/raw/val data/yolo/val
uv run src/detection/cord_to_yolo.py data/cord/raw/test data/yolo/test --generate-yaml

# Step 2: Verify outputs
dir data/yolo/
REM Output:
REM train\       (800 txt files)
REM val\         (100 txt files)
REM test\        (100 txt files)
REM data.yaml

# Step 3: Check a converted file
type data/yolo/train/image_0.txt
REM Output:
REM 0 0.289352 0.177469 0.347222 0.046296
REM 3 0.636574 0.177469 0.115741 0.046296
REM 18 0.450000 0.750000 0.200000 0.050000
```

### Example 3: Training with YOLOv11

Create `train_yolo11.py`:

```python
from ultralytics import YOLO

# Load pre-trained YOLOv11n
model = YOLO('yolo11n.pt')

# Train on your converted dataset
results = model.train(
    data='data/yolo/data.yaml',
    epochs=100,
    imgsz=640,
    device=0,  # GPU device (0) or CPU (use device=-1)
    patience=20,
    save=True,
)

# Test on validation set
val_results = model.val()

# Predict on test set
pred_results = model.predict(source='data/images/test')
```

Run with:
```bash
uv run train_yolo11.py
```

Or install ultralytics first:
```bash
uv pip install ultralytics
uv run train_yolo11.py
```

---

## Troubleshooting

### Problem 1: "No JSON files found"

**Cause:** Input directory is empty or path is wrong

**Solution:**
```bash
# Check if files exist
ls data/cord/raw/train/*.json | head

# Verify path is absolute/relative correctly
uv run src/detection/cord_to_yolo.py `
  "d:\College\sections\receipt-ocr\data\cord\raw\train" `
  "d:\College\sections\receipt-ocr\data\yolo\train"
```

### Problem 2: "ModuleNotFoundError: No module named 'yaml'"

**Cause:** PyYAML not installed

**Solution:**
```bash
uv pip install pyyaml
```

### Problem 3: "Unknown category" warnings

**Cause:** Dataset has categories not in CATEGORY_MAPPING

**Solution:** Add missing categories to `CATEGORY_MAPPING`:

```python
CATEGORY_MAPPING = {
    # ... existing mappings ...
    "your_new_category": 29,  # Add new ID
}
```

### Problem 4: Output files are empty (0 bytes)

**Cause:** JSON parsing failed or no valid annotations

**Solution:**
```bash
# Check a sample JSON file
cat data/cord/raw/train/image_0.json | uv run -c "import json, sys; print(json.dumps(json.load(sys.stdin), indent=2))" | head -50

# Verify image dimensions exist
uv run -c "
import json
with open('data/cord/raw/train/image_0.json') as f:
    data = json.load(f)
    print(data['meta']['image_size'])
"
```

### Problem 5: Invalid normalized coordinates warning

**Cause:** Annotation quad extends beyond image bounds

**Solution:** This is handled automatically (values are clamped to [0,1]). It's usually safe to ignore, but indicates potential annotation issues.

---

## Integration with Training Pipeline

### 1. Prepare Directory Structure
```
data/
├── images/
│   ├── train/     (receipt images)
│   ├── val/
│   └── test/
├── labels/
│   ├── train/     (YOLO txt files)
│   ├── val/
│   └── test/
└── data.yaml
```

### 2. Create Symbolic Links (Optional)

**Windows (PowerShell):**
```powershell
# Link images
New-Item -ItemType SymbolicLink -Path data\images\train -Target ..\..\path\to\actual\images
New-Item -ItemType SymbolicLink -Path data\images\val -Target ..\..\path\to\actual\images
```

**Linux/Mac:**
```bash
# Link images
ln -s path/to/actual/images data/images/train
ln -s path/to/actual/images data/images/val
```

Or copy them:
```bash
cp -r original/images/train data/images/train
cp -r original/images/val data/images/val
cp -r original/images/test data/images/test
```

### 3. Update data.yaml
```yaml
path: /absolute/path/to/data
train: images/train
val: images/val
test: images/test
nc: 29
names:
  0: menu.nm
  1: menu.num
  2: menu.cnt
  # ... etc
```

### 4. Train YOLOv11n

Create `train.py`:
```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.train(data='data/yolo/data.yaml', epochs=100)
```

Run:
```bash
uv pip install ultralytics opencv-python
uv run train.py
```

---

## Performance Tips

1. **Batch Processing:** Script is optimized for 1000+ files
2. **Memory:** Each JSON ~50KB, minimal RAM usage
3. **Speed:** ~500-1000 files/minute on standard hardware
4. **Validation:** Always check a few output .txt files manually
5. **UV Benefits:** Faster package management and script execution

---

## Quick Reference Commands

### Setup
```bash
uv pip install pyyaml
```

### Convert Dataset
```bash
# Train split
uv run src/detection/cord_to_yolo.py data/cord/raw/train data/yolo/train

# Val split
uv run src/detection/cord_to_yolo.py data/cord/raw/val data/yolo/val

# Test split with YAML
uv run src/detection/cord_to_yolo.py data/cord/raw/test data/yolo/test --generate-yaml
```

### Verify Conversion
```bash
# Count converted files
ls data/yolo/train/*.txt | wc -l

# Check first file
head data/yolo/train/image_0.txt

# Check YAML
cat data/yolo/data.yaml
```

### Train YOLOv11n
```bash
uv pip install ultralytics
uv run train.py
```

---

## Summary

| Aspect | Details |
|--------|---------|
| **Input** | CORD JSON annotations (quad coords) |
| **Output** | YOLO11n TXT (normalized coordinates) |
| **Classes** | 29 receipt field types |
| **Format** | `class_id x_center y_center width height` |
| **Speed** | ~1000 files/min |
| **Dependencies** | pyyaml |
| **Python** | 3.8+ |
| **Package Manager** | uv (recommended) |

You're now ready to convert your dataset and train YOLOv11 for receipt field detection!

---

## Additional Resources

- [YOLOv11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [CORD Dataset Paper](https://arxiv.org/abs/1910.14950)
- [UV Package Manager](https://astral.sh/uv/)

---

**Last Updated:** December 18, 2025
**Script Location:** `src/detection/cord_to_yolo.py`
**License:** MIT
