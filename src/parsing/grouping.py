"""
Table/Line Items Extraction for Receipt Parsing.

This module extracts structured line items (products/services) from receipt text:
- Item names
- Item prices

Filters out aggregation lines (totals, subtotals, tax, etc.)
"""

import re
from typing import List, Dict, Optional, Tuple
from .extractor import fix_ocr_number_errors, clean_price_string


# Keywords that indicate aggregation lines (to be excluded)
AGGREGATION_KEYWORDS = [
    'total', 'subtotal', 'sub-total', 'sub total',
    'tax', 'vat', 'gst', 'service charge',
    'discount', 'savings', 'promo',
    'cash', 'change', 'tendered', 'paid',
    'balance', 'due', 'amount',
    'tip', 'gratuity',
    'round', 'rounding',
]


def is_aggregation_line(line: str) -> bool:
    """
    Check if a line contains aggregation keywords.
    
    These lines should be excluded from item extraction to avoid
    duplicating values that are handled by the extractor module.
    
    Args:
        line: Text line from receipt
        
    Returns:
        True if line contains aggregation keywords
    """
    line_lower = line.lower()
    
    for keyword in AGGREGATION_KEYWORDS:
        if keyword in line_lower:
            return True
    
    return False


def extract_price_from_line(line: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Extract item name and price from a receipt line.
    
    Detection logic:
    - Find the last numeric value in the line (the price)
    - Everything before it is the item name
    
    Args:
        line: Single line from receipt text
        
    Returns:
        Tuple of (item_name, price) or (None, None) if not a valid item line
    """
    if not line or not line.strip():
        return None, None
    
    line = line.strip()
    
    # Skip aggregation lines
    if is_aggregation_line(line):
        return None, None
    
    # Pattern to match price at end of line
    # Handles: 12.99, $12.99, 12,99, 1,234.56
    # Also handles OCR errors in digits
    price_pattern = r'[£$€]?\s*([\d,OoIlSsBZz]+[.,]?\d*)\s*$'
    
    match = re.search(price_pattern, line)
    if not match:
        return None, None
    
    price_str = match.group(1)
    price = clean_price_string(price_str)
    
    # Skip if no valid price found
    if price is None or price <= 0:
        return None, None
    
    # Extract item name (everything before the price)
    item_name = line[:match.start()].strip()
    
    # Clean up item name
    # Remove trailing separators, dots, dashes
    item_name = re.sub(r'[\.\-_:]+\s*$', '', item_name).strip()
    
    # Skip if item name is empty or too short
    if not item_name or len(item_name) < 2:
        return None, None
    
    # Skip if item name is just numbers (likely a code/quantity line)
    if re.match(r'^[\d\s]+$', item_name):
        return None, None
    
    return item_name, price


def parse_quantity_line(line: str) -> Tuple[Optional[str], Optional[int], Optional[float]]:
    """
    Parse a line that may include quantity information.
    
    Formats handled:
    - "2 x Item Name    12.99" → (Item Name, 2, 12.99)
    - "Item Name x2     12.99" → (Item Name, 2, 12.99)
    - "2  Item Name     12.99" → (Item Name, 2, 12.99)
    - "Item Name        12.99" → (Item Name, 1, 12.99)
    
    Args:
        line: Single line from receipt text
        
    Returns:
        Tuple of (item_name, quantity, unit_price)
    """
    item_name, total_price = extract_price_from_line(line)
    
    if item_name is None:
        return None, None, None
    
    quantity = 1
    
    # Check for quantity patterns
    # Pattern 1: "2 x Item" or "2x Item" at start
    qty_match = re.match(r'^(\d+)\s*[xX×]\s*(.+)$', item_name)
    if qty_match:
        quantity = int(qty_match.group(1))
        item_name = qty_match.group(2).strip()
    else:
        # Pattern 2: "Item x2" or "Item x 2" at end
        qty_match = re.search(r'^(.+?)\s*[xX×]\s*(\d+)$', item_name)
        if qty_match:
            item_name = qty_match.group(1).strip()
            quantity = int(qty_match.group(2))
        else:
            # Pattern 3: "2 Item" at start (quantity followed by space)
            qty_match = re.match(r'^(\d+)\s{2,}(.+)$', item_name)
            if qty_match:
                quantity = int(qty_match.group(1))
                item_name = qty_match.group(2).strip()
    
    # Calculate unit price
    unit_price = total_price / quantity if quantity > 0 else total_price
    
    return item_name, quantity, unit_price


def extract_line_items(text: str, include_quantity: bool = False) -> List[Dict]:
    """
    Extract all line items from receipt text.
    
    Args:
        text: Full OCR text from receipt
        include_quantity: Whether to parse quantity information
        
    Returns:
        List of dictionaries containing:
        - name: Item name
        - price: Item price (total for the line)
        - quantity: (optional) Quantity if include_quantity=True
        - unit_price: (optional) Unit price if include_quantity=True
    """
    items = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        if include_quantity:
            name, quantity, unit_price = parse_quantity_line(line)
            if name is not None:
                items.append({
                    'name': name,
                    'price': unit_price * quantity,  # Total price
                    'quantity': quantity,
                    'unit_price': unit_price
                })
        else:
            name, price = extract_price_from_line(line)
            if name is not None:
                items.append({
                    'name': name,
                    'price': price
                })
    
    return items


def group_items_by_category(items: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group extracted items by detected categories.
    
    Categories are detected based on common receipt patterns:
    - FOOD: Items containing food-related keywords
    - BEVERAGE: Drinks, coffee, tea, etc.
    - OTHER: Everything else
    
    Args:
        items: List of extracted line items
        
    Returns:
        Dictionary mapping category names to list of items
    """
    food_keywords = ['burger', 'pizza', 'sandwich', 'salad', 'chicken', 'beef', 
                     'fish', 'rice', 'noodle', 'soup', 'fries', 'meal']
    beverage_keywords = ['coffee', 'tea', 'drink', 'juice', 'soda', 'water', 
                         'cola', 'sprite', 'beer', 'wine', 'milk', 'shake']
    
    categories = {
        'FOOD': [],
        'BEVERAGE': [],
        'OTHER': []
    }
    
    for item in items:
        name_lower = item['name'].lower()
        
        if any(kw in name_lower for kw in beverage_keywords):
            categories['BEVERAGE'].append(item)
        elif any(kw in name_lower for kw in food_keywords):
            categories['FOOD'].append(item)
        else:
            categories['OTHER'].append(item)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def calculate_items_subtotal(items: List[Dict]) -> float:
    """
    Calculate subtotal from extracted line items.
    
    Args:
        items: List of extracted line items
        
    Returns:
        Sum of all item prices
    """
    return sum(item.get('price', 0) for item in items)