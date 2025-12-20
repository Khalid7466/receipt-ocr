"""
Key Values Extraction for Receipt Parsing.

This module extracts structured financial data from raw OCR text:
- Transaction dates
- Total amounts

Handles common OCR noise and errors.
"""

import re
from typing import Optional, Tuple
from datetime import datetime


def fix_ocr_number_errors(text: str) -> str:
    """
    Fix common OCR errors where letters are misread as digits or vice versa.
    
    Common OCR mistakes:
    - 'O' or 'o' → '0'
    - 'l' or 'I' → '1'
    - 'S' or 's' → '5'
    - 'B' → '8'
    - 'Z' → '2'
    
    Args:
        text: Raw OCR text that should be a number
        
    Returns:
        Corrected text with letter-to-digit substitutions
    """
    corrections = {
        'O': '0', 'o': '0',
        'l': '1', 'I': '1', 'i': '1',
        'S': '5', 's': '5',
        'B': '8',
        'Z': '2', 'z': '2',
        'g': '9', 'q': '9',
        'D': '0',
    }
    
    result = text
    for letter, digit in corrections.items():
        result = result.replace(letter, digit)
    
    return result


def clean_price_string(price_str: str) -> Optional[float]:
    """
    Clean a price string and convert to float.
    
    Handles:
    - Currency symbols ($, £, €, EGP, etc.)
    - Thousand separators (commas)
    - OCR errors in digits
    - Various decimal formats
    
    Args:
        price_str: Raw price string from OCR
        
    Returns:
        Float value or None if parsing fails
    """
    if not price_str:
        return None
    
    # Remove currency symbols and common prefixes
    cleaned = price_str.strip()
    cleaned = re.sub(r'^[£$€]', '', cleaned)
    cleaned = re.sub(r'\s*(EGP|USD|EUR|GBP|LE|L\.E\.?)\s*', '', cleaned, flags=re.IGNORECASE)
    
    # Fix OCR errors in the remaining string
    cleaned = fix_ocr_number_errors(cleaned)
    
    # Remove thousand separators (commas)
    cleaned = cleaned.replace(',', '')
    
    # Handle different decimal formats
    # Some receipts use comma as decimal separator
    if cleaned.count('.') == 0 and cleaned.count(',') == 1:
        cleaned = cleaned.replace(',', '.')
    
    # Remove any remaining non-numeric characters except decimal point
    cleaned = re.sub(r'[^\d.]', '', cleaned)
    
    # Handle multiple decimal points (take only first)
    parts = cleaned.split('.')
    if len(parts) > 2:
        cleaned = parts[0] + '.' + ''.join(parts[1:])
    
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_date(text: str) -> Optional[str]:
    """
    Extract transaction date from OCR text.
    
    Supports formats:
    - YYYY-MM-DD (ISO format)
    - DD/MM/YYYY
    - MM/DD/YYYY
    - DD-MM-YYYY
    - DD.MM.YYYY
    - Written dates: "Jan 15, 2024", "15 January 2024"
    
    Args:
        text: Raw OCR text from receipt
        
    Returns:
        Date string in YYYY-MM-DD format, or None if not found
    """
    # Define date patterns (ordered by specificity)
    patterns = [
        # ISO format: 2024-01-15
        (r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b', 'YMD'),
        
        # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
        (r'\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b', 'DMY'),
        
        # DD/MM/YY or DD-MM-YY
        (r'\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2})\b', 'DMY_SHORT'),
        
        # Written months: Jan 15, 2024 or 15 Jan 2024
        (r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\b', 'D_MON_Y'),
        (r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})\b', 'MON_D_Y'),
    ]
    
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    for pattern, fmt in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                groups = match.groups()
                
                if fmt == 'YMD':
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                    
                elif fmt == 'DMY':
                    day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                    
                elif fmt == 'DMY_SHORT':
                    day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                    # Convert 2-digit year to 4-digit
                    year = 2000 + year if year < 50 else 1900 + year
                    
                elif fmt == 'D_MON_Y':
                    day = int(groups[0])
                    month = month_map[groups[1].lower()[:3]]
                    year = int(groups[2])
                    
                elif fmt == 'MON_D_Y':
                    month = month_map[groups[0].lower()[:3]]
                    day = int(groups[1])
                    year = int(groups[2])
                else:
                    continue
                
                # Validate date
                if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
                    return f"{year:04d}-{month:02d}-{day:02d}"
                    
            except (ValueError, KeyError):
                continue
    
    return None


def extract_total(text: str) -> Optional[float]:
    """
    Extract total amount from OCR text.
    
    Looks for keywords: Total, Net, Grand Total, Amount Due, etc.
    Handles OCR noise in the price digits.
    
    Args:
        text: Raw OCR text from receipt
        
    Returns:
        Total amount as float, or None if not found
    """
    # Keywords that indicate total amount (case insensitive)
    total_keywords = [
        r'grand\s*total',
        r'total\s*due',
        r'amount\s*due',
        r'balance\s*due',
        r'net\s*total',
        r'net\s*amount',
        r'total\s*amount',
        r'amount\s*payable',
        r'you\s*pay',
        r'to\s*pay',
        r'\btotal\b',
        r'\bnet\b',
        r'\bamount\b',
    ]
    
    # Build pattern to match keyword followed by price
    # Price pattern: optional currency, digits with possible decimal
    price_pattern = r'[£$€]?\s*[\d,OoIlSsBZz]+\.?\d*'
    
    candidates = []
    
    for keyword in total_keywords:
        # Match keyword followed by separator and price
        pattern = rf'{keyword}\s*[:\-]?\s*({price_pattern})'
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            price_str = match.group(1)
            amount = clean_price_string(price_str)
            if amount is not None and amount > 0:
                # Store with keyword priority (earlier in list = higher priority)
                priority = total_keywords.index(keyword)
                candidates.append((priority, amount, match.start()))
    
    if not candidates:
        # Fallback: look for the last/largest number that looks like a total
        # This helps when keywords are not recognized
        pattern = rf'({price_pattern})'
        matches = list(re.finditer(pattern, text))
        
        for match in reversed(matches):  # Start from end (totals usually at bottom)
            price_str = match.group(1)
            amount = clean_price_string(price_str)
            if amount is not None and amount > 0:
                candidates.append((999, amount, match.start()))
                break
    
    if candidates:
        # Sort by priority (lower = better), then by position (later = better for ties)
        candidates.sort(key=lambda x: (x[0], -x[2]))
        return candidates[0][1]
    
    return None


def extract_key_values(text: str) -> dict:
    """
    Extract all key financial values from receipt text.
    
    Args:
        text: Raw OCR text from receipt
        
    Returns:
        Dictionary containing:
        - date: Transaction date (YYYY-MM-DD format) or None
        - total: Total amount (float) or None
    """
    return {
        'date': extract_date(text),
        'total': extract_total(text)
    }