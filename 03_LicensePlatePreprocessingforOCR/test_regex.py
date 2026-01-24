"""
Script để test regex pattern cho biển số Việt Nam
"""
import re

# Các pattern biển số Việt Nam
VIETNAMESE_PLATE_PATTERNS = [
    r'^[0-9A-Z]{2}-[0-9A-Z]{4}\.[0-9A-Z]{2}$',      # XX-XXXX.XX (cho phép chữ ở cuối)
    r'^[0-9A-Z]{2}-[0-9A-Z]{5}\.[0-9A-Z]{2}$',     # XX-XXXXX.XX
    r'^[0-9A-Z]{2}-[0-9A-Z]{4}\.[0-9]{2}$',        # XX-XXXX.XX (chỉ số ở cuối)
    r'^[0-9A-Z]{2}-[0-9A-Z]{5}\.[0-9]{2}$',        # XX-XXXXX.XX (chỉ số ở cuối)
]

def validate_vietnamese_plate(text: str) -> bool:
    """Kiểm tra xem text có match với format biển số Việt Nam không."""
    text_clean = text.replace(' ', '').upper()
    
    print(f"\nTesting: '{text}'")
    print(f"Cleaned: '{text_clean}'")
    print(f"Length: {len(text_clean)}")
    
    for i, pattern in enumerate(VIETNAMESE_PLATE_PATTERNS, 1):
        match = re.match(pattern, text_clean)
        print(f"  Pattern {i}: {pattern}")
        print(f"    Match: {match is not None}")
        if match:
            print(f"    >>> MATCHED!")
            return True
    
    print(f"  >>> NO MATCH")
    return False

# Test với text đã format đúng (từ pipeline)
test_cases = [
    "61-A229.59",      # Đúng format (8 ký tự) - từ 61A22959
    "61A22959",        # Chưa format (8 ký tự) - sẽ FALSE vì chưa có dấu
    "30-1234.56",      # Đúng format (8 ký tự)
    "30A-12345.67",    # Đúng format (9 ký tự) - từ 30A1234567
    "30A-1234.56",     # Đúng format (8 ký tự với chữ)
]

print("\n" + "="*60)
print("TEST 1: Text da format dung (co dau gach ngang va dau cham)")
print("="*60)

print("=" * 60)
print("TESTING VIETNAMESE PLATE REGEX PATTERNS")
print("=" * 60)

for test in test_cases:
    result = validate_vietnamese_plate(test)
    print(f"Result: {'>>> TRUE' if result else '>>> FALSE'}")
    print("-" * 60)

# Test với text chưa format (không có dấu)
print("\n" + "="*60)
print("TEST 2: Text chua format (khong co dau) - can format truoc")
print("="*60)

def format_plate(text: str) -> str:
    """Format text thành đúng pattern"""
    cleaned = re.sub(r'[^0-9A-Z]', '', text.upper())
    if len(cleaned) == 8:
        return f"{cleaned[:2]}-{cleaned[2:6]}.{cleaned[6:]}"
    elif len(cleaned) == 9:
        return f"{cleaned[:2]}-{cleaned[2:7]}.{cleaned[7:]}"
    return cleaned

raw_cases = [
    "61A22959",        # 8 ký tự -> 61-A229.59
    "30A123456",       # 9 ký tự -> 30-A1234.56
    "61A.229.59",      # Có dấu chấm -> clean thành 61A22959 -> 61-A229.59
]

for raw in raw_cases:
    formatted = format_plate(raw)
    print(f"\nRaw: '{raw}'")
    print(f"Formatted: '{formatted}'")
    result = validate_vietnamese_plate(formatted)
    print(f"Result: {'>>> TRUE' if result else '>>> FALSE'}")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)

