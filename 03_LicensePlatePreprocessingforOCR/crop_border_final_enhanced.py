"""
Enhanced ALPR Pipeline for Vietnamese License Plates
=====================================================

Improvements:
1. Deskew and perspective correction
2. Morphological operation tuning
3. Adaptive thresholding (multi-parameter)
4. Text-band extraction optimization
5. Multi-pass OCR waterfall strategy
6. OCR error correction using Vietnamese plate syntax
7. Regex-based validation
8. Optional super-resolution preprocessing
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import easyocr
import re
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

# ================= CONFIG =================
MODEL_PATH = "./model/best.pt"
IMG_PATH = "./img/full_pipeline8.jpg"
SCALE = 2

# Contour filter (for finding "text band")
MIN_AREA = 1200
ASPECT_MIN = 2.0
ASPECT_MAX = 12.0

# Morphology (adaptive based on image size)
K_CLOSE_BASE = (15, 3)
ITER_CLOSE = 1
K_OPEN = (3, 3)
ITER_OPEN = 1

# Vietnamese plate patterns
VIETNAMESE_PLATE_PATTERNS = [
    # Standard: XX-XXXX.XX or XX-XXXXX.XX
    re.compile(r'^[0-9]{2}-[0-9]{4,5}\.[0-9]{2}$'),
    # With letters: XX-XXXX.XX or XX-XXXXX.XX
    re.compile(r'^[0-9A-Z]{2}-[0-9A-Z]{4,5}\.[0-9]{2}$'),
    # Alternative: XX-XXXX.XX (no dot)
    re.compile(r'^[0-9]{2}-[0-9]{4,5}[0-9]{2}$'),
    # Motorcycle: XX-XXXX.XX
    re.compile(r'^[0-9]{2}-[0-9]{4}\.[0-9]{2}$'),
]

# Common OCR character confusion map (for Vietnamese plates)
CHAR_CONFUSION = {
    '0': ['O', 'D', 'Q'],
    '1': ['I', 'L', 'T'],
    '2': ['Z'],
    '5': ['S'],
    '6': ['G'],
    '8': ['B'],
    'B': ['8', 'R'],
    'D': ['0', 'O'],
    'I': ['1', 'L'],
    'O': ['0', 'D', 'Q'],
    'Q': ['O', '0'],
    'S': ['5'],
    'Z': ['2'],
}


@dataclass
class OCRResult:
    text: str
    confidence: float
    method: str
    corrected: bool = False


# ================= DESKEW & PERSPECTIVE CORRECTION =================
def detect_plate_corners(binary: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect plate corners using contour approximation.
    Returns 4 corner points for perspective correction.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Find largest contour
    largest = max(contours, key=cv2.contourArea)
    
    # Approximate polygon
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    
    if len(approx) == 4:
        # Reorder corners: top-left, top-right, bottom-right, bottom-left
        pts = approx.reshape(4, 2)
        
        # Find corners by position
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        top_left = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]
        top_right = pts[np.argmin(diff)]
        bottom_left = pts[np.argmax(diff)]
        
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    
    return None


def correct_perspective(img: np.ndarray, corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply perspective correction and deskew.
    Returns corrected image and transformation matrix.
    """
    h, w = img.shape[:2]
    
    # Calculate bounding box
    x_coords = corners[:, 0]
    y_coords = corners[:, 1]
    
    x_min, x_max = int(x_coords.min()), int(x_coords.max())
    y_min, y_max = int(y_coords.min()), int(y_coords.max())
    
    # Destination points (rectangular)
    width = max(
        np.linalg.norm(corners[0] - corners[1]),
        np.linalg.norm(corners[2] - corners[3])
    )
    height = max(
        np.linalg.norm(corners[1] - corners[2]),
        np.linalg.norm(corners[3] - corners[0])
    )
    
    dst = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    # Get perspective transform
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (int(width), int(height)))
    
    return warped, M


def deskew_image(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Deskew image using Hough line detection.
    Returns deskewed image and skew angle.
    """
    # Detect edges
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    
    # Detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is None or len(lines) == 0:
        return img, 0.0
    
    # Calculate angles
    angles = []
    for rho, theta in lines[:20]:  # Use top 20 lines
        angle = np.degrees(theta) - 90
        if -45 < angle < 45:
            angles.append(angle)
    
    if not angles:
        return img, 0.0
    
    # Use median angle (more robust than mean)
    skew_angle = np.median(angles)
    
    # Rotate to correct skew
    h, w = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    deskewed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    return deskewed, skew_angle


# ================= ENHANCED PREPROCESSING =================
def adaptive_morphology_kernel(img_shape: Tuple[int, int], base_kernel: Tuple[int, int]) -> Tuple[int, int]:
    """
    Adapt morphology kernel size based on image dimensions.
    """
    h, w = img_shape
    scale_factor = min(w / 200, h / 50)  # Normalize to reference size
    
    kw = max(3, int(base_kernel[0] * scale_factor))
    kh = max(3, int(base_kernel[1] * scale_factor))
    
    # Ensure odd numbers
    kw = kw + 1 if kw % 2 == 0 else kw
    kh = kh + 1 if kh % 2 == 0 else kh
    
    return (kw, kh)


def multi_adaptive_threshold(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate multiple adaptive threshold variants.
    Different parameters work better for different plate conditions.
    """
    results = {}
    
    # Variant 1: Standard (current)
    th1 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 5
    )
    results['standard'] = th1
    
    # Variant 2: Larger block size (better for low contrast)
    th2 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 7
    )
    results['large_block'] = th2
    
    # Variant 3: Mean-based (better for uneven lighting)
    th3 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 5
    )
    results['mean'] = th3
    
    # Variant 4: Smaller block (better for high contrast)
    th4 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 3
    )
    results['small_block'] = th4
    
    return results


def clear_border(binary: np.ndarray) -> np.ndarray:
    """Remove connected white regions touching image border."""
    h, w = binary.shape
    cleared = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    for x in range(w):
        if cleared[0, x] == 255:
            cv2.floodFill(cleared, mask, (x, 0), 0)
        if cleared[h - 1, x] == 255:
            cv2.floodFill(cleared, mask, (x, h - 1), 0)
    
    for y in range(h):
        if cleared[y, 0] == 255:
            cv2.floodFill(cleared, mask, (0, y), 0)
        if cleared[y, w - 1] == 255:
            cv2.floodFill(cleared, mask, (w - 1, y), 0)
    
    return cleared


def remove_long_horizontal_lines(binary_inv: np.ndarray, min_len_ratio=0.55) -> np.ndarray:
    """Remove long horizontal components (plate borders)."""
    h, w = binary_inv.shape
    out = binary_inv.copy()
    
    contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        if hh == 0:
            continue
        
        if ww >= int(w * min_len_ratio) and hh <= int(h * 0.20):
            cv2.drawContours(out, [cnt], -1, 0, thickness=-1)
    
    return out


def preprocess_plate_enhanced(img_bgr: np.ndarray, scale: int = 2, 
                              apply_deskew: bool = True,
                              apply_superres: bool = False) -> Dict[str, np.ndarray]:
    """
    Enhanced preprocessing with deskew, super-resolution, and multi-threshold.
    """
    steps = {}
    
    # 1) Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    steps["gray"] = gray
    
    # 2) Optional super-resolution
    if apply_superres:
        # Use EDSR or simple upscaling
        h, w = gray.shape
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        steps["superres"] = gray
    else:
        # Standard resize
        h, w = gray.shape
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    
    steps["resize"] = gray
    
    # 3) Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    steps["blur"] = blur
    
    # 4) Deskew (if enabled)
    if apply_deskew:
        deskewed, skew_angle = deskew_image(blur)
        steps["deskewed"] = deskewed
        steps["skew_angle"] = skew_angle
        blur = deskewed
    
    # 5) Multi-adaptive threshold
    thresh_variants = multi_adaptive_threshold(blur)
    for name, th in thresh_variants.items():
        th_cleared = clear_border(th)
        steps[f"thresh_{name}"] = th_cleared
    
    # Use standard threshold for main pipeline
    th = thresh_variants['standard']
    th = clear_border(th)
    steps["thresh"] = th
    
    # 6) Remove borders
    th2 = remove_long_horizontal_lines(th, min_len_ratio=0.55)
    steps["thresh_noborder"] = th2
    
    # 7) Adaptive morphology
    kernel_close = adaptive_morphology_kernel(th2.shape, K_CLOSE_BASE)
    kernel_close_elem = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_close)
    morph = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel_close_elem, iterations=ITER_CLOSE)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, K_OPEN)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_open, iterations=ITER_OPEN)
    
    steps["morph"] = morph
    
    return steps


# ================= TEXT BAND EXTRACTION =================
def crop_text_band_enhanced(binary_mask: np.ndarray, pad=4) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Enhanced text band extraction with better contour filtering.
    """
    h, w = binary_mask.shape
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dbg = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    
    if not contours:
        return binary_mask, dbg, (0, 0, w, h)
    
    candidates = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if hh == 0:
            continue
        aspect = ww / float(hh)
        
        # Enhanced filtering: consider both area and aspect ratio
        if area > MIN_AREA and ASPECT_MIN <= aspect <= ASPECT_MAX:
            # Score: prefer larger area and better aspect ratio
            score = area * (1.0 / (1.0 + abs(aspect - 5.0)))  # Prefer aspect ~5
            candidates.append((cnt, x, y, ww, hh, area, aspect, score))
    
    if candidates:
        best = max(candidates, key=lambda t: t[6])  # Use score
        best_cnt, x, y, ww, hh, area, aspect, _ = best
        reason = "shape-filter"
    else:
        best_cnt = max(contours, key=cv2.contourArea)
        x, y, ww, hh = cv2.boundingRect(best_cnt)
        area = cv2.contourArea(best_cnt)
        aspect = ww / float(hh) if hh != 0 else 0
        reason = "fallback"
    
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + ww + pad)
    y2 = min(h, y + hh + pad)
    
    crop = binary_mask[y1:y2, x1:x2]
    
    cv2.drawContours(dbg, [best_cnt], -1, (0, 255, 0), 2)
    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    info = [
        f"Pick: {reason}",
        f"Area: {area:.0f}",
        f"Aspect: {aspect:.2f}",
        f"Crop: ({x1},{y1})-({x2},{y2})",
    ]
    y0 = 20
    for s in info:
        cv2.putText(dbg, s, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y0 += 18
    
    return crop, dbg, (x1, y1, x2, y2)


def trim_and_pad_by_mask(gray_img: np.ndarray, binary_mask: np.ndarray,
                         pad=10, min_h=64, denoise=True, debug=False):
    """Trim and pad with denoising."""
    ys, xs = np.where(binary_mask > 128)
    
    if len(xs) == 0 or len(ys) == 0:
        out = cv2.copyMakeBorder(gray_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        return out if not debug else (out, None, (0, 0, gray_img.shape[1], gray_img.shape[0]))
    
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    
    y1 = max(0, y1 - 1)
    y2 = min(binary_mask.shape[0] - 1, y2 + 1)
    x1 = max(0, x1 - 1)
    x2 = min(binary_mask.shape[1] - 1, x2 + 1)
    
    crop_gray = gray_img[y1:y2 + 1, x1:x2 + 1]
    crop_mask = binary_mask[y1:y2 + 1, x1:x2 + 1]
    
    clean = crop_gray.copy()
    clean[crop_mask < 128] = 255
    
    if denoise:
        clean = cv2.fastNlMeansDenoising(clean, None, 10, 7, 21)
    
    clean = cv2.copyMakeBorder(clean, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    
    h, w = clean.shape
    if h < min_h:
        extra = min_h - h
        top = extra // 2
        bot = extra - top
        clean = cv2.copyMakeBorder(clean, top, bot, 0, 0, cv2.BORDER_CONSTANT, value=255)
    
    if debug:
        dbg = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(dbg, f"Trim: ({x1},{y1})-({x2},{y2})", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return clean, dbg, (x1, y1, x2, y2)
    
    return clean


# ================= MULTI-PASS OCR WATERFALL =================
def run_ocr_waterfall(reader: easyocr.Reader, images: List[Tuple[np.ndarray, str]]) -> List[OCRResult]:
    """
    Run OCR on multiple image variants and return results sorted by confidence.
    Waterfall strategy: try best quality first, fallback to alternatives.
    """
    results = []
    
    for img, method in images:
        try:
            ocr_res = reader.readtext(img, detail=1)
            if ocr_res:
                text = "".join([t for (_, t, _) in ocr_res])
                conf = float(np.mean([c for (_, _, c) in ocr_res]))
                results.append(OCRResult(text=text, confidence=conf, method=method))
        except Exception as e:
            print(f"[WARN] OCR failed for {method}: {e}")
            continue
    
    # Sort by confidence (descending)
    results.sort(key=lambda x: x.confidence, reverse=True)
    return results


# ================= VIETNAMESE PLATE VALIDATION & CORRECTION =================
def validate_vietnamese_plate(text: str) -> bool:
    """Check if text matches Vietnamese plate patterns."""
    # KHÔNG xóa dấu gạch ngang và dấu chấm - giữ nguyên để check pattern
    text_clean = text.replace(' ', '').upper()
    
    for pattern in VIETNAMESE_PLATE_PATTERNS:
        if pattern.match(text_clean):
            return True
    
    return False


def correct_ocr_errors(text: str, confidence: float) -> Tuple[str, bool]:
    """
    Correct common OCR errors based on Vietnamese plate syntax.
    Returns (corrected_text, was_corrected).
    """
    if confidence > 0.9:
        return text, False  # High confidence, likely correct
    
    original = text
    corrected = text
    
    # Remove spaces and normalize
    corrected = re.sub(r'\s+', '', corrected)
    
    # Common patterns to fix
    # Pattern: XX-XXXX.XX or XX-XXXXX.XX
    # Fix missing dashes/dots
    if len(corrected) >= 8:
        # Try to insert dash and dot
        if '-' not in corrected and len(corrected) >= 8:
            # Format: XX-XXXX.XX
            if len(corrected) == 8:
                corrected = f"{corrected[:2]}-{corrected[2:6]}.{corrected[6:]}"
            elif len(corrected) == 9:
                corrected = f"{corrected[:2]}-{corrected[2:7]}.{corrected[7:]}"
    
    # Character confusion correction
    for char, alternatives in CHAR_CONFUSION.items():
        for alt in alternatives:
            # Replace if context suggests it should be the correct char
            # (simple heuristic: prefer digits in numeric positions)
            if alt in corrected:
                # Check position: digits in certain positions
                idx = corrected.find(alt)
                if idx < 2 or (idx >= 3 and idx < 7):  # First 2 or middle positions
                    if char.isdigit():
                        corrected = corrected.replace(alt, char, 1)
                        break
    
    was_corrected = (corrected != original)
    return corrected, was_corrected


def apply_vietnamese_plate_rules(text: str) -> str:
    """
    Apply Vietnamese plate formatting rules.
    Standard format: XX-XXXX.XX or XX-XXXXX.XX
    """
    # Xóa HẾT tất cả ký tự đặc biệt (bao gồm cả dấu chấm và gạch ngang)
    # Chỉ giữ lại alphanumeric
    cleaned = re.sub(r'[^0-9A-Z]', '', text.upper())
    
    # Format lại theo đúng pattern
    if len(cleaned) == 8:
        # Format: XX-XXXX.XX
        # Ví dụ: 61A22959 -> 61-A229.59
        return f"{cleaned[:2]}-{cleaned[2:6]}.{cleaned[6:]}"
    elif len(cleaned) == 9:
        # Format: XX-XXXXX.XX
        return f"{cleaned[:2]}-{cleaned[2:7]}.{cleaned[7:]}"
    elif len(cleaned) == 7:
        # Có thể thiếu 1 số
        return f"{cleaned[:2]}-{cleaned[2:6]}.{cleaned[6:]}"
    else:
        # Giữ nguyên nếu không đúng độ dài
        return cleaned


# ================= MAIN PIPELINE =================
if __name__ == "__main__":
    print("[INFO] Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    
    print("[INFO] Loading image...")
    img_full = cv2.imread(IMG_PATH)
    if img_full is None:
        raise FileNotFoundError("Image not found!")
    
    print("[INFO] Running YOLO detection...")
    results = model(img_full)
    
    plates = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            plate_img = img_full[y1:y2, x1:x2]
            if plate_img.size > 0:
                plates.append((plate_img, conf, (x1, y1, x2, y2)))
    
    print(f"[INFO] Found {len(plates)} plate(s)")
    if not plates:
        raise RuntimeError("No plates detected!")
    
    print("[INFO] Initializing OCR engine...")
    ocr_reader = easyocr.Reader(['en'])
    
    for idx, (plate_img, det_conf, bbox) in enumerate(plates, 1):
        print(f"\n[INFO] Processing plate {idx} | det_conf={det_conf:.3f}")
        
        # Enhanced preprocessing
        steps = preprocess_plate_enhanced(
            plate_img, SCALE,
            apply_deskew=True,
            apply_superres=False  # Enable if needed
        )
        
        # Find text band
        crop_mask, dbg_contour, crop_coords = crop_text_band_enhanced(steps["morph"], pad=4)
        
        # Crop gray (blur) by band coords
        cx1, cy1, cx2, cy2 = crop_coords
        blur_band = steps["blur"][cy1:cy2, cx1:cx2]
        
        # Clean + trim + pad
        crop_gray_clean, dbg_trim, trim_coords = trim_and_pad_by_mask(
            blur_band, crop_mask,
            pad=12, min_h=64, denoise=True, debug=True
        )
        
        # Prepare OCR inputs (waterfall strategy)
        ocr_inputs = [
            (crop_gray_clean, "clean_gray"),  # Best quality
            (255 - crop_mask, "binary_inv"),  # Binary inverse
        ]
        
        # Add threshold variants
        for name in ['large_block', 'mean', 'small_block']:
            if f"thresh_{name}" in steps:
                th_variant = steps[f"thresh_{name}"]
                if th_variant.shape == steps["morph"].shape:
                    th_crop = th_variant[cy1:cy2, cx1:cx2]
                    th_trim = trim_and_pad_by_mask(blur_band, th_crop, pad=12, min_h=64, denoise=True)
                    ocr_inputs.append((255 - th_trim, f"thresh_{name}"))
        
        # Run OCR waterfall
        ocr_results = run_ocr_waterfall(ocr_reader, ocr_inputs)
        
        # Process results
        best_result = ocr_results[0] if ocr_results else OCRResult("", 0.0, "none")
        
        # Apply Vietnamese plate correction
        corrected_text, was_corrected = correct_ocr_errors(best_result.text, best_result.confidence)
        formatted_text = apply_vietnamese_plate_rules(corrected_text)
        is_valid = validate_vietnamese_plate(formatted_text)
        
        # Final plate: bỏ hết dấu gạch ngang và dấu chấm (chỉ giữ alphanumeric)
        final_plate = re.sub(r'[^0-9A-Z]', '', formatted_text.upper())
        
        print("\n" + "=" * 60)
        print(f"[PLATE {idx}]")
        print(f"  Detection conf  : {det_conf:.3f}")
        print(f"  OCR (Raw)       : '{best_result.text}' ({best_result.confidence:.3f}) [{best_result.method}]")
        print(f"  OCR (Corrected) : '{corrected_text}' (corrected={was_corrected})")
        print(f"  OCR (Formatted) : '{formatted_text}'")
        print("")
        print(f"  {'='*20} RESULT {'='*20}")
        print(f"  Final Plate     : '{final_plate}'")
        print(f"  Valid Pattern   : {'✅ TRUE' if is_valid else '❌ FALSE'}")
        print(f"  {'='*48}")
        if len(ocr_results) > 1:
            print(f"  Alternatives:")
            for r in ocr_results[1:3]:  # Show top 3
                print(f"    - '{r.text}' ({r.confidence:.3f}) [{r.method}]")
        print("=" * 60)
        
        # Visualization
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        fig.suptitle(f"Enhanced Pipeline - Plate {idx} (Conf: {det_conf:.3f})",
                     fontsize=16, fontweight="bold")
        
        axes[0, 0].imshow(cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("0. Full Image", fontweight="bold", color="blue")
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("1. Detected Plate")
        axes[0, 1].axis("off")
        
        axes[0, 2].imshow(steps["gray"], cmap="gray")
        axes[0, 2].set_title("2. Grayscale")
        axes[0, 2].axis("off")
        
        axes[0, 3].imshow(steps["resize"], cmap="gray")
        axes[0, 3].set_title("3. Resize")
        axes[0, 3].axis("off")
        
        if "deskewed" in steps:
            axes[0, 4].imshow(steps["deskewed"], cmap="gray")
            axes[0, 4].set_title(f"4. Deskewed (angle: {steps['skew_angle']:.2f}°)")
        else:
            axes[0, 4].imshow(steps["blur"], cmap="gray")
            axes[0, 4].set_title("4. Blur")
        axes[0, 4].axis("off")
        
        axes[1, 0].imshow(steps["thresh"], cmap="gray")
        axes[1, 0].set_title("5. Threshold (Standard)")
        axes[1, 0].axis("off")
        
        axes[1, 1].imshow(steps["thresh_noborder"], cmap="gray")
        axes[1, 1].set_title("6. Remove Borders")
        axes[1, 1].axis("off")
        
        axes[1, 2].imshow(steps["morph"], cmap="gray")
        axes[1, 2].set_title("7. Adaptive Morph")
        axes[1, 2].axis("off")
        
        axes[1, 3].imshow(cv2.cvtColor(dbg_contour, cv2.COLOR_BGR2RGB))
        axes[1, 3].set_title("8. Text Band Detection")
        axes[1, 3].axis("off")
        
        axes[1, 4].imshow(crop_gray_clean, cmap="gray")
        title_final = f"9. Final OCR Input\n'{final_plate}'\n(Valid: {is_valid})"
        color = "green" if is_valid else "red"
        axes[1, 4].set_title(title_final, color=color, fontweight="bold", fontsize=9)
        axes[1, 4].axis("off")
        
        # Row 3: Show threshold variants
        if "thresh_large_block" in steps:
            axes[2, 0].imshow(steps["thresh_large_block"], cmap="gray")
            axes[2, 0].set_title("10. Thresh (Large Block)")
            axes[2, 0].axis("off")
        
        if "thresh_mean" in steps:
            axes[2, 1].imshow(steps["thresh_mean"], cmap="gray")
            axes[2, 1].set_title("11. Thresh (Mean)")
            axes[2, 1].axis("off")
        
        axes[2, 2].imshow(cv2.cvtColor(dbg_trim, cv2.COLOR_BGR2RGB))
        axes[2, 2].set_title("12. Trim Debug")
        axes[2, 2].axis("off")
        
        axes[2, 3].axis("off")
        axes[2, 4].axis("off")
        
        plt.tight_layout()
        plt.show()
    
    print("\n[INFO] Done ✅")

