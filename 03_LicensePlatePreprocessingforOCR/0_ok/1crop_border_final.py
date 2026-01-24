import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import easyocr

# ================= CONFIG =================
MODEL_PATH = "./model/best.pt"
IMG_PATH = "./img/car1_full_pipeline.jpg"

SCALE = 2

# contour filter
MIN_AREA = 1200
ASPECT_MIN = 2.2
ASPECT_MAX = 8.0

# morphology kernel
K_CLOSE = (25, 5)
ITER_CLOSE = 2
K_OPEN = (3, 3)
ITER_OPEN = 1


# ================= UTILS =================
def clear_border(binary: np.ndarray):
    """Remove connected white regions touching image border (flood fill)."""
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


# ================= PREPROCESS =================
def preprocess_plate(img_bgr: np.ndarray, scale: int = 2):
    steps = {}

    # 1) grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    steps["gray"] = gray

    # 2) resize
    h, w = gray.shape
    resized = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    steps["resize"] = resized

    # 3) blur
    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    steps["blur"] = blur

    # 4) threshold (INV: chữ trắng nền đen)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5
    )
    th = clear_border(th)
    steps["thresh"] = th

    # 5) morphology (nối chữ)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, K_CLOSE)
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_close, iterations=ITER_CLOSE)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, K_OPEN)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_open, iterations=ITER_OPEN)

    steps["morph"] = morph
    return steps


# ================= CROP TEXT REGION =================
def crop_text_region(binary_mask: np.ndarray, pad=4, debug=False):
    h, w = binary_mask.shape

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = binary_mask.shape
        return binary_mask, cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR), (0, 0, w, h)

    candidates = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if hh == 0:
            continue
        aspect = ww / float(hh)

        if area > MIN_AREA and ASPECT_MIN <= aspect <= ASPECT_MAX:
            candidates.append((cnt, x, y, ww, hh, area, aspect))

    if candidates:
        best = max(candidates, key=lambda t: t[5])
        best_cnt, x, y, ww, hh, area, aspect = best
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

    dbg = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(dbg, [best_cnt], -1, (0, 255, 0), 2)
    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 2)

    info = [
        f"Pick: {reason}",
        f"Area: {area:.0f}",
        f"Aspect: {aspect:.2f}",
        f"Crop: ({x1},{y1})-({x2},{y2})"
    ]
    y0 = 20
    for s in info:
        cv2.putText(dbg, s, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y0 += 18

    return crop, dbg, (x1, y1, x2, y2)


# ================= MAIN =================
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

    for idx, (plate_img, conf, bbox) in enumerate(plates, 1):
        print(f"\n[INFO] Processing plate {idx} | conf={conf:.3f}")

        # Vẽ bounding box trên ảnh gốc
        x1, y1, x2, y2 = bbox
        img_with_bbox = img_full.copy()
        cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img_with_bbox, f"Plate {idx} (conf: {conf:.3f})", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        steps = preprocess_plate(plate_img, SCALE)
        crop_mask, dbg, crop_coords = crop_text_region(steps["morph"], pad=4, debug=True)
        
        # Bước 9: Crop trên ảnh blur dựa vào vùng đã detect và loại bỏ vùng trắng
        x1, y1, x2, y2 = crop_coords
        crop_blur_raw = steps["blur"][y1:y2, x1:x2]
        
        # Tìm vùng text từ mask để crop chính xác hơn
        text_pixels = np.where(crop_mask > 128)
        if len(text_pixels[0]) > 0:
            text_y1 = max(0, np.min(text_pixels[0]) - 2)
            text_y2 = min(crop_mask.shape[0], np.max(text_pixels[0]) + 3)
            text_x1 = max(0, np.min(text_pixels[1]) - 2)
            text_x2 = min(crop_mask.shape[1], np.max(text_pixels[1]) + 3)
            
            # Crop lại trên blur để chỉ giữ vùng text
            crop_blur = crop_blur_raw[text_y1:text_y2, text_x1:text_x2]
            
            # Tạo mask từ crop_mask để set background = trắng (255)
            mask_small = crop_mask[text_y1:text_y2, text_x1:text_x2]
            # Background (vùng không phải text) = trắng
            crop_blur_clean = crop_blur.copy()
            crop_blur_clean[mask_small < 128] = 255  # Set background thành trắng
            crop_blur = crop_blur_clean
        else:
            crop_blur = crop_blur_raw

        # OCR-ready: chữ đen nền trắng (từ binary)
        crop_ocr = 255 - crop_mask

        # ===== BƯỚC 10: OCR =====
        print("[INFO] Running OCR on binary image...")
        ocr_results_binary = ocr_reader.readtext(crop_ocr, detail=1)
        text_binary = ""
        conf_binary = 0.0
        if ocr_results_binary:
            texts = [text for (_, text, _) in ocr_results_binary]
            confs = [conf for (_, _, conf) in ocr_results_binary]
            text_binary = "".join(texts)
            conf_binary = float(np.mean(confs))

        print("[INFO] Running OCR on grayscale image...")
        ocr_results_blur = ocr_reader.readtext(crop_blur, detail=1)
        text_blur = ""
        conf_blur = 0.0
        if ocr_results_blur:
            texts = [text for (_, text, _) in ocr_results_blur]
            confs = [conf for (_, _, conf) in ocr_results_blur]
            text_blur = "".join(texts)
            conf_blur = float(np.mean(confs))

        # In kết quả OCR
        print("\n" + "=" * 60)
        print(f"[PLATE {idx}] OCR Results:")
        print(f"  Detection conf: {conf:.3f}")
        print(f"  OCR (Binary)  : '{text_binary}' (conf: {conf_binary:.3f})")
        print(f"  OCR (Blur)    : '{text_blur}' (conf: {conf_blur:.3f})")
        print("=" * 60)

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f"Full Pipeline - Plate {idx} (Conf: {conf:.3f})", fontsize=16, fontweight="bold")

        # Hàng 1
        axes[0, 0].imshow(cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("0. Full Image + Detection", fontweight="bold", color="blue")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("1. Detected Plate (Cropped)")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(steps["gray"], cmap="gray")
        axes[0, 2].set_title("2. Grayscale")
        axes[0, 2].axis("off")

        axes[0, 3].imshow(steps["resize"], cmap="gray")
        axes[0, 3].set_title("3. Resize")
        axes[0, 3].axis("off")

        axes[0, 4].imshow(steps["blur"], cmap="gray")
        axes[0, 4].set_title("4. Blur")
        axes[0, 4].axis("off")

        # Hàng 2
        axes[1, 0].imshow(steps["thresh"], cmap="gray")
        axes[1, 0].set_title("5. Threshold + Clear Border")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(steps["morph"], cmap="gray")
        axes[1, 1].set_title("6. Morph")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title("7. Debug Contour")
        axes[1, 2].axis("off")

        axes[1, 3].imshow(crop_ocr, cmap="gray")
        title_binary = f"8. Binary OCR\n'{text_binary}' ({conf_binary:.2f})"
        axes[1, 3].set_title(title_binary, color="green", fontsize=9)
        axes[1, 3].axis("off")

        axes[1, 4].imshow(crop_blur, cmap="gray")
        title_blur = f"9. Blur OCR\n'{text_blur}' ({conf_blur:.2f})"
        axes[1, 4].set_title(title_blur, color="red", fontweight="bold", fontsize=9)
        axes[1, 4].axis("off")

        plt.tight_layout()
        plt.show()

    print("\n[INFO] Done.")
