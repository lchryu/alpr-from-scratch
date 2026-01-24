import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import easyocr

# ================= CONFIG =================
MODEL_PATH = "./model/best.pt"
# IMG_PATH = "./img/car1_full_pipeline.jpg"

IMG_PATH = "./img/full_pipeline8.jpg"
SCALE = 2

# contour filter (for finding "text band")
MIN_AREA = 1200
ASPECT_MIN = 2.0
ASPECT_MAX = 12.0

# MORPH (LIGHTER!)
K_CLOSE = (15, 3)     # connect character strokes horizontally (not whole plate border)
ITER_CLOSE = 1
K_OPEN = (3, 3)
ITER_OPEN = 1


# ================= UTILS =================
def clear_border(binary: np.ndarray) -> np.ndarray:
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


def remove_long_horizontal_lines(binary_inv: np.ndarray, min_len_ratio=0.60) -> np.ndarray:
    """
    Remove long horizontal components (often plate borders).
    binary_inv: white(text/edges) on black.
    """
    h, w = binary_inv.shape
    out = binary_inv.copy()

    contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if hh == 0:
            continue

        # long & thin horizontal components
        if ww >= int(w * min_len_ratio) and hh <= int(h * 0.20):
            cv2.drawContours(out, [cnt], -1, 0, thickness=-1)

    return out


def preprocess_plate(img_bgr: np.ndarray, scale: int = 2):
    steps = {}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    steps["gray"] = gray

    h, w = gray.shape
    resized = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    steps["resize"] = resized

    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    steps["blur"] = blur

    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5
    )
    th = clear_border(th)
    steps["thresh"] = th

    # remove long borders BEFORE morph (big impact)
    th2 = remove_long_horizontal_lines(th, min_len_ratio=0.55)
    steps["thresh_noborder"] = th2

    # light morph
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, K_CLOSE)
    morph = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel_close, iterations=ITER_CLOSE)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, K_OPEN)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_open, iterations=ITER_OPEN)

    steps["morph"] = morph
    return steps


def crop_text_band(binary_mask: np.ndarray, pad=4):
    """
    Find best contour region where the text band lies.
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

        if area > MIN_AREA and ASPECT_MIN <= aspect <= ASPECT_MAX:
            candidates.append((cnt, x, y, ww, hh, area, aspect))

    if candidates:
        best_cnt, x, y, ww, hh, area, aspect = max(candidates, key=lambda t: t[5])
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


def trim_and_pad_by_mask(gray_img: np.ndarray,
                         binary_mask: np.ndarray,
                         pad=10,
                         min_h=64,
                         denoise=True,
                         debug=False):
    """
    Stable trim based on non-zero pixels in binary_mask.
    - Crop tight bbox around mask pixels
    - Background -> white
    - White padding
    - Ensure min height
    """
    ys, xs = np.where(binary_mask > 128)

    if len(xs) == 0 or len(ys) == 0:
        out = cv2.copyMakeBorder(gray_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        return out if not debug else (out, None, (0, 0, gray_img.shape[1], gray_img.shape[0]))

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    # safety margin
    y1 = max(0, y1 - 1)
    y2 = min(binary_mask.shape[0] - 1, y2 + 1)
    x1 = max(0, x1 - 1)
    x2 = min(binary_mask.shape[1] - 1, x2 + 1)

    crop_gray = gray_img[y1:y2 + 1, x1:x2 + 1]
    crop_mask = binary_mask[y1:y2 + 1, x1:x2 + 1]

    # background -> white
    clean = crop_gray.copy()
    clean[crop_mask < 128] = 255

    # optional denoise (helps EasyOCR)
    if denoise:
        clean = cv2.fastNlMeansDenoising(clean, None, 10, 7, 21)

    # white padding
    clean = cv2.copyMakeBorder(clean, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

    # min height
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


def run_easyocr(reader, img_gray):
    res = reader.readtext(img_gray, detail=1)
    if not res:
        return "", 0.0
    text = "".join([t for (_, t, _) in res])
    conf = float(np.mean([c for (_, _, c) in res]))
    return text, conf


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

    for idx, (plate_img, det_conf, bbox) in enumerate(plates, 1):
        print(f"\n[INFO] Processing plate {idx} | det_conf={det_conf:.3f}")

        # draw bbox
        x1, y1, x2, y2 = bbox
        img_with_bbox = img_full.copy()
        cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img_with_bbox, f"Plate {idx} (conf: {det_conf:.3f})",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        steps = preprocess_plate(plate_img, SCALE)

        # find text band
        crop_mask, dbg_contour, crop_coords = crop_text_band(steps["morph"], pad=4)

        # crop gray (blur) by band coords
        cx1, cy1, cx2, cy2 = crop_coords
        blur_band = steps["blur"][cy1:cy2, cx1:cx2]

        # clean + trim + pad
        crop_gray_clean, dbg_trim, trim_coords = trim_and_pad_by_mask(
            blur_band, crop_mask,
            pad=12, min_h=64, denoise=True, debug=True
        )

        # OCR on clean gray
        text_gray, conf_gray = run_easyocr(ocr_reader, crop_gray_clean)

        # OCR on binary (optional): make "black text on white"
        crop_ocr = 255 - crop_mask
        text_bin, conf_bin = run_easyocr(ocr_reader, crop_ocr)

        print("\n" + "=" * 60)
        print(f"[PLATE {idx}]")
        print(f"  Detection conf : {det_conf:.3f}")
        print(f"  OCR (Binary)   : '{text_bin}' ({conf_bin:.3f})")
        print(f"  OCR (CleanGray): '{text_gray}' ({conf_gray:.3f})")
        print("=" * 60)

        # ================= VISUALIZE =================
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        fig.suptitle(f"Full Pipeline - Plate {idx} (Conf: {det_conf:.3f})",
                     fontsize=16, fontweight="bold")

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

        axes[1, 0].imshow(steps["thresh"], cmap="gray")
        axes[1, 0].set_title("5. Threshold (INV)")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(steps["thresh_noborder"], cmap="gray")
        axes[1, 1].set_title("6. Thresh (Remove Borders)")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(steps["morph"], cmap="gray")
        axes[1, 2].set_title("7. Light Morph")
        axes[1, 2].axis("off")

        axes[1, 3].imshow(cv2.cvtColor(dbg_contour, cv2.COLOR_BGR2RGB))
        axes[1, 3].set_title("8. Debug Contour")
        axes[1, 3].axis("off")

        axes[1, 4].imshow(crop_ocr, cmap="gray")
        axes[1, 4].set_title(f"9. Binary OCR\n'{text_bin}' ({conf_bin:.2f})",
                             color="green", fontsize=9)
        axes[1, 4].axis("off")

        axes[2, 0].imshow(cv2.cvtColor(dbg_trim, cv2.COLOR_BGR2RGB))
        axes[2, 0].set_title("10. Trim Debug (BBox from mask)")
        axes[2, 0].axis("off")

        axes[2, 1].imshow(blur_band, cmap="gray")
        axes[2, 1].set_title("11. Blur Band (Before Trim)")
        axes[2, 1].axis("off")

        axes[2, 2].imshow(crop_mask, cmap="gray")
        axes[2, 2].set_title("12. Band Mask")
        axes[2, 2].axis("off")

        axes[2, 3].imshow(crop_gray_clean, cmap="gray")
        axes[2, 3].set_title(f"13. CleanGray OCR\n'{text_gray}' ({conf_gray:.2f})",
                             color="red", fontweight="bold", fontsize=9)
        axes[2, 3].axis("off")

        axes[2, 4].axis("off")

        plt.tight_layout()
        plt.show()

    print("\n[INFO] Done ✅")
