from ultralytics import YOLO
import cv2
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ===================== CONFIG =====================
MODEL_PATH = "./model/best.pt"
IMAGE_PATH = "./img/multi_plates.png"

CROP_DIR = "./outputs/crops"
DEBUG_DIR = "./outputs/debug"

BBOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 255)
BBOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
TEXT_THICKNESS = 2

# ===================== SETUP =====================
os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# ===================== LOAD MODEL =====================
print("[INFO] Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)

# ===================== LOAD IMAGE =====================
print("[INFO] Loading image...")
img = cv2.imread(IMAGE_PATH)
assert img is not None, "❌ Không đọc được ảnh!"

h, w, _ = img.shape
print(f"[INFO] Image size: {w}x{h}")

# ===================== DETECTION =====================
print("[INFO] Running detection...")
results = model(img)

debug_img = img.copy()
plate_count = 0

# ===================== PROCESS RESULTS =====================
for r in results:
    for idx, box in enumerate(r.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]

        print(f"\n{'='*60}")
        print(f"[PLATE {idx}]")
        print(f"Class      : {cls_name}")
        print(f"Confidence : {conf:.4f}")
        print(f"Box        : ({x1}, {y1}) -> ({x2}, {y2})")

        # ===================== CROP =====================
        plate_img = img[y1:y2, x1:x2]
        if plate_img.size == 0:
            print("[WARN] Empty crop, skipping...")
            continue

        crop_path = os.path.join(CROP_DIR, f"plate_{idx}.jpg")
        cv2.imwrite(crop_path, plate_img)
        print(f"[SAVED] {crop_path}")

        # ===================== DRAW DEBUG =====================
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)
        label = f"{idx}: {conf:.2f}"
        cv2.putText(
            debug_img,
            label,
            (x1, y1 - 8),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            TEXT_THICKNESS
        )

        plate_count += 1

# ===================== SAVE DEBUG IMAGE =====================
debug_path = os.path.join(DEBUG_DIR, "debug_detection.jpg")
cv2.imwrite(debug_path, debug_img)

print("\n====================== DONE ======================")
print(f"[INFO] Total plates detected: {plate_count}")
print(f"[INFO] Debug image saved at: {debug_path}")
print(f"[INFO] Crops saved in: {CROP_DIR}")
print("==================================================")
