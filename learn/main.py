import warnings
from pathlib import Path

import cv2
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=FutureWarning)

# ===================== CONFIG =====================
MODEL_PATH = Path("./model/best.pt")
IMAGE_PATH = Path("./img/multi_plates.png")

CROP_DIR = Path("./outputs/crops")
DEBUG_DIR = Path("./outputs/debug")

BBOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 255)
BBOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
TEXT_THICKNESS = 2

# ===================== SETUP =====================
CROP_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# ===================== LOAD MODEL =====================
print("[INFO] Loading YOLOv8 model...")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

# ===================== LOAD IMAGE =====================
print("[INFO] Loading image...")
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"❌ Image not found: {IMAGE_PATH}")

img = cv2.imread(str(IMAGE_PATH))
if img is None:
    raise ValueError(f"❌ Failed to read image: {IMAGE_PATH}")

h, w, _ = img.shape
print(f"[INFO] Image size: {w}x{h}")

# ===================== DETECTION =====================
print("[INFO] Running detection...")
results = model(img)

debug_img = img.copy()
plate_count = 0
# print(type(results))
# print(len(results))
# print(type(results[0]))
# Pause here to inspect results
# breakpoint()  # Python debugger - press 'c' to continue, 'q' to quit
# Alternative: 
# input("Press Enter to continue...")  # Simple pause

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

        crop_path = CROP_DIR / f"plate_{idx}.jpg"
        cv2.imwrite(str(crop_path), plate_img)
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
debug_path = DEBUG_DIR / "debug_detection.jpg"
cv2.imwrite(str(debug_path), debug_img)

print("\n====================== DONE ======================")
print(f"[INFO] Total plates detected: {plate_count}")
print(f"[INFO] Debug image saved at: {debug_path}")
print(f"[INFO] Crops saved in: {CROP_DIR}")
print("==================================================")
