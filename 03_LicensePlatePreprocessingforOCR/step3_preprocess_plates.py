import cv2
import os
import numpy as np

# ===================== CONFIG =====================
INPUT_DIR = "./outputs/crops"
OUTPUT_DIR = "./outputs/preprocess"

STEPS = ["gray", "blur", "thresh", "morph", "final"]

# ===================== SETUP =====================
for step in STEPS:
    os.makedirs(os.path.join(OUTPUT_DIR, step), exist_ok=True)

# ===================== PREPROCESS FUNCTION =====================
def preprocess_plate(img):
    steps = {}

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps["gray"] = gray

    # 2. Resize (scale up)
    h, w = gray.shape
    scale = 2
    resized = cv2.resize(gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

    # 3. Gaussian Blur
    blur = cv2.GaussianBlur(resized, (5,5), 0)
    steps["blur"] = blur

    # 4. Adaptive Threshold
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    steps["thresh"] = thresh

    # 5. Morphology (clean noise)
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    steps["morph"] = morph

    steps["final"] = morph
    return steps

# ===================== MAIN =====================
print("[INFO] Running preprocessing pipeline...")

# Check if input directory exists
if not os.path.exists(INPUT_DIR):
    print(f"[ERROR] Input directory '{INPUT_DIR}' does not exist!")
    print(f"[INFO] Please create the directory and add plate images to process.")
    exit(1)

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(img_path)

    if img is None:
        print(f"[WARN] Cannot read {fname}")
        continue

    print(f"[PROCESS] {fname}")

    results = preprocess_plate(img)

    name = os.path.splitext(fname)[0]

    for step, result_img in results.items():
        save_path = os.path.join(OUTPUT_DIR, step, f"{name}_{step}.jpg")
        cv2.imwrite(save_path, result_img)

print("\n====================== DONE ======================")
print("[INFO] Preprocessing completed.")
print("[INFO] Check folder: outputs/preprocess")
print("==================================================")
