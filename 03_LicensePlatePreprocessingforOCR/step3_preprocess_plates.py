import os
from typing import Final
import cv2
import numpy as np

# ===================== CONFIG =====================
INPUT_DIR: Final[str] = "./outputs/crops"
OUTPUT_DIR: Final[str] = "./outputs/preprocess"

STEPS: Final[list[str]] = [
    "01_gray",
    "02_resize",
    "03_blur",
    "04_thresh",
    "05_morph",
    "06_final"
]

# ===================== SETUP =====================
for step in STEPS:
    os.makedirs(os.path.join(OUTPUT_DIR, step), exist_ok=True)

# ===================== PREPROCESS FUNCTION =====================
def preprocess_plate(img: np.ndarray, scale: int = 2) -> dict[str, np.ndarray]:
    steps: dict[str, np.ndarray] = {}

    # 01 - Grayscale
    gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps["01_gray"] = gray

    # 02 - Resize
    h: int
    w: int
    h, w = gray.shape
    resized: np.ndarray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    steps["02_resize"] = resized

    # 03 - Gaussian Blur
    blur: np.ndarray = cv2.GaussianBlur(resized, (5, 5), 0)
    steps["03_blur"] = blur

    # 04 - Adaptive Threshold
    thresh: np.ndarray = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    steps["04_thresh"] = thresh

    # 05 - Morphology
    kernel: np.ndarray = np.ones((3, 3), np.uint8)
    morph: np.ndarray = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    steps["05_morph"] = morph

    # 06 - Final
    steps["06_final"] = morph

    return steps

# ===================== MAIN =====================
print("[INFO] Running preprocessing pipeline...")

if not os.path.exists(INPUT_DIR):
    print(f"[ERROR] Input directory '{INPUT_DIR}' does not exist!")
    print("[INFO] Run Step 2 (crop) first.")
    raise SystemExit(1)

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path: str = os.path.join(INPUT_DIR, fname)
    img: np.ndarray | None = cv2.imread(img_path)

    if img is None:
        print(f"[WARN] Cannot read {fname}")
        continue

    print(f"[PROCESS] {fname}")

    results: dict[str, np.ndarray] = preprocess_plate(img, scale=2)
    name: str = os.path.splitext(fname)[0]

    for step, result_img in results.items():
        save_path: str = os.path.join(OUTPUT_DIR, step, f"{name}_{step}.jpg")
        cv2.imwrite(save_path, result_img)

print("\n====================== DONE ======================")
print("[INFO] Preprocessing completed.")
print("[INFO] Check folder: outputs/preprocess")
print("==================================================")
