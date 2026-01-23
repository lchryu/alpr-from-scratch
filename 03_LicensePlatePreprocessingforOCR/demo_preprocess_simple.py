import cv2
import numpy as np
import os

# Hard code: đường dẫn ảnh
img_path = "./outputs/crops/plate_001.jpg"
output_dir = "./outputs/demo"

# Tạo thư mục output
os.makedirs(output_dir, exist_ok=True)

# Load ảnh
img = cv2.imread(img_path)
if img is None:
    print(f"ERROR: Cannot load image from {img_path}")
    exit(1)
print(f"Original shape: {img.shape}")

# ===== BƯỚC 1: GRAYSCALE =====
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(f"{output_dir}/01_gray.jpg", gray)
print("✓ Step 1: Grayscale")

# ===== BƯỚC 2: RESIZE (phóng to) =====
h, w = gray.shape
scale = 2
resized = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(f"{output_dir}/02_resize.jpg", resized)
print("✓ Step 2: Resize")

# ===== BƯỚC 3: GAUSSIAN BLUR =====
blur = cv2.GaussianBlur(resized, (5, 5), 0)
cv2.imwrite(f"{output_dir}/03_blur.jpg", blur)
print("✓ Step 3: Gaussian Blur")

# ===== BƯỚC 4: ADAPTIVE THRESHOLD =====
thresh = cv2.adaptiveThreshold(
    blur, 
    255,                          # max value
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # method
    cv2.THRESH_BINARY,            # type
    11,                           # block size
    2                             # C constant
)
cv2.imwrite(f"{output_dir}/04_thresh.jpg", thresh)
print("✓ Step 4: Adaptive Threshold")

# ===== BƯỚC 5: MORPHOLOGY (làm sạch noise) =====
kernel = np.ones((3, 3), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
cv2.imwrite(f"{output_dir}/05_morph.jpg", morph)
print("✓ Step 5: Morphology")

print(f"\nDone! Check folder: {output_dir}")

