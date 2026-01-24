import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os

# ===== HARD CODE =====
model_path = "./model/best.pt"
img_path = "./img/multi_plates.png"
output_dir = "./outputs/demo_alpr"

os.makedirs(output_dir, exist_ok=True)

# ===== BƯỚC 1: LOAD MODEL VÀ ẢNH =====
print("[1] Loading YOLO model...")
model = YOLO(model_path)

print("[2] Loading image...")
img = cv2.imread(img_path)
print(f"Image shape: {img.shape}")

# ===== BƯỚC 2: DETECTION (Tìm biển số) =====
print("[3] Running detection...")
results = model(img)

# Lấy biển số đầu tiên
for r in results:
    for box in r.boxes:
        # Lấy tọa độ bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        print(f"Found plate at ({x1}, {y1}) to ({x2}, {y2}), conf: {conf:.3f}")
        
        # Crop biển số
        plate_img = img[y1:y2, x1:x2]
        cv2.imwrite(f"{output_dir}/00_crop.jpg", plate_img)
        print("✓ Saved crop")
        
        # ===== BƯỚC 3: PREPROCESS =====
        # Grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{output_dir}/01_gray.jpg", gray)
        
        # Resize (phóng to)
        h, w = gray.shape
        resized = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"{output_dir}/02_resize.jpg", resized)
        
        # Blur
        blur = cv2.GaussianBlur(resized, (5, 5), 0)
        cv2.imwrite(f"{output_dir}/03_blur.jpg", blur)
        
        # Threshold (nhị phân hóa)
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        cv2.imwrite(f"{output_dir}/04_thresh.jpg", thresh)
        
        # Morphology
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        cv2.imwrite(f"{output_dir}/05_morph.jpg", morph)
        
        print("✓ Preprocessing done")
        
        # ===== BƯỚC 4: OCR (Đọc text) =====
        print("[4] Running OCR...")
        reader = easyocr.Reader(['en'])
        ocr_results = reader.readtext(morph)
        
        # Lấy text và confidence
        texts = []
        confs = []
        for (bbox, text, conf) in ocr_results:
            texts.append(text)
            confs.append(conf)
            print(f"  Found: '{text}' (conf: {conf:.3f})")
        
        final_text = "".join(texts)
        mean_conf = np.mean(confs) if confs else 0.0
        
        print("\n" + "=" * 50)
        print(f"RESULT:")
        print(f"  Detection conf: {conf:.3f}")
        print(f"  OCR text      : {final_text}")
        print(f"  OCR conf      : {mean_conf:.3f}")
        print("=" * 50)
        
        # Chỉ xử lý biển số đầu tiên
        break
    break

print(f"\nDone! Check folder: {output_dir}")

