from os import system
from ultralytics import YOLO
import cv2
import warnings

system("cls")
warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.load.*weights_only.*')

# ==================== CONFIG ====================
# Bounding box parameters
BBOX_COLOR = (0, 255, 0)  # Green
BBOX_THICKNESS = 2

# Text parameters
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
TEXT_COLOR = (0, 255, 0)  # Green
TEXT_THICKNESS = 1

# ==================== LOAD MODEL & IMAGE ====================
model = YOLO("./model/best.pt")
img_path = "./img/multi_plates.png"
img = cv2.imread(img_path)

# ==================== DETECTION ====================
results = model(img)

# ==================== DRAW RESULTS ====================
for r in results:
    for idx, box in enumerate(r.boxes):
        # Extract box information
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls)
        class_name = model.names[class_id]
        
        # Print detection info
        print(f"\n\n{'=' * 50}Box[{idx}]{'=' * 50}")
        print(f"Coordinates: ({x1}, {y1}, {x2}, {y2})")
        print(f"Confidence: {confidence:.4f}")
        print(f"Class: {class_name} (ID: {class_id})")
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)
        
        # Draw text with confidence
        text = f"plate {confidence:.2f}"
        cv2.putText(
            img,
            text,
            (x1, y1 - 5), # text position
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            TEXT_THICKNESS
        )

# ==================== SAVE RESULT ====================
cv2.imwrite("bbox_result.jpg", img)
