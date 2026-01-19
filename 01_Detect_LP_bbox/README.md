# License Plate Detection using YOLOv8

A simple project to detect license plates in images using YOLOv8 object detection model.

## Features

- 🎯 **Multi-plate Detection**: Detects multiple license plates in a single image
- 📊 **Confidence Scores**: Shows detection confidence for each detected plate
- 📦 **Bounding Box Visualization**: Draws green bounding boxes around detected plates
- 🔧 **Easy Configuration**: Simple configuration file for customization
- 📝 **Detailed Output**: Console output with coordinates, confidence, and class information

## Project Structure

```
1Detect_LP_bbox/
├── main.py           # Main detection script
├── config.py         # Configuration file
├── model/
│   └── best.pt      # Pre-trained YOLOv8 model
├── img/
│   └── multi_plates.png    # Input test image
└── bbox_result.jpg   # Output image with detection results
```

## How It Works

1. **Load Model**: Initialize YOLOv8 model from `./model/best.pt`
2. **Load Image**: Read the input image from `./img/multi_plates.png`
3. **Detection**: Run inference on the image using `model(img)`
4. **Extract Results**: For each detected box, extract:
   - Coordinates (x1, y1, x2, y2)
   - Confidence score
   - Class ID and name
5. **Draw & Save**: Draw bounding boxes on image and save as `bbox_result.jpg`

### Input Image

![Input Image](img/multi_plates.png)

## Code Example

Here's a simplified version of the main detection loop:

```python
from ultralytics import YOLO
import cv2

# Load model and image
model = YOLO("./model/best.pt")
img = cv2.imread("./img/multi_plates.png")

# Run detection
results = model(img)

# Process results
for r in results:
    for idx, box in enumerate(r.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_name = model.names[int(box.cls)]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw confidence text
        cv2.putText(img, f"{class_name} {confidence:.2f}", 
                   (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 0), 1)

cv2.imwrite("bbox_result.jpg", img)
```

## Key Code Concepts

### 1. Why `box.conf[0]`?
```python
confidence = float(box.conf[0])
```
- `box.conf` is a **tensor array** (not a simple number)
- `[0]` extracts the **first element** from the tensor
- `float()` converts it to a Python float for easier use

### 2. Class ID Mapping with `model.names`
```python
class_id = int(box.cls)
class_name = model.names[class_id]
```
- Class ID (0, 1, 2...) comes from your **dataset labeling**
- `model.names` is a dictionary that maps: `0 → 'plate'`, `1 → 'other'`, etc.
- This is defined in your model's `data.yaml` file during training
- Shows human-readable class names instead of just numbers

### 3. Using `enumerate()` for Indexing
```python
for idx, box in enumerate(r.boxes):
```
- `enumerate()` provides both the index (0, 1, 2...) and the item
- Useful for tracking and labeling multiple detections

## Sample Output

```
==================================================Box[0]==================================================
Coordinates: (333, 296, 439, 379)
Confidence: 0.8773
Class: plate (ID: 0)

==================================================Box[1]==================================================
Coordinates: (571, 346, 672, 430)
Confidence: 0.8732
Class: plate (ID: 0)

==================================================Box[2]==================================================
Coordinates: (63, 275, 168, 353)
Confidence: 0.8686
Class: plate (ID: 0)

==================================================Box[3]==================================================
Coordinates: (586, 95, 656, 150)
Confidence: 0.6151
Class: plate (ID: 0)
```

## Output Image

The detected license plates are drawn on the image with:
- **Green bounding boxes** around each detected plate
- **Confidence score** displayed above each box (e.g., "plate 0.88")

Saved as `bbox_result.jpg`

![Output Image](bbox_result.jpg)

## Key Learnings

1. **Tensor vs Python Types**: Always convert tensors to native Python types when needed (`float()`, `int()`)
2. **Array Indexing**: Access specific elements with `[0]` instead of treating tensors as scalars
3. **Class Mapping**: Use `model.names` to get human-readable class names from numeric IDs
4. **Loop Indexing**: Use `enumerate()` to track iteration count without manual counter
5. **Code Organization**: Structure code into clear sections (CONFIG, LOAD, DETECTION, DRAW, SAVE)
6. **Constants**: Use UPPER_CASE for configuration values that don't change

## Requirements

- Python 3.8+
- PyTorch
- ultralytics
- opencv-python

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install ultralytics opencv-python torch
```

Or install from requirements (if available):
```bash
pip install -r requirements.txt
```

3. Make sure you have:
   - YOLOv8 model at `./model/best.pt`
   - Test image at `./img/multi_plates.png`

## Usage

Run the detection script:
```bash
python main.py
```

The script will:
1. Load the YOLOv8 model from `./model/best.pt`
2. Process the image from `./img/multi_plates.png`
3. Display detection results in the console
4. Save the output image as `bbox_result.jpg`

## Configuration

You can customize the detection visualization by modifying constants in `main.py`:
- `BBOX_COLOR`: Bounding box color (BGR format)
- `BBOX_THICKNESS`: Thickness of bounding box lines
- `FONT_SCALE`: Size of confidence text
- `TEXT_COLOR`: Color of confidence text

## Output

- **Console output**: Detailed detection information for each box including:
  - Box index
  - Coordinates (x1, y1, x2, y2)
  - Confidence score
  - Class name and ID
- **Output image**: `bbox_result.jpg` with drawn bounding boxes and confidence scores
