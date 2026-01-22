# 🚗 License Plate Detection with YOLOv8

A simple and clean implementation for detecting and cropping license plates from images using **YOLOv8 + OpenCV**.  
Optimized for **ALPR research, self-learning, and graduation thesis projects**.

---

## 📋 Overview

This project uses YOLOv8 to detect license plates in images and automatically crops them into separate files. Perfect for:

- **Research projects** on Automatic License Plate Recognition (ALPR)
- **Learning** computer vision and object detection
- **Thesis projects** requiring license plate detection

### 🎯 What it does:

1. **Input**: An image containing one or more license plates
2. **Detection**: YOLOv8 detects all license plates in the image
3. **Output**:
   - Individual cropped license plate images
   - Debug visualization with bounding boxes

### 📸 Example Results

**Input Image:**

<div align="center">
  <img src="examples/input_sample.png" alt="Input Sample" style="max-width: 800px; width: 100%; height: auto;">
</div>

**Detection Result (with bounding boxes):**

<div align="center">
  <img src="examples/output_debug.jpg" alt="Debug Output" style="max-width: 800px; width: 100%; height: auto;">
</div>

**Cropped License Plates:**
| Plate 1 | Plate 2 | Plate 3 | Plate 4 |
|---------|---------|---------|---------|
| ![Plate 0](examples/plate_0.jpg) | ![Plate 1](examples/plate_1.jpg) | ![Plate 2](examples/plate_2.jpg) | ![Plate 3](examples/plate_3.jpg) |

---

## 📦 Environment Setup (Using Conda - Recommended)

> **Why Conda?** Conda is highly recommended for deep learning projects due to **better dependency management, CUDA compatibility, and reproducibility**.

---

## 1. Create Conda Environment

Create a new conda environment with Python 3.10:

```bash
conda create -n alpr python=3.10 -y
```

**Activate the environment:**

```bash
conda activate alpr
```

**Verify Python version:**

```bash
python --version
```

**Expected output:**

```
Python 3.10.x
```

---

## 2. Install PyTorch

### 🔥 With NVIDIA GPU (Recommended)

If you have an NVIDIA GPU, install PyTorch with CUDA support for faster inference:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**Verify GPU availability:**

```bash
python - << EOF
import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
EOF
```

**Expected output:**

```
CUDA Available: True
GPU: RTX ...
```

### 🐢 CPU Only (No GPU)

If you don't have a GPU or want to use CPU only:

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

> **Note:** CPU inference will be slower but works on any machine.

---

## 3. Install YOLOv8 & OpenCV

Install the required packages:

```bash
pip install ultralytics opencv-python
```

**Verify installation:**

```bash
python -c "from ultralytics import YOLO; import cv2; print('OK')"
```

If you see `OK`, the installation is successful!

---

## 📁 Project Structure

```
learn/
│
├── model/
│   └── best.pt              # YOLOv8 trained model
│
├── img/
│   └── multi_plates.png     # Input test image
│
├── outputs/
│   ├── crops/               # Cropped license plates (output)
│   └── debug/               # Debug visualization (output)
│
├── examples/                # Sample images for documentation
│   ├── input_sample.png     # Input example
│   ├── output_debug.jpg     # Detection visualization
│   └── plate_*.jpg          # Cropped plates examples
│
├── main.py                  # Main detection script
└── README.md                # This file
```

---

## ▶ How to Use

### Step 1: Prepare Your Image

Place your image in the `img/` folder, or update the `IMAGE_PATH` in `main.py`.

### Step 2: Run Detection

```bash
python main.py
```

### Step 3: Check Results

After running, you'll find:

- **Cropped license plates** in: `outputs/crops/`
  - Files named: `plate_0.jpg`, `plate_1.jpg`, `plate_2.jpg`, etc.
- **Debug visualization** in: `outputs/debug/debug_detection.jpg`
  - Shows the original image with bounding boxes around detected plates

### Example Console Output:

```
[INFO] Loading YOLOv8 model...
[INFO] Loading image...
[INFO] Image size: 1920x1080
[INFO] Running detection...

============================================================
[PLATE 0]
Class      : license_plate
Confidence : 0.9234
Box        : (245, 180) -> (420, 230)
[SAVED] outputs/crops/plate_0.jpg
...

====================== DONE ======================
[INFO] Total plates detected: 4
[INFO] Debug image saved at: outputs/debug/debug_detection.jpg
[INFO] Crops saved in: outputs/crops
==================================================
```

---

## ⚡ Quick Install (One-liner)

For experienced users, here's a one-liner to set up everything:

```bash
conda create -n alpr python=3.10 -y && conda activate alpr && conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y && pip install ultralytics opencv-python
```

---

## 🔧 Configuration

You can customize the detection by editing `main.py`:

```python
MODEL_PATH = "./model/best.pt"        # Path to your YOLOv8 model
IMAGE_PATH = "./img/multi_plates.png" # Input image path
CROP_DIR = "./outputs/crops"          # Output directory for crops
DEBUG_DIR = "./outputs/debug"         # Output directory for debug images
```

---

## 🐛 Debugging & Tips

### Pausing Program Execution

When debugging or inspecting results, you may want to pause the program execution. Here are several options:

#### 1. Using `breakpoint()` (Recommended for Debugging)

The `breakpoint()` function (Python 3.7+) enters the Python debugger (pdb), allowing you to:
- Inspect variables and their values
- Execute Python commands
- Step through code line by line

```python
# In your code
results = model(img)
breakpoint()  # Program pauses here
# Press 'c' to continue, 'q' to quit, 'h' for help
```

**Debugger commands:**
- `c` or `continue` - Continue execution
- `q` or `quit` - Exit debugger and program
- `n` or `next` - Execute next line
- `s` or `step` - Step into function calls
- `p <variable>` - Print variable value
- `h` or `help` - Show help

#### 2. Using `input()` (Simple Pause)

For a simple pause without debugger:

```python
# In your code
results = model(img)
input("Press Enter to continue...")  # Program waits for Enter key
```

#### 3. Using `sys.exit()` (Exit Completely)

To exit the program completely:

```python
import sys
# In your code
sys.exit()  # Exits immediately, no continuation
```

#### 4. Using `os.system('pause')` (Windows Only)

Windows-specific pause command:

```python
import os
# In your code
os.system('pause')  # Shows "Press any key to continue..."
```

### Understanding YOLOv8 Results Structure

🧠 **Why `for r in results` even with a single image?**

Even when you pass only **1 image**, YOLOv8 **always returns a list-like results object**. This is by design!

#### The Design Philosophy

YOLOv8 uses a **batch-first mindset** in its API design. Whether you process:
- 1 image
- 10 images (batch)
- 1 folder
- 1 video
- Webcam stream

→ **The API is unified**: it always returns a list-like structure.

#### Example: Inspect the Results Structure

Add this to see the actual structure:

```python
results = model(img)

print(type(results))    # <class 'list'>
print(len(results))     # 1
print(type(results[0])) # <class 'ultralytics.yolo.engine.results.Results'>
```

**Output:**
```
<class 'list'>
1
<class 'ultralytics.yolo.engine.results.Results'>
```

#### Why This Design?

Even with **1 image**, YOLOv8 returns:
```python
results = [Result(img1)]  # List containing 1 Results object
```

**Not:**
```python
results = Result(img1)  # Single object (inconsistent API)
```

**Benefits:**
- ✅ **Unified API** - Same code works for single image, batch, folder, video
- ✅ **Pipeline consistency** - YOLOv8 processes everything as batches internally
- ✅ **Developer-friendly** - No need to remember different return types
- ✅ **Future-proof** - Easy to extend to batch processing

#### Practical Usage

```python
# Works the same way for:
results = model(img)                    # Single image → [Result]
results = model([img1, img2, img3])     # Batch → [Result, Result, Result]
results = model("folder/")              # Folder → [Result, Result, ...]
results = model("video.mp4")            # Video → [Result, Result, ...]

# Always iterate, even for single image:
for r in results:  # ← Always use this pattern
    boxes = r.boxes
    # Process each result
```

### Example: Inspecting Detection Results

```python
# After detection
results = model(img)

# Inspect structure
print(type(results))    # <class 'list'>
print(len(results))     # 1 (for single image)
print(type(results[0])) # <class 'ultralytics.yolo.engine.results.Results'>

# Option 1: Debugger (best for inspection)
breakpoint()

# Option 2: Simple pause
# input("Press Enter to continue...")

# Always iterate, even for single image
for r in results:
    # Process results
    boxes = r.boxes
    # ...
```

---

## 🔁 Export / Rebuild Environment (For Thesis & Reproducibility)

### Export Environment:

Save your environment configuration for sharing or reproducibility:

```bash
conda env export --no-build > environment.yml
```

> **Note:** The `--no-build` flag excludes build information, making the file smaller and more portable across different platforms.

### Rebuild Environment:

Recreate the environment from the exported file:

```bash
conda env create -f environment.yml
```

This is especially useful for:

- **Thesis documentation** - Include `environment.yml` in your project
- **Reproducibility** - Others can recreate your exact environment
- **Backup** - Save your working configuration

---

## 📝 Notes

- The model file (`best.pt`) should be placed in the `model/` folder
- Supported image formats: `.jpg`, `.png`, `.jpeg`
- The script automatically creates output directories if they don't exist
- Detection confidence threshold can be adjusted in the model inference

---

## 🤝 Contributing

Feel free to fork, modify, and use this project for your research or learning purposes!

---

## 📄 License

This project is provided as-is for educational and research purposes.
