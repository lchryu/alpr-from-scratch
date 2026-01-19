# ALPR from Scratch

This repository is a self-learning, step-by-step series on building a License Plate Recognition pipeline from scratch.

## Learning Goals

By following this series, you will learn how to:

- Implement object detection for license plate localization using YOLOv8
- Extract and crop detected regions from images
- Apply image preprocessing techniques to enhance OCR accuracy
- Perform optical character recognition using single and multi-pass approaches
- Validate and postprocess OCR results to produce accurate license plate text
- Understand how individual components compose into a complete ALPR pipeline

## Pipeline Overview

The ALPR pipeline consists of six sequential modules:

1. **01_detect_plate_yolov8** - Detect license plates in images using YOLOv8 object detection
2. **02_crop_plate** - Extract individual license plate regions from detected bounding boxes
3. **03_preprocess_plate** - Enhance cropped images through preprocessing techniques
4. **04_ocr_single_pass** - Perform OCR using a single-pass approach
5. **05_ocr_multi_pass** - Apply advanced OCR with multiple passes and strategies
6. **06_postprocess_validate** - Clean OCR results and validate license plate formats

Each module is designed to be independent and can be used standalone, but they are composable to form a complete end-to-end pipeline.

## Who This Repo Is For

This series is suitable for:

- Students and learners interested in computer vision and deep learning
- Developers seeking to understand ALPR pipeline architecture
- Engineers wanting to build custom license plate recognition systems
- Anyone looking to learn practical image processing and OCR techniques

Basic knowledge of Python and computer vision concepts is assumed.

## What This Repo Is NOT

This repository is:

- **Not a production-ready library** - This is an educational series focused on learning
- **Not a plug-and-play solution** - Modules require understanding and customization
- **Not optimized for deployment** - Code prioritizes clarity and learning over performance
- **Not a complete dataset or training guide** - Focus is on pipeline implementation, not model training

## Project Structure

```
alpr-from-scratch/
├── README.md
├── requirements.txt
├── 1Detect_LP_bbox/           # Module 01: Detect Plate with YOLOv8
├── 02_crop_plate/              # Module 02: Crop Plate Regions
├── 03_preprocess_plate/        # Module 03: Preprocess Plate Images
├── 04_ocr_single_pass/         # Module 04: OCR Single Pass
├── 05_ocr_multi_pass/          # Module 05: OCR Multi Pass
└── 06_postprocess_validate/    # Module 06: Postprocess and Validate
```

## How to Use This Series

1. **Start with Module 01** - Begin with the license plate detection module to understand the foundation
2. **Follow sequentially** - Work through modules in numerical order (01 → 02 → 03 → 04 → 05 → 06)
3. **Read module READMEs** - Each module contains its own README with specific instructions
4. **Experiment and modify** - Use the code as a starting point for your own experiments
5. **Compose modules** - Once familiar with individual modules, combine them to build the complete pipeline

Each module directory contains implementation code and documentation. Refer to individual module READMEs for detailed usage instructions.

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd alpr-from-scratch
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Begin with Module 01:
   ```bash
   cd 1Detect_LP_bbox
   python main.py
   ```

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for detailed dependencies

## Closing Note

This series is designed to be a learning journey. Take your time with each module, experiment with the code, and don't hesitate to modify and extend the implementations. The goal is understanding how each component works and how they fit together to create a complete ALPR system.
