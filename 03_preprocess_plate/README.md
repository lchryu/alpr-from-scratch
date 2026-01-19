# Module 03: Preprocess License Plate Images

Enhance cropped license plate images to improve OCR accuracy through various image preprocessing techniques.

## Purpose in the Pipeline

Raw cropped license plates may have poor lighting, noise, or low contrast. This module applies preprocessing techniques to normalize and enhance the images, making them more suitable for OCR recognition. Preprocessing significantly improves OCR accuracy.

## Key Ideas

- Convert to grayscale for consistent processing
- Apply noise reduction (Gaussian blur, median filter)
- Enhance contrast and brightness (histogram equalization, CLAHE)
- Binarization (thresholding) to create black-and-white images
- Resize/normalize images to optimal dimensions for OCR
- Handle different lighting conditions and image quality

## Input / Output

**Input:**
- Cropped license plate images (from Module 02)
- Preprocessing parameters (threshold values, resize dimensions, etc.)

**Output:**
- Preprocessed license plate images (enhanced, normalized)
- Preprocessing metadata (applied transformations)

## Implementation Notes (Planned)

- Implement multiple preprocessing pipelines (grayscale → denoise → enhance → binarize)
- Use OpenCV for image transformations
- Experiment with different thresholding methods (Otsu, adaptive thresholding)
- Apply morphological operations if needed (opening, closing)
- Support configurable preprocessing steps
- Save intermediate results for debugging

## Previous Module

← [Module 02: Crop Plate Regions](../02_crop_plate/)

## Next Module

→ [Module 04: OCR Single Pass](../04_ocr_single_pass/)

