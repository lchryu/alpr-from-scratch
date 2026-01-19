# Module 02: Crop License Plate Regions

Extract individual license plate regions from detected bounding boxes for further processing.

## Purpose in the Pipeline

After detecting license plates in an image (Module 01), we need to extract each detected plate as a separate image. This module takes the bounding box coordinates and crops the corresponding regions from the original image, preparing them for preprocessing and OCR stages.

## Key Ideas

- Extract rectangular regions using bounding box coordinates (x1, y1, x2, y2)
- Handle multiple detections by cropping each plate separately
- Preserve image quality and aspect ratio during cropping
- Add padding/margin around cropped regions if needed
- Save cropped plates as individual image files or return as array

## Input / Output

**Input:**
- Original image with detected license plates
- Bounding box coordinates from Module 01 (list of boxes with x1, y1, x2, y2)

**Output:**
- Individual cropped license plate images (one per detection)
- Metadata (original coordinates, image dimensions)

## Implementation Notes (Planned)

- Use OpenCV or PIL/Pillow for image cropping operations
- Implement padding option to add margin around detected boxes
- Handle edge cases (boxes extending beyond image boundaries)
- Support batch processing for multiple detections
- Save cropped images with meaningful filenames (e.g., `plate_0.jpg`, `plate_1.jpg`)

## Previous Module

← [Module 01: Detect Plate with YOLOv8](../1Detect_LP_bbox/)

## Next Module

→ [Module 03: Preprocess Plate](../03_preprocess_plate/)

