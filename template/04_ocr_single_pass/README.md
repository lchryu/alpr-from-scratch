# Module 04: OCR Single Pass

Perform Optical Character Recognition (OCR) on preprocessed license plate images using a single-pass approach.

## Purpose in the Pipeline

This module extracts text from preprocessed license plate images using OCR. The single-pass approach applies OCR once to the entire image, suitable for clear, well-preprocessed plates. This is the foundation for text extraction before validation and post-processing.

## Key Ideas

- Use OCR engines (Tesseract OCR, EasyOCR, PaddleOCR, etc.)
- Apply OCR to the entire preprocessed plate image
- Extract raw text predictions with confidence scores
- Handle common OCR errors and character misrecognitions
- Return text strings and confidence metrics

## Input / Output

**Input:**
- Preprocessed license plate images (from Module 03)
- OCR engine configuration (language, character whitelist, etc.)

**Output:**
- Raw OCR text predictions (one per plate)
- Confidence scores for each prediction
- Character-level details (optional)

## Implementation Notes (Planned)

- Integrate Tesseract OCR or modern OCR libraries (EasyOCR, PaddleOCR)
- Configure OCR parameters (language, character set, page segmentation mode)
- Extract confidence scores for quality assessment
- Handle multiple OCR engines for comparison
- Support batch processing of multiple plates
- Log OCR results for analysis

## Previous Module

← [Module 03: Preprocess Plate Images](../03_preprocess_plate/)

## Next Module

→ [Module 05: OCR Multi Pass](../05_ocr_multi_pass/)

