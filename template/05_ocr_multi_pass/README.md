# Module 05: OCR Multi Pass

Apply advanced OCR techniques using multiple passes and strategies to improve recognition accuracy for challenging license plates.

## Purpose in the Pipeline

Some license plates may have poor quality, unusual fonts, or complex backgrounds. This module applies multiple OCR strategies (different preprocessing, multiple OCR engines, region-based recognition) and combines results to achieve higher accuracy than single-pass OCR.

## Key Ideas

- Run OCR multiple times with different preprocessing strategies
- Use multiple OCR engines and combine results (ensemble approach)
- Apply region-based OCR (split plate into character regions)
- Use voting or confidence-weighted aggregation of results
- Handle ambiguous cases where single-pass OCR fails

## Input / Output

**Input:**
- Preprocessed license plate images (from Module 03)
- Multiple OCR strategies/engines configuration

**Output:**
- Best OCR prediction (consensus or highest confidence)
- Alternative predictions with confidence scores
- Strategy metadata (which approach worked best)

## Implementation Notes (Planned)

- Implement multiple preprocessing variations
- Run OCR with different engines (Tesseract, EasyOCR, PaddleOCR)
- Apply ensemble methods (voting, weighted average)
- Implement region-based OCR for character-level recognition
- Compare and rank results by confidence
- Fallback strategies for low-confidence predictions

## Previous Module

← [Module 04: OCR Single Pass](../04_ocr_single_pass/)

## Next Module

→ [Module 06: Postprocess and Validate](../06_postprocess_validate/)

