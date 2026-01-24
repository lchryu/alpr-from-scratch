"""
Module 03: Preprocess License Plate Images

This module applies image preprocessing techniques to enhance cropped license plates
for better OCR recognition accuracy.

TODO: Implement preprocessing pipeline
- Grayscale conversion
- Noise reduction
- Contrast enhancement
- Binarization/thresholding
- Resize normalization
"""

import cv2
import numpy as np
from typing import Optional, Dict


def preprocess_plate(
    image: np.ndarray,
    grayscale: bool = True,
    denoise: bool = True,
    enhance_contrast: bool = True,
    binarize: bool = True,
    resize: Optional[tuple] = None
) -> np.ndarray:
    """
    Preprocess a license plate image for OCR.
    
    Args:
        image: Input license plate image (BGR or RGB)
        grayscale: Convert to grayscale
        denoise: Apply noise reduction
        enhance_contrast: Enhance contrast using histogram equalization
        binarize: Apply thresholding to create binary image
        resize: Optional target size (width, height)
        
    Returns:
        Preprocessed image ready for OCR
    """
    # TODO: Convert to grayscale if needed
    # TODO: Apply denoising (Gaussian blur, median filter)
    # TODO: Enhance contrast (CLAHE, histogram equalization)
    # TODO: Apply binarization (Otsu threshold, adaptive threshold)
    # TODO: Resize if specified
    # TODO: Return preprocessed image
    
    pass


def preprocess_pipeline(image: np.ndarray, config: Dict) -> np.ndarray:
    """
    Apply a configurable preprocessing pipeline.
    
    Args:
        image: Input license plate image
        config: Dictionary with preprocessing parameters
        
    Returns:
        Preprocessed image
    """
    # TODO: Implement configurable preprocessing steps
    # TODO: Support different preprocessing strategies
    # TODO: Return processed image
    
    pass


if __name__ == "__main__":
    # TODO: Example usage
    # Load cropped plate images
    # Apply preprocessing
    # Save preprocessed images
    pass

