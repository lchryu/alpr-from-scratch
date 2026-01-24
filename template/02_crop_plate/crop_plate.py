"""
Module 02: Crop License Plate Regions

This module extracts individual license plate regions from detected bounding boxes.
Each detected plate is cropped from the original image for further processing.

TODO: Implement cropping logic
- Load image and bounding box coordinates
- Extract each detected region
- Apply padding if needed
- Save or return cropped images
"""

import cv2
from typing import List, Tuple, Optional


def crop_plate_regions(
    image_path: str,
    bounding_boxes: List[Tuple[int, int, int, int]],
    padding: int = 0,
    output_dir: Optional[str] = None
) -> List:
    """
    Crop license plate regions from an image based on bounding box coordinates.
    
    Args:
        image_path: Path to the input image
        bounding_boxes: List of bounding boxes as (x1, y1, x2, y2) tuples
        padding: Optional padding to add around each bounding box
        output_dir: Optional directory to save cropped images
        
    Returns:
        List of cropped plate images (numpy arrays or image objects)
    """
    # TODO: Load the image
    # TODO: Iterate through bounding boxes
    # TODO: Crop each region with padding
    # TODO: Handle edge cases (boxes outside image boundaries)
    # TODO: Save cropped images if output_dir is provided
    # TODO: Return list of cropped images
    
    pass


if __name__ == "__main__":
    # TODO: Example usage
    # Load detection results from Module 01
    # Crop plates
    # Save results
    pass

