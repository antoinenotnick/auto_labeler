"""
SAM Segmentation - Reusable object segmentation pipeline.

This package provides a configurable, class-based interface for segmenting
objects in images using the SAM-3 model.

Main Components:
    - SAMSegmenter: Main segmentation pipeline
    - SegmentationResult: Data container for results
    - COCOExporter: Export to COCO JSON format
    - LabelMeExporter: Export to LabelMe JSON format
    - OverlayVisualizer: Create overlay visualizations

Example Usage:
    >>> from sam_segmentation import SAMSegmenter
    >>> segmenter = SAMSegmenter(text_prompt="car")
    >>> results = segmenter.process_directory("./images")
    >>> print(f"Processed {len(results)} images")
"""

from .exporters import COCOExporter, LabelMeExporter
from .segmenter import SegmentationResult, SAMSegmenter
from .utils import mask_to_polygon
from .visualizer import OverlayVisualizer

__version__ = "1.0.0"

__all__ = [
    # Main classes
    "SAMSegmenter",
    "SegmentationResult",

    # Exporters
    "COCOExporter",
    "LabelMeExporter",

    # Visualization
    "OverlayVisualizer",

    # Utilities
    "mask_to_polygon",
]
