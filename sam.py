"""
Simple wrapper script for backward compatibility.

This script provides the same command-line interface as the original sam.py,
but now uses the reusable SAMSegmenter class under the hood.

To use the class in your own code, import it directly:
    from sam_segmentation import SAMSegmenter

    segmenter = SAMSegmenter(text_prompt="car")
    results = segmenter.process_directory("./images")
"""

from pathlib import Path

from sam_segmentation import SAMSegmenter


def main():
    """
    Process every image in ./images:
    - Save per-image LabelMe JSON alongside the image
    - Save a cumulative COCO JSON in ./images
    - Save overlay images to ./images/segmented_images
    """
    base_dir = Path(__file__).parent
    images_dir = base_dir / "images"

    # Initialize segmenter with default settings
    segmenter = SAMSegmenter(
        text_prompt=["Segment each tooth accurately"],  # Prompts the model uses
        labels=["tooth"],  # Labels applied in COCO/LabelMe annotations
        export_format=["coco"],  # Export both formats
        save_overlay=True,
        images_dir=images_dir,  # Set default images directory
        output_dir=images_dir,  # Set output directory
    )

    # Process images (uses class's images_dir)
    segmenter.process_directory()


if __name__ == "__main__":
    main()
