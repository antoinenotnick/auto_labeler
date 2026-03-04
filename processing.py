from pathlib import Path
from sam_segmentation import (
    SAMSegmenter,
    COCOExporter,
)

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

# Visual prompt: bounding box in pixel coordinates (x, y, w, h)
# Example: [100, 100, 200, 300] = top-left (100,100), width 200, height 300
box_prompt = [1000.0, 1550.0, 3000.0, 850.0]  # (x, y, w, h) — replace with your box

def single_image_processing():
    """Process a single image."""
    print("\n" + "=" * 60)
    print("Single Image Processing")
    print("=" * 60)

    segmenter = SAMSegmenter(
        text_prompt = None,
        box_prompt=box_prompt,
        category_name="pole"
    )
    result = segmenter.process_image(SCRIPT_DIR / "images" / "pole-down.jpg") # sample, change if needed

    print(f"Image: {result.image_path.name}")
    print(f"Size: {result.image_size[0]}x{result.image_size[1]}")
    print(f"Detections: {result.num_detections}")
    print(f"Scores: {result.scores}")

    return result


def simple_batch_processing():
    """Simple batch processing with default settings."""
    print("=" * 60)
    print("Simple Batch Processing")
    print("=" * 60)

    segmenter = SAMSegmenter(
        text_prompt = None,
        box_prompt=box_prompt,
        category_name="pole"
    )
    results = segmenter.process_directory(SCRIPT_DIR / "images")

    print(f"\nProcessed {len(results)} images")
    for result in results:
        print(f"  - {result.image_path.name}: {result.num_detections} detections")

    return results 


def export_to_coco(results):
    exporter = COCOExporter(
        category_name="car",
        dataset_name="My Dataset"
    )
    exporter.export(results, "annotations.json")


def custom_export(): # Segments and exports to COCO json
    """Process without exports, then export with custom settings."""
    print("\n" + "=" * 60)
    print("Custom Export")
    print("=" * 60)

    # Process without automatic exports (pass empty list)
    segmenter = SAMSegmenter(
        text_prompt="pole",
        export_format=[],  # No automatic exports
        save_overlay=False,
    )

    results = segmenter.process_directory(SCRIPT_DIR / "images")

    # Custom COCO export with different settings
    exporter = COCOExporter(
        category_name="pole",
        dataset_name="pole dataset",
        polygon_tolerance=1.5,  # More precise polygons
    )
    exporter.export(results, SCRIPT_DIR / "output" / "custom_coco.json")
    print("Custom COCO export complete")


def main():
# Uncomment any function that you would like to use #

    # single_image_processing()
    simple_batch_processing()
    # export_to_coco()
    # custom_export()


if __name__ == "__main__":
    main()