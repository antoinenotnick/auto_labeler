"""Export utilities for COCO and LabelMe formats."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import numpy as np
import torch
from PIL import Image, ImageOps

from .utils import load_image_with_exif, mask_to_polygon

if TYPE_CHECKING:
    from .segmenter import SegmentationResult


class COCOExporter:
    """
    Export segmentation results to COCO JSON format.

    The COCO format is a cumulative dataset format containing all images
    and annotations in a single JSON file.
    """

    def __init__(
        self,
        category_name: str = "object",
        dataset_name: str = "Segmentation Dataset",
        polygon_tolerance: float = 2.0,
        mask_threshold: float = 0.5,
    ):
        """
        Initialize COCO exporter.

        Args:
            category_name: Name of the object category
            dataset_name: Name of the dataset
            polygon_tolerance: Tolerance for polygon approximation
            mask_threshold: Threshold for binarizing masks
        """
        self.category_name = category_name
        self.dataset_name = dataset_name
        self.polygon_tolerance = polygon_tolerance
        self.mask_threshold = mask_threshold

    def export(
        self,
        results: List["SegmentationResult"],
        output_path: Union[str, Path]
    ) -> dict:
        """
        Export multiple segmentation results to COCO JSON format.

        Args:
            results: List of SegmentationResult objects
            output_path: Path to output JSON file

        Returns:
            The generated COCO dictionary
        """
        # Collect all unique categories from all results
        all_categories = set()
        for result in results:
            if hasattr(result, 'categories') and result.categories:
                # Filter out None and empty strings
                valid_categories = [cat for cat in result.categories if cat and isinstance(cat, str) and cat.strip()]
                all_categories.update(valid_categories)

        # If no categories found, use the default category_name
        if not all_categories:
            all_categories = {self.category_name}

        # Build category list with IDs (filter again to be safe)
        category_list = []
        category_to_id = {}
        for cat_id, cat_name in enumerate(sorted(all_categories), start=1):
            if cat_name and isinstance(cat_name, str) and cat_name.strip():
                category_list.append({
                    "id": cat_id,
                    "name": cat_name,
                    "supercategory": "object"
                })
                category_to_id[cat_name] = cat_id

        coco_output = {
            "info": {
                "description": self.dataset_name,
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [],
            "categories": category_list,
            "images": [],
            "annotations": []
        }

        annotation_id = 1

        for img_id, result in enumerate(results, start=1):
            # Add image info
            coco_output["images"].append({
                "id": img_id,
                "file_name": result.image_path.name,
                "width": result.image_size[0],
                "height": result.image_size[1]
            })

            # Convert masks to numpy if needed
            masks = result.masks
            if torch.is_tensor(masks):
                masks = masks.cpu().numpy()

            # Get categories for this result
            result_categories = result.categories if hasattr(result, 'categories') and result.categories else [self.category_name] * len(masks)

            # Process each mask
            for i, mask in enumerate(masks):
                polygons = mask_to_polygon(mask, self.polygon_tolerance, self.mask_threshold)
                if not polygons:
                    continue

                # Calculate bounding box and area
                if mask.ndim > 2:
                    mask = mask.squeeze()
                mask_bool = mask > self.mask_threshold
                if not mask_bool.any():
                    continue

                y_indices, x_indices = np.where(mask_bool)
                x_min, x_max = int(x_indices.min()), int(x_indices.max())
                y_min, y_max = int(y_indices.min()), int(y_indices.max())
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                area = int(mask_bool.sum())

                # Flatten polygons for COCO format
                segmentation = []
                for polygon in polygons:
                    flat_polygon = [coord for point in polygon for coord in point]
                    if len(flat_polygon) >= 6:
                        segmentation.append(flat_polygon)

                if not segmentation:
                    continue

                # Get the category for this mask (ensure it's valid)
                mask_category = result_categories[i] if i < len(result_categories) else self.category_name

                # Handle invalid categories (None, empty string, etc.)
                if not mask_category or not isinstance(mask_category, str) or not mask_category.strip():
                    mask_category = self.category_name

                # Get category_id, default to 1 if not found
                category_id = category_to_id.get(mask_category, 1)

                # Create annotation
                annotation = {
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "attributes": {"fdi_number": ""}
                }

                # Add confidence score if available
                if result.scores is not None and i < len(result.scores):
                    score = result.scores[i]
                    if torch.is_tensor(score):
                        score = score.item()
                    annotation["score"] = float(score)

                coco_output["annotations"].append(annotation)
                annotation_id += 1

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(coco_output, f, indent=2)

        print(f"COCO annotations saved to: {output_path}")
        return coco_output


class LabelMeExporter:
    """
    Export segmentation results to LabelMe JSON format.

    LabelMe format creates one JSON file per image containing polygon
    annotations and metadata.
    """

    def __init__(
        self,
        category_name: str = "object",
        polygon_tolerance: float = 2.0,
        mask_threshold: float = 0.5,
    ):
        """
        Initialize LabelMe exporter.

        Args:
            category_name: Name of the object category
            polygon_tolerance: Tolerance for polygon approximation
            mask_threshold: Threshold for binarizing masks
        """
        self.category_name = category_name
        self.polygon_tolerance = polygon_tolerance
        self.mask_threshold = mask_threshold

    def export(
        self,
        result: "SegmentationResult",
        output_path: Union[str, Path] = None
    ) -> dict:
        """
        Export a single segmentation result to LabelMe JSON format.

        Args:
            result: SegmentationResult object
            output_path: Path to output JSON file (default: same as image with .json extension)

        Returns:
            The generated LabelMe dictionary
        """
        if output_path is None:
            output_path = result.image_path.with_suffix(".json")
        else:
            output_path = Path(output_path)

        labelme_output = {
            "version": "5.0.0",
            "flags": {},
            "shapes": [],
            "imagePath": result.image_path.name,
            "imageData": None,
            "imageHeight": result.image_size[1],
            "imageWidth": result.image_size[0]
        }

        # Convert masks to numpy if needed
        masks = result.masks
        if torch.is_tensor(masks):
            masks = masks.cpu().numpy()

        # Get categories for this result
        result_categories = result.categories if hasattr(result, 'categories') and result.categories else [self.category_name] * len(masks)

        # Process each mask
        for i, mask in enumerate(masks):
            polygons = mask_to_polygon(mask, self.polygon_tolerance, self.mask_threshold)
            for polygon in polygons:
                if len(polygon) < 3:
                    continue

                # Get the category for this mask (ensure it's valid)
                mask_category = result_categories[i] if i < len(result_categories) else self.category_name

                # Handle invalid categories (None, empty string, etc.)
                if not mask_category or not isinstance(mask_category, str) or not mask_category.strip():
                    mask_category = self.category_name

                shape = {
                    "label": mask_category,
                    "points": polygon,
                    "group_id": i + 1,
                    "shape_type": "polygon",
                    "flags": {}
                }

                # Add confidence score as description
                if result.scores is not None and i < len(result.scores):
                    score = result.scores[i]
                    if torch.is_tensor(score):
                        score = score.item()
                    shape["description"] = f"score: {score:.3f}"

                labelme_output["shapes"].append(shape)

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(labelme_output, f, indent=2)

        print(f"LabelMe annotation saved to: {output_path}")
        return labelme_output
