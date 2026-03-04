"""Main segmentation pipeline and result data structures."""

import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .exporters import COCOExporter, LabelMeExporter
from .utils import collect_images, load_image_with_exif
from .visualizer import OverlayVisualizer

# Suppress SAM3 deprecation warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module="sam3.model_builder"
)


@dataclass
class SegmentationResult:
    """
    Container for segmentation results from a single image.

    Attributes:
        image_path: Path to the source image
        image_size: Tuple of (width, height) in pixels
        masks: Numpy array of segmentation masks, shape (N, H, W)
        scores: Numpy array of confidence scores, shape (N,)
        categories: List of category labels for each mask (e.g., ["bus", "bus", "car"])
        timestamp: When the segmentation was performed
    """

    image_path: Path
    image_size: Tuple[int, int]
    masks: np.ndarray
    scores: np.ndarray
    categories: List[str]
    timestamp: datetime

    @property
    def num_detections(self) -> int:
        """Return the number of detected objects."""
        return len(self.masks) if self.masks is not None else 0

    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the result
        """
        return {
            "image_path": str(self.image_path),
            "image_size": self.image_size,
            "num_detections": self.num_detections,
            "categories": self.categories,
            "timestamp": self.timestamp.isoformat(),
        }


class SAMSegmenter:
    """
    Main segmentation pipeline for detecting objects in images using SAM-3.

    This class provides a high-level interface for segmenting objects using
    the SAM-3 model, with configurable export and visualization options.

    Example:
        >>> segmenter = SAMSegmenter(text_prompt="car")
        >>> results = segmenter.process_directory("./images")
        >>> print(f"Processed {len(results)} images")
    """

    def __init__(
        self,
        # Model configuration
        model_path: str = r"C:\Users\h02317\sam3",
        text_prompt: Union[str, List[str]] = "object",
        box_prompt: Optional[Union[Tuple[int, int, int, int], List[int]]] = None,  # (x, y, w, h) in pixels

        # Processing options
        mask_threshold: float = 0.5,
        enable_exif_transpose: bool = True,

        # Export configuration
        export_format: Union[str, List[str]] = "coco",
        save_overlay: bool = True,

        # Input/Output configuration
        images_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        overlay_subdir: str = "segmented_images",

        # Annotation metadata
        category_name: str = "object",
        dataset_name: str = "Segmentation Dataset",
        labels: Optional[List[str]] = None,

        # Visualization options
        overlay_alpha: float = 0.4,
        overlay_colors: Optional[List[Tuple[int, int, int]]] = None,
        contour_thickness: int = 2,

        # Polygon conversion
        polygon_tolerance: float = 2.0,

        # Image filtering
        supported_extensions: Optional[List[str]] = None,
    ):
        """
        Initialize SAMSegmenter.

        Args:
            model_path: Path to SAM3 installation
            text_prompt: Text prompt(s) for segmentation (e.g., "car" or ["bus", "car"]). 
                         Set to None if using only visual (box) prompt.
            box_prompt: Visual prompt as bounding box in pixel coordinates (x, y, w, h):
                        top-left x, top-left y, width, height. Can be a tuple or list.
                        If provided, SAM3 will use this as an exemplar to find all similar
                        objects in the image. Can be combined with text_prompt. 
            mask_threshold: Threshold for binarizing masks (0.0-1.0)
            enable_exif_transpose: Auto-correct image orientation from EXIF
            export_format: Export format(s) - "labelme", "coco", or ["labelme", "coco"]
            save_overlay: Save overlay visualizations
            images_dir: Default directory containing images to process
            output_dir: Base output directory (default: same as images_dir)
            overlay_subdir: Subdirectory name for overlay images
            category_name: Object category name for annotations (used for visual prompt masks)
            dataset_name: Dataset name for COCO export
            labels: Optional list of labels to use for annotations (overrides text prompts for category names)
            overlay_alpha: Transparency for mask overlay (0.0-1.0)
            overlay_colors: Custom color palette for masks
            contour_thickness: Thickness of contour lines in pixels
            polygon_tolerance: Tolerance for polygon approximation
            supported_extensions: List of image extensions to process
        """
        # Store configuration
        self.model_path = model_path
        self.text_prompt = text_prompt
        # Convert box_prompt to tuple if it's a list
        if box_prompt is not None:
            self.box_prompt = tuple(box_prompt) if isinstance(box_prompt, list) else box_prompt
        else:
            self.box_prompt = None
        self.category_name = category_name
        self.mask_threshold = mask_threshold
        self.enable_exif_transpose = enable_exif_transpose
        self.save_overlay = save_overlay
        self.images_dir = Path(images_dir) if images_dir else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.overlay_subdir = overlay_subdir
        self.supported_extensions = supported_extensions
        # Optional per-prompt labels used for annotations (falls back to prompt text)
        self.labels: Optional[List[str]] = labels

        # Parse export format
        if isinstance(export_format, str):
            export_formats = [export_format.lower()]
        else:
            export_formats = [fmt.lower() for fmt in export_format]

        self.export_labelme = "labelme" in export_formats
        self.export_coco = "coco" in export_formats

        # Initialize exporters
        self.coco_exporter = COCOExporter(
            category_name=category_name,
            dataset_name=dataset_name,
            polygon_tolerance=polygon_tolerance,
            mask_threshold=mask_threshold,
        ) if self.export_coco else None

        self.labelme_exporter = LabelMeExporter(
            category_name=category_name,
            polygon_tolerance=polygon_tolerance,
            mask_threshold=mask_threshold,
        ) if self.export_labelme else None

        self.visualizer = OverlayVisualizer(
            alpha=overlay_alpha,
            colors=overlay_colors,
            contour_thickness=contour_thickness,
            mask_threshold=mask_threshold,
        ) if save_overlay else None

        # Model will be loaded lazily on first use
        self._model = None
        self._processor = None

    def _load_model(self):
        """Load SAM3 model and processor (lazy loading)."""
        if self._model is not None:
            return

        # Add SAM3 to Python path
        if self.model_path not in sys.path:
            sys.path.insert(0, self.model_path)

        # Import SAM3 (after adding to path)
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        print("Loading SAM3 model...")
        self._model = build_sam3_image_model()
        self._processor = Sam3Processor(self._model)
        print("Model loaded.")

    def _segment_image(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Run segmentation inference on an image.

        Args:
            image: PIL Image object

        Returns:
            Tuple of (masks, scores, categories) where:
            - masks: numpy array of masks, shape (N, H, W)
            - scores: numpy array of confidence scores, shape (N,)
            - categories: list of category labels for each mask
        """
        # Ensure model is loaded
        self._load_model()

        # Prepare prompts as a list
        if isinstance(self.text_prompt, str):
            prompts = [self.text_prompt] if self.text_prompt else []
        else:
            prompts = self.text_prompt if self.text_prompt else []

        # Optional labels aligned with prompts
        label_overrides = None
        if self.labels:
            if isinstance(self.labels, list) and len(self.labels) == len(prompts):
                label_overrides = self.labels

        # Run inference for each prompt and combine results
        all_masks = []
        all_scores = []
        all_categories = []

        inference_state = self._processor.set_image(image)

        # If a box_prompt is provided, use SAM3 visual (geometric) prompt
        # box_prompt is (x, y, w, h): top-left corner and size in pixels
        if self.box_prompt is not None:
            x, y, w, h = self.box_prompt

            # Normalize to [0, 1] and convert to [cx, cy, w, h] for SAM3
            img_w, img_h = image.size
            cx = (x + w / 2.0) / img_w
            cy = (y + h / 2.0) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            box_norm = [cx, cy, w_norm, h_norm]

            # Positive visual prompt
            output = self._processor.add_geometric_prompt(
                box=box_norm,
                label=True,
                state=inference_state,
            )

            if output["masks"] is not None and len(output["masks"]) > 0:
                masks_cpu = output["masks"].cpu() if hasattr(output["masks"], "cpu") else output["masks"]
                scores_cpu = output["scores"].cpu() if hasattr(output["scores"], "cpu") else output["scores"]
                
                # Convert to numpy and reshape from [N, 1, H, W] to [N, H, W]
                if hasattr(masks_cpu, "numpy"):
                    masks_np = masks_cpu.numpy()
                else:
                    masks_np = np.array(masks_cpu)
                # Squeeze out the channel dimension if present
                if len(masks_np.shape) == 4 and masks_np.shape[1] == 1:
                    masks_np = masks_np.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
                
                if hasattr(scores_cpu, "numpy"):
                    scores_np = scores_cpu.numpy()
                else:
                    scores_np = np.array(scores_cpu)

                num_masks = len(masks_np)
                # Use category_name for visual prompt masks
                label = self.category_name
                categories = [label] * num_masks

                all_masks.append(masks_np)
                all_scores.append(scores_np)
                all_categories.extend(categories)

        # Use text prompt (only if text_prompt is provided and not None)
        if prompts:
            for idx, prompt in enumerate(prompts):
                output = self._processor.set_text_prompt(
                    state=inference_state,
                    prompt=prompt
                )

                if output["masks"] is not None and len(output["masks"]) > 0:
                    # Move tensors from GPU to CPU before converting to numpy
                    masks_cpu = output["masks"].cpu() if hasattr(output["masks"], 'cpu') else output["masks"]
                    scores_cpu = output["scores"].cpu() if hasattr(output["scores"], 'cpu') else output["scores"]
                    
                    # Convert to numpy and reshape from [N, 1, H, W] to [N, H, W]
                    if hasattr(masks_cpu, "numpy"):
                        masks_np = masks_cpu.numpy()
                    else:
                        masks_np = np.array(masks_cpu)
                    # Squeeze out the channel dimension if present
                    if len(masks_np.shape) == 4 and masks_np.shape[1] == 1:
                        masks_np = masks_np.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
                    
                    if hasattr(scores_cpu, "numpy"):
                        scores_np = scores_cpu.numpy()
                    else:
                        scores_np = np.array(scores_cpu)

                    # Track category for each mask from this prompt
                    num_masks = len(masks_np)
                    label = label_overrides[idx] if label_overrides else prompt
                    categories = [label] * num_masks

                    all_masks.append(masks_np)
                    all_scores.append(scores_np)
                    all_categories.extend(categories)

        # Combine masks and scores from all prompts
        if all_masks:
            masks = np.concatenate(all_masks, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            categories = all_categories
        else:
            # No detections from any prompt
            masks = np.array([])
            scores = np.array([])
            categories = []

        return masks, scores, categories

    def process_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ) -> SegmentationResult:
        """
        Process a single image and optionally export results.

        Args:
            image_path: Path to input image
            output_dir: Output directory (overrides constructor setting)

        Returns:
            SegmentationResult object
        """
        image_path = Path(image_path)

        # Determine output directory
        if output_dir:
            out_dir = Path(output_dir)
        elif self.output_dir:
            out_dir = self.output_dir
        else:
            out_dir = image_path.parent

        # Load image
        image = load_image_with_exif(image_path, self.enable_exif_transpose)

        # Run segmentation
        masks, scores, categories = self._segment_image(image)

        # Create result object
        result = SegmentationResult(
            image_path=image_path,
            image_size=(image.width, image.height),
            masks=masks,
            scores=scores,
            categories=categories,
            timestamp=datetime.now()
        )

        # Export LabelMe annotation
        if self.labelme_exporter:
            labelme_path = out_dir / f"{image_path.stem}_labelme.json"
            self.labelme_exporter.export(result, labelme_path)

        # Save overlay visualization (include box_prompt if using visual prompt)
        if self.visualizer:
            overlay_dir = out_dir / self.overlay_subdir
            overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = overlay_dir / f"{image_path.stem}_overlay.png"
            self.visualizer.save(
                result, overlay_path,
                labels=result.categories,
                box_prompt=self.box_prompt,
            )

        return result

    def process_directory(
        self,
        images_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[SegmentationResult]:
        """
        Process all images in a directory.

        Args:
            images_dir: Directory containing images (default: use class's images_dir)
            output_dir: Output directory (default: same as images_dir)

        Returns:
            List of SegmentationResult objects
        """
        # Use class's images_dir if not provided
        if images_dir is None:
            if self.images_dir is None:
                raise ValueError("No images_dir provided. Either pass it to process_directory() or set it in the constructor.")
            images_dir = self.images_dir
        else:
            images_dir = Path(images_dir)

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # Determine output directory
        if output_dir:
            out_dir = Path(output_dir)
        elif self.output_dir:
            out_dir = self.output_dir
        else:
            out_dir = images_dir

        out_dir.mkdir(parents=True, exist_ok=True)

        # Find images
        image_paths = collect_images(images_dir, self.supported_extensions)
        if not image_paths:
            print(f"No images found in {images_dir}")
            return []

        # Load model first (before starting timer)
        self._load_model()

        print(f"Found {len(image_paths)} images. Processing...")

        # Process each image (start timing after model load)
        results = []
        start_time = perf_counter()

        for idx, image_path in enumerate(image_paths, start=1):
            print(f"Processing {idx}/{len(image_paths)}: {image_path.name}")
            try:
                result = self.process_image(image_path, out_dir)
                results.append(result)
            except Exception as e:
                print(f"  Error processing {image_path.name}: {e}")
                continue

        # End timing after image processing
        total_time = perf_counter() - start_time

        # Export COCO annotations
        if self.coco_exporter and results:
            coco_path = out_dir / "annotations_coco.json"
            self.coco_exporter.export(results, coco_path)

        # Print summary
        processed_count = len(results)
        avg_time = total_time / processed_count if processed_count else 0
        print(f"\nDone. Processed {processed_count} images in {total_time:.2f}s "
              f"(avg {avg_time:.2f}s per image)")
        print(f"Outputs saved to: {out_dir}")

        return results

    def export_results(
        self,
        results: Union[SegmentationResult, List[SegmentationResult]],
        output_dir: Union[str, Path]
    ) -> Dict[str, Path]:
        """
        Export results to configured formats.

        Args:
            results: Single result or list of results
            output_dir: Directory for output files

        Returns:
            Dictionary mapping format name to output file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure results is a list
        if isinstance(results, SegmentationResult):
            results = [results]

        exported_files = {}

        # Export COCO
        if self.coco_exporter:
            coco_path = output_dir / "annotations_coco.json"
            self.coco_exporter.export(results, coco_path)
            exported_files["coco"] = coco_path

        # Export LabelMe (one per image)
        if self.labelme_exporter:
            labelme_paths = []
            for result in results:
                labelme_path = output_dir / f"{result.image_path.stem}_labelme.json"
                self.labelme_exporter.export(result, labelme_path)
                labelme_paths.append(labelme_path)
            exported_files["labelme"] = labelme_paths

        # Export overlays
        if self.visualizer:
            overlay_dir = output_dir / self.overlay_subdir
            overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay_paths = []
            for result in results:
                overlay_path = overlay_dir / f"{result.image_path.stem}_overlay.png"
                self.visualizer.save(
                    result, overlay_path,
                    labels=result.categories,
                    box_prompt=self.box_prompt,
                )
                overlay_paths.append(overlay_path)
            exported_files["overlay"] = overlay_paths

        return exported_files
