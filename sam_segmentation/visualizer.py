"""Visualization utilities for segmentation masks."""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from .utils import load_image_with_exif

if TYPE_CHECKING:
    from .segmenter import SegmentationResult


class OverlayVisualizer:
    """
    Create overlay visualizations of segmentation masks on images.

    Renders colored masks with transparency overlaid on the original image,
    with optional contours and labels.
    """

    DEFAULT_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0), (128, 255, 0), (0, 128, 255),
        (255, 0, 128),
    ]

    def __init__(
        self,
        alpha: float = 0.4,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        contour_thickness: int = 2,
        draw_labels: bool = False,
        mask_threshold: float = 0.5,
    ):
        """
        Initialize overlay visualizer.

        Args:
            alpha: Transparency value for mask overlay (0.0 = transparent, 1.0 = opaque)
            colors: List of RGB color tuples for masks (default: predefined palette)
            contour_thickness: Thickness of contour lines in pixels
            draw_labels: If True, draw text labels on masks
            mask_threshold: Threshold for binarizing masks
        """
        self.alpha = alpha
        self.colors = colors if colors is not None else self.DEFAULT_COLORS
        self.contour_thickness = contour_thickness
        self.draw_labels = draw_labels
        self.mask_threshold = mask_threshold

    def render(
        self,
        image: Union[np.ndarray, Image.Image],
        masks: np.ndarray,
        labels: Optional[List[str]] = None,
        box_prompt: Optional[Sequence[Union[int, float]]] = None,
    ) -> np.ndarray:
        """
        Render overlay visualization on image array.

        Args:
            image: Input image (numpy array or PIL Image)
            masks: Segmentation masks, shape (N, H, W)
            labels: Optional list of text labels to draw on each mask
            box_prompt: Optional visual prompt box (x, y, w, h) in pixels to draw on overlay

        Returns:
            Overlay image as numpy array (H, W, 3)
        """
        # Convert PIL Image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        else:
            image = image.copy()

        # Ensure image is RGB
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        overlay = image.copy()

        # Convert masks to numpy if needed
        if torch.is_tensor(masks):
            masks = masks.cpu().numpy()

        # Render each mask
        for i, mask in enumerate(masks):
            if mask.ndim > 2:
                mask = mask.squeeze()

            mask_bool = mask > self.mask_threshold
            if not mask_bool.any():
                continue

            # Get color for this mask
            color = self.colors[i % len(self.colors)]

            # Apply colored overlay with alpha blending
            overlay[mask_bool] = (
                overlay[mask_bool] * (1 - self.alpha) + np.array(color) * self.alpha
            ).astype(np.uint8)

            # Draw contours
            mask_uint8 = (mask_bool.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, self.contour_thickness)

            # Draw labels if enabled
            if self.draw_labels and labels and i < len(labels):
                # Find centroid for label placement
                y_indices, x_indices = np.where(mask_bool)
                if len(x_indices) > 0:
                    cx = int(np.mean(x_indices))
                    cy = int(np.mean(y_indices))

                    # Draw label with background for visibility
                    label_text = labels[i]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2

                    # Black outline
                    cv2.putText(
                        overlay, label_text, (cx - 15, cy + 5),
                        font, font_scale, (0, 0, 0), thickness + 1
                    )
                    # White text
                    cv2.putText(
                        overlay, label_text, (cx - 15, cy + 5),
                        font, font_scale, (255, 255, 255), thickness
                    )

        # Draw visual prompt box if provided (x, y, w, h) — drawn on top so it's visible
        if box_prompt is not None and len(box_prompt) >= 4:
            x, y, w, h = (int(box_prompt[i]) for i in range(4))
            x0, y0 = x, y
            x1, y1 = x + w, y + h
            # Distinct color for the prompt box (e.g. cyan)
            box_color = (0, 255, 255)  # BGR: cyan
            thickness = 3
            cv2.rectangle(overlay, (x0, y0), (x1, y1), box_color, thickness)

        return overlay

    def save(
        self,
        result: "SegmentationResult",
        output_path: Union[str, Path],
        labels: Optional[List[str]] = None,
        box_prompt: Optional[Sequence[Union[int, float]]] = None,
    ):
        """
        Generate and save overlay visualization for a segmentation result.

        Args:
            result: SegmentationResult object
            output_path: Path to save overlay image
            labels: Optional list of text labels to draw on each mask
            box_prompt: Optional visual prompt box (x, y, w, h) in pixels to draw on overlay
        """
        # Load original image
        image = load_image_with_exif(result.image_path, enable_exif=True)
        image_array = np.array(image.convert("RGB"))

        # Render overlay (optionally with visual prompt box)
        overlay = self.render(image_array, result.masks, labels, box_prompt=box_prompt)

        # Save to file
        output_path = Path(output_path)
        Image.fromarray(overlay).save(output_path)

    def save_from_path(
        self,
        image_path: Union[str, Path],
        masks: np.ndarray,
        output_path: Union[str, Path],
        labels: Optional[List[str]] = None
    ):
        """
        Generate and save overlay visualization from image path and masks.

        Args:
            image_path: Path to original image
            masks: Segmentation masks, shape (N, H, W)
            output_path: Path to save overlay image
            labels: Optional list of text labels to draw on each mask
        """
        # Load image
        image = load_image_with_exif(image_path, enable_exif=True)
        image_array = np.array(image.convert("RGB"))

        # Render overlay
        overlay = self.render(image_array, masks, labels)

        # Save to file
        output_path = Path(output_path)
        Image.fromarray(overlay).save(output_path)
