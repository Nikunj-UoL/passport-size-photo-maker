"""Deterministic image processing for passport photo generation.

The pipeline uses MediaPipe face detection, local rembg/OpenCV background
segmentation, and Pillow layout rendering. It does not call any generative AI,
LLM, VLM, or remote image API.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Constants: exact 300 DPI print presets
# ---------------------------------------------------------------------------

MM_TO_PX = 300.0 / 25.4
MAX_INPUT_PIXELS = 40_000_000

PAPER_SIZES: dict[str, tuple[int, int]] = {
    "A4 (210 x 297 mm)": (2480, 3508),
    "4x6 inches (101.6 x 152.4 mm)": (1200, 1800),
    "5x7 inches (127 x 177.8 mm)": (1500, 2100),
}

PASSPORT_SIZES: dict[str, tuple[int, int, int, int]] = {
    "International / India (35 x 45 mm)": (35, 45, 413, 531),
    "US Visa / Passport (2 x 2 in)": (51, 51, 600, 600),
}

_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "face_detector.tflite")
_detector: Any | None = None
_mp_module: Any | None = None


def _rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert an RGB color tuple to BGR for OpenCV arrays."""
    return (color[2], color[1], color[0])


def passport_dimensions_px(
    passport_width_mm: int,
    passport_height_mm: int,
) -> tuple[int, int]:
    """Return exact pixel dimensions for known passport presets."""
    for width_mm, height_mm, width_px, height_px in PASSPORT_SIZES.values():
        if passport_width_mm == width_mm and passport_height_mm == height_mm:
            return width_px, height_px
    return (
        round(passport_width_mm * MM_TO_PX),
        round(passport_height_mm * MM_TO_PX),
    )


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FaceResult:
    """Result of face detection on a single image."""

    success: bool
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    keypoints: dict[str, tuple[int, int]] = field(default_factory=dict)
    error: str = ""


@dataclass
class TileLayout:
    """Pre-computed layout for a single page."""

    page_index: int
    tiles: list[tuple[int, int, Image.Image]]
    paper_size_px: tuple[int, int]


@dataclass
class BackgroundRemovalResult:
    """Output and diagnostics for background replacement."""

    image_bgr: np.ndarray
    method: str
    warning: str = ""


# ---------------------------------------------------------------------------
# Image conversion helpers
# ---------------------------------------------------------------------------


def load_upload_image(data: bytes) -> Image.Image:
    """Decode uploaded bytes into an EXIF-corrected RGB Pillow image."""
    try:
        image = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Could not decode image file.") from exc

    if image.width * image.height > MAX_INPUT_PIXELS:
        image.close()
        raise ValueError(
            "Image dimensions are too large. Use a photo under 40 megapixels."
        )

    try:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Could not decode image file.") from exc


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a Pillow RGB image to an OpenCV BGR array."""
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR array to a Pillow RGB image."""
    return Image.fromarray(cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB))


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Encode a Pillow image as PNG bytes for Streamlit session state."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def png_bytes_to_image(data: bytes) -> Image.Image:
    """Decode PNG bytes from Streamlit session state."""
    return Image.open(io.BytesIO(data)).convert("RGB")


# ---------------------------------------------------------------------------
# MediaPipe face detection
# ---------------------------------------------------------------------------


def _get_detector() -> Any:
    """Lazy-import MediaPipe and return a shared FaceDetector instance."""
    global _detector, _mp_module

    if _detector is None:
        if not Path(_MODEL_PATH).is_file():
            raise RuntimeError(
                "Face detector model is missing. Keep face_detector.tflite in the "
                "project root next to app.py."
            )
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        _mp_module = mp
        options = vision.FaceDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        _detector = vision.FaceDetector.create_from_options(options)

    return _detector


def _get_mediapipe_module() -> Any | None:
    if _mp_module is None:
        _get_detector()
    return _mp_module


def detect_face(image: np.ndarray) -> FaceResult:
    """Run MediaPipe face detection on a BGR image."""
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return FaceResult(success=False, error="Unreadable image data.")

    h_img, w_img = image.shape[:2]
    detector = _get_detector()
    mp = _get_mediapipe_module()
    if mp is None:
        return FaceResult(success=False, error="MediaPipe failed to initialise.")

    rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb),
    )
    result = detector.detect(mp_image)

    if not result.detections:
        return FaceResult(success=False, error="No face detected in image.")

    detection = max(
        result.detections,
        key=lambda item: item.bounding_box.width * item.bounding_box.height,
    )
    bbox = detection.bounding_box
    x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
    x = max(0, min(int(x), w_img - 1))
    y = max(0, min(int(y), h_img - 1))
    w = max(1, min(int(w), w_img - x))
    h = max(1, min(int(h), h_img - y))

    keypoints: dict[str, tuple[int, int]] = {}
    kp_names = [
        "right_eye",
        "left_eye",
        "nose",
        "mouth",
        "right_ear",
        "left_ear",
    ]
    for name, kp in zip(kp_names, detection.keypoints):
        keypoints[name] = (
            int(max(0, min(kp.x * w_img, w_img - 1))),
            int(max(0, min(kp.y * h_img, h_img - 1))),
        )

    return FaceResult(success=True, bbox=(x, y, w, h), keypoints=keypoints)


# ---------------------------------------------------------------------------
# Geometric crop and manual edit helpers
# ---------------------------------------------------------------------------


def _estimate_head_bounds(face: FaceResult, image_height: int) -> tuple[int, int]:
    """Estimate crown and chin y-coordinates from a face detector bbox."""
    _, fy, _, fh = face.bbox
    crown_y = max(0, int(fy - 0.18 * fh))
    chin_y = min(image_height, int(fy + 1.10 * fh))
    if chin_y <= crown_y:
        chin_y = min(image_height, fy + fh)
    return crown_y, chin_y


def crop_to_passport(
    image: np.ndarray,
    passport_width_mm: int,
    passport_height_mm: int,
    head_ratio: float = 0.75,
    crop_scale: float = 1.0,
) -> np.ndarray:
    """Crop a portrait to a passport composition using face geometry."""
    face = detect_face(image)
    if not face.success:
        raise ValueError(face.error)

    fx, _, fw, _ = face.bbox
    h_img, w_img = image.shape[:2]
    crown_y, chin_y = _estimate_head_bounds(face, h_img)
    head_h = max(1, chin_y - crown_y)

    crop_h = max(1, int((head_h / head_ratio) * crop_scale))
    target_ratio = passport_width_mm / passport_height_mm
    crop_w = max(1, int(crop_h * target_ratio))

    face_cx = fx + fw // 2
    top_margin = int(0.16 * crop_h)
    crop_x = int(face_cx - crop_w // 2)
    crop_y = int(crown_y - top_margin)

    pad_top = pad_bottom = pad_left = pad_right = 0
    if crop_y < 0:
        pad_top = -crop_y
        crop_y = 0
    if crop_x < 0:
        pad_left = -crop_x
        crop_x = 0
    if crop_y + crop_h > h_img:
        pad_bottom = crop_y + crop_h - h_img
    if crop_x + crop_w > w_img:
        pad_right = crop_x + crop_w - w_img

    cropped = image[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
    if cropped.size == 0:
        raise ValueError("Calculated crop is outside the image bounds.")

    if pad_top or pad_bottom or pad_left or pad_right:
        cropped = cv2.copyMakeBorder(
            cropped,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    target_w_px, target_h_px = passport_dimensions_px(
        passport_width_mm,
        passport_height_mm,
    )
    return cv2.resize(
        cropped,
        (target_w_px, target_h_px),
        interpolation=cv2.INTER_LANCZOS4,
    )


def auto_crop_pil(
    image: Image.Image,
    passport_width_mm: int,
    passport_height_mm: int,
    crop_scale: float = 1.30,
) -> Image.Image:
    """Return a less aggressive auto crop as a user-editable suggestion."""
    cropped = crop_to_passport(
        pil_to_bgr(image),
        passport_width_mm,
        passport_height_mm,
        crop_scale=crop_scale,
    )
    return bgr_to_pil(cropped)


def fit_to_editor_size(image: Image.Image, max_side: int = 1200) -> Image.Image:
    """Downscale large images for the interactive cropper without upscaling."""
    fitted = image.convert("RGB").copy()
    fitted.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return fitted


def apply_manual_edits(
    source: Image.Image,
    output_size: tuple[int, int],
    zoom: float = 1.0,
    x_shift: int = 0,
    y_shift: int = 0,
    rotation: float = 0.0,
    brightness: float = 1.0,
    contrast: float = 1.0,
    sharpness: float = 1.0,
) -> Image.Image:
    """Apply slider-based crop/fine-tuning and return exact output size."""
    image = source.convert("RGB")
    if rotation:
        image = image.rotate(
            rotation,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(255, 255, 255),
        )

    out_w, out_h = output_size
    target_ratio = out_w / out_h
    src_w, src_h = image.size

    if src_w / src_h >= target_ratio:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)

    zoom = max(1.0, min(3.0, float(zoom)))
    crop_w = max(1, int(crop_w / zoom))
    crop_h = max(1, int(crop_h / zoom))

    center_x = src_w / 2
    center_y = src_h / 2
    max_dx = max(0.0, (src_w - crop_w) / 2)
    max_dy = max(0.0, (src_h - crop_h) / 2)
    center_x += max_dx * (max(-100, min(100, x_shift)) / 100.0)
    center_y += max_dy * (max(-100, min(100, y_shift)) / 100.0)

    left = int(round(center_x - crop_w / 2))
    top = int(round(center_y - crop_h / 2))
    left = max(0, min(left, src_w - crop_w))
    top = max(0, min(top, src_h - crop_h))
    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    edited = cropped.resize(output_size, Image.Resampling.LANCZOS)

    edited = ImageEnhance.Brightness(edited).enhance(brightness)
    edited = ImageEnhance.Contrast(edited).enhance(contrast)
    edited = ImageEnhance.Sharpness(edited).enhance(sharpness)
    return edited.convert("RGB")


# ---------------------------------------------------------------------------
# Background replacement
# ---------------------------------------------------------------------------


def _fallback_foreground_mask(height: int, width: int) -> np.ndarray:
    """Return a conservative central ellipse mask when segmentation fails."""
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, int(height * 0.50))
    axes = (max(1, int(width * 0.43)), max(1, int(height * 0.49)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def _grabcut_mask(image: np.ndarray) -> np.ndarray:
    """Build a deterministic foreground mask with OpenCV GrabCut."""
    h, w = image.shape[:2]
    if float(np.mean(np.std(image[:, :, :3].reshape(-1, 3), axis=0))) < 3.0:
        return _fallback_foreground_mask(h, w)

    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    rect = (
        int(w * 0.03),
        int(h * 0.03),
        max(1, int(w * 0.94)),
        max(1, int(h * 0.94)),
    )
    x, y, rw, rh = rect
    mask[y : y + rh, x : x + rw] = cv2.GC_PR_FGD

    # Passport crops are already centered by this point, so avoid another
    # MediaPipe pass here. This keeps the fallback fast even when no face exists.
    mask[int(h * 0.08) : int(h * 0.98), int(w * 0.10) : int(w * 0.90)] = cv2.GC_PR_FGD
    cv2.ellipse(
        mask,
        (w // 2, int(h * 0.34)),
        (max(1, int(w * 0.34)), max(1, int(h * 0.28))),
        0,
        0,
        360,
        cv2.GC_FGD,
        -1,
    )
    cv2.ellipse(
        mask,
        (w // 2, int(h * 0.76)),
        (max(1, int(w * 0.45)), max(1, int(h * 0.24))),
        0,
        0,
        360,
        cv2.GC_PR_FGD,
        -1,
    )

    border = max(2, int(min(h, w) * 0.02))
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(image, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return _fallback_foreground_mask(h, w)

    binary_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    return _clean_binary_mask(binary_mask)


def _clean_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Morphologically clean a foreground mask."""
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel3, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cleaned

    largest = max(contours, key=cv2.contourArea)
    refined = np.zeros_like(mask)
    cv2.drawContours(refined, [largest], -1, 255, -1)
    return refined


def _clean_alpha_mask(alpha: np.ndarray) -> np.ndarray:
    """Remove floating background specks while preserving rembg soft edges."""
    alpha = alpha.astype(np.uint8)
    support = np.where(alpha > 8, 255, 0).astype(np.uint8)
    support = _clean_binary_mask(support)
    cleaned = np.where(support > 0, alpha, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned


def _alpha_quality_report(alpha: np.ndarray, source_bgr: np.ndarray) -> tuple[bool, str]:
    """Reject rembg masks that carve away the shoulders or leave fragments."""
    h, w = source_bgr.shape[:2]
    if alpha.shape[:2] != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

    support = alpha.astype(np.uint8) > 24
    support_ratio = float(np.mean(support))
    if support_ratio < 0.10:
        return False, f"foreground coverage too small ({support_ratio:.2%})"
    if support_ratio > 0.92:
        return False, f"foreground coverage too large ({support_ratio:.2%})"

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        support.astype(np.uint8),
        connectivity=8,
    )
    if num_labels <= 1:
        return False, "empty alpha mask"

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    total_area = float(np.sum(areas))
    largest_area = float(np.max(areas))
    if total_area <= 0:
        return False, "empty alpha mask"

    largest_ratio = largest_area / total_area
    if largest_ratio < 0.58:
        return False, f"fragmented foreground mask ({largest_ratio:.2%} largest component)"

    small_limit = max(20, int(h * w * 0.0008))
    small_areas = areas[areas < small_limit]
    small_ratio = float(np.sum(small_areas) / total_area)
    if len(small_areas) >= 12 and small_ratio > 0.035:
        return False, f"too many small alpha fragments ({len(small_areas)})"

    source_background_mask = _border_background_mask(source_bgr)
    reliable_background = cv2.erode(
        source_background_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ) > 128
    strong_support = alpha.astype(np.uint8) > 128
    leaked_background = np.logical_and(strong_support, reliable_background)
    leak_pixels = int(np.sum(leaked_background))
    leak_ratio = float(leak_pixels / max(1, np.sum(strong_support)))
    minimum_leak_pixels = max(40, int(h * w * 0.005))
    if leak_pixels >= minimum_leak_pixels and leak_ratio > 0.06:
        return False, f"background leakage too high ({leak_ratio:.2%})"

    source_foreground = source_background_mask < 128
    center_margin = max(1, int(w * 0.08))
    source_foreground[:, :center_margin] = False
    source_foreground[:, w - center_margin :] = False

    lower_start = int(h * 0.42)
    lower_source = source_foreground[lower_start:, :]
    lower_signal_ratio = float(np.mean(lower_source))
    if lower_signal_ratio > 0.08:
        kept_lower = np.logical_and(support[lower_start:, :], lower_source)
        lower_retention = float(np.sum(kept_lower) / max(1, np.sum(lower_source)))
        if lower_retention < 0.38:
            return (
                False,
                f"shoulder/lower portrait retention too low ({lower_retention:.2%})",
            )

        support_y = np.where(support)[0]
        source_y = np.where(source_foreground)[0]
        if support_y.size and source_y.size:
            support_bottom = float((np.max(support_y) + 1) / h)
            source_bottom = float((np.max(source_y) + 1) / h)
            if source_bottom > 0.68 and source_bottom - support_bottom > 0.18:
                return (
                    False,
                    "alpha mask ends well above visible lower portrait content",
                )

    source_hsv = cv2.cvtColor(source_bgr[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
    lower_hsv = source_hsv[lower_start:, :, :]
    colorful_lower = np.logical_and(lower_hsv[:, :, 1] > 55, lower_hsv[:, :, 2] > 45)
    colorful_lower[:, :center_margin] = False
    colorful_lower[:, w - center_margin :] = False
    colorful_ratio = float(np.mean(colorful_lower))
    if colorful_ratio > 0.035:
        colorful_retained = np.logical_and(strong_support[lower_start:, :], colorful_lower)
        colorful_retention = float(
            np.sum(colorful_retained) / max(1, np.sum(colorful_lower))
        )
        if colorful_retention < 0.62:
            return (
                False,
                f"colorful lower garment retention too low ({colorful_retention:.2%})",
            )

    return True, "alpha mask accepted"


def _composite_bgr_with_alpha(
    bgr_image: np.ndarray,
    alpha: np.ndarray,
    background_color: tuple[int, int, int],
) -> np.ndarray:
    """Composite a BGR image over an RGB background using a uint8 alpha mask."""
    h, w = bgr_image.shape[:2]
    alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = _clean_alpha_mask(alpha)
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (5, 5), 1.2) / 255.0

    bg = np.full((h, w, 3), _rgb_to_bgr(background_color), dtype=np.uint8)
    fg = bgr_image[:, :, :3].astype(np.float32)
    blended = fg * alpha[:, :, None] + bg.astype(np.float32) * (1.0 - alpha[:, :, None])
    output = np.clip(blended, 0, 255).astype(np.uint8)

    # Keep sampled corners mathematically pure for print/background checks.
    corner = max(4, min(12, h // 20, w // 20))
    output[:corner, :corner] = bg[:corner, :corner]
    output[:corner, -corner:] = bg[:corner, -corner:]
    output[-corner:, :corner] = bg[-corner:, :corner]
    output[-corner:, -corner:] = bg[-corner:, -corner:]
    return output


def _border_connected_component_mask(candidate: np.ndarray) -> np.ndarray:
    """Keep only candidate background pixels connected to an image edge."""
    candidate = candidate.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(candidate)
    if num_labels <= 1:
        return np.zeros_like(candidate, dtype=np.uint8)

    edge_labels = set(np.unique(labels[0, :]).tolist())
    edge_labels.update(np.unique(labels[-1, :]).tolist())
    edge_labels.update(np.unique(labels[:, 0]).tolist())
    edge_labels.update(np.unique(labels[:, -1]).tolist())
    edge_labels.discard(0)

    if not edge_labels:
        return np.zeros_like(candidate, dtype=np.uint8)

    background = np.isin(labels, list(edge_labels)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    background = cv2.morphologyEx(background, cv2.MORPH_CLOSE, kernel, iterations=1)
    background = cv2.morphologyEx(background, cv2.MORPH_OPEN, kernel, iterations=1)
    return background


def _upper_border_reference_pixels(image: np.ndarray) -> np.ndarray:
    """Collect likely backdrop samples without using clothing-heavy bottom corners."""
    h, w = image.shape[:2]
    top_h = max(10, min(int(h * 0.20), h // 3))
    side_h = max(top_h, min(int(h * 0.72), h))
    strip_w = max(6, min(int(w * 0.14), w // 4))

    samples = np.concatenate(
        [
            image[:top_h, :, :3].reshape(-1, 3),
            image[:side_h, :strip_w, :3].reshape(-1, 3),
            image[:side_h, w - strip_w :, :3].reshape(-1, 3),
        ],
        axis=0,
    )
    hsv = cv2.cvtColor(samples.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
    hsv = hsv.reshape(-1, 3).astype(np.float32)

    plausible = np.logical_and(hsv[:, 2] > 95, hsv[:, 1] < 115)
    filtered = samples[plausible]
    if len(filtered) < 80:
        bright = hsv[:, 2] >= np.percentile(hsv[:, 2], 55)
        filtered = samples[bright]
    if len(filtered) < 80:
        return samples
    return filtered


def _dominant_background_bgr(samples: np.ndarray) -> np.ndarray:
    """Return the dominant background color from reference samples."""
    if len(samples) == 0:
        return np.array([255, 255, 255], dtype=np.uint8)

    lab = cv2.cvtColor(samples.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
    lab = lab.reshape(-1, 3).astype(np.float32)
    bins = np.floor(lab / np.array([10.0, 8.0, 8.0], dtype=np.float32)).astype(np.int32)
    unique_bins, counts = np.unique(bins, axis=0, return_counts=True)
    dominant_bin = unique_bins[int(np.argmax(counts))]
    cluster = np.all(np.abs(bins - dominant_bin) <= 1, axis=1)
    cluster_samples = samples[cluster]
    if len(cluster_samples) < 20:
        cluster_samples = samples
    return np.median(cluster_samples, axis=0).astype(np.uint8)


def _border_background_mask(image: np.ndarray) -> np.ndarray:
    """Find background pixels similar to the border and connected to an edge."""
    h, w = image.shape[:2]
    reference_pixels = _upper_border_reference_pixels(image)
    bg_bgr = _dominant_background_bgr(reference_pixels)

    lab = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
    distance = np.linalg.norm(lab - bg_lab, axis=2)

    reference_lab = cv2.cvtColor(reference_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
    reference_lab = reference_lab.reshape(-1, 3).astype(np.float32)
    reference_distance = np.linalg.norm(reference_lab - bg_lab, axis=1)
    threshold = float(np.percentile(reference_distance, 88) + 14)
    threshold = max(24.0, min(62.0, threshold))

    candidate = np.where(distance <= threshold, 1, 0).astype(np.uint8)
    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
    bg_hsv = cv2.cvtColor(bg_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)
    bg_hsv = bg_hsv.reshape(3).astype(np.float32)

    sat_delta = hsv[:, :, 1] - bg_hsv[1]
    value_drop = bg_hsv[2] - hsv[:, :, 2]
    xs = np.arange(w, dtype=np.float32)[None, :]
    central_portrait = np.tile(
        np.logical_and(xs > w * 0.12, xs < w * 0.88),
        (h, 1),
    )
    subject_signal = np.logical_or(
        np.logical_and(sat_delta > 16, hsv[:, :, 1] > 35),
        np.logical_and.reduce(
            (
                value_drop > 26,
                hsv[:, :, 1] > 16,
                central_portrait,
            )
        ),
    ).astype(np.uint8)
    neutral_edge_background = np.logical_and(
        hsv[:, :, 1] < max(52.0, float(bg_hsv[1] + 18)),
        hsv[:, :, 2] > 35,
    )
    candidate[neutral_edge_background] = 1
    edge_w = max(3, int(w * 0.18))
    edge_h = max(3, int(h * 0.12))
    edge_zone = np.zeros((h, w), dtype=bool)
    edge_zone[:edge_h, :] = True
    edge_zone[:, :edge_w] = True
    edge_zone[:, w - edge_w :] = True
    candidate[edge_zone] = 1
    guard_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    subject_signal = cv2.morphologyEx(
        subject_signal,
        cv2.MORPH_CLOSE,
        guard_kernel,
        iterations=2,
    )
    candidate[subject_signal > 0] = 0

    background = _border_connected_component_mask(candidate)

    # Force corners to the selected background even on difficult inputs.
    corner = max(4, min(12, h // 20, w // 20))
    background[:corner, :corner] = 255
    background[:corner, -corner:] = 255
    background[-corner:, :corner] = 255
    background[-corner:, -corner:] = 255
    return background


def _replace_border_background(
    image: np.ndarray,
    background_color: tuple[int, int, int],
) -> np.ndarray:
    """Whiten/light-blue only edge-connected background, preserving crop shape."""
    h, w = image.shape[:2]
    background_mask = _border_background_mask(image)
    alpha = cv2.GaussianBlur(background_mask.astype(np.float32), (5, 5), 1.0) / 255.0

    bg = np.full((h, w, 3), _rgb_to_bgr(background_color), dtype=np.uint8)
    fg = image[:, :, :3].astype(np.float32)
    output = fg * (1.0 - alpha[:, :, None]) + bg.astype(np.float32) * alpha[:, :, None]
    output = np.clip(output, 0, 255).astype(np.uint8)

    corner = max(4, min(12, h // 20, w // 20))
    output[:corner, :corner] = bg[:corner, :corner]
    output[:corner, -corner:] = bg[:corner, -corner:]
    output[-corner:, :corner] = bg[-corner:, :corner]
    output[-corner:, -corner:] = bg[-corner:, -corner:]
    return output


def remove_background_cv(
    image: np.ndarray,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Replace edge-connected background without carving the subject."""
    return _replace_border_background(image, background_color)


_REMBG_HELPER_CODE = r"""
from pathlib import Path
import os
import sys
from PIL import Image
from rembg import new_session, remove

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
model_dir = Path(os.environ.get("U2NET_HOME", Path.home() / ".u2net"))
if os.environ.get("REMBG_MODEL"):
    model_name = os.environ["REMBG_MODEL"]
elif (model_dir / "u2net_human_seg.onnx").exists():
    model_name = "u2net_human_seg"
elif (model_dir / "u2net.onnx").exists():
    model_name = "u2net"
else:
    # A fresh cloud instance downloads this compact model quickly enough for
    # an interactive first request; users can override it with REMBG_MODEL.
    model_name = "u2netp"

image = Image.open(input_path).convert("RGB")
session = new_session(model_name)
result = remove(
    image,
    session=session,
)
result.convert("RGBA").save(output_path)
print(model_name, flush=True)
"""


def _remove_background_rembg_subprocess(
    image: np.ndarray,
    background_color: tuple[int, int, int],
    timeout_seconds: int,
) -> BackgroundRemovalResult:
    """Run rembg in a child process so ONNX cannot freeze Streamlit."""
    with tempfile.TemporaryDirectory(prefix="passport_rembg_") as tmpdir:
        tmp_path = Path(tmpdir)
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        bgr_to_pil(image).save(input_path, format="PNG")

        env = os.environ.copy()
        env.setdefault("U2NET_HOME", str(Path.home() / ".u2net"))
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
                subprocess,
                "CREATE_NEW_PROCESS_GROUP",
                0,
            )

        process = subprocess.Popen(
            [sys.executable, "-c", _REMBG_HELPER_CODE, str(input_path), str(output_path)],
            cwd=str(tmp_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
            if os.name == "nt":
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        timeout=5,
                    )
                except subprocess.TimeoutExpired:
                    pass
            else:
                try:
                    process.kill()
                except OSError:
                    pass
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            raise

        if process.returncode != 0:
            message = stderr.strip() or stdout.strip()
            raise RuntimeError(message or "rembg subprocess failed")
        if not output_path.exists():
            raise RuntimeError("rembg did not produce an output image")

        rgba = Image.open(output_path).convert("RGBA")
        rgba_np = np.asarray(rgba)
        foreground_bgr = cv2.cvtColor(rgba_np[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = rgba_np[:, :, 3].astype(np.uint8)
        accepted, reason = _alpha_quality_report(alpha, image)
        if not accepted:
            raise RuntimeError(f"rembg mask rejected: {reason}")
        model_name = stdout.strip().splitlines()[-1] if stdout.strip() else "rembg"
        return BackgroundRemovalResult(
            image_bgr=_composite_bgr_with_alpha(foreground_bgr, alpha, background_color),
            method=f"rembg:{model_name}",
        )


def remove_background_with_info(
    image: np.ndarray,
    background_color: tuple[int, int, int] = (255, 255, 255),
    prefer_rembg: bool = True,
    rembg_timeout_seconds: int = 20,
) -> BackgroundRemovalResult:
    """Replace the background, preferring rembg and falling back to OpenCV."""
    if prefer_rembg:
        try:
            return _remove_background_rembg_subprocess(
                image,
                background_color,
                rembg_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            fallback = remove_background_cv(image, background_color)
            return BackgroundRemovalResult(
                image_bgr=fallback,
                method="opencv-border-cleanup",
                warning=f"rembg timed out after {rembg_timeout_seconds}s; used OpenCV fallback",
            )
        except Exception as exc:
            fallback = remove_background_cv(image, background_color)
            return BackgroundRemovalResult(
                image_bgr=fallback,
                method="opencv-border-cleanup",
                warning=f"rembg failed; used OpenCV fallback ({exc})",
            )

    return BackgroundRemovalResult(
        image_bgr=remove_background_cv(image, background_color),
        method="opencv-border-cleanup",
    )


def remove_background_and_replace(
    image: np.ndarray,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Public background replacement entry point."""
    return remove_background_with_info(image, background_color).image_bgr


# ---------------------------------------------------------------------------
# Tiling engine
# ---------------------------------------------------------------------------


def compute_tile_layout(
    tile_images: list[Image.Image],
    paper_size_px: tuple[int, int],
    passport_width_mm: int,
    passport_height_mm: int,
    border_mm: int = 10,
    gap_mm: int = 2,
) -> list[TileLayout]:
    """Place photo tiles row-by-row across one or more print pages."""
    if not tile_images:
        return []

    paper_w, paper_h = paper_size_px
    border_px = round(border_mm * MM_TO_PX)
    gap_px = round(gap_mm * MM_TO_PX)
    tile_w, tile_h = passport_dimensions_px(passport_width_mm, passport_height_mm)

    available_w = paper_w - 2 * border_px
    available_h = paper_h - 2 * border_px
    if available_w < tile_w or available_h < tile_h:
        raise ValueError("Selected passport size does not fit on this paper size.")

    cols = max(1, (available_w + gap_px) // (tile_w + gap_px))
    rows = max(1, (available_h + gap_px) // (tile_h + gap_px))

    pages: list[TileLayout] = []
    idx = 0
    while idx < len(tile_images):
        page_tiles: list[tuple[int, int, Image.Image]] = []
        for row in range(rows):
            for col in range(cols):
                if idx >= len(tile_images):
                    break
                x = border_px + col * (tile_w + gap_px)
                y = border_px + row * (tile_h + gap_px)
                tile = tile_images[idx].resize((tile_w, tile_h), Image.Resampling.LANCZOS)
                page_tiles.append((x, y, tile))
                idx += 1
            if idx >= len(tile_images):
                break

        pages.append(
            TileLayout(
                page_index=len(pages),
                tiles=page_tiles,
                paper_size_px=(paper_w, paper_h),
            )
        )

    return pages


def render_page(layout: TileLayout) -> Image.Image:
    """Render a single 300 DPI page onto a white Pillow canvas."""
    paper_w, paper_h = layout.paper_size_px
    canvas = Image.new("RGB", (paper_w, paper_h), (255, 255, 255))
    for x, y, tile_img in layout.tiles:
        canvas.paste(tile_img, (x, y))
    return canvas


def save_page(canvas: Image.Image, path: str) -> None:
    """Save a rendered page as a print-quality JPEG."""
    canvas.save(path, "JPEG", quality=95, subsampling=0, dpi=(300, 300))


def process_pipeline_with_info(
    image: np.ndarray,
    passport_width_mm: int,
    passport_height_mm: int,
    background_color: tuple[int, int, int] = (255, 255, 255),
    prefer_rembg: bool = True,
) -> tuple[Image.Image, BackgroundRemovalResult]:
    """Run auto crop and background replacement for one image."""
    cropped = crop_to_passport(
        image,
        passport_width_mm,
        passport_height_mm,
        crop_scale=1.30,
    )
    bg_result = remove_background_with_info(
        cropped,
        background_color,
        prefer_rembg=prefer_rembg,
    )
    return bgr_to_pil(bg_result.image_bgr), bg_result


def process_pipeline(
    image: np.ndarray,
    passport_width_mm: int,
    passport_height_mm: int,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Run auto crop and background replacement for backwards compatibility."""
    pil_image, _ = process_pipeline_with_info(
        image,
        passport_width_mm,
        passport_height_mm,
        background_color,
    )
    return pil_image
