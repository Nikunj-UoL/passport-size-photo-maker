"""Deterministic quality checks for generated passport photos."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .imaging import (
    _rgb_to_bgr,
    crop_to_passport,
    detect_face,
    remove_background_and_replace,
)


@dataclass
class VerificationResult:
    """Outcome of the verification loop."""

    passed: bool
    centering_ok: bool = True
    background_purity_ok: bool = True
    head_ratio_ok: bool = True
    errors: list[str] = field(default_factory=list)


def check_face_centering(image: np.ndarray, tolerance_pct: float = 2.0) -> bool:
    """Check that the detected face center is near the image center."""
    face = detect_face(image)
    if not face.success:
        return False

    fx, _, fw, _ = face.bbox
    _, image_w = image.shape[:2]
    face_cx = fx + fw // 2
    image_cx = image_w // 2
    max_offset = int(image_w * tolerance_pct / 100.0)
    return abs(face_cx - image_cx) <= max_offset


def check_background_purity(
    image: np.ndarray,
    corner_size: int = 10,
    pure_color: tuple[int, int, int] = (255, 255, 255),
    threshold: int = 8,
) -> bool:
    """Check that each corner block matches the expected BGR background."""
    h, w = image.shape[:2]
    corner_size = max(1, min(corner_size, h, w))
    corners = [
        image[0:corner_size, 0:corner_size, :3],
        image[0:corner_size, w - corner_size : w, :3],
        image[h - corner_size : h, 0:corner_size, :3],
        image[h - corner_size : h, w - corner_size : w, :3],
    ]

    expected = np.array(pure_color, dtype=np.int16)
    for corner in corners:
        delta = np.abs(corner.astype(np.int16) - expected)
        if np.any(delta > threshold):
            return False
    return True


def measure_head_ratio(image: np.ndarray) -> float:
    """Estimate crown-to-chin height as a ratio of canvas height."""
    face = detect_face(image)
    if not face.success:
        return 0.0

    _, fy, _, fh = face.bbox
    image_h = image.shape[0]
    crown_y = max(0, int(fy - 0.15 * fh))
    chin_y = min(image_h, int(fy + 1.05 * fh))
    if chin_y <= crown_y:
        return 0.0
    return (chin_y - crown_y) / image_h


def check_head_ratio(
    image: np.ndarray,
    raw_image: np.ndarray | None = None,
    target_ratio: float = 0.75,
    tolerance: float = 0.06,
) -> bool:
    """Verify the estimated crown-to-chin ratio is near the target."""
    del raw_image
    actual_ratio = measure_head_ratio(image)
    return abs(actual_ratio - target_ratio) <= tolerance


def _foreground_mask_from_background(
    bgr_image: np.ndarray,
    background_color_rgb: tuple[int, int, int],
) -> np.ndarray:
    """Infer a foreground mask from distance to the expected background."""
    background_bgr = np.array(_rgb_to_bgr(background_color_rgb), dtype=np.int16)
    diff = np.abs(bgr_image[:, :, :3].astype(np.int16) - background_bgr)
    distance = np.max(diff, axis=2)
    return np.where(distance > 12, 255, 0).astype(np.uint8)


def apply_alpha_to_background(
    bgr_image: np.ndarray,
    alpha: np.ndarray,
    background_color: tuple[int, int, int],
) -> np.ndarray:
    """Paste a BGR image onto an RGB background color using alpha."""
    h, w = bgr_image.shape[:2]
    bg = np.full((h, w, 3), _rgb_to_bgr(background_color), dtype=np.uint8)
    alpha_norm = alpha.astype(np.float32) / 255.0
    fg = bgr_image[:, :, :3].astype(np.float32)
    blended = fg * alpha_norm[:, :, None] + bg.astype(np.float32) * (1.0 - alpha_norm[:, :, None])
    return np.clip(blended, 0, 255).astype(np.uint8)


def verify_passport_photo(
    processed_img: np.ndarray,
    raw_img: np.ndarray,
    passport_width_mm: int,
    passport_height_mm: int,
    background_color: tuple[int, int, int] = (255, 255, 255),
    max_attempts: int = 3,
    head_ratio: float = 0.75,
    allow_recrop: bool = True,
    enforce_face_centering: bool = True,
    enforce_head_ratio: bool = True,
) -> tuple[np.ndarray, VerificationResult]:
    """Run deterministic verification with up to max_attempts corrections."""
    current = processed_img[:, :, :3].copy()
    crop_scale = 1.0
    last_errors: list[str] = []
    bg_bgr = _rgb_to_bgr(background_color)

    for attempt in range(1, max_attempts + 1):
        center_ok = not enforce_face_centering or check_face_centering(current)
        bg_ok = check_background_purity(current, pure_color=bg_bgr)
        ratio_value = measure_head_ratio(current)
        ratio_ok = not enforce_head_ratio or abs(ratio_value - head_ratio) <= 0.06

        errors: list[str] = []
        if enforce_face_centering and not center_ok:
            errors.append(f"attempt {attempt}: face centering failed")
        if not bg_ok:
            errors.append(f"attempt {attempt}: background purity failed")
        if enforce_head_ratio and not ratio_ok:
            errors.append(
                f"attempt {attempt}: head ratio failed "
                f"({ratio_value:.3f}, target {head_ratio:.3f})"
            )
        last_errors = errors

        if center_ok and bg_ok and ratio_ok:
            return current, VerificationResult(
                passed=True,
                centering_ok=True,
                background_purity_ok=True,
                head_ratio_ok=True,
            )

        if not bg_ok:
            mask = _foreground_mask_from_background(current, background_color)
            kernel = np.ones((2, 2), np.uint8)
            eroded = cv2.erode(mask, kernel, iterations=1)
            current = apply_alpha_to_background(current, eroded, background_color)

        if (
            allow_recrop
            and (
                (enforce_face_centering and not center_ok)
                or (enforce_head_ratio and not ratio_ok)
            )
            and attempt < max_attempts
        ):
            if ratio_value > head_ratio:
                crop_scale *= 1.05
            elif ratio_value > 0:
                crop_scale *= 0.95

            try:
                cropped = crop_to_passport(
                    raw_img,
                    passport_width_mm,
                    passport_height_mm,
                    head_ratio=head_ratio,
                    crop_scale=crop_scale,
                )
                current = remove_background_and_replace(cropped, background_color)
            except ValueError:
                pass

    return current, VerificationResult(
        passed=False,
        centering_ok=(
            True if not enforce_face_centering else check_face_centering(current)
        ),
        background_purity_ok=check_background_purity(current, pure_color=bg_bgr),
        head_ratio_ok=(
            True
            if not enforce_head_ratio
            else check_head_ratio(current, target_ratio=head_ratio)
        ),
        errors=last_errors or ["All verification attempts exhausted."],
    )


def extract_alpha_channel(bgr_image: np.ndarray) -> np.ndarray | None:
    """Backward-compatible helper for older callers."""
    if bgr_image.ndim == 3 and bgr_image.shape[2] == 4:
        return bgr_image[:, :, 3].copy()
    if bgr_image.ndim == 3:
        return _foreground_mask_from_background(bgr_image, (255, 255, 255))
    return None
