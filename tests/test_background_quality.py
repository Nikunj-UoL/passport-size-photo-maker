"""Regression tests for portrait background-mask quality."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from utils.imaging import _alpha_quality_report


def portrait_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Return a simple person on a neutral wall and its clean alpha mask."""
    height, width = 531, 413
    wall_bgr = (184, 194, 204)
    shirt_bgr = (170, 105, 145)
    skin_bgr = (125, 165, 205)

    source = np.full((height, width, 3), wall_bgr, dtype=np.uint8)
    alpha = np.zeros((height, width), dtype=np.uint8)

    cv2.ellipse(alpha, (width // 2, 180), (92, 142), 0, 0, 360, 255, -1)
    shoulders = np.array(
        [[32, height - 1], [78, 342], [145, 294], [268, 294], [335, 342], [381, height - 1]],
        dtype=np.int32,
    )
    cv2.fillPoly(alpha, [shoulders], 255)

    source[alpha > 0] = shirt_bgr
    face = np.zeros_like(alpha)
    cv2.ellipse(face, (width // 2, 180), (92, 142), 0, 0, 360, 255, -1)
    source[face > 0] = skin_bgr
    return source, alpha


class BackgroundQualityTests(unittest.TestCase):
    def test_clean_portrait_mask_is_accepted(self) -> None:
        source, alpha = portrait_fixture()
        accepted, reason = _alpha_quality_report(alpha, source)
        self.assertTrue(accepted, reason)

    def test_wall_blobs_attached_to_shoulders_are_rejected(self) -> None:
        source, alpha = portrait_fixture()
        leaky_alpha = alpha.copy()

        # These neutral-wall islands touch the shoulder silhouette, matching the
        # jagged beige chunks seen in the generated passport sheet.
        cv2.rectangle(leaky_alpha, (8, 300), (105, 455), 255, -1)
        cv2.rectangle(leaky_alpha, (307, 310), (404, 450), 255, -1)

        accepted, reason = _alpha_quality_report(leaky_alpha, source)
        self.assertFalse(accepted, reason)
        self.assertIn("background leakage", reason)


if __name__ == "__main__":
    unittest.main()
