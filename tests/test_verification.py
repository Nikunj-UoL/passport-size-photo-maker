"""Regression tests for passport-photo verification behavior."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from utils.cv_checks import verify_passport_photo


class VerificationTests(unittest.TestCase):
    def test_approved_manual_crop_is_not_rejected_for_head_ratio(self) -> None:
        processed = np.full((531, 413, 3), 255, dtype=np.uint8)
        raw = processed.copy()

        with (
            patch("utils.cv_checks.check_face_centering", return_value=False),
            patch("utils.cv_checks.check_background_purity", return_value=True),
            patch("utils.cv_checks.measure_head_ratio", return_value=0.62),
            patch("utils.cv_checks.crop_to_passport") as recrop,
        ):
            corrected, result = verify_passport_photo(
                processed,
                raw,
                35,
                45,
                max_attempts=1,
                allow_recrop=False,
                enforce_face_centering=False,
                enforce_head_ratio=False,
            )

        self.assertTrue(result.passed, result.errors)
        self.assertTrue(np.array_equal(corrected, processed))
        recrop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
