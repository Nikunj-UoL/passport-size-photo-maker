"""Regression checks for Streamlit Community Cloud dependencies."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def manifest_entries(filename: str) -> list[str]:
    return [
        line.strip()
        for line in (ROOT / filename).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


class DeploymentManifestTests(unittest.TestCase):
    def test_mediapipe_gles_runtime_is_installed(self) -> None:
        """MediaPipe's Linux shared library needs libGLESv2.so.2 on upload."""
        self.assertIn("libgles2", manifest_entries("packages.txt"))

    def test_only_one_opencv_distribution_is_declared(self) -> None:
        opencv_distributions = [
            requirement
            for requirement in manifest_entries("requirements.txt")
            if requirement.lower().startswith("opencv-")
        ]
        self.assertEqual(["opencv-contrib-python==4.13.0.92"], opencv_distributions)


if __name__ == "__main__":
    unittest.main()
