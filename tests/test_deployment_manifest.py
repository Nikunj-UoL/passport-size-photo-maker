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
    def test_mediapipe_gpu_runtimes_are_installed(self) -> None:
        """MediaPipe's Linux shared library needs GLES and EGL on upload."""
        packages = manifest_entries("packages.txt")
        required_packages = {
            "libGLESv2.so.2": "libgles2",
            "libEGL.so.1": "libegl1",
        }
        for library, package in required_packages.items():
            with self.subTest(library=library):
                self.assertIn(package, packages)

    def test_only_one_opencv_distribution_is_declared(self) -> None:
        opencv_distributions = [
            requirement
            for requirement in manifest_entries("requirements.txt")
            if requirement.lower().startswith("opencv-")
        ]
        self.assertEqual(["opencv-contrib-python==4.13.0.92"], opencv_distributions)


if __name__ == "__main__":
    unittest.main()
