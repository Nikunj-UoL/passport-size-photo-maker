"""End-to-end smoke test for the Streamlit passport photo app.

The test starts Streamlit, opens the UI in Playwright, uploads a real photo,
generates a sheet, and verifies the result is visible. It does not require a
pre-existing server.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlparse
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from utils.imaging import MM_TO_PX, PAPER_SIZES, passport_dimensions_px


ROOT = Path(__file__).resolve().parent
PHOTO_ENV = os.environ.get("PASSPORT_TEST_PHOTO")
PHOTO = Path(PHOTO_ENV).expanduser() if PHOTO_ENV else None
URL = os.environ.get("PASSPORT_TEST_URL", "http://localhost:8501")
PORT = int(os.environ.get("PASSPORT_TEST_PORT", urlparse(URL).port or 8501))
HEALTH_URL = f"{URL}/_stcore/health"
ARTIFACT_DIR = ROOT / "output" / "playwright"


def validate_downloaded_sheet(path: Path, source_photo: Path) -> int:
    """Catch the bad generated-output case where rembg leaves only a head."""
    image = Image.open(path).convert("RGB")
    expected_size = PAPER_SIZES["A4 (210 x 297 mm)"]
    if image.size != expected_size:
        raise RuntimeError(f"unexpected downloaded sheet size: {image.size}")

    tile_w, tile_h = passport_dimensions_px(35, 45)
    border = round(10 * MM_TO_PX)
    tile = image.crop((border, border, border + tile_w, border + tile_h))
    tile_array = np.asarray(tile)

    not_background = np.any(tile_array < 245, axis=2)
    lower_start = int(tile_h * 0.48)
    lower_not_background = not_background[lower_start:, :]
    lower_pixels = int(np.sum(lower_not_background))
    if lower_pixels < 3500:
        raise RuntimeError(
            "downloaded sheet appears to have lost the lower portrait/shoulders"
        )

    torso_band = not_background[int(tile_h * 0.55) : int(tile_h * 0.92), :]
    hole_pixels = 0
    span_pixels = 0
    wide_rows = 0
    for row in torso_band:
        xs = np.where(row)[0]
        if len(xs) > 80:
            wide_rows += 1
            span = row[xs.min() : xs.max() + 1]
            span_pixels += len(span)
            hole_pixels += int(np.sum(~span))

    hole_ratio = hole_pixels / max(1, span_pixels)
    if wide_rows > 30 and hole_ratio > 0.035:
        raise RuntimeError(
            f"downloaded sheet has fragmented lower portrait mask: {hole_ratio:.2%}"
        )

    source_image = Image.open(source_photo).convert("RGB")
    source_aspect = source_image.width / max(1, source_image.height)
    tile_aspect = tile_w / tile_h
    if abs(source_aspect - tile_aspect) < 0.06:
        source_tile = source_image.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        source_array = np.asarray(source_tile)
        y0 = int(tile_h * 0.55)
        x0 = int(tile_w * 0.50)
        source_region = source_array[y0 : int(tile_h * 0.95), x0:, :]
        output_region = tile_array[y0 : int(tile_h * 0.95), x0:, :]

        source_span = np.max(source_region, axis=2) - np.min(source_region, axis=2)
        output_span = np.max(output_region, axis=2) - np.min(output_region, axis=2)
        source_colorful = np.logical_and(source_span > 35, np.max(source_region, axis=2) > 80)
        source_colorful_ratio = float(np.mean(source_colorful))
        if source_colorful_ratio > 0.08:
            output_colorful = np.logical_and(
                output_span > 25,
                np.any(output_region < 235, axis=2),
            )
            retained = float(
                np.sum(np.logical_and(source_colorful, output_colorful))
                / max(1, np.sum(source_colorful))
            )
            if retained < 0.55:
                raise RuntimeError(
                    f"downloaded sheet erased colorful lower garment: {retained:.2%}"
                )

    return lower_pixels


def wait_for_server(process: subprocess.Popen, timeout: float = 45.0) -> None:
    """Wait for Streamlit's health endpoint to become available."""
    deadline = time.time() + timeout
    last_error: Exception | None = None

    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Streamlit exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=1.0) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)

    raise RuntimeError(f"Streamlit did not become healthy: {last_error}")


def server_is_healthy() -> bool:
    """Return True when an existing Streamlit server is already ready."""
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=1.0) as response:
            return response.status == 200
    except Exception:
        return False


def main() -> int:
    if PHOTO is None:
        print(
            "Set PASSPORT_TEST_PHOTO to the path of a portrait image before running this test.",
            flush=True,
        )
        return 2
    if not PHOTO.is_file():
        print(f"missing_photo={PHOTO}", flush=True)
        return 2

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ARTIFACT_DIR / "streamlit_e2e.log"
    log_file = None
    process: subprocess.Popen | None = None

    try:
        if server_is_healthy():
            print("using_existing_server=True", flush=True)
        else:
            log_file = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    "app.py",
                    "--server.headless=true",
                    f"--server.port={PORT}",
                ],
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            wait_for_server(process)
            print("using_existing_server=False", flush=True)

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1366, "height": 900})
            page.goto(URL, wait_until="networkidle", timeout=30_000)
            page.get_by_text("Passport Photo Grid Generator").wait_for(timeout=15_000)

            file_input = page.locator("input[type=file]")
            file_input.set_input_files(str(PHOTO))
            page.get_by_text("Total photos requested").wait_for(timeout=20_000)
            page.get_by_text("Edit Crop and Photo").wait_for(timeout=30_000)
            page.get_by_text("Edited crop preview used for generation").wait_for(
                timeout=60_000
            )
            page.get_by_text("Drag crop box + sliders", exact=True).click()
            page.get_by_text("Drag the crop box").wait_for(timeout=60_000)

            zoom_slider = page.get_by_role("slider").first
            zoom_slider.wait_for(timeout=20_000)
            zoom_slider.press("ArrowRight")
            page.wait_for_timeout(750)

            page.get_by_role("button", name="Generate Print Sheets").click()
            page.get_by_text("Generated").wait_for(timeout=120_000)
            page.get_by_role("button", name="Download Page 1 (JPEG)").wait_for(
                timeout=30_000
            )

            download_path = ARTIFACT_DIR / "streamlit_e2e_page1.jpg"
            with page.expect_download(timeout=30_000) as download_info:
                page.get_by_role("button", name="Download Page 1 (JPEG)").click()
            download = download_info.value
            download.save_as(str(download_path))
            lower_pixels = validate_downloaded_sheet(download_path, PHOTO)

            screenshot = ARTIFACT_DIR / "streamlit_e2e.png"
            page.screenshot(path=str(screenshot), full_page=True)

            text = page.locator("body").inner_text(timeout=10_000)
            print(f"photo={PHOTO}", flush=True)
            print("rendered_title=True", flush=True)
            print("editor_visible=True", flush=True)
            print("edit_control_interacted=True", flush=True)
            print("generated_success=True", flush=True)
            print(f"download={download_path}", flush=True)
            print(f"download_lower_portrait_pixels={lower_pixels}", flush=True)
            print(f"screenshot={screenshot}", flush=True)
            print(f"body_excerpt={text[:300].encode('ascii', 'replace').decode()}", flush=True)

            browser.close()

    except PlaywrightTimeoutError as exc:
        print(f"playwright_timeout={exc}", flush=True)
        return 1
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log_file is not None:
            log_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
