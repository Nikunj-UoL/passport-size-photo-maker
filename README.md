# Passport Photo Grid Generator

A privacy-focused Streamlit application that automatically crops and aligns portrait photos, replaces their backgrounds using local deterministic image processing, checks passport-photo geometry, and arranges multiple copies on print-ready 300 DPI sheets.

The processing pipeline runs locally. It does not call generative AI, LLM, VLM, or remote image APIs.

## Features

- Upload and process multiple JPG or PNG portraits
- Set names and individual or uniform photo quantities
- Create 35 x 45 mm India/international or 2 x 2 inch US photos
- Choose white or light-blue backgrounds
- Adjust crop, zoom, position, rotation, brightness, contrast, and sharpness
- Export A4, 4 x 6 inch, or 5 x 7 inch layouts at 300 DPI
- Automatically create multiple pages with 10 mm borders and 2 mm cutting gaps

## Requirements

- Python 3.10 or newer
- Windows, macOS, or Linux

`rembg` downloads its local ONNX segmentation model the first time background removal runs. After that, processing can run locally using the cached model.

## Installation

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the application

```bash
python -m streamlit run app.py
```

Open the local address shown by Streamlit, normally `http://localhost:8501`.

## Optional end-to-end test

The browser test needs Playwright Chromium and a real portrait image:

```bash
python -m pip install -r requirements-dev.txt
python -m playwright install chromium
```

In PowerShell:

```powershell
$env:PASSPORT_TEST_PHOTO = "C:\path\to\portrait.jpg"
python final_test.py
```

Test screenshots and generated sheets are written under `output/`, which is excluded from Git.

## Project structure

```text
app.py                  Streamlit interface and workflow
face_detector.tflite    Local MediaPipe face-detection model
final_test.py           Optional Playwright end-to-end smoke test
requirements.txt        Python dependencies
requirements-dev.txt    Optional end-to-end test dependencies
utils/imaging.py        Cropping, background replacement, and sheet layout
utils/cv_checks.py      Deterministic passport-photo quality checks
```

## Privacy

Uploaded images are processed within the running application and generated files are placed in a temporary directory for download. Do not commit personal portrait photos or generated output sheets to a public repository.
