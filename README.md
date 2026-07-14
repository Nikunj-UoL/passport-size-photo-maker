# Passport Photo Grid Generator

A local-first Streamlit application that automatically crops and aligns portrait photos, replaces their backgrounds using deterministic image processing, checks passport-photo geometry, and arranges multiple copies on print-ready 300 DPI sheets.

The processing pipeline runs inside the application runtime. It does not call generative AI, LLM, VLM, or remote image APIs.

[![Deploy on Streamlit Community Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

## Use it as a website

GitHub Pages cannot execute this application's Python image-processing backend. Deploy the GitHub repository with **Streamlit Community Cloud** to receive a public `https://...streamlit.app` link:

1. Upload or push this complete repository to GitHub.
2. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Select **Create app**, then choose your repository.
4. Set the branch to `main` and the entrypoint to `app.py`.
5. Open **Advanced settings** and select **Python 3.12**.
6. Choose an available app URL and select **Deploy**.
7. Put the resulting `https://your-app-name.streamlit.app` address in the GitHub repository's **Website** field.

No secrets are required. The first background-removal request may take longer while the small local ONNX model is downloaded and cached by the hosted application. Future GitHub pushes automatically update the deployed app.

## Features

- Upload and process multiple JPG or PNG portraits
- Protect hosted resources with per-batch upload and decoded-image limits
- Set names and individual or uniform photo quantities
- Create 35 x 45 mm India/international or 2 x 2 inch US photos
- Choose white or light-blue backgrounds
- Adjust crop, zoom, position, rotation, brightness, contrast, and sharpness
- Export A4, 4 x 6 inch, or 5 x 7 inch layouts at 300 DPI
- Automatically create multiple pages with 10 mm borders and 2 mm cutting gaps

## Requirements

- Python 3.11, 3.12, or 3.13 (Python 3.12 recommended for deployment)
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
packages.txt            Linux libraries for Streamlit Community Cloud
.streamlit/config.toml  Hosted application configuration
utils/imaging.py        Cropping, background replacement, and sheet layout
utils/cv_checks.py      Deterministic passport-photo quality checks
```

## Privacy

When run locally, photos remain on your computer. When deployed, uploaded photos are sent to your Streamlit Community Cloud application for processing. Generated temporary files are deleted after their download data is prepared, but you should still review Streamlit's current hosting and privacy terms before processing sensitive identity photos. Do not commit personal portraits or generated output sheets to a public repository.
