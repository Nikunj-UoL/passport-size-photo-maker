"""Streamlit frontend for the passport photo grid generator."""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

try:
    from streamlit_cropper import st_cropper
except Exception:
    st_cropper = None

from utils.cv_checks import verify_passport_photo
from utils.imaging import (
    PAPER_SIZES,
    PASSPORT_SIZES,
    apply_manual_edits,
    auto_crop_pil,
    bgr_to_pil,
    compute_tile_layout,
    fit_to_editor_size,
    image_to_png_bytes,
    load_upload_image,
    pil_to_bgr,
    png_bytes_to_image,
    remove_background_with_info,
    render_page,
    save_page,
)


st.set_page_config(
    page_title="Passport Photo Grid Generator",
    layout="wide",
)

st.title("Passport Photo Grid Generator")


def _positive_env_int(name: str, default: int) -> int:
    """Read a positive integer environment variable with a safe fallback."""
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


REMBG_TIMEOUT_SECONDS = _positive_env_int("REMBG_TIMEOUT_SECONDS", 45)
MAX_UPLOAD_FILES = 10
MAX_TOTAL_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_TOTAL_PHOTOS = 100


def _default_editor_settings() -> dict[str, float | int | str]:
    return {
        "mode": "Auto crop + sliders",
        "zoom": 1.0,
        "x_shift": 0,
        "y_shift": 0,
        "rotation": 0.0,
        "brightness": 1.0,
        "contrast": 1.0,
        "sharpness": 1.0,
    }


def _init_state() -> None:
    defaults = {
        "uploaded_file_data": {},
        "names": {},
        "quantities": {},
        "editor_settings": {},
        "edited_crop_bytes": {},
        "editor_meta": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _decode_preview(data: bytes) -> Image.Image | None:
    try:
        image = load_upload_image(data)
        image.thumbnail((120, 160))
        return image
    except ValueError:
        return None


@st.cache_data(show_spinner=False)
def _auto_crop_bytes(data: bytes, width_mm: int, height_mm: int) -> tuple[bytes, str]:
    """Return a cached automatic crop suggestion as PNG bytes."""
    source = load_upload_image(data)
    try:
        crop = auto_crop_pil(source, width_mm, height_mm)
        return image_to_png_bytes(crop), ""
    except ValueError as exc:
        target_w, target_h = _passport_pixel_size(width_mm, height_mm)
        fitted = ImageOps.fit(
            source,
            (target_w, target_h),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.45),
        )
        return image_to_png_bytes(fitted), str(exc)


def _passport_pixel_size(width_mm: int, height_mm: int) -> tuple[int, int]:
    for preset_w, preset_h, width_px, height_px in PASSPORT_SIZES.values():
        if preset_w == width_mm and preset_h == height_mm:
            return width_px, height_px
    return round(width_mm * 300 / 25.4), round(height_mm * 300 / 25.4)


def _build_zip(paths: list[str]) -> bytes:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, path in enumerate(paths, start=1):
            archive.write(path, arcname=f"output_page_{index}.jpg")
    zip_buf.seek(0)
    return zip_buf.getvalue()


def _settings_for_file(fname: str, passport_label: str) -> dict[str, float | int | str]:
    meta = st.session_state.editor_meta.get(fname)
    if meta != passport_label:
        st.session_state.editor_settings[fname] = _default_editor_settings()
        st.session_state.editor_meta[fname] = passport_label
        st.session_state.edited_crop_bytes.pop(fname, None)
    return st.session_state.editor_settings.setdefault(fname, _default_editor_settings())


def _render_photo_editor(
    fname: str,
    data: bytes,
    passport_label: str,
    passport_width_mm: int,
    passport_height_mm: int,
    output_size_px: tuple[int, int],
) -> None:
    """Render crop/edit controls and persist the edited crop in session state."""
    settings = _settings_for_file(fname, passport_label)

    try:
        source = load_upload_image(data)
        auto_png, auto_warning = _auto_crop_bytes(data, passport_width_mm, passport_height_mm)
        auto_crop = png_bytes_to_image(auto_png)
    except ValueError as exc:
        st.warning(f"{fname}: {exc}")
        st.session_state.edited_crop_bytes.pop(fname, None)
        return

    with st.expander(f"Edit Photo: {fname}", expanded=True):
        top_cols = st.columns([1, 1, 1])
        with top_cols[0]:
            st.image(source, caption="Original", width=180)
        with top_cols[1]:
            st.image(auto_crop, caption="Auto crop suggestion", width=180)
        with top_cols[2]:
            if st.button("Reset edit", key=f"reset_{fname}"):
                st.session_state.editor_settings[fname] = _default_editor_settings()
                st.session_state.edited_crop_bytes.pop(fname, None)
                st.rerun()

        if auto_warning:
            st.warning(f"Auto crop fallback used: {auto_warning}")

        mode_options = ["Auto crop + sliders", "Drag crop box + sliders"]
        if st_cropper is None:
            mode_options = ["Auto crop + sliders"]
            if settings.get("mode") != "Auto crop + sliders":
                settings["mode"] = "Auto crop + sliders"

        current_mode = str(settings.get("mode", "Auto crop + sliders"))
        if current_mode not in mode_options:
            current_mode = mode_options[0]

        settings["mode"] = st.radio(
            "Crop control",
            options=mode_options,
            index=mode_options.index(current_mode),
            horizontal=True,
            key=f"mode_{fname}",
        )

        edit_cols = st.columns([1.25, 1, 1])
        with edit_cols[0]:
            if settings["mode"] == "Drag crop box + sliders" and st_cropper is not None:
                crop_source = st_cropper(
                    fit_to_editor_size(source),
                    realtime_update=True,
                    box_color="#2563eb",
                    aspect_ratio=output_size_px,
                    key=f"cropper_{fname}_{passport_label}",
                )
                st.caption("Drag the crop box, then fine-tune with the sliders.")
            else:
                crop_source = auto_crop
                st.image(crop_source, caption="Crop source", width=240)

        with edit_cols[1]:
            settings["zoom"] = st.slider(
                "Zoom",
                min_value=1.0,
                max_value=2.5,
                value=float(settings.get("zoom", 1.0)),
                step=0.05,
                key=f"zoom_{fname}",
            )
            settings["x_shift"] = st.slider(
                "Horizontal position",
                min_value=-100,
                max_value=100,
                value=int(settings.get("x_shift", 0)),
                step=1,
                key=f"x_shift_{fname}",
            )
            settings["y_shift"] = st.slider(
                "Vertical position",
                min_value=-100,
                max_value=100,
                value=int(settings.get("y_shift", 0)),
                step=1,
                key=f"y_shift_{fname}",
            )
            settings["rotation"] = st.slider(
                "Rotation",
                min_value=-10.0,
                max_value=10.0,
                value=float(settings.get("rotation", 0.0)),
                step=0.25,
                key=f"rotation_{fname}",
            )

        with edit_cols[2]:
            settings["brightness"] = st.slider(
                "Brightness",
                min_value=0.70,
                max_value=1.30,
                value=float(settings.get("brightness", 1.0)),
                step=0.02,
                key=f"brightness_{fname}",
            )
            settings["contrast"] = st.slider(
                "Contrast",
                min_value=0.70,
                max_value=1.40,
                value=float(settings.get("contrast", 1.0)),
                step=0.02,
                key=f"contrast_{fname}",
            )
            settings["sharpness"] = st.slider(
                "Sharpness",
                min_value=0.50,
                max_value=2.00,
                value=float(settings.get("sharpness", 1.0)),
                step=0.05,
                key=f"sharpness_{fname}",
            )

        edited = apply_manual_edits(
            crop_source,
            output_size_px,
            zoom=float(settings["zoom"]),
            x_shift=int(settings["x_shift"]),
            y_shift=int(settings["y_shift"]),
            rotation=float(settings["rotation"]),
            brightness=float(settings["brightness"]),
            contrast=float(settings["contrast"]),
            sharpness=float(settings["sharpness"]),
        )
        st.session_state.edited_crop_bytes[fname] = image_to_png_bytes(edited)
        st.image(edited, caption="Edited crop preview used for generation", width=220)


_init_state()


# ---------------------------------------------------------------------------
# Sidebar layout controls
# ---------------------------------------------------------------------------

st.sidebar.header("Layout")

paper_label = st.sidebar.selectbox(
    "Paper Size",
    options=list(PAPER_SIZES.keys()),
    index=0,
)
paper_size_px = PAPER_SIZES[paper_label]

passport_label = st.sidebar.selectbox(
    "Passport Photo Size",
    options=list(PASSPORT_SIZES.keys()),
    index=0,
)
psp_w_mm, psp_h_mm, psp_w_px, psp_h_px = PASSPORT_SIZES[passport_label]

bg_color_option = st.sidebar.selectbox(
    "Background Colour",
    options=["White (255, 255, 255)", "Light Blue (240, 248, 255)"],
    index=0,
)
bg_color = (255, 255, 255) if bg_color_option.startswith("White") else (240, 248, 255)

st.sidebar.caption(
    f"Photo: {psp_w_mm} x {psp_h_mm} mm ({psp_w_px} x {psp_h_px} px)"
)
st.sidebar.caption(f"Sheet: {paper_size_px[0]} x {paper_size_px[1]} px at 300 DPI")


# ---------------------------------------------------------------------------
# Uploads and per-person controls
# ---------------------------------------------------------------------------

st.header("1. Upload Photos")

uploaded_files = st.file_uploader(
    "Choose portrait photos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

upload_selection_valid = True
if uploaded_files and len(uploaded_files) > MAX_UPLOAD_FILES:
    st.error(f"Upload at most {MAX_UPLOAD_FILES} photos in one batch.")
    upload_selection_valid = False
elif uploaded_files and sum(file.size for file in uploaded_files) > MAX_TOTAL_UPLOAD_BYTES:
    st.error("The total upload size must not exceed 50 MB per batch.")
    upload_selection_valid = False

stored_keys = set(st.session_state.uploaded_file_data.keys())
current_keys = (
    {file.name for file in (uploaded_files or [])}
    if upload_selection_valid
    else stored_keys
)

for key in stored_keys - current_keys:
    st.session_state.uploaded_file_data.pop(key, None)
    st.session_state.names.pop(key, None)
    st.session_state.quantities.pop(key, None)
    st.session_state.editor_settings.pop(key, None)
    st.session_state.edited_crop_bytes.pop(key, None)
    st.session_state.editor_meta.pop(key, None)

if upload_selection_valid and uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.uploaded_file_data:
            data = file.read()
            st.session_state.uploaded_file_data[file.name] = data
            st.session_state.names[file.name] = Path(file.name).stem
            st.session_state.quantities[file.name] = 4

st.header("2. Configure People")

if not st.session_state.uploaded_file_data:
    st.info("Upload at least one JPG or PNG portrait.")
else:
    uniform = st.checkbox(
        "Apply uniform quantity to all individuals",
        key="uniform_checkbox",
    )
    global_qty = 4
    if uniform:
        global_qty = int(
            st.number_input(
                "Number of photos per person",
                min_value=1,
                max_value=100,
                value=4,
                step=1,
                key="global_qty",
            )
        )

    valid_preview_count = 0
    for fname, data in st.session_state.uploaded_file_data.items():
        cols = st.columns([1, 2, 1])

        with cols[0]:
            preview = _decode_preview(data)
            if preview is None:
                st.warning("Unreadable image")
            else:
                valid_preview_count += 1
                st.image(preview, width=120)

        with cols[1]:
            st.session_state.names[fname] = st.text_input(
                "Name",
                value=st.session_state.names.get(fname, Path(fname).stem),
                key=f"name_{fname}",
            )

        with cols[2]:
            if uniform:
                st.number_input(
                    "Photos",
                    min_value=1,
                    max_value=100,
                    value=global_qty,
                    step=1,
                    key=f"qty_display_{fname}",
                    disabled=True,
                )
                st.session_state.quantities[fname] = global_qty
            else:
                st.session_state.quantities[fname] = int(
                    st.number_input(
                        "Photos",
                        min_value=1,
                        max_value=100,
                        value=int(st.session_state.quantities.get(fname, 4)),
                        step=1,
                        key=f"qty_{fname}",
                    )
                )

    total_photos = sum(int(qty) for qty in st.session_state.quantities.values())
    st.caption(f"Total photos requested: {total_photos}")
    if total_photos > MAX_TOTAL_PHOTOS:
        st.error(
            f"Request at most {MAX_TOTAL_PHOTOS} photos per generation to keep "
            "the hosted app responsive."
        )
    if valid_preview_count == 0:
        st.warning("No readable previews were found in the current upload set.")


# ---------------------------------------------------------------------------
# Edit crop/photo controls
# ---------------------------------------------------------------------------

st.header("3. Edit Crop and Photo")

if not st.session_state.uploaded_file_data:
    st.info("Upload photos first to unlock crop and edit controls.")
else:
    if st_cropper is None:
        st.warning(
            "Drag cropper dependency is not installed. Slider editing is still available."
        )
    for fname, data in st.session_state.uploaded_file_data.items():
        _render_photo_editor(
            fname,
            data,
            passport_label,
            psp_w_mm,
            psp_h_mm,
            (psp_w_px, psp_h_px),
        )


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

st.header("4. Generate Print Sheets")

requested_photo_count = sum(
    int(qty) for qty in st.session_state.quantities.values()
)
can_process = (
    bool(st.session_state.uploaded_file_data)
    and requested_photo_count <= MAX_TOTAL_PHOTOS
)
if st.button("Generate Print Sheets", type="primary", disabled=not can_process):
    progress_bar = st.progress(0.0, text="Initialising")
    status_text = st.empty()

    processed_images: list[tuple[str, Image.Image]] = []
    errors: list[str] = []
    warnings: list[str] = []
    file_items = list(st.session_state.uploaded_file_data.items())
    rembg_available = True

    for index, (fname, data) in enumerate(file_items, start=1):
        person_name = st.session_state.names.get(fname, Path(fname).stem).strip()
        person_name = person_name or Path(fname).stem
        qty = int(st.session_state.quantities.get(fname, 1))

        status_text.text(f"Processing {person_name} ({index}/{len(file_items)})")
        progress_bar.progress((index - 1) / len(file_items), text=f"Processing {person_name}")

        try:
            raw_pil = load_upload_image(data)
            if raw_pil.width < psp_w_px or raw_pil.height < psp_h_px:
                raise ValueError(
                    f"Image resolution is too low ({raw_pil.width} x {raw_pil.height} px). "
                    f"Use an image at least {psp_w_px} x {psp_h_px} px."
                )
            raw_bgr = pil_to_bgr(raw_pil)

            manual_crop_used = fname in st.session_state.edited_crop_bytes
            if manual_crop_used:
                crop_pil = png_bytes_to_image(st.session_state.edited_crop_bytes[fname])
            else:
                crop_pil = auto_crop_pil(raw_pil, psp_w_mm, psp_h_mm)

            crop_pil = crop_pil.resize((psp_w_px, psp_h_px), Image.Resampling.LANCZOS)
            bg_result = remove_background_with_info(
                pil_to_bgr(crop_pil),
                background_color=bg_color,
                prefer_rembg=rembg_available,
                rembg_timeout_seconds=REMBG_TIMEOUT_SECONDS,
            )
            if bg_result.warning:
                warnings.append(f"{person_name}: {bg_result.warning}")
                rembg_available = False
            else:
                warnings.append(f"{person_name}: background removed with {bg_result.method}")

            corrected_bgr, verification = verify_passport_photo(
                bg_result.image_bgr,
                raw_bgr,
                psp_w_mm,
                psp_h_mm,
                background_color=bg_color,
                max_attempts=3,
                allow_recrop=not manual_crop_used,
                enforce_face_centering=not manual_crop_used,
                enforce_head_ratio=not manual_crop_used,
            )

            if not verification.passed:
                errors.append(
                    f"{person_name}: passport checks failed - "
                    + "; ".join(verification.errors)
                )
                continue

            final_pil = bgr_to_pil(corrected_bgr)
            for _ in range(qty):
                processed_images.append((person_name, final_pil.copy()))

        except ValueError as exc:
            errors.append(f"{person_name}: {exc}")
        except Exception as exc:
            errors.append(f"{person_name}: unexpected error - {exc}")

    if not processed_images:
        progress_bar.empty()
        status_text.empty()
        st.error("No images were successfully processed.")
        for error in errors:
            st.warning(error)
        st.stop()

    progress_bar.progress(1.0, text="Laying out sheets")
    status_text.text("Generating print layout")

    try:
        tile_images = [image for _, image in processed_images]
        layouts = compute_tile_layout(
            tile_images,
            paper_size_px,
            psp_w_mm,
            psp_h_mm,
        )
    except ValueError as exc:
        progress_bar.empty()
        status_text.empty()
        st.error(str(exc))
        st.stop()

    output_dir = tempfile.mkdtemp(prefix="passport_")
    output_paths: list[str] = []

    try:
        for page_index, layout in enumerate(layouts, start=1):
            canvas = render_page(layout)
            path = os.path.join(output_dir, f"output_page_{page_index}.jpg")
            save_page(canvas, path)
            output_paths.append(path)

        progress_bar.progress(1.0, text="Done")
        status_text.empty()

        st.success(
            f"Generated {len(output_paths)} page(s) with "
            f"{len(processed_images)} photo(s)."
        )

        st.subheader("Preview")
        preview_cols = st.columns(min(len(output_paths), 3))
        for index, path in enumerate(output_paths, start=1):
            with preview_cols[(index - 1) % len(preview_cols)]:
                with Image.open(path) as preview_source:
                    preview = preview_source.copy()
                preview.thumbnail((400, 600))
                st.image(preview, caption=f"Page {index}")

        st.subheader("Download")
        if len(output_paths) == 1:
            download_data = Path(output_paths[0]).read_bytes()
            st.download_button(
                label="Download Page 1 (JPEG)",
                data=download_data,
                file_name="passport_sheet.jpg",
                mime="image/jpeg",
            )
        else:
            download_data = _build_zip(output_paths)
            st.download_button(
                label=f"Download All Pages (ZIP, {len(output_paths)} files)",
                data=download_data,
                file_name="passport_sheets.zip",
                mime="application/zip",
            )
    finally:
        # Download widgets retain the bytes. Always remove hosted temporary
        # passport files, including when rendering or ZIP creation fails.
        shutil.rmtree(output_dir, ignore_errors=True)

    if errors or warnings:
        with st.expander("Processing Details"):
            for warning in warnings:
                st.info(warning)
            for error in errors:
                st.warning(error)
