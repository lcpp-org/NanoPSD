# SPDX-License-Identifier: GPL-3.0-or-later
#
# NanoPSD: Automated Nanoparticle Shape Distribution Analysis
# Copyright (C) 2026 Md Fazlul Huq
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Interactive Selection Utilities for NanoPSD
===========================================

This module provides optional, user-driven input helpers that let a user
override automatic detection when it fails or when only part of an image
should be analyzed.

Currently provides:
  - select_roi_interactive: drag a rectangle to select the analysis region.

Design notes:
  - Kept in a dedicated module so the main analysis pipeline never imports
    UI code unless the user explicitly requested interactive behavior.
  - Uses cv2.selectROI (available since OpenCV 3.0, no new dependency).
  - On user cancel (Escape / window close with no selection), returns None
    so the caller can decide how to exit cleanly.
"""

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from scripts.preprocessing.clahe_filter import compute_full_image_otsu


# Window title shown to the user during ROI selection
_ROI_WINDOW_TITLE = "NanoPSD - Drag a rectangle to select ROI, ENTER to confirm, ESC to cancel"


def select_roi_interactive(
    image_path: str,
    max_display_dim: int = 1200,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Prompt the user to drag a rectangle on the image and return the ROI.

    Parameters
    ----------
    image_path : str
        Path to the input image. Read from disk with cv2.imread (color).
    max_display_dim : int, default=1200
        If either dimension of the image exceeds this, the displayed window
        is scaled down proportionally for convenience. The returned
        coordinates are always in ORIGINAL image space, not the scaled
        display space.

    Returns
    -------
    (x, y, w, h) or None
        Bounding box of the selected region in original-image pixel
        coordinates. Returns None if the user cancels (Esc / window close)
        or selects a zero-area region.

    Notes
    -----
    - Uses cv2.selectROI which blocks until the user confirms (Enter /
      Space) or cancels (Esc).
    - A scaled preview is used purely so large images fit on screen; the
      coordinates are scaled back up before returning.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image for ROI selection: {image_path}")

    h, w = img.shape[:2]

    # Scale down for display if the image is large, but keep coordinates in
    # the ORIGINAL image space.
    scale = 1.0
    if max(h, w) > max_display_dim:
        scale = max_display_dim / float(max(h, w))
        display = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        display = img

    print("\n" + "=" * 60)
    print("INTERACTIVE ROI SELECTION")
    print("=" * 60)
    print(f"Image: {os.path.basename(image_path)}  ({w} x {h} px)")
    print("Instructions:")
    print("  - Drag a rectangle with the mouse to select a region.")
    print("  - Press ENTER or SPACE to confirm.")
    print("  - Press ESC (or close window) to cancel.")
    print("=" * 60 + "\n")

    cv2.namedWindow(_ROI_WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    try:
        # showCrosshair=True gives the user visible guide lines while dragging
        # fromCenter=False means the first click is a corner, not the center
        rx, ry, rw, rh = cv2.selectROI(
            _ROI_WINDOW_TITLE, display, showCrosshair=True, fromCenter=False
        )
    finally:
        cv2.destroyWindow(_ROI_WINDOW_TITLE)
        # Small pump so the window actually closes on some platforms
        cv2.waitKey(1)

    # User cancelled or drew a zero-area box
    if rw <= 0 or rh <= 0:
        logging.info("ROI selection cancelled or empty.")
        return None

    # Scale coordinates back to original image space if we scaled the display
    if scale != 1.0:
        rx = int(round(rx / scale))
        ry = int(round(ry / scale))
        rw = int(round(rw / scale))
        rh = int(round(rh / scale))

    # Clamp to image bounds (defensive)
    rx = max(0, min(rx, w - 1))
    ry = max(0, min(ry, h - 1))
    rw = max(1, min(rw, w - rx))
    rh = max(1, min(rh, h - ry))

    logging.info(
        f"ROI selected: x={rx}, y={ry}, w={rw}, h={rh} "
        f"(image size {w}x{h})"
    )
    return (rx, ry, rw, rh)


def crop_to_cache(
    image_path: str,
    roi: Tuple[int, int, int, int],
    cache_dir: str = "outputs/.cache",
) -> Tuple[str, int, int, float]:
    """
    Crop an image to the given ROI and save to a cache file that preserves
    the original stem, so downstream output filenames are unchanged.

    Also returns the ORIGINAL full image's grayscale min/max intensity, so
    the preprocessing stage can normalize the crop using the same intensity
    range as the full image. This prevents noise amplification when the
    crop's own min/max is narrower than the full image's (e.g., when the
    user crops out the pure-black scale bar region).

    The cache file is ALWAYS saved as PNG (lossless), regardless of the
    input format. JPEG re-compression introduces 8x8 blocking artifacts
    that cause Otsu thresholding to produce thousands of spurious regions.

    Parameters
    ----------
    image_path : str
        Path to the original image.
    roi : (x, y, w, h)
        Bounding box to crop.
    cache_dir : str
        Directory for the temporary cropped image.

    Returns
    -------
    (out_path, orig_min, orig_max, orig_otsu)
        out_path : str
            Path to the cached cropped image (always .png).
        orig_min : int
            Minimum grayscale intensity of the full original image (0-255).
        orig_max : int
            Maximum grayscale intensity of the full original image (0-255).
        orig_otsu : float
            Otsu binary threshold computed on the full original image
            (after normalize → CLAHE → blur). Used to keep the crop's
            thresholding consistent with the full image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image for cropping: {image_path}")

    # Compute min/max from the ORIGINAL full image in grayscale space, matching
    # what preprocess_image will see when it reads the image with IMREAD_GRAYSCALE.
    # Preprocessing will use these as the normalization anchor so the cropped
    # image is stretched using the full image's intensity range, preventing
    # noise amplification when the crop's own min/max is narrower.
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        orig_min, orig_max = 0, 255  # safe fallback
    else:
        orig_min = int(np.min(gray))
        orig_max = int(np.max(gray))

    x, y, w, h = roi
    cropped = img[y:y + h, x:x + w]

    os.makedirs(cache_dir, exist_ok=True)
    # Force PNG extension to avoid lossy re-encoding (especially for JPEG input).
    # JPEG re-compression introduces 8x8 blocking artifacts that cause Otsu
    # thresholding to produce thousands of spurious regions.
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(cache_dir, stem + ".png")
    cv2.imwrite(out_path, cropped)
    # Compute Otsu threshold on the FULL original image so the crop can
    # use the same binary cutoff. This keeps segmentation behavior
    # consistent between full-image and cropped-image runs.
    orig_otsu = compute_full_image_otsu(image_path, norm_min=orig_min, norm_max=orig_max)
    if orig_otsu is None:
        orig_otsu = 127.0  # safe fallback

    logging.info(
        f"Cached cropped ROI to: {out_path} "
        f"(original intensity range: {orig_min}-{orig_max}, "
        f"original Otsu threshold: {orig_otsu:.1f})"
    )
    return out_path, orig_min, orig_max, orig_otsu


def delete_cache_file(path: str) -> None:
    """Delete a cache file, ignoring missing-file errors."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError as e:
        logging.warning(f"Could not delete cache file {path}: {e}")