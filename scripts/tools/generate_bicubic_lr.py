"""Generate MATLAB-like bicubic LR images from an HR directory tree."""

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from basicsr.utils.matlab_functions import imresize
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def generate_lr(image_path: Path, output_path: Path, scale: int) -> None:
    """Downsample an RGB image using BasicSR MATLAB-like bicubic."""

    with Image.open(image_path) as image:
        hr = np.asarray(image.convert("RGB"), dtype=np.uint8)

    height, width = hr.shape[:2]

    if height % scale != 0 or width % scale != 0:
        raise ValueError(
            f"{image_path}: HR size ({height}, {width}) "
            f"is not divisible by scale {scale}"
        )

    # BasicSR expects HWC float32 in [0, 1].
    hr_float = hr.astype(np.float32) / 255.0

    lr_float = imresize(
        hr_float,
        scale=1.0 / scale,
        antialiasing=True,
    )

    # Match conventional 8-bit image quantization.
    lr = np.clip(
        np.round(lr_float * 255.0),
        0,
        255,
    ).astype(np.uint8)

    expected_shape = (height // scale, width // scale, 3)
    if lr.shape != expected_shape:
        raise ValueError(
            f"Unexpected LR shape: {lr.shape}, expected {expected_shape}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(lr, mode="RGB").save(output_path, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate bicubic LR images with BasicSR imresize."
    )

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    input_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()

    if not input_root.is_dir():
        parser.error(f"Input directory does not exist: {input_root}")

    if args.scale < 2:
        parser.error("--scale must be >= 2")

    if input_root == output_root or input_root in output_root.parents:
        parser.error("Output directory must not be inside input directory")

    images = sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if not images:
        parser.error("No supported images found")

    LOGGER.info("Input: %s", input_root)
    LOGGER.info("Output: %s", output_root)
    LOGGER.info("Scale: x%d", args.scale)
    LOGGER.info("Images found: %d", len(images))

    processed = 0
    skipped = 0
    failed = 0

    for image_path in tqdm(images, desc="Generating LR", unit="img"):
        relative_path = image_path.relative_to(input_root)
        output_path = (output_root / relative_path).with_suffix(".png")

        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        if args.dry_run:
            processed += 1
            continue

        try:
            generate_lr(image_path, output_path, args.scale)
            processed += 1
        except Exception:
            failed += 1
            LOGGER.exception("Failed to process %s", image_path)

    LOGGER.info(
        "Finished. Processed: %d | Skipped: %d | Failed: %d",
        processed,
        skipped,
        failed,
    )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
