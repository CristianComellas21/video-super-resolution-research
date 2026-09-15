from pathlib import Path
import logging

import torch
import torch.utils.data as data
import torch.nn.functional as F

from datasets.DatasetVideoRecurrent import _list_files, _read_img_seq, _sequence_dirs

logger = logging.getLogger(__name__)


class Dataset(data.Dataset):
    """Video dataset that reads only GT/HR sequences and generates LR on-the-fly.

    Expected structure:
    dataroot_gt
    ├── seq_a
    │   ├── frame000
    │   ├── frame001
    │   ├── ...
    ├── group_1
    │   ├── seq_b
    │   │   ├── frame000
    │   │   ├── frame001
    │   │   ├── ...

    Args:
        dataroot_gt (str): Data root path for GT/HR.
        seq_len (int): Length of each sequence window.
        scale (int): Downscale factor for generated LR.
        stride (int): Step size between windows.
        meta_info_file (str | None): Optional list of relative sequence paths.
        max_depth (int): Max folder depth to search for sequences.
    """

    def __init__(
        self,
        dataroot_gt,
        seq_len,
        dataname=None,
        scale=4,
        stride=1,
        meta_info_file=None,
        max_depth=2,
        max_samples=None,
        spatial_crop_size=64,
        spatial_crop_mode="random",
        *args,
        **kwargs,
    ):
        super(Dataset, self).__init__()
        self.gt_root = Path(dataroot_gt)
        self.seq_len = int(seq_len)
        self.scale = int(scale)
        self.stride = int(stride)
        self.max_samples = max_samples
        self.spatial_crop_size = int(spatial_crop_size) if spatial_crop_size else None
        self.spatial_crop_mode = str(spatial_crop_mode).lower()

        if self.spatial_crop_mode not in {"random", "center"}:
            raise ValueError(
                f"spatial_crop_mode must be 'random' or 'center', got {spatial_crop_mode}"
            )

        logger.info(
            f"Init HR->LR dataset: gt_root={self.gt_root} seq_len={self.seq_len} "
            f"scale={self.scale} stride={self.stride} dataname={dataname} "
            f"meta_info_file={meta_info_file} max_depth={max_depth} "
            f"max_samples={self.max_samples} spatial_crop_size={self.spatial_crop_size} "
            f"spatial_crop_mode={self.spatial_crop_mode}"
        )

        if self.seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if self.scale <= 0:
            raise ValueError("scale must be > 0")
        if self.stride <= 0:
            raise ValueError("stride must be > 0")

        self.samples = []
        self._gt_paths = {}

        gt_dirs = _sequence_dirs(self.gt_root, meta_info_file=meta_info_file, max_depth=max_depth)
        gt_map = {p.relative_to(self.gt_root).as_posix(): p for p in gt_dirs}
        folders = sorted(gt_map)

        if not folders:
            raise ValueError("No subfolders with image sequences found in GT root")

        logger.info(f"Found {len(folders)} GT subfolders")

        for subfolder_name in folders:
            subfolder_gt = gt_map[subfolder_name]
            img_paths_gt = _list_files(subfolder_gt)

            if not img_paths_gt:
                logger.debug(f"Skipping empty folder: {subfolder_name}")
                continue

            num_frames = len(img_paths_gt)
            if num_frames < self.seq_len:
                logger.debug(
                    f"Skipping short folder: {subfolder_name} ({num_frames} frames)"
                )
                continue

            self._gt_paths[subfolder_name] = [str(p) for p in img_paths_gt]

            max_start = num_frames - self.seq_len
            for start in range(0, max_start + 1, self.stride):
                self.samples.append((subfolder_name, start))

        logger.info(f"Built {len(self.samples)} samples")

        if self.max_samples is not None:
            if isinstance(self.max_samples, str):
                self.max_samples = int(self.max_samples)
            if self.max_samples <= 0:
                raise ValueError("max_samples must be > 0")
            if self.max_samples < len(self.samples):
                self.samples = self.samples[: self.max_samples]
                logger.info(f"Capped samples to {len(self.samples)}")

    def __getitem__(self, index):
        folder, start = self.samples[index]
        end = start + self.seq_len

        gt_paths = self._gt_paths[folder][start:end]
        imgs_gt = _read_img_seq(gt_paths)

        _, _, h, w = imgs_gt.shape
        h_lq = h // self.scale
        w_lq = w // self.scale
        if h_lq <= 0 or w_lq <= 0:
            raise ValueError(
                f"Invalid spatial size for scale={self.scale}: got HR ({h}, {w}) in folder {folder}"
            )

        h_crop = h_lq * self.scale
        w_crop = w_lq * self.scale
        imgs_gt = imgs_gt[..., :h_crop, :w_crop]

        imgs_lq = F.interpolate(
            imgs_gt,
            size=(h_lq, w_lq),
            mode="bicubic",
            antialias=True,
            align_corners=False
        )

        if self.spatial_crop_size is not None:
            imgs_lq, imgs_gt = self._apply_spatial_crop(imgs_lq, imgs_gt)

        logger.debug(f"Sample {index}: folder={folder} start={start} end={end}")
        logger.debug(f"  gt_paths: {gt_paths} (type={type(gt_paths)}, len={len(gt_paths)})")
        logger.debug(f"  imgs_lq: type={type(imgs_lq)}, shape={imgs_lq.shape}")
        logger.debug(f"  imgs_gt: type={type(imgs_gt)}, shape={imgs_gt.shape}")
        if gt_paths:
            logger.debug(f"  First gt_path: {gt_paths[0]}")

        return {
            "L": imgs_lq,
            "H": imgs_gt,
            "folder": str(folder),
            "lq_path": gt_paths,
            "gt_path": gt_paths,
            "start": start,
            "idx": f"{start}/{len(self._gt_paths[folder])}",
        }

    def _apply_spatial_crop(self, imgs_lq, imgs_gt):
        """Apply a consistent spatial crop to LQ and GT sequences.

        Crop is performed after generating LQ from GT, keeping GT/LQ alignment:
        - LQ crop size is spatial_crop_size
        - GT crop size is spatial_crop_size * scale
        """
        if imgs_lq.ndim != 4 or imgs_gt.ndim != 4:
            raise ValueError(
                f"Expected 4D tensors (T, C, H, W), got "
                f"imgs_lq.ndim={imgs_lq.ndim}, imgs_gt.ndim={imgs_gt.ndim}"
            )

        t_lq, c_lq, h_lq, w_lq = imgs_lq.shape
        t_gt, c_gt, h_gt, w_gt = imgs_gt.shape

        if t_lq != t_gt:
            raise ValueError(
                f"Temporal mismatch: LQ has {t_lq} frames, GT has {t_gt} frames."
            )

        if c_lq != c_gt:
            raise ValueError(
                f"Channel mismatch: LQ has {c_lq} channels, GT has {c_gt} channels."
            )

        crop_size_lq = int(self.spatial_crop_size)
        crop_size_gt = crop_size_lq * self.scale

        if crop_size_lq <= 0:
            raise ValueError(f"spatial_crop_size must be > 0, got {crop_size_lq}")

        if h_gt != h_lq * self.scale or w_gt != w_lq * self.scale:
            raise ValueError(
                f"GT/LQ size mismatch: "
                f"LQ=({h_lq}, {w_lq}), GT=({h_gt}, {w_gt}), scale={self.scale}. "
                f"Expected GT=({h_lq * self.scale}, {w_lq * self.scale})."
            )

        if h_lq < crop_size_lq or w_lq < crop_size_lq:
            raise ValueError(
                f"LQ image size ({h_lq}x{w_lq}) is smaller than crop size "
                f"({crop_size_lq}x{crop_size_lq})."
            )

        h_max_lq = h_lq - crop_size_lq
        w_max_lq = w_lq - crop_size_lq

        if self.spatial_crop_mode == "center":
            h_start_lq = h_max_lq // 2
            w_start_lq = w_max_lq // 2
        else:
            h_start_lq = torch.randint(0, h_max_lq + 1, (1,)).item()
            w_start_lq = torch.randint(0, w_max_lq + 1, (1,)).item()

        h_start_gt = h_start_lq * self.scale
        w_start_gt = w_start_lq * self.scale

        imgs_lq_cropped = imgs_lq[
            :,
            :,
            h_start_lq:h_start_lq + crop_size_lq,
            w_start_lq:w_start_lq + crop_size_lq,
        ].contiguous()

        imgs_gt_cropped = imgs_gt[
            :,
            :,
            h_start_gt:h_start_gt + crop_size_gt,
            w_start_gt:w_start_gt + crop_size_gt,
        ].contiguous()

        logger.debug(
            "Applied spatial crop: "
            f"mode={self.spatial_crop_mode}; "
            f"LQ start=({h_start_lq}, {w_start_lq}), size={crop_size_lq}; "
            f"GT start=({h_start_gt}, {w_start_gt}), size={crop_size_gt}; "
            f"LQ ({h_lq}x{w_lq}) -> ({imgs_lq_cropped.shape[-2]}x{imgs_lq_cropped.shape[-1]}), "
            f"GT ({h_gt}x{w_gt}) -> ({imgs_gt_cropped.shape[-2]}x{imgs_gt_cropped.shape[-1]})"
        )

        return imgs_lq_cropped, imgs_gt_cropped
        

    def __len__(self):
        return len(self.samples)

    def get_images_names(self):
        names = []
        for folder, _start in self.samples:
            names.append(str(Path(folder)))
        return names

    def get_original_image_names(self):
        names = []
        for folder, start in self.samples:
            end = start + self.seq_len
            gt_paths = self._gt_paths[folder][start:end]
            if gt_paths:
                center_path = Path(gt_paths[len(gt_paths) // 2])
                names.append(str(Path(folder) / center_path.name))
            else:
                names.append(str(Path(folder) / f"start_{start:06d}"))
        return names

    def get_sample_names(self):
        names = []
        for folder, start in self.samples:
            names.append(str(Path(folder) / f"start_{start:06d}"))
        return names
