from pathlib import Path
import logging

import numpy as np
import torch
import torch.utils.data as data
from skimage import io as skio

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

logger = logging.getLogger(__name__)

def _list_files(dir_path, suffixes=IMAGE_SUFFIXES):
    dir_path = Path(dir_path)
    # Stable, hidden-file-free listing for frame ordering.
    files = [
        p
        for p in dir_path.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and (p.suffix.lower() in suffixes)
    ]
    return sorted(files)


def _read_img_seq(path):
    if isinstance(path, (list, tuple)):
        img_paths = [Path(p) for p in path]
    else:
        img_paths = _list_files(path)

    if not img_paths:
        raise ValueError("No image files found for the requested sequence")

    imgs = []
    logger.debug(f"Reading {len(img_paths)} frames")
    for p in img_paths:
        # Read as HWC, normalize to [0, 1], and convert to CHW.
        img = skio.imread(str(p))
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        if img.shape[2] == 1:
            # Expand grayscale to 3 channels to keep batch shapes consistent.
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        img = img.astype(np.float32) / 255.0
        # Ensure contiguous memory to avoid non-resizable storage in default collate.
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        imgs.append(torch.from_numpy(img).float())
    return torch.stack(imgs, dim=0)


def _sequence_dirs(root, meta_info_file=None, max_depth=2):
    root = Path(root)
    if meta_info_file:
        meta_info_file = Path(meta_info_file)
        lines = [line.strip() for line in meta_info_file.read_text().splitlines()]
        return [root / line for line in lines if line]

    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")

    seq_dirs = []
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if depth >= max_depth:
            continue
        for entry in current.iterdir():
            if not entry.is_dir():
                continue
            if _list_files(entry):
                seq_dirs.append(entry)
            else:
                stack.append((entry, depth + 1))
    return seq_dirs


class Dataset(data.Dataset):
    """Generic video test dataset with sliding windows over subfolders.

    Expected structure (any depth, same for LQ/GT):
    dataroot
    ├── seq_a
    │   ├── frame000
    │   ├── frame001
    │   ├── ...
    ├── group_1
    │   ├── seq_b
    │   │   ├── frame000
    │   │   ├── frame001
    │   │   ├── ...
    │   ├── seq_c
    │   │   ├── frame000
    │   │   ├── frame001
    │   │   ├── ...

    Args:
        dataroot_gt (str): Data root path for GT.
        dataroot_lq (str): Data root path for LQ.
        seq_len (int): Length of each sequence window.
        stride (int): Step size between windows.
        meta_info_file (str | None): Optional list of relative sequence paths.
        max_depth (int): Max folder depth to search for sequences.
    """

    def __init__(
        self,
        dataroot_gt,
        dataroot_lq,
        seq_len,
        dataname=None,
        stride=1,
        meta_info_file=None,
        max_depth=2,
        num_folders=None,
        shuffle_folders=False,
        folders_seed=None,
        sequences_per_folder=None,
        shuffle_within_folder=False,
        sequences_seed=None,
        max_samples=None,
        spatial_crop_size: int | tuple[int, int] | None =64,
        scale=4,
        spatial_crop_mode="random",
        use_hflip=False,
        use_vflip=False,
        use_rot=False,
        random_reverse=False,
        *args,
        **kwargs,
    ):
        super(Dataset, self).__init__()
        self.gt_root = Path(dataroot_gt)
        self.lq_root = Path(dataroot_lq)
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.num_folders = None if num_folders is None else int(num_folders)
        self.shuffle_folders = bool(shuffle_folders)
        self.folders_seed = None if folders_seed is None else int(folders_seed)
        self.sequences_per_folder = None if sequences_per_folder is None else int(sequences_per_folder)
        self.shuffle_within_folder = bool(shuffle_within_folder)
        self.sequences_seed = None if sequences_seed is None else int(sequences_seed)
        self.max_samples = max_samples
        if spatial_crop_size is None:
            self.spatial_crop_size = None
        elif isinstance(spatial_crop_size, int):
            self.spatial_crop_size = int(spatial_crop_size)
        else:
            crop_size = tuple(int(s) for s in spatial_crop_size)
            if len(crop_size) != 2:
                raise ValueError(
                    "spatial_crop_size must be an int or a tuple of two integers"
                )
            self.spatial_crop_size = crop_size
        self.scale = int(scale)
        self.spatial_crop_mode = str(spatial_crop_mode).lower()
        self.use_hflip = bool(use_hflip)
        self.use_vflip = bool(use_vflip)
        self.use_rot = bool(use_rot)
        self.random_reverse = bool(random_reverse)

        if self.spatial_crop_mode not in {"random", "center"}:
            raise ValueError(
                f"spatial_crop_mode must be 'random' or 'center', got {spatial_crop_mode}"
            )

        logger.info(
            f"Init dataset: gt_root={self.gt_root} lq_root={self.lq_root} "
            f"seq_len={self.seq_len} stride={self.stride} dataname={dataname} "
            f"meta_info_file={meta_info_file} max_depth={max_depth} "
            f"max_samples={self.max_samples} spatial_crop_size={self.spatial_crop_size} "
            f"scale={self.scale} spatial_crop_mode={self.spatial_crop_mode} "
            f"use_hflip={self.use_hflip} use_vflip={self.use_vflip} "
            f"use_rot={self.use_rot} random_reverse={self.random_reverse}"
        )

        if self.seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if self.stride <= 0:
            raise ValueError("stride must be > 0")

        self.samples = []
        self._lq_paths = {}
        self._gt_paths = {}

        lq_dirs = _sequence_dirs(self.lq_root, meta_info_file=meta_info_file, max_depth=max_depth)
        gt_dirs = _sequence_dirs(self.gt_root, meta_info_file=meta_info_file, max_depth=max_depth)
        lq_map = {p.relative_to(self.lq_root).as_posix(): p for p in lq_dirs}
        gt_map = {p.relative_to(self.gt_root).as_posix(): p for p in gt_dirs}
        lq_keys = set(lq_map)
        gt_keys = set(gt_map)
        if lq_keys != gt_keys:
            missing_in_gt = sorted(lq_keys - gt_keys)
            missing_in_lq = sorted(gt_keys - lq_keys)
            raise ValueError(
                f"GT/LQ structure mismatch. Missing in GT: {missing_in_gt[:5]} "
                f"Missing in LQ: {missing_in_lq[:5]}"
            )

        common_folders = sorted(lq_keys)

        if not common_folders:
            raise ValueError("No matching subfolders found between LQ and GT roots")

        logger.info(f"Found {len(common_folders)} matching subfolders")

        # Optionally limit the number of folders used for sampling.
        if self.num_folders is not None:
            if self.num_folders <= 0:
                raise ValueError("num_folders must be > 0")
            if self.num_folders < len(common_folders):
                if self.shuffle_folders:
                    rng = np.random.RandomState(self.folders_seed) if self.folders_seed is not None else np.random.RandomState()
                    sampled = list(rng.choice(common_folders, size=self.num_folders, replace=False))
                    common_folders = sampled
                else:
                    common_folders = common_folders[: self.num_folders]
                logger.info(f"Filtered to {len(common_folders)} folders (num_folders={self.num_folders}, shuffle={self.shuffle_folders})")

        # Build a list of (folder, start_idx) windows across all subfolders.
        for subfolder_name in common_folders:
            subfolder_lq = lq_map[subfolder_name]
            subfolder_gt = gt_map[subfolder_name]
            img_paths_lq = _list_files(subfolder_lq)
            img_paths_gt = _list_files(subfolder_gt)

            if not img_paths_lq or not img_paths_gt:
                logger.debug(f"Skipping empty folder: {subfolder_name}")
                continue

            num_frames = len(img_paths_lq)
            if num_frames != len(img_paths_gt):
                raise AssertionError(
                    f"Different number of images in lq ({num_frames}) "
                    f"and gt folders ({len(img_paths_gt)})"
                )

            if num_frames < self.seq_len:
                logger.debug(
                    f"Skipping short folder: {subfolder_name} ({num_frames} frames)"
                )
                continue

            # Save paths as strings to avoid problems with Path objects in collate.
            self._lq_paths[subfolder_name] = [str(p) for p in img_paths_lq]
            self._gt_paths[subfolder_name] = [str(p) for p in img_paths_gt]

            max_start = num_frames - self.seq_len
            starts = list(range(0, max_start + 1, self.stride))

            # Optionally limit the number of sequences per folder.
            if self.sequences_per_folder is not None and self.sequences_per_folder > 0:
                if self.sequences_per_folder < len(starts):
                    if self.shuffle_within_folder:
                        rng = np.random.RandomState(self.sequences_seed) if self.sequences_seed is not None else np.random.RandomState()
                        starts = list(rng.choice(starts, size=self.sequences_per_folder, replace=False))
                        starts.sort()
                    else:
                        starts = starts[: self.sequences_per_folder]

            for start in starts:
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

        # Slice the window paths and load the corresponding frames.
        lq_paths = self._lq_paths[folder][start:end]
        gt_paths = self._gt_paths[folder][start:end]

        imgs_lq = _read_img_seq(lq_paths)
        imgs_gt = _read_img_seq(gt_paths)
        
        # Apply spatial crops if configured (consistent across all frames)
        if self.spatial_crop_size is not None:
            imgs_lq, imgs_gt = self._apply_spatial_crop(imgs_lq, imgs_gt)
            
        # Apply train-time augmentations if configured (consistent across all frames)
        imgs_lq, imgs_gt = self._apply_augmentations(imgs_lq, imgs_gt)
        
        # print types and shapes of variables for debugging and also first element of lists
        logger.debug(f"Sample {index}: folder={folder} start={start} end={end}")
        logger.debug(f"  lq_paths: {lq_paths} (type={type(lq_paths)}, len={len(lq_paths)})")
        logger.debug(f"  gt_paths: {gt_paths} (type={type(gt_paths)}, len={len(gt_paths)})")
        logger.debug(f"  imgs_lq: type={type(imgs_lq)}, shape={imgs_lq.shape}")
        logger.debug(f"  imgs_gt: type={type(imgs_gt)}, shape={imgs_gt.shape}")
        if lq_paths:
            logger.debug(f"  First lq_path: {lq_paths[0]}")
        if gt_paths:
            logger.debug(f"  First gt_path: {gt_paths[0]}")

        return {
            "L": imgs_lq,
            "H": imgs_gt,
            "folder": str(folder),
            "lq_path": lq_paths,
            "gt_path": gt_paths,
            "start": start,
            "idx": f"{start}/{len(self._lq_paths[folder])}",
        }

    def _apply_spatial_crop(self, imgs_lq, imgs_gt):
        """Apply a consistent spatial crop to LQ and GT sequences.

        The same spatial region is cropped from all frames in the LQ sequence, and
        the corresponding scaled region is cropped from the GT sequence.

        Args:
            imgs_lq (torch.Tensor): LQ sequence of shape (T, C, H_lq, W_lq).
            imgs_gt (torch.Tensor): GT sequence of shape (T, C, H_gt, W_gt), where
                H_gt = H_lq * scale and W_gt = W_lq * scale.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                Cropped LQ and GT sequences.

        Raises:
            ValueError: If tensor shapes are invalid, temporal lengths do not match,
                GT/LQ scale relation is inconsistent, or crop size is too large.
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

        if self.spatial_crop_size is None:
            return imgs_lq, imgs_gt

        if isinstance(self.spatial_crop_size, int):
            crop_h_lq = crop_w_lq = int(self.spatial_crop_size)
        else:
            crop_h_lq, crop_w_lq = self.spatial_crop_size

        if crop_h_lq <= 0 or crop_w_lq <= 0:
            raise ValueError(
                "spatial_crop_size must contain values > 0, got "
                f"{self.spatial_crop_size}"
            )
        if self.scale <= 0:
            raise ValueError(f"scale must be > 0, got {self.scale}")

        crop_h_gt = crop_h_lq * self.scale
        crop_w_gt = crop_w_lq * self.scale

        # Validate that GT is exactly the expected scaled version of LQ.
        if h_gt != h_lq * self.scale or w_gt != w_lq * self.scale:
            raise ValueError(
                f"GT/LQ size mismatch: "
                f"LQ=({h_lq}, {w_lq}), GT=({h_gt}, {w_gt}), scale={self.scale}. "
                f"Expected GT=({h_lq * self.scale}, {w_lq * self.scale})."
            )

        # Validate crop fits inside LQ. If it fits in LQ, it will fit in GT too
        # because of the exact scale relation checked above.
        if h_lq < crop_h_lq or w_lq < crop_w_lq:
            raise ValueError(
                f"LQ image size ({h_lq}x{w_lq}) is smaller than crop size "
                f"({crop_h_lq}x{crop_w_lq})."
            )

        h_max_lq = h_lq - crop_h_lq
        w_max_lq = w_lq - crop_w_lq

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
            h_start_lq:h_start_lq + crop_h_lq,
            w_start_lq:w_start_lq + crop_w_lq,
        ].contiguous()

        imgs_gt_cropped = imgs_gt[
            :,
            :,
            h_start_gt:h_start_gt + crop_h_gt,
            w_start_gt:w_start_gt + crop_w_gt,
        ].contiguous()

        logger.debug(
            "Applied spatial crop: "
            f"mode={self.spatial_crop_mode}; "
            f"LQ start=({h_start_lq}, {w_start_lq}), size=({crop_h_lq}, {crop_w_lq}); "
            f"GT start=({h_start_gt}, {w_start_gt}), size=({crop_h_gt}, {crop_w_gt}); "
            f"LQ ({h_lq}x{w_lq}) -> ({imgs_lq_cropped.shape[-2]}x{imgs_lq_cropped.shape[-1]}), "
            f"GT ({h_gt}x{w_gt}) -> ({imgs_gt_cropped.shape[-2]}x{imgs_gt_cropped.shape[-1]})"
        )

        return imgs_lq_cropped, imgs_gt_cropped


    def _apply_augmentations(self, imgs_lq, imgs_gt):
        """Apply consistent train-time augmentations to LQ and GT sequences.

        Args:
            imgs_lq (torch.Tensor): LQ sequence with shape (T, C, H, W).
            imgs_gt (torch.Tensor): GT sequence with shape (T, C, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Augmented LQ and GT sequences.
        """
        if imgs_lq.ndim != 4 or imgs_gt.ndim != 4:
            raise ValueError(
                f"Expected 4D tensors (T, C, H, W), got "
                f"imgs_lq.ndim={imgs_lq.ndim}, imgs_gt.ndim={imgs_gt.ndim}"
            )

        if imgs_lq.shape[0] != imgs_gt.shape[0]:
            raise ValueError(
                f"Temporal mismatch: LQ has {imgs_lq.shape[0]} frames, "
                f"GT has {imgs_gt.shape[0]} frames."
            )

        # Temporal reverse
        if self.random_reverse and torch.rand(1).item() < 0.5:
            imgs_lq = torch.flip(imgs_lq, dims=[0])
            imgs_gt = torch.flip(imgs_gt, dims=[0])

        # Horizontal flip: flip width dimension
        if self.use_hflip and torch.rand(1).item() < 0.5:
            imgs_lq = torch.flip(imgs_lq, dims=[-1])
            imgs_gt = torch.flip(imgs_gt, dims=[-1])

        # Vertical flip: flip height dimension
        if self.use_vflip and torch.rand(1).item() < 0.5:
            imgs_lq = torch.flip(imgs_lq, dims=[-2])
            imgs_gt = torch.flip(imgs_gt, dims=[-2])

        # Random rotation by 0, 90, 180, or 270 degrees
        if self.use_rot:
            k = torch.randint(0, 4, (1,)).item()
            if k != 0:
                imgs_lq = torch.rot90(imgs_lq, k=k, dims=[-2, -1])
                imgs_gt = torch.rot90(imgs_gt, k=k, dims=[-2, -1])

        return imgs_lq.contiguous(), imgs_gt.contiguous()

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
            lq_paths = self._lq_paths[folder][start:end]
            if lq_paths:
                center_path = Path(lq_paths[len(lq_paths) // 2])
                names.append(str(Path(folder) / center_path.name))
            else:
                names.append(str(Path(folder) / f"start_{start:06d}"))
        return names

    def get_sample_names(self):
        names = []
        for folder, start in self.samples:
            names.append(str(Path(folder) / f"start_{start:06d}"))
        return names