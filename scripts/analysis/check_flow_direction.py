from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io as skio

from basicsr.archs.arch_util import flow_warp
from basicsr.archs.spynet_arch import SpyNet


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_frames(seq_dir: Path) -> list[Path]:
    frames = [
        p
        for p in seq_dir.iterdir()
        if p.is_file() and (p.suffix.lower() in IMAGE_SUFFIXES) and not p.name.startswith(".")
    ]
    return sorted(frames)


def read_img(path: Path, device: str) -> torch.Tensor:
    img = skio.imread(str(path))
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    ten = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return ten.to(device)


def warp(src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    return flow_warp(src, flow.permute(0, 2, 3, 1))


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


@torch.no_grad()
def evaluate_sequence(
    seq_dir: Path,
    spynet: SpyNet,
    device: str,
    max_triplets: int | None = None,
) -> dict[str, float | int]:
    frame_paths = list_frames(seq_dir)
    if len(frame_paths) < 3:
        raise ValueError(f"Need at least 3 frames in {seq_dir}, found {len(frame_paths)}")

    idxs = list(range(1, len(frame_paths) - 1))
    if max_triplets is not None:
        idxs = idxs[: max_triplets]

    prev_correct_better = 0
    next_correct_better = 0

    prev_err_correct = []
    prev_err_wrong = []
    next_err_correct = []
    next_err_wrong = []

    for t in idxs:
        prev = read_img(frame_paths[t - 1], device)
        cur = read_img(frame_paths[t], device)
        nxt = read_img(frame_paths[t + 1], device)

        # SpyNet(a,b) follows a -> b convention in this setup.
        flow_fwd = spynet(prev, cur)  # prev -> cur
        flow_bwd = spynet(cur, prev)  # cur -> prev

        # To reconstruct cur from prev via flow_warp, correct flow is cur -> prev.
        prev_w_correct = warp(prev, flow_bwd)
        prev_w_wrong = warp(prev, flow_fwd)

        e_prev_correct = mae(prev_w_correct, cur)
        e_prev_wrong = mae(prev_w_wrong, cur)
        prev_err_correct.append(e_prev_correct)
        prev_err_wrong.append(e_prev_wrong)
        if e_prev_correct < e_prev_wrong:
            prev_correct_better += 1

        flow_fwd_cn = spynet(cur, nxt)  # cur -> next
        flow_bwd_nc = spynet(nxt, cur)  # next -> cur

        # To reconstruct cur from next via flow_warp, correct flow is cur -> next.
        next_w_correct = warp(nxt, flow_fwd_cn)
        next_w_wrong = warp(nxt, flow_bwd_nc)

        e_next_correct = mae(next_w_correct, cur)
        e_next_wrong = mae(next_w_wrong, cur)
        next_err_correct.append(e_next_correct)
        next_err_wrong.append(e_next_wrong)
        if e_next_correct < e_next_wrong:
            next_correct_better += 1

    n = len(idxs)
    return {
        "triplets": n,
        "prev_correct_better_count": prev_correct_better,
        "next_correct_better_count": next_correct_better,
        "prev_correct_better_ratio": prev_correct_better / max(n, 1),
        "next_correct_better_ratio": next_correct_better / max(n, 1),
        "prev_mae_correct_mean": float(np.mean(prev_err_correct)),
        "prev_mae_wrong_mean": float(np.mean(prev_err_wrong)),
        "next_mae_correct_mean": float(np.mean(next_err_correct)),
        "next_mae_wrong_mean": float(np.mean(next_err_wrong)),
    }


def main():
    parser = ArgumentParser(description="Sanity-check flow direction for flow_warp + SpyNet")
    parser.add_argument("--seq_dir", type=str, required=True, help="Directory with ordered video frames")
    parser.add_argument(
        "--spynet",
        type=str,
        default="/home/cristiancomellas/phd/super-resolution/code/checkpoints/flow/spynet/spynet_sintel_final-3d2a1287.pth",
        help="Path to spynet weights",
    )
    parser.add_argument("--max_triplets", type=int, default=30, help="Evaluate at most this many center frames")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence dir does not exist: {seq_dir}")
    if not Path(args.spynet).exists():
        raise FileNotFoundError(f"SpyNet weights not found: {args.spynet}")

    spynet = SpyNet(args.spynet).to(args.device)
    spynet.eval()

    stats = evaluate_sequence(
        seq_dir=seq_dir,
        spynet=spynet,
        device=args.device,
        max_triplets=args.max_triplets,
    )

    print("=== Flow Direction Sanity Check ===")
    print(f"sequence: {seq_dir}")
    print(f"triplets: {stats['triplets']}")
    print()
    print("[prev -> cur reconstruction]")
    print(f"correct flow (cur->prev) better ratio: {stats['prev_correct_better_ratio']:.3f}")
    print(f"mean MAE correct: {stats['prev_mae_correct_mean']:.6f}")
    print(f"mean MAE wrong:   {stats['prev_mae_wrong_mean']:.6f}")
    print()
    print("[next -> cur reconstruction]")
    print(f"correct flow (cur->next) better ratio: {stats['next_correct_better_ratio']:.3f}")
    print(f"mean MAE correct: {stats['next_mae_correct_mean']:.6f}")
    print(f"mean MAE wrong:   {stats['next_mae_wrong_mean']:.6f}")

    ok_prev = stats["prev_mae_correct_mean"] < stats["prev_mae_wrong_mean"]
    ok_next = stats["next_mae_correct_mean"] < stats["next_mae_wrong_mean"]
    print()
    print(f"conclusion_prev: {'PASS' if ok_prev else 'FAIL'}")
    print(f"conclusion_next: {'PASS' if ok_next else 'FAIL'}")


if __name__ == "__main__":
    main()
