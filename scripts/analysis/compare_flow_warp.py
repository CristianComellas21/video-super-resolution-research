from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import torch
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


def tensor_to_uint8_img(t: torch.Tensor) -> np.ndarray:
    t = t.clamp(0.0, 1.0)
    arr = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return arr


def warp(src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    return flow_warp(src, flow.permute(0, 2, 3, 1))


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def psnr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    mse = torch.mean((a - b) ** 2).item()
    return float(-10.0 * np.log10(max(mse, eps)))


def flow_to_rgb(flow: torch.Tensor) -> np.ndarray:
    try:
        from torchvision.utils import flow_to_image
    except Exception as exc:
        raise RuntimeError(
            "torchvision no está disponible; no puedo visualizar flujo. "
            "Instala torchvision para habilitar flow_to_image."
        ) from exc

    flow_img = flow_to_image(flow.detach().cpu())
    return flow_img.squeeze(0).permute(1, 2, 0).numpy()


def load_spynet(spynet_path: Path, device: str) -> SpyNet:
    if not spynet_path.exists():
        raise FileNotFoundError(f"No existe checkpoint de SpyNet: {spynet_path}")
    net = SpyNet(str(spynet_path)).to(device)
    net.eval()
    return net


def load_raft(device: str, variant: str = "large"):
    try:
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )
    except Exception as exc:
        raise RuntimeError(
            "No pude importar RAFT desde torchvision. "
            "Necesitas torchvision>=0.14 con modelos optical_flow."
        ) from exc

    variant = variant.lower()
    if variant == "small":
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=True)
    else:
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=True)

    model = model.to(device)
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess, variant


def _pad_to_multiple_of_8(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    _, _, h, w = a.shape
    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8
    if pad_h == 0 and pad_w == 0:
        return a, b, 0, 0

    # Pad right/bottom only, keeping original coordinates at top-left.
    a_pad = torch.nn.functional.pad(a, (0, pad_w, 0, pad_h), mode="replicate")
    b_pad = torch.nn.functional.pad(b, (0, pad_w, 0, pad_h), mode="replicate")
    return a_pad, b_pad, pad_h, pad_w


@torch.no_grad()
def infer_spynet(spynet: SpyNet, ref: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    # Flow from ref(target) -> img(source) so we can warp img into ref coordinates.
    return spynet(ref, img)


@torch.no_grad()
def infer_raft(raft_model, raft_preprocess, ref: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    _, _, h, w = ref.shape
    ref_pad, img_pad, _, _ = _pad_to_multiple_of_8(ref, img)
    ref_pp, img_pp = raft_preprocess(ref_pad, img_pad)
    flow_list = raft_model(ref_pp, img_pp)
    flow = flow_list[-1]
    return flow[:, :, :h, :w]


def save_results(
    output_dir: Path,
    ref_name: str,
    img_name: str,
    method: str,
    flow: torch.Tensor,
    warped: torch.Tensor,
    ref: torch.Tensor,
    save_visuals: bool = True,
) -> dict[str, float]:
    if save_visuals:
        warped_img = tensor_to_uint8_img(warped)
        skio.imsave(output_dir / f"warped_{method}_{img_name}_on_{ref_name}.png", warped_img)

        try:
            flow_img = flow_to_rgb(flow)
            skio.imsave(output_dir / f"flow_{method}_{ref_name}_to_{img_name}.png", flow_img)
        except RuntimeError:
            pass

        diff = (warped - ref).abs().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        diff_img = tensor_to_uint8_img(diff)
        skio.imsave(output_dir / f"diff_{method}_{img_name}_vs_{ref_name}.png", diff_img)

    return {
        "mae": mae(warped, ref),
        "psnr": psnr(warped, ref),
    }


@torch.no_grad()
def evaluate_pair(
    ref: torch.Tensor,
    img: torch.Tensor,
    output_dir: Path,
    ref_name: str,
    img_name: str,
    spynet: SpyNet,
    raft_model,
    raft_preprocess,
    raft_variant: str | None,
    save_visuals: bool,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}

    flow_spy = infer_spynet(spynet, ref, img)
    warped_spy = warp(img, flow_spy)
    metrics["spynet"] = save_results(
        output_dir=output_dir,
        ref_name=ref_name,
        img_name=img_name,
        method="spynet",
        flow=flow_spy,
        warped=warped_spy,
        ref=ref,
        save_visuals=save_visuals,
    )

    if raft_model is not None and raft_preprocess is not None and raft_variant is not None:
        flow_raft = infer_raft(raft_model, raft_preprocess, ref, img)
        warped_raft = warp(img, flow_raft)
        metrics[f"raft_{raft_variant}"] = save_results(
            output_dir=output_dir,
            ref_name=ref_name,
            img_name=img_name,
            method=f"raft_{raft_variant}",
            flow=flow_raft,
            warped=warped_raft,
            ref=ref,
            save_visuals=save_visuals,
        )

    return metrics


def summarize_metrics(per_pair: list[dict]) -> dict[str, dict[str, float]]:
    if len(per_pair) == 0:
        return {}

    methods = sorted({m for row in per_pair for m in row["metrics"].keys()})
    summary: dict[str, dict[str, float]] = {}
    for method in methods:
        maes = [row["metrics"][method]["mae"] for row in per_pair if method in row["metrics"]]
        psnrs = [row["metrics"][method]["psnr"] for row in per_pair if method in row["metrics"]]
        summary[method] = {
            "pairs": len(maes),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "psnr_mean": float(np.mean(psnrs)),
            "psnr_std": float(np.std(psnrs)),
        }
    return summary


def main():
    parser = ArgumentParser(description="Compara warping LR usando SpyNet vs RAFT")
    parser.add_argument("--ref", type=str, default=None, help="Imagen de referencia (target)")
    parser.add_argument("--img", type=str, default=None, help="Imagen a warpear (source)")
    parser.add_argument("--seq_dir", type=str, default=None, help="Directorio con frames ordenados")
    parser.add_argument(
        "--spynet",
        type=str,
        default="/home/cristiancomellas/phd/super-resolution/code/checkpoints/flow/spynet/spynet_sintel_final-3d2a1287.pth",
        help="Checkpoint de SpyNet",
    )
    parser.add_argument("--raft_variant", type=str, default="large", choices=["large", "small"])
    parser.add_argument("--no_raft", action="store_true", help="Si se activa, solo corre SpyNet")
    parser.add_argument("--max_pairs", type=int, default=None, help="Máximo de pares consecutivos a evaluar")
    parser.add_argument(
        "--save_visuals",
        action="store_true",
        help="Si se activa, guarda warped/flow/diff por cada par (puede ocupar bastante disco)",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/flow_compare")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    has_pair = (args.ref is not None) and (args.img is not None)
    has_seq = args.seq_dir is not None
    if has_pair == has_seq:
        raise ValueError("Debes usar exactamente uno de estos modos: (--ref y --img) o --seq_dir")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spynet = load_spynet(Path(args.spynet), args.device)
    raft_model = None
    raft_preprocess = None
    raft_variant = None
    if not args.no_raft:
        raft_model, raft_preprocess, raft_variant = load_raft(args.device, args.raft_variant)

    if has_pair:
        ref_path = Path(args.ref)
        img_path = Path(args.img)
        if not ref_path.exists():
            raise FileNotFoundError(f"No existe --ref: {ref_path}")
        if not img_path.exists():
            raise FileNotFoundError(f"No existe --img: {img_path}")

        ref = read_img(ref_path, args.device)
        img = read_img(img_path, args.device)
        if ref.shape != img.shape:
            raise ValueError(f"ref e img deben tener mismo shape, got {tuple(ref.shape)} vs {tuple(img.shape)}")

        ref_name = ref_path.stem
        img_name = img_path.stem
        skio.imsave(output_dir / f"ref_{ref_name}.png", tensor_to_uint8_img(ref))
        skio.imsave(output_dir / f"img_{img_name}.png", tensor_to_uint8_img(img))

        metrics = evaluate_pair(
            ref=ref,
            img=img,
            output_dir=output_dir,
            ref_name=ref_name,
            img_name=img_name,
            spynet=spynet,
            raft_model=raft_model,
            raft_preprocess=raft_preprocess,
            raft_variant=raft_variant,
            save_visuals=True,
        )

        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print("=== Comparación flow + warp (LR) ===")
        print(f"ref: {ref_path}")
        print(f"img: {img_path}")
        print(f"output_dir: {output_dir}")
        for method, vals in metrics.items():
            print(f"[{method}] MAE={vals['mae']:.6f} | PSNR={vals['psnr']:.3f} dB")
        return

    seq_dir = Path(args.seq_dir)
    if not seq_dir.exists():
        raise FileNotFoundError(f"No existe --seq_dir: {seq_dir}")

    frame_paths = list_frames(seq_dir)
    if len(frame_paths) < 2:
        raise ValueError(f"Se necesitan al menos 2 frames en {seq_dir}, encontrados {len(frame_paths)}")

    pair_indices = list(range(len(frame_paths) - 1))
    if args.max_pairs is not None:
        pair_indices = pair_indices[: args.max_pairs]

    per_pair: list[dict] = []
    for idx in pair_indices:
        ref_path = frame_paths[idx]
        img_path = frame_paths[idx + 1]

        ref = read_img(ref_path, args.device)
        img = read_img(img_path, args.device)
        if ref.shape != img.shape:
            raise ValueError(
                f"Shape distinto en par consecutivo {ref_path.name}, {img_path.name}: "
                f"{tuple(ref.shape)} vs {tuple(img.shape)}"
            )

        pair_name = f"{ref_path.stem}_to_{img_path.stem}"
        pair_out_dir = output_dir / pair_name
        pair_out_dir.mkdir(parents=True, exist_ok=True)

        metrics = evaluate_pair(
            ref=ref,
            img=img,
            output_dir=pair_out_dir,
            ref_name=ref_path.stem,
            img_name=img_path.stem,
            spynet=spynet,
            raft_model=raft_model,
            raft_preprocess=raft_preprocess,
            raft_variant=raft_variant,
            save_visuals=args.save_visuals,
        )

        per_pair.append(
            {
                "pair": pair_name,
                "ref": str(ref_path),
                "img": str(img_path),
                "metrics": metrics,
            }
        )

    summary = summarize_metrics(per_pair)

    with open(output_dir / "metrics_per_pair.json", "w", encoding="utf-8") as f:
        json.dump(per_pair, f, indent=2)

    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Comparación flow + warp (LR) - secuencia ===")
    print(f"seq_dir: {seq_dir}")
    print(f"pairs_evaluated: {len(per_pair)}")
    print(f"output_dir: {output_dir}")
    for method, vals in summary.items():
        print(
            f"[{method}] pairs={vals['pairs']} | "
            f"MAE(mean±std)={vals['mae_mean']:.6f}±{vals['mae_std']:.6f} | "
            f"PSNR(mean±std)={vals['psnr_mean']:.3f}±{vals['psnr_std']:.3f} dB"
        )


if __name__ == "__main__":
    main()
