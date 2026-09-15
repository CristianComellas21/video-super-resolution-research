from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import csv
import json
import shutil
import subprocess
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io as skio


warnings.filterwarnings(
    "ignore",
    message=r".*torchvision\.transforms\.functional_tensor module is deprecated.*",
    category=UserWarning,
)

from basicsr.utils.matlab_functions import imresize


def read_image_rgb(path: Path) -> np.ndarray:
    img = skio.imread(str(path))
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] >= 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255.0).round().astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def np_img_to_tensor(img: np.ndarray) -> torch.Tensor:
    ten = torch.from_numpy(img.astype(np.float32) / 255.0)
    return ten.permute(2, 0, 1).unsqueeze(0)


def tensor_to_uint8_img(t: torch.Tensor) -> np.ndarray:
    t = t.clamp(0.0, 1.0)
    out = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return out


def downsample_torch_bicubic(img: np.ndarray, out_h: int, out_w: int, antialias: bool) -> np.ndarray:
    x = np_img_to_tensor(img)
    y = F.interpolate(
        x,
        size=(out_h, out_w),
        mode="bicubic",
        align_corners=False,
        antialias=antialias,
    )
    return tensor_to_uint8_img(y)


def downsample_basicsr_matlab(img: np.ndarray, scale: float) -> np.ndarray:
    x = np_img_to_tensor(img).squeeze(0)
    y = imresize(x, scale=scale)
    if y.ndim != 3:
        raise ValueError(f"Salida inesperada de BasicSR imresize: shape={tuple(y.shape)}")
    return tensor_to_uint8_img(y.unsqueeze(0))


def get_imagemagick_command() -> list[str]:
    magick = shutil.which("magick")
    if magick is not None:
        return [magick]

    convert = shutil.which("convert")
    if convert is not None:
        return [convert]

    raise RuntimeError(
        "No se encontro ImageMagick. Instala 'magick' (IM7) o 'convert' (IM6) y vuelve a ejecutar."
    )


def downsample_imagemagick(
    in_path: Path,
    out_path: Path,
    out_h: int,
    out_w: int,
    filter_name: str,
) -> None:
    cmd = get_imagemagick_command() + [
        str(in_path),
        "-filter",
        filter_name,
        "-resize",
        f"{out_w}x{out_h}!",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float32) / 255.0
    y = b.astype(np.float32) / 255.0
    return float(np.mean(np.abs(x - y)))


def psnr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    x = a.astype(np.float32) / 255.0
    y = b.astype(np.float32) / 255.0
    mse = np.mean((x - y) ** 2)
    return float(-10.0 * np.log10(max(float(mse), eps)))


def pairwise_metrics(results: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    names = sorted(results.keys())
    out: dict[str, dict[str, float]] = {}
    for i, a_name in enumerate(names):
        for b_name in names[i + 1 :]:
            a = results[a_name]
            b = results[b_name]
            key = f"{a_name}__vs__{b_name}"
            out[key] = {
                "mae": mae(a, b),
                "psnr": psnr(a, b),
            }
    return out


def metrics_against_reference(
    results: dict[str, np.ndarray],
    reference: np.ndarray,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for method_name in sorted(results.keys()):
        method_img = results[method_name]
        out[method_name] = {
            "mae": mae(method_img, reference),
            "psnr": psnr(method_img, reference),
        }
    return out


def print_summary(metrics: dict[str, dict[str, float]]) -> None:
    print("\nComparacion pairwise (imagenes downsampleadas):")
    print("-" * 72)
    print(f"{'par':48s} {'MAE':>10s} {'PSNR(dB)':>10s}")
    print("-" * 72)
    for key in sorted(metrics.keys()):
        row = metrics[key]
        print(f"{key:48s} {row['mae']:10.6f} {row['psnr']:10.4f}")
    print("-" * 72)


def print_reference_summary(
    metrics: dict[str, dict[str, float]],
    reference_label: str,
) -> None:
    print(f"\nComparacion contra referencia: {reference_label}")
    print("-" * 72)
    print(f"{'metodo':32s} {'MAE':>10s} {'PSNR(dB)':>10s}")
    print("-" * 72)
    for method_name in sorted(metrics.keys()):
        row = metrics[method_name]
        print(f"{method_name:32s} {row['mae']:10.6f} {row['psnr']:10.4f}")
    print("-" * 72)


def write_reference_csv(
    csv_path: Path,
    reference_label: str,
    metrics: dict[str, dict[str, float]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "reference", "mae", "psnr"],
        )
        writer.writeheader()
        for method_name in sorted(metrics.keys()):
            row = metrics[method_name]
            writer.writerow(
                {
                    "method": method_name,
                    "reference": reference_label,
                    "mae": f"{row['mae']:.8f}",
                    "psnr": f"{row['psnr']:.8f}",
                }
            )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Compara 4 metodos de downsampling bicubic sobre una imagen: "
            "PyTorch, BasicSR(MATLAB-like), ImageMagick Catrom e ImageMagick Cubic."
        )
    )
    parser.add_argument("--input", type=str, required=True, help="Ruta de la imagen de entrada")
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help=(
            "Ruta de imagen de referencia para evaluar las salidas contra ella "
            "(por ejemplo, un LR de REDS)."
        ),
    )
    parser.add_argument(
        "--reference_label",
        type=str,
        default="REDS",
        help="Etiqueta legible de la referencia para reportes/CSV (por defecto: REDS)",
    )
    parser.add_argument("--scale", type=int, default=4, help="Factor de downsampling")
    parser.add_argument("--output_dir", type=str, default="./scripts/outputs/downsample_compare")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help=(
            "Directorio adicional opcional donde guardar resumen contra referencia (CSV/JSON). "
            "Si no se indica, se usa output_dir y todo queda en una sola carpeta."
        ),
    )
    parser.add_argument(
        "--torch_antialias",
        dest="torch_antialias",
        action="store_true",
        help="Activa antialias en el bicubic de PyTorch",
    )
    parser.add_argument(
        "--no_torch_antialias",
        dest="torch_antialias",
        action="store_false",
        help="Desactiva antialias en el bicubic de PyTorch",
    )
    parser.set_defaults(torch_antialias=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"No existe la imagen de entrada: {in_path}")

    if args.scale <= 0:
        raise ValueError(f"scale debe ser > 0, recibido: {args.scale}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir) if args.results_dir is not None else output_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    img = read_image_rgb(in_path)
    h, w, _ = img.shape
    out_h = h // args.scale
    out_w = w // args.scale
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"El factor scale={args.scale} deja tamano invalido para imagen {w}x{h}."
        )

    print(f"Entrada: {in_path}")
    print(f"Tamano original: {w}x{h}")
    print(f"Tamano salida:   {out_w}x{out_h} (scale={args.scale})")

    stem = in_path.stem

    gt_hr_output_path = output_dir / f"{stem}_GT_HR.png"
    gt_hr_results_path = results_dir / f"{stem}_GT_HR.png"
    skio.imsave(str(gt_hr_output_path), img)
    if gt_hr_results_path != gt_hr_output_path:
        skio.imsave(str(gt_hr_results_path), img)

    torch_img = downsample_torch_bicubic(
        img=img,
        out_h=out_h,
        out_w=out_w,
        antialias=args.torch_antialias,
    )
    basicsr_img = downsample_basicsr_matlab(img=img, scale=1.0 / args.scale)

    torch_path = output_dir / f"{stem}_down_torch_bicubic.png"
    basicsr_path = output_dir / f"{stem}_down_basicsr_matlab_like.png"
    catrom_path = output_dir / f"{stem}_down_imagemagick_catrom.png"
    cubic_path = output_dir / f"{stem}_down_imagemagick_cubic.png"

    skio.imsave(str(torch_path), torch_img)
    skio.imsave(str(basicsr_path), basicsr_img)

    downsample_imagemagick(in_path, catrom_path, out_h, out_w, "Catrom")
    downsample_imagemagick(in_path, cubic_path, out_h, out_w, "Cubic")

    catrom_img = read_image_rgb(catrom_path)
    cubic_img = read_image_rgb(cubic_path)

    results = {
        "torch_bicubic": torch_img,
        "basicsr_matlab_like": basicsr_img,
        "imagemagick_catrom": catrom_img,
        "imagemagick_cubic": cubic_img,
    }

    metrics = pairwise_metrics(results)
    print_summary(metrics)

    outputs_map = {
        "torch_bicubic": torch_path,
        "basicsr_matlab_like": basicsr_path,
        "imagemagick_catrom": catrom_path,
        "imagemagick_cubic": cubic_path,
    }

    reference_metrics: dict[str, dict[str, float]] | None = None
    reference_csv_path: Path | None = None
    reference_in_output_path: Path | None = None
    reference_in_results_path: Path | None = None

    if args.reference is not None:
        reference_path = Path(args.reference)
        if not reference_path.exists():
            raise FileNotFoundError(f"No existe la referencia: {reference_path}")
        reference_label = str(args.reference_label).strip() or "REDS"

        ref_img = read_image_rgb(reference_path)
        ref_h, ref_w, _ = ref_img.shape
        if ref_h != out_h or ref_w != out_w:
            raise ValueError(
                "La referencia no coincide con el tamano esperado de salida. "
                f"Esperado: {out_w}x{out_h}, recibido: {ref_w}x{ref_h}."
            )

        reference_metrics = metrics_against_reference(results=results, reference=ref_img)
        print_reference_summary(reference_metrics, reference_label)

        reference_in_output_path = output_dir / f"{stem}_reference_{reference_label}.png"
        reference_in_results_path = results_dir / f"{stem}_reference_{reference_label}.png"
        skio.imsave(str(reference_in_output_path), ref_img)
        if reference_in_results_path != reference_in_output_path:
            skio.imsave(str(reference_in_results_path), ref_img)

        reference_csv_path = output_dir / f"{stem}_reference_{reference_label}_metrics.csv"
        write_reference_csv(
            csv_path=reference_csv_path,
            reference_label=reference_label,
            metrics=reference_metrics,
        )

        results_csv_path = results_dir / f"{stem}_reference_{reference_label}_metrics.csv"
        if results_csv_path != reference_csv_path:
            write_reference_csv(
                csv_path=results_csv_path,
                reference_label=reference_label,
                metrics=reference_metrics,
            )

    report = {
        "input": str(in_path),
        "reference": str(args.reference_label) if args.reference is not None else None,
        "saved_input_copy": {
            "output_dir": str(gt_hr_output_path),
            "results_dir": str(gt_hr_results_path),
        },
        "original_size_wh": [w, h],
        "output_size_wh": [out_w, out_h],
        "scale": int(args.scale),
        "torch_antialias": bool(args.torch_antialias),
        "outputs": {
            "torch_bicubic": str(outputs_map["torch_bicubic"]),
            "basicsr_matlab_like": str(outputs_map["basicsr_matlab_like"]),
            "imagemagick_catrom": str(outputs_map["imagemagick_catrom"]),
            "imagemagick_cubic": str(outputs_map["imagemagick_cubic"]),
        },
        "pairwise_metrics": metrics,
        "reference_metrics": reference_metrics,
        "reference_summary_csv": str(reference_csv_path) if reference_csv_path is not None else None,
        "saved_reference_copy": {
            "output_dir": str(reference_in_output_path) if reference_in_output_path is not None else None,
            "results_dir": str(reference_in_results_path) if reference_in_results_path is not None else None,
        },
    }

    report_path = output_dir / f"{stem}_downsampling_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    results_report_path = results_dir / f"{stem}_downsampling_report.json"
    if results_report_path != report_path:
        with results_report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(f"\nResultados guardados en: {output_dir}")
    print(f"Reporte JSON: {report_path}")
    if results_dir != output_dir:
        print(f"Resultados (resumen) tambien guardados en: {results_dir}")
        print(f"Reporte JSON (results): {results_report_path}")
    if reference_csv_path is not None:
        print(f"CSV referencia (output): {reference_csv_path}")
        if results_dir != output_dir:
            print(f"CSV referencia (results): {results_dir / f'{stem}_reference_metrics.csv'}")


if __name__ == "__main__":
    main()
