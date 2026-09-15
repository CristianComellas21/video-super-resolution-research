from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch


def _ensure_import_paths(iart_repo_dir: Path) -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    if not iart_repo_dir.exists():
        raise FileNotFoundError(f"No existe el repositorio IART en: {iart_repo_dir}")

    if str(iart_repo_dir) not in sys.path:
        sys.path.insert(0, str(iart_repo_dir))


def _register_package(package_name: str, package_dir: Path) -> None:
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package


def _load_module(fullname: str, file_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(fullname, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No pude construir un loader para {fullname} desde {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)


def _prime_legacy_imports(code_root: Path, iart_repo_dir: Path) -> None:
    # Some legacy checkpoints reference these modules during unpickling.
    for module_name in ("models.IARTOriginal", "models", "archs.iart_arch", "archs"):
        sys.modules.pop(module_name, None)

    _register_package("models", code_root / "models")
    _load_module("models.IARTOriginal", code_root / "models" / "IARTOriginal.py")

    _register_package("archs", iart_repo_dir / "archs")
    _load_module("archs.iart_arch", iart_repo_dir / "archs" / "iart_arch.py")


def _extract_state_dict(raw):
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "params", "params_ema", "net_g", "network_g", "model"):
            value = raw.get(key)
            if isinstance(value, dict):
                return value
    return raw


def _clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            cleaned[key] = value
            continue

        new_key = key
        for prefix in ("module.", "net_g.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def export_state_dict(ckpt_path: Path, output_path: Path, iart_repo_dir: Path | None = None) -> None:
    code_root = Path(__file__).resolve().parent
    if iart_repo_dir is not None:
        _ensure_import_paths(iart_repo_dir)
        _prime_legacy_imports(code_root, iart_repo_dir)
    else:
        default_iart_dir = code_root.parent / "IART"
        _ensure_import_paths(default_iart_dir)
        _prime_legacy_imports(code_root, default_iart_dir)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"No existe checkpoint: {ckpt_path}")

    load_kwargs = {"map_location": "cpu"}
    try:
        raw = torch.load(str(ckpt_path), weights_only=True, **load_kwargs)
    except TypeError:
        raw = torch.load(str(ckpt_path), **load_kwargs)
    except Exception:
        raw = torch.load(str(ckpt_path), **load_kwargs)
    state_dict = _extract_state_dict(raw)

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "No pude encontrar un state_dict dentro del checkpoint. "
            "Prueba a abrir el .ckpt en el entorno original o indica otro contenedor."
        )

    cleaned = _clean_state_dict(state_dict)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cleaned, output_path)

    print(f"Saved state_dict with {len(cleaned)} tensors to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta solo el state_dict de un checkpoint Lightning .ckpt")
    parser.add_argument("checkpoint", type=Path, help="Ruta al archivo .ckpt de entrada")
    parser.add_argument("--output", type=Path, default=None, help="Ruta de salida .pth/.pt (por defecto: <ckpt>.state_dict.pth)")
    parser.add_argument(
        "--iart-repo-dir",
        type=Path,
        default=None,
        help="Ruta al repo IART para resolver imports legacy como 'archs' antes de torch.load",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = args.checkpoint.with_suffix("")
        output_path = output_path.with_name(f"{output_path.name}.state_dict.pth")

    export_state_dict(args.checkpoint, output_path, args.iart_repo_dir)


if __name__ == "__main__":
    main()