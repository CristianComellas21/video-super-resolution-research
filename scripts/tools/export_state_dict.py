import argparse
from pathlib import Path

import torch


def load_state_dict(
    checkpoint_path: Path,
) -> dict:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        return checkpoint["state_dict"]

    return checkpoint


def extract_submodule(
    state_dict: dict,
    module_path: str,
) -> dict:
    prefix = f"{module_path}."

    extracted = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }

    if not extracted:
        available_roots = sorted(
            {
                key.split(".")[0]
                for key in state_dict
            }
        )

        raise ValueError(
            f"No parameters found for module '{module_path}'. "
            f"Available top-level modules: {available_roots}"
        )

    return extracted


def save_state_dict(
    state_dict: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        state_dict,
        output_path,
    )

    print(
        f"[OK] {len(state_dict)} tensors -> {output_path}"
    )


def export_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    modules: list[str] | None = None,
    root_module: str = "model",
) -> None:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_dict = load_state_dict(
        checkpoint_path,
    )

    if root_module:
        state_dict = extract_submodule(
            state_dict,
            root_module,
        )

    save_state_dict(
        state_dict,
        output_dir / "model.pth",
    )

    for module_path in modules or []:
        module_state_dict = extract_submodule(
            state_dict,
            module_path,
        )

        filename = (
            module_path.replace(".", "_")
            + ".pth"
        )

        save_state_dict(
            module_state_dict,
            output_dir / filename,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Export portable PyTorch state_dict files "
            "from a Lightning checkpoint."
        )
    )

    parser.add_argument(
        "checkpoint_path",
        type=Path,
        help="Path to the input checkpoint.",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where portable weights are written.",
    )

    parser.add_argument(
        "--modules",
        nargs="+",
        default=None,
        help=(
            "Optional model submodules to export in addition "
            "to the full model."
        ),
    )

    parser.add_argument(
        "--root-module",
        type=str,
        default="model",
        help=(
            "Root module inside the Lightning state_dict. "
            "Default: 'model'. Use '' to disable."
        ),
    )

    args = parser.parse_args()

    export_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        modules=args.modules,
        root_module=args.root_module,
    )