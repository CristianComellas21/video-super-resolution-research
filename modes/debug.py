
import gc
import json
import logging
import time
from pathlib import Path

import hydra.utils as hy_utils
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from ptflops import get_model_complexity_info
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

log = logging.getLogger(__name__)

MIB = 1024**2


# ============================================================
# DATASET
# ============================================================

def extract_input(sample, device: torch.device) -> torch.Tensor:
    """
    Extract model input following LitModule.__get_xy.

    Supported:
        dict:  {"L": x, "H": y, ...}
        tuple: (x, y, metadata)
        tensor: x

    Dataset samples are assumed to be unbatched.
    """
    if isinstance(sample, dict):
        if "L" not in sample:
            raise KeyError(
                f"Expected key 'L'. Available keys: {list(sample.keys())}"
            )
        x = sample["L"]

    elif isinstance(sample, (tuple, list)):
        if not sample:
            raise ValueError("Empty dataset sample")
        x = sample[0]

    else:
        x = sample

    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"Expected Tensor as input, got {type(x).__name__}"
        )

    # Image: (C,H,W) -> (B,C,H,W)
    # Video: (T,C,H,W) -> (B,T,C,H,W)
    if x.ndim not in (3, 4):
        raise ValueError(
            f"Expected unbatched image/video, got shape {tuple(x.shape)}"
        )

    return x.unsqueeze(0).contiguous().to(device)


def get_output_info(output: torch.Tensor) -> dict:
    """Get output shape and temporal length."""
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            f"Expected Tensor model output, got {type(output).__name__}"
        )

    if output.ndim == 5:
        output_frames = int(output.shape[1])
    elif output.ndim == 4:
        output_frames = 1
    else:
        raise ValueError(
            f"Unsupported output shape: {tuple(output.shape)}"
        )

    return {
        "Output shape": str(tuple(output.shape)),
        "Output frames": output_frames,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info(model: nn.Module) -> dict:
    """Count parameters, buffers and model memory."""
    parameters = list(model.parameters())
    buffers = list(model.buffers())

    total_params = sum(p.numel() for p in parameters)
    trainable_params = sum(
        p.numel() for p in parameters if p.requires_grad
    )
    frozen_params = total_params - trainable_params

    params_bytes = sum(
        p.numel() * p.element_size() for p in parameters
    )
    buffers_bytes = sum(
        b.numel() * b.element_size() for b in buffers
    )

    return {
        "Parameters (M)": total_params / 1e6,
        "Trainable parameters (M)": trainable_params / 1e6,
        "Frozen parameters (M)": frozen_params / 1e6,
        "Parameters memory (MiB)": params_bytes / MIB,
        "Buffers memory (MiB)": buffers_bytes / MIB,
        "Model size (MiB)": (
            params_bytes + buffers_bytes
        ) / MIB,
    }


def measure_complexity(
    model: nn.Module,
    input_shape: tuple[int, ...],
) -> dict:
    """
    Estimate MACs using ptflops (aten backend).

    FLOPs are approximated as 2 * MACs.
    Unsupported operations may not be counted.
    """
    metrics = {
        "MACs (G)": None,
        "FLOPs estimated (G)": None,
    }

    try:
        with torch.inference_mode():
            macs, _ = get_model_complexity_info(
                model,
                input_res=input_shape,
                as_strings=False,
                backend="pytorch",
                print_per_layer_stat=False,
                verbose=False,
            )

        if macs is not None:
            metrics["MACs (G)"] = float(macs) / 1e9
            metrics["FLOPs estimated (G)"] = (
                2.0 * float(macs) / 1e9
            )

    except Exception as exc:
        log.warning(
            "Could not measure model complexity: %s",
            exc,
        )

    return metrics


# ============================================================
# INFERENCE BENCHMARK
# ============================================================

def benchmark_inference(
    model: nn.Module,
    x: torch.Tensor,
    warmup: int = 20,
    iterations: int = 100,
) -> dict:
    """
    Benchmark end-to-end model forward inference.

    Measures:
        - Mean latency
        - Standard deviation
        - Median and p95/p99 latency
        - Throughput
        - Peak PyTorch CUDA memory

    CUDA timings use CUDA events.
    """
    if iterations < 1 or warmup < 0:
        raise ValueError(
            "iterations must be >= 1 and warmup must be >= 0"
        )

    model.eval()

    device = x.device
    use_cuda = device.type == "cuda"

    times = []

    log.info("Warmup: %d iterations", warmup)

    with torch.inference_mode():
        # ----------------------------------------------------
        # WARMUP
        # ----------------------------------------------------
        for _ in range(warmup):
            output = model(x)
            del output

        if use_cuda:
            torch.cuda.synchronize(device)

        # ----------------------------------------------------
        # MEMORY BASELINE
        # ----------------------------------------------------
        if use_cuda:
            baseline_allocated = torch.cuda.memory_allocated(device)
            baseline_reserved = torch.cuda.memory_reserved(device)

            torch.cuda.reset_peak_memory_stats(device)

        # ----------------------------------------------------
        # TIMED INFERENCE
        # ----------------------------------------------------
        log.info("Benchmark: %d iterations", iterations)

        for _ in range(iterations):
            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                output = model(x)
                end.record()

                end.synchronize()
                elapsed_ms = start.elapsed_time(end)

            else:
                start_time = time.perf_counter()
                output = model(x)
                elapsed_ms = (
                    time.perf_counter() - start_time
                ) * 1000.0

            times.append(elapsed_ms)
            del output

        # ----------------------------------------------------
        # MEMORY PEAK
        # ----------------------------------------------------
        if use_cuda:
            torch.cuda.synchronize(device)

            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)

    times = np.asarray(times, dtype=np.float64)

    mean_ms = float(times.mean())
    batch_size = int(x.shape[0])

    metrics = {
        "Latency mean (ms)": mean_ms,
        "Latency std (ms)": float(times.std()),
        "Latency median (ms)": float(np.median(times)),
        "Latency p95 (ms)": float(np.percentile(times, 95)),
        "Latency p99 (ms)": float(np.percentile(times, 99)),
        "Latency min (ms)": float(times.min()),
        "Latency max (ms)": float(times.max()),
        "Throughput (samples/s)": (
            batch_size * 1000.0 / mean_ms
        ),
    }

    if use_cuda:
        metrics.update({
            "Baseline allocated (MiB)":
                baseline_allocated / MIB,

            "Baseline reserved (MiB)":
                baseline_reserved / MIB,

            "Peak VRAM allocated (MiB)":
                peak_allocated / MIB,

            "Peak VRAM reserved (MiB)":
                peak_reserved / MIB,

            "Additional peak allocated (MiB)":
                max(
                    0,
                    peak_allocated - baseline_allocated,
                ) / MIB,

            "Additional peak reserved (MiB)":
                max(
                    0,
                    peak_reserved - baseline_reserved,
                ) / MIB,
        })

    return metrics


# ============================================================
# PYTORCH PROFILER
# ============================================================

def profile_model(
    model: nn.Module,
    x: torch.Tensor,
    output_dir: Path,
) -> None:
    """
    Run one inference with PyTorch profiler.

    This measurement is independent of the latency benchmark.
    """
    use_cuda = x.device.type == "cuda"

    activities = [ProfilerActivity.CPU]

    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    log.info("Starting PyTorch profiler...")

    with torch.inference_mode():
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:

            with record_function("model_inference"):
                output = model(x)

                if use_cuda:
                    torch.cuda.synchronize(x.device)

            del output

    sort_key = (
        "self_device_time_total"
        if use_cuda
        else "self_cpu_time_total"
    )

    table = prof.key_averages().table(
        sort_by=sort_key,
        row_limit=30,
    )

    log.info(
        "\n=== PYTORCH PROFILER ===\n%s",
        table,
    )

    trace_path = output_dir / "profiler_trace.json"

    prof.export_chrome_trace(str(trace_path))

    (output_dir / "profiler_summary.txt").write_text(
        table,
        encoding="utf-8",
    )

    log.info("Profiler trace: %s", trace_path)


# ============================================================
# RESULTS
# ============================================================

def save_results(
    results: dict,
    output_dir: Path,
) -> None:
    """Save benchmark results in CSV and JSON formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([results])

    csv_path = output_dir / "benchmark.csv"
    json_path = output_dir / "benchmark.json"

    df.to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )

    log.info(
        "\n=== MODEL BENCHMARK ===\n%s",
        df.T.to_string(header=False),
    )

    log.info("CSV saved to: %s", csv_path)
    log.info("JSON saved to: %s", json_path)


# ============================================================
# MAIN DEBUG ENTRY POINT
# ============================================================

def debug(cfg):
    log.info("Starting model benchmark...")

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------
    accelerator = str(
        cfg.trainer.get("accelerator", "auto")
    ).lower()

    if accelerator in ("gpu", "cuda", "auto") and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    warmup = int(cfg.get("debug_warmup", 20))
    iterations = int(cfg.get("debug_iterations", 100))
    run_profiler = bool(cfg.get("debug_profiler", False))
    run_complexity = bool(cfg.get("debug_complexity", True))

    matmul_precision = str(
        cfg.get("debug_matmul_precision", "medium")
    )
    torch.set_float32_matmul_precision(matmul_precision)

    output_dir = Path(
        HydraConfig.get().runtime.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Device: %s", device)

    if device.type == "cuda":
        log.info(
            "GPU: %s",
            torch.cuda.get_device_name(device),
        )

    # --------------------------------------------------------
    # MODEL SETUP
    # --------------------------------------------------------
    model = hy_utils.instantiate(cfg.model)
    model = model.to(device)
    model.eval()

    model_name = cfg.model._target_.split(".")[-1]

    log.info("Model: %s", model_name)

    # --------------------------------------------------------
    # DATASET SETUP
    # --------------------------------------------------------
    dataset_val = hy_utils.instantiate(
        cfg.dataset,
        **cfg.dataset.val,
    )

    sample = dataset_val[0]
    x = extract_input(sample, device)

    log.info("Input shape: %s", tuple(x.shape))

    # --------------------------------------------------------
    # OUTPUT INFORMATION
    # --------------------------------------------------------
    with torch.inference_mode():
        output = model(x)

        output_info = get_output_info(output)

        del output

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    log.info(
        "Output shape: %s",
        output_info["Output shape"],
    )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------
    model_info = get_model_info(model)

    # --------------------------------------------------------
    # INFERENCE BENCHMARK
    # --------------------------------------------------------
    runtime_metrics = benchmark_inference(
        model,
        x,
        warmup=warmup,
        iterations=iterations,
    )

    # --------------------------------------------------------
    # VIDEO THROUGHPUT
    # --------------------------------------------------------
    output_frames = output_info["Output frames"]

    runtime_metrics["Throughput (frames/s)"] = (
        runtime_metrics["Throughput (samples/s)"]
        * output_frames
    )

    # --------------------------------------------------------
    # COMPUTATIONAL COMPLEXITY
    # --------------------------------------------------------
    if run_complexity:
        complexity_metrics = measure_complexity(
            model,
            input_shape=tuple(x.shape[1:]),
        )
    else:
        complexity_metrics = {}

    # --------------------------------------------------------
    # HARDWARE INFORMATION
    # --------------------------------------------------------
    hardware_info = {
        "Device": str(device),
        "GPU": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "PyTorch version": torch.__version__,
        "CUDA version": (
            torch.version.cuda
            if device.type == "cuda"
            else None
        ),
        "Matmul precision": matmul_precision,
    }

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------
    results = {
        "Model": model_name,
        **hardware_info,
        "Input shape": str(tuple(x.shape)),
        **output_info,
        "Warmup iterations": warmup,
        "Benchmark iterations": iterations,
        **model_info,
        **complexity_metrics,
        **runtime_metrics,
    }

    save_results(results, output_dir)

    # --------------------------------------------------------
    # OPTIONAL PROFILER
    # --------------------------------------------------------
    if run_profiler:
        profile_model(model, x, output_dir)

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------
    del x, sample, dataset_val, model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    log.info("Benchmark finished!")

    return pd.DataFrame([results])
