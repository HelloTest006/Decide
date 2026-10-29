"""
Decide & DecideX Inference Runtime.
Over 10x faster than Laya (20x-85x on GPU) with 89-93% accuracy on typed decisions.
"""
from pathlib import Path
from typing import Optional, Union
import torch

from .config import DecideConfig, QTYPES, QTYPE_NAMES
from .model import DecideModel, MonotonicCardinalityCalibrator
from .tokenizer import FastDecideTokenizer, render_options_with_keys
from .agent import DecideAgent
from .router import DecideRouter
from .runtime import DecideRuntime, StructuredStateEnricher, BoundedLRUCache
from .server import serve_runtime
from .training import (
    rlcd_v2_loss,
    generate_multi_workflow_dataset,
    collate_training_batch,
    evaluate_model,
    train_decide_line,
)

__version__ = "1.0.0"


def load(
    model_line: str = "decide",
    weights_dir: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> DecideAgent:
    """Load a trained Decide (`decide`) or DecideX (`decidex`) System-1 Decision Agent."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    m = model_line.lower().strip()
    is_x = m in ("decidex", "decide-x", "large", "x", "decidex-v1")
    cfg = DecideConfig.decidex() if is_x else DecideConfig.decide()
    model = DecideModel(cfg)

    base_weights = Path(weights_dir) if weights_dir else (Path(__file__).resolve().parent.parent / "weights")
    sub = "decidex" if is_x else "decide"
    ckpt = base_weights / sub / "model.safetensors"
    if ckpt.exists():
        from safetensors.torch import load_file
        sd = {k: v.float() for k, v in load_file(str(ckpt)).items()}
        model.load_state_dict(sd, strict=True)

    model = model.to(dev).eval()
    return DecideAgent(model, FastDecideTokenizer(cfg.vocab_size), dev)


Router = DecideRouter

__all__ = [
    "DecideConfig",
    "DecideModel",
    "MonotonicCardinalityCalibrator",
    "FastDecideTokenizer",
    "DecideAgent",
    "DecideRouter",
    "DecideRuntime",
    "StructuredStateEnricher",
    "BoundedLRUCache",
    "serve_runtime",
    "rlcd_v2_loss",
    "generate_multi_workflow_dataset",
    "collate_training_batch",
    "evaluate_model",
    "train_decide_line",
    "Router",
    "load",
    "QTYPES",
    "QTYPE_NAMES",
]
