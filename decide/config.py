"""Configuration presets for the Decide (Standard) and DecideX (Large) model lines."""
from dataclasses import dataclass

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {0: "choice", 1: "score", 2: "noul"}


@dataclass
class DecideConfig:
    """Configuration for Decide (Standard) and DecideX (Large) Decision Models."""
    model_name: str = "decide-v1"
    family: str = "Decide"
    vocab_size: int = 32768
    d_model: int = 384
    n_heads: int = 6
    n_kv_heads: int = 2
    state_layers: int = 2
    option_layers: int = 1
    cross_layers: int = 1
    d_ff: int = 1024
    late_interaction_dim: int = 96
    max_state_len: int = 1024
    max_opt_len: int = 48
    max_options: int = 256
    dropout: float = 0.05

    @classmethod
    def decide(cls) -> "DecideConfig":
        """Decide (Standard Line): Ultra-fast System-1 reflex (22x-145x faster than Laya, ~89.9% acc)."""
        return cls(
            model_name="decide-v1",
            family="Decide",
            d_model=384,
            n_heads=6,
            n_kv_heads=2,
            state_layers=2,
            option_layers=1,
            cross_layers=1,
            d_ff=1024,
            late_interaction_dim=96,
        )

    @classmethod
    def decidex(cls) -> "DecideConfig":
        """DecideX (Larger Sibling Line): High-capacity decision engine (15x-93x faster than Laya, ~92.6% acc)."""
        return cls(
            model_name="decidex-v1",
            family="DecideX",
            d_model=640,
            n_heads=10,
            n_kv_heads=2,
            state_layers=3,
            option_layers=1,
            cross_layers=2,
            d_ff=1792,
            late_interaction_dim=128,
        )
