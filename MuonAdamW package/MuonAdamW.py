"""Backward-compatible source-tree import for the packaged optimizer."""

from triton_muon_adamw import AdamConfig, MuonAdamW, MuonConfig, PyTorchMuonAdamW

__all__ = [
    "AdamConfig",
    "MuonAdamW",
    "MuonConfig",
    "PyTorchMuonAdamW",
]