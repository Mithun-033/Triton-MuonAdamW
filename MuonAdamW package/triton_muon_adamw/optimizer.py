from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
import torch.nn as nn  #noqa: PLR0402

from .adamw.triton_adam import TritonAdamW
from .muon.triton_muon import TritonMuon


ParameterGroups: TypeAlias = dict[str, list[nn.Parameter]]


@dataclass
class AdamConfig:
    """Hyperparameters for the AdamW branch of :class:`MuonAdamW`.

    Parameters
    ----------
    lr:
        Learning rate used by AdamW.
    betas:
        Coefficients used for the first- and second-moment running averages.
    eps:
        Term added to the denominator for numerical stability.
    weight_decay:
        Decoupled AdamW weight decay coefficient.
    """

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-7
    weight_decay: float = 0.01


@dataclass
class MuonConfig:
    """Hyperparameters for the Muon branch of :class:`MuonAdamW`.

    Parameters
    ----------
    lr:
        Learning rate used by Muon.
    beta:
        Momentum coefficient used by the Muon update.
    eps:
        Term used to avoid division by zero during normalization.
    weight_decay:
        Decoupled weight decay coefficient.
    coeffs:
        Coefficients for the Newton-Schulz polynomial approximation.
    nesterov:
        Whether to use Nesterov momentum.
    ns_steps:
        Number of Newton-Schulz iterations used to orthogonalize updates.
    adjust_lr_fn:
        Learning-rate adjustment strategy for Muon matrix updates.
    """

    lr: float = 1e-3
    beta: float = 0.95
    eps: float = 1e-7
    weight_decay: float = 0.1
    coeffs: tuple[float, float, float] = (3.445, -4.775, 2.0315)
    nesterov: bool = True
    ns_steps: int = 5
    adjust_lr_fn: Literal["original", "match_rms_adamw", "spectral_unclamped"] = "original"


class MuonAdamW(TritonMuon, TritonAdamW):
    """A fused optimizer that routes model parameters to Muon or AdamW.

    Two-dimensional parameters are updated with Muon, except input embeddings
    and output heads. All other trainable parameters, including biases,
    normalization weights, and embeddings, use the Triton AdamW branch.

    Parameters
    ----------
    model_params:
        The model to inspect when ``parameter_split="auto"`` or a dictionary
        with ``"adam"`` and ``"muon"`` parameter lists when using explicit
        splitting.
    muon_config:
        Hyperparameters for the Muon branch.
    adam_config:
        Hyperparameters for the AdamW branch.
    parameter_split:
        ``"auto"`` classifies parameters by shape and excludes embeddings.
        ``"explicit"`` uses the two lists supplied in ``model_params``.

    Notes
    -----
    The optimizer expects CUDA tensors because its update kernels are written
    in Triton. In auto mode, the model must provide
    ``get_input_embeddings()`` and ``get_output_embeddings()`` methods.
    """

    def __init__(
        self,
        model_params: nn.Module | ParameterGroups,
        muon_config: MuonConfig = MuonConfig(),  # noqa: B008
        adam_config: AdamConfig = AdamConfig(),  # noqa: B008
        parameter_split: Literal["auto", "explicit"] = "auto",
    ) -> None:
        adam_params = []
        muon_params = []
        if parameter_split == "explicit" and not isinstance(model_params, dict):
            raise ValueError("When parameter_split is 'explicit', params must be a dictionary with keys 'adam' and 'muon' and their corresponding parameters.")

        if parameter_split == "auto" and isinstance(model_params, dict):
            raise ValueError("When parameter_split is 'auto', params must be an nn.Module (ie the model itself), not a dictionary.")

        if parameter_split == "explicit":
            adam_params = model_params.get("adam", [])
            muon_params = model_params.get("muon", [])

        elif parameter_split == "auto":
            embed = model_params.get_input_embeddings().weight
            head = model_params.get_output_embeddings().weight if model_params.get_output_embeddings() is not None else None

            seen_ptrs= set()
            for p in model_params.parameters():
                if not p.requires_grad or p.data_ptr() in seen_ptrs:
                    continue
                seen_ptrs.add(p.data_ptr())

                if p.ndim == 2 and p is not embed and p is not head:
                    muon_params.append(p)
                else:
                    adam_params.append(p)

        self.adamw = TritonAdamW(
            params=adam_params,
            lr=adam_config.lr,
            betas=adam_config.betas,
            weight_decay=adam_config.weight_decay,
            eps=adam_config.eps,
        )
        self.muon = TritonMuon(
            params=muon_params,
            lr=muon_config.lr,
            beta=muon_config.beta,
            eps=muon_config.eps,
            weight_decay=muon_config.weight_decay,
            coeffs=muon_config.coeffs,
            nesterov=muon_config.nesterov,
            ns_steps=muon_config.ns_steps,
            adjust_lr_fn=muon_config.adjust_lr_fn,
        )

        self.param_group = {
            "adam": self.adamw.model_params,
            "muon": self.muon.model_params,
        }

    def step(self) -> None:
        """Apply one update with both the AdamW and Muon branches."""
        self.adamw.step()
        self.muon.step()

    def zero_grad(self) -> None:
        """Reset gradients for parameters managed by both branches."""
        self.adamw.zero_grad()
        self.muon.zero_grad()

    def state_dict(self) -> dict[str, object]:
        """Return the serialized state for the AdamW and Muon branches."""
        return self.param_group

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Load branch state previously returned by :meth:`state_dict`."""
        if "adam" in state_dict:
            self.adamw.model_params = state_dict["adam"]
        if "muon" in state_dict:
            self.muon.model_params = state_dict["muon"]


class PyTorchMuonAdamW:
    """PyTorch reference implementation for comparing optimizer performance.

    This class mirrors :class:`MuonAdamW` but delegates updates to
    ``torch.optim.AdamW`` and ``torch.optim.Muon``. It is intended for
    correctness and benchmark comparisons, not for the Triton fast path.
    """

    def __init__(
        self,
        model_params: nn.Module | ParameterGroups,
        muon_config: MuonConfig = MuonConfig(),  # noqa: B008
        adam_config: AdamConfig = AdamConfig(),  # noqa: B008
        parameter_split: Literal["auto", "explicit"] = "auto",
    ) -> None:
        adam_params = []
        muon_params = []
        if parameter_split == "explicit" and not isinstance(model_params, dict):
            raise ValueError("When parameter_split is 'explicit', params must be a dictionary with keys 'adam' and 'muon' and their corresponding parameters.")

        if parameter_split == "auto" and isinstance(model_params, dict):
            raise ValueError("When parameter_split is 'auto', params must be an nn.Module (ie the model itself), not a dictionary.")

        if parameter_split == "explicit":
            adam_params = model_params.get("adam", [])
            muon_params = model_params.get("muon", [])

        elif parameter_split == "auto":
            embed = model_params.get_input_embeddings().weight
            head = model_params.get_output_embeddings().weight if model_params.get_output_embeddings() is not None else None

            seen_ptrs = set()
            for p in model_params.parameters():
                if not p.requires_grad or p.data_ptr() in seen_ptrs:
                    continue
                seen_ptrs.add(p.data_ptr())

                if p.ndim == 2 and p is not embed and p is not head:
                    muon_params.append(p)
                else:
                    adam_params.append(p)

        self.adamw = torch.optim.AdamW(
            params=adam_params,
            lr=adam_config.lr,
            betas=adam_config.betas,
            weight_decay=adam_config.weight_decay,
            eps=adam_config.eps,
            fused=True,
        )
        self.muon = torch.optim.Muon(
            params=muon_params,
            lr=muon_config.lr,
            momentum=muon_config.beta,
            weight_decay=muon_config.weight_decay,
            nesterov=muon_config.nesterov,
            ns_steps=muon_config.ns_steps,
            adjust_lr_fn=muon_config.adjust_lr_fn,
        )

    def step(self) -> None:
        """Apply one reference AdamW and Muon update."""
        self.adamw.step()
        self.muon.step()

    def zero_grad(self) -> None:
        """Reset gradients for both reference optimizers."""
        self.adamw.zero_grad()
        self.muon.zero_grad()

    def state_dict(self) -> dict[str, object]:
        """Return the serialized state of both reference optimizers."""
        return {
            "adam": self.adamw.state_dict(),
            "muon": self.muon.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Load a state dictionary produced by :meth:`state_dict`."""
        if "adam" in state_dict:
            self.adamw.load_state_dict(state_dict["adam"])
        if "muon" in state_dict:
            self.muon.load_state_dict(state_dict["muon"])

    