from triton_adam import TritonAdamW
from triton_muon import TritonMuon

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal
from collections.abc import Iterable

@dataclass
class AdamConfig:
    lr: float = 1e-3
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-7
    weight_decay: float = 0.01


@dataclass 
class MuonConfig:
    lr : float = 1e-3
    beta : float = 0.95
    eps : float = 1e-7
    weight_decay : float = 0.1
    coeffs: tuple[float, float, float] = (3.445, -4.775, 2.0315)
    nesterov : bool = True
    ns_steps : int = 5
    adjust_lr_fn : Literal["original", "match_rms_adamw", "spectral_unclamped"] = "original"

class MuonAdamW(TritonMuon, TritonAdamW):
    def __init__(
        self,
        model_params: nn.Module | dict[str, nn.Parameter],
        muon_config: MuonConfig = MuonConfig(),  # noqa: B008
        adam_config: AdamConfig = AdamConfig(),  # noqa: B008
        parameter_split : Literal["auto", "explicit"] = "auto"
    ):
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

    def step(self):
        self.adamw.step()
        self.muon.step()

    def zero_grad(self):
        self.adamw.zero_grad()
        self.muon.zero_grad()

    def state_dict(self):
        return self.param_group 

    def load_state_dict(self, state_dict):
        if "adam" in state_dict:
            self.adamw.model_params = state_dict["adam"]
        if "muon" in state_dict:
            self.muon.model_params = state_dict["muon"]

    