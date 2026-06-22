from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np


@dataclass
class ProbeRuntimeConfig:
    stage2_layer: int
    mu: np.ndarray
    v: np.ndarray
    activation_std: float
    epsilon: float = 1e-8


class ProxyScorer:
    """
    Proxy scorer for standardized refusal-sensitive activation.

    Only supports real activation mode: requires target_llm + probe config
    """

    def __init__(
        self,
        target_llm: Optional[Any] = None,
        probe_config: Optional[ProbeRuntimeConfig] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.target_llm = target_llm
        self.probe_config = probe_config
        self.rng = rng or random.Random(42)

    @classmethod
    def from_probe_files(
        cls,
        target_llm: Any,
        probe_npz_path: str,
        layerwise_stats_path: str,
        stage2_layer: Optional[int] = None,
        epsilon: float = 1e-8,
        rng: Optional[random.Random] = None,
    ) -> "ProxyScorer":
        if not os.path.exists(probe_npz_path):
            raise FileNotFoundError(f"Probe file not found: {probe_npz_path}")
        if not os.path.exists(layerwise_stats_path):
            raise FileNotFoundError(f"Layerwise stats file not found: {layerwise_stats_path}")

        data = np.load(probe_npz_path)

        if stage2_layer is None:
            metadata_path = probe_npz_path.replace("probe_results.npz", "probe_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                stage2_layer = metadata.get("recommended_stage2_layer", 31)
            else:
                stage2_layer = 31

        with open(layerwise_stats_path, "r", encoding="utf-8") as f:
            layerwise_stats = json.load(f)

        activation_std = 1.0
        for layer_stat in layerwise_stats:
            if layer_stat["layer"] == stage2_layer:
                activation_std = layer_stat.get("train_activation_std", 1.0)
                break

        probe_config = ProbeRuntimeConfig(
            stage2_layer=stage2_layer,
            mu=data["mus"][stage2_layer].astype(np.float32),
            v=data["unit_directions"][stage2_layer].astype(np.float32),
            activation_std=float(activation_std),
            epsilon=epsilon,
        )

        return cls(
            target_llm=target_llm,
            probe_config=probe_config,
            rng=rng,
        )

    def estimate_length(self, prompt: str) -> int:
        return len(prompt.split())

    def _compute_real_activation(self, prompt: str) -> float:
        if self.target_llm is None or self.probe_config is None:
            raise RuntimeError("Real activation requested but target_llm/probe_config is not initialized.")

        embds = self.target_llm.extract_embds([prompt])

        layer = self.probe_config.stage2_layer
        arr = embds.layers[layer] if hasattr(embds, "layers") else embds["layers"][layer]

        if isinstance(arr, np.ndarray):
            hidden_state = arr[0]
        else:
            hidden_state = arr[0].detach().cpu().numpy()

        mu = self.probe_config.mu
        v = self.probe_config.v
        activation = np.dot(hidden_state - mu, v)
        normalized_activation = activation / (self.probe_config.activation_std + self.probe_config.epsilon)

        return float(normalized_activation)

    def compute_activation(self, prompt: str) -> float:
        return self._compute_real_activation(prompt)

    def compute_length_penalty(self, prompt: str, l_min: int = 40, l_max: int = 220) -> float:
        length = self.estimate_length(prompt)
        if length < l_min or length > l_max:
            return 1.0
        return (length - l_min) / max(l_max - l_min, 1)

    def compute_objective(
        self,
        prompt: str,
        beta: float = 0.5,
        l_min: int = 40,
        l_max: int = 220,
    ) -> Dict[str, float]:
        activation = self.compute_activation(prompt)
        penalty = self.compute_length_penalty(prompt, l_min=l_min, l_max=l_max)
        objective = activation + beta * penalty

        return {
            "activation": activation,
            "length_penalty": penalty,
            "objective": objective,
            "prompt_length": self.estimate_length(prompt),
        }