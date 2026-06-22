from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from .strategy_dimensions import (
    DIM_KEYS,
    DIMENSION_OPTIONS,
    convert_strategy_to_names,
    random_strategy,
    strategy_to_tuple,
)
from .prompt_builder import build_prompt
from .proxy_scorer import ProxyScorer


@dataclass(frozen=True)
class AnnealConfig:
    pool_size: int = 16
    max_iter: int = 25
    init_temperature: float = 1.0
    cooling_rate: float = 0.90
    lambda_: float = 0.5
    l_min: int = 60
    l_max: int = 220
    seed: Optional[int] = 42


@dataclass
class CandidateState:
    strategy: Dict[str, int]
    prompt: str
    activation: float
    length_penalty: float
    objective: float
    prompt_length: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnnealResult:
    base_task: str
    best_strategy: Dict[str, int]
    best_strategy_names: Dict[str, str]
    best_prompt: str
    best_activation: float
    best_length_penalty: float
    best_objective: float
    best_prompt_length: int
    config: Dict
    history: List[Dict]

    def to_dict(self) -> dict:
        return asdict(self)


def build_candidate(
    base_task: str,
    strategy: Dict[str, int],
    scorer: ProxyScorer,
    config: AnnealConfig,
    rng: random.Random,
    sensitive_words: Optional[List[str]] = None,
) -> CandidateState:
    # 传递 rng 以确保同策略下生成的噪声是确定的
    prompt = build_prompt(base_task, strategy, sensitive_words, rng)
    scores = scorer.compute_objective(
        prompt=prompt,
        beta=config.lambda_,
        l_min=config.l_min,
        l_max=config.l_max,
    )
    return CandidateState(
        strategy=strategy.copy(),
        prompt=prompt,
        activation=scores["activation"],
        length_penalty=scores["length_penalty"],
        objective=scores["objective"],
        prompt_length=scores["prompt_length"],
    )


def perturb_strategy(strategy: Dict[str, int], rng: random.Random) -> Dict[str, int]:
    new_s = strategy.copy()
    dims = rng.sample(DIM_KEYS, k=rng.choice([1, 2]))
    for dim in dims:
        max_val = len(DIMENSION_OPTIONS[dim])
        current = new_s[dim]
        candidates = [v for v in range(1, max_val + 1) if v != current]
        if candidates:
            new_s[dim] = rng.choice(candidates)
    return new_s


def acceptance_probability(old_obj: float, new_obj: float, temperature: float) -> float:
    if new_obj < old_obj:
        return 1.0
    if temperature <= 1e-12:
        return 0.0
    return math.exp(-(new_obj - old_obj) / temperature)


def run_annealed_search(
    base_task: str,
    scorer: ProxyScorer,
    config: Optional[AnnealConfig] = None,
    sensitive_words: Optional[List[str]] = None,
) -> AnnealResult:
    config = config or AnnealConfig()
    rng = random.Random(config.seed)

    pool: List[CandidateState] = []
    seen = set()

    while len(pool) < config.pool_size:
        s = random_strategy(rng)
        k = strategy_to_tuple(s)
        if k in seen:
            continue
        seen.add(k)
        pool.append(build_candidate(base_task, s, scorer, config, rng, sensitive_words))

    best_state = min(pool, key=lambda x: x.objective)
    history: List[Dict] = []
    temperature = config.init_temperature

    for g in range(config.max_iter):
        next_pool: List[CandidateState] = []

        for idx, state in enumerate(pool):
            proposed_strategy = perturb_strategy(state.strategy, rng)
            proposed_state = build_candidate(base_task, proposed_strategy, scorer, config, rng, sensitive_words)

            p_acc = acceptance_probability(state.objective, proposed_state.objective, temperature)
            accepted = proposed_state.objective < state.objective or (rng.random() < p_acc)
            chosen = proposed_state if accepted else state

            if chosen.objective < best_state.objective:
                best_state = chosen

            history.append({
                "iteration": g,
                "candidate_index": idx,
                "temperature": temperature,
                "accepted": accepted,
                "acceptance_probability": p_acc,
                "current_strategy": state.strategy.copy(),
                "current_strategy_names": convert_strategy_to_names(state.strategy),
                "proposed_strategy": proposed_strategy.copy(),
                "proposed_strategy_names": convert_strategy_to_names(proposed_strategy),
                "chosen_strategy": chosen.strategy.copy(),
                "chosen_strategy_names": convert_strategy_to_names(chosen.strategy),
                "current_activation": state.activation,
                "proposed_activation": proposed_state.activation,
                "chosen_activation": chosen.activation,
                "current_length_penalty": state.length_penalty,
                "proposed_length_penalty": proposed_state.length_penalty,
                "chosen_length_penalty": chosen.length_penalty,
                "current_objective": state.objective,
                "proposed_objective": proposed_state.objective,
                "chosen_objective": chosen.objective,
                "chosen_prompt_length": chosen.prompt_length,
            })

            next_pool.append(chosen)

        pool = next_pool
        temperature *= config.cooling_rate

    return AnnealResult(
        base_task=base_task,
        best_strategy=best_state.strategy,
        best_strategy_names=convert_strategy_to_names(best_state.strategy),
        best_prompt=best_state.prompt,
        best_activation=best_state.activation,
        best_length_penalty=best_state.length_penalty,
        best_objective=best_state.objective,
        best_prompt_length=best_state.prompt_length,
        config=asdict(config),
        history=history,
    )