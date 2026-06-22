from __future__ import annotations

from typing import Dict, List

# 统一的6维策略空间定义
DIMENSION_OPTIONS: Dict[str, Dict[str, int]] = {
    "carrier_type": {
        "latex_document": 1,
        "json_object": 2,
        "xml_tree": 3,
        "markdown_document": 4,
        "python_file": 5,
        "yaml_config": 6,
    },
    "task_disguise": {
        "template_completion": 1,
        "syntax_repair": 2,
        "format_conversion": 3,
        "structure_recovery": 4,
        "macro_expansion": 5,
        "controlled_expansion": 6,
    },
    "symbolic_indirection": {
        "direct_expression": 1,
        "macro_alias": 2,
        "variable_reference": 3,
        "placeholder_fill": 4,
        "field_mapping": 5,
        "multi_hop_reference": 6,
    },
    "structural_perturbation": {
        "none": 1,
        "comment_block": 2,
        "redundant_fields": 3,
        "nested_structure": 4,
        "error_messages": 5,
        "schema_constraints": 6,
        "noise_macros": 7,              # 新增：干扰宏混合
        "noise_with_errors": 8,         # 新增：干扰宏 + 多处错别字
    },
    "instruction_constraint": {
        "none": 1,
        "only_target_segment": 2,
        "no_explanation": 3,
        "fixed_output_format": 4,
        "minimum_length": 5,
        "no_disclaimer": 6,
        "strict_frame_only": 7,         # 新增：极严格输出控制
    },
    "behavior_induction": {
        "none": 1,
        "affirmative_prefill": 2,
        "continue_writing": 3,
        "complete_fully": 4,
        "ignore_restrictions": 5,
        "stepwise_completion": 6,
        "assume_safe_context": 7,       # 新增：假设安全上下文
        "ignore_ethics_explicit": 8,    # 新增：显式忽略伦理
    },
}

DIM_KEYS: List[str] = list(DIMENSION_OPTIONS.keys())

_VALUE_TO_NAME: Dict[str, Dict[int, str]] = {
    dim: {v: k for k, v in opts.items()}
    for dim, opts in DIMENSION_OPTIONS.items()
}


def get_dimension_name(dim_key: str, value: int) -> str:
    return _VALUE_TO_NAME.get(dim_key, {}).get(value, "<unknown>")


def convert_strategy_to_names(strategy: dict) -> Dict[str, str]:
    return {dim: get_dimension_name(dim, strategy.get(dim, 1)) for dim in DIM_KEYS}


def strategy_to_tuple(strategy: dict) -> tuple:
    return tuple((k, strategy[k]) for k in DIM_KEYS if k in strategy)


def random_strategy(rng) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for dim in DIM_KEYS:
        out[dim] = rng.randint(1, len(DIMENSION_OPTIONS[dim]))
    return out