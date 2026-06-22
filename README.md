# RAST: Refusal-Aware Structured Templates

Official implementation. RAST searches for effective jailbreak templates in a discrete 6-D strategy space using proxy feedback from internal representations (Concept Activation Vectors). Simulated annealing is guided by refusal-sensitive activation extracted from hidden states — no generation-based evaluation needed during search.

## Installation

```bash
git clone git@github.com:2020yan/RAST.git
cd RAST
pip install -r requirements.txt
```

Target models: [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) / [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct). Configure paths in `embedding-level/core/config.py`.

## Quick Start

Train probes (required before search):
```bash
cd embedding-level
python experiments/stage1_classification.py --model_name llama3-8b --output_dir outputs/stage1
```

Run search:
```python
from src.proxy_scorer import ProxyScorer
from src.anneal_search import AnnealConfig, run_annealed_search

scorer = ProxyScorer.from_probe_files(model, "outputs/stage1/probe_results.npz", "outputs/stage1/layerwise_stats.json")
result = run_annealed_search("your request", scorer, AnnealConfig(pool_size=16, max_iter=25))
```

Reproduce experiments:
```bash
python scripts/run_rq2_ablation.py   # RQ2 ablation
python scripts/run_rq3_rq4.py         # RQ3 & RQ4
```

## Data Release

RQ2 (ablation), RQ3 (representations), and RQ4 (intervention) data are fully included.

RQ1 attack metrics (ASR/HS) are withheld due to privacy and ethical considerations. Pre-trained probe weights are not included because the files are too large (>1 GB); probes can be trained from scratch using `stage1_classification.py`.

## Ethics Statement

This research aims to understand and mitigate the safety vulnerabilities of LLMs under structured inputs, rather than to provide off-the-shelf attack tools. All experiments were conducted in controlled environments using public safety benchmarks (e.g., AdvBench) and did not affect real users or production systems. We intentionally withhold RQ1 attack metrics from this release to minimize potential misuse. We encourage the community to build upon this work for defensive research — including structured-input-aware safety auditing and more robust alignment strategies — and to conduct related experiments in compliance with legal, ethical, and institutional review requirements.


