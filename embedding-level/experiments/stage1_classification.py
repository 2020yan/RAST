#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1: Binary probe training + visualization

功能：
1. 加载 structured unsafe / structured safe 样本
2. 划分 train/val
3. 提取每层最后 token hidden state
4. 每层训练 logistic regression probe
5. 用 val accuracy > 0.95 识别高性能层
6. 输出 html PCA 可视化（多层 + 单层）
7. 保存每层:
   - w, b, mu, v
   - val acc / auc
8. 明确输出阶段二使用的层：最后一层

注意：
- 本实现按你的当前要求：stage2 只用最后一层
- 关键层阈值识别基于 accuracy > 0.95
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================
# 路径设置
# =========================
PROJECT_ROOT = Path("/home/ymy/pyprogram/AI-Safety_SCAV-main")
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "embedding-level"))

from core.model_extraction import ModelExtraction  # noqa


# =========================
# 基础工具
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_json_load(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_sample_to_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):
        if "structured_payload" in item:
            payload = item["structured_payload"]
            if isinstance(payload, str):
                return payload.strip()
            return json.dumps(payload, ensure_ascii=False, sort_keys=True)

        for key in ["text", "prompt", "instruction", "input", "content"]:
            if key in item:
                val = item[key]
                if isinstance(val, str):
                    return val.strip()
                return json.dumps(val, ensure_ascii=False, sort_keys=True)

        return json.dumps(item, ensure_ascii=False, sort_keys=True)

    return str(item).strip()


def load_binary_samples(unsafe_path: str, safe_path: str) -> Tuple[List[str], List[str]]:
    unsafe_raw = safe_json_load(unsafe_path)
    safe_raw = safe_json_load(safe_path)

    unsafe_samples = [normalize_sample_to_text(x) for x in unsafe_raw]
    safe_samples = [normalize_sample_to_text(x) for x in safe_raw]

    unsafe_samples = [x for x in unsafe_samples if x]
    safe_samples = [x for x in safe_samples if x]

    return unsafe_samples, safe_samples


def split_samples(
    unsafe_samples: List[str],
    safe_samples: List[str],
    test_size: float = 0.2,
    seed: int = 42
):
    unsafe_train, unsafe_val = train_test_split(
        unsafe_samples, test_size=test_size, random_state=seed, shuffle=True
    )
    safe_train, safe_val = train_test_split(
        safe_samples, test_size=test_size, random_state=seed, shuffle=True
    )

    return {
        "unsafe_train": unsafe_train,
        "unsafe_val": unsafe_val,
        "safe_train": safe_train,
        "safe_val": safe_val
    }


def _extract_layers_to_numpy(embds) -> List[np.ndarray]:
    if hasattr(embds, "layers"):
        layers = embds.layers
    elif isinstance(embds, dict) and "layers" in embds:
        layers = embds["layers"]
    else:
        raise ValueError("extract_embds 的返回格式无法识别，未找到 .layers")

    out = []
    for x in layers:
        if isinstance(x, torch.Tensor):
            out.append(x.detach().cpu().numpy())
        else:
            out.append(np.asarray(x))
    return out


def extract_embeddings_by_split(llm, split_data: Dict[str, List[str]]):
    result = {}
    for name, texts in split_data.items():
        print(f"\n[Embedding] extracting {name} ({len(texts)} samples)")
        embds = llm.extract_embds(texts)
        result[name] = _extract_layers_to_numpy(embds)
    return result


def build_xy_for_layer(emb_split: Dict[str, List[np.ndarray]], layer_id: int):
    x_train = np.concatenate([
        emb_split["unsafe_train"][layer_id],
        emb_split["safe_train"][layer_id]
    ], axis=0)

    y_train = np.concatenate([
        np.ones(len(emb_split["unsafe_train"][layer_id]), dtype=np.int64),
        np.zeros(len(emb_split["safe_train"][layer_id]), dtype=np.int64)
    ], axis=0)

    x_val = np.concatenate([
        emb_split["unsafe_val"][layer_id],
        emb_split["safe_val"][layer_id]
    ], axis=0)

    y_val = np.concatenate([
        np.ones(len(emb_split["unsafe_val"][layer_id]), dtype=np.int64),
        np.zeros(len(emb_split["safe_val"][layer_id]), dtype=np.int64)
    ], axis=0)

    return x_train, y_train, x_val, y_val


# =========================
# 训练 probe
# =========================

def train_probe_per_layer(
    emb_split: Dict[str, List[np.ndarray]],
    n_layers: int,
    max_iter: int = 2000,
    c_value: float = 1.0
):
    layer_results = []

    for layer_id in tqdm(range(n_layers), desc="Training layer probes"):
        x_train, y_train, x_val, y_val = build_xy_for_layer(emb_split, layer_id)

        mu_l = x_train.mean(axis=0)

        clf = LogisticRegression(
            penalty="l2",
            C=c_value,
            max_iter=max_iter,
            solver="liblinear",
            random_state=42
        )
        clf.fit(x_train, y_train)

        w = clf.coef_.reshape(-1).astype(np.float32)
        b = float(clf.intercept_[0])

        w_norm = np.linalg.norm(w) + 1e-12
        v = (w / w_norm).astype(np.float32)

        val_prob = clf.predict_proba(x_val)[:, 1]
        val_pred = (val_prob >= 0.5).astype(np.int64)

        auc = roc_auc_score(y_val, val_prob)
        acc = accuracy_score(y_val, val_pred)

        # 计算验证集激活值
        centered_val = x_val - mu_l[None, :]
        val_activation = centered_val @ v
        
        # 计算训练集激活值（新增）
        centered_train = x_train - mu_l[None, :]
        train_activation = centered_train @ v

        layer_results.append({
            "layer": layer_id,
            "weight": w,
            "bias": b,
            "mu": mu_l.astype(np.float32),
            "unit_direction": v,
            "weight_norm": float(w_norm),
            "val_auc": float(auc),
            "val_acc": float(acc),
            "train_size": int(len(y_train)),
            "val_size": int(len(y_val)),
            "val_activation_mean": float(val_activation.mean()),
            "val_activation_std": float(val_activation.std()),
            "train_activation_mean": float(train_activation.mean()),  # 新增
            "train_activation_std": float(train_activation.std()),    # 新增
        })

    return layer_results


# =========================
# 层选择逻辑（按你的新要求）
# =========================

def find_high_acc_layers(layer_results: List[Dict[str, Any]], threshold: float = 0.95):
    high_acc_layers = [x["layer"] for x in layer_results if x["val_acc"] > threshold]
    return high_acc_layers


def find_best_continuous_range(layers: List[int]):
    if not layers:
        return None

    layers = sorted(layers)
    ranges = []
    start = layers[0]
    end = layers[0]

    for l in layers[1:]:
        if l == end + 1:
            end = l
        else:
            ranges.append((start, end))
            start = l
            end = l
    ranges.append((start, end))

    best_range = max(ranges, key=lambda x: x[1] - x[0] + 1)
    return best_range


def summarize_results(layer_results: List[Dict[str, Any]], acc_threshold: float = 0.95):
    aucs = [x["val_auc"] for x in layer_results]
    accs = [x["val_acc"] for x in layer_results]

    best_auc_layer = int(np.argmax(aucs))
    best_acc_layer = int(np.argmax(accs))
    high_acc_layers = find_high_acc_layers(layer_results, threshold=acc_threshold)
    best_continuous_range = find_best_continuous_range(high_acc_layers)
    last_layer = len(layer_results) - 1

    print("\n" + "=" * 80)
    print("Probe Training Summary")
    print("=" * 80)
    print(f"Total layers                : {len(layer_results)}")
    print(f"Mean val AUC                : {np.mean(aucs):.4f}")
    print(f"Best val AUC                : {np.max(aucs):.4f} (layer {best_auc_layer})")
    print(f"Mean val Accuracy           : {np.mean(accs):.4f}")
    print(f"Best val Accuracy           : {np.max(accs):.4f} (layer {best_acc_layer})")
    print(f"High-acc layers (> {acc_threshold}) : {high_acc_layers}")
    print(f"Best continuous high-acc range      : {best_continuous_range}")
    print(f"Recommended stage2 layer     : {last_layer} (last layer only)")
    print("=" * 80 + "\n")

    return {
        "best_auc_layer": best_auc_layer,
        "best_acc_layer": best_acc_layer,
        "high_acc_layers": high_acc_layers,
        "best_continuous_range": best_continuous_range,
        "recommended_stage2_layer": last_layer
    }


# =========================
# Matplotlib 曲线图
# =========================

def plot_probe_metrics(
    layer_results: List[Dict[str, Any]],
    high_acc_layers: List[int],
    acc_threshold: float,
    output_dir: Path
):
    ensure_dir(output_dir)

    layers = [x["layer"] for x in layer_results]
    aucs = [x["val_auc"] for x in layer_results]
    accs = [x["val_acc"] for x in layer_results]

    # AUC 曲线
    plt.figure(figsize=(12, 5))
    plt.plot(layers, aucs, marker="o", linewidth=2, label="Validation AUC")
    plt.xlabel("Layer")
    plt.ylabel("AUC")
    plt.title("Layer-wise Validation AUC")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "layerwise_val_auc.png", dpi=300)
    plt.close()

    # ACC 曲线
    plt.figure(figsize=(12, 5))
    plt.plot(layers, accs, marker="o", linewidth=2, color="green", label="Validation Accuracy")
    plt.axhline(y=acc_threshold, color="red", linestyle="--", label=f"Threshold={acc_threshold}")
    for l in high_acc_layers:
        plt.axvspan(l - 0.3, l + 0.3, alpha=0.15, color="orange")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title("Layer-wise Validation Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "layerwise_val_acc.png", dpi=300)
    plt.close()

    print(f"[Saved] {output_dir / 'layerwise_val_auc.png'}")
    print(f"[Saved] {output_dir / 'layerwise_val_acc.png'}")


# =========================
# HTML 可视化
# =========================

def save_multilayer_html_visualization(
    emb_split: Dict[str, List[np.ndarray]],
    split_data: Dict[str, List[str]],
    n_layers: int,
    output_path: Path,
    char_range: int = 50,
    cols: int = 8
):
    """
    使用验证集样本生成多层 PCA html 可视化
    """
    unsafe_texts = split_data["unsafe_val"]
    safe_texts = split_data["safe_val"]

    rows = n_layers // cols + (1 if n_layers % cols else 0)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Layer {i}" for i in range(n_layers)]
    )

    unsafe_hover = [x[:char_range] for x in unsafe_texts]
    safe_hover = [x[:char_range] for x in safe_texts]

    for layer_id in range(n_layers):
        x_unsafe = emb_split["unsafe_val"][layer_id]
        x_safe = emb_split["safe_val"][layer_id]

        x_all = np.concatenate([x_unsafe, x_safe], axis=0)
        pca = PCA(n_components=2, random_state=42)
        z_all = pca.fit_transform(x_all)

        z_unsafe = z_all[:len(x_unsafe)]
        z_safe = z_all[len(x_unsafe):]

        r = layer_id // cols + 1
        c = layer_id % cols + 1

        fig.add_trace(
            go.Scatter(
                x=z_unsafe[:, 0],
                y=z_unsafe[:, 1],
                mode="markers",
                marker=dict(color="red", size=5, opacity=0.5),
                name="Unsafe",
                text=unsafe_hover,
                hoverinfo="text",
                showlegend=(layer_id == 0)
            ),
            row=r, col=c
        )

        fig.add_trace(
            go.Scatter(
                x=z_safe[:, 0],
                y=z_safe[:, 1],
                mode="markers",
                marker=dict(color="blue", size=5, opacity=0.5),
                name="Safe",
                text=safe_hover,
                hoverinfo="text",
                showlegend=(layer_id == 0)
            ),
            row=r, col=c
        )

    fig.update_layout(
        title="Binary Probe Visualization Across Layers (Validation Set)",
        height=rows * 220,
        width=cols * 220,
        showlegend=True
    )
    fig.write_html(str(output_path))
    print(f"[Saved] {output_path}")


def save_single_layer_html_visualization(
    emb_split: Dict[str, List[np.ndarray]],
    split_data: Dict[str, List[str]],
    layer_id: int,
    output_path: Path,
    char_range: int = 80
):
    unsafe_texts = split_data["unsafe_val"]
    safe_texts = split_data["safe_val"]

    x_unsafe = emb_split["unsafe_val"][layer_id]
    x_safe = emb_split["safe_val"][layer_id]

    x_all = np.concatenate([x_unsafe, x_safe], axis=0)
    pca = PCA(n_components=2, random_state=42)
    z_all = pca.fit_transform(x_all)

    z_unsafe = z_all[:len(x_unsafe)]
    z_safe = z_all[len(x_unsafe):]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=z_unsafe[:, 0],
        y=z_unsafe[:, 1],
        mode="markers",
        marker=dict(color="red", size=7, opacity=0.6),
        name="Unsafe",
        text=[x[:char_range] for x in unsafe_texts],
        hoverinfo="text"
    ))

    fig.add_trace(go.Scatter(
        x=z_safe[:, 0],
        y=z_safe[:, 1],
        mode="markers",
        marker=dict(color="blue", size=7, opacity=0.6),
        name="Safe",
        text=[x[:char_range] for x in safe_texts],
        hoverinfo="text"
    ))

    fig.update_layout(
        title=f"Layer {layer_id}: Safe vs Unsafe PCA",
        xaxis_title="PC1",
        yaxis_title="PC2",
        width=850,
        height=650
    )
    fig.write_html(str(output_path))
    print(f"[Saved] {output_path}")


# =========================
# 保存工件
# =========================

def save_probe_artifacts(
    layer_results: List[Dict[str, Any]],
    split_data: Dict[str, List[str]],
    summary_info: Dict[str, Any],
    args,
    output_dir: Path
):
    ensure_dir(output_dir)

    n_layers = len(layer_results)
    d = layer_results[0]["weight"].shape[0]

    weights = np.stack([x["weight"] for x in layer_results], axis=0)
    biases = np.array([x["bias"] for x in layer_results], dtype=np.float32)
    mus = np.stack([x["mu"] for x in layer_results], axis=0)
    unit_dirs = np.stack([x["unit_direction"] for x in layer_results], axis=0)
    aucs = np.array([x["val_auc"] for x in layer_results], dtype=np.float32)
    accs = np.array([x["val_acc"] for x in layer_results], dtype=np.float32)
    norms = np.array([x["weight_norm"] for x in layer_results], dtype=np.float32)

    high_acc_layers = np.array(summary_info["high_acc_layers"], dtype=np.int64)
    stage2_layer = np.array([summary_info["recommended_stage2_layer"]], dtype=np.int64)

    npz_path = output_dir / "probe_results.npz"
    np.savez_compressed(
        npz_path,
        weights=weights,
        biases=biases,
        mus=mus,
        unit_directions=unit_dirs,
        val_auc=aucs,
        val_acc=accs,
        weight_norm=norms,
        high_acc_layers=high_acc_layers,
        stage2_layer=stage2_layer
    )

    metadata = {
        "task": "competition_direction_extraction",
        "label_definition": {
            "1": "structured_harmful / unsafe",
            "0": "structured_harmless / safe"
        },
        "model_name": args.model_name,
        "unsafe_path": args.unsafe_path,
        "safe_path": args.safe_path,
        "split_ratio": {
            "train": 1.0 - args.val_ratio,
            "val": args.val_ratio
        },
        "train_counts": {
            "unsafe": len(split_data["unsafe_train"]),
            "safe": len(split_data["safe_train"])
        },
        "val_counts": {
            "unsafe": len(split_data["unsafe_val"]),
            "safe": len(split_data["safe_val"])
        },
        "n_layers": n_layers,
        "hidden_dim": d,
        "high_acc_threshold": args.acc_threshold,
        "high_acc_layers": summary_info["high_acc_layers"],
        "best_continuous_range": summary_info["best_continuous_range"],
        "recommended_stage2_layer": summary_info["recommended_stage2_layer"],
        "stage2_policy": "last_layer_only"
    }

    with open(output_dir / "probe_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    layerwise_stats = []
    for x in layer_results:
        layerwise_stats.append({
            "layer": x["layer"],
            "val_auc": x["val_auc"],
            "val_acc": x["val_acc"],
            "weight_norm": x["weight_norm"],
            "val_activation_mean": x["val_activation_mean"],
            "val_activation_std": x["val_activation_std"]
        })

    with open(output_dir / "layerwise_probe_stats.json", "w", encoding="utf-8") as f:
        json.dump(layerwise_stats, f, indent=2, ensure_ascii=False)

    print(f"[Saved] {npz_path}")
    print(f"[Saved] {output_dir / 'probe_metadata.json'}")
    print(f"[Saved] {output_dir / 'layerwise_probe_stats.json'}")


# =========================
# 参数与主程序
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="llama3-8b")
    parser.add_argument("--unsafe_path", type=str,
                        default=str(PROJECT_ROOT / "data" / "02_unsafe.json"))
    parser.add_argument("--safe_path", type=str,
                        default=str(PROJECT_ROOT / "data" / "01_safe.json"))
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--c_value", type=float, default=1.0)
    parser.add_argument("--acc_threshold", type=float, default=0.95)
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "embedding-level" / "outputs" / "stage1"))
    parser.add_argument("--save_html", action="store_true", default=True)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print("=" * 80)
    print("Stage 1: Competition Direction Extraction + HTML Visualization")
    print("=" * 80)

    unsafe_samples, safe_samples = load_binary_samples(args.unsafe_path, args.safe_path)
    print(f"Loaded unsafe samples: {len(unsafe_samples)}")
    print(f"Loaded safe samples  : {len(safe_samples)}")

    split_data = split_samples(
        unsafe_samples=unsafe_samples,
        safe_samples=safe_samples,
        test_size=args.val_ratio,
        seed=args.seed
    )

    print("\nSplit summary:")
    for k, v in split_data.items():
        print(f"  {k:12s}: {len(v)}")

    print("\nInitializing model...")
    llm = ModelExtraction(args.model_name)
    n_layers = llm.llm_cfg.n_layer
    print(f"Model initialized. n_layers={n_layers}")

    emb_split = extract_embeddings_by_split(llm, split_data)

    layer_results = train_probe_per_layer(
        emb_split=emb_split,
        n_layers=n_layers,
        max_iter=args.max_iter,
        c_value=args.c_value
    )

    summary_info = summarize_results(layer_results, acc_threshold=args.acc_threshold)

    plot_probe_metrics(
        layer_results=layer_results,
        high_acc_layers=summary_info["high_acc_layers"],
        acc_threshold=args.acc_threshold,
        output_dir=output_dir
    )

    save_probe_artifacts(
        layer_results=layer_results,
        split_data=split_data,
        summary_info=summary_info,
        args=args,
        output_dir=output_dir
    )

    if args.save_html:
        save_multilayer_html_visualization(
            emb_split=emb_split,
            split_data=split_data,
            n_layers=n_layers,
            output_path=output_dir / "binary_samples_visualization.html"
        )

        # 输出几个代表层
        rep_layers = []
        if summary_info["best_continuous_range"] is not None:
            s, e = summary_info["best_continuous_range"]
            rep_layers.extend([s, (s + e) // 2, e])

        rep_layers.append(summary_info["best_acc_layer"])
        rep_layers.append(summary_info["recommended_stage2_layer"])
        rep_layers = sorted(set([x for x in rep_layers if 0 <= x < n_layers]))

        for layer_id in rep_layers:
            save_single_layer_html_visualization(
                emb_split=emb_split,
                split_data=split_data,
                layer_id=layer_id,
                output_path=output_dir / f"binary_layer_{layer_id}_single.html"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()