import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =========================
# Config
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
csv_path = str(PROJECT_ROOT / "results/pre_experiment/k5/summary.csv")
out_dir = "figures"
os.makedirs(out_dir, exist_ok=True)

ks = [1, 4, 8, 12, 16]
rec_k = 12
rec_g = 8

# =========================
# Paper-style settings (two-column friendly)
# =========================
sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 6,
    "legend.title_fontsize": 6,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

palette = {
    1: "#4C72B0",
    4: "#55A868",
    8: "#C44E52",
    12: "#8172B2",
    16: "#CCB974",
}

# =========================
# Load CSV
# =========================
df = pd.read_csv(csv_path)
df = df.sort_values("Iteration").reset_index(drop=True)

required_cols = ["Iteration", "AvgIterTime(s)"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

for k in ks:
    needed = [
        f"K={k}_ASR(%)",
        f"K={k}_AvgHS",
        f"K={k}_AvgQ",
        f"K={k}_AvgEvalTime(s)",
        f"K={k}_AvgTotalTime(s)",
    ]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

# =========================
# Convert to long format
# =========================
records = []
for _, row in df.iterrows():
    for k in ks:
        records.append({
            "Iteration": int(row["Iteration"]),
            "K": k,
            "AvgIterTime": float(row["AvgIterTime(s)"]),
            "ASR": float(row[f"K={k}_ASR(%)"]),
            "AvgHS": float(row[f"K={k}_AvgHS"]),
            "AvgQ": float(row[f"K={k}_AvgQ"]),
            "AvgEvalTime": float(row[f"K={k}_AvgEvalTime(s)"]),
            "AvgTotalTime": float(row[f"K={k}_AvgTotalTime(s)"]),
        })

long_df = pd.DataFrame(records)

rec_df = long_df[(long_df["K"] == rec_k) & (long_df["Iteration"] == rec_g)]
if len(rec_df) == 0:
    raise ValueError(f"Recommended point K={rec_k}, G={rec_g} not found.")
best_point = rec_df.iloc[0]

xvals = sorted(long_df["Iteration"].unique())
xticks = xvals[::2] if len(xvals) > 10 else xvals

# =========================
# Figure 1: ASR vs Iteration
# =========================
fig, ax = plt.subplots(figsize=(3.4, 2.5))  # two-column friendly single-panel size

for k in ks:
    sub = long_df[long_df["K"] == k]
    ax.plot(
        sub["Iteration"],
        sub["ASR"],
        marker="o",
        label=f"$K={k}$",
        color=palette[k],
    )

ax.scatter(
    best_point["Iteration"],
    best_point["ASR"],
    s=25,
    color=palette[rec_k],
    edgecolor="black",
    linewidth=0.6,
    zorder=5,
)

ax.annotate(
    f"($K={rec_k}, G={rec_g}$)",
    xy=(best_point["Iteration"], best_point["ASR"]),
    xytext=(9.2, 84.5),
    textcoords="data",
    arrowprops=dict(arrowstyle="->", lw=0.8, color="black"),
    fontsize=7.5,
    ha="left",
    va="bottom",
)

ax.set_xlabel("Iteration budget $G$")
ax.set_ylabel("ASR (%)")
ax.set_xlim(min(xvals), max(xvals))
ax.set_ylim(20, 102)
ax.set_xticks(xticks)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

leg = ax.legend(
    title="$K$",
    loc="lower right",
    frameon=True,
    ncol=1,
    borderpad=0.3,
    handlelength=1.6,
)
leg.get_frame().set_alpha(0.9)

plt.tight_layout(pad=0.3)
plt.savefig(os.path.join(out_dir, "rq2_asr_vs_iteration_by_k.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(out_dir, "rq2_asr_vs_iteration_by_k.png"), dpi=400, bbox_inches="tight")
plt.close()

# =========================
# Figure 2: Total Time vs Iteration
# =========================
fig, ax = plt.subplots(figsize=(3.4, 2.5))

for k in ks:
    sub = long_df[long_df["K"] == k]
    ax.plot(
        sub["Iteration"],
        sub["AvgTotalTime"],
        marker="s",
        label=f"$K={k}$",
        color=palette[k],
    )

# Removed AvgIterTime line as requested

ax.scatter(
    best_point["Iteration"],
    best_point["AvgTotalTime"],
    s=25,
    color=palette[rec_k],
    edgecolor="black",
    linewidth=0.6,
    zorder=5,
)

ax.annotate(
    f"($K={rec_k}, G={rec_g}$)",
    xy=(best_point["Iteration"], best_point["AvgTotalTime"]),
    xytext=(9.5, best_point["AvgTotalTime"] + 4.5),
    textcoords="data",
    arrowprops=dict(arrowstyle="->", lw=0.8, color="black"),
    fontsize=7.5,
    ha="left",
    va="bottom",
)

ax.set_xlabel("Iteration budget $G$")
ax.set_ylabel("Time/sample (s)")
ax.set_xlim(min(xvals), max(xvals))
ax.set_xticks(xticks)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

leg = ax.legend(
    title="$K$",
    loc="lower right",
    frameon=True,
    ncol=1,
    borderpad=0.3,
    handlelength=1.6,
)
leg.get_frame().set_alpha(0.9)

plt.tight_layout(pad=0.3)
plt.savefig(os.path.join(out_dir, "rq2_time_vs_iteration_by_k.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(out_dir, "rq2_time_vs_iteration_by_k.png"), dpi=400, bbox_inches="tight")
plt.close()

# =========================
# Figure 3: ASR Heatmap (appendix optional)
# =========================
heatmap_df = long_df.pivot(index="K", columns="Iteration", values="ASR")

fig, ax = plt.subplots(figsize=(6.8, 1.9))  # wide and compact for two-column appendix
sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".0f",
    cmap="YlOrRd",
    cbar_kws={"label": "ASR (%)"},
    linewidths=0.35,
    linecolor="white",
    annot_kws={"size": 7},
    ax=ax
)

ax.set_xlabel("Iteration budget $G$")
ax.set_ylabel("$K$")
ax.tick_params(axis="x", labelrotation=0)
ax.tick_params(axis="y", rotation=0)

plt.tight_layout(pad=0.3)
plt.savefig(os.path.join(out_dir, "rq2_asr_heatmap.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(out_dir, "rq2_asr_heatmap.png"), dpi=400, bbox_inches="tight")
plt.close()

print(f"Saved figures to: {out_dir}")
print("Generated:")
print(" - rq2_asr_vs_iteration_by_k.(pdf/png)")
print(" - rq2_time_vs_iteration_by_k.(pdf/png)")
print(" - rq2_asr_heatmap.(pdf/png)")