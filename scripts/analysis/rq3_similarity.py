import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Config
# =========================
out_dir = "figures"
os.makedirs(out_dir, exist_ok=True)

# If you want, you can also load this from a json file.
data = {
    "plain_pair": {
        "normalized": [
            1.0000005960464478,
            0.9788009524345398,
            0.9458519816398621,
            0.9589555859565735,
            0.933606743812561,
            0.9029810428619385,
            0.9008454084396362,
            0.864829421043396,
            0.8282117247581482,
            0.8039238452911377,
            0.7579572796821594,
            0.7262853980064392,
            0.6834648251533508,
            0.7015109062194824,
            0.6211203932762146,
            0.6127352118492126,
            0.5369850397109985,
            0.49044081568717957,
            0.4720268249511719,
            0.43841320276260376,
            0.4230363070964813,
            0.3926292359828949,
            0.3756392300128937,
            0.3384867012500763,
            0.32846546173095703,
            0.3510545790195465,
            0.3751111328601837,
            0.37367984652519226,
            0.38785865902900696,
            0.3689488172531128,
            0.3799780309200287,
            0.3555046021938324
        ]
    },
    "failed_template_pair": {
        "normalized": [
            1.0000005960464478,
            0.9385520815849304,
            0.9089011549949646,
            0.9310250878334045,
            0.9340057969093323,
            0.9418591856956482,
            0.9465833902359009,
            0.9352962374687195,
            0.9330047369003296,
            0.9082276821136475,
            0.899005115032196,
            0.8947718739509583,
            0.8825506567955017,
            0.874651312828064,
            0.8188275694847107,
            0.787544846534729,
            0.6985389590263367,
            0.613257884979248,
            0.5623809695243835,
            0.5444391369819641,
            0.4996531009674072,
            0.46341952681541443,
            0.4393291175365448,
            0.41653963923454285,
            0.40903162956237793,
            0.42467746138572693,
            0.43702802062034607,
            0.4281383454799652,
            0.44132256507873535,
            0.44806811213493347,
            0.4616430699825287,
            0.4639812707901001
        ]
    },
    "successful_template_pair": {
        "normalized": [
            1.0000005960464478,
            0.9992133975028992,
            0.9990566968917847,
            0.9987977743148804,
            0.9983152151107788,
            0.9979783892631531,
            0.9976423382759094,
            0.9969147443771362,
            0.9969327449798584,
            0.9961602687835693,
            0.9965190291404724,
            0.995762050151825,
            0.9956068992614746,
            0.995134711265564,
            0.992536187171936,
            0.9911167621612549,
            0.9892767071723938,
            0.9857261776924133,
            0.9832504391670227,
            0.9833012223243713,
            0.9822894334793091,
            0.9800539612770081,
            0.980021595954895,
            0.9751913547515869,
            0.9743853807449341,
            0.9733611941337585,
            0.9687846302986145,
            0.9692329168319702,
            0.9660469889640808,
            0.9619615077972412,
            0.9596085548400879,
            0.9536244869232178
        ]
    }
}

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
    "plain_pair": "#4C72B0",
    "failed_template_pair": "#55A868",
    "successful_template_pair": "#C44E52",
}

markers = {
    "plain_pair": "o",
    "failed_template_pair": "s",
    "successful_template_pair": "^",
}

labels = {
    "plain_pair": "Plain pair",
    "failed_template_pair": "Failed-template pair",
    "successful_template_pair": "Successful-template pair",
}

# =========================
# Prepare data
# =========================
plain = data["plain_pair"]["normalized"]
failed = data["failed_template_pair"]["normalized"]
success = data["successful_template_pair"]["normalized"]

num_layers = len(plain)
assert len(failed) == num_layers and len(success) == num_layers, "All curves must have the same length."

layers = list(range(1, num_layers + 1))
xticks = layers[::4]
if xticks[-1] != layers[-1]:
    xticks.append(layers[-1])

# =========================
# Plot main figure
# =========================
fig, ax = plt.subplots(figsize=(3.4, 2.5))

for key in ["plain_pair", "failed_template_pair", "successful_template_pair"]:
    ax.plot(
        layers,
        data[key]["normalized"],
        marker=markers[key],
        label=labels[key],
        color=palette[key],
    )

ax.set_xlabel("Layer")
ax.set_ylabel("Average cosine similarity")
ax.set_xlim(1, num_layers)
ax.set_ylim(0.30, 1.02)
ax.set_xticks(xticks)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

leg = ax.legend(
    loc="lower left",
    frameon=True,
    ncol=1,
    borderpad=0.3,
    handlelength=1.6,
)
leg.get_frame().set_alpha(0.9)

plt.tight_layout(pad=0.3)
plt.savefig(os.path.join(out_dir, "alignment_results.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(out_dir, "alignment_results.png"), dpi=400, bbox_inches="tight")
plt.close()

print(f"Saved figures to: {out_dir}")
print("Generated:")
print(" - alignment_results.pdf")
print(" - alignment_results.png")