#!/usr/bin/env python3
"""Central-finding figure: the command interface exposes little state-dependent
efficiency headroom. (a) the per-episode optimal command scale is nearly constant;
(b) an omniscient per-state oracle barely beats the best single fixed scale.
Data computed on paired-successful episodes (identical initial states)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = json.load(open("/tmp/claude-1000/-home-thing1-skill-discover/ce49fb81-69ac-4332-8458-218dbe336606/scratchpad/finding.json"))
go2, h1 = D["go2"], D["h1"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.8))

# ---- Panel (a): winning-scale distribution ----
scales = ["0.75", "0.90", "1.0"]
gw = [go2["win"]["0.75"], go2["win"]["0.9"], go2["win"]["1.0"]]
hw = [h1["win"]["0.75"], h1["win"]["0.9"], h1["win"]["1.0"]]
x = np.arange(3); w = 0.36
axA.bar(x - w/2, [100*v for v in gw], w, label=f"Go2 ($n{{=}}{go2['n']}$)", color="#d95f02")
axA.bar(x + w/2, [100*v for v in hw], w, label=f"H1 ($n{{=}}{h1['n']}$)", color="#7570b3")
axA.set_xticks(x); axA.set_xticklabels([f"scale {s}" for s in scales])
axA.set_ylabel("episodes where scale is COT-optimal (%)", fontsize=9)
axA.set_title("(a) The optimal command scale is nearly state-independent", fontsize=9.5)
axA.legend(fontsize=8, loc="upper right"); axA.grid(axis="y", alpha=0.25)
axA.text(0.02, 0.90, "one scale wins\n74–78% of states", transform=axA.transAxes,
         fontsize=8.5, style="italic", va="top")

# ---- Panel (b): COT relative to best single fixed scale ----
def rel(r):
    b = r["scaleCOT"]["scaled_target_command_75"]
    return {
        "per-state\noracle": r["oracle"]/b,
        "best fixed\nscale (0.75)": 1.0,
        "fixed 0.90": r["scaleCOT"]["scaled_target_command_90"]/b,
        "VGCC": r["vgcc"]/b,
        "direct": r["scaleCOT"]["direct_target_command"]/b,
    }
g, h = rel(go2), rel(h1)
labels = list(g.keys())
xi = np.arange(len(labels))
axB.axhline(1.0, color="#1b9e77", lw=1.2, ls="--", alpha=0.8)
# shade the achievable headroom band (oracle..best fixed)
axB.axhspan(min(g["per-state\noracle"], h["per-state\noracle"]), 1.0,
            color="#1b9e77", alpha=0.12)
axB.scatter(xi, [g[l] for l in labels], s=90, marker="D", color="#d95f02",
            edgecolors="black", linewidths=0.7, zorder=3, label=f"Go2")
axB.scatter(xi, [h[l] for l in labels], s=90, marker="o", color="#7570b3",
            edgecolors="black", linewidths=0.7, zorder=3, label=f"H1")
axB.set_xticks(xi); axB.set_xticklabels(labels, fontsize=8)
axB.set_ylabel("cost of transport\n(relative to best fixed scale)", fontsize=9)
axB.set_title("(b) A per-state oracle barely beats one fixed scale", fontsize=9.5)
axB.legend(fontsize=8, loc="upper left")
axB.grid(axis="y", alpha=0.25)
axB.annotate(f"total adaptivity headroom:\nGo2 {100*(1-go2['oracle']/go2['scaleCOT']['scaled_target_command_75']):.1f}%,  "
             f"H1 {100*(1-h1['oracle']/h1['scaleCOT']['scaled_target_command_75']):.1f}%",
             xy=(0, 0.99), xytext=(0.5, 0.055), textcoords="axes fraction",
             fontsize=8.2, style="italic",
             arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

fig.tight_layout()
out = "paper/figures/finding.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
