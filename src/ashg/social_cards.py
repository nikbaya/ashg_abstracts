"""Social preview cards (1200x630 PNGs) for the landing, Find talks and Compare pages.

Illustrative art only, with no real titles, names or numbers: the network is random.
Usage: python src/ashg/social_cards.py [docs/img]
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

BG, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
TALK, POSTER, RULE = "#2a78d6", "#e34948", "#e4e3df"
CARDS = {  # file -> (kicker, title, subtitle)
    "social_index.png": ("ASHG abstracts", "What ASHG\nabstracts say,\nwho gets talks",
                         "Topics, wording and presenters over time,\nand what leans toward talks over posters"),
    "social_similar.png": ("ASHG abstracts · Find talks", "Find the\nabstracts closest\nto yours",
                           "Pick an abstract and see its closest\nmatches at ASHG 2026, by text or authors"),
    "social_compare.png": ("ASHG abstracts · Compare", "How alike are\ntwo abstracts?",
                           "Text match, rank among each other's closest\nabstracts, and the words they share"),
}


def canvas():
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200); ax.set_ylim(0, 630); ax.axis("off")
    ax.set_aspect("equal")
    return fig, ax


def text(ax, kicker, title, sub):
    ax.add_patch(plt.Rectangle((0, 0), 1200, 10, color=TALK, lw=0))
    ax.add_patch(plt.Rectangle((600, 0), 600, 10, color=POSTER, lw=0))
    ax.text(70, 540, kicker, fontsize=21, color=MUTED, va="center")
    ax.text(70, 375, title, fontsize=42, fontweight="bold", color=INK, va="center", linespacing=1.2)
    ax.text(70, 190, sub, fontsize=21, color=MUTED, va="top", linespacing=1.5)


def art_index(ax):
    for (x, c) in ((880, TALK), (1020, POSTER)):
        ax.add_patch(Circle((x, 330), 125, color=c, alpha=0.82, lw=0))
    ax.add_patch(Circle((880, 330), 125, fill=False, ec=BG, lw=3))
    ax.text(825, 330, "talks", fontsize=20, color="white", ha="center", va="center", fontweight="bold")
    ax.text(1080, 330, "posters", fontsize=20, color="white", ha="center", va="center", fontweight="bold")


def art_similar(ax):
    rng = np.random.default_rng(7)
    cx, cy = 910, 330
    pts = []
    for r, n in ((85, 5), (150, 9), (215, 13)):
        for a in np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, 1):
            pts.append((cx + r * np.cos(a) * 1.0 + rng.normal(0, 9), cy + r * np.sin(a) * 0.95 + rng.normal(0, 9), r))
    for x, y, r in pts:
        if r == 85:
            ax.plot([cx, x], [cy, y], color=RULE, lw=2.5, zorder=1)
    for x, y, r in pts:
        ax.add_patch(Circle((x, y), {85: 20, 150: 15, 215: 10}[r], color=POSTER if rng.random() < 0.55 else TALK,
                            alpha={85: 0.9, 150: 0.65, 215: 0.4}[r], lw=0, zorder=2))
    ax.add_patch(Circle((cx, cy), 40, color=TALK, ec=BG, lw=4, zorder=3))
    ax.add_patch(Circle((cx, cy), 14, color="white", lw=0, zorder=4))


def art_compare(ax):
    ax.add_patch(Circle((870, 330), 125, color=TALK, alpha=0.8, lw=0))
    ax.add_patch(Circle((1010, 330), 125, color=POSTER, alpha=0.8, lw=0))
    ax.add_patch(Circle((870, 330), 125, fill=False, ec=BG, lw=3))
    # lens where the circles overlap
    t = np.linspace(-1, 1, 200)
    h = np.sqrt(125 ** 2 - 70 ** 2)
    ys = 330 + h * t
    xl = 1010 - np.sqrt(125 ** 2 - (ys - 330) ** 2)
    xr = 870 + np.sqrt(125 ** 2 - (ys - 330) ** 2)
    ax.fill_betweenx(ys, xl, xr, color="#7a4aa8", alpha=0.95, lw=0)
    for i, w in enumerate((60, 44, 54)):
        ax.plot([940 - w / 2, 940 + w / 2], [375 - i * 45, 375 - i * 45], color="white", lw=5, solid_capstyle="round")


ART = {"social_index.png": art_index, "social_similar.png": art_similar, "social_compare.png": art_compare}


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "docs/img")
    out.mkdir(parents=True, exist_ok=True)
    for name, (kicker, title, sub) in CARDS.items():
        fig, ax = canvas()
        text(ax, kicker, title, sub)
        ART[name](ax)
        fig.savefig(out / name, dpi=100, facecolor=BG)
        plt.close(fig)
        print("wrote", out / name)


if __name__ == "__main__":
    main()
