"""Generate topology and interference diagrams for the ChannelHub manual.

Output: 4 PNG files in D:/MSG平台_cc/docs/figures/
  fig_hex_topology.png   - 7-cell hexagonal topology
  fig_linear_topology.png - HSR linear topology
  fig_interference.png    - Neighbor-cell interference model
  fig_workflow.png        - Platform end-to-end workflow
"""

import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, RegularPolygon
import numpy as np

OUT = Path("D:/MSG平台_cc/docs/figures")
OUT.mkdir(exist_ok=True)

# -- Consistent style --
BLUE = "#1677ff"
GREEN = "#52c41a"
RED = "#fa541c"
PURPLE = "#722ed1"
CYAN = "#13c2c2"
PINK = "#eb2f96"
ORANGE = "#faad14"
DARK_BLUE = "#2f54eb"
GREY = "#595959"
LIGHT_BLUE_FILL = "rgba(22,119,255,0.06)"

SITE_COLORS = [BLUE, GREEN, RED, PURPLE, CYAN, PINK, ORANGE, DARK_BLUE,
               "#a0d911", "#f5222d", "#1890ff", "#597ef7"]

plt.rcParams.update({
    "font.family": ["SimHei", "Microsoft YaHei", "sans-serif"],
    "axes.unicode_minus": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


# ============================================================
# Fig 1: 7-cell hexagonal topology
# ============================================================
def hex_corners(cx, cy, r):
    angles = [math.pi / 3 * i - math.pi / 6 for i in range(6)]
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]


def hex_ring_positions(isd):
    r = isd / math.sqrt(3)
    positions = [(0, 0)]  # center
    dx = isd * math.sqrt(3) / 2
    dy = isd * 1.5 / math.sqrt(3)
    ring1 = [
        (isd, 0), (isd / 2, isd * math.sqrt(3) / 2),
        (-isd / 2, isd * math.sqrt(3) / 2), (-isd, 0),
        (-isd / 2, -isd * math.sqrt(3) / 2), (isd / 2, -isd * math.sqrt(3) / 2),
    ]
    positions.extend(ring1)
    return positions, r


def draw_hex_topology():
    fig, ax = plt.subplots(1, 1, figsize=(7, 6.5))
    isd = 500
    positions, cell_r = hex_ring_positions(isd)

    # Draw hexagons
    for i, (cx, cy) in enumerate(positions):
        hex_patch = RegularPolygon(
            (cx, cy), numVertices=6, radius=cell_r,
            orientation=0,
            facecolor="#e8f4ff" if i == 0 else "#f0f7ff",
            edgecolor="#91caff" if i == 0 else "#bcdcff",
            linewidth=1.5 if i == 0 else 1,
            zorder=1)
        ax.add_patch(hex_patch)

    # Draw sector lines (3 sectors per site, azimuths 0/120/240)
    for i, (cx, cy) in enumerate(positions):
        for az in [0, 120, 240]:
            angle = math.radians(90 - az)
            length = cell_r * 0.75
            ex = cx + length * math.cos(angle)
            ey = cy + length * math.sin(angle)
            ax.plot([cx, ex], [cy, ey], color="#d0d0d0", linewidth=0.8,
                    linestyle="--", zorder=2)

    # Place UEs (random-ish inside cells)
    rng = np.random.default_rng(42)
    ue_xs, ue_ys = [], []
    for cx, cy in positions:
        n_ue = rng.integers(2, 5)
        for _ in range(n_ue):
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(0.15, 0.7) * cell_r
            ux = cx + dist * math.cos(angle)
            uy = cy + dist * math.sin(angle)
            ue_xs.append(ux)
            ue_ys.append(uy)

    ax.scatter(ue_xs, ue_ys, s=12, c=GREY, alpha=0.5, zorder=4, label="终端 (UE)")

    # Draw BS markers
    for i, (cx, cy) in enumerate(positions):
        color = SITE_COLORS[i % len(SITE_COLORS)]
        ax.plot(cx, cy, "o", markersize=12, color="white", markeredgecolor=color,
                markeredgewidth=2, zorder=5)
        ax.plot(cx, cy, "o", markersize=5, color=color, zorder=6)
        ax.text(cx, cy + cell_r * 0.18, f"BS{i}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=color, zorder=7)

    # Serving cell highlight
    ax.text(0, -cell_r * 0.25, "服务小区", ha="center", va="top",
            fontsize=9, color=BLUE, fontstyle="italic", zorder=7)

    # ISD annotation
    ax.annotate("", xy=(positions[1][0], positions[1][1]),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.2))
    mid_x = positions[1][0] / 2
    mid_y = positions[1][1] / 2
    ax.text(mid_x + 30, mid_y + 30, f"ISD = {isd} m",
            fontsize=9, color="#333", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#ccc", alpha=0.9))

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE,
                   markeredgecolor=BLUE, markersize=8, label="基站 (BS)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY,
                   markersize=5, alpha=0.5, label="终端 (UE)"),
        mpatches.Patch(facecolor="#e8f4ff", edgecolor="#91caff", label="服务小区"),
        mpatches.Patch(facecolor="#f0f7ff", edgecolor="#bcdcff", label="邻区"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.9, edgecolor="#ddd")

    ax.set_xlim(-isd * 1.5, isd * 1.5)
    ax.set_ylim(-isd * 1.3, isd * 1.3)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.grid(True, alpha=0.15, linestyle="-")
    ax.set_title("7 小区六边形蜂窝拓扑（3 扇区）", fontsize=13, fontweight="bold", pad=12)

    fig.savefig(OUT / "fig_hex_topology.png")
    plt.close(fig)
    print("  -> fig_hex_topology.png")


# ============================================================
# Fig 2: Linear HSR topology
# ============================================================
def draw_linear_topology():
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    isd = 1000
    n_sites = 6
    track_offset = 150

    # Draw track centerline
    total_len = (n_sites - 1) * isd
    ax.plot([-200, total_len + 200], [0, 0], color=RED, linewidth=2,
            linestyle="--", alpha=0.6, zorder=2)
    ax.text(total_len + 250, 0, "轨道中心线", fontsize=8, color=RED,
            va="center", ha="left")

    # Draw BS positions (alternating sides)
    for i in range(n_sites):
        x = i * isd
        y = track_offset if i % 2 == 0 else -track_offset
        color = SITE_COLORS[i % len(SITE_COLORS)]

        # Coverage circle
        circle = plt.Circle((x, y), isd * 0.45, facecolor="#f0f7ff",
                           edgecolor="#bcdcff", linewidth=1,
                           linestyle="--", alpha=0.5, zorder=1)
        ax.add_patch(circle)

        # BS marker
        ax.plot(x, y, "o", markersize=12, color="white", markeredgecolor=color,
                markeredgewidth=2, zorder=5)
        ax.plot(x, y, "o", markersize=5, color=color, zorder=6)
        label_y = y + (50 if y > 0 else -50)
        ax.text(x, label_y, f"BS{i}", ha="center", va="center",
                fontsize=8, fontweight="bold", color=color, zorder=7)

        # Line to track
        ax.plot([x, x], [y, 0], color="#ccc", linewidth=0.8, linestyle=":", zorder=2)

    # Draw train
    train_x_start = 1.3 * isd
    train_x_end = 2.7 * isd
    train_w = train_x_end - train_x_start
    train_h = 40
    train_rect = FancyBboxPatch(
        (train_x_start, -train_h / 2), train_w, train_h,
        boxstyle="round,pad=8",
        facecolor=(250/255, 84/255, 28/255, 0.08), edgecolor=RED,
        linewidth=1.5, linestyle="--", zorder=3)
    ax.add_patch(train_rect)

    # UEs inside train
    rng = np.random.default_rng(123)
    n_ue = 8
    for j in range(n_ue):
        ux = train_x_start + 30 + (train_w - 60) * j / (n_ue - 1)
        uy = rng.uniform(-12, 12)
        ax.plot(ux, uy, "o", markersize=3, color=GREY, alpha=0.5, zorder=4)

    ax.text((train_x_start + train_x_end) / 2, train_h / 2 + 18,
            "列车 (8 UE)", ha="center", fontsize=9, color=RED,
            fontweight="bold", zorder=7)

    # Speed arrow
    ax.annotate("", xy=(train_x_end + 60, 0),
                xytext=(train_x_end + 10, 0),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.text(train_x_end + 70, 8, "v = 350 km/h", fontsize=8, color=RED, va="bottom")

    # Track offset annotation
    ax.annotate("", xy=(500, track_offset), xytext=(500, 0),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1))
    ax.text(560, track_offset / 2, f"偏移 {track_offset} m", fontsize=8, color="#333", va="center")

    # ISD annotation
    ax.annotate("", xy=(isd, -track_offset - 60), xytext=(0, -track_offset - 60),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1))
    ax.text(isd / 2, -track_offset - 75, f"ISD = {isd} m", fontsize=8,
            color="#333", ha="center")

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE,
                   markeredgecolor=BLUE, markersize=8, label="基站 (BS)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY,
                   markersize=5, alpha=0.5, label="终端 (UE)"),
        plt.Line2D([0], [0], color=RED, linewidth=1.5, linestyle="--",
                   alpha=0.6, label="轨道中心线"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8,
              framealpha=0.9, edgecolor="#ddd")

    ax.set_xlim(-400, total_len + 600)
    ax.set_ylim(-track_offset - 180, track_offset + 180)
    ax.set_aspect("equal")
    ax.set_xlabel("沿轨道方向 (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.grid(True, alpha=0.15, linestyle="-")
    ax.set_title("高铁线性拓扑部署", fontsize=13, fontweight="bold", pad=12)

    fig.savefig(OUT / "fig_linear_topology.png")
    plt.close(fig)
    print("  -> fig_linear_topology.png")


# ============================================================
# Fig 3: Neighbor-cell interference model with precoding
# ============================================================
def draw_interference():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))

    # Cell positions (3-cell simplified)
    isd = 500
    cell_r = isd / math.sqrt(3)
    positions = [(0, 0), (isd, 0), (isd / 2, isd * math.sqrt(3) / 2)]
    labels = ["BS₀ (服务)", "BS₁ (邻区)", "BS₂ (邻区)"]
    colors = [BLUE, RED, PURPLE]

    # Draw hexagons
    for i, (cx, cy) in enumerate(positions):
        hex_patch = RegularPolygon(
            (cx, cy), numVertices=6, radius=cell_r,
            orientation=0,
            facecolor="#e8f4ff" if i == 0 else "#fff5f5" if i == 1 else "#f9f0ff",
            edgecolor=colors[i], linewidth=1.5, alpha=0.5, zorder=1)
        ax.add_patch(hex_patch)

    # Target UE Q position (in serving cell)
    q_x, q_y = 120, -60
    ax.plot(q_x, q_y, "s", markersize=10, color=BLUE, zorder=8)
    ax.text(q_x + 25, q_y - 20, "Q (目标UE)", fontsize=9, color=BLUE,
            fontweight="bold", zorder=9)

    # Scheduled UEs P_k in neighbor cells
    p1_x, p1_y = isd - 100, 80
    p2_x, p2_y = isd / 2 + 60, isd * math.sqrt(3) / 2 - 80

    ax.plot(p1_x, p1_y, "D", markersize=7, color=RED, zorder=8)
    ax.text(p1_x + 25, p1_y, "P₁ (邻区调度UE)", fontsize=8, color=RED, zorder=9)

    ax.plot(p2_x, p2_y, "D", markersize=7, color=PURPLE, zorder=8)
    ax.text(p2_x + 25, p2_y, "P₂ (邻区调度UE)", fontsize=8, color=PURPLE, zorder=9)

    # Draw BS markers
    for i, (cx, cy) in enumerate(positions):
        ax.plot(cx, cy, "^", markersize=16, color="white", markeredgecolor=colors[i],
                markeredgewidth=2.5, zorder=6)
        ax.plot(cx, cy, "^", markersize=9, color=colors[i], zorder=7)
        offset_y = cell_r * 0.15 + 20
        ax.text(cx, cy + offset_y, labels[i], ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=colors[i], zorder=9,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none"))

    # Serving link: BS0 -> Q (solid blue, thick)
    ax.annotate("", xy=(q_x - 8, q_y + 5), xytext=(15, -15),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.5,
                               connectionstyle="arc3,rad=0.05"))
    ax.text(60, -50, "H(BS₀→Q)", fontsize=8, color=BLUE,
            rotation=10, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none"))

    # Interference link: BS1 -> Q (dashed red)
    ax.annotate("", xy=(q_x + 8, q_y + 5), xytext=(isd - 15, -10),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8,
                               linestyle="--",
                               connectionstyle="arc3,rad=-0.08"))
    ax.text(isd / 2 + 40, -55, "H(BS₁→Q)·W₁W₁ᴴ", fontsize=8, color=RED,
            fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none"))

    # Interference link: BS2 -> Q (dashed purple)
    ax.annotate("", xy=(q_x + 3, q_y + 10), xytext=(isd / 2 - 10, isd * math.sqrt(3) / 2 - 15),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.8,
                               linestyle="--",
                               connectionstyle="arc3,rad=0.12"))
    ax.text(100, isd * math.sqrt(3) / 4 - 20, "H(BS₂→Q)·W₂W₂ᴴ", fontsize=8,
            color=PURPLE, fontstyle="italic", rotation=-50,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none"))

    # Precoding link: BS1 -> P1 (dotted, thin)
    ax.annotate("", xy=(p1_x - 5, p1_y - 5), xytext=(isd - 10, 10),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2,
                               linestyle=":", alpha=0.7,
                               connectionstyle="arc3,rad=0.1"))
    ax.text(isd - 60, 55, "H(BS₁→P₁)\n→SVD→W₁", fontsize=7, color=RED,
            ha="center", alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#fff5f5", alpha=0.8, edgecolor="#ffccc7"))

    # Precoding link: BS2 -> P2
    ax.annotate("", xy=(p2_x - 5, p2_y + 3), xytext=(isd / 2 + 5, isd * math.sqrt(3) / 2 - 10),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.2,
                               linestyle=":", alpha=0.7,
                               connectionstyle="arc3,rad=-0.1"))
    ax.text(isd / 2 + 70, isd * math.sqrt(3) / 2 - 45, "H(BS₂→P₂)\n→SVD→W₂",
            fontsize=7, color=PURPLE, ha="center", alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#f9f0ff", alpha=0.8, edgecolor="#d3adf7"))

    # Formula box
    formula_text = (
        "邻区预编码投影:\n"
        "H_proj(BS_k->Q) = W_k * W_k^H * H(BS_k->Q)\n"
        "W_k = SVD(H(BS_k->P_k))[:, :rank]"
    )
    ax.text(isd * 0.5, -cell_r * 0.8, formula_text,
            fontsize=8.5, color="#333", ha="center", va="top",
            family="SimHei",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbe6",
                     edgecolor="#ffe58f", alpha=0.95),
            zorder=10)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=BLUE, linewidth=2.5, label="服务链路 H(BS₀→Q)"),
        plt.Line2D([0], [0], color=RED, linewidth=1.8, linestyle="--",
                   label="投影后干扰 H·WW^H"),
        plt.Line2D([0], [0], color="#666", linewidth=1.2, linestyle=":",
                   label="邻区DL信道 H(BSₖ→Pₖ)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=BLUE,
                   markersize=7, label="目标终端 Q"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=RED,
                   markersize=6, label="邻区调度终端 Pₖ"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7.5,
              framealpha=0.95, edgecolor="#ddd")

    ax.set_xlim(-cell_r - 50, isd + cell_r + 50)
    ax.set_ylim(-cell_r - 30, isd * math.sqrt(3) / 2 + cell_r * 0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.grid(True, alpha=0.12, linestyle="-")
    ax.set_title("邻区 DL 预编码投影干扰建模", fontsize=13, fontweight="bold", pad=12)

    fig.savefig(OUT / "fig_interference.png")
    plt.close(fig)
    print("  -> fig_interference.png")


# ============================================================
# Fig 4: Platform workflow
# ============================================================
def draw_workflow():
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 3)
    ax.axis("off")

    boxes = [
        (0.5, 1, "数据采集\n(Simulate)", BLUE),
        (2.5, 1, "Bridge\n特征提取", GREEN),
        (4.5, 1, "数据集\n划分/锁定", CYAN),
        (6.5, 1, "数据导出\n(HDF5/WDS)", ORANGE),
        (8.5, 1, "外部训练\n(独立平台)", "#999"),
    ]
    boxes2 = [
        (8.5, -0.3, "模型上传\n(.pt/.ckpt)", PURPLE),
        (6.5, -0.3, "模型评估\n(锁定测试集)", RED),
        (4.5, -0.3, "排行榜\n指标对比", DARK_BLUE),
    ]

    bw, bh = 1.5, 0.8

    def draw_box(x, y, text, color, is_external=False):
        style = "round,pad=0.1"
        rect = FancyBboxPatch(
            (x - bw / 2, y - bh / 2), bw, bh,
            boxstyle=style, facecolor="white",
            edgecolor=color, linewidth=2,
            linestyle="--" if is_external else "-",
            zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=9, fontweight="bold", color=color, zorder=4)

    # Top row (data pipeline)
    for i, (x, y, text, color) in enumerate(boxes):
        draw_box(x, y, text, color, is_external=(i == 4))

    # Arrows top row
    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i + 1][0] - bw / 2 - 0.05, boxes[i][1]),
                    xytext=(boxes[i][0] + bw / 2 + 0.05, boxes[i][1]),
                    arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.5))

    # Bottom row (model pipeline)
    for x, y, text, color in boxes2:
        draw_box(x, y, text, color)

    # Arrow: external -> upload (down)
    ax.annotate("", xy=(8.5, -0.3 + bh / 2 + 0.05),
                xytext=(8.5, 1 - bh / 2 - 0.05),
                arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.5))

    # Arrows bottom row (right to left)
    for i in range(len(boxes2) - 1):
        ax.annotate("", xy=(boxes2[i + 1][0] + bw / 2 + 0.05, boxes2[i][1]),
                    xytext=(boxes2[i][0] - bw / 2 - 0.05, boxes2[i][1]),
                    arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.5))

    # Row labels
    ax.text(-0.3, 1, "数据\n流水线", ha="center", va="center",
            fontsize=8, color="#999", fontstyle="italic")
    ax.text(-0.3, -0.3, "评估\n流水线", ha="center", va="center",
            fontsize=8, color="#999", fontstyle="italic")

    # Platform boundary
    rect_plat = FancyBboxPatch(
        (-0.6, -0.9), 8.2, 2.8,
        boxstyle="round,pad=0.15", facecolor="none",
        edgecolor="#1677ff", linewidth=1.5, linestyle=":", alpha=0.5, zorder=0)
    ax.add_patch(rect_plat)
    ax.text(3.5, 2.1, "ChannelHub 平台", fontsize=10, color=BLUE,
            fontweight="bold", ha="center", alpha=0.7)

    # External boundary
    rect_ext = FancyBboxPatch(
        (7.5, 0.4), 2, 1.4,
        boxstyle="round,pad=0.15", facecolor="none",
        edgecolor="#999", linewidth=1.5, linestyle=":", alpha=0.5, zorder=0)
    ax.add_patch(rect_ext)
    ax.text(8.5, 2.0, "外部平台", fontsize=9, color="#999",
            fontweight="bold", ha="center", alpha=0.7)

    ax.set_title("平台端到端工作流", fontsize=13, fontweight="bold", pad=12)

    fig.savefig(OUT / "fig_workflow.png")
    plt.close(fig)
    print("  -> fig_workflow.png")


if __name__ == "__main__":
    print("Generating figures...")
    draw_hex_topology()
    draw_linear_topology()
    draw_interference()
    draw_workflow()
    print("All figures generated.")
