"""
Generates the publication-grade Architectural Framework diagram for Adaptive Learning Gate (ALG).
Ensures zero text overlap, clean bounding boxes, distinct color coding, and non-intersecting flow arrows.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_architecture_diagram(output_paths: list[Path]):
    fig, ax = plt.subplots(figsize=(15.5, 9.8), dpi=300)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 9.8)
    ax.axis('off')

    # Color Palette
    c_bg = "#FFFFFF"
    c_title = "#1A252F"
    c_sub = "#566573"

    fig.patch.set_facecolor(c_bg)
    ax.set_facecolor(c_bg)

    # ------------------ TITLE & SUBTITLE ------------------
    ax.text(7.75, 9.35, "Adaptive Learning Gate (ALG) — Architectural Framework", 
            ha='center', va='center', fontsize=18, fontweight='bold', color=c_title, family='sans-serif')
    ax.text(7.75, 8.95, "Selective Continual Training via Tri-Signal Utility Optimization & Bounded Deferral Quarantine", 
            ha='center', va='center', fontsize=11, fontstyle='italic', color=c_sub, family='sans-serif')

    def draw_box(x, y, w, h, bg_color, border_color, border_width=2.0, radius=0.25):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=border_width,
            zorder=3
        )
        ax.add_patch(box)
        return box

    # ------------------ TOP ROW PIPELINE (Y: 4.9 to 8.5) ------------------
    
    # 1. STREAM INTAKE BOX
    draw_box(0.5, 5.0, 2.7, 3.5, "#F0F4F8", "#2B4C7E", 2.2)
    ax.text(1.85, 8.15, "1. Stream Intake", ha='center', va='center', fontsize=12, fontweight='bold', color="#1B365D")
    ax.text(1.85, 7.80, "(x, y_obs)", ha='center', va='center', fontsize=10.5, fontstyle='italic', color="#334E68")
    
    stream_desc = (
        "• Continuous Sample Stream\n"
        "• Uncurated Non-Stationary\n"
        "• Sensor & Blur Noise (15%)\n"
        "• Symmetric Label Flips (20-40%)\n"
        "• Input: 3×32×32 Tensor"
    )
    ax.text(0.7, 6.40, stream_desc, ha='left', va='center', fontsize=9.5, color="#243B53", linespacing=1.6)

    # Arrow 1 -> 2
    ax.annotate("", xy=(3.8, 6.75), xytext=(3.2, 6.75),
                arrowprops=dict(arrowstyle="-|>", color="#2B4C7E", lw=2.5, mutation_scale=18), zorder=4)

    # 2. NEURAL BACKBONE BOX
    draw_box(3.8, 5.0, 2.7, 3.5, "#F8F9FA", "#2C3E50", 2.2)
    ax.text(5.15, 8.15, "2. Neural Backbone", ha='center', va='center', fontsize=12, fontweight='bold', color="#1A252F")
    ax.text(5.15, 7.80, "f_θ(x) [ConvNet Backbone]", ha='center', va='center', fontsize=10.5, fontstyle='italic', color="#4A5568")

    backbone_desc = (
        "• Forward Pass (No Grad)\n"
        "• 3-Stage ConvNet Backbone\n"
        "• Latent Rep: h = φ_θ(x) ∈ ℝ⁵¹²\n"
        "• Class Logits: z ∈ ℝ¹⁰\n"
        "• Softmax Probabilities p(c|x)"
    )
    ax.text(4.0, 6.40, backbone_desc, ha='left', va='center', fontsize=9.5, color="#2D3748", linespacing=1.6)

    # Arrow 2 -> 3
    ax.annotate("", xy=(7.1, 6.75), xytext=(6.5, 6.75),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=2.5, mutation_scale=18), zorder=4)

    # 3. TRI-SIGNAL EXTRACTOR CONTAINER
    draw_box(7.1, 4.8, 3.8, 3.7, "#FDF7F2", "#D35400", 2.2)
    ax.text(9.00, 8.22, "3. Tri-Signal Extractor", ha='center', va='center', fontsize=12.5, fontweight='bold', color="#A04000")

    # Sub-box 3A: Uncertainty
    draw_box(7.35, 7.10, 3.3, 0.85, "#FBEEE6", "#E59866", 1.4, radius=0.15)
    ax.text(9.00, 7.65, "Predictive Uncertainty U(x)", ha='center', va='center', fontsize=10, fontweight='bold', color="#935116")
    ax.text(9.00, 7.32, "Normalized Shannon Entropy ∈ [0, 1]", ha='center', va='center', fontsize=8.5, color="#7E5109")

    # Sub-box 3B: Novelty
    draw_box(7.35, 6.05, 3.3, 0.85, "#FBEEE6", "#E59866", 1.4, radius=0.15)
    ax.text(9.00, 6.60, "Latent Prototype Novelty N(x)", ha='center', va='center', fontsize=10, fontweight='bold', color="#935116")
    ax.text(9.00, 6.27, "Cosine Dist to Class Centroid c_k ∈ [0, 1]", ha='center', va='center', fontsize=8.5, color="#7E5109")

    # Sub-box 3C: Reliability
    draw_box(7.35, 5.00, 3.3, 0.85, "#FBEEE6", "#E59866", 1.4, radius=0.15)
    ax.text(9.00, 5.55, "Composite Reliability R(x)", ha='center', va='center', fontsize=10, fontweight='bold', color="#935116")
    ax.text(9.00, 5.22, "R_input(x) × R_pred(x) ∈ [0, 1]", ha='center', va='center', fontsize=8.5, color="#7E5109")

    # Arrow 3 -> 4
    ax.annotate("", xy=(11.5, 6.75), xytext=(10.9, 6.75),
                arrowprops=dict(arrowstyle="-|>", color="#D35400", lw=2.5, mutation_scale=18), zorder=4)

    # 4. GATING DECISION ENGINE BOX
    draw_box(11.5, 4.8, 3.5, 3.7, "#EAF7EF", "#1E8449", 2.2)
    ax.text(13.25, 8.22, "4. Gating Decision Engine", ha='center', va='center', fontsize=12, fontweight='bold', color="#145A32")

    engine_desc = (
        "Expected Utility Optimization:\n"
        "  B(x) = 0.5·U(x) + 0.5·N(x)\n"
        "  H(x) = (1 - R(x))·(1 + U(x))\n"
        "  U_net(x) = 0.5·(B - H + 1)\n\n"
        "Calibrated Policy Thresholds:\n"
        "  • LEARN : U_net ≥ 0.55 (τ_learn)\n"
        "  • DEFER : 0.35 ≤ U_net < 0.55\n"
        "  • IGNORE: U_net < 0.35 (τ_defer)"
    )
    ax.text(11.75, 6.45, engine_desc, ha='left', va='center', fontsize=9.0, color="#0E6251", linespacing=1.45)

    # ------------------ BOTTOM ROW: ACTIONS (Y: 0.8 to 3.6) ------------------

    # ACTION A: LEARN (Left)
    draw_box(0.5, 0.8, 4.2, 2.8, "#EAFAF1", "#27AE60", 2.2)
    ax.text(2.6, 3.25, "ACTION: LEARN", ha='center', va='center', fontsize=12, fontweight='bold', color="#145A32")
    learn_desc = (
        "• Execute Gradient Backpropagation\n"
        "• Compute Cross-Entropy Loss L(f(x), y)\n"
        "• Update Model Weights: θ ← θ - η∇L\n"
        "• Update Class Centroid Prototype c_k:\n"
        "   c_k ← (1-α)c_k + α·h\n"
        "• Preserves Representation Orthogonality"
    )
    ax.text(0.7, 1.95, learn_desc, ha='left', va='center', fontsize=9.2, color="#196F3D", linespacing=1.55)

    # ACTION B: IGNORE (Center)
    draw_box(5.6, 0.8, 4.2, 2.8, "#FDEDEC", "#C0392B", 2.2)
    ax.text(7.7, 3.25, "ACTION: IGNORE", ha='center', va='center', fontsize=12, fontweight='bold', color="#78281F")
    ignore_desc = (
        "• High Poison Hazard or Redundant\n"
        "• Zero Backward Computation (Detach)\n"
        "• Discard Instance Immediately\n"
        "• Intercepts 96.25% of Corrupted Data\n"
        "• Saves 61.6% of Backward FLOPs\n"
        "• Active Firewall Against Poisoning"
    )
    ax.text(5.8, 1.95, ignore_desc, ha='left', va='center', fontsize=9.2, color="#922B21", linespacing=1.55)

    # ACTION C: DEFER (Right)
    draw_box(10.7, 0.8, 4.2, 2.8, "#FEF9E7", "#D4AC0D", 2.2)
    ax.text(12.8, 3.25, "ACTION: DEFER", ha='center', va='center', fontsize=12, fontweight='bold', color="#7D6608")
    defer_desc = (
        "• Epistemic Ambiguity Quarantine\n"
        "• Enqueue to Bounded FIFO Queue Q_defer\n"
        "• Bounded Buffer Capacity (M = 128)\n"
        "• Periodic Re-evaluation (Every 10 Steps)\n"
        "• Rescore with Evolved Model State\n"
        "• Promotes to LEARN if Matured / Purge"
    )
    ax.text(10.9, 1.95, defer_desc, ha='left', va='center', fontsize=9.2, color="#9A7D0A", linespacing=1.55)

    # ------------------ CLEAN BUS ROUTING (BOX 4 -> ACTION BOXES) ------------------
    # Channel line: from bottom of Decision Engine (X=13.25, Y=4.8) down to Y=4.20
    ax.plot([13.25, 13.25], [4.8, 4.20], color="#1E8449", lw=2.2, zorder=2)
    # Horizontal bus bar from X=3.2 to X=13.25 at Y=4.20
    ax.plot([3.2, 13.25], [4.20, 4.20], color="#1E8449", lw=2.2, zorder=2)

    # Drop 1: Down to LEARN (X=3.2)
    ax.annotate("", xy=(3.2, 3.6), xytext=(3.2, 4.20),
                arrowprops=dict(arrowstyle="-|>", color="#27AE60", lw=2.5, mutation_scale=16), zorder=4)
    ax.text(3.2, 4.45, "LEARN: U_net ≥ 0.55", fontsize=9.0, fontweight='bold', color="#1E8449",
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.25", fc="#EAFAF1", ec="#27AE60", lw=1.2), zorder=5)

    # Drop 2: Down to IGNORE (X=7.7)
    ax.annotate("", xy=(7.7, 3.6), xytext=(7.7, 4.20),
                arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=2.5, mutation_scale=16), zorder=4)
    ax.text(7.7, 4.45, "IGNORE: U_net < 0.35", fontsize=9.0, fontweight='bold', color="#922B21",
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.25", fc="#FDEDEC", ec="#C0392B", lw=1.2), zorder=5)

    # Drop 3: Down to DEFER (X=12.8)
    ax.annotate("", xy=(12.8, 3.6), xytext=(12.8, 4.20),
                arrowprops=dict(arrowstyle="-|>", color="#D4AC0D", lw=2.5, mutation_scale=16), zorder=4)
    ax.text(12.8, 4.45, "DEFER: 0.35 ≤ U_net < 0.55", fontsize=8.5, fontweight='bold', color="#7D6608",
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.25", fc="#FEF9E7", ec="#D4AC0D", lw=1.2), zorder=5)

    # ------------------ FEEDBACK LOOPS ------------------
    # Feedback from LEARN back up to NEURAL BACKBONE (From X=1.5, Y=3.6 curving up to X=4.2, Y=5.0)
    ax.annotate("", xy=(4.2, 5.0), xytext=(1.2, 3.6),
                arrowprops=dict(arrowstyle="-|>", color="#27AE60", lw=2.2, ls="--",
                                connectionstyle="arc3,rad=0.35", mutation_scale=16), zorder=4)
    ax.text(1.7, 4.50, "Weights & Prototypes\nUpdated", fontsize=8.0, fontweight='bold',
            color="#1E8449", ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFF", ec="#27AE60", lw=1.2), zorder=5)

    # Re-eval loop from DEFER back up to GATING DECISION (from top right of DEFER X=14.5, Y=3.6 curving up to X=14.8, Y=5.5)
    ax.annotate("", xy=(14.8, 5.5), xytext=(14.5, 3.6),
                arrowprops=dict(arrowstyle="-|>", color="#B7950B", lw=2.0, ls=":",
                                connectionstyle="arc3,rad=-0.3", mutation_scale=15), zorder=4)
    ax.text(14.95, 4.45, "Re-evaluate\nCadence\n(K = 10)", fontsize=7.5, fontweight='bold',
            color="#7D6608", ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec="#B7950B", lw=1.0), zorder=5)

    plt.tight_layout()

    for p in output_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"Saved architectural diagram to: {p}")

    plt.close()

if __name__ == "__main__":
    paths = [
        Path("research/report/fig_architecture.png"),
        Path("research/plots/fig_architecture.png")
    ]
    create_architecture_diagram(paths)
