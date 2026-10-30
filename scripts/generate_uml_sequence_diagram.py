"""
Generates the publication-grade UML Sequence Diagram for Adaptive Learning Gate (ALG).
Visualizes the temporal message-passing flow across:
1. Data Stream Source
2. Neural Backbone (Forward Pass)
3. Tri-Signal Extractor
4. Expected Utility Engine
5. Bounded Quarantine Queue
6. Continual Optimizer (Backward Pass)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_uml_sequence_diagram(output_path: Path):
    fig, ax = plt.subplots(figsize=(14.5, 8.5), dpi=300)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # Background
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Title
    ax.text(7.25, 8.15, "Adaptive Learning Gate (ALG) — UML Sequence Diagram",
            ha='center', va='center', fontsize=16, fontweight='bold', color='#1A252F', family='sans-serif')
    ax.text(7.25, 7.80, "Sequential Execution Flow and Auditable Decision Lifelines in Continual Streaming",
            ha='center', va='center', fontsize=10.5, fontstyle='italic', color='#566573', family='sans-serif')

    # Lifeline actors
    actors = [
        ("Data Stream\n(x_t, y_t)", 1.2, "#2B4C7E"),
        ("Backbone\nf_theta", 3.6, "#155724"),
        ("Tri-Signal\nExtractor", 6.0, "#0C5460"),
        ("Utility Engine\nE[U]", 8.4, "#856404"),
        ("Quarantine\nQueue (FIFO)", 10.8, "#6C757D"),
        ("Optimizer\nUpdate (theta)", 13.2, "#721C24")
    ]

    y_top = 7.3
    y_bot = 0.8

    # Draw Actor Boxes and Lifeline Dashes
    for name, x, color in actors:
        # Box
        box = patches.FancyBboxPatch(
            (x - 0.9, y_top - 0.5), 1.8, 0.6,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor="#F8F9FA", edgecolor=color, linewidth=2.0, zorder=4
        )
        ax.add_patch(box)
        ax.text(x, y_top - 0.2, name, ha='center', va='center', fontsize=9.5, fontweight='bold', color=color)

        # Dashed lifeline
        ax.plot([x, x], [y_top - 0.6, y_bot], color="#BDC3C7", linestyle="--", linewidth=1.5, zorder=1)

    # Helper for messages
    def draw_msg(x1, x2, y, text, color="#2C3E50", is_return=False, note=""):
        ls = "--" if is_return else "-"
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, linestyle=ls, mutation_scale=14), zorder=3)
        mid_x = (x1 + x2) / 2
        ax.text(mid_x, y + 0.12, text, ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=color)
        if note:
            ax.text(mid_x, y - 0.14, note, ha='center', va='top', fontsize=7.5, fontstyle='italic', color='#555555')

    def draw_self_loop(x, y, text, color="#2C3E50"):
        ax.annotate("", xy=(x + 0.05, y - 0.25), xytext=(x + 0.05, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                    connectionstyle="arc3,rad=-1.2", mutation_scale=12), zorder=3)
        ax.text(x + 0.6, y - 0.12, text, ha='left', va='center', fontsize=8.0, color=color, fontweight='bold')

    # Step 1: Stream intake to Backbone
    draw_msg(1.2, 3.6, 6.3, "1: forward_pass(x_t)", "#2B4C7E", note="Input: 3x32x32 image tensor")

    # Step 2: Backbone returns embedding & logits
    draw_msg(3.6, 6.0, 5.7, "2: extract_signals(z_t, p_t, y_t)", "#155724", note="Latent rep z_t in R^512, class logits p_t")

    # Step 3: Tri-signal self computation
    draw_self_loop(6.0, 5.2, "Compute H(p_t), d_proto(z_t), R(z_t, y_t)", "#0C5460")

    # Step 4: Pass signals to Utility Engine
    draw_msg(6.0, 8.4, 4.4, "3: evaluate_utility(H, d_proto, R)", "#0C5460", note="Vector: [Uncertainty, Novelty, Reliability]")

    # Step 5: Utility Engine computes E[U]
    draw_self_loop(8.4, 3.9, "E[U] = w_u*H + w_n*d_proto - w_r*(1-R)", "#856404")

    # Step 6: Decision Branching Box (Alternative frames)
    # Action A: LEARN
    alt_box = patches.Rectangle((0.6, 1.2), 13.2, 2.3, facecolor="#EBF5FB", edgecolor="#3498DB", linewidth=1.2, linestyle=":", zorder=2)
    ax.add_patch(alt_box)
    ax.text(0.7, 3.35, "[Alt: Decision Policy]", fontsize=8.5, fontweight='bold', color="#2980B9")

    # If E[U] > tau_learn -> LEARN
    ax.text(0.7, 3.0, "case 1: E[U] >= tau_learn (LEARN)", fontsize=8, fontweight='bold', color="#196F3D")
    draw_msg(8.4, 13.2, 2.8, "4a: backward_update(Loss(p_t, y_t))", "#196F3D", note="Gradient step + Prototypes updated")

    # If E[U] < tau_defer -> IGNORE
    ax.text(0.7, 2.35, "case 2: E[U] < tau_defer (IGNORE)", fontsize=8, fontweight='bold', color="#922B21")
    draw_self_loop(8.4, 2.1, "4b: Drop sample (0 FLOPs backward, audit log written)", "#922B21")

    # Else -> DEFER
    ax.text(0.7, 1.7, "case 3: tau_defer <= E[U] < tau_learn (DEFER)", fontsize=8, fontweight='bold', color="#D68910")
    draw_msg(8.4, 10.8, 1.5, "4c: enqueue(x_t, y_t, t_arrival)", "#D68910", note="Push to FIFO Bounded Buffer (capacity B=100)")

    # Step 7: Periodic Re-evaluation from Queue
    draw_msg(10.8, 8.4, 0.95, "5: periodic_recheck(deferred_batch)", "#6C757D", is_return=True, note="Matured representation re-adjudicates utility")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated UML Sequence Diagram at: {output_path}")

if __name__ == "__main__":
    out = Path("research/report/fig_uml_sequence.png")
    create_uml_sequence_diagram(out)
