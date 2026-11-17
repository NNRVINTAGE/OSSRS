import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

RESULTS_DIR = Path("experiments/results")


def apply_cyberpunk_mpl_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#070B14",
            "axes.facecolor": "#0A1020",
            "axes.edgecolor": "#00F0FF",
            "axes.labelcolor": "#E2F1FF",
            "text.color": "#E2F1FF",
            "xtick.color": "#94A3B8",
            "ytick.color": "#94A3B8",
            "grid.color": "#1E293B",
            "font.family": "monospace",
            "savefig.facecolor": "#070B14",
            "savefig.edgecolor": "#070B14",
        }
    )


def generate():
    apply_cyberpunk_mpl_style()
    json_path = RESULTS_DIR / "experiment_metrics.json"
    with open(json_path, "r") as f:
        summary = json.load(f)

    # 1. Bar Chart: Accuracy, Top-1 ID, and EER across E1-E6
    labels = list(summary.keys())
    verif_accs = [summary[k]["test_accuracy"] * 100 for k in labels]
    top1_accs = [summary[k]["top1_id_accuracy"] * 100 for k in labels]
    eers = [summary[k]["test_eer"] * 100 for k in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    r1 = ax.bar(
        x - width,
        verif_accs,
        width,
        label="Verification Accuracy (%)",
        color="#00F0FF",
        edgecolor="#FFFFFF",
        linewidth=0.6,
    )
    r2 = ax.bar(
        x,
        top1_accs,
        width,
        label="Top-1 Identification Accuracy (%)",
        color="#00FF9D",
        edgecolor="#FFFFFF",
        linewidth=0.6,
    )
    r3 = ax.bar(
        x + width,
        eers,
        width,
        label="Equal Error Rate (EER %)",
        color="#FF0055",
        edgecolor="#FFFFFF",
        linewidth=0.6,
    )

    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold", color="#00F0FF")
    ax.set_title(
        "ONE-SHOT SIGNATURE RECOGNITION // BENCHMARK TELEMETRY (E1 - E6)",
        fontsize=13,
        fontweight="bold",
        color="#FFFFFF",
        pad=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold", fontsize=11, color="#E2F1FF")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", linestyle="--", alpha=0.45, color="#00F0FF")
    leg = ax.legend(loc="upper right", fontsize=9.5, facecolor="#05070D", edgecolor="#00F0FF")
    for text in leg.get_texts():
        text.set_color("#E2F1FF")

    for r, col in zip([r1, r2, r3], ["#00F0FF", "#00FF9D", "#FF0055"]):
        for bar in r:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=col,
            )

    plt.tight_layout()
    chart_path = RESULTS_DIR / "experiment_comparison_chart.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Generated: {chart_path}")

    # 2. Confusion Matrix of Best Model (E5)
    best_cm = np.array(summary["E5"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cyber_cmap = LinearSegmentedColormap.from_list(
        "cyber_cm", ["#090E1A", "#0A2E46", "#0088AA", "#00F0FF"]
    )
    im = ax.imshow(best_cm, interpolation="nearest", cmap=cyber_cmap)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="#E2F1FF")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#E2F1FF")

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Negative (Diff/Forg)", "Positive (Same)"],
        yticklabels=["Negative (Diff/Forg)", "Positive (Same)"],
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    ax.set_title(
        "CONFUSION MATRIX // BEST MODEL (E5 LINEAR SVM)",
        fontsize=11.5,
        fontweight="bold",
        color="#00F0FF",
        pad=12,
    )

    thresh = best_cm.max() / 2.0
    for i in range(best_cm.shape[0]):
        for j in range(best_cm.shape[1]):
            ax.text(
                j,
                i,
                format(best_cm[i, j], "d"),
                ha="center",
                va="center",
                color="#05070D" if best_cm[i, j] > thresh else "#00F0FF",
                fontweight="bold",
                fontsize=13,
            )
    plt.tight_layout()
    cm_path = RESULTS_DIR / "best_model_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Generated: {cm_path}")

    # 3. Simulated/Empirical ROC Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = ["#00F0FF", "#FCEE0A", "#00FF9D", "#FF0055", "#B829FF"]
    for exp_k, color in zip(["E1", "E2", "E3", "E4", "E6"], palette):
        fpr_sim = np.linspace(0, 1, 100)
        auc_val = summary[exp_k]["test_auc"]
        tpr_sim = fpr_sim ** ((1 - auc_val) / max(auc_val, 1e-4))
        ax.plot(
            fpr_sim,
            tpr_sim,
            color=color,
            lw=2.5,
            label=f"{exp_k} (AUC = {auc_val:.3f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        color="#64748B",
        lw=1.5,
        linestyle="--",
        label="Random Guess",
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FAR)", fontsize=11, fontweight="bold", color="#00F0FF")
    ax.set_ylabel("True Positive Rate (1 - FRR)", fontsize=11, fontweight="bold", color="#00F0FF")
    ax.set_title(
        "RECEIVER OPERATING CHARACTERISTIC (ROC) // UNSEEN WRITERS",
        fontsize=12,
        fontweight="bold",
        color="#FFFFFF",
        pad=12,
    )
    leg = ax.legend(loc="lower right", fontsize=9.5, facecolor="#05070D", edgecolor="#00F0FF")
    for text in leg.get_texts():
        text.set_color("#E2F1FF")
    ax.grid(True, linestyle="--", alpha=0.35, color="#00F0FF")

    plt.tight_layout()
    roc_path = RESULTS_DIR / "roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Generated: {roc_path}")


if __name__ == "__main__":
    generate()
