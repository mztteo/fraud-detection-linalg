import numpy as np
import shap
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.model_selection import train_test_split
from logistic_regression import sigmoid

def compute_shap(X_train, X_test, y_test, w, b, n_background=100, n_explain=50):


    # SHAP a besoin de données denses
    X_train_dense = X_train.toarray()
    X_test_dense  = X_test.toarray()

    # background = sous-ensemble pour estimer les valeurs de base
    background = X_train_dense[:n_background]

    # explainer
    explainer = shap.LinearExplainer(
        (w, b),           # (coefficients, intercept)
        background,
        feature_perturbation="interventional"
    )

    # on explique les n_explain premières transactions du test
    X_explain = X_test_dense[:n_explain]
    shap_values = explainer.shap_values(X_explain)

    return shap_values, X_explain
def plot_shap(shap_values, X_explain, y_test, save_path="shap_summary.png"):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── 1. Importance globale des features ──
    # moyenne des |SHAP values| par feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-15:]  # top 15 features

    axes[0].barh(
        [f"Feature {i}" for i in top_idx],
        mean_abs_shap[top_idx],
        color=['#1a3a6b' if v > mean_abs_shap.mean() else '#aab8cc'
               for v in mean_abs_shap[top_idx]]
    )
    axes[0].set_xlabel("Importance SHAP moyenne |E[φᵢ]|", fontsize=11)
    axes[0].set_title("Top 15 features — importance globale", fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

    # ── 2. SHAP values pour une fraude vs une transaction légitime ──
    # trouver une fraude et un légitime dans X_explain
    y_sub = y_test[:len(X_explain)]
    fraud_idx  = np.where(y_sub == 1)[0]
    legit_idx  = np.where(y_sub == 0)[0]

    if len(fraud_idx) > 0 and len(legit_idx) > 0:
        fi = fraud_idx[0]
        li = legit_idx[0]

        # top features actives pour ces deux transactions
        top_features = np.argsort(np.abs(shap_values[fi]))[-10:]

        x = np.arange(len(top_features))
        w_bar = 0.35

        axes[1].bar(x - w_bar/2, shap_values[fi][top_features],
                    width=w_bar, color='#cc0000', alpha=0.8, label='Fraude')
        axes[1].bar(x + w_bar/2, shap_values[li][top_features],
                    width=w_bar, color='#2d6a2d', alpha=0.8, label='Légitime')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"F{i}" for i in top_features], fontsize=9)
        axes[1].axhline(0, color='black', lw=0.8)
        axes[1].set_ylabel("Valeur SHAP φᵢ", fontsize=11)
        axes[1].set_title("Contributions SHAP — fraude vs légitime", fontsize=11, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("Analyse SHAP — Explicabilité du modèle de détection de fraude",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"SHAP sauvegardé → {save_path}")

    return mean_abs_shap

# à ajouter dans shap_analysis.py

def plot_shap_waterfall(shap_values, X_explain, y_test, save_path="shap_waterfall.png"):
    """
    Waterfall plot pour une transaction frauduleuse précise.
    Montre comment chaque feature pousse vers fraude ou légitime.
    """
    y_sub = y_test[:len(X_explain)]
    fraud_idx = np.where(y_sub == 1)[0]

    if len(fraud_idx) == 0:
        print("Pas de fraude dans X_explain")
        return

    fi = fraud_idx[0]
    sv = shap_values[fi]

    # garder seulement les features non nulles et les trier
    nonzero = np.where(X_explain[fi] != 0)[0]
    if len(nonzero) == 0:
        nonzero = np.argsort(np.abs(sv))[-10:]

    top = nonzero[np.argsort(np.abs(sv[nonzero]))[-10:]]
    top = top[np.argsort(sv[top])]  # trier par valeur

    vals = sv[top]
    labels = [f"Feature {i}\n(val={X_explain[fi,i]:.2f})" for i in top]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#cc0000' if v > 0 else '#2d6a2d' for v in vals]
    bars = ax.barh(labels, vals, color=colors, alpha=0.85)

    # valeur de base (expected value)
    ax.axvline(0, color='black', lw=0.8, ls='--')

    for bar, val in zip(bars, vals):
        ax.text(val + (0.001 if val >= 0 else -0.001),
                bar.get_y() + bar.get_height()/2,
                f"{val:+.4f}",
                va='center', ha='left' if val >= 0 else 'right',
                fontsize=9, color='#333333')

    ax.set_xlabel("Contribution SHAP φᵢ\n(positif → pousse vers fraude, négatif → pousse vers légitime)", fontsize=10)
    ax.set_title(f"Waterfall SHAP — Transaction frauduleuse #{fi}\n"
                 f"(P(fraude) calculée par le modèle)", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # légende
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#cc0000', alpha=0.85, label='Pousse vers fraude (+)'),
        Patch(color='#2d6a2d', alpha=0.85, label='Pousse vers légitime (−)'),
    ], loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Waterfall sauvegardé → {save_path}")

