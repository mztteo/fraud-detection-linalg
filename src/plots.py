import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from logistic_regression import sigmoid

def plot_all(A, B, w, b, threshold, y_test, X_test, metrics, path="resultats.png"):

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Détection de Fraude Bancaire — Résultats",
                 fontsize=13, fontweight='bold')

    # ── 1. Convergence des solveurs ──
    rest_gs, rest_sor = [], []

    w_temp = np.zeros(len(B))
    for k in range(30):
        for i in range(len(B)):
            s = np.dot(A[i], w_temp) - A[i,i] * w_temp[i]
            w_temp[i] = (B[i] - s) / A[i,i]
        rest_gs.append(np.linalg.norm(A @ w_temp - B))

    w_temp = np.zeros(len(B))
    for k in range(30):
        for i in range(len(B)):
            s = np.dot(A[i], w_temp) - A[i,i] * w_temp[i]
            w_gs_i = (B[i] - s) / A[i,i]
            w_temp[i] = 1.2 * w_gs_i + (1 - 1.2) * w_temp[i]
        rest_sor.append(np.linalg.norm(A @ w_temp - B))

    axes[0].semilogy(rest_gs, color='#4fffb0', label='Gauss-Seidel', linewidth=2)
    axes[0].semilogy(rest_sor, color='#6e8fff', label='SOR ω=1.2', linewidth=2)
    axes[0].set_title("Convergence des solveurs")
    axes[0].set_xlabel("Itération")
    axes[0].set_ylabel("Résidu ‖Aw-b‖")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── 2. Courbe ROC ──
    y_hat_test = sigmoid(X_test @ w + b)
    fpr, tpr, _ = roc_curve(y_test, y_hat_test)
    axes[1].plot(fpr, tpr, color='#6e8fff', linewidth=2,
                 label=f"AUC = {metrics['auc']:.3f}")
    axes[1].plot([0,1], [0,1], '--', color='gray', linewidth=1, label='Aléatoire')
    axes[1].set_title("Courbe ROC")
    axes[1].set_xlabel("Taux faux positifs")
    axes[1].set_ylabel("Taux vrais positifs")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ── 3. Matrice de confusion ──
    tp, fp, fn, tn = metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn']
    cm = np.array([[tn, fp], [fn, tp]])
    axes[2].imshow(cm, cmap='Blues')
    axes[2].set_xticks([0,1])
    axes[2].set_yticks([0,1])
    axes[2].set_xticklabels(['Prédit Légit', 'Prédit Fraude'])
    axes[2].set_yticklabels(['Réel Légit', 'Réel Fraude'])
    axes[2].set_title("Matrice de confusion (test)")
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, str(int(cm[i,j])),
                        ha='center', va='center',
                        fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Graphique sauvegardé → {path}")
