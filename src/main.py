import numpy as np
from data_simulation import generate_data, split_data
from solvers import build_system, jacobi, gauss_seidel, sor
from logistic_regression import train, find_best_threshold, evaluate
from predict import save_model
from plots import plot_all

# ── 1. Données ──
print("=" * 50)
print("1. GÉNÉRATION DES DONNÉES")
print("=" * 50)
X, y = generate_data()
X_train, X_test, y_train, y_test = split_data(X, y)
print(f"Train : {X_train.shape[0]} transactions")
print(f"Test  : {X_test.shape[0]} transactions")
print(f"Fraudes train : {y_train.sum():.0f}")
print(f"Fraudes test  : {y_test.sum():.0f}")

# ── 2. Solveurs ──
print("\n" + "=" * 50)
print("2. SOLVEURS ITÉRATIFS")
print("=" * 50)
A, B = build_system(X_train, y_train)
w_j,   iters_j   = jacobi(A, B)
w_gs,  iters_gs  = gauss_seidel(A, B)
w_sor, iters_sor = sor(A, B, omega=1.2)

# ── 3. Régression logistique ──
print("\n" + "=" * 50)
print("3. RÉGRESSION LOGISTIQUE")
print("=" * 50)
w, b, sample_weights, history = train(X_train, y_train)

from data_simulation import apply_smote

# ── 3b. Entraînement avec SMOTE ──
print("\n" + "=" * 50)
print("3b. RÉGRESSION LOGISTIQUE AVEC SMOTE")
print("=" * 50)

X_train_sm, y_train_sm = apply_smote(X_train, y_train)
print(f"Après SMOTE → {y_train_sm.sum():.0f} fraudes / {len(y_train_sm)} total")

w_sm, b_sm, _, _ = train(X_train_sm, y_train_sm)
threshold_sm = find_best_threshold(X_train_sm, y_train_sm, w_sm, b_sm)
metrics_sm = evaluate(X_test, y_test, w_sm, b_sm, threshold_sm)

# ── 4. Seuil optimal ──
print("\n" + "=" * 50)
print("4. SEUIL OPTIMAL")
print("=" * 50)
threshold = find_best_threshold(X_train, y_train, w, b)

# ── 5. Évaluation ──
print("\n" + "=" * 50)
print("5. ÉVALUATION SUR LE TEST")
print("=" * 50)
metrics = evaluate(X_test, y_test, w, b, threshold)

# ── 6. Sauvegarde ──
print("\n" + "=" * 50)
print("6. SAUVEGARDE")
print("=" * 50)
save_model(w, b, threshold)

# ── 7. Graphiques ──
print("\n" + "=" * 50)
print("7. GRAPHIQUES")
print("=" * 50)
plot_all(A, B, w, b, threshold, y_test, X_test, metrics)

from shap_analysis import compute_shap, plot_shap, plot_shap_waterfall

# ── 8. SHAP ──
print("\n" + "=" * 50)
print("8. ANALYSE SHAP")
print("=" * 50)

shap_values, X_explain = compute_shap(X_train, X_test, y_test, w, b)
mean_abs_shap = plot_shap(shap_values, X_explain, y_test)
plot_shap_waterfall(shap_values, X_explain, y_test)

# top 5 features les plus importantes
top5 = np.argsort(mean_abs_shap)[-5:][::-1]
print("\nTop 5 features les plus importantes :")
for rank, idx in enumerate(top5, 1):
    print(f"  {rank}. Feature {idx:3d} — SHAP moyen = {mean_abs_shap[idx]:.4f}")
