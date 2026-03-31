# 🛡️ Détection d'Anomalies Bancaires : Algèbre & ML From Scratch

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📌 Présentation du Projet
Ce projet présente la conception et l'implémentation complète d'un pipeline de détection d'anomalies bancaires, construit entièrement à partir des fondements de l'algèbre linéaire. L'objectif est de démontrer que les outils de l'analyse matricielle constituent le socle commun du machine learning et de l'optimisation numérique.

### Points clés techniques :
* **Matrices Creuses (Format CSR) :** Exploitation de la sparsité naturelle des transactions pour réduire l'empreinte mémoire d'un facteur 100.
* **Solveurs Itératifs :** Implémentation *from scratch* de Jacobi, Gauss-Seidel et SOR.
* **ML Natif :** Régression logistique entraînée par descente de gradient mini-batch avec gestion du déséquilibre.
* **Explicabilité (SHAP) :** Attribution des contributions de chaque variable à la décision de détection selon la théorie des jeux coopératifs.

## 📊 Performances du Modèle
* **Score de discrimination :** ROC-AUC de **0.967**
* **Capacité de détection (avec SMOTE) :** Rappel de **74%**, soit 37 fraudes détectées sur 50.
* **Optimisation Numérique :** Convergence de Gauss-Seidel en seulement **14 itérations**.

## 🛠️ Architecture du Dépôt
Le pipeline est organisé en modules à responsabilités séparées pour garantir la testabilité :
* `src/solvers.py` : Algorithmes de résolution numérique.
* `src/logistic_regression.py` : Moteur d'entraînement et calcul du gradient.
* `src/shap_analysis.py` : Analyse d'importance et Waterfall plots.
* `models/fraud_model.npz` : Paramètres du modèle sérialisés.

## 🧠 Insights Théoriques
Le projet démontre que le gradient de la log-loss et les équations normales se ramènent à la même opération fondamentale : le **produit matrice-vecteur creux** $\mathbf{X}^\top \mathbf{v}$.

La convergence de nos solveurs est validée par le **théorème d'Ostrowski-Reich** appliqué à notre matrice SDP (Symétrique Définie Positive) de conditionnement $\kappa \approx 7$.

## 🚀 Installation & Utilisation
1. **Clonez le dépôt :**
   ```bash
   git clone [https://github.com/votre-utilisateur/votre-repo.git](https://github.com/votre-utilisateur/votre-repo.git)
