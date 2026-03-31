import numpy as np
from logistic_regression import sigmoid

def save_model(w, b, threshold, path="fraud_model.npz"):
    np.savez(path,
        w=w,
        b=np.array([b]),
        threshold=np.array([threshold])
    )
    print(f"Modèle sauvegardé → {path}")

def load_model(path="fraud_model.npz"):
    model = np.load(path)
    return model['w'], model['b'][0], model['threshold'][0]

def score_transaction(transaction, w, b, threshold):
    """
    transaction : np.array de taille n_features
    retourne    : (probabilité, verdict)
    """
    proba = sigmoid(transaction @ w + b)
    verdict = "FRAUDE" if proba >= threshold else "LEGITIME"
    return proba, verdict


if __name__ == "__main__":
    # exemple d'utilisation
    w, b, threshold = load_model("fraud_model.npz")

    # transaction test
    tx = np.zeros(200)
    tx[3]  = 0.8
    tx[17] = 0.5
    tx[42] = 1.2

    proba, verdict = score_transaction(tx, w, b, threshold)
    print(f"P(fraude)  = {proba:.4f}")
    print(f"Seuil τ*   = {threshold:.2f}")
    print(f"Verdict    = {verdict}")
