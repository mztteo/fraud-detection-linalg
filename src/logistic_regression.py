import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_sample_weights(y):
    n = len(y)
    n_fraud = y.sum()
    n_legit = n - n_fraud
    return np.where(y == 1,
        n / (2 * n_fraud),   # ~10 pour les fraudes
        n / (2 * n_legit)    # ~0.53 pour les légitimes
    )

def train(X_train, y_train, lambda_r=1e-3,
          lr=0.1, n_epochs=45, batch_size=256):

    n = len(y_train)
    w = np.zeros(X_train.shape[1])
    b = 0.0
    sample_weights = compute_sample_weights(y_train)
    history = []

    pbar = tqdm(range(n_epochs), desc="Entraînement")

    for epoch in pbar:
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            sw_batch = sample_weights[batch_idx]
            n_b = len(batch_idx)

            y_hat = sigmoid(X_batch @ w + b)
            err = sw_batch * (y_hat - y_batch)
            grad_w = X_batch.T.dot(err) / n_b + lambda_r * w
            grad_b = err.mean()
            w -= lr * grad_w
            b -= lr * grad_b

        if epoch % 5 == 0:
            y_hat_all = sigmoid(X_train @ w + b)
            loss = -np.mean(
                sample_weights * (
                    y_train * np.log(y_hat_all + 1e-9) +
                    (1 - y_train) * np.log(1 - y_hat_all + 1e-9)
                )
            )
            history.append((epoch, loss))
            print(f"Époque {epoch:2d} | loss = {loss:.4f}")

    return w, b, sample_weights, history

def find_best_threshold(X_train, y_train, w, b):
    y_hat = sigmoid(X_train @ w + b)
    best_threshold, best_f1 = 0.5, 0.0

    for tau in np.arange(0.1, 0.9, 0.01):
        preds = (y_hat >= tau).astype(float)
        f1 = f1_score(y_train, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = tau

    print(f"Seuil optimal : {best_threshold:.2f} | F1 : {best_f1:.3f}")
    return best_threshold

def evaluate(X_test, y_test, w, b, threshold):
    y_hat = sigmoid(X_test @ w + b)
    y_pred = (y_hat >= threshold).astype(float)

    tp = ((y_pred == 1) & (y_test == 1)).sum()
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    tn = ((y_pred == 0) & (y_test == 0)).sum()

    rappel    = tp / (tp + fn)
    precision = tp / (tp + fp)
    auc       = roc_auc_score(y_test, y_hat)

    print(f"\nMatrice de confusion :")
    print(f"  TN={tn:.0f}  FP={fp:.0f}")
    print(f"  FN={fn:.0f}   TP={tp:.0f}")
    print(f"\nRappel    : {rappel:.3f}")
    print(f"Précision : {precision:.3f}")
    print(f"ROC-AUC   : {auc:.3f}")

    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                rappel=rappel, precision=precision, auc=auc)
