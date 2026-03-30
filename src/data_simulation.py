import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split

def generate_data(n_transactions=5000, n_features=200,
                  density=0.01, random_state=42):
    X = sparse.random(n_transactions, n_features,
                      density=density, format="csr",
                      random_state=random_state)
    true_weights = np.random.rand(n_features)
    scores = X.dot(true_weights)
    y = (scores > np.percentile(scores, 95)).astype(float)
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state,
                            stratify=y)

from imblearn.over_sampling import SMOTE

def apply_smote(X_train, y_train, random_state=42):
    """
    SMOTE génère des exemples synthétiques de la classe minoritaire
    en interpolant entre les exemples existants.
    → avant : 3800 légitimes, 200 fraudes
    → après : 3800 légitimes, 3800 fraudes (équilibré)
    """
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train.toarray(), y_train)
    # on re-convertit en sparse après SMOTE
    from scipy.sparse import csr_matrix
    return csr_matrix(X_res), y_res
