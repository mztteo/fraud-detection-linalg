import numpy as np

def jacobi(A, b, max_iter=500, tol=1e-6):
    n = len(b)
    w = np.zeros(n)

    for k in range(max_iter):
        w_new = np.zeros(n)
        for i in range(n):
            s = sum(A[i,j] * w[j] for j in range(n) if j != i)
            w_new[i] = (b[i] - s) / A[i,i]
        residual = np.linalg.norm(A @ w_new - b)
        w = w_new
        if residual < tol:
            print(f"Jacobi : convergé en {k+1} itérations")
            return w, k+1

    print(f"Jacobi : diverge après {max_iter} itérations")
    return w, max_iter


def gauss_seidel(A, b, max_iter=500, tol=1e-6):
    n = len(b)
    w = np.zeros(n)

    for k in range(max_iter):
        for i in range(n):
            s = np.dot(A[i], w) - A[i,i] * w[i]
            w[i] = (b[i] - s) / A[i,i]
        residual = np.linalg.norm(A @ w - b)
        if residual < tol:
            print(f"Gauss-Seidel : convergé en {k+1} itérations")
            return w, k+1

    print(f"Gauss-Seidel : diverge après {max_iter} itérations")
    return w, max_iter


def sor(A, b, omega=1.2, max_iter=500, tol=1e-6):
    n = len(b)
    w = np.zeros(n)

    for k in range(max_iter):
        for i in range(n):
            s = np.dot(A[i], w) - A[i,i] * w[i]
            w_gs = (b[i] - s) / A[i,i]
            w[i] = omega * w_gs + (1 - omega) * w[i]
        residual = np.linalg.norm(A @ w - b)
        if residual < tol:
            print(f"SOR (ω={omega}) : convergé en {k+1} itérations")
            return w, k+1

    print(f"SOR (ω={omega}) : diverge après {max_iter} itérations")
    return w, max_iter


def build_system(X_train, y_train, lambda_r=1e-3):
    XtX = X_train.T.dot(X_train).toarray()
    A = XtX + lambda_r * np.eye(X_train.shape[1])
    b = X_train.T.dot(y_train)
    return A, b
