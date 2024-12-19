import numpy as np

def random_orthogonal_matrix(n, eps = 0):
    """ Генерирует случайную ортогональную матрицу с помощью QR-разложения """
    X = np.random.normal(0, 1, size=(n, n)) + eps * np.eye(n) # Используем регуляризацию гребнем для стабильности
    Q, R = np.linalg.qr(X)
    return Q