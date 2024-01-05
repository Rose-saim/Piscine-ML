from prediction import predict_
import numpy as np
# from prediction import predict_
def gradient(x, y, theta):
    """
    Computes the gradient of the mean squared error loss function with respect to the parameters theta.

    Args:
    - x: Feature matrix (numpy array), shape (m, n+1) where m is the number of examples and n is the number of features.
    - y: Target values (numpy array), shape (m, 1).
    - theta: Model parameters (numpy array), shape (n+1, 1).

    Returns:
    - gradient: Vector of partial derivatives (numpy array), shape (n+1, 1).
    """
    m = len(y)
    
    # Ajout du terme de biais à la matrice des caractéristiques
    ones_column = np.ones((m, 1))
    X = np.hstack((ones_column, x))
    
    # Calcul des erreurs entre les prédictions et les valeurs cibles
    X_th_y = predict_(x, theta) - y
    
    # Calcul du gradient en utilisant la formule
    gradient = (1 / m) * np.matmul(X.T, X_th_y)
    return gradient
