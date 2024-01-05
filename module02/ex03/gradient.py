from prediction import predict_

import numpy as np
# from prediction import predict_
def simple_gradient(x, y, theta):
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
    X = np.c_[np.ones(m), x]
    # Calcul des erreurs entre les prédictions et les valeurs cibles
    X_th_y = np.dot(x, theta) - y
    
    # Calcul du gradient en utilisant la formule
    gradient = (1 / m) * np.matmul(X.T, X_th_y)
    return gradient


import numpy as np
x = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1],
[ -8, -4, 6],
[ -5, -9, 6],
[ 1, -5, 11],
[ 9, -11, 8]])
y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
theta1 = np.array([3,0.5,-6]).reshape((-1, 1))
# Example :
print(simple_gradient(x, y, theta1))
# Output:
# array([[ -33.71428571], [ -37.35714286], [183.14285714], [-393.]])
# Example :
theta2 = np.array([0,0,0]).reshape((-1, 1))
print(simple_gradient(x, y, theta2))
# Output:
# array([[ -0.71428571], [ 0.85714286], [23.28571429], [-26.42857143]])