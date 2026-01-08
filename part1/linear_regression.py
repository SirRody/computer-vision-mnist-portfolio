import numpy as np

def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1
    """
   
    n, d_plus_1 = X.shape
    d = d_plus_1 - 1  # since we have d features + 1 bias term
    
    # Create identity matrix of appropriate size
    I = np.eye(d_plus_1)
    
    # Compute the closed-form solution: θ = (X^T X + λI)^(-1) X^T Y
    # Note: X.T @ X is matrix multiplication (dot product) in NumPy
    X_T_X = X.T @ X
    X_T_Y = X.T @ Y
    
    # Add regularization term: λI
    regularized_X_T_X = X_T_X + lambda_factor * I
    
    # Compute the inverse and solve for theta
    theta = np.linalg.inv(regularized_X_T_X) @ X_T_Y
    
    return theta
    raise NotImplementedError

def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
