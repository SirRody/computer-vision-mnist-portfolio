import numpy as np

def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # Compute the dot product matrix: X @ Y^T
    # This gives an (n, m) matrix where element (i, j) is the dot product of X[i] and Y[j]
    dot_matrix = X @ Y.T
    
    # Apply polynomial kernel: (dot_product + c)^p
    kernel_matrix = (dot_matrix + c) ** p
    
    return kernel_matrix
    raise NotImplementedError



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    n = X.shape[0]
    m = Y.shape[0]
    kernel_matrix = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            # Compute squared Euclidean distance
            squared_distance = np.sum((X[i] - Y[j]) ** 2)
            # Apply RBF kernel
            kernel_matrix[i, j] = np.exp(-gamma * squared_distance)
    
    return kernel_matrix
    raise NotImplementedError
