import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    
    # Compute (theta * X^T) / temp_parameter
    dot_products = (theta @ X.T) / temp_parameter  # (k, n)
    
    # Numerical stability: subtract max for each column
    max_vals = np.max(dot_products, axis=0, keepdims=True)  # (1, n)
    stable_exp = np.exp(dot_products - max_vals)  # (k, n)
    
    # Normalize
    H = stable_exp / np.sum(stable_exp, axis=0, keepdims=True)  # (k, n)
    
    return H
    raise NotImplementedError

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
     
    n = X.shape[0]
    k = theta.shape[0]
    
    # Compute dot products: theta @ X.T, shape: (k, n)
    dot_products = theta @ X.T  # (k, n)
    
    # Scale by temperature
    scaled_dot_products = dot_products / temp_parameter  # (k, n)
    
    # For numerical stability: subtract max for each data point (column)
    max_per_column = np.max(scaled_dot_products, axis=0, keepdims=True)  # (1, n)
    shifted = scaled_dot_products - max_per_column  # (k, n)
    
    # Compute log-sum-exp for each column (data point)
    # log(sum(exp(shifted))) = max + log(sum(exp(shifted - max)))
    exp_shifted = np.exp(shifted)  # (k, n)
    sum_per_column = np.sum(exp_shifted, axis=0, keepdims=True)  # (1, n)
    log_sum_exp = max_per_column + np.log(sum_per_column)  # (1, n)
    
    # For each data point, get the score for the true class
    # Y contains indices [0, k-1], we need to get the corresponding row from scaled_dot_products
    true_class_scores = scaled_dot_products[Y, np.arange(n)]  # (n,)
    
    # Negative log-likelihood: -1/n * sum(true_class_score - log_sum_exp)
    log_likelihood_per_point = true_class_scores - log_sum_exp.flatten()  # (n,)
    negative_log_likelihood = -np.mean(log_likelihood_per_point)
    
    # Regularization term
    regularization_term = (lambda_factor / 2) * np.sum(theta ** 2)
    
    # Total cost
    total_cost = negative_log_likelihood + regularization_term
    
    return total_cost

    raise NotImplementedError

def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    
    n = X.shape[0]
    k = theta.shape[0]
    
    # Compute probabilities
    P = compute_probabilities(X, theta, temp_parameter)
    
    # Create sparse indicator matrix I
    rows = Y
    cols = np.arange(n)
    data = np.ones(n)
    I = sparse.coo_matrix((data, (rows, cols)), shape=(k, n))
    
    # Compute gradient efficiently using sparse operations
    # We need to compute: (P - I) @ X
    # This equals: P @ X - I @ X
    
    # Compute P @ X (dense)
    P_X = P @ X  # Shape: (k, d)
    
    # Compute I @ X (sparse multiplication, result is dense)
    I_X = I @ X  # Shape: (k, d)
    
    # Compute (P - I) @ X = P @ X - I @ X
    gradient_data_term = P_X - I_X
    
    # Scale by 1/(τn)
    scaled_gradient = gradient_data_term / (temp_parameter * n)
    
    # Add regularization and update theta
    gradient = scaled_gradient + lambda_factor * theta
    theta = theta - alpha * gradient
    
    return theta
    raise NotImplementedError

def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
     # Use modulo 3 to convert digits to their mod 3 values
    train_y_mod3 = train_y % 3
    test_y_mod3 = test_y % 3
    
    return train_y_mod3, test_y_mod3
    
    raise NotImplementedError

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    # First, we need to add a column of ones to X for the bias term
    # X has shape (n, d-1), we need to add bias to make it (n, d)
    #X_with_bias = np.hstack([np.ones([X.shape[0], 1]), X])
    
    # Use get_classification to predict the original digits (0-9)
    predicted_digits = get_classification(X, theta, temp_parameter)
    
    # Convert predicted digits to mod 3 labels
    predicted_mod3 = predicted_digits % 3
    
    # Calculate error rate
    test_error = np.mean(predicted_mod3 != Y)
    
    return test_error
    
    raise NotImplementedError

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
