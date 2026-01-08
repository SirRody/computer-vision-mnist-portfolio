import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *

#######################################################################
# 1. Introduction
#######################################################################

# Load MNIST data:
train_x, train_y, test_x, test_y = get_MNIST_data()
# Plot the first 20 images of the training set.
plot_images(train_x[0:20, :])

#######################################################################
# 2. Linear Regression with Closed Form Solution
#######################################################################


def run_linear_regression_on_MNIST(lambda_factor=1000):
    """
    Trains linear regression, classifies test data, computes test error on test set

    Returns:
        Final test error
    """
    # Load MNIST data
    train_x, train_y, test_x, test_y = get_MNIST_data()
    
    # Add bias term (column of ones) to features
    # train_x has shape (60000, 784), we add column to make (60000, 785)
    train_x_b = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_b = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    
    # Train linear regression model using closed form solution
    theta = closed_form(train_x_b, train_y, lambda_factor)
    
    # Make predictions on test set
    # test_x_b has shape (10000, 785), theta has shape (785,)
    # Result will be (10000,) - continuous predictions
    test_predictions = test_x_b.dot(theta)
    
    # Round predictions to nearest integer (0-9)
    # Note: np.round returns float, convert to int
    test_predictions = np.round(test_predictions).astype(int)
    
    # Clip predictions to valid range [0, 9]
    # Some predictions might be outside range after rounding
    test_predictions = np.clip(test_predictions, 0, 9)
    
    # Calculate error rate: fraction of incorrect predictions
    error_rate = np.mean(test_predictions != test_y)
    
    return error_rate

print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=1))


#######################################################################
# 3. Support Vector Machine
#######################################################################


def run_svm_one_vs_rest_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1
    pred_test_y = one_vs_rest_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


print('SVM one vs. rest test_error:', run_svm_one_vs_rest_on_MNIST())


def run_multiclass_svm_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = multi_class_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


print('Multiclass SVM test_error:', run_multiclass_svm_on_MNIST())

#######################################################################
# 4. Multinomial (Softmax) Regression and Gradient Descent
#######################################################################


def run_softmax_on_MNIST(temp_parameter=1):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta.pkl.gz")

    # Convert labels to mod 3
    train_y_mod3, test_y_mod3 = update_y(train_y, test_y)
    
    # Compute test error for mod 3 labels using the current model (theta trained on 10 classes)
    test_error_mod3 = compute_test_error_mod3(test_x, test_y_mod3, theta, temp_parameter)
    
    print('test_error_mod3 =', test_error_mod3)
    
    return test_error

print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=1))

# We find the error rate for temp_parameter = [.5, 1.0, 2.0]

#######################################################################
# 6. Changing Labels
#######################################################################



def run_softmax_on_MNIST_mod3(temp_parameter=1):
    """
    Trains Softmax regression on digit (mod 3) classifications.

    See run_softmax_on_MNIST for more info.
    """
    
    # Load MNIST data
    train_x, train_y, test_x, test_y = get_MNIST_data()
    
    # Convert labels to mod 3
    train_y_mod3, test_y_mod3 = update_y(train_y, test_y)
    
    # Train softmax regression with 3 classes (k=3)
    theta, cost_function_history = softmax_regression(train_x, train_y_mod3, temp_parameter, 
                                                     alpha=0.3, lambda_factor=1.0e-4, k=3, 
                                                     num_iterations=150)
    
    # Plot cost function over time
    plot_cost_function_over_time(cost_function_history)
    
    # Compute test error for mod 3 classification
    test_error = compute_test_error_mod3(test_x, test_y_mod3, theta, temp_parameter)
    
    # Save the model parameters theta to disk (optional, but consistent with the original)
    write_pickle_data(theta, "./theta_mod3.pkl.gz")
    
    return test_error

    raise NotImplementedError


# Run run_softmax_on_MNIST_mod3(), report the error rate

print('softmax test_error_mod3 (retrained) =', run_softmax_on_MNIST_mod3(temp_parameter=1))

#######################################################################
# 7. Classification Using Manually Crafted Features
#######################################################################

## Dimensionality reduction via PCA ##

n_components = 18

###Correction note:  the following 4 lines have been modified since release.
train_x_centered, feature_means = center_data(train_x)
pcs = principal_components(train_x_centered)
train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)


print("Training softmax regression on 18-dimensional PCA features...")
theta_pca, cost_function_history_pca = softmax_regression(train_pca, train_y, temp_parameter=1, 
                                                         alpha=0.3, lambda_factor=1.0e-4, 
                                                         k=10, num_iterations=150)

# Compute test error on PCA features
test_error_pca = compute_test_error(test_pca, test_y, theta_pca, temp_parameter=1)
print('Softmax test error on 18-dimensional PCA features:', test_error_pca)

#       Use the plot_PC function in features.py to produce scatterplot
#       of the first 100 MNIST images, as represented in the space spanned by the
#       first 2 principal components found above.
plot_PC(train_x[range(000, 100), ], pcs, train_y[range(000, 100)], feature_means)#feature_means added since release


#       Use the reconstruct_PC function in features.py to show
#       the first and second MNIST images as reconstructed solely from
#       their 18-dimensional principal component representation.
#       Compare the reconstructed images with the originals.
firstimage_reconstructed = reconstruct_PC(train_pca[0, ], pcs, n_components, train_x, feature_means)#feature_means added since release
plot_images(firstimage_reconstructed)
plot_images(train_x[0, ])

secondimage_reconstructed = reconstruct_PC(train_pca[1, ], pcs, n_components, train_x, feature_means)#feature_means added since release
plot_images(secondimage_reconstructed)
plot_images(train_x[1, ])


## Cubic Kernel ##
# TODO: Find the 10-dimensional PCA representation of the training and test set
# ======================================================================
# 10-dimensional PCA for Cubic Features
# ======================================================================

print("\n=== Computing 10-dimensional PCA for Cubic Features ===")
n_components_10 = 10

# Project onto first 10 principal components
train_pca10 = project_onto_PC(train_x, pcs, n_components_10, feature_means)
test_pca10 = project_onto_PC(test_x, pcs, n_components_10, feature_means)

print(f"train_pca10 shape: {train_pca10.shape}")
print(f"test_pca10 shape: {test_pca10.shape}")


# ======================================================================
# Cubic Feature Mapping
# ======================================================================

print("\n=== Applying Cubic Feature Mapping ===")
train_cube = cubic_features(train_pca10)
test_cube = cubic_features(test_pca10)
# train_cube (and test_cube) is a representation of our training (and test) data
# after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.


#       Train your softmax regression model using (train_cube, train_y)
#       and evaluate its accuracy on (test_cube, test_y).
# ======================================================================
# Train Softmax on Cubic Features
# ======================================================================

print("\n=== Training Softmax Regression on Cubic Features ===")
theta_cube, cost_history_cube = softmax_regression(train_cube, train_y, temp_parameter=1,
                                                   alpha=0.3, lambda_factor=1.0e-4, 
                                                   k=10, num_iterations=150)

test_error_cube = compute_test_error(test_cube, test_y, theta_cube, temp_parameter=1)
print(f'Softmax test error on cubic features: {test_error_cube}')

# ======================================================================
# Polynomial SVM (Cubic Kernel) on 10-dimensional PCA
# ======================================================================

print("\n=== Training Polynomial SVM (degree=3) on 10-dimensional PCA ===")

# Import SVC for kernel SVM (if not already imported)
from sklearn.svm import SVC

# Create polynomial SVM with degree 3
poly_svm = SVC(kernel='poly', degree=3, random_state=0)

# Train on 10-dimensional PCA features
poly_svm.fit(train_pca10, train_y)

# Predict on test data
pred_test_y_poly = poly_svm.predict(test_pca10)

# Compute test error
test_error_poly_svm = np.mean(pred_test_y_poly != test_y)
print(f'Polynomial SVM (degree=3) test error on 10-dimensional PCA: {test_error_poly_svm}')

# ======================================================================
# RBF SVM on 10-dimensional PCA
# ======================================================================

print("\n=== Training RBF SVM on 10-dimensional PCA ===")

# Create RBF SVM (SVC with kernel='rbf')
rbf_svm = SVC(kernel='rbf', random_state=0)

# Train on 10-dimensional PCA features
rbf_svm.fit(train_pca10, train_y)

# Predict on test data
pred_test_y_rbf = rbf_svm.predict(test_pca10)

# Compute test error
test_error_rbf_svm = np.mean(pred_test_y_rbf != test_y)
print(f'RBF SVM test error on 10-dimensional PCA: {test_error_rbf_svm}')