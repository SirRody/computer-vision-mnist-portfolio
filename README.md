# Computer Vision: MNIST Digit Recognition Portfolio

## 🎯 Project Overview
This portfolio demonstrates the complete evolution of computer vision techniques through the classic MNIST digit recognition challenge. Starting with basic machine learning and progressing to sophisticated deep learning models, this project showcases how different approaches solve the same visual pattern recognition problem.

## 📁 Project Structure

**Part 1: Traditional Machine Learning**
- `linear_regression.py` - Basic linear models
- `softmax.py` - Multi-class classification  
- `svm.py` - Support Vector Machines
- `features.py` - Feature engineering
- `kernel.py` - Kernel methods
- `main.py` - Runs all experiments

**Part 2: Deep Learning**
- `neural_nets.py` - Neural network from scratch (NumPy)
- `nnet_fc.py` - Fully connected network (PyTorch)
- `nnet_cnn.py` - Convolutional neural network (PyTorch)
- `mlp.py` - Multi-layer perceptron for two digits
- `conv.py` - CNN for overlapping digits
- `train_utils.py` - Training utilities

## 🔍 The Problem
The MNIST dataset contains 70,000 handwritten digits (0-9) that humans recognize easily but computers historically struggled with. Each 28×28 pixel image presents challenges: handwriting variations, different stroke styles, and positional inconsistencies. The advanced challenge adds two overlapping digits that must be recognized separately.

## 🚧 Challenges Faced
1. **High Dimensionality**: 784 features per image
2. **Non-linear Patterns**: Digits aren't linearly separable
3. **Overlapping Digits**: Predicting two digits simultaneously
4. **Performance Optimization**: Balancing accuracy with training time
5. **Generalization**: Working on unseen handwriting styles

## 📊 Results Comparison

| Method | Architecture | Single-Digit | Two-Digit | Key Insight |
|--------|--------------|--------------|-----------|-------------|
| Linear Regression | Basic linear | 23.03% | N/A | Too simple for images |
| SVM (RBF Kernel) | Kernel method | 93.64% | N/A | Great with feature engineering |
| Neural Network | Manual (NumPy) | 85%+ | N/A | Educational foundation |
| Fully Connected | 1 hidden layer | 97.46% | 92.26% | Deep learning advantage |
| **Convolutional** | **2 conv layers** | **99.19%** | **97.74%** | **Best for visual patterns** |

**Key Finding**: Convolutional Neural Networks outperformed all other methods, demonstrating superior capability for image recognition.

## 🌍 Real-World Applications
These techniques power technologies you use daily:

**Everyday Uses:**
- Postal mail sorting and zip code reading
- Banking check processing
- Educational test grading

**Advanced Applications:**
- Self-driving car traffic sign recognition
- Medical imaging analysis
- Retail "Just Walk Out" systems (Amazon Go)
- Accessibility tools for visually impaired

**Business Impact:** A 99% accurate system processes 10,000 documents with only 100 errors vs. 2,300 errors with linear methods—saving hundreds of work hours monthly.

## 🎓 Conclusion
This project shows a fundamental truth in computer vision: **while simple models provide insights, deep learning—particularly convolutional networks—delivers production-ready performance.** The progression from 23% to 99% accuracy mirrors the industry's journey from basic algorithms to today's AI revolution.

The skills demonstrated form the foundation for tackling complex computer vision problems in healthcare, autonomous systems, and beyond.

---

## 👨‍💻 Author: Rodrick
**Data Scientist & Machine Learning Engineer**

A results-driven professional with expertise in transforming theoretical machine learning concepts into practical, production-ready solutions. This project represents a comprehensive journey through computer vision—from fundamental algorithms to cutting-edge deep learning—demonstrating both breadth of knowledge and depth of implementation skill. Passionate about leveraging AI to solve real-world problems and create tangible business value.

*"The true test of understanding isn't just knowing algorithms exist, but implementing them to solve progressively harder problems."*