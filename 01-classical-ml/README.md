# 📚 01-classical-ml: Fundamentals of Machine Learning

This directory contains essential, hands-on tutorials for foundational **Classical Machine Learning** algorithms, implemented primarily using the **scikit-learn** library in Python.

The notebooks are designed to build a strong theoretical and practical understanding, progressing from crucial data preprocessing to implementing and tuning core classification and regression models.

## 🎯 Notebooks

| \# | Notebook | Topic | Key Concepts Covered | 
 | ----- | ----- | ----- | ----- | 
| 1 | `01-feature-scaling.ipynb` | **Feature Scaling** | Understanding **Normalization** (Min-Max) vs. **Standardization** (Z-score), and their critical role in preparing data for distance-based and gradient-descent algorithms (e.g., k-NN, SVMs, Logistic Regression). | 
| 2 | `02-k-nearest-neighbors.ipynb` | **k-Nearest Neighbors (k-NN)** | The concept of **lazy learning**, how distance metrics (Euclidean, Manhattan) influence prediction, the **bias/variance trade-off** in choosing the value of *k*, and applications for both Classification and Regression. | 
| 3 | `03-decision-trees.ipynb` | **Decision Trees** | The structure and decision process of tree-based models, measures for node splitting (e.g., Gini Impurity), diagnosing **overfitting**, and the importance of **pruning** (e.g., using `max_depth`) for generalization and model interpretability. | 
| 4 | `04-support-vector-machines.ipynb` | **Support Vector Machines (SVM)** | Intuition behind finding the **optimal separating hyperplane**, the difference between Hard Margin and Soft Margin, the crucial **Kernel Trick** (RBF, Polynomial), and the absolute necessity of **feature scaling** for kernel methods. | 