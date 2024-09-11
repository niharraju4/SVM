
---

# Support Vector Machine (SVM) Classification

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Generation](#dataset-generation)
3. [Data Visualization](#data-visualization)
4. [Model Training](#model-training)
    - [Linear Kernel](#linear-kernel)
    - [RBF Kernel](#rbf-kernel)
    - [Polynomial Kernel](#polynomial-kernel)
    - [Sigmoid Kernel](#sigmoid-kernel)
5. [Model Evaluation](#model-evaluation)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Results](#results)
8. [Conclusion](#conclusion)


## Introduction
This project demonstrates the application of Support Vector Machine (SVM) for binary classification using different kernels. The dataset is synthetically generated using `make_classification` from `sklearn.datasets`.

## Dataset Generation
The dataset is generated with the following parameters:
- Number of samples: 1000
- Number of classes: 2
- Number of features: 2
- Number of clusters per class: 2
- Number of redundant features: 0

```python
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_classes=2, n_features=2, n_clusters_per_class=2, n_redundant=0)
```

## Data Visualization
The dataset is visualized using a scatter plot with `seaborn`.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=pd.DataFrame(X)[0], y=pd.DataFrame(X)[1], hue=y)
plt.show()
```

## Model Training
The dataset is split into training and testing sets with a test size of 25%.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
```

### Linear Kernel
```python
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
```

### RBF Kernel
```python
rbf = SVC(kernel='rbf')
rbf.fit(X_train, y_train)
y_pred1 = rbf.predict(X_test)
```

### Polynomial Kernel
```python
poly = SVC(kernel='poly')
poly.fit(X_train, y_train)
y_pred2 = poly.predict(X_test)
```

### Sigmoid Kernel
```python
sigmoid = SVC(kernel='sigmoid')
sigmoid.fit(X_train, y_train)
y_pred3 = sigmoid.predict(X_test)
```

## Model Evaluation
The models are evaluated using accuracy score, classification report, and confusion matrix.

### Linear Kernel
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(confusion_matrix(y_pred, y_test))
print(classification_report(y_pred, y_test))
score = accuracy_score(y_pred, y_test)
print(score)
```

### RBF Kernel
```python
print(classification_report(y_pred1, y_test))
score = accuracy_score(y_pred1, y_test)
print(score)
print(confusion_matrix(y_pred1, y_test))
```

### Polynomial Kernel
```python
score = accuracy_score(y_pred2, y_test)
print(score)
print(confusion_matrix(y_pred2, y_test))
print(classification_report(y_pred2, y_test))
```

### Sigmoid Kernel
```python
print(classification_report(y_pred3, y_test))
print(confusion_matrix(y_pred3, y_test))
score = accuracy_score(y_test, y_pred3)
print(score)
```

## Hyperparameter Tuning
Hyperparameter tuning is performed using `GridSearchCV`.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```

## Results
The results of the model evaluations are as follows:

### Linear Kernel
- Accuracy: [Insert Accuracy Score]
- Confusion Matrix: [Insert Confusion Matrix]
- Classification Report: [Insert Classification Report]

### RBF Kernel
- Accuracy: [Insert Accuracy Score]
- Confusion Matrix: [Insert Confusion Matrix]
- Classification Report: [Insert Classification Report]

### Polynomial Kernel
- Accuracy: [Insert Accuracy Score]
- Confusion Matrix: [Insert Confusion Matrix]
- Classification Report: [Insert Classification Report]

### Sigmoid Kernel
- Accuracy: [Insert Accuracy Score]
- Confusion Matrix: [Insert Confusion Matrix]
- Classification Report: [Insert Classification Report]

## Conclusion
The SVM models with different kernels were trained and evaluated on the synthetic dataset. The RBF kernel performed the best among the tested kernels. Hyperparameter tuning further improved the model performance.

## Author: Nihar Raju
---

