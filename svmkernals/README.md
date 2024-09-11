
# Table of Contents
1. [Introduction](#introduction)
2. [Dataset Generation](#dataset-generation)
3. [Data Preparation](#data-preparation)
4. [Feature Engineering](#feature-engineering)
5. [Train-Test Split](#train-test-split)
6. [Visualization](#visualization)
7. [Support Vector Classification (SVC)](#support-vector-classification-svc)
8. [Results](#results)

## Introduction
This documentation provides a detailed guide to perform classification on a synthetic dataset using Support Vector Classification (SVC). The process involves generating the dataset, preparing the data, engineering features, splitting the data into training and testing sets, visualizing the data, training the SVC model with different kernels, and evaluating the model's performance.

**Author: Nihar Raju**

## Dataset Generation
The dataset is generated using mathematical functions to create two concentric circles.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Generate data for the first circle
x = np.linspace(-5.0, 5.0, 100)
y = np.sqrt(10**2 - x**2)
y = np.hstack([y, -y])
x = np.hstack([x, -x])

# Generate data for the second circle
x1 = np.linspace(-5.0, 5.0, 100)
y1 = np.sqrt(5**2 - x1**2)
y1 = np.hstack([y1, -y1])
x1 = np.hstack([x1, -x1])

# Plot the data
plt.scatter(y, x)
plt.scatter(y1, x1)
```

## Data Preparation
The generated data is prepared into a DataFrame, and labels are assigned.

```python
# Create DataFrame for the first circle
df1 = pd.DataFrame(np.vstack([y, x]).T, columns=['X1', 'X2'])
df1['Y'] = 0

# Create DataFrame for the second circle
df2 = pd.DataFrame(np.vstack([y1, x1]).T, columns=['X1', 'X2'])
df2['Y'] = 1

# Concatenate the DataFrames
df = pd.concat([df1, df2], ignore_index=True)

# Display the first and last few rows of the DataFrame
df.head(5)
df.tail()
```

## Feature Engineering
Additional features are engineered to improve the model's performance.

```python
# Engineer new features
df['X1_Square'] = df['X1'] ** 2
df['X2_Square'] = df['X2'] ** 2
df['X1*X2'] = df['X1'] * df['X2']

# Display the first few rows of the DataFrame with new features
df.head()
```

## Train-Test Split
The dataset is split into training and testing sets.

```python
from sklearn.model_selection import train_test_split

# Define features and target
X = df[['X1', 'X2', 'X1_Square', 'X2_Square', 'X1*X2']]
y = df['Y']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Display the first few rows of the training set
X_train.head()
```

## Visualization
The data is visualized using 3D scatter plots to understand the distribution of features.

```python
import plotly.express as px

# Create a 3D scatter plot for the original features
fig = px.scatter_3d(df, x='X1', y='X2', z='X1*X2', color='Y')
fig.show()

# Create a 3D scatter plot for the engineered features
fig = px.scatter_3d(df, x='X1_Square', y='X2_Square', z='X1*X2', color='Y')
fig.show()
```

## Support Vector Classification (SVC)
The SVC model is trained with different kernels, and the performance is evaluated.

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Train and evaluate the SVC model with a linear kernel
classifier = SVC(kernel="linear")
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Accuracy (Linear Kernel):", accuracy_score(y_test, y_pred))

# Train and evaluate the SVC model with a polynomial kernel
classifier = SVC(kernel="poly")
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Accuracy (Polynomial Kernel):", accuracy_score(y_test, y_pred))

# Train and evaluate the SVC model with an RBF kernel
classifier = SVC(kernel="rbf")
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Accuracy (RBF Kernel):", accuracy_score(y_test, y_pred))

# Train and evaluate the SVC model with a sigmoid kernel
classifier = SVC(kernel="sigmoid")
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Accuracy (Sigmoid Kernel):", accuracy_score(y_test, y_pred))
```

## Results
The results include the accuracy scores for the SVC model with different kernels.

### Accuracy Scores
- **Linear Kernel**: [Displayed in the output]
- **Polynomial Kernel**: [Displayed in the output]
- **RBF Kernel**: [Displayed in the output]
- **Sigmoid Kernel**: [Displayed in the output]

### Detailed Steps

#### Step 1: Dataset Generation
- **Generating Data**: Data for two concentric circles is generated using mathematical functions.
- **Plotting Data**: The generated data is plotted using scatter plots.

#### Step 2: Data Preparation
- **Creating DataFrames**: DataFrames for the two circles are created and concatenated.
- **Assigning Labels**: Labels are assigned to the data points.

#### Step 3: Feature Engineering
- **Engineering Features**: Additional features (squared terms and interaction term) are engineered to improve the model's performance.

#### Step 4: Train-Test Split
- **Splitting the Dataset**: The dataset is split into training and testing sets using an 75-25 split.

#### Step 5: Visualization
- **3D Scatter Plots**: The data is visualized using 3D scatter plots to understand the distribution of features.

#### Step 6: Support Vector Classification (SVC)
- **Training the Model**: The SVC model is trained with different kernels (linear, polynomial, RBF, sigmoid).
- **Evaluating the Model**: The accuracy scores for the different kernels are calculated and displayed.

This documentation provides a comprehensive guide to performing classification on a synthetic dataset using SVC and evaluating the model's performance with different kernels.

**Author: Nihar Raju**
