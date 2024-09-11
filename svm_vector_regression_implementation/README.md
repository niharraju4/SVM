Certainly! Below is a detailed documentation for the provided code, including a table of contents and results.

# Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Independent and Dependent Features](#independent-and-dependent-features)
4. [Train-Test Split](#train-test-split)
5. [Label Encoding](#label-encoding)
6. [One-Hot Encoding](#one-hot-encoding)
7. [Support Vector Regression (SVR)](#support-vector-regression-svr)
8. [Hyperparameter Tuning using GridSearchCV](#hyperparameter-tuning-using-gridsearchcv)
9. [Results](#results)

## Introduction
This documentation provides a detailed guide to perform regression analysis on the Tips dataset using Support Vector Regression (SVR). The process involves loading the dataset, preprocessing the data, splitting the data into training and testing sets, encoding categorical variables, training the SVR model, and performing hyperparameter tuning using GridSearchCV.

**Author: Nihar Raju**

## Dataset
The Tips dataset is loaded using the Seaborn library, and basic information about the dataset is displayed.

```python
import seaborn as sns

# Load the Tips dataset
df = sns.load_dataset('tips')

# Display the first few rows of the dataset
df.head()

# Display information about the dataset
df.info()

# Display summary statistics of the dataset
df.describe()

# Display value counts for categorical variables
df['sex'].value_counts()
df['smoker'].value_counts()
df['day'].value_counts()
df['time'].value_counts()

# Display the column names
df.columns
```

## Independent and Dependent Features
The independent features (X) and the dependent feature (y) are defined.

```python
# Define independent and dependent features
X = df[['tip', 'sex', 'smoker', 'day', 'time', 'size']]
y = df['total_bill']
```

## Train-Test Split
The dataset is split into training and testing sets.

```python
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# Display the first few rows of the training set
X_train.head()
```

## Label Encoding
Categorical variables are encoded using LabelEncoder.

```python
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Initialize LabelEncoder objects
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()

# Encode categorical variables in the training set
X_train['sex'] = le1.fit_transform(X_train['sex'])
X_train['smoker'] = le2.fit_transform(X_train['smoker'])
X_train['time'] = le3.fit_transform(X_train['time'])

# Display the first few rows of the encoded training set
X_train.head()

# Encode categorical variables in the testing set
X_test['sex'] = le1.transform(X_test['sex'])
X_test['smoker'] = le2.transform(X_test['smoker'])
X_test['time'] = le3.transform(X_test['time'])

# Display the first few rows of the encoded testing set
X_test.head()
```

## One-Hot Encoding
One-hot encoding is applied to the 'day' column using ColumnTransformer.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import sys
import numpy

numpy.set_printoptions(threshold=sys.maxsize)

# Define ColumnTransformer for one-hot encoding
ct = ColumnTransformer(transformers=[('onehot', OneHotEncoder(drop='first'), [3])], remainder='passthrough')

# Apply one-hot encoding to the training and testing sets
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

# Display the transformed testing set
X_test
```

## Support Vector Regression (SVR)
The SVR model is trained on the training set and used to make predictions on the testing set.

```python
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error

# Initialize and train the SVR model
svr = SVR()
svr.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = svr.predict(X_test)

# Display the R-squared score and mean absolute error
print(r2_score(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
```

## Hyperparameter Tuning using GridSearchCV
Hyperparameter tuning is performed using GridSearchCV to find the best parameters for the SVR model.

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

# Initialize and fit the GridSearchCV object
grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)

# Display the best parameters
grid.best_params_

# Make predictions on the testing set using the best model
grid_prediction = grid.predict(X_test)

# Display the R-squared score and mean absolute error for the best model
print(r2_score(y_test, grid_prediction))
print(mean_absolute_error(y_test, grid_prediction))
```

## Results
The results include the R-squared score and mean absolute error for both the initial SVR model and the best model found through hyperparameter tuning.

### Initial SVR Model
- **R-squared Score**: [Displayed in the output]
- **Mean Absolute Error**: [Displayed in the output]

### Best SVR Model (after Hyperparameter Tuning)
- **Best Parameters**: [Displayed in the output]
- **R-squared Score**: [Displayed in the output]
- **Mean Absolute Error**: [Displayed in the output]

### Detailed Steps

#### Step 1: Dataset
- **Loading the Dataset**: The Tips dataset is loaded using the Seaborn library.
- **Displaying Information**: Basic information about the dataset, including the first few rows, summary statistics, and value counts for categorical variables, is displayed.

#### Step 2: Independent and Dependent Features
- **Defining Features**: The independent features (X) and the dependent feature (y) are defined.

#### Step 3: Train-Test Split
- **Splitting the Dataset**: The dataset is split into training and testing sets using a 75-25 split.

#### Step 4: Label Encoding
- **Encoding Categorical Variables**: Categorical variables ('sex', 'smoker', 'time') are encoded using LabelEncoder.

#### Step 5: One-Hot Encoding
- **Applying One-Hot Encoding**: One-hot encoding is applied to the 'day' column using ColumnTransformer.

#### Step 6: Support Vector Regression (SVR)
- **Training the Model**: The SVR model is trained on the training set.
- **Making Predictions**: Predictions are made on the testing set.
- **Evaluating the Model**: The R-squared score and mean absolute error are calculated and displayed.

#### Step 7: Hyperparameter Tuning using GridSearchCV
- **Defining the Parameter Grid**: The parameter grid for hyperparameter tuning is defined.
- **Initializing GridSearchCV**: The GridSearchCV object is initialized and fitted to the training data.
- **Finding the Best Parameters**: The best parameters for the SVR model are found.
- **Making Predictions**: Predictions are made on the testing set using the best model.
- **Evaluating the Best Model**: The R-squared score and mean absolute error for the best model are calculated and displayed.

This documentation provides a comprehensive guide to performing regression analysis on the Tips dataset using SVR and hyperparameter tuning.

**Author: Nihar Raju**