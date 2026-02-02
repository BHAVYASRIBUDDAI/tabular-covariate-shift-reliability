from google.colab import files

uploaded = files.upload()

import pandas as pd

columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

data = pd.read_csv('adult.data', header=None, names=columns, sep=',\s*', engine='python')
print(data.head())

# Replace '?' with NaN
data.replace('?', pd.NA, inplace=True)

# Drop rows with missing values
data.dropna(inplace=True)

print(f"Dataset size after dropping missing: {data.shape}")

categorical_features = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]

numerical_features = [
    'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'
]

target = 'income'

import matplotlib.pyplot as plt

for col in numerical_features:
    plt.hist(data[col], bins=30)
    plt.title(col)
    plt.show()

# Training: age 25-50
train_data = data[data['age'] <= 50]

# Shifted test: age 51+
test_data = data[data['age'] > 50]

print(f"Train size: {train_data.shape}, Test size (shifted): {test_data.shape}")

from sklearn.preprocessing import OneHotEncoder, StandardScaler

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_train_cat = encoder.fit_transform(train_data[categorical_features])
X_test_cat = encoder.transform(test_data[categorical_features])

# Standardize numerical features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(train_data[numerical_features])
X_test_num = scaler.transform(test_data[numerical_features])

# Combine features
import numpy as np

X_train = np.hstack([X_train_num, X_train_cat])
X_test = np.hstack([X_test_num, X_test_cat])

# Encode target
y_train = (train_data[target] == '>50K').astype(int).values
y_test = (test_data[target] == '>50K').astype(int).values

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
