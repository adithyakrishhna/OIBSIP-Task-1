# Oasis Infybyte Data Science Internship
## Iris Flower Classification Machine Learning Project

This project is part of the Oasis Infybyte Data Science Internship and focuses on building a machine learning model to classify Iris flowers into different species based on their features.

### Steps to Build the Machine Learning Model

1. **Importing the Dataset**
   - Load the Iris dataset from a CSV file.
   - Explore the dataset by displaying the first 10 rows and checking for any missing values.

2. **Visualizing the Dataset**
   - Visualize the data using box plots to understand the distribution of each feature.
   - Create a heatmap to visualize the correlation between features.

3. **Data Preparation**
   - Drop the 'Id' column from the dataset as it is not needed for classification.
   - Map the species names ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica') to numerical values (1, 2, 3) for classification.
   - Split the dataset into feature variables (X) and the target variable (y).

4. **Training the Algorithm**
   - Split the data into training and testing sets using `train_test_split`.
   - Train a Linear Regression model on the training data.
   - Evaluate the model's performance using the `model.score` method.

5. **Making Predictions**
   - Use the trained model to make predictions on the test data.

6. **Model Evaluation**
   - Calculate the mean squared error to evaluate the model's accuracy.

### Code Example

Here is an example of the code used in this project:

```python
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing Dataset
df = pd.read_csv('Iris.csv')

# Visualizing the Dataset
plt.boxplot(df['SepalLengthCm'])
# ... (box plots for other features)

sns.heatmap(df.corr())

# Data Preparation
df.drop('Id', axis=1, inplace=True)
sp = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
df['Species'] = [sp[i] for i in df['Species']]

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

# Training the Algorithm
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Making Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("Mean squared error: %.2f" % np.mean((y_pred - y_test) ** 2))
