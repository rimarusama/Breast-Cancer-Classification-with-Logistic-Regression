# Import necessary libraries
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Breast Cancer dataset from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# Create a DataFrame to hold the dataset with feature names as column names
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)

# Add a 'label' column to the DataFrame to hold target values
data_frame['label'] = breast_cancer_dataset.target

# Separate features (X) and target labels (Y)
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=None)

# Create a Logistic Regression model
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train, Y_train)

# Make predictions on the training data and calculate training accuracy
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

# Print the training accuracy
print('Accuracy on training data: ', training_data_accuracy)

# Make predictions on the test data and calculate test accuracy
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

# Print the test accuracy
print('Accuracy on test data: ', test_data_accuracy)

# Define input data as a tuple
input_data = (10.51, 23.09, 66.85, 334.2, 0.1015, 0.06797, 0.02495, 0.01875, 0.1695, 0.06556, 0.2868, 1.143, 2.289, 20.56, 0.01017, 0.01443, 0.01861, 0.0125, 0.03464, 0.001971, 10.93, 24.22, 70.1, 362.7, 0.1143, 0.08614, 0.04158, 0.03125, 0.2227, 0.06777)

# Convert input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the input data to be in the format expected by the model
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make a prediction using the trained model
prediction = model.predict(input_data_reshaped)

# Print the prediction (Positive or Negative)
if prediction == True:
    print('Positive')
else:
    print('Negative')
