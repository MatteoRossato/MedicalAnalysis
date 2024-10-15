# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (replace 'medical_data.csv' with your dataset)
# The dataset should have features and a target variable 'has_condition'
# Example features could be age, blood pressure, cholesterol level, etc.
data = pd.read_csv('medical_data.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Preprocess the data
# Assume the target variable is 'has_condition' and the features are all other columns
X = data.drop('has_condition', axis=1)  # Features
y = data['has_condition']                 # Target variable

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy:.2f}")

# Print a classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

