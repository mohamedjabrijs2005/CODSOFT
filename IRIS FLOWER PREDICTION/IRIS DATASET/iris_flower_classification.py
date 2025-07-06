
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# --- IRIS FLOWER CLASSIFICATION ---
# The Iris flower dataset consists of three species: setosa, versicolor, and virginica.
# These species can be distinguished based on their measurements (sepal length, sepal width, petal length, petal width).
# Objective: Train a machine learning model that can learn from these measurements and accurately classify the Iris flowers into their respective species.
# This script loads the data, visualizes it, trains a classifier, and evaluates its performance.

# Load the dataset
iris = pd.read_csv('IRIS.csv')


# --- Data Overview ---
print("First 5 rows of the dataset:")
print(iris.head())
print("\nDataset info:")
print(iris.info())
print("\nClass distribution:")
print(iris['species'].value_counts())

# --- Visualizations ---
plt.figure(figsize=(8, 6))
sns.countplot(x='species', data=iris)
plt.title('Count of Each Iris Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()

# Pairplot for all features colored by species
sns.pairplot(iris, hue='species', diag_kind='hist')
plt.suptitle('Pairplot of Features by Species', y=1.02)
plt.show()

# Boxplot for each feature by species
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
plt.figure(figsize=(12, 8))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='species', y=feature, data=iris)
    plt.title(f'{feature.capitalize()} by Species')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(iris[features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# Features and target
X = iris.drop('species', axis=1)
y = iris['species']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)


# Evaluate
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)
cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix Visualization
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# --- Model Results ---
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)
