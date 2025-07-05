# IRIS FLOWER CLASSIFICATION

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 2. Load the Dataset
df = pd.read_csv(r'c:\Users\safik\Desktop\jabs 2\archive (1)\IRIS.csv')

# 3. Explore the Data
print("First 5 rows:\n", df.head())
print("\nInfo:")
print(df.info())
print("\nDescribe:\n", df.describe())
print("\nSpecies value counts:\n", df['species'].value_counts())

# 4. Visualize the Data
plt.figure(figsize=(8,6))
sns.pairplot(df, hue='species')
plt.suptitle('Pairplot of Iris Features by Species', y=1.02)
plt.show()

# 5. Preprocess the Data
# Encode species labels
le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['species'])

# Features and target
X = df.drop(['species', 'species_encoded'], axis=1)
y = df['species_encoded']

# Feature scaling (optional, but good practice)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 7. Build and Train the Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 8. Evaluate the Model
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 9. Visualize Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 10. Predict on New Data Example
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example: sepal_length, sepal_width, petal_length, petal_width
sample_scaled = scaler.transform(sample)
predicted_class = le.inverse_transform(model.predict(sample_scaled))
print("\nPredicted species for sample:", predicted_class[0])