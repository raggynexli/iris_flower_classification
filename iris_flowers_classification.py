import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. DATASET ---
print("\n=== DATASET LOADED ===")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Save CSV for your folder
X.to_csv('iris.csv', index=False)
print("Saved dataset to 'iris.csv'")
print("\nFirst 5 rows:")
print(X.head())

# --- 2. TRAIN & TEST DATA ---
print("\n=== TRAINING & TESTING DATA ===")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data: {len(X_train)} rows")
print(f"Testing data:  {len(X_test)} rows")

# --- 3. TRAIN MODEL ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- 4. PERFORMANCE ---
print("\n=== MODEL PERFORMANCE ===")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# --- 5. PREDICTION ---
print("\n=== PREDICTION OUTPUT ===")
print("Enter the measurements for the new flower:")
sepal_l = float(input("Sepal Length: "))
sepal_w = float(input("Sepal Width: "))
petal_l = float(input("Petal Length: "))
petal_w = float(input("Petal Width: "))

new_flower = [[sepal_l, sepal_w, petal_l, petal_w]]
predicted_class = model.predict(new_flower)
print(f"\nInput: {new_flower[0]}")
print(f"Predicted Species: {iris.target_names[predicted_class[0]].upper()}")

# --- 6. GRAPH ---
print("\n=== GRAPH OUTPUT ===")
print("Opening the confusion matrix graph...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.show()