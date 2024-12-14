import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


file_path = "online_payments_fraud.csv" 
data = pd.read_csv(file_path)


print(data.head())
print(data.info())
print(data.describe())


sns.countplot(x="isFraud", data=data)
plt.title("Distribution of Fraudulent Transactions")
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


data = data.drop(columns=["nameOrig", "nameDest"])


data = data.dropna()


X = data.drop(columns=["isFraud"])
y = data["isFraud"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


example_data = X_test.iloc[0].values.reshape(1, -1)
example_prediction = model.predict(example_data)
print(f"Example Prediction: {'Fraud' if example_prediction[0] == 1 else 'Not Fraud'}")


import joblib
joblib.dump(model, "fraud_detection_model.pkl")
print("Model saved as 'fraud_detection_model.pkl'.")


with open("README.md", "w") as f:
    f.write("# Online Payments Fraud Detection\n\n")
    f.write("## Project Overview\n")
    f.write("This project uses a Decision Tree Classifier to predict fraudulent online payment transactions.\n\n")
    f.write("## Dataset Features\n")
    f.write("- `step`: Represents a unit of time where 1 step equals 1 hour.\n")
    f.write("- `type`: Type of online transaction.\n")
    f.write("- `amount`: The amount of the transaction.\n")
    f.write("- `oldbalanceOrg`: Initial balance of the origin account.\n")
    f.write("- `newbalanceOrig`: New balance of the origin account.\n")
    f.write("- `oldbalanceDest`: Initial balance of the destination account.\n")
    f.write("- `newbalanceDest`: New balance of the destination account.\n")
    f.write("- `isFraud`: Whether the transaction is fraudulent (1) or not (0).\n\n")
    f.write("## Results\n")
    f.write(f"- Model Accuracy: {accuracy:.2f}\n\n")
    f.write("## Example Prediction\n")
    f.write(f"An example prediction result: {'Fraud' if example_prediction[0] == 1 else 'Not Fraud'}\n\n")
    f.write("## Libraries Used\n")
    f.write("- Pandas\n- NumPy\n- Matplotlib\n- Seaborn\n- Scikit-learn\n- Joblib\n")

print("README.md file created.")
