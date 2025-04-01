# ====== Import Libraries ======
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

# ✅ Load Dataset
file_path = r"D:\cyber security\major project\real time\Road_Accident_Data_Reduced.csv"

df = pd.read_csv(file_path)

# ✅ Separate features and target
y = df['Accident_Severity_Num']  # Target variable
X = df.drop('Accident_Severity_Num', axis=1)  # Features only

# ✅ Select only numerical features
X_numerical = X.select_dtypes(include=['number', 'float64', 'int64'])

# ✅ Load the new scaler
new_scaler_filepath = r"D:\cyber security\major project\real time\new_scaler_logistic.pkl"
scaler = joblib.load(new_scaler_filepath)

# ✅ Scale the numerical features using the new scaler
X_scaled = scaler.transform(X_numerical)

# ✅ Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Train Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)

# ✅ Make predictions
y_pred_logistic = logistic_model.predict(X_test)

# ✅ Model evaluation
logistic_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_logistic),
    "Precision": precision_score(y_test, y_pred_logistic, average='weighted'),
    "Recall": recall_score(y_test, y_pred_logistic, average='weighted'),
    "F1 Score": f1_score(y_test, y_pred_logistic, average='weighted'),
    "ROC-AUC": roc_auc_score(y_test, logistic_model.predict_proba(X_test), multi_class='ovr')
}

# ✅ Display metrics
print("\n🔥 Logistic Regression Model Evaluation:")
for metric, value in logistic_metrics.items():
    print(f"➡️ {metric}: {value:.4f}")

# ✅ Save the Logistic Regression model
logistic_model_filepath = r"D:\cyber security\major project\real time\logistic_model.pkl"
joblib.dump(logistic_model, logistic_model_filepath)
print(f"\n✅ Logistic Regression model saved as '{logistic_model_filepath}'!")

# ====== Graph Visualizations ======

# ✅ Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay.from_estimator(logistic_model, X_test, y_test, ax=ax, cmap='Blues')
plt.title('🔥 Confusion Matrix - Logistic Regression')
plt.show()

# ✅ Plot ROC Curve
y_prob = logistic_model.predict_proba(X_test)

plt.figure(figsize=(10, 6))

# Plot ROC for each class
for i in range(y_prob.shape[1]):
    fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('🔥 ROC Curve - Logistic Regression')
plt.legend()
plt.show()

# ✅ Bar Graph for Model Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [logistic_metrics[metric] for metric in metrics]

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['#4caf50', '#2196f3', '#ff9800', '#f44336'])
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('🔥 Model Evaluation Metrics - Logistic Regression')

# Add value labels on top of each bar
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=12)

plt.ylim(0, 1.1)
plt.show()
