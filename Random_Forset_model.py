# ====== Import Libraries ======
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# ====== Load the Reduced Dataset ======
file_path = r"D:\cyber security\major project\real time\Road_Accident_Data_Reduced.csv"
df = pd.read_csv(file_path)

# ====== Prepare Features and Target ======
y = df['Accident_Severity_Num']  # Target

# ====== Load the Scaler ======
scaler_path = r"D:\cyber security\major project\real time\new_scaler_logistic.pkl"
scaler = joblib.load(scaler_path)

# ✅ Match the Feature Names
# Extract the original feature names used during scaler fitting
original_features = scaler.feature_names_in_

# Select only the matching columns
X = df[original_features]

# ✅ Scale the matched features
X_scaled = scaler.transform(X)

# ✅ Split the Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ====== Train the Random Forest Model ======
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# ✅ Save the Model
output_path = r"D:\cyber security\major project\real time\random_forest_model.pkl"
joblib.dump(random_forest_model, output_path)
print(f"\n✅ Random Forest model saved as '{output_path}'!")

# ====== Make Predictions ======
y_pred_rf = random_forest_model.predict(X_test)
y_prob_rf = random_forest_model.predict_proba(X_test)[:, 1]  # For ROC Curve

# ====== Evaluate Model ======
accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf, average='weighted')
recall = recall_score(y_test, y_pred_rf, average='weighted')
f1 = f1_score(y_test, y_pred_rf, average='weighted')
roc_auc = roc_auc_score(y_test, random_forest_model.predict_proba(X_test), multi_class='ovr')

# ====== Print Metrics ======
print("\n🔥 Random Forest Model Evaluation:")
print(f"➡️ Accuracy: {accuracy:.4f}")
print(f"➡️ Precision: {precision:.4f}")
print(f"➡️ Recall: {recall:.4f}")
print(f"➡️ F1 Score: {f1:.4f}")
print(f"➡️ ROC-AUC: {roc_auc:.4f}")

# ====== Graphical Visualizations ======

# 1️⃣ Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 2️⃣ ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_rf, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.show()

# 3️⃣ Feature Importance Plot
feature_importances = random_forest_model.feature_importances_
feature_names = original_features

# Sort feature importance for better visualization
sorted_indices = feature_importances.argsort()

plt.figure(figsize=(12, 8))
plt.barh(range(len(feature_importances)), feature_importances[sorted_indices], align='center')
plt.yticks(range(len(feature_importances)), [feature_names[i] for i in sorted_indices])
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.title('Feature Importance - Random Forest')
plt.show()
