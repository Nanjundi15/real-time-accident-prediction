# ====== Import Libraries ======
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ====== Load the Dataset ======
file_path = r"D:\cyber security\major project\real time\Road_Accident_Data_Reduced.csv"  # Your PyCharm local path
data = pd.read_csv(file_path)
print("✅ File loaded successfully!")

# ====== Convert Object Columns to Categorical ======
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].astype('category')

print("✅ Categorical encoding applied!")

# ====== Prepare Features and Target ======
X = data.drop('Accident_Severity_Num', axis=1)  # Features
y = data['Accident_Severity_Num']               # Target

# ====== Split the Data ======
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== Train the XGBoost Model with Categorical Support ======
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    enable_categorical=True,
    random_state=42
)

model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ====== Make Predictions ======
y_pred = model.predict(X_test)

# ====== Evaluate the Model ======
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# ====== Display the Results ======
print("\n✅ Model Accuracy:", accuracy)
print("\n📊 Confusion Matrix:\n", conf_matrix)
print("\n🛠️ Classification Report:\n", report)
