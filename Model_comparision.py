# ====== Import Libraries ======
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ====== Load the Models and Scaler ======
model_paths = {
    "Logistic Regression": r"D:\cyber security\major project\real time\logistic_model.pkl",
    "Decision Tree": r"D:\cyber security\major project\real time\decision_tree_model.pkl",
    "Random Forest": r"D:\cyber security\major project\real time\random_forest_model.pkl",
    "XGBoost": r"D:\cyber security\major project\real time\xgboost_model_fixed.pkl"  # Retrained XGBoost model
}

scaler_path = r"D:\cyber security\major project\real time\new_scaler_logistic.pkl"
scaler = joblib.load(scaler_path)

# ====== Load the Dataset ======
file_path = r"D:\cyber security\major project\real time\Road_Accident_Data_Reduced.csv"
df = pd.read_csv(file_path)

# ====== Prepare Features and Target ======
y = df['Accident_Severity_Num']

# Match the Feature Names (features used during scaler fitting)
original_features = scaler.feature_names_in_
X = df[original_features]

# Scale the features
X_scaled = scaler.transform(X)

# Convert scaled data to DataFrame with matching columns
X_scaled_df = pd.DataFrame(X_scaled, columns=original_features)

# ====== Evaluate All Models ======
results = {}

for model_name, model_path in model_paths.items():
    print(f"\n🔥 Evaluating: {model_name}")

    # Load the model
    model = joblib.load(model_path)

    # Use model.feature_names_in_ if available; otherwise, use original_features
    expected_features = getattr(model, "feature_names_in_", original_features)

    # Align the test features
    X_test_matched = X_scaled_df[expected_features]

    # Make predictions
    y_pred = model.predict(X_test_matched)

    # Calculate metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y, model.predict_proba(X_test_matched), multi_class='ovr')

    # Store results
    results[model_name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }

# ====== Improved Circular Bar Plot with Plotly ======
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
angles = np.linspace(0, 360, len(metrics) + 1)[:-1]  # Circular positioning

fig = go.Figure()

# Improved color palette
colors = ['#FF5733', '#33FF57', '#337BFF', '#FF33A6']

# Add each model with enhanced styling
for idx, (model_name, model_results) in enumerate(results.items()):
    values = [model_results[metric] for metric in metrics]

    # Close the loop by adding the first value again
    values.append(values[0])
    angles_with_loop = np.append(angles, angles[0])

    # Add polar trace
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=angles_with_loop,
        fill='toself',
        name=model_name,
        hoverinfo='text',
        text=[f'{metric}: {value:.2f}' for metric, value in zip(metrics, values[:-1])],
        line=dict(color=colors[idx], width=3, dash='solid'),  # Smoother line
        marker=dict(size=8, symbol='circle', opacity=0.9),  # Larger markers
        opacity=0.85
    ))

# Customize layout with enhanced aesthetics
fig.update_layout(
    title={
        'text': "Model Performance Comparison (Interactive Circular Plot)",
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 26, 'color': '#2c3e50'}
    },
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            showline=False,
            tickfont=dict(size=12, color='#333'),
            gridcolor='#ccc'
        ),
        angularaxis=dict(
            tickmode='array',
            tickvals=angles,
            ticktext=metrics,
            rotation=90,
            direction='clockwise',
            tickfont=dict(size=14, color='#2c3e50')
        )
    ),
    legend=dict(
        x=1,
        y=1,
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='rgba(0,0,0,0.1)',
        borderwidth=1,
        font=dict(size=13)
    ),
    margin=dict(t=80, b=60, l=60, r=60),
    template='plotly_dark'  # Dark theme for contrast
)

# Display the interactive plot
fig.show()
