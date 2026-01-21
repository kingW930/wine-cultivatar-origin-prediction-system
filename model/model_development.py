"""
Wine Cultivar Origin Prediction Model Development
==================================================
This script loads the Wine dataset, preprocesses it, trains a Random Forest
classifier, evaluates it, and saves the model for use in the web application.

Features Used (6 selected):
- alcohol
- malic_acid
- flavanoids
- color_intensity
- hue
- proline

Target: cultivar (wine origin class: 0, 1, or 2)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os

# ============================================================================
# 1. LOAD THE WINE DATASET
# ============================================================================
print("="*60)
print("WINE CULTIVAR ORIGIN PREDICTION - MODEL DEVELOPMENT")
print("="*60)

# Load dataset from sklearn
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['cultivar'] = wine.target

print("\n[1] Dataset Loaded Successfully!")
print(f"    Total samples: {len(df)}")
print(f"    Total features: {len(wine.feature_names)}")
print(f"    Target classes: {list(np.unique(wine.target))}")

# Display feature names
print("\n    Available features:")
for i, name in enumerate(wine.feature_names):
    print(f"      {i+1}. {name}")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n[2] Data Preprocessing...")

# Check for missing values
missing_values = df.isnull().sum().sum()
print(f"    Missing values: {missing_values}")

if missing_values > 0:
    df = df.dropna()
    print(f"    Dropped rows with missing values. New size: {len(df)}")
else:
    print("    No missing values found - dataset is clean!")

# ============================================================================
# 3. FEATURE SELECTION (6 Features)
# ============================================================================
print("\n[3] Feature Selection...")

# Select 6 features from the allowed list
selected_features = [
    'alcohol',
    'malic_acid',
    'flavanoids',
    'color_intensity',
    'hue',
    'proline'
]

print(f"    Selected {len(selected_features)} features:")
for feat in selected_features:
    print(f"      - {feat}")

# Extract features and target
X = df[selected_features]
y = df['cultivar']

print(f"\n    Feature matrix shape: {X.shape}")
print(f"    Target vector shape: {y.shape}")

# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================
print("\n[4] Splitting Data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"    Training set size: {len(X_train)}")
print(f"    Test set size: {len(X_test)}")

# ============================================================================
# 5. FEATURE SCALING (Mandatory)
# ============================================================================
print("\n[5] Feature Scaling...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("    Applied StandardScaler to normalize feature ranges")
print(f"    Training data mean (after scaling): {X_train_scaled.mean():.6f}")
print(f"    Training data std (after scaling): {X_train_scaled.std():.6f}")

# ============================================================================
# 6. MODEL TRAINING - Random Forest Classifier
# ============================================================================
print("\n[6] Training Random Forest Classifier...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("    Model trained successfully!")

# ============================================================================
# 7. MODEL EVALUATION
# ============================================================================
print("\n[7] Model Evaluation...")

# Predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n    PERFORMANCE METRICS:")
print(f"    {'='*40}")
print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"    Precision: {precision:.4f}")
print(f"    Recall:    {recall:.4f}")
print(f"    F1-Score:  {f1:.4f}")

print(f"\n    CLASSIFICATION REPORT:")
print(f"    {'='*40}")
print(classification_report(y_test, y_pred, target_names=['Cultivar 1', 'Cultivar 2', 'Cultivar 3']))

# Feature importance
print(f"\n    FEATURE IMPORTANCE:")
print(f"    {'='*40}")
importance = model.feature_importances_
for feat, imp in sorted(zip(selected_features, importance), key=lambda x: x[1], reverse=True):
    print(f"    {feat:25s}: {imp:.4f}")

# ============================================================================
# 8. SAVE MODEL AND SCALER
# ============================================================================
print("\n[8] Saving Model and Scaler...")

# Create model directory if it doesn't exist
model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, 'wine_cultivar_model.pkl')

# Save model and scaler together
model_data = {
    'model': model,
    'scaler': scaler,
    'features': selected_features
}

joblib.dump(model_data, model_path)
print(f"    Model saved to: {model_path}")

# ============================================================================
# 9. VERIFICATION - Test Loading
# ============================================================================
print("\n[9] Verification - Loading saved model...")

loaded_data = joblib.load(model_path)
loaded_model = loaded_data['model']
loaded_scaler = loaded_data['scaler']
loaded_features = loaded_data['features']

# Test prediction
test_sample = X_test.iloc[0:1]
test_sample_scaled = loaded_scaler.transform(test_sample)
prediction = loaded_model.predict(test_sample_scaled)

print(f"    Model loaded successfully!")
print(f"    Test prediction: Cultivar {prediction[0] + 1}")
print(f"    Actual value: Cultivar {y_test.iloc[0] + 1}")

print("\n" + "="*60)
print("MODEL DEVELOPMENT COMPLETE!")
print("="*60)
print(f"\nModel file: {model_path}")
print(f"Algorithm: Random Forest Classifier")
print(f"Features: {selected_features}")
print(f"Accuracy: {accuracy*100:.2f}%")
