# train_model.py - COMPLETELY CORRECTED VERSION (NO XGBOOST)
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, RidgeCV

print("="*60)
print("HOUSE PRICE PREDICTION - MODEL TRAINING")
print("="*60)

# Step 1: Load data
print("\nStep 1: Loading data...")
try:
    df = pd.read_csv('House Price Prediction Dataset.csv')
    print(f"    Data loaded successfully!")
    print(f"    Dataset shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")
except FileNotFoundError:
    print("   ERROR: CSV file not found!")
    print("   Please make sure 'House Price Prediction Dataset.csv' is in the same folder")
    exit()

# Step 2: Feature Engineering
print("\n🔧 Step 2: Feature Engineering...")

# Create age feature
current_year = 2024
df['Age'] = current_year - df['YearBuilt']
print(f"   Created 'Age' feature (current year: {current_year})")

# Create total rooms feature
df['TotalRooms'] = df['Bedrooms'] + df['Bathrooms']
print(f" Created 'TotalRooms' feature")

# Create area per room feature
df['AreaPerRoom'] = df['Area'] / (df['TotalRooms'] + 1)
print(f"   Created 'AreaPerRoom' feature")

# Step 3: Encode categorical variables
print("\nStep 3: Encoding categorical variables...")

label_encoders = {}
categorical_cols = ['Location', 'Condition', 'Garage']

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_Encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"   Encoded '{col}' -> '{col}_Encoded'")
    print(f"      Unique values: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Step 4: Prepare features
print("\nStep 4: Preparing features...")

feature_cols = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt', 
                'Location_Encoded', 'Condition_Encoded', 'Garage_Encoded',
                'Age', 'TotalRooms', 'AreaPerRoom']

X = df[feature_cols]
y = np.log1p(df['Price'])  # Log transform target

print(f"    Features selected: {len(feature_cols)} features")
print(f"    Feature names: {feature_cols}")
print(f"    Target: Price (log-transformed)")

# Step 5: Split data
print("\n Step 5: Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Training set: {len(X_train)} samples (80%)")
print(f"   Test set: {len(X_test)} samples (20%)")

# Step 6: Scale features
print("\n Step 6: Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"   Features scaled using StandardScaler")

# Step 7: Train models
print("\n Step 7: Training models...")

# Train individual models (NO XGBOOST)
print("\n   Training individual models:")

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
print(f"  Random Forest trained")

gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train_scaled, y_train)
print(f"   Gradient Boosting trained")

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
print(f"   Ridge trained")

lasso = Lasso(alpha=0.001, max_iter=10000)
lasso.fit(X_train_scaled, y_train)
print(f"   Lasso trained")

# Step 8: Create stacking ensemble
print("\n Step 8: Creating Stacking Ensemble...")

base_models = [
    ('ridge', ridge),
    ('lasso', lasso),
    ('gb', gb),
    ('rf', rf)
]

stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
    cv=5,
    n_jobs=-1
)

print("   Training stacking model (this may take a minute)...")
stacking_model.fit(X_train_scaled, y_train)
print(f"   Stacking Ensemble trained successfully!")

# Step 9: Quick evaluation
print("\nStep 9: Quick evaluation on test set...")

# Predict on test set
y_pred_log = stacking_model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

# Calculate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mae = mean_absolute_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

print(f"   Test Set Performance:")
print(f"      RMSE: ${rmse:,.2f}")
print(f"      MAE: ${mae:,.2f}")
print(f"      R² Score: {r2:.4f}")

# Step 10: Save model artifacts
print("\nStep 10: Saving model artifacts...")

model_artifacts = {
    'model': stacking_model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_names': feature_cols,
    'model_performance': {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
}

joblib.dump(model_artifacts, 'house_price_model.pkl')
print(f" Model saved as 'house_price_model.pkl'")

# Step 11: Test prediction example
print("\n🔍 Step 11: Testing with sample prediction...")

# Take first test sample
sample = X_test.iloc[0:1]
sample_scaled = scaler.transform(sample)
sample_pred_log = stacking_model.predict(sample_scaled)[0]
sample_pred = np.expm1(sample_pred_log)
sample_actual = np.expm1(y_test.iloc[0])

print(f"\n   Sample House Features:")
for col in feature_cols:
    print(f"      {col}: {sample.iloc[0][col]}")
print(f"\n   Prediction Result:")
print(f"      Actual Price: ${sample_actual:,.2f}")
print(f"      Predicted Price: ${sample_pred:,.2f}")
print(f"      Difference: ${abs(sample_actual - sample_pred):,.2f}")
print(f"      Accuracy: {100 - (abs(sample_actual - sample_pred)/sample_actual*100):.1f}%")

print("\n" + "="*60)
print("TRAINING COMPLETE! Model is ready to use.")
print("="*60)
print("\nNext steps:")
print("1. Run 'py app.py' to start the web application")
print("2. Open http://localhost:5000 in your browser")
print("3. Start predicting house prices!")