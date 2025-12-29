# ---------------------------------------------
# 1. Import Required Libraries
# ---------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Metrics and Selection
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE

# The Star of the Show: XGBoost
from xgboost import XGBClassifier


# ---------------------------------------------
# 2. Load Dataset
# ---------------------------------------------
df = pd.read_csv("/Users/rithiiks/Desktop/sdr/brithish_airways_job/customer_booking.csv", encoding="ISO-8859-1")


# ---------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------
# Weekend flight feature
df['is_weekend_flight'] = df['flight_day'].isin(['Sat', 'Sun']).astype(int)

# Count of extra services selected
df['extras_selected'] = (
    df['wants_extra_baggage'] +
    df['wants_preferred_seat'] +
    df['wants_in_flight_meals']
)

# Stay duration relative to flight time
df['stay_per_flight_hour'] = df['length_of_stay'] / df['flight_duration']

# Drop Route (As per your previous logic)
df.drop(columns=['route'], inplace=True)

# Encode Categorical Variables
df = pd.get_dummies(
    df,
    columns=['sales_channel', 'trip_type', 'booking_origin', 'flight_day'],
    drop_first=True
)


# ---------------------------------------------
# 4. Split Features and Target
# ---------------------------------------------
X = df.drop('booking_complete', axis=1)
y = df['booking_complete']


# ---------------------------------------------
# 5. Train-Test Split & SMOTE
# ---------------------------------------------
# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE
# We use sampling_strategy=0.5 to not "over-fake" the data
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Original Training Shape: {X_train.shape}")
print(f"Resampled Training Shape: {X_train_resampled.shape}")


# ---------------------------------------------
# 6. Train XGBoost Model (Updated for XGBoost 2.0+)
# ---------------------------------------------
# We define the model
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    gamma=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    # --- MOVED THESE PARAMETERS HERE ---
    eval_metric=['logloss', 'auc'],  # Define metrics here
    early_stopping_rounds=50         # Define early stopping here
)

# We fit the model
print("\nTraining XGBoost...")
xgb.fit(
    X_train_resampled, 
    y_train_resampled,
    eval_set=[(X_train_resampled, y_train_resampled), (X_test, y_test)],
    verbose=50  # Print progress every 50 trees
)

# ---------------------------------------------
# 6.5. OPTIMIZATION: Remove "Low Insight" Features
# ---------------------------------------------
from sklearn.feature_selection import SelectFromModel

print("\n--- Optimizing Based on Insights (Feature Selection) ---")

# 1. Use the trained XGBoost model to find the best features
# "threshold" determines the cutoff. 'median' means "keep the top 50% features"
# You can also use a specific number like threshold=0.01
selection = SelectFromModel(xgb, threshold='median', prefit=True)

# 2. Transform the Training and Test sets to keep ONLY those features
X_train_selected = selection.transform(X_train_resampled)
X_test_selected = selection.transform(X_test)

print(f"Original Feature Count: {X_train_resampled.shape[1]}")
print(f"Selected Feature Count: {X_train_selected.shape[1]}")

# 3. Retrain a new XGBoost model on just the "High Insight" features
xgb_optimized = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,             
    min_child_weight=5,      # Increased to reduce overfitting further
    gamma=1.0,               # Increased to reduce overfitting further
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric=['logloss', 'auc'],
    early_stopping_rounds=50
)

print("\nRetraining on optimized features...")
xgb_optimized.fit(
    X_train_selected, 
    y_train_resampled,
    eval_set=[(X_train_selected, y_train_resampled), (X_test_selected, y_test)],
    verbose=50
)

# 4. Check the New Performance
y_pred_opt = xgb_optimized.predict(X_test_selected)
y_prob_opt = xgb_optimized.predict_proba(X_test_selected)[:, 1]

print("\n--- Optimized Model Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred_opt))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_opt))


# ---------------------------------------------
# 8. Cross-Validation (Robust Check)
# ---------------------------------------------
print("\nRunning Cross-Validation...")

# Define a NEW simple model for CV without early stopping complications
# (Early stopping is great for training, but hard to use with simple cross_val_score)
xgb_cv = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    gamma=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='auc' 
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# We use the simple model here to avoid the "Must have validation dataset" error
cv_scores = cross_val_score(xgb_cv, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

print("Mean CV ROC-AUC:", cv_scores.mean())


# ---------------------------------------------
# 9. Feature Importance (XGBoost Style)
# ---------------------------------------------
plt.figure(figsize=(10, 6))
importances = xgb.feature_importances_
features = X.columns
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)

plt.barh(feat_df['Feature'], feat_df['Importance'])
plt.gca().invert_yaxis()
plt.title("Top 10 Features (XGBoost)")
plt.xlabel("Importance Score")
plt.show()


# ---------------------------------------------
# 10. Training Progress (Loss & AUC Curves)
# ---------------------------------------------
# XGBoost stores the history in .evals_result()
results = xgb.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(14, 6))

# Plot Log Loss
plt.subplot(1, 2, 1)
plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
plt.plot(x_axis, results['validation_1']['logloss'], label='Test')
plt.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.grid(True)

# Plot AUC
plt.subplot(1, 2, 2)
plt.plot(x_axis, results['validation_0']['auc'], label='Train')
plt.plot(x_axis, results['validation_1']['auc'], label='Test')
plt.legend()
plt.ylabel('AUC')
plt.title('XGBoost AUC Score')
plt.grid(True)

plt.show()


# ---------------------------------------------
# 11. Optimal Threshold Moving
# ---------------------------------------------
# 1. Calculate the probabilities (This fixes the NameError)
y_prob = xgb.predict_proba(X_test)[:, 1]

# 2. Calculate curve details
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# 3. Find optimal threshold
optimal_threshold = 0.30
y_pred_optimized = (y_prob >= optimal_threshold).astype(int)

print(f"\n--- Performance at Threshold {optimal_threshold} ---")
print(classification_report(y_test, y_pred_optimized))

# ---------------------------------------------
# 12. Model Explainability (SHAP) - The "Why"
# ---------------------------------------------
import shap

# 1. Initialize the SHAP explainer with your model
# We use the optimized model (xgb or xgb_optimized)
explainer = shap.TreeExplainer(xgb)

# 2. Calculate SHAP values for the Test set
# (We only take the first 1000 rows to make it fast)
shap_values = explainer.shap_values(X_test.iloc[:1000])

# 3. Plot Summary (Global View)
# This shows which features matter most across ALL customers
plt.figure(figsize=(10, 6))
plt.title("What drives customer bookings? (SHAP Summary)")
shap.summary_plot(shap_values, X_test.iloc[:1000], show=True)

# 4. Plot Individual Explanation (Local View)
# Let's look at the very first customer in the test set
# This answers: "Why did specific Customer #1 get this score?"
shap.initjs() # Required if running in Notebooks
print("\nExplaining the prediction for the first customer in Test Set:")

# Force plot for the first customer
# Note: In a script, we save this as an HTML file to open in a browser
force_plot = shap.force_plot(
    explainer.expected_value, 
    shap_values[0,:], 
    X_test.iloc[0,:], 
    matplotlib=True,
    show=False
)
plt.savefig('single_customer_explanation.png')
print("Saved single_customer_explanation.png")
