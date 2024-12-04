# Initialize XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the model
xgb.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb.predict(X_test)

# Evaluate performance
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
