# Initialize Logistic Regression classifier
lr = LogisticRegression(max_iter=1000)

# Use One-vs-Rest classifier
ovr_lr = OneVsRestClassifier(lr)

# Train the model
ovr_lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lr = ovr_lr.predict(X_test)

# Evaluate performance
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=mlb.classes_))
