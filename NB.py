# Initialize Naive Bayes classifier
nb = MultinomialNB()

# Use One-vs-Rest classifier
ovr_nb = OneVsRestClassifier(nb)

# Train the model
ovr_nb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ovr_nb.predict(X_test)

# Evaluate performance
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))
