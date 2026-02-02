//import libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Logistic Regression
lr = LogisticRegression(max_iter=1000)  # Increase iterations to converge
lr.fit(X_train, y_train)

# Predictions
y_train_pred_lr = lr.predict(X_train)
y_test_pred_lr = lr.predict(X_test)

# Evaluation
print("Logistic Regression Performance")
print("Train Accuracy:", accuracy_score(y_train, y_train_pred_lr))
print("Train F1 Score:", f1_score(y_train, y_train_pred_lr))
print("Shifted Test Accuracy:", accuracy_score(y_test, y_test_pred_lr))
print("Shifted Test F1 Score:", f1_score(y_test, y_test_pred_lr))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_train_pred_rf = rf.predict(X_train)
y_test_pred_rf = rf.predict(X_test)

# Evaluation
print("\nRandom Forest Performance")
print("Train Accuracy:", accuracy_score(y_train, y_train_pred_rf))
print("Train F1 Score:", f1_score(y_train, y_train_pred_rf))
print("Shifted Test Accuracy:", accuracy_score(y_test, y_test_pred_rf))
print("Shifted Test F1 Score:", f1_score(y_test, y_test_pred_rf))

//confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_test_pred_lr)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Logistic Regression Confusion Matrix on Shifted Test Set")
plt.show()
