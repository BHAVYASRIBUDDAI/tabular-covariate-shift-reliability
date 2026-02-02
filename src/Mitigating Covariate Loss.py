##Mitigating Covariate Loss
# We reuse the shift detector idea, but now only train it to
# distinguish train vs test distributions

from sklearn.linear_model import LogisticRegression

# Create labels again: 0 = train, 1 = test
X_shift_full = np.vstack([X_train, X_test])
y_shift_full = np.hstack([
    np.zeros(len(X_train)),
    np.ones(len(X_test))
])

# Train a simple classifier to detect shift
shift_model = LogisticRegression(max_iter=1000)
shift_model.fit(X_shift_full, y_shift_full)

# Probability that a training sample looks like test data
train_shift_probs = shift_model.predict_proba(X_train)[:, 1]

# Convert probabilities into importance weights
# Adding small epsilon to avoid division issues
epsilon = 1e-6
sample_weights = train_shift_probs / (1 - train_shift_probs + epsilon)


##Retrain Logistic Regression with Weights
# Weighted Logistic Regression
lr_weighted = LogisticRegression(max_iter=1000)

lr_weighted.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)

# Predictions on shifted test set
y_test_pred_lr_w = lr_weighted.predict(X_test)
y_test_prob_lr_w = lr_weighted.predict_proba(X_test)[:, 1]


##Performance evaluation
from sklearn.metrics import accuracy_score, f1_score

print("Weighted Logistic Regression")
print("Shifted Test Accuracy:", accuracy_score(y_test, y_test_pred_lr_w))
print("Shifted Test F1:", f1_score(y_test, y_test_pred_lr_w))


##Recheck calibaration
# Reliability diagram
reliability_diagram(y_test, y_test_prob_lr_w)

# ECE
ece_lr_weighted = expected_calibration_error(y_test, y_test_prob_lr_w)
print("Weighted LR ECE:", ece_lr_weighted)

