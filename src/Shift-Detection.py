Create a Shift-Detection Dataset
//import numpy as np

# Combine features
X_shift = np.vstack([X_train, X_test])

# Labels: 0 = train, 1 = test
y_shift = np.hstack([
    np.zeros(X_train.shape[0]),
    np.ones(X_test.shape[0])
])

print(X_shift.shape, y_shift.shape)

//Train a Shift Detector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Split for shift detection
X_sd_train, X_sd_test, y_sd_train, y_sd_test = train_test_split(
    X_shift, y_shift, test_size=0.3, random_state=42, stratify=y_shift
)

# Train detector
shift_detector = LogisticRegression(max_iter=1000)
shift_detector.fit(X_sd_train, y_sd_train)

# Predict probabilities
y_sd_pred = shift_detector.predict_proba(X_sd_test)[:, 1]

# Evaluate
auc = roc_auc_score(y_sd_test, y_sd_pred)
print("Covariate Shift Detection AUC:", auc)




