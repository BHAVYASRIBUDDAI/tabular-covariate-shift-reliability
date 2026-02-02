//Prediction Probabilities

# Logistic Regression probabilities
lr_probs_test = lr.predict_proba(X_test)[:, 1]

# Random Forest probabilities
rf_probs_test = rf.predict_proba(X_test)[:, 1]

//Reliability Diagram (Confidence vs Accuracy)
import numpy as np
import matplotlib.pyplot as plt

def reliability_diagram(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    bin_acc = []
    bin_conf = []

    for i in range(n_bins):
        idx = bin_ids == i
        if np.sum(idx) > 0:
            bin_acc.append(np.mean(y_true[idx]))
            bin_conf.append(np.mean(y_prob[idx]))

    plt.plot(bin_conf, bin_acc, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


##Plot for shifted test data
reliability_diagram(y_test, lr_probs_test)
reliability_diagram(y_test, rf_probs_test)



##Expected Calibration Error (ECE)
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    ece = 0
    for i in range(n_bins):
        idx = bin_ids == i
        if np.sum(idx) > 0:
            acc = np.mean(y_true[idx])
            conf = np.mean(y_prob[idx])
            ece += np.abs(acc - conf) * np.sum(idx) / len(y_true)

    return ece


##Compute ECE
ece_lr = expected_calibration_error(y_test, lr_probs_test)
ece_rf = expected_calibration_error(y_test, rf_probs_test)

print("Logistic Regression ECE:", ece_lr)
print("Random Forest ECE:", ece_rf)





