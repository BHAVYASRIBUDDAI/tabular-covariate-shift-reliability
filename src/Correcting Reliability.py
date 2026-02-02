#Split a Small “Validation-under-Shift” Set
from sklearn.model_selection import train_test_split

# Split test set into calibration + evaluation
X_calib, X_eval, y_calib, y_eval = train_test_split(
    X_test, y_test, test_size=0.7, random_state=42
)

#Learn Temperature
import numpy as np
from scipy.optimize import minimize

def temperature_scale(probs, T):
    return 1 / (1 + np.exp(-np.log(probs / (1 - probs)) / T))

def calibration_loss(T, probs, labels):
    scaled_probs = temperature_scale(probs, T)
    return np.mean((scaled_probs - labels) ** 2)

# Optimize temperature on shifted validation data
opt_result = minimize(
    calibration_loss,
    x0=[1.0],
    args=(y_test_prob_lr_w, y_test),
    bounds=[(0.1, 10)]
)

best_T = opt_result.x[0]
print("Learned temperature:", best_T)


#Apply Calibaration
calibrated_probs = temperature_scale(y_test_prob_lr_w, best_T)


#Evaluate Reliability Again
# Reliability diagram after calibration
reliability_diagram(y_test, calibrated_probs)

# ECE after calibration
ece_calibrated = expected_calibration_error(y_test, calibrated_probs)
print("Calibrated ECE:", ece_calibrated)




