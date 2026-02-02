# Quantifying Model Reliability under Covariate Shift in Tabular Machine Learning

This project studies how classical machine learning models behave under covariate shift
using the UCI Adult Income (Census) dataset. The focus is on **model reliability and calibration**,
rather than just raw accuracy.

Covariate shift occurs when the distribution of input features changes between training and deployment.
Even if overall accuracy remains reasonable, the predicted confidence can be misleading.
This project demonstrates how to detect, mitigate, and correct these reliability issues in tabular ML pipelines.

---
##PROBLEM STATEMENT:

Machine learning models are typically trained under the assumption that the training and test data are drawn from the same distribution. However, in real-world deployments, this assumption often breaks due to changes in data collection processes, population demographics, or environmental conditions.

One common form of distribution shift is covariate shift, where the input feature distribution 
𝑃(𝑋)
P(X) changes between training and deployment, while the conditional label distribution 
𝑃(𝑌∣𝑋)
P(Y∣X) remains unchanged.

This project investigates how covariate shift affects not only predictive accuracy but also model reliability and calibration in classical tabular machine learning models. Rather than focusing solely on performance metrics, we study how confident a model is in its predictions under distribution shift and whether those confidence estimates remain trustworthy.

##DATASET

This project uses the UCI Adult Income dataset, a widely used tabular dataset for binary classification tasks. The objective is to predict whether an individual’s annual income exceeds $50K based on demographic and socioeconomic attributes.

COVARIATE SHIFT METHODOLOGY:

To study model reliability under distribution shift, we introduce covariate shift by deliberately altering the input feature distribution between training and test sets while keeping the label generation mechanism unchanged.

##SHIFT CONSTRUCTION:

Covariate shift is simulated using feature-based subpopulation filtering:
The training set is sampled from one region of the feature space (e.g., individuals within a restricted age or education range).
The test set is sampled from a different region of the same feature space, resulting in a changed marginal distribution 
𝑃(𝑋)P(X).

This approach ensures:
𝑃train(𝑋)≠𝑃test(𝑋)Ptrain(X)=Ptest(X)

##SHIFT DETECTION:

A binary classifier (discriminator) is trained to distinguish between training and test samples based solely on input features. High classification accuracy of this discriminator indicates the presence of significant covariate shift.
The predicted probabilities from this discriminator are further used to estimate importance weights, enabling reweighted training to partially correct for distribution mismatch.

MODELS AND TRAINING:
To analyze the effects of covariate shift on both predictive performance and reliability, we evaluate two widely used classical machine learning models:

##MODELS:

Logistic Regression
Chosen for its interpretability and well-calibrated probability estimates under standard i.i.d. conditions. It serves as a strong linear baseline for studying calibration degradation under shift.

Random Forest Classifier
A non-linear ensemble model capable of capturing complex feature interactions. Random Forests often achieve higher accuracy but may produce poorly calibrated confidence estimates, especially under distribution shift.


##TRAINING PIPELINE
The training procedure follows a standardized pipeline to ensure fair comparison across models:

Data preprocessing:
One-hot encoding for categorical features
Feature standardization for continuous variables

Train–test split:
Training data sampled from the original distribution
Test data sampled from a shifted distribution

Model training:
Models are trained using standard empirical risk minimization
Hyperparameters are kept fixed across experiments to isolate the effect of covariate shift

Shift-aware training :
Importance weights estimated using a discriminator model
Weighted loss functions used during training to mitigate covariate shift

##EVALUATION METRICS:
Traditional performance metrics such as accuracy often fail to capture how reliable a model’s confidence estimates are under distribution shift. To address this, we evaluate models using both predictive performance and calibration-based reliability metrics.

##PERFORMANCE METRICS:

Accuracy
Measures the proportion of correctly classified samples. While informative under i.i.d. settings, accuracy alone can be misleading under covariate shift.

F1 Score
Accounts for class imbalance by combining precision and recall, providing a more robust measure of classification performance.

##RELIABILTY AND CALIBARATION METRICS:

Expected Calibration Error (ECE)
Quantifies the difference between predicted confidence and empirical accuracy by partitioning predictions into confidence bins. Lower ECE values indicate better-calibrated models.

Reliability Diagrams
Visualize calibration by comparing predicted confidence against observed accuracy across bins. Deviations from the diagonal indicate miscalibration.


##CONCLUSION:
This project examined the impact of covariate shift on both predictive performance and model reliability in tabular machine learning models. Through controlled experiments, we demonstrated that distribution shift can significantly degrade model calibration, often more severely than classification accuracy.

Our findings highlight that models may remain seemingly accurate while producing unreliable and overconfident predictions, posing risks in real-world deployment. Importance-weighted training and post-hoc calibration techniques were shown to partially mitigate these effects, though neither fully restores performance to i.i.d. conditions.
These results emphasize the importance of evaluating machine learning systems beyond accuracy, particularly in settings where confidence estimates inform downstream decisions.


---
##Install Dependencies
pip install -r requirements.txt


