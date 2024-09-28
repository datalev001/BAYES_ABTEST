import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats

# Read the data into a pandas DataFrame
data = pd.read_csv('data_uplift.csv')

# Create a treatment indicator variable
# We'll define 'Buy One Get One' as the treatment and 'No Offer' as the control
data = data[data['offer'].isin(['Buy One Get One', 'No Offer'])]

# Create the treatment indicator: 1 for treatment group, 0 for control group
data['treatment'] = np.where(data['offer'] == 'Buy One Get One', 1, 0)

# Encode categorical variables using one-hot encoding
# The categorical variables are 'zip_code' and 'channel'
data = pd.get_dummies(data, columns=['zip_code', 'channel'], drop_first=True)

# Prepare the feature matrix X and the target vector y
# Exclude 'conversion' (target variable) and 'offer' (since we now have 'treatment')
X = data.drop(columns=['conversion', 'offer'])
y = data['conversion']

# Add interaction terms between treatment and other features
# We'll create interaction terms by multiplying 'treatment' with each feature
# Get the list of feature names excluding 'treatment'
feature_names = [col for col in X.columns if col != 'treatment' and col != 'const']

# Create interaction terms
for feature in feature_names:
    X[f'{feature}_x_treatment'] = X[feature] * X['treatment']

# Add a constant term (intercept) to the model
X = sm.add_constant(X)

# Fit the logistic regression model
logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary())

#Refit the logistic regression model using only significant variables
# Keeping 'zip_code_Urban', 'treatment', 'recency', 'used_discount', 'used_bogo', and 'is_referral'
significant_features = ['const', 'zip_code_Urban', 'treatment', 'recency', 
                        'used_discount', 'used_bogo', 'is_referral']
X_significant = X[significant_features]
logit_model_significant = sm.Logit(y, X_significant)
result_significant = logit_model_significant.fit()
print(result_significant.summary())

#Predict probabilities for both treatment and control groups using the significant model
X_control_significant = X_significant.copy()
X_control_significant['treatment'] = 0

X_treatment_significant = X_significant.copy()
X_treatment_significant['treatment'] = 1

probs_control_significant = result_significant.predict(X_control_significant)
probs_treatment_significant = result_significant.predict(X_treatment_significant)

#Compute the estimated uplift for each individual
uplift_significant = probs_treatment_significant - probs_control_significant

# Compute the average uplift across all individuals
average_uplift_significant = uplift_significant.mean()
print(f"Estimated average uplift (significant variables): {average_uplift_significant:.3f}")

#Estimate the standard error of the average uplift
std_error_significant = uplift_significant.std(ddof=1) / np.sqrt(len(uplift_significant))

# Perform a t-test to assess whether the average uplift is significantly different from zero
t_stat_significant = average_uplift_significant / std_error_significant
p_value_significant = (1 - t.cdf(abs(t_stat_significant), df=len(uplift_significant) - 1)) * 2

print(f"T-statistic (significant variables): {t_stat_significant:.3f}")
print(f"P-value (significant variables): {p_value_significant:.3f}")

#Calculate the confidence interval for the average uplift using significant variables
t_critical_significant = t.ppf((1 + confidence_level) / 2, df=len(uplift_significant) - 1)
margin_of_error_significant = t_critical_significant * std_error_significant
lower_bound_significant = average_uplift_significant - margin_of_error_significant
upper_bound_significant = average_uplift_significant + margin_of_error_significant

print(f"{int(confidence_level*100)}% Confidence interval for average uplift (significant variables): ({lower_bound_significant:.3f}, {upper_bound_significant:.3f})")
