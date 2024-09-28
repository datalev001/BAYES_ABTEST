import pymc3 as pm
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import arviz as az
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('data_uplift.csv')
data = data[data['offer'].isin(['Buy One Get One', 'No Offer'])]
data['treatment'] = np.where(data['offer'] == 'Buy One Get One', 1, 0)

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['zip_code', 'channel'], drop_first=True)

# Selecting only the specified significant features
X = data[['zip_code_Urban', 'treatment', 'recency', 'used_discount', 'used_bogo', 'is_referral']]
y = data['conversion']

# Add a constant (intercept) term
X = sm.add_constant(X)

# Split the data into training and testing sets for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Take a random sample of the training data for testing purposes
sample_size = 10000  # Adjust this number based on your testing needs
if len(X_train) > sample_size:
    X_train_sample = X_train.sample(n=sample_size, random_state=42)
    y_train_sample = y_train.loc[X_train_sample.index]
else:
    X_train_sample = X_train.copy()
    y_train_sample = y_train.copy()

# Perform Bayesian logistic regression using PyMC3 on the sampled data
with pm.Model() as model:
    # Priors for coefficients
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    coeffs = {}
    for col in X_train_sample.columns:
        coeffs[col] = pm.Normal(col, mu=0, sigma=10)
    
    # Logistic regression model
    linear_combination = intercept
    for col in X_train_sample.columns:
        linear_combination += coeffs[col] * X_train_sample[col].values
    
    theta = pm.Deterministic('theta', pm.math.sigmoid(linear_combination))
    
    # Likelihood
    y_obs = pm.Bernoulli('y_obs', p=theta, observed=y_train_sample.values)
    
    # Sample from the posterior with a larger sample size
    trace = pm.sample(10000, tune=5000, cores=1, random_seed=42, return_inferencedata=True)

# Extract posterior means of the coefficients
coeff_means = {var_name: trace.posterior[var_name].mean().values for var_name in ['intercept'] + list(coeffs.keys())}

# Prepare test data for treatment and control groups
def prepare_group_data(X, treatment_value):
    X_group = X.copy()
    X_group['treatment'] = treatment_value
    return X_group

X_test_control = prepare_group_data(X_test, 0)
X_test_treatment = prepare_group_data(X_test, 1)

# Compute predicted probabilities for both groups
def compute_probs(X_group, coeff_means):
    # Initialize linear_combination as an array with intercept value
    linear_combination = np.full(X_group.shape[0], coeff_means['intercept'])
    for col in X_group.columns:
        if col in coeff_means:
            linear_combination += coeff_means[col] * X_group[col].values
    probs = 1 / (1 + np.exp(-linear_combination))
    return probs

probs_control = compute_probs(X_test_control, coeff_means)
probs_treatment = compute_probs(X_test_treatment, coeff_means)

# Compute the estimated uplift for each individual
uplift = probs_treatment - probs_control
average_uplift = uplift.mean()
print(f"\nEstimated average uplift (Bayesian): {average_uplift:.4f}")

# Estimate the credible interval for the average uplift
uplift_samples = []
n_chains = trace.posterior.dims['chain']
n_draws = trace.posterior.dims['draw']
for idx_chain in range(n_chains):
    for idx_draw in range(n_draws):
        # Extract coefficients for this sample
        coeff_sample = {var_name: trace.posterior[var_name].values[idx_chain, idx_draw] for var_name in ['intercept'] + list(coeffs.keys())}
        # Compute probabilities
        probs_control_sample = compute_probs(X_test_control, coeff_sample)
        probs_treatment_sample = compute_probs(X_test_treatment, coeff_sample)
        uplift_sample = probs_treatment_sample - probs_control_sample
        uplift_samples.append(uplift_sample.mean())

# Calculate 95% credible interval
lower_bound = np.percentile(uplift_samples, 2.5)
upper_bound = np.percentile(uplift_samples, 97.5)
print(f"95% Credible interval for average uplift: ({lower_bound:.4f}, {upper_bound:.4f})")

# Optional: Plot the uplift distribution
plt.figure(figsize=(10,6))
sns.histplot(uplift_samples, kde=True)
plt.title('Posterior Distribution of Average Uplift (PyMC3)')
plt.xlabel('Average Uplift')
plt.ylabel('Density')
plt.show()

'''
Estimated average uplift (Bayesian): 0.0330
95% Credible interval for average uplift: (0.0251, 0.0450)

coefficient posterio 
{'intercept': array(-0.58997382),
 'const': array(-1.77239157),
 'zip_code_Urban': array(-0.07312064),
 'treatment': array(0.30684654),
 'recency': array(-0.06667285),
 'used_discount': array(0.7582661),
 'used_bogo': array(0.9213419),
 'is_referral': array(-0.6163699)}
'''

