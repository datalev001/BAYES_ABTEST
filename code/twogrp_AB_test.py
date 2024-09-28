import numpy as np
import pandas as pd
import scipy.stats as stats
import pystan

np.random.seed(42)
n_control = 500  # Number of control group users
n_test = 500     # Number of test group users

# Prior belief: Control group conversion rate ~ 6%, Test group ~ 8%
control_prior_conversion = 0.06
test_prior_conversion = 0.08

# Generate conversion data with prior knowledge
control_data_prior = np.random.binomial(1, control_prior_conversion, n_control)
test_data_prior = np.random.binomial(1, test_prior_conversion, n_test)

# No prior beliefs, same data structure as above but randomly generated without bias
n_control_raw = 500
n_test_raw = 500

# Random conversion rates to simulate real-world data
control_conversion_rate_raw = np.random.uniform(0.04, 0.10)
test_conversion_rate_raw = np.random.uniform(0.04, 0.10)

control_data_raw = np.random.binomial(1, control_conversion_rate_raw, n_control_raw)
test_data_raw = np.random.binomial(1, test_conversion_rate_raw, n_test_raw)

# Bayesian A/B Test with Prior Information
prior_control = stats.beta(2, 30)
prior_test = stats.beta(5, 30)

# Update with observed data
posterior_control = stats.beta(2 + control_data_prior.sum(), 30 + n_control - control_data_prior.sum())
posterior_test = stats.beta(5 + test_data_prior.sum(), 30 + n_test - test_data_prior.sum())

# Posterior samples
samples_control = posterior_control.rvs(100000)
samples_test = posterior_test.rvs(100000)

# Compare the samples
posterior_prob_test_better = np.mean(samples_test > samples_control)
print(f"Probability that Test is better than Control (with prior): {posterior_prob_test_better:.4f}")

# Credible intervals
credible_interval_control = np.percentile(samples_control, [2.5, 97.5])
credible_interval_test = np.percentile(samples_test, [2.5, 97.5])
print(f"95% Credible Interval for Control (with prior): {credible_interval_control}")
print(f"95% Credible Interval for Test (with prior): {credible_interval_test}")

# Bayesian A/B Test without Prior Information
posterior_control_raw = stats.beta(1 + control_data_raw.sum(), 1 + n_control_raw - control_data_raw.sum())
posterior_test_raw = stats.beta(1 + test_data_raw.sum(), 1 + n_test_raw - test_data_raw.sum())

# Posterior samples
samples_control_raw = posterior_control_raw.rvs(100000)
samples_test_raw = posterior_test_raw.rvs(100000)

# Compare the samples
posterior_prob_test_better_raw = np.mean(samples_test_raw > samples_control_raw)
print(f"Probability that Test is better than Control (without prior): {posterior_prob_test_better_raw:.4f}")

# Credible intervals
credible_interval_control_raw = np.percentile(samples_control_raw, [2.5, 97.5])
credible_interval_test_raw = np.percentile(samples_test_raw, [2.5, 97.5])
print(f"95% Credible Interval for Control (without prior): {credible_interval_control_raw}")
print(f"95% Credible Interval for Test (without prior): {credible_interval_test_raw}")

# Traditional t-test for comparison
control_mean = control_data_raw.mean()
test_mean = test_data_raw.mean()

# Performing t-test
t_stat, p_val = stats.ttest_ind(control_data_raw, test_data_raw)
print(f"T-test p-value: {p_val:.4f}")
print(f"Control mean conversion rate: {control_mean:.4f}")
print(f"Test mean conversion rate: {test_mean:.4f}")

#Validating Bayesian Advantage over T-test
control_ci = np.percentile(np.random.choice(control_data_raw, (100000, n_control_raw)).mean(axis=1), [2.5, 97.5])
test_ci = np.percentile(np.random.choice(test_data_raw, (100000, n_test_raw)).mean(axis=1), [2.5, 97.5])

print(f"Traditional 95% Confidence Interval for Control: {control_ci}")
print(f"Traditional 95% Confidence Interval for Test: {test_ci}")

# Compare Bayesian Credible Intervals vs Traditional Confidence Intervals
print(f"Bayesian Credible Interval is more informative due to incorporation of prior knowledge and provides a direct probability of one group being better than the other, which traditional t-test lacks.")

