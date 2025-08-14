import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# What is the true probability p that someone will sign up if we use method A?
# We have data: 16 people were asked, 6 said yes.
# We want to represent our uncertainty about p and use Bayesian statistics to do so.

# Number of draws for the Bayesian analysis
n_draws = 100000

# Sample size
n = 16

# Number of observed "yes" responses
observed_data = 6

# Here you sample n_draws draws from the prior into a pandas Series
# (to have convenient methods available for historgrams and descriptive statistics) 

# We act as if we know nothing about p before seeing the data.
# Therefore, we assume evenly distributed values between 0 and 1.
# Each value is a “guess” for the true probability p.
prior = pd.Series(np.random.uniform(0, 1, size = n_draws))

# It's always a good idea to visualize the prior distribution as Histogram
# Prior-Histogramm
# A bar in the histogram at a specific p-value shows how many of the total parameter values drawn fell into the corresponding value interval around this p-value. The width of the interval is determined by the histogram setting (“bin width”).
ig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left histogram – Frequency
axes[0].hist(prior, bins=10)
axes[0].set_title("Prior distribution of p")
axes[0].set_xlabel("p")
axes[0].set_ylabel("Frequency")
axes[0].grid(True)

# Right histogram – Probability (relative frequency)
axes[1].hist(prior, bins=10, weights=np.ones_like(prior) / len(prior))
axes[1].set_title("Prior: Relative frequency per bin of the rate of sign-up")
axes[1].set_xlabel("Rate of sign-up (p)")
axes[1].set_ylabel("Relative frequency (Probability)")
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Defining the generative model
# Says: If the true probability p = parameters, how many yes answers do we expect in 16 surveys?
# The binomial distribution provides exactly such random numbers.
def generative_model(parameters):
    return(np.random.binomial(16, parameters))

# Here you simulate data using the parameters from the prior and the generative model.
# For each of our n_draws guesses for p, we draw a random number from the binomial distribution and use it to simulate the number of yes answers.
sim_data = list()
for p in prior:
    sim_data.append(generative_model(p))

# Here you filter off all draws (simulated data) that match the observed data
# We only keep the p-values for which the model generated exactly 6 Yes responses.
# These are the plausible values for p after we have seen the data → Posterior.
posterior = prior[list(map(lambda x: x == observed_data, sim_data))]

# The posterior distribution indicates: “How likely is each possible value of 𝑝 after I have seen the data?”
# Posterior-Histogramm
# x-axis (p): possible values for the true probability of success p of method A 
# y-axis (Frequency): How many of the posterior samples drawn were in this value range.
# (What is the probability that, given a registration probability p, exactly 6 registrations will be observed in a sample of 16 people?
# For example: With 𝑝 = 0.40, this event was observed approximately 430 times in 100,000 simulations, while with  p=0.70 it occurred only about 10 times.)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left histogram – Frequency
axes[0].hist(posterior, bins=10)
axes[0].set_title(f"Posterior of p | X = {observed_data}")
axes[0].set_xlabel("p")
axes[0].set_ylabel("Frequency")
axes[0].grid(True)

# Right histogram – Probability (relative frequency)
axes[1].hist(posterior, bins=10, weights=np.ones_like(posterior) / len(posterior))
axes[1].set_title("Posterior rate of sign up")
axes[1].set_xlabel("Posterior rate of sign up")
axes[1].set_ylabel("Probability")
axes[1].grid(True)

plt.tight_layout()
plt.show()


# Now you can summarize the posterior, where a common summary is to take the mean or the median posterior, 
# and perhaps a 95% quantile interval.

print('Number of draws left (Nur x der n_draws  passten exakt zur beobachteten Zahl): %d, Posterior median: %.3f, Posterior quantile interval: %.3f-%.3f' % 
      (len(posterior), posterior.median(), posterior.quantile(.025), posterior.quantile(.975)))


# Probability that I will get 6 yeses again with p_m = posterior.median()
print('Probability of getting 6 hits with p_m = posterior.median(): %.3f' % 
      st.binom.pmf(6, n, posterior.median()))


# Proportion of posterior samples that are > 0.2
prob_A_gt_20_v1 = np.mean(posterior > 0.2)   
prob_A_gt_20_v2 = sum(posterior > 0.2) / len(posterior) 
print(prob_A_gt_20_v1, prob_A_gt_20_v2)


#  If method A was used on 100 people what would be number of sign-ups?
# with p = “estimated registration probability per person” from the posterior distribution.
signups = pd.Series([np.random.binomial(n = 100, p = p) for p in posterior])


plt.figure()
signups.hist(bins=10)
plt.title(f"Sign-up 95% quantile interval 17-63")
plt.xlabel("Sign-ups")
plt.ylabel("Frequency bei 100 000 people")
plt.tight_layout()
plt.show() 

print('Sign-up 95%% quantile interval %d-%d' % tuple(signups.quantile([.025, .975]).values))

# Median number of sign-ups
print(f"Median der signups: {signups.median()}")

