/* A model for comparing two sets of measurements of non-negative metrics.
 *
 * Each data set is modeled using a single Gamma distribution. Priors on the
 * parameters of the Gamma distribution are based on the mean of the Gamma
 * distributions being assumed to stem from an exponential distribution while
 * the variance lies in an interval derived from the variance in the data.
 */
data {
  // Number of runs in the first data set.
  int<lower=1> n_runs1;

  // Number of runs in the second data set.
  int<lower=1> n_runs2;

  // (Uncensored) units for the first method to be compared.
  vector[n_runs1] y1;

  // (Uncensored) units for the second method to be compared.
  vector[n_runs2] y2;

  // Factor for the lower and upper bound on the variance of the prior on the
  // gamma distribution.
  real<lower=0, upper=1> var_lower;
  real<lower=var_lower> var_upper;

  // Hyperprior parameter. Assume that with a probability of 90%, the real data
  // means lie in [min(y)/100, min(y)/100 + mean_upper].
  real<lower=0> mean_upper;

  // Number of censored data points for each of the methods.
  int<lower=0> n_censored1;
  int<lower=0> n_censored2;

  // Value above which data has been censored.
  real<lower=0> censoring_point;
}

transformed data {
  // 90% of (exponentially distributed) unshifted means lie in [0, mean_upper].
  real lambda_mean1 = - log(1 - 0.9) / mean_upper;
  real lambda_mean2 = - log(1 - 0.9) / mean_upper;

  // Cached upper and lower bounds of the variance for y1 and y2.
  real var_y1_upper = var_upper * variance(y1);
  real var_y1_lower = var_lower * variance(y1);
  real var_y2_upper = var_upper * variance(y2);
  real var_y2_lower = var_lower * variance(y2);

  // Minimum mean to consider for each of the components, a hundredth of the minimum of the data.
  real min_mean1 = min(y1) / 100;
  real min_mean2 = min(y2) / 100;
}

parameters {
  // Unshifted mean parameters for the Gamma distributions.
  real<lower=0> mean1_unshifted;
  real<lower=0> mean2_unshifted;

  // Beta parameters for the Gamma distributions, constrained by the variance bounds.
  real<lower=(mean1_unshifted + min_mean1) / var_y1_upper, upper=(mean1_unshifted + min_mean1) / var_y1_lower> beta1;
  real<lower=(mean2_unshifted + min_mean2) / var_y2_upper, upper=(mean2_unshifted + min_mean2) / var_y2_lower> beta2;

  // Censored data points for y1 and y2.
  array[n_censored1] real<lower=censoring_point> y1_cens;
  array[n_censored2] real<lower=censoring_point> y2_cens;
}

transformed parameters {
  // Shifted mean parameters for the Gamma distributions.
  real<lower=min_mean1> mean1 = mean1_unshifted + min_mean1;
  real<lower=min_mean2> mean2 = mean2_unshifted + min_mean2;

  // Alpha parameters for the Gamma distributions.
  real<lower=0> alpha1 = mean1 * beta1;
  real<lower=0> alpha2 = mean2 * beta2;
}

model {
  // Prior distributions for the unshifted mean parameters.
  mean1_unshifted ~ exponential(lambda_mean1);
  mean2_unshifted ~ exponential(lambda_mean2);

  // Prior distributions for the beta parameters, constrained by the variance bounds.
  beta1 ~ uniform(mean1 / var_y1_upper, mean1 / var_y1_lower);
  beta2 ~ uniform(mean2 / var_y2_upper, mean2 / var_y2_lower);

  // Likelihoods for the uncensored data.
  y1 ~ gamma(alpha1, beta1);
  y2 ~ gamma(alpha2, beta2);
  // Likelihoods for the censored data.
  y1_cens ~ gamma(alpha1, beta1);
  y2_cens ~ gamma(alpha2, beta2);
}

generated quantities {
  // Calculate the difference in means.
  real difference_mean = mean1 - mean2;

  // Posterior predictive check.
  array[n_runs1] real y1_rep;
  for (i in 1:n_runs1) {
    y1_rep[i] = gamma_rng(alpha1, beta1);
  }
  array[n_runs2] real y2_rep;
  for (i in 1:n_runs2) {
    y2_rep[i] = gamma_rng(alpha2, beta2);
  }
}
