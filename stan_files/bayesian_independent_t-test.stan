data {
  // Number of runs in the first data set.
  int<lower=1> n_runs1;

  // Number of runs in the second data set.
  int<lower=1> n_runs2;

  // Units for the first method to be compared.
  vector[n_runs1] y1;

  // Units for the second method to be compared.
  vector[n_runs2] y2;
}

transformed data {
  // Combine the two data sets into one vector.
  vector[n_runs1 + n_runs2] y = append_row(y1, y2);

  // Hyperparameters for the priors on mu1 and mu2.
  real S = 1000 * sd(y); // Scale for the normal prior.
  real M = mean(y); // Mean for the normal prior.

  // Hyperparameters for the priors on sigma1 and sigma2.
  real L = 1.0 / 1000 * sd(y); // Lower bound for the uniform prior.
  real H = 1000 * sd(y); // Upper bound for the uniform prior.

  // Hyperparameter for the prior on nu.
  real lambda = 1.0 / 29;
}

parameters {
  // Means of the metrics for the two methods.
  real mu1;
  real mu2;

  // Standard deviations of the metrics for the two methods.
  real<lower=L,upper=H> sigma1;
  real<lower=L,upper=H> sigma2;

  // Degrees of freedom parameter for the Student's t-distribution.
  real<lower = 1, upper=50> nu;
}

model {
  // Priors for the means.
  mu1 ~ normal(M, S);
  mu2 ~ normal(M, S);

  // Priors for the standard deviations.
  // non-informative
  sigma1 ~ uniform(L, H);
  sigma2 ~ uniform(L, H);

   // weakly-informative
   // sigma1 ~ cauchy(0, 5);
   // sigma2 ~ cauchy(0, 5);


  // Prior for the degrees of freedom.
  nu ~ exponential(lambda);

  // Likelihood for the data.
  y1 ~ student_t(nu, mu1, sigma1);
  y2 ~ student_t(nu, mu2, sigma2);
}

generated quantities {
  // Calculate the difference in means.
  real difference_mean = mu1 - mu2;

  // Calculate the difference in standard deviations.
  real difference_sigma = sigma1 - sigma2;

  // Calculate the effect size.
  real effect_size = (mu1 - mu2) / sqrt((sigma1^2 + sigma2^2) / 2);

  // Posterior predictive checks.
  array [n_runs1] real y1_rep; // Replicated data for y1.
  array [n_runs2] real y2_rep; // Replicated data for y2.
  for (i in 1:n_runs1) y1_rep[i] = student_t_rng(nu, mu1, sigma1);
  for (i in 1:n_runs2) y2_rep[i] = student_t_rng(nu, mu2, sigma2);
}
