data {
  int<lower=1> N;                       // number of observations
  int<lower=2> K;                       // number of groups
  array[N] int<lower=1, upper=K> g;     // group index for each observation
  vector[N] y;                          // response
  real y_mean;
  real<lower=0> y_sd;
}
parameters {
  real mu0;                              // grand mean
  real<lower=0> sigma_mu;                // between-group SD
  vector[K] mu_raw;                      // standardized group deviations
  real<lower=0> sigma_y;                 // within-group SD
  real<lower=0> nu_minus_one;            // df - 1 for Student-t
}
transformed parameters {
  vector[K] mu;
  real<lower=1> nu;
  mu = mu0 + sigma_mu * mu_raw;          // group means
  nu = nu_minus_one + 1;
}
model {
  // Weakly-informative priors guided by data scale
  mu0 ~ normal(y_mean, 10 * y_sd + 1e-8);
  sigma_mu ~ cauchy(0, y_sd + 1e-8);     // half-Cauchy via positivity constraint
  sigma_y ~ cauchy(0, y_sd + 1e-8);      // half-Cauchy via positivity constraint
  mu_raw ~ normal(0, 1);
  nu_minus_one ~ exponential(1.0 / 29.0);

  // Likelihood (robust Student-t)
  for (n in 1:N) {
    y[n] ~ student_t(nu, mu[g[n]], sigma_y);
  }
}
generated quantities {
  vector[N] y_rep;
  matrix[K, K] diff_mu;
  real<lower=0, upper=1> eta_sq;

  // Posterior predictive
  for (n in 1:N) {
    y_rep[n] = student_t_rng(nu, mu[g[n]], sigma_y);
  }

  // Pairwise mean differences
  for (i in 1:K) {
    for (j in 1:K) {
      diff_mu[i, j] = mu[i] - mu[j];
    }
  }

  // Proportion of variance explained (ANOVA-style)
  eta_sq = square(sigma_mu) / (square(sigma_mu) + square(sigma_y));
}
