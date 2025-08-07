data {
  int<lower=2> n; // Number of tasks (must be at least 2)
  vector[n] y1;   // Performance statistics for the first method
  vector[n] y2;   // Performance statistics for the second method
  real<lower=0, upper=1> rho; // Fraction of data used as test set, indicating the correlation introduced by cross-validation
}

transformed data {
  vector[n] x = y1 - y2; // Compute the difference in performance statistics between the two methods for each task
  real mean_x = mean(x); // Calculate the mean of the differences
  real nu = n - 1.0; // Degrees of freedom for the Student's t-distribution (n-1 because it's based on n samples)
  real<lower=0> var_x = variance(x); // Estimate the variance of the differences
  real<lower=0> corrected_var = (1.0 / n + rho / (1.0 - rho)) * var_x; // Corrected variance using Nadeau-Bengio's correction
  real scale = sqrt(corrected_var); // Calculate the scale parameter incorporating the correlation factor
}

parameters {
  real mu; // Mean difference of the differences, the parameter to be estimated
}

model {
  // Prior for mu as a Student's t-distribution with degrees of freedom nu, mean, and scale
  mu ~ student_t(nu, mean_x, scale);
}

generated quantities {
  // For posterior predictive checking.
  vector[n] x_rep; // Replicated data for posterior predictive check
  vector[n] y1_rep; // Replicated y1 data
  vector[n] y2_rep; // Replicated y2 data

  // Generate replicated data based on the posterior distribution
  for (i in 1:n) {
    x_rep[i] = student_t_rng(nu, mu, scale);
    y1_rep[i] = y2[i] + x_rep[i]; // Calculate y1_rep from y2 and x_rep
    y2_rep[i] = y1[i] - x_rep[i]; // Calculate y2_rep from y1 and x_rep
  }
}
