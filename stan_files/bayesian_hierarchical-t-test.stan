data {
    real delta_low; // Lower bound for delta0
    real delta_high; // Upper bound for delta0
    real std0_low; // Lower bound for std0
    real std0_high; // Upper bound for std0
    real std_low; // Lower bound for sigma
    real std_high; // Upper bound for sigma
    int<lower=2> n_samples; // Number of samples
    int<lower=1> q; // Number of datasets
    matrix[q, n_samples] x; // Matrix of observations
    real rho; // Correlation parameter
    real alpha_upper; // Upper bound for the alpha parameter of the gamma distribution
    real alpha_lower; // Lower bound for the alpha parameter of the gamma distribution
    real beta_upper; // Upper bound for the beta parameter of the gamma distribution
    real beta_lower; // Lower bound for the beta parameter of the gamma distribution
}

transformed data {
    vector[n_samples] H; // Vector of ones
    vector[n_samples] mean0; // Mean vector (all zeros)
    matrix[n_samples, n_samples] inv_M; // Inverse of the covariance matrix
    real det_M; // Determinant of the covariance matrix

    // Calculate the determinant of the covariance matrix
    det_M = (1 + (n_samples - 1) * rho) * pow((1 - rho), n_samples - 1);

    // Initialize the mean vector, H vector, and inverse covariance matrix
    for (j in 1:n_samples) {
        mean0[j] = 0;
        H[j] = 1;
        for (i in 1:n_samples) {
            if (j == i)
                inv_M[j, i] = (1 + (n_samples - 2) * rho) * pow((1 - rho), n_samples - 2);
            else
                inv_M[j, i] = -rho * pow((1 - rho), n_samples - 2);
        }
    }
    inv_M = inv_M / det_M; // Normalize the inverse covariance matrix
}

parameters {
    real<lower=delta_low, upper=delta_high> delta0; // Mean of the delta parameter
    real<lower=std0_low, upper=std0_high> std0; // Standard deviation of the delta parameter
    vector[q] delta; // Vector of delta parameters for each dataset
    vector<lower=std_low, upper=std_high>[q] sigma; // Vector of standard deviations for each dataset
    real<lower=0> nu_minus_one; // Degrees of freedom minus one for the t-distribution
    real<lower=alpha_lower, upper=alpha_upper> gamma_alpha; // Alpha parameter for the gamma distribution
    real<lower=beta_lower, upper=beta_upper> gamma_beta; // Beta parameter for the gamma distribution
}

transformed parameters {
    real<lower=1> nu; // Degrees of freedom for the t-distribution
    matrix[q, n_samples] diff; // Difference matrix between observations and delta
    vector[q] diag_quad; // Diagonal elements of the quadratic form
    vector[q] one_over_sigma2; // Inverse of the variance
    vector[q] log_det_sigma; // Logarithm of the determinant of the covariance matrix
    vector[q] log_lik; // Log-likelihood

    nu = nu_minus_one + 1; // Degrees of freedom for the t-distribution
    one_over_sigma2 = rep_vector(1, q) ./ sigma; // Inverse of the variance
    one_over_sigma2 = one_over_sigma2 ./ sigma; // Squared inverse of the variance
    diff = x - rep_matrix(delta, n_samples); // Difference between observations and delta

    // Calculate the diagonal elements of the quadratic form
    diag_quad = diagonal(quad_form(inv_M, diff'));
    log_det_sigma = 2 * n_samples * log(sigma) + log(det_M); // Logarithm of the determinant of the covariance matrix
    log_lik = -0.5 * log_det_sigma - 0.5 * n_samples * log(6.283); // Log-likelihood part 1
    log_lik = log_lik - 0.5 * one_over_sigma2 .* diag_quad; // Log-likelihood part 2
}


model {
    nu_minus_one ~ gamma(gamma_alpha, gamma_beta); // Prior for nu_minus_one
    delta ~ student_t(nu, delta0, std0); // Prior for delta
    target += sum(log_lik); // Increment the target log probability with the log-likelihood
}

generated quantities {
    matrix[q, n_samples] x_rep; // Matrix to store replicated data

    for (i in 1:q) {
        for (j in 1:n_samples) {
            x_rep[i, j] = student_t_rng(nu, delta[i], sigma[i]);
        }
    }
}
