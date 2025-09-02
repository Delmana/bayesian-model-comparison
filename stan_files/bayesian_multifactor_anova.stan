
// bayesian_multifactor_anova.stan
// Robust Bayesian multifactor ANOVA (up to 3 factors) with optional interactions
// Following Kruschke (2015) Ch. 20. Sum-to-zero constraints via centering.
// Student-t likelihood for robustness.

functions {
  // Center a vector to sum to zero (subtract mean)
  vector center_sum_to_zero(vector v) {
    return v - mean(v);
  }

  // Double-center a matrix so row/col sums are zero (add grand mean to compensate)
  matrix double_center(matrix M) {
    int R = rows(M);
    int C = cols(M);
    vector[R] rmean;
    vector[C] cmean;
    real gmean = mean(to_vector(M));
    for (r in 1:R) rmean[r] = mean(M[r]);
    for (c in 1:C) cmean[c] = mean(col(M, c));
    matrix[R, C] out;
    for (r in 1:R) {
      for (c in 1:C) {
        out[r, c] = M[r, c] - rmean[r] - cmean[c] + gmean;
      }
    }
    return out;
  }

  // Sample variance helper
  real sample_variance(vector x) {
    int N = num_elements(x);
    real m = mean(x);
    real ss = dot_self(x - rep_vector(m, N));
    return ss / (N - 1);
  }
}

data {
  int<lower=1> N;
  vector[N] y;
  real y_mean;
  real<lower=0> y_sd;

  int<lower=1, upper=3> K;         // number of factors provided
  int<lower=0, upper=1> include_interactions;

  // Up to 3 factors, each with J_k levels. If a factor not used, J_k=0.
  int<lower=0> J_1;
  int<lower=0> J_2;
  int<lower=0> J_3;

  // Observation indices (1..J_k). If not used, filled with 1s.
  array[N] int<lower=1> idx_1;
  array[N] int<lower=1> idx_2;
  array[N] int<lower=1> idx_3;
}

parameters {
  real alpha;

  // Main effects (centered to sum to zero when J_k>0)
  vector[J_1] a1_raw;
  vector[J_2] a2_raw;
  vector[J_3] a3_raw;

  // Hierarchical SDs for main effects
  real<lower=0> sigma_a1;
  real<lower=0> sigma_a2;
  real<lower=0> sigma_a3;

  // Interactions (if enabled)
  matrix[J_1, J_2] ab_raw;
  matrix[J_1, J_3] ac_raw;
  matrix[J_2, J_3] bc_raw;
  array[J_1] matrix[J_2, J_3] abc_raw; // 3-way as an array of J1 matrices of size J2xJ3

  // Hierarchical SDs for interactions
  real<lower=0> sigma_ab;
  real<lower=0> sigma_ac;
  real<lower=0> sigma_bc;
  real<lower=0> sigma_abc;

  // Likelihood scale & df
  real<lower=0> sigma_y;
  real<lower=0> nu_minus_one;
}

transformed parameters {
  real nu = nu_minus_one + 1;

  // Centered main effects (sum to zero if J_k>0)
  vector[J_1] a1 = (J_1 > 0) ? center_sum_to_zero(a1_raw) : rep_vector(0.0, 0);
  vector[J_2] a2 = (J_2 > 0) ? center_sum_to_zero(a2_raw) : rep_vector(0.0, 0);
  vector[J_3] a3 = (J_3 > 0) ? center_sum_to_zero(a3_raw) : rep_vector(0.0, 0);

  // Centered interactions
  matrix[J_1, J_2] ab = (include_interactions == 1 && J_1>0 && J_2>0) ? double_center(ab_raw) : rep_matrix(0.0, J_1, J_2);
  matrix[J_1, J_3] ac = (include_interactions == 1 && J_1>0 && J_3>0) ? double_center(ac_raw) : rep_matrix(0.0, J_1, J_3);
  matrix[J_2, J_3] bc = (include_interactions == 1 && J_2>0 && J_3>0) ? double_center(bc_raw) : rep_matrix(0.0, J_2, J_3);

  // For 3-way, we zero-center each J2xJ3 slice and across slices (J1) as well
  array[J_1] matrix[J_2, J_3] abc;
  if (include_interactions == 1 && J_1>0 && J_2>0 && J_3>0) {
    // first double-center each slice
    for (i1 in 1:J_1) {
      abc[i1] = double_center(abc_raw[i1]);
    }
    // then center across slices per position to make sum over i1 zero
    if (J_1 > 0) {
      for (j2 in 1:J_2) {
        for (j3 in 1:J_3) {
          real m = 0;
          for (i1 in 1:J_1) m += abc[i1][j2, j3];
          m /= J_1;
          for (i1 in 1:J_1) abc[i1][j2, j3] -= m;
        }
      }
    }
  } else {
    for (i1 in 1:J_1) abc[i1] = rep_matrix(0.0, J_2, J_3);
  }
}

model {
  // Priors
  alpha ~ normal(y_mean, 10 * y_sd);

  if (J_1 > 0) {
    sigma_a1 ~ student_t(3, 0, y_sd);
    a1_raw ~ normal(0, sigma_a1);
  }
  if (J_2 > 0) {
    sigma_a2 ~ student_t(3, 0, y_sd);
    a2_raw ~ normal(0, sigma_a2);
  }
  if (J_3 > 0) {
    sigma_a3 ~ student_t(3, 0, y_sd);
    a3_raw ~ normal(0, sigma_a3);
  }

  if (include_interactions == 1) {
    if (J_1>0 && J_2>0) {
      sigma_ab ~ student_t(3, 0, y_sd);
      to_vector(ab_raw) ~ normal(0, sigma_ab);
    } else {
      sigma_ab ~ normal(0, 1); // weak prior placeholder
    }
    if (J_1>0 && J_3>0) {
      sigma_ac ~ student_t(3, 0, y_sd);
      to_vector(ac_raw) ~ normal(0, sigma_ac);
    } else {
      sigma_ac ~ normal(0, 1);
    }
    if (J_2>0 && J_3>0) {
      sigma_bc ~ student_t(3, 0, y_sd);
      to_vector(bc_raw) ~ normal(0, sigma_bc);
    } else {
      sigma_bc ~ normal(0, 1);
    }
    if (J_1>0 && J_2>0 && J_3>0) {
      sigma_abc ~ student_t(3, 0, y_sd);
      for (i1 in 1:J_1) to_vector(abc_raw[i1]) ~ normal(0, sigma_abc);
    } else {
      sigma_abc ~ normal(0, 1);
    }
  } else {
    sigma_ab ~ normal(0, 1);
    sigma_ac ~ normal(0, 1);
    sigma_bc ~ normal(0, 1);
    sigma_abc ~ normal(0, 1);
  }

  sigma_y ~ student_t(3, 0, y_sd);
  nu_minus_one ~ exponential(1.0 / 29.0);

  // Likelihood
  for (n in 1:N) {
    real mu_n = alpha;
    if (J_1 > 0) mu_n += a1[idx_1[n]];
    if (J_2 > 0) mu_n += a2[idx_2[n]];
    if (J_3 > 0) mu_n += a3[idx_3[n]];

    if (include_interactions == 1) {
      if (J_1>0 && J_2>0) mu_n += ab[idx_1[n], idx_2[n]];
      if (J_1>0 && J_3>0) mu_n += ac[idx_1[n], idx_3[n]];
      if (J_2>0 && J_3>0) mu_n += bc[idx_2[n], idx_3[n]];
      if (J_1>0 && J_2>0 && J_3>0) mu_n += abc[idx_1[n]][idx_2[n], idx_3[n]];
    }
    y[n] ~ student_t(nu, mu_n, sigma_y);
  }
}

generated quantities {
  vector[N] y_rep;
  real eta_sq;

  // Posterior predictive
  for (n in 1:N) {
    real mu_n = alpha;
    if (J_1 > 0) mu_n += a1[idx_1[n]];
    if (J_2 > 0) mu_n += a2[idx_2[n]];
    if (J_3 > 0) mu_n += a3[idx_3[n]];

    if (include_interactions == 1) {
      if (J_1>0 && J_2>0) mu_n += ab[idx_1[n], idx_2[n]];
      if (J_1>0 && J_3>0) mu_n += ac[idx_1[n], idx_3[n]];
      if (J_2>0 && J_3>0) mu_n += bc[idx_2[n], idx_3[n]];
      if (J_1>0 && J_2>0 && J_3>0) mu_n += abc[idx_1[n]][idx_2[n], idx_3[n]];
    }
    y_rep[n] = student_t_rng(nu, mu_n, sigma_y);
  }

  // Total eta^2 using variance of linear predictor across observed design
  {
    vector[N] mu_vec;
    for (n in 1:N) {
      real mu_n = alpha;
      if (J_1 > 0) mu_n += a1[idx_1[n]];
      if (J_2 > 0) mu_n += a2[idx_2[n]];
      if (J_3 > 0) mu_n += a3[idx_3[n]];

      if (include_interactions == 1) {
        if (J_1>0 && J_2>0) mu_n += ab[idx_1[n], idx_2[n]];
        if (J_1>0 && J_3>0) mu_n += ac[idx_1[n], idx_3[n]];
        if (J_2>0 && J_3>0) mu_n += bc[idx_2[n], idx_3[n]];
        if (J_1>0 && J_2>0 && J_3>0) mu_n += abc[idx_1[n]][idx_2[n], idx_3[n]];
      }
      mu_vec[n] = mu_n;
    }
    real var_mu = sample_variance(mu_vec);
    eta_sq = var_mu / (var_mu + square(sigma_y));
  }

  // --- Marginal means & pairwise diffs per factor ---
  // We average cell means equally across levels of the *other* factors.
  // To keep output sizes modest, we store only packed upper-triangular diffs.

  array[(J_1 > 1) ? (J_1 * (J_1 - 1) / 2) : 0] real diff_A;
array[(J_2 > 1) ? (J_2 * (J_2 - 1) / 2) : 0] real diff_B;
array[(J_3 > 1) ? (J_3 * (J_3 - 1) / 2) : 0] real diff_C;

  {
    // Build full cell mean array for equal-weight marginalization
    // Dimensions default to at least 1 for ease of loops
    int A = (J_1 > 0) ? J_1 : 1;
    int B = (J_2 > 0) ? J_2 : 1;
    int C = (J_3 > 0) ? J_3 : 1;
    array[A] matrix[B, C] mu_cell;

    for (i1 in 1:A) {
      for (i2 in 1:B) {
        for (i3 in 1:C) {
          real mu = alpha;
          if (J_1 > 0) mu += a1[i1];
          if (J_2 > 0) mu += a2[i2];
          if (J_3 > 0) mu += a3[i3];
          if (include_interactions == 1) {
            if (J_1>0 && J_2>0) mu += ab[i1, i2];
            if (J_1>0 && J_3>0) mu += ac[i1, i3];
            if (J_2>0 && J_3>0) mu += bc[i2, i3];
            if (J_1>0 && J_2>0 && J_3>0) mu += abc[i1][i2, i3];
          }
          mu_cell[i1][i2, i3] = mu;
        }
      }
    }

    // Marginals
    if (J_1 > 0) {
      vector[J_1] mu_A;
      for (i1 in 1:J_1) {
        real s = 0;
        for (i2 in 1:B) for (i3 in 1:C) s += mu_cell[i1][i2, i3];
        mu_A[i1] = s / (B * C);
      }
      // packed differences
      if (J_1 > 1) {
        int p = 1;
        for (i in 1:(J_1-1)) {
          for (j in (i+1):J_1) {
            diff_A[p] = mu_A[i] - mu_A[j];
            p += 1;
          }
        }
      }
    }
    if (J_2 > 0) {
      vector[J_2] mu_B;
      for (i2 in 1:J_2) {
        real s = 0;
        for (i1 in 1:A) for (i3 in 1:C) s += mu_cell[i1][i2, i3];
        mu_B[i2] = s / (A * C);
      }
      if (J_2 > 1) {
        int p = 1;
        for (i in 1:(J_2-1)) {
          for (j in (i+1):J_2) {
            diff_B[p] = mu_B[i] - mu_B[j];
            p += 1;
          }
        }
      }
    }
    if (J_3 > 0) {
      vector[J_3] mu_C;
      for (i3 in 1:J_3) {
        real s = 0;
        for (i1 in 1:A) for (i2 in 1:B) s += mu_cell[i1][i2, i3];
        mu_C[i3] = s / (A * B);
      }
      if (J_3 > 1) {
        int p = 1;
        for (i in 1:(J_3-1)) {
          for (j in (i+1):J_3) {
            diff_C[p] = mu_C[i] - mu_C[j];
            p += 1;
          }
        }
      }
    }
  }
}
