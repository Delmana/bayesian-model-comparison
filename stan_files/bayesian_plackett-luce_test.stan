data {
    int<lower=1> n; // number of instances (observations)
    int<lower=2> m; // number of algorithms

    // Matrix with all the rankings, one per row
    array[n, m] int ranks;

    // Vector of weights for each instance
    vector[n] weights;

    // Parameters for Dirichlet prior
    vector[m] alpha;
}

transformed data {
    // Initialize the order array to store the position of each algorithm in the ranking
    array[n, m] int order;

    // Fill the order array based on the rankings
    for (s in 1:n) {
        for (i in 1:m) {
            // ranks[s, i] indicates the rank of algorithm i in instance s
            // order[s, ranks[s, i]] = i means that in instance s, the algorithm ranked at position ranks[s, i] is i
            order[s, ranks[s, i]] = i;
        }
    }
}

parameters {
    // Vector of ratings for each algorithm
    // The simplex constrains the ratings to sum to 1
    simplex[m] ratings;
}

transformed parameters {
    real loglik; // Log likelihood
    real rest; // Intermediate calculation for the denominator

    loglik = 0; // Initialize log likelihood to 0

    // Calculate the log likelihood
    for (s in 1:n) {
        for (i in 1:(m-1)) {
            rest = 0; // Reset the rest for each new term

            // Sum the ratings for the algorithms from position i to m
            for (j in i:m) {
                rest = rest + ratings[order[s, j]];
            }

            // Update the log likelihood
            loglik = loglik + log(weights[s] * ratings[order[s, i]] / rest);
        }
    }
}

model {
    // Dirichlet prior for the ratings
    ratings ~ dirichlet(alpha);

    // Add the log likelihood to the target (posterior log density)
    target += loglik;
}

generated quantities {
    array[n, m] int ranks_rep; // Array to store the replicated rankings

    for (s in 1:n) {
        array[m] real cum_ratings; // Array to store the cumulative ratings
        array[m] real adjusted_ratings; // Array to store adjusted ratings for sampling

        // Initialize the adjusted ratings
        for (i in 1:m) {
            adjusted_ratings[i] = ratings[i];
        }

        // Compute the cumulative ratings
        cum_ratings[1] = adjusted_ratings[1];
        for (j in 2:m) {
            cum_ratings[j] = cum_ratings[j-1] + adjusted_ratings[j];
        }

        array[m] int indices; // Array to store the selected ranks

        // Sample ranks without replacement
        for (i in 1:m) {
            real u = uniform_rng(0, cum_ratings[m]); // Sample a uniform random number
            int rank = 1;

            // Determine the rank based on the cumulative ratings
            while (u > cum_ratings[rank]) {
                rank += 1;
            }

            ranks_rep[s, i] = rank; // Assign the sampled rank
            indices[i] = rank; // Store the rank for adjustment

            // Adjust the cumulative ratings to avoid negative values
            real adjustment = adjusted_ratings[rank];
            for (j in rank:m) {
                cum_ratings[j] -= adjustment;
            }

            // Set the selected rating to zero to prevent re-selection
            adjusted_ratings[rank] = 0;
        }

        // Re-sort the ranks_rep array to match the original order
        for (i in 1:m) {
            ranks_rep[s, indices[i]] = i;
        }
    }
}
