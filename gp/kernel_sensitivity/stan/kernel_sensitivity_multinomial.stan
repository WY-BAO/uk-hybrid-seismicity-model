// Catalogue-conditioned multinomial spatial GP for kernel sensitivity.
// The only case selector is the covariance correlation function.

functions {
  real kernel_correlation(real distance, real rho, int kernel_id) {
    real r = distance / rho;
    if (kernel_id == 1) {
      return exp(-0.5 * square(r));
    } else if (kernel_id == 2) {
      // This tail is already zero in double precision. Guarding it prevents
      // an infinity-times-zero NaN for rejected extreme warmup proposals.
      if (r > 1e50) {
        return 0;
      }
      real z = sqrt(5.0) * r;
      return exp(log(1 + z + (5.0 / 3.0) * square(r)) - z);
    } else if (kernel_id == 3) {
      if (r > 1e50) {
        return 0;
      }
      real z = sqrt(3.0) * r;
      return exp(log1p(z) - z);
    } else if (kernel_id == 4) {
      return exp(-r);
    } else if (kernel_id == 5) {
      return square(inv(1 + 0.25 * square(r)));
    } else if (kernel_id == 6) {
      return inv(1 + square(r));
    } else {
      if (r < 1) {
        return square(square(1 - r)) * (1 + 4 * r);
      }
      return 0;
    }
  }

  matrix kernel_covariance(
      array[] vector x,
      real alpha,
      real rho,
      int kernel_id) {
    int N = size(x);
    matrix[N, N] K;
    for (i in 1:N) {
      K[i, i] = square(alpha);
      if (i < N) {
        for (j in (i + 1):N) {
          real distance = sqrt(dot_self(x[i] - x[j]));
          real covariance = square(alpha)
                            * kernel_correlation(distance, rho, kernel_id);
          K[i, j] = covariance;
          K[j, i] = covariance;
        }
      }
    }
    return K;
  }
}

data {
  int<lower=1> N;
  int<lower=1> D;
  array[N] vector[D] x;
  array[N] int<lower=0> count;
  vector<lower=0>[N] cell_area_km2;
  real<lower=0> distance_scale_km;
  real<lower=0> jitter;

  real<lower=0> alpha_prior_sd;
  real rho_prior_logmean;
  real<lower=0> rho_prior_logsd;
  int<lower=1, upper=7> kernel_id;
}

parameters {
  real<lower=1e-12> alpha;
  real<lower=0> rho;
  vector[N] eta;
}

transformed parameters {
  vector[N] gp_effect;
  simplex[N] spatial_probability;
  {
    matrix[N, N] K = kernel_covariance(x, alpha, rho, kernel_id);
    matrix[N, N] L_K;
    vector[N] raw_gp_effect;
    for (n in 1:N) {
      K[n, n] += jitter;
    }
    L_K = cholesky_decompose(K);
    raw_gp_effect = L_K * eta;
    gp_effect = raw_gp_effect - mean(raw_gp_effect);
    spatial_probability = softmax(log(cell_area_km2) + gp_effect);
  }
}

model {
  rho ~ lognormal(rho_prior_logmean, rho_prior_logsd);
  alpha ~ normal(0, alpha_prior_sd);
  eta ~ std_normal();

  count ~ multinomial(spatial_probability);
}

generated quantities {
  real rho_km = rho * distance_scale_km;
}
