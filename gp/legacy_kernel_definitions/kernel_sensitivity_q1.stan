// Earlier covariance-kernel sensitivity definition retained for comparison.
//
// All data, priors, likelihood terms, non-centred parameterisation, and
// generated direct-rate diagnostics match the baseline baseline.  The
// only fitted-case selector is kernel_id.

functions {
  matrix kernel_covariance(
      array[] vector x,
      real alpha,
      real rho,
      int kernel_id) {
    int N = size(x);

    if (kernel_id == 1) {
      return gp_exp_quad_cov(x, alpha, rho);
    } else if (kernel_id == 2) {
      return gp_matern52_cov(x, alpha, rho);
    } else if (kernel_id == 3) {
      return gp_matern32_cov(x, alpha, rho);
    } else if (kernel_id == 4) {
      return gp_exponential_cov(x, alpha, rho);
    } else {
      matrix[N, N] K;
      for (i in 1:N) {
        K[i, i] = square(alpha);
        if (i < N) {
          for (j in (i + 1):N) {
            real d = sqrt(dot_self(x[i] - x[j]));
            real r = d / rho;
            real correlation;
            if (kernel_id == 5) {
              // Rational quadratic with fixed q = 1.
              correlation = inv(1 + 0.5 * square(r));
            } else if (kernel_id == 6) {
              // Fixed Cauchy form.
              correlation = inv(1 + square(r));
            } else {
              // Compactly supported Wendland C2 covariance.
              if (r < 1) {
                correlation = square(square(1 - r)) * (4 * r + 1);
              } else {
                correlation = 0;
              }
            }
            K[i, j] = square(alpha) * correlation;
            K[j, i] = K[i, j];
          }
        }
      }
      return K;
    }
  }
}

data {
  int<lower=1> N;
  int<lower=1> D;
  array[N] vector[D] x;
  array[N] int<lower=0> count;
  vector[N] log_exposure_area;
  vector<lower=0>[N] cell_area_km2;
  real<lower=0> distance_scale_km;
  real<lower=0> jitter;

  real a_prior_mean;
  real<lower=0> a_prior_sd;
  real<lower=0> alpha_prior_sd;
  real rho_prior_logmean;
  real<lower=0> rho_prior_logsd;
  int<lower=1, upper=7> kernel_id;
}

parameters {
  real a;
  real<lower=0> alpha;
  real<lower=0> rho;
  vector[N] eta;
}

transformed parameters {
  vector[N] gp_effect;
  {
    matrix[N, N] K = kernel_covariance(x, alpha, rho, kernel_id);
    matrix[N, N] L_K;
    for (n in 1:N) {
      K[n, n] += jitter;
    }
    L_K = cholesky_decompose(K);
    gp_effect = L_K * eta;
  }
}

model {
  rho ~ lognormal(rho_prior_logmean, rho_prior_logsd);
  alpha ~ normal(0, alpha_prior_sd);
  a ~ normal(a_prior_mean, a_prior_sd);
  eta ~ std_normal();

  count ~ poisson_log(log_exposure_area + a + gp_effect);
}

generated quantities {
  vector[N] diagnostic_direct_gp_rate_density;
  vector[N] diagnostic_direct_gp_rate;
  vector[N] log_lik;
  real diagnostic_direct_gp_total;
  real rho_km = rho * distance_scale_km;

  diagnostic_direct_gp_rate_density = exp(a + gp_effect);
  diagnostic_direct_gp_rate =
    diagnostic_direct_gp_rate_density .* cell_area_km2;
  diagnostic_direct_gp_total = sum(diagnostic_direct_gp_rate);
  for (n in 1:N) {
    log_lik[n] = poisson_log_lpmf(
      count[n] | log_exposure_area[n] + a + gp_effect[n]
    );
  }
}
