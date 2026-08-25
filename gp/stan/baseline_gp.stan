// Baseline multinomial spatial Gaussian process.
//
// The observed catalogue total is conditioned on. The GP estimates only the
// relative spatial allocation; the independent L5 posterior supplies the
// UK-wide Mw >= 2 activity rate during post-processing.

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
    matrix[N, N] K = gp_exp_quad_cov(x, alpha, rho);
    matrix[N, N] L_K;
    vector[N] raw_gp_effect;
    for (n in 1:N) {
      K[n, n] += jitter;
    }
    L_K = cholesky_decompose(K);
    raw_gp_effect = L_K * eta;

    // The multinomial likelihood cannot identify an additive constant. The
    // centring removes that uninformative component without changing softmax.
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
