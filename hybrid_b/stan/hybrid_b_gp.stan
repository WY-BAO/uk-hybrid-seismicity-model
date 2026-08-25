// Source-informed gridded Gaussian Process (Hybrid B).
// The six zero-source cells and their eight events are outside this model.
// The observed model-domain total is conditioned on; L5 supplies the UK total
// activity rate only during post-processing.

data {
  int<lower=1> N;
  int<lower=1> D;
  array[N] vector[D] x;
  array[N] int<lower=0> count;
  vector<lower=0>[N] p_source;
  int<lower=1> expected_modelled_count;
  real<lower=0> distance_scale_km;
  real<lower=0> jitter;

  real<lower=0> alpha_prior_sd;
  real rho_prior_logmean;
  real<lower=0> rho_prior_logsd;
}

transformed data {
  if (abs(sum(p_source) - 1.0) > 1e-10) {
    reject("Model-domain p_source must sum to one; sum=", sum(p_source));
  }
  if (min(p_source) <= 0.0) {
    reject("Every model-domain p_source value must be strictly positive");
  }
  if (sum(count) != expected_modelled_count) {
    reject("Model-domain counts do not match expected_modelled_count");
  }
}

parameters {
  real<lower=1e-12> alpha;
  real<lower=0> rho;
  vector[N] eta;
}

transformed parameters {
  vector[N] f_correction;
  simplex[N] p_hybrid;
  {
    matrix[N, N] K = gp_exp_quad_cov(x, alpha, rho);
    matrix[N, N] L_K;
    vector[N] f_raw;
    vector[N] log_weight;
    for (n in 1:N) {
      K[n, n] += jitter;
    }
    L_K = cholesky_decompose(K);
    f_raw = L_K * eta;

    f_correction = f_raw - mean(f_raw);
    log_weight = log(p_source) + f_correction;
    p_hybrid = softmax(log_weight);
  }
}

model {
  rho ~ lognormal(rho_prior_logmean, rho_prior_logsd);
  alpha ~ normal(0, alpha_prior_sd);
  eta ~ std_normal();

  count ~ multinomial(p_hybrid);
}

generated quantities {
  real rho_km = rho * distance_scale_km;
}
