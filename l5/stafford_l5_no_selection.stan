// L4 comparison model without the catalogue-selection term.

functions {
  real mw2ml(real mw) {
    real a = 0.0376;
    real b = 0.646;
    real c = 0.53 - mw;
    real disc = fmax(square(b) - 4.0 * a * c, 0.0);
    return (-b + sqrt(disc)) / (2.0 * a);
  }

  real grunthal_slope(real ml) {
    return 0.0752 * ml + 0.646;
  }

  real grunthal_sigma_mw(real ml) {
    // Gruenthal et al. (2009) conversion variance as implemented in ONR T983.
    real var_mw = (
      0.97 * pow(ml, 4)
      - 12.4 * pow(ml, 3)
      + 58.4 * square(ml)
      - 120.0 * ml
      + 921.0
    ) * 1e-4;
    return sqrt(fmax(var_mw, 0.0));
  }

  real sigma_conversion_ml(real ml) {
    return grunthal_sigma_mw(ml) / abs(grunthal_slope(ml));
  }

  real completeness_time_for_ml(
    real ml,
    int n_rows,
    vector min_ml,
    vector exposure_time
  ) {
    real exposure = exposure_time[1];
    for (j in 1:n_rows) {
      if (ml >= min_ml[j]) {
        exposure = exposure_time[j];
      }
    }
    return exposure;
  }
}

data {
  int<lower=1> N;
  vector[N] ml_reported;
  vector<lower=0>[N] sigma_ml;
  real<lower=0> sigma_round;
  real<lower=0> dm_ml;

  real mw_min;
  real mw_max;
  real mw_floor;
  real ml_detection_threshold;

  int<lower=1> n_exposure_rows;
  vector[n_exposure_rows] exposure_min_ml;
  vector<lower=0>[n_exposure_rows] exposure_time;

  int<lower=1> n_quad;

  real beta_prior_mean;
  real<lower=0> beta_prior_sd;
  real lambda_prior_mean;
  real<lower=0> lambda_prior_sd;
}

transformed data {
  real dmw = (mw_max - mw_floor) / n_quad;
  real sigma_ml_average = mean(sigma_ml);
  vector[n_quad] mw_grid;
  vector[n_quad] t_grid;
  vector[n_quad] p_select_grid;
  matrix[N, n_quad] obs_kernel;

  for (q in 1:n_quad) {
    real mw_q = mw_floor + (q - 0.5) * dmw;
    real ml_q = mw2ml(mw_q);
    real sigma_conv = sigma_conversion_ml(ml_q);
    real t_q = completeness_time_for_ml(
      ml_q,
      n_exposure_rows,
      exposure_min_ml,
      exposure_time
    );
    real sigma_select = sqrt(
      square(sigma_ml_average) + square(sigma_conv) + square(sigma_round)
    );

    mw_grid[q] = mw_q;
    t_grid[q] = t_q;
    p_select_grid[q] = 1.0;

    for (i in 1:N) {
      real sigma_total_i = sqrt(
        square(sigma_ml[i]) + square(sigma_conv) + square(sigma_round)
      );
      obs_kernel[i, q] = t_q * exp(normal_lpdf(ml_reported[i] | ml_q, sigma_total_i));
    }
  }
}

parameters {
  real<lower=0.1, upper=5.0> beta;
  real<lower=1e-6, upper=10000.0> lambda_floor;
}

model {
  real gr_norm;
  vector[n_quad] gr_weight;
  real expected_count;
  vector[N] event_intensity;

  beta ~ normal(beta_prior_mean, beta_prior_sd);
  lambda_floor ~ lognormal(lambda_prior_mean, lambda_prior_sd);

  gr_norm = 1.0 - exp(-beta * (mw_max - mw_floor));
  for (q in 1:n_quad) {
    gr_weight[q] = beta / gr_norm * exp(-beta * (mw_grid[q] - mw_floor)) * dmw;
  }

  expected_count = lambda_floor * dot_product(t_grid .* p_select_grid, gr_weight);
  event_intensity = obs_kernel * gr_weight;

  target += sum(log(event_intensity)) + N * log(lambda_floor) - expected_count;
}

generated quantities {
  real b = beta / log(10.0);
  real frac_above_mw_min =
    (
      exp(-beta * (mw_min - mw_floor))
      - exp(-beta * (mw_max - mw_floor))
    )
    / (1.0 - exp(-beta * (mw_max - mw_floor)));
  real lambda_mw_min = lambda_floor * frac_above_mw_min;
}
