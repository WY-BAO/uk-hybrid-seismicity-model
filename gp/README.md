# Multinomial spatial GP

The GP models the relative distribution of the 1,013 retained earthquakes over
132 one-degree cells. Cell probabilities are area-adjusted and sum to one. The
annual activity rate is introduced afterwards by pairing GP draws with the L5
national-rate posterior.

The baseline uses an exponentiated-quadratic covariance, a half-normal prior
for the GP amplitude, and a lognormal prior for the length scale. Separate
directories contain the grid, prior, and kernel sensitivity analyses.
