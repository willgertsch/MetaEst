using GaussianMixtures
using Distributions
using StatsPlots
using MetaEst

# using the same data as for my MetaEst testing
p = MixtureModel(Normal, [(-1, 4/9), (1, 4/9)], [.5,.5]);
y = rand(p, 1000);
density(y)

# fit model using EM
gm = GMM(2, y)
@show gm

# extract parameters
w = gm.w
μ = vec(gm.μ)
σ = sqrt.(vec(gm.Σ))

# compute log-likelihood on same objective function
obs = gaussmixObs(y, 2)
gaussmix_ll(obs, w, μ, σ)