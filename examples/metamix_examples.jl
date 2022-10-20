# test out metamix function
using MetaEst
using Distributions
using Random

# generate data
p = MixtureModel(Normal, [(-1, 4/9), (1, 4/9)], [.5,.5]);
Random.seed!(1234);
y = rand(p, 1000);

bounds = [0. 0. -10. -10. 0. 0.; 1. 1. 10. 10. 10. 10.];

out = metamix(
    y,
    2,
    bounds,
    1234,
    parameters = [0.],
    alg = "ECA",
    swarm_size = 100,
    max_iter = 1000
)

# results
out.w
out.μ
out.σ
out.loglik