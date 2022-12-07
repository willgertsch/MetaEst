using MetaEst
using Random
using Distributions
using StatsBase
using Metaheuristics
using DataFrames
using Gadfly


# generate data set
# Model:
# Yᵢ ∼ N(Zᵢ'(β₁ + β₂δᵢ), σ²)
# P(δᵢ = 1) = ilogit(Xᵢ'γ)
# parameter true values
β₁ = [80., 0.] # regression parameters baseline
β₂ = [0., 30.] # increases due to latent subgroup
γ = [-1.39, 1.79] # rate of Gene X in women = .6, in men = .2
σ = 1.

Random.seed!(1234)
N = 100;
# sample men and women with 3/2 ratio
sex = StatsBase.sample([0., 1.], Weights([.6, .4]), N);
X = hcat(ones(N), sex);
# sample Gene X data based on logistic regression model
geneX = rand.(Bernoulli.(ilogit.(X * γ)));

# assign half to treatment
treat = repeat([0., 1.], inner = Int(N/2));
Z = hcat(ones(N), treat);

# generate from the mixture distribution
Y = Vector{Float64}(undef, N)
for i in 1:N
    if geneX[i] == 1.
        μ = Z[i, :]' * (β₁ + β₂)
        Y[i] = rand(Normal(μ, σ))
    else
        μ = Z[i, :]' * β₁
        Y[i] = rand(Normal(μ, σ))
    end
end

# final dataset
# write to CSV to be used in R
d = DataFrame((Y=Y, treat=treat, sex=sex, geneX=geneX));
using CSV
CSV.write("examples/clinicaltrial.csv", d)

# see bimodal distributions in treated group
plot(d,
x = "Y", xgroup = "treat",
Geom.subplot_grid(Geom.density))

# what happens if we control for sex?
plot(d,
x = "Y", xgroup = "treat", color = "sex",
Geom.subplot_grid(Geom.density))
# we get more precision, but bimodal shape remains unexplained

# controlling for latent geneX
plot(d,
x = "Y", xgroup = "treat", color = "geneX",
Geom.subplot_grid(Geom.density))
# bimodel effect is fully accounted for

# fit
obs = LnmObs(Y, X, Z)
mod = LnmModel(obs)
MetaEst.fit!(
    mod,
    DE()
)


# fit using all algorithms
results = fit_all!(mod)

df = DataFrame(results, ["loglik", "β₁₁", "β₁₂", "β₂₁", "β₂₂", "γ₁", "γ₂", "σ"])[1:7, :];
df.method = ["ECA", "DE", "PSO", "SA", "WOA", "GA", "εDE"];
select!(df, :method, :); # reorder columns
sort!(df, :loglik, rev = true)

# simulation study comparing different algorithms
# and at different sample sizes
Nsims = 1000 # run 1000 simulations
sample_sizes = [30, 50, 100, 300, 800]
#sample_sizes = [30, 50]
simulation_results = Vector{Matrix{Float64}}(undef, Nsims * length(sample_sizes));
for i in 1:Nsims * length(sample_sizes)
    simulation_results[i] = Matrix{Float64}(undef, 8, 9)
end

# setting - parameter true values
β₁ = [80., 0.] # regression parameters baseline
β₂ = [0., 30.] # increases due to latent subgroup
γ = [-1.39, 1.79] # rate of Gene X in women = .6, in men = .2
σ = 1.


# parallel loop
Random.seed!(4321)
# tool for putting data in results matrix
for i in eachindex(sample_sizes)

    N_i = sample_sizes[i]

    println("Simulating for sample size ", N_i)

    # simulate in parallel
    Threads.@threads for j in 1:Nsims

        # generate simulated data
        # sample men and women with 3/2 ratio
        sex = StatsBase.sample([0., 1.], Weights([.6, .4]), N_i);
        X = hcat(ones(N_i), sex);
        # sample Gene X data based on logistic regression model
        geneX = rand.(Bernoulli.(ilogit.(X * γ)));

        # assign half to treatment
        treat = repeat([0., 1.], inner = Int(N_i/2));
        Z = hcat(ones(N_i), treat);

        # generate from the mixture distribution
        Y = Vector{Float64}(undef, N_i)
        for i in 1:N_i
            if geneX[i] == 1.
                μ = Z[i, :]' * (β₁ + β₂)
                Y[i] = rand(Normal(μ, σ))
            else
                μ = Z[i, :]' * β₁
                Y[i] = rand(Normal(μ, σ))
            end
        end

        # fit all models to data
        obs = LnmObs(Y, X, Z)
        mod = LnmModel(obs)

        index = (i*Nsims - Nsims + j)
        #println(index)
        simulation_results[index][:, 1:8] .= fit_all!(mod)
        simulation_results[index][:, 9] .= N_i # save sample size
    end
end

# process data
# convert to data frame
# first step concatenates all matrices in vector of matrices
df = DataFrame(reduce(vcat, simulation_results), 
["loglik", "β₁₁", "β₁₂", "β₂₁", "β₂₂", "γ₁", "γ₂", "σ", "n"]);

# create methods column
df.method = repeat(["ECA", "DE", "PSO", "SA", "WOA", "GA", "εDE", "EM"], Nsims*length(sample_sizes));
select!(df, :method, :); # reorder columns

# write to CSV for analysis in R
using CSV
CSV.write("examples/subgroup_simulation_study.csv", df)