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
# set bounds
minY = minimum(Y)
maxY = maximum(Y)
rangeY = maxY - minY
sdY = √(var(Y))
bounds = [
    minY -rangeY -rangeY 0. -10. -5. 0.;
    maxY rangeY rangeY rangeY 10. 5. sdY]
obs = LnmObs(Y, X, Z)
mod = LnmModel(obs)
MetaEst.fit!(
    mod,
    DE(),
    bounds
)


# fit using all algorithms
results = fit_all!(mod, bounds)

df = DataFrame(results, ["loglik", "β₁₁", "β₁₂", "β₂₁", "β₂₂", "γ₁", "γ₂", "σ"])[1:7, :];
df.method = ["ECA", "DE", "PSO", "SA", "WOA", "GA", "εDE"];
select!(df, :method, :); # reorder columns
sort!(df, :loglik, rev = true)

# confidence intervals using bootstrapping
M = confint!(mod, 1000, 80, "DE", bounds)

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


# simulation study for confidence interval coverage
Nsims = 5
# setting - parameter true values
β₁ = [80., 0.] # regression parameters baseline
β₂ = [0., 30.] # increases due to latent subgroup
γ = [-1.39, 1.79] # rate of Gene X in women = .6, in men = .2
σ = 1.
θ = [80., 0., 0., 30., -1.39, 1.79, 1.]

function coverage(θ, lb, ub)
    if θ >= lb && θ <= ub
        return 1
    else
        return 0
    end
end

# no parallized loop because confint is parallel
N = 100
bootsamples = 1000
bootss = 80
coverage_counts_DE = zeros(length(θ))
coverage_counts_εDE = zeros(length(θ))
for i in 1:Nsims

    println("Simulation ", i, "/", Nsims)

    # generate dataset
    mod = generate_lnm(N, β₁, β₂, γ, σ)
    # compute confidence intervals for each algorithm
    # only using GA, DE
    # have to define bounds here for GA
    
    M_DE = confint!(mod, bootsamples, bootss, "DE")
    M_εDE = confint!(mod, bootsamples, bootss, "εDE")

    # check coverage
    for j in eachindex(θ)
        coverage_counts_DE[j] += coverage(θ[j], M_DE[j, 2], M_DE[j, 3])
    end

    for j in eachindex(θ)
        coverage_counts_εDE[j] += coverage(θ[j], M_DE[j, 2], M_DE[j, 3])
    end



end

# save results
coverage_DF = DataFrame(DE = coverage_counts_DE, εDE = coverage_counts_εDE)
using CSV
CSV.write("examples/subgroup_CI_coverage.csv", coverage_DF)


# analysis using ATN data
using CSV
analysis_data = CSV.read("C:/Users/wgertsch/Desktop/subgroup/data/study2.csv", DataFrame)

# process data
n = size(analysis_data, 1)
Y = analysis_data[!, "log10_vl_24m"]
Z = analysis_data[!, ["log10_vl_baseline", "intv_binary"]]
Z = hcat(ones(n), Z)
Z = Matrix(Z)
X = analysis_data[!, ["sex", "log10_vl_baseline", "black"]]
X = hcat(ones(n), X)
X = Matrix(X)

# fit model
obs = LnmObs(Y, X, Z)
mod = LnmModel(obs)
minY = minimum(Y)
maxY = maximum(Y)
rangeY = maxY - minY
sdY = √(var(Y))
bounds = [
    minY -rangeY -rangeY -rangeY -rangeY 0. -10. -5. -5. -5. 0.;
    maxY rangeY rangeY rangeY rangeY rangeY 10. 5. 5. 5. sdY
]
bounds = [

]
Random.seed!(1234)
MetaEst.fit!(
    mod,
    DE(),
    bounds
)

mod.β₁
mod.β₂
mod.γ
mod.σ

# try fitting all models
Random.seed!(1234567)
results = fit_all!(mod, bounds)
df = DataFrame(results, ["loglik", "β₁₁", "β₁₂", "β₁₃", "β₂₁", "β₂₂", "β₂₃", "γ₀", "γ₁", "γ₂", "γ₃", "σ"])[1:7, :];
df.method = ["ECA", "DE", "PSO", "SA", "WOA", "GA", "εDE"];
select!(df, :method, :); # reorder columns
sort!(df, :loglik, rev = true)
print(df)

# confidence interval
M = confint!(mod, 1000, 60, "DE", bounds)
print(M)