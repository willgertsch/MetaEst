# main simulation study
using MetaEst

# check number of threads is maxed out
# Macbook Air (2020) has 8
# Desktop has Ryzen 5 3600 with 12
@assert Threads.nthreads() >= 8

# test for a single algorithm
# result = SimTwoGaussMixEst(
#     "EM",
#     [30, 50],
#     [.5, .5, -1., 1., 4/9, 4/9],
#     1000
# )

# parameters for study
N = 1000
sample_sizes = [30, 50, 100, 150, 200, 300];

# define cases
θ1 = [2/3, 1/3, 0., 0., 1, .1^2];
θ2 = [.5, .5, -1., 1., 4/9, 4/9];
θ3 = [3/4, 1/4, 0., 3/2, 1., 1/9];
θ4 = [1/4, 3/4, 0., 3/2, 1/9, 1.];

# EM
EM_results_case1 = SimTwoGaussMixEst(
    "EM",
    sample_sizes,
    θ1,
    N
);

EM_results_case2 = SimTwoGaussMixEst(
    "EM",
    sample_sizes,
    θ2,
    N
);

EM_results_case3 = SimTwoGaussMixEst(
    "EM",
    sample_sizes,
    θ3,
    N
);

EM_results_case4 = SimTwoGaussMixEst(
    "EM",
    sample_sizes,
    θ4,
    N
);

# DE
DE_results_case1 = SimTwoGaussMixEst(
    "DE",
    sample_sizes,
    θ1,
    N
);

DE_results_case2 = SimTwoGaussMixEst(
    "DE",
    sample_sizes,
    θ2,
    N
);

DE_results_case3 = SimTwoGaussMixEst(
    "DE",
    sample_sizes,
    θ3,
    N
);

DE_results_case4 = SimTwoGaussMixEst(
    "DE",
    sample_sizes,
    θ4,
    N
);

# PSO
PSO_results_case1 = SimTwoGaussMixEst(
    "PSO",
    sample_sizes,
    θ1,
    N
);

PSO_results_case2 = SimTwoGaussMixEst(
    "PSO",
    sample_sizes,
    θ2,
    N
);

PSO_results_case3 = SimTwoGaussMixEst(
    "PSO",
    sample_sizes,
    θ3,
    N
);

PSO_results_case4 = SimTwoGaussMixEst(
    "PSO",
    sample_sizes,
    θ4,
    N
);

# SA
SA_results_case1 = SimTwoGaussMixEst(
    "SA",
    sample_sizes,
    θ1,
    N
);

SA_results_case2 = SimTwoGaussMixEst(
    "SA",
    sample_sizes,
    θ2,
    N
);

SA_results_case3 = SimTwoGaussMixEst(
    "SA",
    sample_sizes,
    θ3,
    N
);

SA_results_case4 = SimTwoGaussMixEst(
    "SA",
    sample_sizes,
    θ4,
    N
);

# WOA
WOA_results_case1 = SimTwoGaussMixEst(
    "WOA",
    sample_sizes,
    θ1,
    N
);

WOA_results_case2 = SimTwoGaussMixEst(
    "WOA",
    sample_sizes,
    θ2,
    N
);

WOA_results_case3 = SimTwoGaussMixEst(
    "WOA",
    sample_sizes,
    θ3,
    N
);

WOA_results_case4 = SimTwoGaussMixEst(
    "WOA",
    sample_sizes,
    θ4,
    N
);

# ϵDE
ϵDE_results_case1 = SimTwoGaussMixEst(
    "ϵDE",
    sample_sizes,
    θ1,
    N
);

ϵDE_results_case2 = SimTwoGaussMixEst(
    "ϵDE",
    sample_sizes,
    θ2,
    N
);

ϵDE_results_case3 = SimTwoGaussMixEst(
    "ϵDE",
    sample_sizes,
    θ3,
    N
);

ϵDE_results_case4 = SimTwoGaussMixEst(
    "ϵDE",
    sample_sizes,
    θ4,
    N
);

# ECA
ECA_results_case1 = SimTwoGaussMixEst(
    "ECA",
    sample_sizes,
    θ1,
    N
);

ECA_results_case2 = SimTwoGaussMixEst(
    "ECA",
    sample_sizes,
    θ2,
    N
);

ECA_results_case3 = SimTwoGaussMixEst(
    "ECA",
    sample_sizes,
    θ3,
    N
);

ECA_results_case4 = SimTwoGaussMixEst(
    "ECA",
    sample_sizes,
    θ4,
    N
);

# process data
using DataFrames


# collect results into data frames
EM_df = DataFrame(vcat(
    EM_results_case1,
    EM_results_case2,
    EM_results_case3,
    EM_results_case4
), :auto);

DE_df = DataFrame(vcat(
    DE_results_case1,
    DE_results_case2,
    DE_results_case3,
    DE_results_case4
), :auto);

PSO_df = DataFrame(vcat(
    PSO_results_case1,
    PSO_results_case2,
    PSO_results_case3,
    PSO_results_case4
), :auto);

SA_df = DataFrame(vcat(
    SA_results_case1,
    SA_results_case2,
    SA_results_case3,
    SA_results_case4
), :auto);

WOA_df = DataFrame(vcat(
    WOA_results_case1,
    WOA_results_case2,
    WOA_results_case3,
    WOA_results_case4
), :auto);

ϵDE_df = DataFrame(vcat(
    ϵDE_results_case1,
    ϵDE_results_case2,
    ϵDE_results_case3,
    ϵDE_results_case4
), :auto);

ECA_df = DataFrame(vcat(
    ECA_results_case1,
    ECA_results_case2,
    ECA_results_case3,
    ECA_results_case4
), :auto);

# add columns for algorithms
# add column for case number and sample sizes
case_numbers = repeat(1:4, inner = size(sample_sizes, 1));
Ns = repeat(sample_sizes, 4);
insertcols!(EM_df, 1, :algorithm => "EM", :case => case_numbers, :n => Ns);
insertcols!(DE_df, 1, :algorithm => "DE", :case => case_numbers, :n => Ns);
insertcols!(ECA_df, 1, :algorithm => "ECA", :case => case_numbers, :n => Ns);
insertcols!(PSO_df, 1, :algorithm => "PSO", :case => case_numbers, :n => Ns);
insertcols!(SA_df, 1, :algorithm => "SA", :case => case_numbers, :n => Ns);
insertcols!(WOA_df, 1, :algorithm => "WOA", :case => case_numbers, :n => Ns);
insertcols!(ϵDE_df, 1, :algorithm => "ϵDE", :case => case_numbers, :n => Ns);

# combine into a single data frame
df = vcat(
    EM_df,
    DE_df,
    ECA_df,
    PSO_df,
    SA_df,
    WOA_df,
    ϵDE_df
);

# rename columns
rename!(df, [
    :algorithm,
    :case,
    :n,
    :RMSE_w1,
    :RMSE_w2,
    :RMSE_μ1,
    :RMSE_μ2,
    :RMSE_σ1,
    :RMSE_σ2,
    :bias_w1,
    :bias_w2,
    :bias_μ1,
    :bias_μ2,
    :bias_σ1,
    :bias_σ2,
    :median_loglik
]);

# save to a file
using CSV
CSV.write("simulation-study/simulation_study.csv", df)

# load back in
dfr = CSV.read("simulation-study/simulation_study.csv", DataFrame)

# plots
using Gadfly

# plot for loglikelihood
plot(dfr, 
x=:n, xgroup=:case, y=:median_loglik, color=:algorithm,
 Geom.subplot_grid(Geom.point, Geom.line))

 # RMSE
 plot(dfr, 
x=:n, xgroup=:case, y=:RMSE_w1, color=:algorithm,
 Geom.subplot_grid(Geom.point, Geom.line))

 plot(dfr, 
 x=:n, xgroup=:case, y=:RMSE_w2, color=:algorithm,
  Geom.subplot_grid(Geom.point, Geom.line))

  plot(dfr, 
  x=:n, xgroup=:case, y=:RMSE_μ1, color=:algorithm,
   Geom.subplot_grid(Geom.point, Geom.line))

plot(dfr, 
x=:n, xgroup=:case, y=:RMSE_μ2, color=:algorithm,
Geom.subplot_grid(Geom.point, Geom.line))

plot(dfr, 
x=:n, xgroup=:case, y=:RMSE_σ1, color=:algorithm,
Geom.subplot_grid(Geom.point, Geom.line))

plot(dfr, 
x=:n, xgroup=:case, y=:RMSE_σ2, color=:algorithm,
Geom.subplot_grid(Geom.point, Geom.line))

# bias
plot(dfr, 
x=:n, xgroup=:case, y=:bias_w1, color=:algorithm,
Geom.subplot_grid(Geom.point, Geom.line))

plot(dfr, 
x=:n, xgroup=:case, y=:bias_w2, color=:algorithm,
Geom.subplot_grid(Geom.point, Geom.line))

plot(dfr, 
x=:n, xgroup=:case, y=:bias_μ1, color=:algorithm,
Geom.subplot_grid(Geom.point, Geom.line))

plot(dfr, 
x=:n, xgroup=:case, y=:bias_μ2, color=:algorithm,
Geom.subplot_grid(Geom.point, Geom.line))

plot(dfr, 
x=:n, xgroup=:case, y=:bias_σ1, color=:algorithm,
Geom.subplot_grid(Geom.point, Geom.line))

plot(dfr, 
x=:n, xgroup=:case, y=:bias_σ2, color=:algorithm,
Geom.subplot_grid(Geom.point, Geom.line))

# plot this all in a loop
using Cairo
using Fontconfig
for col in names(dfr)[4:end]
    p = plot(dfr, 
    x=:n, xgroup=:case, y=col, color=:algorithm,
    Geom.subplot_grid(Geom.point, Geom.line))

    img = PNG(string("simulation-study/plots/", col, ".png"))
    draw(img, p)
end