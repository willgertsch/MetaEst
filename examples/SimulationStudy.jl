# main simulation study
using MetaEst

# check number of threads is maxed out
@assert Threads.nthreads() == 8

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

DE_results_case4 = SimTwoGaussMixEst(
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

DE_results_case3 = SimTwoGaussMixEst(
    "DE",
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

