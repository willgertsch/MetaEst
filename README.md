# MetaEst
A Julia package for parameter estimation of statistical models using metaheuristic optimization algorithms. The code currently supports several variations on mixture models. Examples include 1-dimensional Gaussian mixtures, 1-dimensional mixtures for flood frequency analysis, and logistic-normal models for subgroup analysis. I plan on using this code-base for several parts of my thesis focused on parameter estimation and automatic tuning of algorithms.

## Metaheuristics for Subgroup Discovery
Even in a randomized study, there can be significant differences between subgroups of the patient population. This can lead to significant treatment effect heterogeneity where some participants might experience a treatment benefit while others do not. In many situations, these subgroups are unknown. Therefore, a common approach is to use clustering methods such as mixture models to identify distinct subgroups of the population that may respond to the treatment in different ways. The EM algorithm is commonly used to fit these models, but it can often converge to local optima. This is problematic for unsupervised learning tasks because a local optimum may represent a completely different clustering than the true optimum clustering. Metaheuristic optimization algorithms have stochastic components which allow them to avoid local optima and make them well suited to clustering problems. The goal of this project is to demostrate that metaheuristic algorithms can be used for fitting models such as mixture-of-experts, latent class analysis, and mixture hidden markov models and that they can outperform the EM algorithm. Other future work includes automatic tuning of algorithms and hybrid EM-metaheuristic algorithms.
