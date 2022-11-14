library(dplyr)
library(ggplot2)

df = read.csv("~/JuliaProjects/MetaEst/examples/clinicaltrial.csv") %>% as.data.frame()

# numbers for table 1
df %>% count(treat)

df %>%
group_by(treat) %>%
summarise(mean = mean(Y), sd = sd(Y))

df %>%
summarise(mean = mean(Y), sd = sd(Y))

count(df, treat, sex)

# statistical tests
t.test(df[df$treat==1,]$Y, df[df$treat==0,]$Y)
chisq.test(df[df$treat==1,]$sex, df[df$treat==0,]$sex)

# plots
ggplot(df, aes(x = Y)) +
geom_histogram() +
facet_wrap(~treat) +
theme_bw()

plot(density(df[df$treat==1,]$Y))

mod = lm(Y ~ sex * treat, df)
summary(mod)


# process results of simulation study
sim_data = read.csv("examples/subgroup_simulation_study.csv") %>%
as.data.frame() %>%
filter(method != "EM")

# rename columns
colnames(sim_data) = c("method", "loglik", "beta11", "beta12",
"beta21", "beta22", "gamma1", "gamma2", "sigma", "n")

rmse = function(theta, thetahat) {
    e = theta - thetahat
    e2 = e^2
    mse = sum(e2)/length(thetahat)
    return(sqrt(mse))
}

# calculate summary statistics
sim_data_sum = sim_data %>%
group_by(method, n) %>%
summarise(
    median_ll = median(loglik),
    beta11_rmse = rmse(80, beta11),
    beta12_rmse = rmse(0, beta12),
    beta21_rmse = rmse(0, beta21),
    beta22_rmse = rmse(30, beta22),
    gamma1_rmse = rmse(-1.39, gamma1),
    gamma2_rmse = rmse(1.79, gamma2),
    sigma_rmse = rmse(1, sigma)
) %>%
filter(method != "SA")
sim_data_sum

# plots for interesting parameters
# log-like
ggplot(sim_data_sum, aes(x = n, y = median_ll, color = method)) +
geom_point() + geom_line() +
theme_bw() +
labs(x = "Sample size", y = "Median log-likelihood")

# treament effect null group RMSE
ggplot(sim_data_sum, aes(x = n, y = beta12_rmse, color = method)) +
geom_point() + geom_line() +
theme_bw() +
labs(x = "Sample size", y = "baseline treatment effect RMSE")

# interaction effect
ggplot(sim_data_sum, aes(x = n, y = beta22_rmse, color = method)) +
geom_point() + geom_line() +
theme_bw() +
labs(x = "Sample size", y = "Interaction effect RMSE")
