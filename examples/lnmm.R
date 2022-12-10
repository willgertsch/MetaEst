library(dplyr)
library(ggplot2)
library(tibble)


df = read.csv("~/JuliaProjects/MetaEst/examples/longitudinal_data.csv") %>% 
as_tibble()

colnames(df) = c("id", "Y", "Z-intercept", "treatment", "visit", "interaction",
"X-intercept", "sex", "race", "geneX")

ids = sample(unique(df$id), 20)


df %>%
ggplot(data = ., aes(x = visit, y = Y, group = id)) +
geom_point() + geom_line() +
theme_bw()

df %>%
filter(id %in% ids) %>%
ggplot(data = ., aes(x = visit, y = Y, group = id, color = as.factor(treatment))) +
geom_point() + geom_line() +
theme_bw()

df %>%
ggplot(data = ., aes(x = visit, y = Y, color = as.factor(treatment))) +
geom_smooth(method = "lm")

df %>%
ggplot(data = ., aes(x = visit, y = Y, color = as.factor(treatment))) +
geom_smooth(method = "lm") +
facet_wrap(~geneX)

count(df, geneX)

library(lme4)
library(lmerTest)
mod = lmer(Y ~ visit*treatment + (1|id), data = df)
summary(mod)


mod2 = lmer(Y ~ visit*treatment*geneX + (1|id), data = df)
summary(mod2)
