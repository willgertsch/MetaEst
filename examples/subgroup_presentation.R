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
