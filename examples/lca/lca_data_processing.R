library(dplyr)

# load raw data
load(file = "examples/lca/data/carcinoma.rda")

df = as.data.frame(apply(carcinoma, 2, as.factor))

colnames(carcinoma)

model.matrix(~A - 1, data = df)

# process data
varnames = colnames(df)
M = model.matrix(~eval(parse(text=varnames[1])) - 1, data = df)
for (i in 2:ncol(df)) {
    M = cbind(
        M, model.matrix(~eval(parse(text=varnames[i])) - 1, data = df)
    )
}
colnames(M) = 1:ncol(M)

# save processed file
write.csv(M, file = "examples/lca/data/carcinoma.csv", 
row.names = F)

