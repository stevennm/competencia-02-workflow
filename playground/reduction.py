import polars as pl

# load data
df = pl.read_parquet("data/competencia_02_target.parquet")

# reduce the dataset to weight less than 60mb
df = df.sample(fraction=0.025)

# save the dataset to csv
df.write_csv("data/competencia_02_target_reduced.csv")