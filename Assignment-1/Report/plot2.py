import pandas as pd
import matplotlib.pyplot as plt

names = [
    ("SIFT", "s1"),     
    ("AKAZE", "s1"),     
    ("SIFT", "s2"),     
    ("AKAZE", "s2"),     
    ("SIFT", "s3"),     
    ("AKAZE", "s3"),     
    ]

dfs = []

for method, name in names:
    dfs.append(pd.read_csv(f"distance_res_[{method}]-{name}.csv"))
    
counts = []
for df in dfs:
    counts.append(df["match distances"].count())

plt.bar(list(map(lambda x: x[0] + " " + x[1], names)), counts)

plt.title(f"Chart of matches")

plt.savefig("histogram-of-matches.png")

plt.show()
