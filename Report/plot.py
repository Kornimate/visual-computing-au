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

fig, axes = plt.subplots(1, len(dfs), figsize=(5 * 20, 4))

# Plot each histogram
for ax, col in zip(axes, range(0,len(dfs))):
    ax.hist(dfs[col]["match distances"].dropna(), bins=20, edgecolor="black")
    ax.set_title(f"Histogram of {names[col][1]} ({names[col][0]})")

# Adjust layout to avoid overlap
plt.tight_layout()

# Save the figure
plt.savefig("histograms.png")

plt.show()
