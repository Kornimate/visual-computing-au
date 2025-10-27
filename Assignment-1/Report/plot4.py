import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({              #    |   
    "x": [ 1, 3, 5, 10, 1, 3, 5, 10, 1, 3, 5, 10],
    "y": [0.357143,0.428571,0.428571,0.5, 0.252874,0.482759,0.62069,0.655172, 0.355263,0.631579,0.842105,0.855263],
    "group": ["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2", "s3", "s3", "s3", "s3"]
})

groups = df.groupby("group")

for name, group in groups:
    plt.plot(group["x"], group["y"], marker="o", label=name)

plt.xlabel("Threshold")
plt.ylabel("Inliner ratio")
plt.title("Threshold vs. Panorama alignment error")
plt.legend(title="Group")
plt.savefig("scatter_plot2.png")
plt.show()
