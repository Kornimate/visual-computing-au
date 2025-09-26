import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({              #    |   
    "x": [ 90, 126, 1020, 808, 1186, 686],
    "y": [1.2653, 2.8425 ,73.0941, 8.6341, 92.2921, 7.3694],
    "group": ["SIFT", "AKAZE", "SIFT", "AKAZE", "SIFT", "AKAZE"]
})

groups = df.groupby("group")

for name, group in groups:
    plt.plot(group["x"], group["y"], marker="o", label=name)

plt.xlabel("Sum of Key points")
plt.ylabel("Elapsed Time")
plt.title("Key point count vs. Matching time")
plt.legend(title="Group")
plt.savefig("scatter_plot.png")
plt.show()
