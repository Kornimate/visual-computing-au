import matplotlib.pyplot as plt

# Example data
num_objects = [0, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 999]
fps = [1045, 1039, 1021, 1009, 910, 773, 684, 537, 427, 352, 307, 271, 241, 213]

plt.figure(figsize=(8, 5))
plt.plot(num_objects, fps, marker='o', linewidth=2)

plt.xlabel("# of Objects")
plt.ylabel("FPS")
plt.title("Objects vs FPS Performance")
plt.grid(True)

plt.show()
