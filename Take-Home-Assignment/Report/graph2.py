shapes = [
    "Sphere",
    "Donut",
    "Pyramid",
    "Cube",
    "Cuboid",
    "Pentagonal Prism",
    "8-sided Object",
]
import matplotlib.pyplot as plt

avg_fps = [1021, 428, 1029, 1042, 1039, 1023, 1008]

plt.figure(figsize=(10, 5))
bars = plt.bar(shapes, avg_fps)

plt.axhline(y=1045, color='red', linestyle='--', linewidth=2)
plt.text(-0.7, 1045, "No Shape Average FPS: 1045", color="red", va="bottom")

for bar, value in zip(bars, avg_fps):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() / 2,
        str(value),
        ha='center', va='center', color='white', fontsize=10, fontweight='bold'
    )

plt.xlabel("Shape")
plt.ylabel("Average FPS")
plt.title("Average FPS per Shape")
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

