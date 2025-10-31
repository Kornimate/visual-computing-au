import matplotlib.pyplot as plt
import statistics

# Sample data
x = list(range(1,21))
data1 = []
data2 = []

with open("./HD.txt") as f:
    data1 = f.readlines()

data1 = list(map(lambda x: x.split(" "), data1))

with open("./LOW.txt") as f:
    data2 = f.readlines()

data2 = list(map(lambda x: x.split(" "), data2))
    
y1 = list(map(lambda x: int(x[1]), data1))[:20]
y2 = list(map(lambda x: int(x[1]), data2))[:20]

print(statistics.mean(y1))
print(statistics.mean(y2))

# Plot multiple lines
plt.plot(x, y1, label='HD')
plt.plot(x, y2, label='LOW')

# Add labels and title
plt.xlabel('Time (s)')
plt.ylabel('FPS')
plt.title('HD vs. LOW Resolution')

# Show legend
plt.legend()

# Display plot
plt.show()