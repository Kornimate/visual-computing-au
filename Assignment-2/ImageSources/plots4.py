import matplotlib.pyplot as plt
import statistics

# Sample data
x = list(range(1,21))
data1 = []
data2 = []

with open("./DEBUG.txt") as f:
    data1 = f.readlines()

data1 = list(map(lambda x: x.split(" "), data1))

with open("./RELEASE.txt") as f:
    data2 = f.readlines()

data2 = list(map(lambda x: x.split(" "), data2))
    
y1 = list(map(lambda x: int(x[1]),filter(lambda x: x[2]=="[CPU]", data1)))[:20]
y2 = list(map(lambda x: int(x[1]),filter(lambda x: x[2]=="[GPU]", data1)))[:20]
y3 = list(map(lambda x: int(x[1]),filter(lambda x: x[2]=="[CPU]", data2)))[:20]
y4 = list(map(lambda x: int(x[1]),filter(lambda x: x[2]=="[GPU]", data2)))[:20]

print(statistics.mean(y1))
print(statistics.mean(y2))
print(statistics.mean(y3))
print(statistics.mean(y4))

# Plot multiple lines
plt.plot(x, y1, label='DEBUG CPU')
plt.plot(x, y2, label='DEBUG GPU')
plt.plot(x, y3, label='RELEASE CPU')
plt.plot(x, y4, label='RELEASE GPU')

# Add labels and title
plt.xlabel('Time (s)')
plt.ylabel('FPS')
plt.title('DEBUG vs. RELEASE Build')

# Show legend
plt.legend()

# Display plot
plt.show()