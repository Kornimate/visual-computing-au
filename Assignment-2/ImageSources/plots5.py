import matplotlib.pyplot as plt
import statistics

# Sample data
x = list(range(1,21))
data1 = []

with open("./Transformations.txt") as f:
    data1 = f.readlines()

data1 = list(map(lambda x: x.split(" "), data1))

print(data1[0][-1])
print(data1[0][-2])
    
y1 = list(map(lambda x: int(x[1]),filter(lambda x: x[2]=="[CPU]" and x[-1].strip() == "0,0", data1)))[:20]
y2 = list(map(lambda x: int(x[1]),filter(lambda x: x[2]=="[GPU]" and x[-1].strip() == "0,0", data1)))[:20]
y3 = list(map(lambda x: int(x[1]),filter(lambda x: x[2]=="[CPU]" and x[-1].strip() != "0,0", data1)))[:20]
y4 = list(map(lambda x: int(x[1]),filter(lambda x: x[2]=="[GPU]" and x[-1].strip() != "0,0", data1)))[:20]

print(statistics.mean(y1))
print(statistics.mean(y2))
print(statistics.mean(y3))
print(statistics.mean(y4))

# Plot multiple lines
plt.plot(x, y1, label='NO TRANSFORM CPU')
plt.plot(x, y2, label='NO TRANSFORM GPU')
plt.plot(x, y3, label='TRANSFORM CPU')
plt.plot(x, y4, label='TRANSFORM GPU')

# Add labels and title
plt.xlabel('Time (s)')
plt.ylabel('FPS')
plt.title('TRANSFORM vs. NO TRANSFORM')

# Show legend
plt.legend()

# Display plot
plt.show()