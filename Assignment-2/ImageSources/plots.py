import matplotlib.pyplot as plt
import statistics

# Sample data
x = list(range(1,21))
data1 = []

with open("./Filters-HD.txt") as f:
    data1 = f.readlines()

data1 = list(map(lambda x: x.split(" "), data1))
    
y1 = list(map(lambda x: int(x[1]),filter(lambda x: x[2]=="[CPU]" and x[5]=="Pixelate", data1)))[:20]
y2 = list(map(lambda x: int(x[1]),filter(lambda x: x[2]=="[CPU]" and x[5]=="SinCity", data1)))[:20]
y3 = list(map(lambda x: int(x[1]),filter(lambda x: x[2]=="[CPU]" and x[5]=="Comic", data1)))[:20]
y4 = list(map(lambda x: int(x[1]),filter(lambda x: x[2]=="[GPU]" and x[5]=="Pixelate", data1)))[:20]
y5 = list(map(lambda x: int(x[1]),filter(lambda x: x[2]=="[GPU]" and x[5]=="SinCity", data1)))[:20]
y6 = list(map(lambda x: int(x[1]),filter(lambda x: x[2]=="[GPU]" and x[5]=="Comic", data1)))[:20]

print(statistics.mean(y1))
print(statistics.mean(y2))
print(statistics.mean(y3))
print(statistics.mean(y4))
print(statistics.mean(y5))
print(statistics.mean(y6))

# Plot multiple lines
plt.plot(x, y1, label='CPU Pixelate')
plt.plot(x, y2, label='CPU SinCity')
plt.plot(x, y3, label='CPU Comic')
plt.plot(x, y4, label='GPU Pixelate')
plt.plot(x, y5, label='GPU Sinity')
plt.plot(x, y6, label='GPU Comic')

# Add labels and title
plt.xlabel('Time (s)')
plt.ylabel('FPS')
plt.title('HD Resolution FPS')

# Show legend
plt.legend()

# Display plot
plt.show()