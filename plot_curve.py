import matplotlib.pyplot as plt
import os

# 定义数据点
x1, y1 = [1, 2, 3], [1, 2, 3]
x2, y2 = [10, 20, 30], [10, 20, 30]
x3, y3 = [100, 200, 300], [10, 20, 30]

# 创建一个新的绘图
plt.figure()

# 绘制折线1
plt.plot(x1, y1, color='red', marker='o', label='Line 1')

# 绘制折线2
plt.plot(x2, y2, color='green', marker='o', label='Line 2')

# 绘制折线3
plt.plot(x3, y3, color='blue', marker='o', label='Line 3')

# 添加标题和标签
plt.title('Line Plot with Three Lines')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图例
plt.legend()

# 创建保存路径
save_path = '/home/Visualizetrajectories/curve'
os.makedirs(save_path, exist_ok=True)

# 保存图像
plt.savefig(os.path.join(save_path, 'line_plot.png'))

# 显示图形
plt.show()
