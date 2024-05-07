import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# x 轴的数据
x = np.array([0.001, 0.01, 0.1, 1, 10])

# y 轴的数据
acc = np.array([0.906, 0.933, 0.950, 0.921, 0.278])
nmi = np.array([0.837, 0.878, 0.900, 0.853, 0.209])

# 创建折线图
plt.plot(x, acc, marker='o', linestyle='-', color='r', label='ACC')  # 绘制 ACC 折线
plt.plot(x, nmi, marker='x', linestyle='--', color='b', label='NMI')  # 绘制 NMI 折线

# 设置 x 轴为对数尺度
plt.xscale("log")

# 设置主要刻度
major_ticks = [0.001, 0.01, 0.1, 1, 10]  # 仅显示这些主要刻度
plt.gca().xaxis.set_major_locator(ticker.FixedLocator(major_ticks))

# 关闭次要刻度和次要网格线
plt.gca().xaxis.set_minor_locator(ticker.NullLocator())  # 不显示次要刻度
plt.grid(True, which='major', linestyle='--', linewidth=0.5)  # 仅显示主要网格线

# 添加标签和标题
plt.xlabel(r"$\lambda$")
plt.ylabel("")
plt.ylim(0.2, 1)
plt.legend()  # 显示图例

plt.savefig("./ckpt/lambda.pdf", bbox_inches='tight', pad_inches=0)
